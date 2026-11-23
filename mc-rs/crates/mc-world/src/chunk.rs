//! [`ChunkSection`] / [`Chunk`] storage and the `S21PacketChunkData` / `S26PacketMapChunkBulk`
//! byte-layout unpacker.
//!
//! java: `world/chunk/Chunk.java`, `world/chunk/storage/ExtendedBlockStorage.java`,
//! `world/chunk/NibbleArray.java`,
//! `network/play/server/S21PacketChunkData.java` (`getExtractedData`/`Chunk.fillChunk`)
//!
//! # Wire layout (hazard H6)
//!
//! A chunk packet's payload is **not** interleaved per section. For the `N` sections selected
//! by the bitmask, in ascending section-index order, it is four back-to-back regions:
//!
//! 1. `N * 4096` block states, each a **little-endian `u16`** (low byte first) — `8192*N` bytes.
//! 2. `N * 2048` bytes of block-light nibbles.
//! 3. `N * 2048` bytes of sky-light nibbles — **only present when the dimension has sky**
//!    (`has_sky_light`). Reading this unconditionally desyncs every byte after it, which is
//!    exactly H6.
//! 4. `256` bytes of biome ids — only when the packet is `ground_up_continuous` (a full-column
//!    send), otherwise absent.
//!
//! Within a section, block/light arrays are indexed `y << 8 | z << 4 | x`
//! (`ExtendedBlockStorage.get`/`NibbleArray.getCoordinateIndex`), and a nibble array packs two
//! consecutive indices per byte, low nibble first (`NibbleArray.isLowerNibble`).

use std::sync::Arc;

use mc_data::BlockState;
use mc_math::BlockPos;

use crate::WorldError;

/// One 16x16x16 slice of a [`Chunk`]. java: `ExtendedBlockStorage`
#[derive(Debug, Clone)]
pub struct ChunkSection {
    blocks: [u16; 4096],
    block_light: [u8; 2048],
    sky_light: [u8; 2048],
    // Conservative between full packet replacements: removing a provider may leave this true.
    may_contain_tile_provider: bool,
    /// How many cells hold a raw block value other than 0 (air). Kept
    /// exact by every write to `blocks` (`set_block`, the packet unpacker); `0` proves the
    /// section reads `Blocks.air` in every cell, exactly like an absent section.
    non_air: u16,
}

/// `MC_W19_O2_AIR_SECTIONS=0` turns off the raw-air section skips in
/// the heightmap and skylight descents (read once).
pub(crate) fn air_sections_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MC_W19_O2_AIR_SECTIONS").map_or(true, |v| v != "0"))
}

#[inline]
const fn local_index(x: u8, y: u8, z: u8) -> usize {
    // java: ExtendedBlockStorage.get/set, NibbleArray.getCoordinateIndex — `y << 8 | z << 4 | x`
    ((y as usize) << 8) | ((z as usize) << 4) | (x as usize)
}

#[inline]
const fn nibble_get(arr: &[u8; 2048], x: u8, y: u8, z: u8) -> u8 {
    // java: NibbleArray.getFromIndex
    let idx = local_index(x, y, z);
    let byte = arr[idx >> 1];
    if idx & 1 == 0 {
        byte & 0x0F
    } else {
        (byte >> 4) & 0x0F
    }
}

#[inline]
fn nibble_set(arr: &mut [u8; 2048], x: u8, y: u8, z: u8, value: u8) {
    // java: NibbleArray.setIndex
    let idx = local_index(x, y, z);
    let i = idx >> 1;
    if idx & 1 == 0 {
        arr[i] = (arr[i] & 0xF0) | (value & 0x0F);
    } else {
        arr[i] = (arr[i] & 0x0F) | ((value & 0x0F) << 4);
    }
}

impl ChunkSection {
    /// A section of all-air blocks and zero light.
    #[inline]
    #[must_use]
    pub fn empty() -> Self {
        Self {
            blocks: [0; 4096],
            block_light: [0; 2048],
            sky_light: [0; 2048],
            may_contain_tile_provider: false,
            non_air: 0,
        }
    }

    /// Whether every cell holds the raw air value 0, so the section
    /// reads exactly like an absent one (`Blocks.air` everywhere: opacity 0, `Material.air`).
    /// O(1).
    #[inline]
    #[must_use]
    pub(crate) fn is_raw_air(&self) -> bool {
        self.non_air == 0
    }

    /// Recount [`Self::non_air`] after a test wrote `blocks` directly.
    #[cfg(test)]
    pub(crate) fn recount_non_air(&mut self) {
        self.non_air = self.blocks.iter().filter(|&&raw| raw != 0).count() as u16;
    }

    /// The count [`Self::is_raw_air`] reads, for the invariant test.
    #[cfg(test)]
    pub(crate) fn non_air_count(&self) -> u16 {
        self.non_air
    }

    /// The block state at a section-local coordinate (each `0..=15`).
    #[inline]
    #[must_use]
    pub fn block(&self, x: u8, y: u8, z: u8) -> BlockState {
        let state = BlockState::from_raw(self.blocks[local_index(x, y, z)]);
        // java: ExtendedBlockStorage.get asks Block.BLOCK_STATE_IDS for the stored value and
        // substitutes air when the registry has no such state. Keep the raw value in storage
        // (as Java's char[] does), but never expose an impossible state through a block lookup.
        if state.is_legal() {
            state
        } else {
            BlockState::AIR
        }
    }

    /// The packed block state before registry validation. Tile-entity scans use this only to
    /// reject the overwhelmingly common non-provider ids before paying `BlockState::is_legal`.
    /// Callers that expose a block to gameplay or rendering must continue to use [`Self::block`].
    #[inline]
    pub(crate) fn raw_block(&self, x: u8, y: u8, z: u8) -> BlockState {
        BlockState::from_raw(self.blocks[local_index(x, y, z)])
    }

    /// Sets the block state at a section-local coordinate (each `0..=15`).
    #[inline]
    pub fn set_block(&mut self, x: u8, y: u8, z: u8, state: BlockState) {
        let cell = &mut self.blocks[local_index(x, y, z)];
        // Add before subtracting: a non-zero old value is already
        // counted, so this cannot go below zero.
        self.non_air = self.non_air + u16::from(state.id != 0) - u16::from(*cell != 0);
        *cell = state.id;
        self.may_contain_tile_provider |= state.props().has_tile_entity;
    }

    /// False proves there is no legal tile provider; true may outlive a removed provider.
    #[inline]
    #[must_use]
    pub(crate) fn may_contain_tile_provider(&self) -> bool {
        self.may_contain_tile_provider
    }

    /// The block-light nibble (`0..=15`) at a section-local coordinate.
    #[inline]
    #[must_use]
    pub fn block_light(&self, x: u8, y: u8, z: u8) -> u8 {
        nibble_get(&self.block_light, x, y, z)
    }

    /// Sets the block-light nibble. `value` is masked to 4 bits.
    #[inline]
    pub fn set_block_light(&mut self, x: u8, y: u8, z: u8, value: u8) {
        nibble_set(&mut self.block_light, x, y, z, value);
    }

    /// The sky-light nibble (`0..=15`) at a section-local coordinate.
    ///
    /// Meaningless (and left at whatever the section was initialised with) in dimensions
    /// without sky — callers should consult [`crate::World`]'s dimension flag rather than
    /// trust this blindly, mirroring vanilla's `provider.getHasNoSky()` guard.
    #[inline]
    #[must_use]
    pub fn sky_light(&self, x: u8, y: u8, z: u8) -> u8 {
        nibble_get(&self.sky_light, x, y, z)
    }

    /// Sets the sky-light nibble. `value` is masked to 4 bits.
    #[inline]
    pub fn set_sky_light(&mut self, x: u8, y: u8, z: u8, value: u8) {
        nibble_set(&mut self.sky_light, x, y, z, value);
    }

    /// True when every block in this section is air. java: `ExtendedBlockStorage.isEmpty`
    ///
    /// Vanilla tracks this with an incremental `blockRefCount`; we recompute it by scanning,
    /// since S08 has no hot path that calls this per tick (S11's dirty-set already tells the
    /// mesher which sections changed).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        // java: ExtendedBlockStorage.removeInvalidBlocks/get treats every unregistered packed
        // value as air before rebuilding blockRefCount; an invalid raw value must therefore not
        // make an otherwise empty section non-empty.
        self.blocks
            .iter()
            .all(|&raw| BlockState::from_raw(raw).props().is_air())
    }
}

/// What [`Chunk::set_block_relit`] left for [`crate::World`] to finish — the tail of
/// `Chunk.setBlockState` that vanilla runs against `worldObj` and so cannot live on a chunk
/// that has no back-reference to its world.
///
/// java: `Chunk.setBlockState:717-741`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkSetOutcome {
    /// The write was refused. Either the new state equals the old one (`iblockstate == state`,
    /// `Chunk.java:673`) or it was air into a section that does not exist (`Chunk.java:688`) —
    /// both return `null` from `setBlockState`, which makes `World.setBlockState` return `false`
    /// **without** running `checkLight`.
    Unchanged,
    /// The write landed. Each field names one thing `World` still owes.
    Changed {
        /// The state that was overwritten, for `World.setBlockState`'s opacity/emission compare.
        old: BlockState,
        /// `generateSkylightMap()` is owed: a section was created at or above the heightmap.
        regenerate_skylight: bool,
        /// `relightBlock(x, <this y>, z)` is owed. World-absolute Y, as `relightBlock` takes it.
        relight_at: Option<i32>,
    },
}

/// A chunk column: 16 vertical sections plus one biome id per `(x, z)` sub-column.
/// java: `Chunk`
#[derive(Debug, Clone)]
pub struct Chunk {
    sections: [Option<Arc<ChunkSection>>; 16],
    biomes: [u8; 256],
    /// java: `Chunk.heightMap` (`Chunk.java:66`), indexed `z << 4 | x`.
    ///
    /// **`i32`, not `u8`.** `generateHeightMap` writes `l` where `l` starts at
    /// `getTopFilledSegment() + 16` = up to **256**, so a `u8` would wrap a column whose y=255
    /// is opaque to `0` — the exact value `relightBlock`'s `heightMap[..] & 255` mask exists to
    /// cope with. Vanilla's field is `int[]`; so is this.
    pub(crate) height_map: [i32; 256],
    /// java: `Chunk.precipitationHeightMap` (`Chunk.java:56`), same `z << 4 | x` index, with
    /// vanilla's `-999` "not computed" sentinel and its `-1` "nothing in this column" answer.
    pub(crate) precipitation_height_map: [i32; 256],
    /// java: `Chunk.heightMapMinimum` (`:96`) — `getLowestHeight`, which `recheckGaps` compares
    /// across the four horizontal neighbours.
    pub(crate) height_map_minimum: i32,
    /// java: `Chunk.updateSkylightColumns` (`:88`) — the per-column "this column's sky-light
    /// gap needs rechecking" flags `propagateSkylightOcclusion` sets. Indexed `x + z * 16`,
    /// which is numerically the same as the heightmap's `z << 4 | x` for local coordinates.
    pub(crate) update_skylight_columns: [bool; 256],
    /// java: `Chunk.isGapLightingUpdated` (`:73`) — the "any column flagged" short-circuit
    /// `func_150804_b` tests before paying for `recheckGaps` at all.
    pub(crate) is_gap_lighting_updated: bool,
    /// java: `Chunk.queuedLightChecks` (`:104`), initialised to **4096** by the constructor
    /// (`:114`), i.e. "already finished". `resetRelightChecks` puts it back to `0`.
    pub(crate) queued_light_checks: i32,
}

impl Chunk {
    /// A freshly loaded chunk: no sections, and biomes set to vanilla's `-1` "unset" sentinel
    /// (`Arrays.fill(blockBiomeArray, (byte) -1)`, `Chunk.java:128`).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            sections: std::array::from_fn(|_| None),
            biomes: [0xFF; 256],
            height_map: [0; 256],
            // java: `Arrays.fill(this.precipitationHeightMap, -999)` (`Chunk.java:127`)
            precipitation_height_map: [-999; 256],
            height_map_minimum: 0,
            update_skylight_columns: [false; 256],
            is_gap_lighting_updated: false,
            // java: `Chunk.java:114` — a chunk starts with its relight sweep *finished*;
            // `resetRelightChecks` is what arms it, and on the client only
            // `handleChunkData` calls that.
            queued_light_checks: 4096,
        }
    }

    /// The section at vertical index `0..=15`, if loaded.
    #[inline]
    #[must_use]
    pub fn section(&self, index: u8) -> Option<&ChunkSection> {
        self.sections.get(index as usize)?.as_deref()
    }

    /// Mutable access to the section at vertical index `0..=15`, if loaded.
    #[inline]
    #[must_use]
    pub fn section_mut(&mut self, index: u8) -> Option<&mut ChunkSection> {
        let section = self.sections.get_mut(index as usize)?.as_mut()?;
        Some(Arc::make_mut(section))
    }

    /// The section at vertical index `0..=15`, creating an empty one if absent.
    #[inline]
    pub fn section_or_insert(&mut self, index: u8) -> &mut ChunkSection {
        let section =
            self.sections[index as usize].get_or_insert_with(|| Arc::new(ChunkSection::empty()));
        Arc::make_mut(section)
    }

    /// All 16 section slots, bottom-to-top order as stored.
    #[inline]
    #[must_use]
    pub fn sections(&self) -> &[Option<Arc<ChunkSection>>; 16] {
        &self.sections
    }

    /// The block state at a chunk-local coordinate: `x`/`z` are `0..=15`, `y` is the full
    /// world height `0..=255`.
    #[must_use]
    pub fn block_at(&self, x: u8, y: u8, z: u8) -> BlockState {
        match self.section(y >> 4) {
            Some(section) => section.block(x, y & 15, z),
            None => BlockState::AIR,
        }
    }

    /// Sets the block state at a chunk-local coordinate, creating the target section if it
    /// does not exist yet (matching `Chunk.setBlockState`'s "create the storage on demand"
    /// behaviour).
    pub fn set_block_at(&mut self, x: u8, y: u8, z: u8, state: BlockState) {
        self.section_or_insert(y >> 4)
            .set_block(x, y & 15, z, state);
    }

    // ---- heightmap ---------------------------------------------------------

    /// The Y coordinate of the *base* of the highest section that exists, or `0` when the
    /// column has none. java: `Chunk.getTopFilledSegment` (`:186`)
    #[must_use]
    pub fn top_filled_segment(&self) -> i32 {
        for i in (0..16usize).rev() {
            if self.sections[i].is_some() {
                return (i as i32) << 4;
            }
        }
        0
    }

    /// The heightmap value at a chunk-local `(x, z)`: one *above* the topmost block with a
    /// non-zero light opacity, so `y >= height_value(x, z)` is exactly "this cell sees sky".
    /// java: `Chunk.getHeightValue` (`:178`)
    #[inline]
    #[must_use]
    pub fn height_value(&self, x: u8, z: u8) -> i32 {
        self.height_map[((z as usize) << 4) | (x as usize)]
    }

    /// The lowest heightmap value in this column, `i32::MAX` for an all-transparent chunk (a
    /// vanilla quirk: `generateHeightMap` only lowers the minimum for columns where it *found*
    /// something opaque). java: `Chunk.getLowestHeight` (`:1685`), `heightMapMinimum`
    #[inline]
    #[must_use]
    pub fn lowest_height(&self) -> i32 {
        self.height_map_minimum
    }

    /// Whether a chunk-local cell has an unobstructed view of the sky.
    /// java: `Chunk.canSeeSky` (`:904`)
    #[inline]
    #[must_use]
    pub fn can_see_sky(&self, x: u8, y: i32, z: u8) -> bool {
        y >= self.height_value(x, z)
    }

    /// `Block.getLightOpacity` at a chunk-local cell, `0` outside `y = 0..=255`.
    /// java: `Chunk.getBlockLightOpacity(int, int, int)` (`:508`)
    #[must_use]
    pub fn light_opacity(&self, x: u8, y: i32, z: u8) -> i32 {
        if !(0..256).contains(&y) {
            return 0;
        }
        i32::from(self.block_at(x, y as u8, z).props().light_opacity)
    }

    /// Rebuilds [`Chunk::height_map`] (and this port's eagerly-filled
    /// [`Chunk::precipitation_height_map`]) from the block data.
    /// java: `Chunk.generateHeightMap` (`:210`), called by `fillChunk` (`:1374`)
    ///
    /// Two deliberate transcription notes:
    /// * vanilla does **not** clear `heightMap` first, so a column with nothing opaque in it
    ///   keeps whatever value it had (`0` for a fresh chunk) and does not lower
    ///   `heightMapMinimum`. Reproduced, because a partial (`!ground_up_continuous`) packet
    ///   re-runs this on a chunk that already has values.
    /// * vanilla writes `-999` into `precipitationHeightMap` here and computes the real value
    ///   lazily in `getPrecipitationHeight`. This fills it during the same descent instead:
    ///   the values are identical, and the lazy version needs `&mut self` on what is otherwise
    ///   a pure read (`WorldColumns` holds `&World`).
    pub fn generate_height_map(&mut self) {
        let top = self.top_filled_segment();
        let air_sections = air_sections_enabled();
        self.height_map_minimum = i32::MAX;
        for x in 0..16u8 {
            for z in 0..16u8 {
                let idx = ((z as usize) << 4) | (x as usize);
                // One descent, not two: vanilla's heightmap loop
                // (`for (int l = i + 16; l > 0; --l) { getBlock0(j, l - 1, k) … }`) and
                // `getPrecipitationHeight`'s (`while (blockpos.getY() > 0 …)`) walk the same
                // cells with different predicates, and the block lookup is the expensive half.
                //
                // Section-aware: an absent section reads `Blocks.air` in every cell (opacity 0,
                // `Material.air`, neither movement-blocking nor liquid), so neither predicate can
                // fire inside one and skipping it visits exactly the cells that can change
                // anything, top-down as before. On a void map this is most of the column.
                let mut height_done = false;
                let mut precipitation = -1;
                'column: for si in (0..=(top >> 4) as usize).rev() {
                    let Some(section) = self.sections[si].as_deref() else {
                        continue;
                    };
                    // A raw-air section reads `Blocks.air` in every cell,
                    // exactly like an absent one, so it is skipped the same way.
                    if air_sections && section.is_raw_air() {
                        continue;
                    }
                    for ly in (0..16u8).rev() {
                        if height_done && precipitation != -1 {
                            break 'column;
                        }
                        let y = ((si as i32) << 4) | i32::from(ly);
                        let props = section.block(x, ly, z).props();
                        if !height_done && props.light_opacity != 0 {
                            self.height_map[idx] = y + 1;
                            if y + 1 < self.height_map_minimum {
                                self.height_map_minimum = y + 1;
                            }
                            height_done = true;
                        }
                        // `getPrecipitationHeight` stops at y = 1, one short of the heightmap's
                        // y = 0.
                        if precipitation == -1
                            && y > 0
                            && (props.material.blocks_movement() || props.material.is_liquid())
                        {
                            precipitation = y + 1;
                        }
                    }
                }
                self.precipitation_height_map[idx] = precipitation;
            }
        }
    }

    /// The pre-09-relight cell-by-cell [`Chunk::generate_height_map`], kept as the reference the
    /// section-aware descent is differentially tested against.
    #[cfg(test)]
    pub(crate) fn generate_height_map_reference(&mut self) {
        let top = self.top_filled_segment();
        self.height_map_minimum = i32::MAX;
        for x in 0..16u8 {
            for z in 0..16u8 {
                let idx = ((z as usize) << 4) | (x as usize);
                let mut height_done = false;
                let mut precipitation = -1;
                for y in (0..=(top + 15)).rev() {
                    if height_done && precipitation != -1 {
                        break;
                    }
                    let props = self.block_at(x, y as u8, z).props();
                    if !height_done && props.light_opacity != 0 {
                        self.height_map[idx] = y + 1;
                        if y + 1 < self.height_map_minimum {
                            self.height_map_minimum = y + 1;
                        }
                        height_done = true;
                    }
                    // `getPrecipitationHeight` stops at y = 1, one short of the heightmap's y = 0.
                    if precipitation == -1
                        && y > 0
                        && (props.material.blocks_movement() || props.material.is_liquid())
                    {
                        precipitation = y + 1;
                    }
                }
                self.precipitation_height_map[idx] = precipitation;
            }
        }
    }

    /// java: the scan inside `Chunk.getPrecipitationHeight` (`:1195-1214`) — the Y *above* the
    /// topmost block whose material either blocks movement or is a liquid, or `-1` when the
    /// column has none. Rain stops at materials that block movement or are liquids (glass and
    /// leaves do; torches, tall grass, and snow layers do not).
    #[must_use]
    fn compute_precipitation_height(&self, x: u8, z: u8) -> i32 {
        // java: `int l = this.getTopFilledSegment() + 15;`
        let mut y = self.top_filled_segment() + 15;
        while y > 0 {
            let material = self.block_at(x, y.clamp(0, 255) as u8, z).props().material;
            if material.blocks_movement() || material.is_liquid() {
                return y + 1;
            }
            y -= 1;
        }
        -1
    }

    /// The Y precipitation falls to in this column, from the cache when it is valid and from a
    /// fresh scan when [`Chunk::set_block_relit`] invalidated it.
    /// java: `Chunk.getPrecipitationHeight` (`:1186`)
    #[must_use]
    pub fn precipitation_height(&self, x: u8, z: u8) -> i32 {
        let cached = self.precipitation_height_map[((z as usize) << 4) | (x as usize)];
        if cached == -999 {
            self.compute_precipitation_height(x, z)
        } else {
            cached
        }
    }

    // ---- the light-aware block write ---------------------------------------

    /// `Chunk.setBlockState`'s write plus its chunk-local lighting bookkeeping, reporting back
    /// what only [`crate::World`] can finish. java: `Chunk.setBlockState` (`:658-741`)
    ///
    /// Deliberately *not* `set_block_at`: this one refuses a no-op write and refuses to
    /// allocate a section for air, both of which vanilla signals by returning `null` — and a
    /// `null` return is what stops `World.setBlockState` from running `checkLight` at all.
    pub(crate) fn set_block_relit(
        &mut self,
        x: u8,
        y: i32,
        z: u8,
        state: BlockState,
    ) -> ChunkSetOutcome {
        let idx = ((z as usize) << 4) | (x as usize);
        // java: `if (j >= this.precipitationHeightMap[l] - 1) this.precipitationHeightMap[l] = -999;`
        if y >= self.precipitation_height_map[idx] - 1 {
            self.precipitation_height_map[idx] = -999;
        }
        let i1 = self.height_map[idx];
        let old = self.block_at(x, y as u8, z);
        if old == state {
            return ChunkSetOutcome::Unchanged;
        }

        let mut regenerate_skylight = false;
        if self.section((y >> 4) as u8).is_none() {
            // java: `if (block == Blocks.air) return null;` — writing air into a hole is not
            // worth 12 KiB of section, and vanilla declines it.
            // Java compares `state.getBlock()` with Blocks.air, after registry resolution. That
            // includes impossible air metadata and every unregistered packed state, not only the
            // one canonical `BlockState::AIR` value.
            if state.props().is_air() {
                return ChunkSetOutcome::Unchanged;
            }
            regenerate_skylight = y >= i1;
        }
        self.section_or_insert((y >> 4) as u8)
            .set_block(x, (y & 15) as u8, z, state);

        let mut relight_at = None;
        if !regenerate_skylight {
            let new_opacity = i32::from(state.props().light_opacity);
            if new_opacity > 0 {
                if y >= i1 {
                    relight_at = Some(y + 1);
                }
            } else if y == i1 - 1 {
                relight_at = Some(y);
            }
        }

        ChunkSetOutcome::Changed {
            old,
            regenerate_skylight,
            relight_at,
        }
    }

    /// java: `Chunk.generateSkylightMap` (`:246`) — the heightmap rebuild *plus* a fresh
    /// top-down sky-light descent per column. Run when a section is created at or above the
    /// heightmap, which is the one case where the stored nibbles cannot be trusted.
    ///
    /// The heightmap half of `generateSkylightMap` is character-for-character
    /// `generateHeightMap` (`getBlockLightOpacity(x, y, z)` is `getBlock0(x, y, z)
    /// .getLightOpacity()`), so it is not written out twice.
    /// Returns the inclusive Y range it actually wrote sky light into, or `None` when it wrote
    /// none — which is the union of the `notifyLightSet` calls vanilla makes one per written
    /// cell, and therefore exactly what the caller has to invalidate.
    pub fn generate_skylight_map(&mut self, has_sky_light: bool) -> Option<(i32, i32)> {
        let top = self.top_filled_segment();
        self.generate_height_map();
        if !has_sky_light {
            return None;
        }
        let mut written: Option<(i32, i32)> = None;
        // While every present section from the top down is raw air,
        // every column enters it at 15 (absent sections above it are skipped at 15, and an air
        // cell costs nothing at full sky) and writes 15 into all 16 of its cells: so each such
        // section ends with 15 everywhere. Fill it in one write instead of 256 columns x 16
        // cells, then start every column below the run with `k1` still 15: the same nibbles,
        // the same written range, the same copy-on-write (a written section is detached). Section
        // 0 stays with the column loop, which never writes y = 0.
        let mut start = top + 16 - 1;
        if air_sections_enabled() {
            let mut si = (top >> 4) as usize;
            while si >= 1 {
                if let Some(slot) = self.sections[si].as_mut() {
                    if !slot.is_raw_air() {
                        break;
                    }
                    Arc::make_mut(slot).sky_light = [0xFF; 2048];
                    let (lo, hi) = ((si as i32) << 4, ((si as i32) << 4) + 15);
                    written = Some(match written {
                        None => (lo, hi),
                        Some((a, b)) => (a.min(lo), b.max(hi)),
                    });
                }
                start = ((si as i32) << 4) - 1;
                si -= 1;
            }
        }
        for x in 0..16u8 {
            for z in 0..16u8 {
                let mut k1 = 15i32;
                let mut i1 = start;
                loop {
                    // `i1 >= 1` here: the loop exits before it drops lower.
                    let si = (i1 >> 4) as usize;
                    if self.sections[si].is_some() {
                        let (next_i1, next_k1, done) =
                            self.sky_descent_run(x, z, si, i1, k1, &mut written);
                        if done {
                            break;
                        }
                        (i1, k1) = (next_i1, next_k1);
                        continue;
                    }
                    // Section-aware skip. While `k1` is still 15, every cell of an absent section
                    // has opacity 0, so `j1` stays 0 (the "costs 1" rule needs `k1 != 15`), `k1`
                    // stays 15 and, with no storage, nothing is written: only `i1` moves. Jump it
                    // to the cell below the section, keeping the loop's `i1 <= 0` exit (so y = 0
                    // is still never visited). Once `k1 < 15` each air cell costs 1, so at most 14
                    // more cells run cell by cell below (and none of them can be written).
                    if k1 == 15 {
                        i1 = ((si as i32) << 4) - 1;
                        if i1 <= 0 {
                            break;
                        }
                        continue;
                    }
                    let mut j1 = self.light_opacity(x, i1, z);
                    // java: the first transparent cell below an opaque one still costs 1.
                    if j1 == 0 && k1 != 15 {
                        j1 = 1;
                    }
                    k1 -= j1;
                    if k1 > 0 {
                        if let Some(section) = self.section_mut((i1 >> 4) as u8) {
                            section.set_sky_light(x, (i1 & 15) as u8, z, k1 as u8);
                            written = Some(match written {
                                None => (i1, i1),
                                Some((lo, hi)) => (lo.min(i1), hi.max(i1)),
                            });
                        }
                    }
                    i1 -= 1;
                    if i1 <= 0 || k1 <= 0 {
                        break;
                    }
                }
            }
        }
        written
    }

    /// One present section's stretch of [`Chunk::generate_skylight_map`]'s per-column descent,
    /// entered at cell `i1` (inside section `si`) with running value `k1`. Returns the next
    /// `(i1, k1)` and whether the column's descent has finished.
    ///
    /// Same cells, values, exits and `written` range as the per-cell loop. The one difference is
    /// that `Arc::make_mut` runs once, before the first write, rather than once per written cell
    /// (an atomic CAS each, ~20k per call on a void chunk). The copy-on-write decision is the
    /// same: a section shared with a render snapshot is detached iff it is written at all.
    fn sky_descent_run(
        &mut self,
        x: u8,
        z: u8,
        si: usize,
        mut i1: i32,
        mut k1: i32,
        written: &mut Option<(i32, i32)>,
    ) -> (i32, i32, bool) {
        let base = (si as i32) << 4;
        let slot = self.sections[si]
            .as_mut()
            .expect("the caller checked this section is present");
        // Every cell of a raw-air section has opacity 0: no lookups.
        let raw_air = air_sections_enabled() && slot.is_raw_air();
        let mut j1 = if raw_air {
            0
        } else {
            i32::from(slot.block(x, (i1 & 15) as u8, z).props().light_opacity)
        };
        // java: the first transparent cell below an opaque one still costs 1.
        if j1 == 0 && k1 != 15 {
            j1 = 1;
        }
        k1 -= j1;
        if k1 <= 0 {
            // No write, and `k1 <= 0` ends the descent.
            return (i1 - 1, k1, true);
        }
        let section = Arc::make_mut(slot);
        loop {
            // `k1 > 0` here: this cell is written.
            section.set_sky_light(x, (i1 & 15) as u8, z, k1 as u8);
            *written = Some(match *written {
                None => (i1, i1),
                Some((lo, hi)) => (lo.min(i1), hi.max(i1)),
            });
            i1 -= 1;
            if i1 <= 0 {
                return (i1, k1, true);
            }
            if i1 < base {
                return (i1, k1, false);
            }
            let mut j1 = if raw_air {
                0
            } else {
                i32::from(section.block(x, (i1 & 15) as u8, z).props().light_opacity)
            };
            if j1 == 0 && k1 != 15 {
                j1 = 1;
            }
            k1 -= j1;
            if k1 <= 0 {
                return (i1 - 1, k1, true);
            }
        }
    }

    /// The pre-09-relight cell-by-cell [`Chunk::generate_skylight_map`] (heightmap half included),
    /// kept as the reference the section-aware descent is differentially tested against.
    #[cfg(test)]
    pub(crate) fn generate_skylight_map_reference(
        &mut self,
        has_sky_light: bool,
    ) -> Option<(i32, i32)> {
        let top = self.top_filled_segment();
        self.generate_height_map_reference();
        if !has_sky_light {
            return None;
        }
        let mut written: Option<(i32, i32)> = None;
        for x in 0..16u8 {
            for z in 0..16u8 {
                let mut k1 = 15i32;
                let mut i1 = top + 16 - 1;
                loop {
                    let mut j1 = self.light_opacity(x, i1, z);
                    if j1 == 0 && k1 != 15 {
                        j1 = 1;
                    }
                    k1 -= j1;
                    if k1 > 0 {
                        if let Some(section) = self.section_mut((i1 >> 4) as u8) {
                            section.set_sky_light(x, (i1 & 15) as u8, z, k1 as u8);
                            written = Some(match written {
                                None => (i1, i1),
                                Some((lo, hi)) => (lo.min(i1), hi.max(i1)),
                            });
                        }
                    }
                    i1 -= 1;
                    if i1 <= 0 || k1 <= 0 {
                        break;
                    }
                }
            }
        }
        written
    }

    /// Whether two columns hold identical storage: section presence, every block, light
    /// nibble and tile-provider flag, both height maps and the light bookkeeping. Test-only
    /// support for the light-engine differential tests.
    #[cfg(test)]
    pub(crate) fn same_storage(&self, other: &Self) -> bool {
        let sections_equal = self
            .sections
            .iter()
            .zip(other.sections.iter())
            .all(|(a, b)| match (a, b) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    Arc::ptr_eq(a, b)
                        || (a.blocks == b.blocks
                            && a.block_light == b.block_light
                            && a.sky_light == b.sky_light
                            && a.may_contain_tile_provider == b.may_contain_tile_provider)
                }
                _ => false,
            });
        sections_equal
            && self.biomes == other.biomes
            && self.height_map == other.height_map
            && self.precipitation_height_map == other.precipitation_height_map
            && self.height_map_minimum == other.height_map_minimum
            && self.update_skylight_columns == other.update_skylight_columns
            && self.is_gap_lighting_updated == other.is_gap_lighting_updated
            && self.queued_light_checks == other.queued_light_checks
    }

    /// The first half of `Chunk.relightBlock` (`:400-408`): where the heightmap *would* move to.
    /// `None` is vanilla's `if (j != i)` failing — the column did not change and nothing else in
    /// `relightBlock` runs.
    ///
    /// Split from [`Chunk::relight_finish`] because vanilla calls
    /// `worldObj.markBlocksDirtyVertical(…, j, i)` **between** the two halves, at `:412` — i.e.
    /// with the heightmap still holding its *old* value. That ordering is load-bearing:
    /// `markBlocksDirtyVertical` runs `checkLightFor(SKY, …)` up the column, and `getRawLight`'s
    /// sky base case is `canSeeSky`, which reads the heightmap. Writing the new height first
    /// made the parity fixtures `roof_one_block` and `slab_only` disagree with vanilla over
    /// hundreds of cells.
    pub(crate) fn relight_begin(&self, x: u8, y: i32, z: u8) -> Option<(i32, i32)> {
        let idx = ((z as usize) << 4) | (x as usize);
        // java: `int i = this.heightMap[z << 4 | x] & 255;` — the mask that makes a heightmap of
        // exactly 256 (a column opaque at y=255) read back as 0.
        let i = self.height_map[idx] & 255;
        let mut j = if y > i { y } else { i };
        while j > 0 {
            let below = j - 1;
            // Section-aware: every cell of an absent section has opacity 0, so this descent
            // steps through all of them; land on the section's base in one move instead.
            if below < 256 && self.sections[(below >> 4) as usize].is_none() {
                j = (below >> 4) << 4;
                continue;
            }
            if self.light_opacity(x, below, z) != 0 {
                break;
            }
            j -= 1;
        }
        if j == i {
            None
        } else {
            Some((i, j))
        }
    }

    /// The pre-09-relight cell-by-cell [`Chunk::relight_begin`], kept as a differential
    /// reference.
    #[cfg(test)]
    pub(crate) fn relight_begin_reference(&self, x: u8, y: i32, z: u8) -> Option<(i32, i32)> {
        let idx = ((z as usize) << 4) | (x as usize);
        let i = self.height_map[idx] & 255;
        let mut j = if y > i { y } else { i };
        while j > 0 && self.light_opacity(x, j - 1, z) == 0 {
            j -= 1;
        }
        if j == i {
            None
        } else {
            Some((i, j))
        }
    }

    /// The rest of `Chunk.relightBlock` (`:413-506`): commit the new height, rewrite the sky
    /// column through it, and report the Y range `updateSkylightNeighborHeight` needs.
    /// `i`/`j` are [`Chunk::relight_begin`]'s pair, deliberately *not* re-derived — vanilla
    /// carries the same two stale locals across its own re-entrant `checkLightFor` calls.
    pub(crate) fn relight_finish(
        &mut self,
        x: u8,
        z: u8,
        i: i32,
        j: i32,
        has_sky_light: bool,
    ) -> crate::light::RelightOutcome {
        let idx = ((z as usize) << 4) | (x as usize);
        self.height_map[idx] = j;
        let mut written: Option<(i32, i32)> = None;
        let note = |written: &mut Option<(i32, i32)>, y: i32| {
            *written = Some(match *written {
                None => (y, y),
                Some((lo, hi)) => (lo.min(y), hi.max(y)),
            });
        };

        if has_sky_light {
            if j < i {
                // The column got shorter: everything between is now open sky.
                for j1 in j..i {
                    if let Some(section) = self.section_mut((j1 >> 4) as u8) {
                        section.set_sky_light(x, (j1 & 15) as u8, z, 15);
                        note(&mut written, j1);
                    }
                }
            } else {
                // The column got taller: everything between is now shadowed.
                for i1 in i..j {
                    if let Some(section) = self.section_mut((i1 >> 4) as u8) {
                        section.set_sky_light(x, (i1 & 15) as u8, z, 0);
                        note(&mut written, i1);
                    }
                }
            }
            // Then walk down from the new height attenuating by each cell's opacity.
            let mut k1 = 15i32;
            let mut jj = j;
            while jj > 0 && k1 > 0 {
                jj -= 1;
                let mut i2 = self.light_opacity(x, jj, z);
                if i2 == 0 {
                    i2 = 1;
                }
                k1 -= i2;
                if k1 < 0 {
                    k1 = 0;
                }
                if let Some(section) = self.section_mut((jj >> 4) as u8) {
                    section.set_sky_light(x, (jj & 15) as u8, z, k1 as u8);
                    note(&mut written, jj);
                }
            }
        }

        let l1 = self.height_map[idx];
        let neighbour_range = if l1 < i { (l1, i) } else { (i, l1) };
        if l1 < self.height_map_minimum {
            self.height_map_minimum = l1;
        }
        crate::light::RelightOutcome {
            neighbour_range,
            written,
        }
    }

    /// java: `Chunk.propagateSkylightOcclusion` (`:316`) — flags one column for `recheckGaps`.
    #[inline]
    pub(crate) fn propagate_skylight_occlusion(&mut self, x: u8, z: u8) {
        self.update_skylight_columns[(x as usize) + (z as usize) * 16] = true;
        self.is_gap_lighting_updated = true;
    }

    /// Whether any column in this chunk is flagged for `recheckGaps`.
    /// java: `Chunk.isGapLightingUpdated` (`:73`)
    #[inline]
    #[must_use]
    pub fn is_gap_lighting_updated(&self) -> bool {
        self.is_gap_lighting_updated
    }

    /// How far the round-robin relight sweep has got, `0..=4096`. `4096` means "finished", which
    /// is also where a chunk starts. java: `Chunk.queuedLightChecks` (`:104`)
    #[inline]
    #[must_use]
    pub fn queued_light_checks(&self) -> i32 {
        self.queued_light_checks
    }

    /// java: `Chunk.resetRelightChecks` (`:1429`) — arms the round-robin relight sweep.
    /// `NetHandlerPlayClient:768` calls this after any partial chunk packet, and after *every*
    /// chunk packet in a dimension whose provider is not `WorldProviderSurface`.
    #[inline]
    pub fn reset_relight_checks(&mut self) {
        self.queued_light_checks = 0;
    }

    /// The biome id at a chunk-local `(x, z)`, each `0..=15`, when no generated-biome source
    /// is available. java: `Chunk.getBiome` (index `z << 4 | x`, `Chunk.java:1386`)
    ///
    /// The stored byte is vanilla's `-1`/`0xFF` "never received" sentinel for any column that
    /// has only ever had non-`ground_up_continuous` packets. A real `Chunk.getBiome` call asks
    /// its `WorldChunkManager` to generate and cache that cell instead. Use
    /// [`Self::biome_with_generated`] where that source is available; this wire-only convenience
    /// retains the old Plains answer solely for callers which have no provider context.
    #[inline]
    #[must_use]
    pub fn biome(&self, x: u8, z: u8) -> u8 {
        let raw = self.biomes[((z as usize) << 4) | (x as usize)];
        // java: Chunk.getBiome returns Plains when BiomeGenBase.getBiome(raw) is null. The
        // mc-data resolver reproduces both that case and the 0xFF/unset case.
        mc_data::biome(raw).id
    }

    /// Resolves the biome at `position`, generating and caching an unset `0xFF` cell exactly as
    /// `Chunk.getBiome(BlockPos, WorldChunkManager)` does.
    ///
    /// `generated` represents `WorldChunkManager.getBiomeGenerator(position, plains)`: return
    /// `Some(id)` for the generated biome, or `None` only when that source itself has no answer,
    /// in which case vanilla's supplied Plains default (`id = 1`) is used. The generated answer
    /// is normalized through the registered biome table before it is cached, so an invalid custom
    /// provider value cannot leave the sentinel live forever.
    ///
    /// Unlike [`Self::biome`], this takes `&mut self` because vanilla writes the first generated
    /// result into `blockBiomeArray`; later reads of the same local `(x, z)` do not re-query the
    /// source, even if their Y differs.
    #[must_use]
    pub fn biome_with_generated(
        &mut self,
        position: BlockPos,
        generated: impl FnOnce(BlockPos) -> Option<u8>,
    ) -> u8 {
        let x = (position.x & 15) as usize;
        let z = (position.z & 15) as usize;
        let index = (z << 4) | x;
        let raw = self.biomes[index];
        if raw == u8::MAX {
            // Java's manager accepts `BiomeGenBase.plains` as its default. Its normal generator
            // always returns a registered BiomeGenBase; normalize the public closure seam too.
            let id = generated(position).map_or(1, |id| mc_data::biome(id).id);
            self.biomes[index] = id;
            return id;
        }
        mc_data::biome(raw).id
    }

    /// Sets the biome id at a chunk-local `(x, z)`, each `0..=15`.
    #[inline]
    pub fn set_biome(&mut self, x: u8, z: u8, biome: u8) {
        self.biomes[((z as usize) << 4) | (x as usize)] = biome;
    }

    /// The raw 256-entry biome array, `z << 4 | x` order.
    #[inline]
    #[must_use]
    pub fn biomes(&self) -> &[u8; 256] {
        &self.biomes
    }

    /// Applies one `S21PacketChunkData` / `S26PacketMapChunkBulk` payload to this chunk.
    ///
    /// `data` is the packet's raw extracted byte array; `primary_bitmask` is its section
    /// bitmask (`dataSize` in the Java); `ground_up_continuous` is the
    /// packet's "full column" flag (`field_149279_g` in `S21PacketChunkData`, always `true`
    /// for a bulk-packet entry); `has_sky_light` reflects the *dimension*, not the packet —
    /// callers get it from [`crate::World`]'s dimension flag (H6).
    ///
    /// java: `Chunk.fillChunk`
    pub fn fill_from_packet(
        &mut self,
        ground_up_continuous: bool,
        primary_bitmask: u16,
        has_sky_light: bool,
        data: &[u8],
    ) -> Result<(), WorldError> {
        let expected =
            expected_chunk_data_len(primary_bitmask, has_sky_light, ground_up_continuous);
        // java: Chunk.fillChunk advances a cursor over the fields it knows and never checks the
        // array tail. A short array faults, while a proxy-padded array is accepted and ignored.
        if data.len() < expected {
            return Err(WorldError::ChunkDataLength {
                expected,
                actual: data.len(),
            });
        }

        let mut i = 0usize;

        // 1. Blocks: little-endian u16 per cell, all selected sections back to back.
        for s in 0..16u8 {
            if primary_bitmask & (1 << s) != 0 {
                let section = self.section_or_insert(s);
                let mut may_contain_tile_provider = false;
                let mut non_air = 0u16;
                for cell in section.blocks.iter_mut() {
                    let raw = u16::from(data[i]) | (u16::from(data[i + 1]) << 8);
                    *cell = raw;
                    non_air += u16::from(raw != 0);
                    may_contain_tile_provider |= BlockState::from_raw(raw).props().has_tile_entity;
                    i += 2;
                }
                section.may_contain_tile_provider = may_contain_tile_provider;
                section.non_air = non_air;
            } else if ground_up_continuous {
                self.sections[s as usize] = None;
            }
        }

        // 2. Block light: one 2048-byte nibble array per selected section.
        for s in 0..16u8 {
            if primary_bitmask & (1 << s) != 0 {
                let section = self.section_or_insert(s);
                section.block_light.copy_from_slice(&data[i..i + 2048]);
                i += 2048;
            }
        }

        // 3. Sky light: same shape as block light, but only in dimensions that have sky (H6).
        if has_sky_light {
            for s in 0..16u8 {
                if primary_bitmask & (1 << s) != 0 {
                    let section = self.section_or_insert(s);
                    section.sky_light.copy_from_slice(&data[i..i + 2048]);
                    i += 2048;
                }
            }
        }

        // 4. Biomes: once per column, only on a full-column send.
        if ground_up_continuous {
            self.biomes.copy_from_slice(&data[i..i + 256]);
            i += 256;
        }

        debug_assert_eq!(
            i, expected,
            "expected_chunk_data_len drifted from the unpack loop"
        );

        // java: `Chunk.fillChunk:1374` — the last thing `fillChunk` does before touching tile
        // entities. Note it is `generateHeightMap`, *not* `generateSkylightMap`: the server
        // already sent correct light, so a chunk packet costs one descent per column and no
        // flood fill at all.
        self.generate_height_map();
        Ok(())
    }

    /// One full-column (`ground_up_continuous`) payload unpacked into a fresh column: exactly the
    /// `Chunk::empty()` + `fill_from_packet(true, ..)` that [`crate::World::apply_chunk_data`] and
    /// [`crate::World::apply_chunk_bulk`] run in place, as a value another thread can build.
    /// Install the result with [`crate::World::install_full_column`] (#15 / NET-1).
    ///
    /// # Errors
    ///
    /// Exactly `fill_from_packet`'s: a payload shorter than the bitmask requires.
    pub fn from_full_column_packet(
        primary_bitmask: u16,
        has_sky_light: bool,
        data: &[u8],
    ) -> Result<Chunk, WorldError> {
        let mut chunk = Chunk::empty();
        chunk.fill_from_packet(true, primary_bitmask, has_sky_light, data)?;
        Ok(chunk)
    }
}

/// The byte length `Chunk::fill_from_packet` expects for a given bitmask/flags combination.
/// java: `S21PacketChunkData.func_180737_a`
#[must_use]
pub fn expected_chunk_data_len(
    primary_bitmask: u16,
    has_sky_light: bool,
    ground_up_continuous: bool,
) -> usize {
    let sections = primary_bitmask.count_ones() as usize;
    let blocks = sections * 2 * 16 * 16 * 16;
    let block_light = sections * 16 * 16 * 16 / 2;
    let sky_light = if has_sky_light {
        sections * 16 * 16 * 16 / 2
    } else {
        0
    };
    let biomes = if ground_up_continuous { 256 } else { 0 };
    blocks + block_light + sky_light + biomes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pre-NET-2 `generate_height_map`, cell by cell through `block_at`, kept verbatim as the
    /// oracle for the section-skipping version.
    fn generate_height_map_by_cells(chunk: &mut Chunk) {
        let top = chunk.top_filled_segment();
        chunk.height_map_minimum = i32::MAX;
        for x in 0..16u8 {
            for z in 0..16u8 {
                let idx = ((z as usize) << 4) | (x as usize);
                let mut height_done = false;
                let mut precipitation = -1;
                for y in (0..=(top + 15)).rev() {
                    if height_done && precipitation != -1 {
                        break;
                    }
                    let props = chunk.block_at(x, y as u8, z).props();
                    if !height_done && props.light_opacity != 0 {
                        chunk.height_map[idx] = y + 1;
                        if y + 1 < chunk.height_map_minimum {
                            chunk.height_map_minimum = y + 1;
                        }
                        height_done = true;
                    }
                    if precipitation == -1
                        && y > 0
                        && (props.material.blocks_movement() || props.material.is_liquid())
                    {
                        precipitation = y + 1;
                    }
                }
                chunk.precipitation_height_map[idx] = precipitation;
            }
        }
    }

    /// `non_air` stays the exact number of non-zero raw cells through
    /// every write path: `set_block` in any order (air over air, block over air, air over block,
    /// block over block, illegal ids), clearing a section back to air, and the packet unpacker
    /// (all-air, sparse and full sections, one column refilled), so `is_raw_air` is exact.
    #[test]
    fn w19_o2_non_air_count_tracks_every_block_write() {
        let recount = |s: &ChunkSection| s.blocks.iter().filter(|&&raw| raw != 0).count() as u16;
        let mut seed = 0x05EE_D0A2_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut section = ChunkSection::empty();
        assert!(section.is_raw_air());
        let mut peak = 0;
        for step in 0..20_000 {
            let (x, y, z) = (
                (next() % 16) as u8,
                (next() % 16) as u8,
                (next() % 16) as u8,
            );
            let raw = match next() % 4 {
                0 | 1 => 0,
                2 => 1 << 4,
                _ => (next() >> 16) as u16 & 0x1FFF,
            };
            section.set_block(x, y, z, BlockState::from_raw(raw));
            let exact = recount(&section);
            assert_eq!(section.non_air_count(), exact, "step {step}");
            assert_eq!(section.is_raw_air(), exact == 0, "step {step}");
            peak = peak.max(exact);
        }
        assert!(
            peak > 1000,
            "the writes must fill the section well past empty: {peak}"
        );
        for y in 0..16u8 {
            for z in 0..16u8 {
                for x in 0..16u8 {
                    section.set_block(x, y, z, BlockState::AIR);
                }
            }
        }
        assert_eq!(section.non_air_count(), 0);
        assert!(section.is_raw_air());

        let mut chunk = Chunk::empty();
        for round in 0..3u16 {
            let mask: u16 = 0b0111 << round;
            let mut data = Vec::new();
            for s in 0..3u32 {
                for cell in 0..4096u32 {
                    let raw: u16 = match s {
                        0 => 0,
                        1 if cell % 97 == u32::from(round) => 1 << 4,
                        1 => 0,
                        _ => (cell % 400) as u16 + 1,
                    };
                    data.extend_from_slice(&raw.to_le_bytes());
                }
            }
            data.resize(data.len() + 3 * 2048 * 2 + 256, 0);
            chunk
                .fill_from_packet(true, mask, true, &data)
                .expect("a well-formed payload");
            for si in 0..16u8 {
                if let Some(section) = chunk.section(si) {
                    assert_eq!(section.non_air_count(), recount(section), "{round} {si}");
                }
            }
            assert!(chunk.section(round as u8).expect("sent").is_raw_air());
            assert!(!chunk.section(round as u8 + 1).expect("sent").is_raw_air());
            assert!(!chunk.section(round as u8 + 2).expect("sent").is_raw_air());
        }
    }

    /// NET-2: skipping absent sections must leave `height_map`, `precipitation_height_map` and
    /// `height_map_minimum` bit-identical to the cell descent — including vanilla's "keep the old
    /// value when nothing is opaque" quirk (seeded heightmaps), illegal raw ids (read as air),
    /// gaps between present sections, and all-air columns.
    #[test]
    fn height_map_section_skip_matches_the_cell_descent() {
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for case in 0..300 {
            let mut chunk = Chunk::empty();
            // Case 1 is the all-air column (bitmask 0 in S26): no section at all.
            let present = if case == 1 { 0 } else { next() as u16 };
            // Sparse, void-map-like columns most of the time; dense ones sometimes.
            let air_per_mille = [990u64, 900, 500, 0][case % 4];
            for s in 0..16u8 {
                if present & (1 << s) == 0 && case % 7 != 0 {
                    continue;
                }
                let section = chunk.section_or_insert(s);
                for cell in section.blocks.iter_mut() {
                    let roll = next();
                    *cell = if roll % 1000 < air_per_mille {
                        0
                    } else {
                        // Any raw value: legal states of every block, and illegal ids.
                        (roll >> 16) as u16 & 0x1FFF
                    };
                }
                // This fixture writes `blocks` directly.
                section.recount_non_air();
            }
            if case % 3 == 0 {
                // A partial packet re-runs the descent over an already-filled heightmap.
                for value in &mut chunk.height_map {
                    *value = (next() % 257) as i32;
                }
            }
            let mut oracle = chunk.clone();
            generate_height_map_by_cells(&mut oracle);
            chunk.generate_height_map();
            assert_eq!(chunk.height_map, oracle.height_map, "case {case}");
            assert_eq!(
                chunk.precipitation_height_map, oracle.precipitation_height_map,
                "case {case}"
            );
            assert_eq!(
                chunk.height_map_minimum, oracle.height_map_minimum,
                "case {case}"
            );
        }
    }

    /// Measurement aid for #15 / NET-2, not a gate: per-column cost of the heightmap descent
    /// (cell walk vs section skip) on void-map shapes.
    /// `cargo test -p mc-world --lib heightmap_cost_probe -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing probe; run explicitly"]
    fn heightmap_cost_probe() {
        use std::time::Instant;
        // An island column: sections 3-4 present, a 3-block-thick floor at y=60..62, air above.
        let mut island = Chunk::empty();
        for x in 0..16u8 {
            for z in 0..16u8 {
                for y in 60..63u8 {
                    island.set_block_at(x, y, z, BlockState::new(mc_data::BlockId(1), 0));
                }
                island.set_block_at(x, 70, z, BlockState::new(mc_data::BlockId(0), 0));
            }
        }
        let air = Chunk::empty();
        let runs = 2000u32;
        for (name, column) in [("island", &island), ("all-air", &air)] {
            let mut work = column.clone();
            let start = Instant::now();
            for _ in 0..runs {
                generate_height_map_by_cells(&mut work);
            }
            let cells = start.elapsed() / runs;
            let start = Instant::now();
            for _ in 0..runs {
                work.generate_height_map();
            }
            let skip = start.elapsed() / runs;
            eprintln!("heightmap_cost_probe {name}: cell walk {cells:?}, section skip {skip:?}");
        }
    }

    #[test]
    fn section_indexing_round_trips() {
        let mut s = ChunkSection::empty();
        for x in 0..16u8 {
            for y in 0..16u8 {
                for z in 0..16u8 {
                    let state = BlockState::from_raw(
                        u16::from(x) ^ (u16::from(y) << 4) ^ (u16::from(z) << 9),
                    );
                    s.set_block(x, y, z, state);
                    assert_eq!(s.blocks[local_index(x, y, z)], state.id, "{x},{y},{z}");
                }
            }
        }
    }

    #[test]
    fn nibble_packing_matches_java_low_high_order() {
        let mut arr = [0u8; 2048];
        // (0,0,0) -> index 0 (even -> low nibble), (1,0,0) -> index 1 (odd -> high nibble),
        // both land in byte 0.
        nibble_set(&mut arr, 0, 0, 0, 0xA);
        nibble_set(&mut arr, 1, 0, 0, 0xB);
        assert_eq!(arr[0], 0xBA);
        assert_eq!(nibble_get(&arr, 0, 0, 0), 0xA);
        assert_eq!(nibble_get(&arr, 1, 0, 0), 0xB);
    }

    #[test]
    fn nibble_values_are_masked_to_four_bits() {
        let mut arr = [0u8; 2048];
        nibble_set(&mut arr, 5, 5, 5, 0xFF);
        assert_eq!(nibble_get(&arr, 5, 5, 5), 0x0F);
    }

    #[test]
    fn chunk_block_at_maps_section_index_and_local_y() {
        let mut c = Chunk::empty();
        c.set_block_at(3, 20, 7, BlockState::new(mc_data::BlockId(1), 0));
        // y=20 -> section 1, local y=4
        assert_eq!(
            c.section(1).unwrap().block(3, 4, 7).id,
            BlockState::new(mc_data::BlockId(1), 0).id
        );
        assert_eq!(c.block_at(3, 20, 7).block().0, 1);
        assert!(c.section(0).is_none());
    }

    #[test]
    fn cloned_sections_share_until_block_and_light_mutations_detach() {
        let pos = (1, 4, 2);
        let original_state = BlockState::new(mc_data::BlockId(1), 0);
        let changed_state = BlockState::new(mc_data::BlockId(2), 0);
        let mut live = Chunk::empty();
        live.set_block_at(pos.0, pos.1, pos.2, original_state);
        live.set_block_at(1, 20, 2, original_state);
        live.section_mut(0)
            .unwrap()
            .set_block_light(pos.0, pos.1, pos.2, 3);

        let mut snapshot = live.clone();
        assert!(Arc::ptr_eq(
            live.sections[0].as_ref().unwrap(),
            snapshot.sections[0].as_ref().unwrap(),
        ));
        assert!(Arc::ptr_eq(
            live.sections[1].as_ref().unwrap(),
            snapshot.sections[1].as_ref().unwrap(),
        ));

        snapshot.set_block_at(pos.0, pos.1, pos.2, changed_state);
        snapshot
            .section_mut(0)
            .unwrap()
            .set_block_light(pos.0, pos.1, pos.2, 9);
        assert_eq!(live.block_at(pos.0, pos.1, pos.2), original_state);
        assert_eq!(snapshot.block_at(pos.0, pos.1, pos.2), changed_state);
        assert_eq!(live.section(0).unwrap().block_light(pos.0, pos.1, pos.2), 3);
        assert_eq!(
            snapshot
                .section(0)
                .unwrap()
                .block_light(pos.0, pos.1, pos.2),
            9
        );
        assert!(!Arc::ptr_eq(
            live.sections[0].as_ref().unwrap(),
            snapshot.sections[0].as_ref().unwrap(),
        ));
        assert!(Arc::ptr_eq(
            live.sections[1].as_ref().unwrap(),
            snapshot.sections[1].as_ref().unwrap(),
        ));

        let second_snapshot = live.clone();
        live.section_mut(0)
            .unwrap()
            .set_sky_light(pos.0, pos.1, pos.2, 15);
        assert_eq!(live.section(0).unwrap().sky_light(pos.0, pos.1, pos.2), 15);
        assert_eq!(
            second_snapshot
                .section(0)
                .unwrap()
                .sky_light(pos.0, pos.1, pos.2),
            0
        );
        assert!(!Arc::ptr_eq(
            live.sections[0].as_ref().unwrap(),
            second_snapshot.sections[0].as_ref().unwrap(),
        ));
    }

    #[test]
    fn absent_sections_are_created_independently_after_clone() {
        let mut live = Chunk::empty();
        let mut snapshot = live.clone();

        live.set_block_at(3, 35, 7, BlockState::new(mc_data::BlockId(1), 0));
        assert!(live.section(2).is_some());
        assert!(snapshot.section(2).is_none());

        snapshot.section_or_insert(2).set_block_light(3, 3, 7, 11);
        assert_eq!(snapshot.section(2).unwrap().block_light(3, 3, 7), 11);
        assert_eq!(live.section(2).unwrap().block_light(3, 3, 7), 0);
        assert!(!Arc::ptr_eq(
            live.sections[2].as_ref().unwrap(),
            snapshot.sections[2].as_ref().unwrap(),
        ));
    }

    #[test]
    fn full_packet_replacement_and_removal_preserve_cloned_sections() {
        let old_state = BlockState::new(mc_data::BlockId(1), 0);
        let new_state = BlockState::new(mc_data::BlockId(2), 0);
        let mut live = Chunk::empty();
        live.set_block_at(0, 0, 0, old_state);
        live.set_block_at(0, 16, 0, old_state);
        let snapshot = live.clone();

        let mut replacement = vec![0; expected_chunk_data_len(1, true, true)];
        replacement[0] = new_state.id as u8;
        replacement[1] = (new_state.id >> 8) as u8;
        live.fill_from_packet(true, 1, true, &replacement).unwrap();
        assert_eq!(live.block_at(0, 0, 0), new_state);
        assert_eq!(snapshot.block_at(0, 0, 0), old_state);
        assert!(live.section(1).is_none());
        assert_eq!(snapshot.block_at(0, 16, 0), old_state);
        assert!(!Arc::ptr_eq(
            live.sections[0].as_ref().unwrap(),
            snapshot.sections[0].as_ref().unwrap(),
        ));

        let before_removal = live.clone();
        live.fill_from_packet(true, 0, true, &[0; 256]).unwrap();
        assert!(live.section(0).is_none());
        assert_eq!(before_removal.block_at(0, 0, 0), new_state);
        assert!(!Arc::ptr_eq(
            before_removal.sections[0].as_ref().unwrap(),
            snapshot.sections[0].as_ref().unwrap(),
        ));
    }

    #[test]
    fn partial_packet_detaches_selected_and_preserves_unselected_shared_section() {
        let old_state = BlockState::new(mc_data::BlockId(1), 0);
        let new_state = BlockState::new(mc_data::BlockId(4), 0);
        let mut original = Chunk::empty();
        original.set_block_at(0, 0, 0, old_state);
        original.set_block_at(0, 16, 0, old_state);
        let mut updated = original.clone();

        let mut partial = vec![0; expected_chunk_data_len(1, true, false)];
        partial[0] = new_state.id as u8;
        partial[1] = (new_state.id >> 8) as u8;
        updated.fill_from_packet(false, 1, true, &partial).unwrap();

        assert_eq!(original.block_at(0, 0, 0), old_state);
        assert_eq!(updated.block_at(0, 0, 0), new_state);
        assert!(Arc::ptr_eq(
            original.sections[1].as_ref().unwrap(),
            updated.sections[1].as_ref().unwrap(),
        ));
        assert!(!Arc::ptr_eq(
            original.sections[0].as_ref().unwrap(),
            updated.sections[0].as_ref().unwrap(),
        ));
    }

    #[test]
    fn world_render_snapshot_sections_isolate_block_and_light_writes_both_ways() {
        let pos = BlockPos::new(1, 64, 2);
        let original_state = BlockState::new(mc_data::BlockId(1), 0);
        let live_state = BlockState::new(mc_data::BlockId(4), 0);
        let snapshot_state = BlockState::new(mc_data::BlockId(16), 0);
        let mut world = crate::World::new();
        let mut payload = vec![0; expected_chunk_data_len(1 << 4, true, true)];
        let block_offset =
            ((pos.y as usize & 15) << 8) | ((pos.z as usize & 15) << 4) | (pos.x as usize & 15);
        let initial_id = original_state.id.to_le_bytes();
        payload[block_offset * 2] = initial_id[0];
        payload[block_offset * 2 + 1] = initial_id[1];
        world
            .apply_chunk_data(pos.chunk_pos(), true, 1 << 4, &payload)
            .unwrap();
        let mut snapshot = world.render_snapshot();

        world.set_block(pos, live_state);
        world.set_light_for(crate::SkyBlock::Block, pos, 7);
        assert_eq!(world.block_at(pos), live_state);
        assert_eq!(world.block_light_at(pos), 7);
        assert_eq!(snapshot.block_at(pos), original_state);
        assert_eq!(snapshot.block_light_at(pos), 0);

        snapshot.set_block(pos, snapshot_state);
        snapshot.set_light_for(crate::SkyBlock::Block, pos, 11);
        assert_eq!(snapshot.block_at(pos), snapshot_state);
        assert_eq!(snapshot.block_light_at(pos), 11);
        assert_eq!(world.block_at(pos), live_state);
        assert_eq!(world.block_light_at(pos), 7);
    }

    #[test]
    fn biome_index_is_z_major() {
        let mut c = Chunk::empty();
        c.set_biome(1, 2, 12);
        assert_eq!(c.biomes()[2 * 16 + 1], 12);
        assert_eq!(c.biome(1, 2), 12);
    }

    #[test]
    fn unregistered_biome_ids_resolve_to_plains_without_rewriting_wire_storage() {
        let mut c = Chunk::empty();
        for raw in [40, 42, 127, 168, 254, 255] {
            c.set_biome(3, 5, raw);
            assert_eq!(c.biome(3, 5), 1, "raw biome {raw}");
            assert_eq!(c.biomes()[5 * 16 + 3], raw, "raw biome {raw}");
        }

        c.set_biome(3, 5, 12);
        assert_eq!(c.biome(3, 5), 12, "registered ids must not collapse");
    }

    #[test]
    fn unset_biome_queries_generated_source_once_and_caches_its_answer() {
        let mut c = Chunk::empty();
        let p = BlockPos::new(-17, 64, -33);
        assert_eq!(
            c.biome_with_generated(p, |got| {
                assert_eq!(
                    got, p,
                    "the provider receives the unmodified world position"
                );
                Some(6) // swamp, deliberately not Plains
            }),
            6
        );
        // -17/-33 map to local (15, 15), preserving Java's `& 15` behaviour for negative
        // columns rather than truncating toward zero.
        assert_eq!(c.biomes()[15 * 16 + 15], 6);
        assert_eq!(
            c.biome_with_generated(BlockPos::new(-17, 200, -33), |_| -> Option<u8> {
                panic!("a cached generated biome must not call its provider again")
            }),
            6
        );
    }

    #[test]
    fn received_biome_never_queries_generated_source() {
        let mut c = Chunk::empty();
        c.set_biome(3, 5, 12);
        assert_eq!(
            c.biome_with_generated(BlockPos::new(3, 64, 5), |_| -> Option<u8> {
                panic!("a received wire biome is authoritative")
            }),
            12
        );
    }

    #[test]
    fn generated_source_none_uses_the_explicit_plains_default_and_caches_it() {
        let mut c = Chunk::empty();
        let p = BlockPos::new(1, 64, 2);
        assert_eq!(c.biome_with_generated(p, |_| None), 1);
        assert_eq!(c.biomes()[2 * 16 + 1], 1);
    }

    #[test]
    fn expected_len_matches_java_formula() {
        // One section, overworld, full column: 8192 + 2048 + 2048 + 256.
        assert_eq!(
            expected_chunk_data_len(0b1, true, true),
            8192 + 2048 + 2048 + 256
        );
        // Same section, no sky (nether/end): no skylight region.
        assert_eq!(expected_chunk_data_len(0b1, false, true), 8192 + 2048 + 256);
        // Partial (non-full-column) update: no biome region.
        assert_eq!(
            expected_chunk_data_len(0b1, true, false),
            8192 + 2048 + 2048
        );
        // Empty bitmask, full column: only biomes (an unload-shaped payload).
        assert_eq!(expected_chunk_data_len(0, true, true), 256);
    }

    #[test]
    fn fill_from_packet_rejects_truncated_data() {
        let mut c = Chunk::empty();
        let err = c.fill_from_packet(true, 0b1, true, &[0u8; 10]).unwrap_err();
        assert!(matches!(err, WorldError::ChunkDataLength { .. }));
    }

    #[test]
    fn fill_from_packet_ignores_bytes_after_the_vanilla_payload() {
        let expected = expected_chunk_data_len(0, true, true);
        let mut data = vec![7u8; expected];
        data.extend_from_slice(&[99, 100, 101]);

        let mut c = Chunk::empty();
        c.fill_from_packet(true, 0, true, &data).unwrap();
        assert!(c.sections().iter().all(Option::is_none));
        assert!(c.biomes().iter().all(|&biome| biome == 7));

        let short = &data[..expected - 1];
        assert!(matches!(
            c.fill_from_packet(true, 0, true, short),
            Err(WorldError::ChunkDataLength { .. })
        ));
    }

    #[test]
    fn empty_section_reports_empty() {
        let mut s = ChunkSection::empty();
        assert!(s.is_empty());
        s.set_block(0, 0, 0, BlockState::new(mc_data::BlockId(1), 0));
        assert!(!s.is_empty());
    }

    #[test]
    fn invalid_stored_states_read_as_air_and_do_not_make_a_section_nonempty() {
        for raw in [1, 15, 0x0FF0, 0xFFFF] {
            let mut s = ChunkSection::empty();
            let invalid = BlockState::from_raw(raw);
            assert!(!invalid.is_legal(), "negative control: {raw:#06x}");
            s.set_block(4, 5, 6, invalid);
            assert_eq!(s.block(4, 5, 6), BlockState::AIR, "raw {raw:#06x}");
            assert!(s.is_empty(), "raw {raw:#06x}");
        }

        let mut control = ChunkSection::empty();
        let stone = BlockState::new(mc_data::BlockId(1), 0);
        control.set_block(4, 5, 6, stone);
        assert_eq!(control.block(4, 5, 6), stone);
        assert!(!control.is_empty());
    }

    #[test]
    fn writing_an_air_block_variant_into_a_missing_section_is_unchanged() {
        for raw in [1, 15, 0x0FF0, 0xFFFF] {
            let mut c = Chunk::empty();
            let air = BlockState::from_raw(raw);
            assert!(air.props().is_air(), "negative control: {raw:#06x}");
            assert_eq!(
                c.set_block_relit(2, 35, 7, air),
                ChunkSetOutcome::Unchanged,
                "raw {raw:#06x}"
            );
            assert!(c.section(2).is_none(), "raw {raw:#06x} allocated storage");
        }

        let mut control = Chunk::empty();
        let stone = BlockState::new(mc_data::BlockId(1), 0);
        assert!(matches!(
            control.set_block_relit(2, 35, 7, stone),
            ChunkSetOutcome::Changed { .. }
        ));
        assert!(control.section(2).is_some());
    }
    #[test]
    fn provider_hint_is_sticky_for_writes_and_survives_clone() {
        let provider = BlockState::new(mc_data::BlockId(54), 2);
        let stone = BlockState::new(mc_data::BlockId(1), 0);
        let mut section = ChunkSection::empty();
        assert!(!section.may_contain_tile_provider());
        section.set_block_light(1, 2, 3, 15);
        section.set_sky_light(1, 2, 3, 15);
        section.set_block(1, 2, 3, stone);
        assert!(!section.may_contain_tile_provider());
        section.set_block(1, 2, 3, provider);
        assert!(section.may_contain_tile_provider());
        section.set_block(1, 2, 3, stone);
        assert!(section.may_contain_tile_provider());
        assert!(section.clone().may_contain_tile_provider());
    }

    #[test]
    fn invalid_provider_metadata_does_not_create_a_tile_hint() {
        let raw = BlockState::from_raw((54u16 << 4) | 15);
        assert!(!raw.is_legal());
        let mut section = ChunkSection::empty();
        section.set_block(4, 5, 6, raw);
        assert!(!section.may_contain_tile_provider());
        assert_eq!(section.block(4, 5, 6), BlockState::AIR);
    }

    #[test]
    fn complete_packet_section_recomputes_provider_hint() {
        let provider = BlockState::new(mc_data::BlockId(54), 2);
        let mut chunk = Chunk::empty();
        let mut data = vec![0u8; expected_chunk_data_len(1, false, true)];
        let offset = local_index(1, 2, 3) * 2;
        data[offset..offset + 2].copy_from_slice(&provider.id.to_le_bytes());
        chunk.fill_from_packet(true, 1, false, &data).unwrap();
        assert!(chunk.section(0).unwrap().may_contain_tile_provider());
        assert_eq!(chunk.block_at(1, 2, 3), provider);
        data[offset..offset + 2].fill(0);
        chunk.fill_from_packet(true, 1, false, &data).unwrap();
        assert!(!chunk.section(0).unwrap().may_contain_tile_provider());
    }
}
