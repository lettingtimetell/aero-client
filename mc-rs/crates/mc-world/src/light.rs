//! Client-side light propagation.
//!
//! java: `world/World.java`'s `checkLight` (`:2767`), `getRawLight` (`:2783`) and
//! `checkLightFor` (`:2836`), plus the chunk-side helpers `Chunk.relightBlock` (`:400`),
//! `generateSkylightMap` (`:246`), `propagateSkylightOcclusion` (`:316`), `recheckGaps` (`:322`)
//! and `enqueueRelightChecks` (`:1439`).
//!
//! # Why the client has to do this at all
//!
//! 1.8's wire protocol has **no light-update packet**. `S23PacketBlockChange` carries a state id
//! and nothing else, and the server never re-sends a column after a single block change. Both
//! sides therefore recompute light independently: `World.setBlockState:369` calls `checkLight`
//! with the `!isRemote` guard applied only to the *neighbour-notify* half, never to the light
//! half. Without this module a torch lights only its own cell, a wall broken into a sealed room
//! stays black, and a mineshaft is dark for the rest of the session.
//!
//! # Shape
//!
//! Vanilla's `Chunk` holds a back-reference to its `World` and calls into it mid-write
//! (`relightBlock` → `worldObj.markBlocksDirtyVertical` → `worldObj.checkLightFor`). The arena-index rule forbids that cycle, so the split here is: [`crate::Chunk`] does everything that is
//! chunk-local and *reports* what is owed ([`crate::chunk::ChunkSetOutcome`],
//! [`RelightOutcome`]), and every method that needs the whole world lives in this file as a
//! second `impl World` block.
//!
//! # Cost
//!
//! `checkLightFor` is a bounded flood fill: at most 32,768 queue entries, and every push is
//! gated on `|dx| + |dy| + |dz| < 17`, so one call can never leave a 17-block ball. The scratch
//! queue is a `World` field rather than a per-call allocation, exactly as vanilla's
//! `lightUpdateBlockList` is. The periodic repair pass ([`World::tick_chunk_light`]) takes an
//! explicit per-tick column budget instead of vanilla's wall-clock one, so a replay is
//! deterministic.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use mc_data::BlockState;
use mc_math::{BlockPos, ChunkPos, ChunkSectionPos};

use crate::chunk::ChunkSetOutcome;
use crate::{Chunk, ChunkSection, World};

/// One storage write collected by [`World::apply_block_changes`].  The chunk-local half of
/// `Chunk.setBlockState` can run for every entry without entering a world-wide light flood fill;
/// the world finishes all collected writes together below.  java: `Chunk.setBlockState:658-741`
/// followed by the client `S22PacketMultiBlockChange` loop.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PendingBlockChange {
    pub(crate) position: BlockPos,
    pub(crate) state: BlockState,
    pub(crate) old: BlockState,
    pub(crate) outcome: ChunkSetOutcome,
}

/// The cached `World.activeChunkSet` behind [`World::tick_chunk_relight_active_around`].
///
/// java: `World.setActivePlayerChunksAndCheckLight` rebuilds the set from scratch every tick:
/// `(2r+1)²` inserts per player, 12k–22k SipHash inserts for 20 players (#6/C1-2/TICK-1). The set
/// is a pure function of the deduplicated player chunk centres and `r`, so it is rebuilt only
/// when that key changes, directly in this port's `(x, z)` replay order and without hashing.
/// No RNG is involved: the client's `playMoodSoundAndCheckLight` mood branch is `!isRemote`, and
/// the one `light_check_rng` sample is drawn by the caller from its player list, not from this
/// set.
#[derive(Debug, Default, Clone)]
pub(crate) struct ActiveLightChunks {
    /// The key: deduplicated player chunk centres sorted by `(x, z)`, plus the radius.
    centres: Vec<ChunkPos>,
    radius: i32,
    /// Whether `centres`/`radius` describe `ordered`. False until the first keyed build and
    /// after [`World::tick_chunk_relight_active`] replaced `ordered` with an arbitrary set.
    keyed: bool,
    /// `activeChunkSet`, sorted by `(x, z)` and deduplicated.
    pub(crate) ordered: Vec<ChunkPos>,
    /// Every entry of `ordered[..scan_from]` is in `World::previous_active_light_chunks`.
    scan_from: usize,
    /// The columns the last call consumed, in visit order (reused allocation).
    pub(crate) picked: Vec<ChunkPos>,
    /// Scratch for key normalisation and the union sweep.
    scratch: Vec<ChunkPos>,
    zs: Vec<i32>,
}

impl ActiveLightChunks {
    /// Re-key on `centres`/`radius`. Returns `true` when `ordered` was rebuilt.
    fn update(&mut self, centres: &[ChunkPos], radius: i32) -> bool {
        self.scratch.clear();
        self.scratch.extend_from_slice(centres);
        self.scratch.sort_unstable_by_key(|pos| (pos.x, pos.z));
        self.scratch.dedup();
        if self.keyed && self.radius == radius && self.scratch == self.centres {
            return false;
        }
        std::mem::swap(&mut self.centres, &mut self.scratch);
        self.radius = radius;
        self.keyed = true;
        chunk_square_union(&self.centres, radius, &mut self.zs, &mut self.ordered);
        true
    }
}

/// The union of the `(2r+1)²` chunk squares around `centres`, sorted by `(x, z)` and
/// deduplicated: exactly what collecting `centre + (dx, dz)` for `dx, dz in -r..=r` into a set
/// and sorting it yields, without hashing.
///
/// `centres` must be sorted by `(x, z)`. The sweep walks x and keeps the sorted multiset of the
/// z of every centre whose x-range covers the current x; equal-width z-intervals sorted by start
/// merge in one pass.
pub(crate) fn chunk_square_union(
    centres: &[ChunkPos],
    radius: i32,
    zs: &mut Vec<i32>,
    out: &mut Vec<ChunkPos>,
) {
    out.clear();
    if centres.is_empty() || radius < 0 {
        return;
    }
    // A centre within `radius` of the i32 range is far outside the 30M-block world (a bogus
    // teleport); keep the literal per-square construction there, with release-mode wrapping.
    let limit = i32::MAX - radius - 1;
    if centres
        .iter()
        .any(|c| c.x > limit || c.x < -limit || c.z > limit || c.z < -limit)
    {
        for c in centres {
            for dx in -radius..=radius {
                for dz in -radius..=radius {
                    out.push(ChunkPos::new(c.x.wrapping_add(dx), c.z.wrapping_add(dz)));
                }
            }
        }
        out.sort_unstable_by_key(|pos| (pos.x, pos.z));
        out.dedup();
        return;
    }
    zs.clear();
    let (mut enter, mut leave) = (0, 0);
    let mut x = centres[0].x - radius;
    let x_end = centres[centres.len() - 1].x + radius;
    while x <= x_end {
        while enter < centres.len() && centres[enter].x - radius <= x {
            let z = centres[enter].z;
            let at = zs.partition_point(|&v| v < z);
            zs.insert(at, z);
            enter += 1;
        }
        // Equal widths: squares leave in the order they entered.
        while leave < enter && centres[leave].x + radius < x {
            let z = centres[leave].z;
            let at = zs.partition_point(|&v| v < z);
            zs.remove(at);
            leave += 1;
        }
        if zs.is_empty() {
            // No square covers this x. Some centre has not entered yet, otherwise
            // `x > x_end`; jump to its first column.
            x = centres[enter].x - radius;
            continue;
        }
        let mut start = zs[0] - radius;
        let mut end = zs[0] + radius;
        for &z in &zs[1..] {
            if z - radius > end + 1 {
                out.extend((start..=end).map(|z| ChunkPos::new(x, z)));
                start = z - radius;
            }
            end = z + radius;
        }
        out.extend((start..=end).map(|z| ChunkPos::new(x, z)));
        x += 1;
    }
}

/// Merge `add` into `into`; both sorted by `(x, z)` and disjoint.
fn merge_sorted_disjoint(into: &mut Vec<ChunkPos>, add: &[ChunkPos]) {
    let Some(first) = add.first() else {
        return;
    };
    if into
        .last()
        .is_none_or(|last| (last.x, last.z) < (first.x, first.z))
    {
        // The steady state: the cycle advances past every consumed column.
        into.extend_from_slice(add);
        return;
    }
    let mut a = into.len();
    into.extend_from_slice(add);
    let mut b = add.len();
    let mut w = into.len();
    while b > 0 {
        w -= 1;
        if a > 0 && (into[a - 1].x, into[a - 1].z) > (add[b - 1].x, add[b - 1].z) {
            into[w] = into[a - 1];
            a -= 1;
        } else {
            into[w] = add[b - 1];
            b -= 1;
        }
    }
}

/// Which of the two light channels a query or an update is about. java: `world/EnumSkyBlock.java`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SkyBlock {
    /// Daylight, propagated downward at full strength through anything transparent.
    /// `EnumSkyBlock.SKY`, `defaultLightValue = 15`.
    Sky,
    /// Emitted light (torches, lava, glowstone). `EnumSkyBlock.BLOCK`, `defaultLightValue = 0`.
    Block,
}

impl SkyBlock {
    /// What an unloaded column reports for this channel. java: `EnumSkyBlock.defaultLightValue`
    #[inline]
    #[must_use]
    pub const fn default_light_value(self) -> u8 {
        match self {
            SkyBlock::Sky => 15,
            SkyBlock::Block => 0,
        }
    }
}

/// The six neighbour offsets in `EnumFacing.values()` order: DOWN, UP, NORTH, SOUTH, WEST, EAST.
///
/// The order is load-bearing, not cosmetic: `getRawLight` returns early the moment a neighbour
/// pushes the running maximum to `14`, so a different iteration order can legitimately produce
/// `14` where vanilla produces `15` (and vice versa) for the same cell.
const FACINGS: [(i32, i32, i32); 6] = [
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
    (-1, 0, 0),
    (1, 0, 0),
];

/// `EnumFacing.Plane.HORIZONTAL`'s iteration order, which is **NORTH, EAST, SOUTH, WEST**
/// (`EnumFacing.Plane.facings():454`) — not the declaration order of the enum.
const HORIZONTAL: [(i32, i32); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

/// The spread pass's push order, which `checkLightFor` writes out by hand as
/// `west, east, down, up, north, south` (`World.java:2931-2962`) rather than looping
/// `EnumFacing.values()`.
///
/// This is **not** the same order as [`FACINGS`], and the difference is observable: the fill's
/// 17-block reach is measured from its origin, so a different visit order changes which cells at
/// the frontier are reached before the bound cuts the expansion off. Using `values()` here made
/// two of the nine parity fixtures disagree with vanilla in exactly that boundary shell.
const SPREAD_ORDER: [(i32, i32, i32); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

/// Vanilla's `lightUpdateBlockList` length (`World.java:158`). Every push is bounds-checked
/// against it, so a fill that would overflow simply stops expanding rather than growing.
const LIGHT_UPDATE_LIST_LEN: usize = 32768;

/// What [`crate::Chunk::relight_finish`] left for [`World`] to do. java: the trailing
/// `worldObj.…` calls of `Chunk.relightBlock` (`:495-503`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RelightOutcome {
    /// `updateSkylightNeighborHeight(…, <0>, <1>)` for the four horizontal neighbours and for
    /// the column itself.
    pub(crate) neighbour_range: (i32, i32),
    /// The inclusive Y range sky light was actually written into, or `None`. This is the union
    /// of the per-cell `notifyLightSet` boxes, and marking only it is what keeps a single block
    /// placement from dirtying all 144 sections of a 3×3 column neighbourhood.
    pub(crate) written: Option<(i32, i32)>,
}

/// A block-local `(x, z)` pair. Java's `& 15` on a negative `int` two's-complements to the same
/// low-four-bits answer Rust's `&` gives, so negative coordinates need no separate case.
#[inline]
fn local_xz(p: BlockPos) -> (u8, u8) {
    ((p.x & 15) as u8, (p.z & 15) as u8)
}

/// java: `World.isValid` (`:240`) — inside the ±30,000,000 horizontal limit and `0..256` tall.
#[inline]
fn is_valid(p: BlockPos) -> bool {
    p.x >= -30_000_000
        && p.z >= -30_000_000
        && p.x < 30_000_000
        && p.z < 30_000_000
        && (0..256).contains(&p.y)
}

// ---- resolved-column light access ------------------------------------
//
// The reference transcription below pays one SipHash `chunks` lookup per light/block read and
// two `get_mut`s per write: ~15-20 hashes per flood-fill cell, plus 9-16 `contains_key`s per
// `checkLightFor` origin for `isAreaLoaded`. Nothing about the *algorithm* needs that: every
// column a fill can touch is known before it starts. The helpers here are the same `World`
// reads written once against [`ColumnRead`], so the cached probe (no-op origin test) and the
// fill's resolved view share one transcription. Order of reads, writes, pushes and the 17-block
// bound are unchanged, so light, storage and the dirty set stay bit-identical; the
// `light_equivalence` tests run both engines side by side to pin that.

/// Read access to resident columns. `None` is the client provider's blank `EmptyChunk`.
trait ColumnRead {
    fn column(&self, cpos: ChunkPos) -> Option<&Chunk>;
}

/// [`World::light_for`] against a [`ColumnRead`]. Identical arms, including the absent-section
/// `canSeeSky ? default : 0` answer that ignores the dimension's sky flag (as `Chunk.getLightFor`
/// does).
#[inline]
fn light_for_in(src: &impl ColumnRead, has_sky: bool, kind: SkyBlock, p: BlockPos) -> u8 {
    let p = if p.y < 0 {
        BlockPos::new(p.x, 0, p.z)
    } else {
        p
    };
    if !is_valid(p) {
        return kind.default_light_value();
    }
    let (x, z) = local_xz(p);
    let Some(chunk) = src.column(p.chunk_pos()) else {
        return kind.default_light_value();
    };
    match chunk.section((p.y >> 4) as u8) {
        None => {
            if chunk.can_see_sky(x, p.y, z) {
                kind.default_light_value()
            } else {
                0
            }
        }
        Some(section) => match kind {
            SkyBlock::Sky if !has_sky => 0,
            SkyBlock::Sky => section.sky_light(x, (p.y & 15) as u8, z),
            SkyBlock::Block => section.block_light(x, (p.y & 15) as u8, z),
        },
    }
}

/// [`World::block_at`] against a [`ColumnRead`], **without** the DEBUG_WORLD branch: callers
/// only use it when [`World::uses_reference_light_engine`] is `false`, which excludes that type.
#[inline]
fn block_at_in(src: &impl ColumnRead, p: BlockPos) -> BlockState {
    if !(0..256).contains(&p.y) {
        return BlockState::AIR;
    }
    match src.column(p.chunk_pos()) {
        Some(chunk) => {
            let (x, z) = local_xz(p);
            chunk.block_at(x, p.y as u8, z)
        }
        None => BlockState::AIR,
    }
}

/// [`World::raw_light`] against a [`ColumnRead`]: same early outs, same `FACINGS` order.
#[inline]
fn raw_light_in(src: &impl ColumnRead, has_sky: bool, p: BlockPos, kind: SkyBlock) -> i32 {
    if kind == SkyBlock::Sky {
        let (x, z) = local_xz(p);
        // `World::can_see_sky`: `false` for an absent column.
        if src
            .column(p.chunk_pos())
            .is_some_and(|c| c.can_see_sky(x, p.y, z))
        {
            return 15;
        }
    }
    let props = block_at_in(src, p).props();
    let mut i = match kind {
        SkyBlock::Sky => 0,
        SkyBlock::Block => i32::from(props.light_emission),
    };
    let mut j = i32::from(props.light_opacity);
    if j >= 15 && props.light_emission > 0 {
        j = 1;
    }
    if j < 1 {
        j = 1;
    }
    if j >= 15 {
        return 0;
    }
    if i >= 14 {
        return i;
    }
    for (dx, dy, dz) in FACINGS {
        let n = BlockPos::new(p.x + dx, p.y + dy, p.z + dz);
        let k = i32::from(light_for_in(src, has_sky, kind, n)) - j;
        if k > i {
            i = k;
        }
        if i >= 14 {
            return i;
        }
    }
    i
}

// ---- interior-cell reads -----------------
//
// A flood fill spends most of its time reading the six neighbours of each queued cell, and the
// generic reads above resolve the column, the section and the validity of every neighbour on
// its own. When a cell lies at least one block inside a present section on every axis, all six
// neighbours are in that same section, in the same column and equally valid, so their blocks
// and light can be read straight from the section's arrays. These reads return exactly what
// `block_at_in` and `light_for_in` return for those cells (the present-section arms), so the
// fill reads, writes and queues the same cells in the same order. About two thirds of the cells
// of a fill are interior; the rest (section borders, absent sections) take the generic reads.

/// `MC_W19_O2_LIGHT_FAST=0` turns the interior-cell reads off (read once).
fn light_fast_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MC_W19_O2_LIGHT_FAST").map_or(true, |v| v != "0"))
}

/// A cell one block or more inside a present section on every axis, with its column and section
/// resolved once.
#[derive(Clone, Copy)]
struct InteriorCell<'a> {
    chunk: &'a Chunk,
    section: &'a ChunkSection,
    /// Section-local coordinates, each `1..=14`.
    x: u8,
    y: u8,
    z: u8,
}

impl InteriorCell<'_> {
    /// `light_for_in` of the neighbour at `(dx, dy, dz)` (each `-1..=1`): its present-section arm.
    #[inline]
    fn light(self, kind: SkyBlock, has_sky: bool, dx: i32, dy: i32, dz: i32) -> u8 {
        let (x, y, z) = (
            (i32::from(self.x) + dx) as u8,
            (i32::from(self.y) + dy) as u8,
            (i32::from(self.z) + dz) as u8,
        );
        match kind {
            SkyBlock::Sky if !has_sky => 0,
            SkyBlock::Sky => self.section.sky_light(x, y, z),
            SkyBlock::Block => self.section.block_light(x, y, z),
        }
    }

    /// `block_at_in` of the neighbour at `(dx, dy, dz)` (each `-1..=1`).
    #[inline]
    fn block(self, dx: i32, dy: i32, dz: i32) -> BlockState {
        self.section.block(
            (i32::from(self.x) + dx) as u8,
            (i32::from(self.y) + dy) as u8,
            (i32::from(self.z) + dz) as u8,
        )
    }
}

/// The cell at `p` if it is an [`InteriorCell`] of a column `src` holds: valid (a column never
/// straddles the ±30,000,000 border, which is a multiple of 16, and an interior `y` is inside
/// `1..=254`), its section present, and `1..=14` on every section-local axis.
#[inline]
fn interior_cell(src: &impl ColumnRead, p: BlockPos) -> Option<InteriorCell<'_>> {
    let (x, y, z) = (p.x & 15, p.y & 15, p.z & 15);
    if !(1..=14).contains(&x) || !(1..=14).contains(&y) || !(1..=14).contains(&z) {
        return None;
    }
    if !is_valid(p) {
        return None;
    }
    let chunk = src.column(p.chunk_pos())?;
    let section = chunk.section((p.y >> 4) as u8)?;
    Some(InteriorCell {
        chunk,
        section,
        x: x as u8,
        y: y as u8,
        z: z as u8,
    })
}

/// [`raw_light_in`] for an [`InteriorCell`] at world height `y`: the same early outs, the same
/// `FACINGS` order, with every read taken from the cell's own column and section.
#[inline]
fn raw_light_interior(cell: InteriorCell<'_>, y: i32, has_sky: bool, kind: SkyBlock) -> i32 {
    if kind == SkyBlock::Sky && cell.chunk.can_see_sky(cell.x, y, cell.z) {
        return 15;
    }
    let props = cell.block(0, 0, 0).props();
    let mut i = match kind {
        SkyBlock::Sky => 0,
        SkyBlock::Block => i32::from(props.light_emission),
    };
    let mut j = i32::from(props.light_opacity);
    if j >= 15 && props.light_emission > 0 {
        j = 1;
    }
    if j < 1 {
        j = 1;
    }
    if j >= 15 {
        return 0;
    }
    if i >= 14 {
        return i;
    }
    for (dx, dy, dz) in FACINGS {
        let k = i32::from(cell.light(kind, has_sky, dx, dy, dz)) - j;
        if k > i {
            i = k;
        }
        if i >= 14 {
            return i;
        }
    }
    i
}

/// Whether `checkLightFor(kind, p)` on a resident area would change nothing: its stored value
/// already equals `getRawLight`, so neither the darkening nor the spread pass queues anything
/// and no cell is written. Pure reads, so evaluating it immediately before the call is exact.
#[inline]
fn origin_is_noop(src: &impl ColumnRead, has_sky: bool, kind: SkyBlock, p: BlockPos) -> bool {
    // The same two reads from one resolution when `p` is interior: a void
    // bridge block runs ~360 of these tests and nearly all are no-ops.
    if light_fast_enabled() {
        if let Some(cell) = interior_cell(src, p) {
            return raw_light_interior(cell, p.y, has_sky, kind)
                == i32::from(cell.light(kind, has_sky, 0, 0, 0));
        }
    }
    raw_light_in(src, has_sky, p, kind) == i32::from(light_for_in(src, has_sky, kind, p))
}

/// `World::mark_dirty_block_range`'s body over an explicit set, so [`FillView`] can mark while
/// it holds the column borrows.
fn mark_dirty_block_range_into(
    dirty: &mut mc_core::FastSet<ChunkSectionPos>,
    x1: i32,
    y1: i32,
    z1: i32,
    x2: i32,
    y2: i32,
    z2: i32,
) {
    for cx in (x1 >> 4)..=(x2 >> 4) {
        for cy in (y1 >> 4)..=(y2 >> 4) {
            if !(0..16).contains(&cy) {
                continue;
            }
            for cz in (z1 >> 4)..=(z2 >> 4) {
                dirty.insert(ChunkSectionPos::new(cx, cy, cz));
            }
        }
    }
}

/// Read-only access for the light checks of one world column `(x, z)`: its own chunk and the
/// chunks holding its four horizontal neighbours, resolved once (≤3 lookups). Any other column
/// falls back to the hashed map, so every answer equals [`World`]'s.
struct ColumnProbe<'w> {
    chunks: &'w mc_core::FastMap<ChunkPos, Arc<Chunk>>,
    keys: [ChunkPos; 3],
    cols: [Option<&'w Chunk>; 3],
}

impl<'w> ColumnProbe<'w> {
    fn around(chunks: &'w mc_core::FastMap<ChunkPos, Arc<Chunk>>, x: i32, z: i32) -> Self {
        let centre = ChunkPos::new(x >> 4, z >> 4);
        // Only an edge cell has a horizontal neighbour in another chunk, and only on one side.
        let x_side = if x & 15 == 0 { x - 1 } else { x + 1 };
        let z_side = if z & 15 == 0 { z - 1 } else { z + 1 };
        let keys = [
            centre,
            ChunkPos::new(x_side >> 4, z >> 4),
            ChunkPos::new(x >> 4, z_side >> 4),
        ];
        let first = chunks.get(&centre).map(Arc::as_ref);
        let resolve = |key: ChunkPos| {
            if key == centre {
                first
            } else {
                chunks.get(&key).map(Arc::as_ref)
            }
        };
        let cols = [first, resolve(keys[1]), resolve(keys[2])];
        Self { chunks, keys, cols }
    }
}

impl ColumnRead for ColumnProbe<'_> {
    #[inline]
    fn column(&self, cpos: ChunkPos) -> Option<&Chunk> {
        for (key, col) in self.keys.iter().zip(self.cols) {
            if *key == cpos {
                return col;
            }
        }
        self.chunks.get(&cpos).map(Arc::as_ref)
    }
}

/// How far, per horizontal axis, one `checkLightFor` call can read: queued cells stay within
/// Manhattan distance 17 of the origin (a push needs its parent at `< 17`), and `getRawLight`
/// reads one neighbour further. So every column a fill touches lies in
/// `((o - 18) >> 4) ..= ((o + 18) >> 4)` per axis — at most [`VIEW_SPAN`] columns.
const VIEW_REACH: i32 = 18;
/// Columns per axis in a [`FillView`].
const VIEW_SPAN: i32 = 4;

/// The ≤4×4 columns one `checkLightFor` flood fill can touch, resolved once with
/// `HashMap::get_disjoint_mut` (16 lookups) instead of 15-20 lookups per visited cell. Columns
/// are only `Arc::make_mut`-ed when written, exactly as before, so a render snapshot still shares
/// every column the fill merely reads.
struct FillView<'w> {
    base_cx: i32,
    base_cz: i32,
    cols: [Option<&'w mut Arc<Chunk>>; 16],
    dirty: &'w mut mc_core::FastSet<ChunkSectionPos>,
    /// The per-cell `notifyLightSet` boxes, one bit per `(column slot, section y)`. `dirty` is a
    /// set that nothing reads during a fill, so inserting the union once at the end is exact.
    dirty_bits: [u64; 4],
    has_sky: bool,
}

impl<'w> FillView<'w> {
    fn new(
        chunks: &'w mut mc_core::FastMap<ChunkPos, Arc<Chunk>>,
        dirty: &'w mut mc_core::FastSet<ChunkSectionPos>,
        has_sky: bool,
        origin: BlockPos,
    ) -> Self {
        let base_cx = (origin.x - VIEW_REACH) >> 4;
        let base_cz = (origin.z - VIEW_REACH) >> 4;
        let keys: [ChunkPos; 16] = std::array::from_fn(|i| {
            ChunkPos::new(
                base_cx + (i as i32 % VIEW_SPAN),
                base_cz + (i as i32 / VIEW_SPAN),
            )
        });
        // Distinct keys by construction, so this cannot panic.
        let cols = chunks.get_disjoint_mut(keys.each_ref());
        Self {
            base_cx,
            base_cz,
            cols,
            dirty,
            dirty_bits: [0; 4],
            has_sky,
        }
    }

    #[inline]
    fn slot(&self, cpos: ChunkPos) -> Option<usize> {
        let dx = cpos.x - self.base_cx;
        let dz = cpos.z - self.base_cz;
        if (0..VIEW_SPAN).contains(&dx) && (0..VIEW_SPAN).contains(&dz) {
            Some((dx + dz * VIEW_SPAN) as usize)
        } else {
            None
        }
    }

    /// java: `notifyLightSet` → `markBlocksForUpdate(p - 1 .. p + 1)`, as [`World::mark_dirty_box`].
    fn mark_box(&mut self, p: BlockPos) {
        for cx in ((p.x - 1) >> 4)..=((p.x + 1) >> 4) {
            for cy in ((p.y - 1) >> 4)..=((p.y + 1) >> 4) {
                if !(0..16).contains(&cy) {
                    continue;
                }
                for cz in ((p.z - 1) >> 4)..=((p.z + 1) >> 4) {
                    match self.slot(ChunkPos::new(cx, cz)) {
                        Some(i) => {
                            let bit = i * 16 + cy as usize;
                            self.dirty_bits[bit >> 6] |= 1u64 << (bit & 63);
                        }
                        None => {
                            self.dirty.insert(ChunkSectionPos::new(cx, cy, cz));
                        }
                    }
                }
            }
        }
    }

    fn flush_dirty(&mut self) {
        for (word_index, word) in self.dirty_bits.iter().enumerate() {
            let mut bits = *word;
            while bits != 0 {
                let bit = word_index * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let (slot, cy) = ((bit / 16) as i32, (bit % 16) as i32);
                self.dirty.insert(ChunkSectionPos::new(
                    self.base_cx + slot % VIEW_SPAN,
                    cy,
                    self.base_cz + slot / VIEW_SPAN,
                ));
            }
        }
        self.dirty_bits = [0; 4];
    }

    /// [`World::set_light_for`] on the resolved columns: same refusal rules, same
    /// create-then-`generateSkylightMap`-then-write order, same dirty boxes.
    fn set_light(&mut self, kind: SkyBlock, p: BlockPos, value: u8) {
        if !is_valid(p) {
            return;
        }
        let cpos = p.chunk_pos();
        let Some(index) = self.slot(cpos) else {
            debug_assert!(false, "light fill wrote outside its resolved view: {p:?}");
            return;
        };
        let has_sky = self.has_sky;
        // java: the blank `EmptyChunk.setLightFor` is an empty method.
        let Some(column) = self.cols[index].as_deref_mut() else {
            return;
        };
        let chunk = Arc::make_mut(column);
        let si = (p.y >> 4) as u8;
        let created = chunk.section(si).is_none();
        if !created && light_fast_enabled() {
            // The section is present, so `section_or_insert` returns the
            // very section `section_mut` would after it: write through it (one copy-on-write
            // check instead of two). Same write, same dirty box.
            let (x, z) = local_xz(p);
            let ly = (p.y & 15) as u8;
            let section = chunk.section_or_insert(si);
            match kind {
                SkyBlock::Sky => {
                    if has_sky {
                        section.set_sky_light(x, ly, z, value);
                    }
                }
                SkyBlock::Block => section.set_block_light(x, ly, z, value),
            }
            self.mark_box(p);
            return;
        }
        chunk.section_or_insert(si);
        if created {
            // `World::generate_skylight_map(cpos)` on the column already in hand.
            if let Some((lo, hi)) = chunk.generate_skylight_map(has_sky) {
                let (x, z) = (cpos.x * 16, cpos.z * 16);
                mark_dirty_block_range_into(
                    self.dirty,
                    x - 1,
                    lo - 1,
                    z - 1,
                    x + 16,
                    hi + 1,
                    z + 16,
                );
            }
        }
        let (x, z) = local_xz(p);
        let ly = (p.y & 15) as u8;
        let Some(section) = chunk.section_mut(si) else {
            return;
        };
        match kind {
            SkyBlock::Sky => {
                if has_sky {
                    section.set_sky_light(x, ly, z, value);
                }
            }
            SkyBlock::Block => section.set_block_light(x, ly, z, value),
        }
        self.mark_box(p);
    }

    #[inline]
    fn light(&self, kind: SkyBlock, p: BlockPos) -> u8 {
        light_for_in(self, self.has_sky, kind, p)
    }

    /// [`World::check_light_for`]'s body after the `isAreaLoaded` guard — line for line.
    fn check_light_for(&mut self, kind: SkyBlock, p: BlockPos, list: &mut [i32]) {
        let fast = light_fast_enabled();
        let has_sky = self.has_sky;
        let mut i = 0usize;
        let mut j = 0usize;
        let k = i32::from(self.light(kind, p));
        let l = raw_light_in(&*self, self.has_sky, p, kind);
        let (i1, j1, k1) = (p.x, p.y, p.z);

        if l > k {
            list[j] = 133_152;
            j += 1;
        } else if l < k {
            list[j] = 133_152 | (k << 18);
            j += 1;
            while i < j {
                let l1 = list[i];
                i += 1;
                let i2 = (l1 & 63) - 32 + i1;
                let j2 = ((l1 >> 6) & 63) - 32 + j1;
                let k2 = ((l1 >> 12) & 63) - 32 + k1;
                let l2 = (l1 >> 18) & 15;
                let bp = BlockPos::new(i2, j2, k2);
                if i32::from(self.light(kind, bp)) != l2 {
                    continue;
                }
                self.set_light(kind, bp, 0);
                if l2 <= 0 {
                    continue;
                }
                if (i2 - i1).abs() + (j2 - j1).abs() + (k2 - k1).abs() >= 17 {
                    continue;
                }
                // Resolved after the write above, so it reads what the
                // generic reads below would.
                let cell = if fast {
                    interior_cell(&*self, bp)
                } else {
                    None
                };
                for (dx, dy, dz) in FACINGS {
                    let (i4, j4, k4) = (i2 + dx, j2 + dy, k2 + dz);
                    let (opacity, light) = match cell {
                        Some(cell) => (
                            cell.block(dx, dy, dz).props().light_opacity,
                            cell.light(kind, has_sky, dx, dy, dz),
                        ),
                        None => {
                            let np = BlockPos::new(i4, j4, k4);
                            (
                                block_at_in(&*self, np).props().light_opacity,
                                self.light(kind, np),
                            )
                        }
                    };
                    let l4 = i32::from(opacity).max(1);
                    if i32::from(light) == l2 - l4 && j < list.len() {
                        list[j] = (i4 - i1 + 32)
                            | ((j4 - j1 + 32) << 6)
                            | ((k4 - k1 + 32) << 12)
                            | ((l2 - l4) << 18);
                        j += 1;
                    }
                }
            }
            i = 0;
        }

        while i < j {
            let i5 = list[i];
            i += 1;
            let j5 = (i5 & 63) - 32 + i1;
            let k5 = ((i5 >> 6) & 63) - 32 + j1;
            let l5 = ((i5 >> 12) & 63) - 32 + k1;
            let bp = BlockPos::new(j5, k5, l5);
            // Both reads of an interior cell from one resolution.
            let (i6, j6) = match if fast {
                interior_cell(&*self, bp)
            } else {
                None
            } {
                Some(cell) => (
                    i32::from(cell.light(kind, has_sky, 0, 0, 0)),
                    raw_light_interior(cell, k5, has_sky, kind),
                ),
                None => (
                    i32::from(self.light(kind, bp)),
                    raw_light_in(&*self, has_sky, bp, kind),
                ),
            };
            if j6 == i6 {
                continue;
            }
            self.set_light(kind, bp, j6.clamp(0, 15) as u8);
            if j6 <= i6 {
                continue;
            }
            let within = (j5 - i1).abs() + (k5 - j1).abs() + (l5 - k1).abs() < 17;
            if !within || j >= list.len() - 6 {
                continue;
            }
            // Resolved after the write, like the generic reads.
            let cell = if fast {
                interior_cell(&*self, bp)
            } else {
                None
            };
            for (dx, dy, dz) in SPREAD_ORDER {
                let n = BlockPos::new(j5 + dx, k5 + dy, l5 + dz);
                let light = match cell {
                    Some(cell) => cell.light(kind, has_sky, dx, dy, dz),
                    None => self.light(kind, n),
                };
                if i32::from(light) < j6 {
                    list[j] = (n.x - i1 + 32) | ((n.y - j1 + 32) << 6) | ((n.z - k1 + 32) << 12);
                    j += 1;
                }
            }
        }
    }
}

impl ColumnRead for FillView<'_> {
    #[inline]
    fn column(&self, cpos: ChunkPos) -> Option<&Chunk> {
        let Some(index) = self.slot(cpos) else {
            debug_assert!(false, "light fill read outside its resolved view: {cpos:?}");
            return None;
        };
        self.cols[index].as_deref().map(|column| &**column)
    }
}

impl World {
    // ---- area/height queries ------------------------------------------------

    /// java: `World.isAreaLoaded(BlockPos, int, boolean)` (`:269`) → the private box form
    /// (`:294`).
    ///
    /// `allow_empty = true` is **always satisfied on the client**, and that is vanilla, not a
    /// shortcut: `isChunkLoaded` is `chunkProvider.chunkExists(x, z) && (allowEmpty || …)`, and
    /// `ChunkProviderClient.chunkExists` returns `true` unconditionally
    /// (`ChunkProviderClient.java:42-45`). Only the `allowEmpty = false` form — the one
    /// `checkLightFor` uses — actually tests residency, because a real `Chunk.isEmpty()` is
    /// always `false` (`Chunk.java:1115`) while the provider's blank `EmptyChunk.isEmpty()` is
    /// `true` (`EmptyChunk.java:169`).
    #[must_use]
    pub fn is_area_loaded(&self, center: BlockPos, radius: i32, allow_empty: bool) -> bool {
        // java: the box corners are `center -/+ radius` in wrapping `int` arithmetic (an S08 can
        // put the player, and so this light sample, near `Integer.MAX_VALUE`). A wrapped box
        // has `start > end`, so Java's loop is empty and the area counts as loaded.
        let (y_start, y_end) = (center.y.wrapping_sub(radius), center.y.wrapping_add(radius));
        // java: `yEnd >= 0 && yStart < 256`, checked before any chunk lookup.
        if y_end < 0 || y_start >= 256 {
            return false;
        }
        if allow_empty {
            return true;
        }
        for cx in (center.x.wrapping_sub(radius) >> 4)..=(center.x.wrapping_add(radius) >> 4) {
            for cz in (center.z.wrapping_sub(radius) >> 4)..=(center.z.wrapping_add(radius) >> 4) {
                if !self.is_chunk_loaded(ChunkPos::new(cx, cz)) {
                    return false;
                }
            }
        }
        true
    }

    /// The heightmap value for a world column — one above the topmost light-blocking block.
    /// `0` for an unloaded column, matching `EmptyChunk.getHeightValue` (`:33`).
    /// java: `World.getHeight(BlockPos)` (`:684`), the `.getY()` of its result.
    #[must_use]
    pub fn height(&self, x: i32, z: i32) -> i32 {
        if !(-30_000_000..30_000_000).contains(&x) || !(-30_000_000..30_000_000).contains(&z) {
            // java: `getSeaLevel() + 1`; the client's sea level is 63.
            return 64;
        }
        self.chunk(ChunkPos::new(x >> 4, z >> 4))
            .map_or(0, |c| c.height_value((x & 15) as u8, (z & 15) as u8))
    }

    /// java: `World.getChunksLowestHorizon` (`:710`) — `recheckGaps` compares this across the
    /// four horizontal neighbours to find a sky-light discontinuity at a column boundary.
    #[must_use]
    pub fn chunks_lowest_horizon(&self, x: i32, z: i32) -> i32 {
        if !(-30_000_000..30_000_000).contains(&x) || !(-30_000_000..30_000_000).contains(&z) {
            return 64;
        }
        self.chunk(ChunkPos::new(x >> 4, z >> 4))
            .map_or(0, crate::Chunk::lowest_height)
    }

    /// Whether this cell has an unobstructed view of the sky. `false` for an unloaded column,
    /// matching `EmptyChunk.canSeeSky` (`:102`) — which is *not* the same default as
    /// [`World::sky_light_at`]'s, and vanilla is inconsistent here too.
    /// java: `World.canSeeSky` (`:568`) → `Chunk.canSeeSky` (`:904`)
    #[must_use]
    pub fn can_see_sky(&self, p: BlockPos) -> bool {
        let (x, z) = local_xz(p);
        self.chunk(p.chunk_pos())
            .is_some_and(|c| c.can_see_sky(x, p.y, z))
    }

    /// The Y precipitation falls to in this column. java: `World.getPrecipitationHeight` →
    /// `Chunk.getPrecipitationHeight` (`:1186`).
    ///
    /// The client provider's shared `EmptyChunk` inherits `Chunk`'s scan, but that scan reads
    /// `this.getBlock(pos)` (`Chunk.java:1201`), which `EmptyChunk` overrides to air: an absent
    /// column answers `-1` whatever states the blank chunk holds (see [`World::block_at`]).
    /// Keep that sentinel here instead of turning an absent column into a synthetic ground
    /// level; render callers that need a non-negative Y can normalize it at their own boundary.
    #[must_use]
    pub fn precipitation_height(&self, x: i32, z: i32) -> i32 {
        match self.chunk(ChunkPos::new(x >> 4, z >> 4)) {
            Some(chunk) => chunk.precipitation_height((x & 15) as u8, (z & 15) as u8),
            None => -1,
        }
    }

    // ---- single-channel light access ------------------------------------------------------

    /// One channel's stored nibble at `p`. java: `World.getLightFor` (`:789`) →
    /// `Chunk.getLightFor` (`:780`)
    #[must_use]
    pub fn light_for(&self, kind: SkyBlock, p: BlockPos) -> u8 {
        // java: `if (pos.getY() < 0) pos = new BlockPos(pos.getX(), 0, pos.getZ());`
        let p = if p.y < 0 {
            BlockPos::new(p.x, 0, p.z)
        } else {
            p
        };
        if !is_valid(p) {
            return kind.default_light_value();
        }
        let (x, z) = local_xz(p);
        let Some(chunk) = self.chunk(p.chunk_pos()) else {
            // java: the provider's blank `EmptyChunk.getLightFor` (`:67`).
            return kind.default_light_value();
        };
        match chunk.section((p.y >> 4) as u8) {
            // java: `extendedblockstorage == null ? (canSeeSky(pos) ? defaultLightValue : 0)`
            None => {
                if chunk.can_see_sky(x, p.y, z) {
                    kind.default_light_value()
                } else {
                    0
                }
            }
            Some(section) => match kind {
                SkyBlock::Sky if !self.has_sky_light() => 0,
                SkyBlock::Sky => section.sky_light(x, (p.y & 15) as u8, z),
                SkyBlock::Block => section.block_light(x, (p.y & 15) as u8, z),
            },
        }
    }

    /// Writes one channel's nibble and tells the mesher. java: `World.setLightFor` (`:811`) →
    /// `Chunk.setLightFor` (`:789`) + `notifyLightSet` (`:824`)
    pub fn set_light_for(&mut self, kind: SkyBlock, p: BlockPos, value: u8) {
        if !is_valid(p) {
            return;
        }
        let has_sky = self.has_sky_light();
        let cpos = p.chunk_pos();
        let (x, z) = local_xz(p);
        let ly = (p.y & 15) as u8;
        let si = (p.y >> 4) as u8;
        // java: on the client an absent column is the blank `EmptyChunk`, whose `setLightFor`
        // is an empty method (`EmptyChunk.java:72`).
        let Some(chunk) = self.chunks.get_mut(&cpos).map(Arc::make_mut) else {
            return;
        };
        // java: `if (extendedblockstorage == null) { … new ExtendedBlockStorage(…);
        //        this.generateSkylightMap(); }`
        let created = chunk.section(si).is_none();
        chunk.section_or_insert(si);
        if created {
            self.generate_skylight_map(cpos);
        }
        // `generateSkylightMap` deliberately runs before this write. Besides matching Java's
        // order, this matters when the new section is an air gap under a translucent overhang:
        // the top-down descent writes the same nibble, and must not erase the caller's value.
        let Some(section) = self
            .chunks
            .get_mut(&cpos)
            .map(Arc::make_mut)
            .and_then(|chunk| chunk.section_mut(si))
        else {
            return;
        };
        match kind {
            SkyBlock::Sky => {
                if has_sky {
                    section.set_sky_light(x, ly, z, value);
                }
            }
            SkyBlock::Block => section.set_block_light(x, ly, z, value),
        }
        self.mark_dirty_box(p);
    }

    /// The darkened light query. java: `World.getLight(pos, true)` (`:626`) via
    /// `Chunk.getLightSubtracted` (`:818`).
    ///
    /// `amount` is vanilla's `skylightSubtracted`, which
    /// `mc_render::sky::Environment::skylight_subtracted` computes from the celestial angle and
    /// the weather. `0` reproduces [`World::light_at`].
    /// Two vanilla quirks are reproduced rather than tidied, and one is not:
    /// * the missing-section arm returns `15 - amount` **without** consulting `canSeeSky`, which
    ///   is *not* what [`World::light_for`]'s equivalent arm does — vanilla is inconsistent here;
    /// * an unloaded column is `0`, not `15`, because that arm is `EmptyChunk.getLightSubtracted`
    ///   (`:76`) rather than `EmptyChunk.getLightFor` (`:67`);
    /// * Java's intermediate `l = l - amount` can go negative (sky 2, amount 11), but the
    ///   non-negative block-light channel then wins the following `max`; saturating the sky
    ///   subtraction at zero is therefore result-equivalent.
    #[must_use]
    pub fn light_subtracted(&self, p: BlockPos, amount: u8) -> u8 {
        // java: `World.getLight(pos, true)` returns 15 outside the horizontal world border,
        // before either the neighbour-brightness branch or the Y guards.
        if !(-30_000_000..30_000_000).contains(&p.x) || !(-30_000_000..30_000_000).contains(&p.z) {
            return 15;
        }
        if self.block_at(p).props().use_neighbor_brightness {
            // java: UP, EAST, WEST, SOUTH, NORTH. Only the maximum is observable, but spelling
            // the order out keeps this port aligned with World.java:626-650.
            return [
                BlockPos::new(p.x, p.y + 1, p.z),
                BlockPos::new(p.x + 1, p.y, p.z),
                BlockPos::new(p.x - 1, p.y, p.z),
                BlockPos::new(p.x, p.y, p.z + 1),
                BlockPos::new(p.x, p.y, p.z - 1),
            ]
            .into_iter()
            .map(|neighbour| self.light_subtracted_direct(neighbour, amount))
            .max()
            .unwrap_or(0);
        }
        self.light_subtracted_direct(p, amount)
    }

    /// `World.getLight(pos, false)` / the non-neighbour half of the public query above.
    #[must_use]
    fn light_subtracted_direct(&self, p: BlockPos, amount: u8) -> u8 {
        // java: each recursive `getLight(pos, false)` repeats the horizontal border guard.
        if !(-30_000_000..30_000_000).contains(&p.x) || !(-30_000_000..30_000_000).contains(&p.z) {
            return 15;
        }
        if p.y < 0 {
            return 0;
        }
        let p = if p.y >= 256 {
            BlockPos::new(p.x, 255, p.z)
        } else {
            p
        };
        let Some(chunk) = self.chunk(p.chunk_pos()) else {
            return 0;
        };
        let (x, z) = local_xz(p);
        let ly = (p.y & 15) as u8;
        let Some(section) = chunk.section((p.y >> 4) as u8) else {
            return if self.has_sky_light() && amount < 15 {
                15 - amount
            } else {
                0
            };
        };
        let sky = if self.has_sky_light() {
            section.sky_light(x, ly, z)
        } else {
            0
        };
        sky.saturating_sub(amount)
            .max(section.block_light(x, ly, z))
    }

    // ---- the flood fill ------------------------------------------------------

    /// What this cell's light level *should* be, from its own emission and its six neighbours.
    /// java: `World.getRawLight` (`:2783`)
    #[must_use]
    fn raw_light(&self, p: BlockPos, kind: SkyBlock) -> i32 {
        if kind == SkyBlock::Sky && self.can_see_sky(p) {
            return 15;
        }
        let props = self.block_at(p).props();
        let mut i = match kind {
            SkyBlock::Sky => 0,
            SkyBlock::Block => i32::from(props.light_emission),
        };
        let mut j = i32::from(props.light_opacity);
        // java: an opaque *emitter* (a jack-o'-lantern is opacity 255, emission 15) is treated
        // as opacity 1 so its own light can escape.
        if j >= 15 && props.light_emission > 0 {
            j = 1;
        }
        if j < 1 {
            j = 1;
        }
        if j >= 15 {
            return 0;
        }
        if i >= 14 {
            return i;
        }
        for (dx, dy, dz) in FACINGS {
            let n = BlockPos::new(p.x + dx, p.y + dy, p.z + dz);
            let k = i32::from(self.light_for(kind, n)) - j;
            if k > i {
                i = k;
            }
            if i >= 14 {
                return i;
            }
        }
        i
    }

    /// Both channels at once. java: `World.checkLight` (`:2767`)
    pub fn check_light(&mut self, p: BlockPos) -> bool {
        if self.uses_reference_light_engine() {
            let mut flag = false;
            if self.has_sky_light() {
                flag |= self.check_light_for(SkyBlock::Sky, p);
            }
            flag |= self.check_light_for(SkyBlock::Block, p);
            return flag;
        }
        // Both channels ask `isAreaLoaded(p, 17, false)` of the same position, and no light
        // write loads or unloads a column, so the second answer always equals the first.
        if !self.is_area_loaded(p, 17, false) {
            return false;
        }
        if self.has_sky_light() {
            self.check_light_for_resident(SkyBlock::Sky, p);
        }
        self.check_light_for_resident(SkyBlock::Block, p);
        true
    }

    /// One channel's bounded flood fill. java: `World.checkLightFor` (`:2836`)
    ///
    /// Returns `false` without doing anything when the 17-block ball around `p` is not fully
    /// loaded — the guard that stops the client from "fixing" light against columns it has not
    /// received and baking a dark seam into the mesh.
    pub fn check_light_for(&mut self, kind: SkyBlock, p: BlockPos) -> bool {
        if !self.is_area_loaded(p, 17, false) {
            return false;
        }
        self.check_light_for_in_loaded_area(kind, p)
    }

    /// [`Self::check_light`] for a `p` whose guard result the caller already holds:
    /// `area_loaded` must be `self.is_area_loaded(p, 17, false)`.
    ///
    /// That guard is 9–16 `is_chunk_loaded` probes per channel, more than the rest of a no-op
    /// check. `enqueue_relight_checks` asks it for up to 128 cells sharing one `(x, z)` (y in
    /// `0..256` never trips its y bound), and no light write loads or unloads a column, so one
    /// answer serves them all (NF1).
    fn check_light_area_known(&mut self, p: BlockPos, area_loaded: bool) -> bool {
        if !area_loaded {
            // Both channels' `checkLightFor` return `false` before touching anything.
            return false;
        }
        let mut flag = false;
        if self.has_sky_light() {
            flag |= self.check_light_for_in_loaded_area(SkyBlock::Sky, p);
        }
        flag |= self.check_light_for_in_loaded_area(SkyBlock::Block, p);
        flag
    }

    /// [`Self::check_light_for`] past its residency guard: the reference engine or the resolved-
    /// column one, which produce bit-identical light. `check_light_area_known` relies on this.
    fn check_light_for_in_loaded_area(&mut self, kind: SkyBlock, p: BlockPos) -> bool {
        if self.uses_reference_light_engine() {
            self.check_light_for_reference(kind, p);
        } else {
            self.check_light_for_resident(kind, p);
        }
        true
    }

    /// Whether every `checkLightFor` runs through [`Self::check_light_for_reference`].
    /// DEBUG_WORLD must: its `getBlockState` ignores chunk storage, which the resolved views
    /// read directly.
    #[inline]
    fn uses_reference_light_engine(&self) -> bool {
        self.reference_light_engine || self.is_debug_world()
    }

    /// Selects the original hashed-lookup light engine (`true`) or the resolved-column one
    /// (`false`, the default). Results are bit-identical; this exists for differential tests and
    /// `examples/light_perf.rs`.
    #[doc(hidden)]
    pub fn set_reference_light_engine(&mut self, reference: bool) {
        self.reference_light_engine = reference;
    }

    /// `checkLightFor` once `isAreaLoaded(p, 17, false)` is known to hold. A provable no-op
    /// (stored value already equals `getRawLight`) is answered from ≤3 resolved columns; anything
    /// else runs the fill over a [`FillView`].
    fn check_light_for_resident(&mut self, kind: SkyBlock, p: BlockPos) {
        let has_sky = self.has_sky_light();
        let noop = {
            let probe = ColumnProbe::around(&self.chunks, p.x, p.z);
            origin_is_noop(&probe, has_sky, kind, p)
        };
        if !noop {
            self.check_light_for_fill(kind, p);
        }
    }

    /// The flood fill proper, on resolved columns.
    fn check_light_for_fill(&mut self, kind: SkyBlock, p: BlockPos) {
        let has_sky = self.has_sky_light();
        // Taken out exactly as the reference engine does; see there.
        let mut list = std::mem::take(&mut self.light_update_block_list);
        if list.len() != LIGHT_UPDATE_LIST_LEN {
            list.resize(LIGHT_UPDATE_LIST_LEN, 0);
        }
        let mut view = FillView::new(&mut self.chunks, &mut self.dirty, has_sky, p);
        view.check_light_for(kind, p, &mut list);
        view.flush_dirty();
        self.light_update_block_list = list;
    }

    /// `checkLightFor(SKY, (x, y, z))` for every `y` in `lo..=hi`, ascending — the loop body of
    /// `markBlocksDirtyVertical` and `updateSkylightNeighborHeight`. A void bridge block runs
    /// ~391 of these and nearly all are no-ops.
    ///
    /// Exactness: `isAreaLoaded(p, 17, false)` depends on `y` only through its `-17..273` guard
    /// (its residency half is a function of `x, z`, and no light write loads or unloads a
    /// column), so it is hoisted; and each origin's no-op test is evaluated immediately before
    /// the origin would run, against the state every earlier origin left, so a cell written by
    /// an earlier fill is seen exactly as the sequential calls would see it.
    fn check_sky_light_column(&mut self, x: i32, z: i32, lo: i32, hi: i32) {
        if self.uses_reference_light_engine() {
            for y in lo..=hi {
                self.check_light_for(SkyBlock::Sky, BlockPos::new(x, y, z));
            }
            return;
        }
        if lo > hi || !self.is_area_loaded(BlockPos::new(x, 0, z), 17, false) {
            return;
        }
        let has_sky = self.has_sky_light();
        let mut y = lo;
        loop {
            let next = {
                let probe = ColumnProbe::around(&self.chunks, x, z);
                (y..=hi).find(|&yy| {
                    yy + 17 >= 0
                        && yy - 17 < 256
                        && !origin_is_noop(&probe, has_sky, SkyBlock::Sky, BlockPos::new(x, yy, z))
                })
            };
            let Some(origin) = next else {
                return;
            };
            self.check_light_for_fill(SkyBlock::Sky, BlockPos::new(x, origin, z));
            if origin >= hi {
                return;
            }
            y = origin + 1;
        }
    }

    /// The original transcription of `checkLightFor`'s body after the `isAreaLoaded` guard:
    /// one hashed column lookup per read and two per write. Kept verbatim as the reference the
    /// resolved-column engine is tested against, and for DEBUG_WORLD.
    #[allow(clippy::too_many_lines)] // A line-for-line transcription; splitting it hides drift.
    fn check_light_for_reference(&mut self, kind: SkyBlock, p: BlockPos) {
        // Taken out so the fill can hold `&mut self` for `set_light_for`; put back below. The
        // buffer is reused across calls exactly as vanilla's `lightUpdateBlockList` field is.
        let mut list = std::mem::take(&mut self.light_update_block_list);
        if list.len() != LIGHT_UPDATE_LIST_LEN {
            list.resize(LIGHT_UPDATE_LIST_LEN, 0);
        }

        let mut i = 0usize;
        let mut j = 0usize;
        let k = i32::from(self.light_for(kind, p));
        let l = self.raw_light(p, kind);
        let (i1, j1, k1) = (p.x, p.y, p.z);

        if l > k {
            // Brightening: seed the spread pass with the origin and let it run below.
            list[j] = 133_152;
            j += 1;
        } else if l < k {
            // Darkening: first erase the region this cell used to light, remembering its
            // frontier, then re-brighten from that frontier in the second pass.
            list[j] = 133_152 | (k << 18);
            j += 1;
            while i < j {
                let l1 = list[i];
                i += 1;
                let i2 = (l1 & 63) - 32 + i1;
                let j2 = ((l1 >> 6) & 63) - 32 + j1;
                let k2 = ((l1 >> 12) & 63) - 32 + k1;
                let l2 = (l1 >> 18) & 15;
                let bp = BlockPos::new(i2, j2, k2);
                if i32::from(self.light_for(kind, bp)) != l2 {
                    continue;
                }
                self.set_light_for(kind, bp, 0);
                if l2 <= 0 {
                    continue;
                }
                // java: `MathHelper.abs_int` — the 17-block reach bound.
                if (i2 - i1).abs() + (j2 - j1).abs() + (k2 - k1).abs() >= 17 {
                    continue;
                }
                for (dx, dy, dz) in FACINGS {
                    let (i4, j4, k4) = (i2 + dx, j2 + dy, k2 + dz);
                    let np = BlockPos::new(i4, j4, k4);
                    let l4 = i32::from(self.block_at(np).props().light_opacity).max(1);
                    if i32::from(self.light_for(kind, np)) == l2 - l4 && j < list.len() {
                        list[j] = (i4 - i1 + 32)
                            | ((j4 - j1 + 32) << 6)
                            | ((k4 - k1 + 32) << 12)
                            | ((l2 - l4) << 18);
                        j += 1;
                    }
                }
            }
            i = 0;
        }

        // The spread pass: for every queued cell, recompute and push any neighbour it can now
        // brighten. Runs for both the brightening seed and the darkening frontier.
        while i < j {
            let i5 = list[i];
            i += 1;
            let j5 = (i5 & 63) - 32 + i1;
            let k5 = ((i5 >> 6) & 63) - 32 + j1;
            let l5 = ((i5 >> 12) & 63) - 32 + k1;
            let bp = BlockPos::new(j5, k5, l5);
            let i6 = i32::from(self.light_for(kind, bp));
            let j6 = self.raw_light(bp, kind);
            if j6 == i6 {
                continue;
            }
            self.set_light_for(kind, bp, j6.clamp(0, 15) as u8);
            if j6 <= i6 {
                continue;
            }
            let within = (j5 - i1).abs() + (k5 - j1).abs() + (l5 - k1).abs() < 17;
            if !within || j >= list.len() - 6 {
                continue;
            }
            for (dx, dy, dz) in SPREAD_ORDER {
                let n = BlockPos::new(j5 + dx, k5 + dy, l5 + dz);
                if i32::from(self.light_for(kind, n)) < j6 {
                    list[j] = (n.x - i1 + 32) | ((n.y - j1 + 32) << 6) | ((n.z - k1 + 32) << 12);
                    j += 1;
                }
            }
        }

        self.light_update_block_list = list;
    }

    // ---- heightmap maintenance ----------------------------------------

    /// java: `Chunk.generateSkylightMap` (`:246`) — the full-column sky descent, run when a
    /// section is created at or above the heightmap.
    pub(crate) fn generate_skylight_map(&mut self, cpos: ChunkPos) {
        let has_sky = self.has_sky_light();
        let Some(chunk) = self.chunks.get_mut(&cpos).map(Arc::make_mut) else {
            return;
        };
        let written = chunk.generate_skylight_map(has_sky);
        // java: one `notifyLightSet` per written cell. Their union over a whole chunk footprint
        // is this box, marked in one go rather than re-hashing the same section key a thousand
        // times — and bounded to the Y range actually written, which is usually one or two
        // section layers rather than all sixteen.
        if let Some((lo, hi)) = written {
            let (x, z) = (cpos.x * 16, cpos.z * 16);
            self.mark_dirty_block_range(x - 1, lo - 1, z - 1, x + 16, hi + 1, z + 16);
        }
    }

    /// java: `Chunk.relightBlock` (`:400`) — the heightmap moved, so the sky column above and
    /// below it has to be rewritten and its horizontal neighbours re-checked.
    pub(crate) fn relight_block(&mut self, cpos: ChunkPos, x: u8, y: i32, z: u8) {
        let has_sky = self.has_sky_light();
        let Some((i, j)) = self.chunk(cpos).and_then(|c| c.relight_begin(x, y, z)) else {
            // java: `if (j != i)` failed — `relightBlock` does nothing at all.
            return;
        };
        let wx = (cpos.x << 4) + i32::from(x);
        let wz = (cpos.z << 4) + i32::from(z);
        // java: `Chunk.java:412`, **before** the heightmap is written. See `relight_begin`.
        self.mark_blocks_dirty_vertical(wx, wz, j, i);
        let outcome = {
            let Some(chunk) = self.chunks.get_mut(&cpos).map(Arc::make_mut) else {
                return;
            };
            chunk.relight_finish(x, z, i, j, has_sky)
        };
        // java: `notifyLightSet` per written cell — one column's worth, not a whole chunk's.
        if let Some((lo, hi)) = outcome.written {
            self.mark_dirty_block_range(wx - 1, lo - 1, wz - 1, wx + 1, hi + 1, wz + 1);
        }
        if has_sky {
            let (start, end) = outcome.neighbour_range;
            for (dx, dz) in HORIZONTAL {
                self.update_skylight_neighbor_height(wx + dx, wz + dz, start, end);
            }
            self.update_skylight_neighbor_height(wx, wz, start, end);
        }
    }

    /// java: `World.markBlocksDirtyVertical` (`:453`)
    fn mark_blocks_dirty_vertical(&mut self, x: i32, z: i32, y1: i32, y2: i32) {
        let (lo, hi) = if y1 > y2 { (y2, y1) } else { (y1, y2) };
        if self.has_sky_light() {
            self.check_sky_light_column(x, z, lo, hi);
        }
        self.mark_dirty_block_range(x, lo, z, x, hi, z);
    }

    /// java: `Chunk.updateSkylightNeighborHeight` (`:384`)
    fn update_skylight_neighbor_height(&mut self, x: i32, z: i32, start_y: i32, end_y: i32) {
        if end_y > start_y && self.is_area_loaded(BlockPos::new(x, 0, z), 16, true) {
            self.check_sky_light_column(x, z, start_y, end_y - 1);
        }
    }

    /// java: `Chunk.checkSkylightNeighborHeight` (`:370`)
    fn check_skylight_neighbor_height(&mut self, x: i32, z: i32, max_value: i32) {
        let i = self.height(x, z);
        // java: `maxValue + 1` is int arithmetic and wraps. `recheckGaps` passes
        // `getChunksLowestHorizon`, which is `Integer.MAX_VALUE` for a column whose chunk has
        // nothing opaque; the wrapped end then fails `endY > startY`. A plain `+` panics there in
        // an overflow-checked (dev) build instead.
        if i > max_value {
            self.update_skylight_neighbor_height(x, z, max_value, i.wrapping_add(1));
        } else if i < max_value {
            self.update_skylight_neighbor_height(x, z, i, max_value.wrapping_add(1));
        }
    }

    // ---- the periodic repair pass --------------------------------------------

    /// java: `Chunk.recheckGaps` (`:322`) — repairs sky-light discontinuities at column
    /// boundaries by comparing `getChunksLowestHorizon` across the four horizontal neighbours.
    ///
    /// `only_one` is vanilla's `p_150803_1_`, which `func_150804_b` passes as
    /// `worldObj.isRemote` — so on the client this returns after the **first** flagged column,
    /// one per chunk per tick.
    pub fn recheck_gaps(&mut self, cpos: ChunkPos, only_one: bool) {
        // java: `isAreaLoaded(new BlockPos(x*16+8, 0, z*16+8), 16)` — the two-argument overload,
        // i.e. `allowEmpty = true`, which is unconditionally satisfied on the client.
        if !self.is_area_loaded(BlockPos::new(cpos.x * 16 + 8, 0, cpos.z * 16 + 8), 16, true) {
            return;
        }
        for i in 0..16u8 {
            for jj in 0..16u8 {
                let idx = (i as usize) + (jj as usize) * 16;
                let flagged = {
                    let Some(chunk) = self.chunks.get_mut(&cpos).map(Arc::make_mut) else {
                        return;
                    };
                    let flagged = chunk.update_skylight_columns[idx];
                    if flagged {
                        chunk.update_skylight_columns[idx] = false;
                    }
                    flagged
                };
                if !flagged {
                    continue;
                }
                let k = self.chunk(cpos).map_or(0, |c| c.height_value(i, jj));
                let l = cpos.x * 16 + i32::from(i);
                let i1 = cpos.z * 16 + i32::from(jj);
                let mut j1 = i32::MAX;
                for (dx, dz) in HORIZONTAL {
                    j1 = j1.min(self.chunks_lowest_horizon(l + dx, i1 + dz));
                }
                self.check_skylight_neighbor_height(l, i1, j1);
                for (dx, dz) in HORIZONTAL {
                    self.check_skylight_neighbor_height(l + dx, i1 + dz, k);
                }
                if only_one {
                    return;
                }
            }
        }
        if let Some(chunk) = self.chunks.get_mut(&cpos).map(Arc::make_mut) {
            chunk.is_gap_lighting_updated = false;
        }
    }

    /// java: `Chunk.enqueueRelightChecks` (`:1439`) — the round-robin sweep that re-runs
    /// `checkLight` around any light source that sat near a chunk edge when the column loaded.
    /// Advances at most 8 of the chunk's 4096 slots, exactly as vanilla does.
    pub fn enqueue_relight_checks(&mut self, cpos: ChunkPos) {
        // Read-only first: `Arc::make_mut` below deep-copies a column that a render snapshot
        // still shares, and a finished sweep (`>= 4096`) writes nothing (TICK-6). Same early
        // returns as the loop's own checks.
        if self
            .chunk(cpos)
            .is_none_or(|chunk| chunk.queued_light_checks >= 4096)
        {
            return;
        }
        let base_x = cpos.x << 4;
        let base_z = cpos.z << 4;
        for _ in 0..8 {
            let q = {
                let Some(chunk) = self.chunks.get_mut(&cpos).map(Arc::make_mut) else {
                    return;
                };
                if chunk.queued_light_checks >= 4096 {
                    return;
                }
                let q = chunk.queued_light_checks;
                chunk.queued_light_checks += 1;
                q
            };
            let j = (q % 16) as u8; // section index
            let k = (q / 16 % 16) as u8; // local x
            let l = (q / 256) as u8; // local z

            // Every cell of this slot shares `(x, z)`: one `is_area_loaded(bp, 17, false)`
            // answers all their `check_light` guards (see `check_light_area_known`).
            let mut area_loaded = None;
            // `World::block_at` answers DEBUG_WORLD before consulting any column.
            let debug_world = self.is_debug_world();
            for i1 in 0..16u8 {
                let bp = BlockPos::new(
                    base_x + i32::from(k),
                    (i32::from(j) << 4) + i32::from(i1),
                    base_z + i32::from(l),
                );
                let edge = i1 == 0 || i1 == 15 || k == 0 || k == 15 || l == 0 || l == 15;
                // One probe of this column answers the "interesting" test and every neighbour
                // read that stays inside it. The sweep rewrites light only, never blocks (a
                // section `set_light_for` creates is all air), so all six emissions can be read
                // before any neighbour's `check_light` runs, in the same order.
                let mut emitting = [false; 6];
                let interesting = {
                    let column = self.chunk(cpos);
                    let interesting = match column.and_then(|c| c.section(j)) {
                        None => edge,
                        Some(section) => {
                            section.block(k, i1, l).props().material == mc_data::Material::Air
                        }
                    };
                    if interesting {
                        for (emits, (dx, dy, dz)) in emitting.iter_mut().zip(FACINGS) {
                            let np = BlockPos::new(bp.x + dx, bp.y + dy, bp.z + dz);
                            let state = match column {
                                Some(chunk)
                                    if !debug_world
                                        && np.chunk_pos() == cpos
                                        && (0..256).contains(&np.y) =>
                                {
                                    let (x, z) = local_xz(np);
                                    chunk.block_at(x, np.y as u8, z)
                                }
                                _ => self.block_at(np),
                            };
                            *emits = state.props().light_emission > 0;
                        }
                    }
                    interesting
                };
                if !interesting {
                    continue;
                }
                for ((dx, dy, dz), emits) in FACINGS.into_iter().zip(emitting) {
                    if emits {
                        self.check_light(BlockPos::new(bp.x + dx, bp.y + dy, bp.z + dz));
                    }
                }
                let loaded = *area_loaded.get_or_insert_with(|| self.is_area_loaded(bp, 17, false));
                self.check_light_area_known(bp, loaded);
            }
        }
    }

    /// Run the client's gap-repair half of the periodic light pass.
    ///
    /// java: `ChunkProviderClient.unloadQueuedChunks` (`:109-116`) invokes
    /// `Chunk.func_150804_b` for **every** entry in `chunkListing` on every client tick. The
    /// elapsed-time argument only changes whether `Chunk.recheckGaps` stops after its first
    /// flagged cell; it does not limit which columns are visited. A replay cannot use Java's
    /// wall-clock cutoff deterministically, so this keeps the existing one-flagged-cell client
    /// behaviour (`only_one = true`) while preserving the important every-column cadence.
    ///
    /// Returns the number of resident columns visited, including columns whose gap flag is
    /// already clear. Keeping this pass separate from [`Self::tick_chunk_relight`] is load-bearing:
    /// the relight budget belongs to the other vanilla loop and must not throttle this sweep.
    pub fn tick_chunk_gap_light(&mut self) -> usize {
        let loaded = self.chunks.len();
        // java: `Chunk.func_150804_b` guards `recheckGaps` with both this flag and
        // `!worldObj.provider.getHasNoSky()`.
        if !self.has_sky_light() {
            return loaded;
        }
        // Only flagged columns do any work, and nothing reachable from `recheck_gaps` sets a
        // flag (only block writes call `propagate_skylight_occlusion`), so the sorted flagged
        // subset is the same visit sequence as sorting every loaded column and skipping the
        // clear ones. In steady state the subset is empty: no Vec, no sort, no per-column
        // SipHash probe (TICK-6).
        let mut flagged: Vec<ChunkPos> = self
            .chunks
            .iter()
            .filter(|(_, chunk)| chunk.is_gap_lighting_updated)
            .map(|(pos, _)| *pos)
            .collect();
        // `chunks` is a `HashMap`; sort so the repair order is deterministic when more than one
        // flagged column changes neighbouring light in the same tick.
        flagged.sort_unstable_by_key(|pos| (pos.x, pos.z));
        for cpos in &flagged {
            if self.chunk(*cpos).is_some_and(|c| c.is_gap_lighting_updated) {
                self.recheck_gaps(*cpos, true);
            }
        }
        loaded
    }

    /// Run the bounded active-column relight half of the periodic light pass.
    ///
    /// java: `WorldClient.updateBlocks` (`:120-151`) hands at most **10** newly-active columns
    /// to `playMoodSoundAndCheckLight`, whose `checkLight` tail calls
    /// `Chunk.enqueueRelightChecks` (`World.java:2665-2666`). The explicit budget is retained as
    /// an argument so world-level tests and deterministic replays can choose a smaller sample.
    /// The queue is refilled only when empty **on entry**, so one call never visits a column
    /// twice and no column is starved: every loaded column gets its turn before any gets a
    /// second. Gap rechecks deliberately do not happen here; [`Self::tick_chunk_gap_light`]
    /// owns that every-loaded-column pass.
    pub fn tick_chunk_relight(&mut self, columns_per_tick: usize) -> usize {
        if columns_per_tick == 0 {
            return 0;
        }
        if self.light_tick_queue.is_empty() {
            let mut all: Vec<ChunkPos> = self.loaded_chunks().map(|(pos, _)| pos).collect();
            // `loaded_chunks` iterates a `HashMap`, whose order differs between processes.
            // Sorting is what makes this pass replay-deterministic, the same reason
            // `drain_dirty_sections` returns a `BTreeSet`.
            all.sort_unstable_by_key(|pos| (pos.x, pos.z));
            self.light_tick_queue = all;
        }
        let mut visited = 0;
        while visited < columns_per_tick {
            let Some(cpos) = self.light_tick_queue.pop() else {
                break;
            };
            if !self.is_chunk_loaded(cpos) {
                continue;
            }
            visited += 1;
            self.enqueue_relight_checks(cpos);
        }
        visited
    }

    /// Run `WorldClient.updateBlocks`' active-column relight cycle.
    ///
    /// Java first retains only still-active entries in `previousActiveChunkSet`. Once that set
    /// has caught up with the complete `activeChunkSet`, it clears the previous set and begins a
    /// new cycle. At most `columns_per_tick` entries absent from the previous set are consumed;
    /// an active but not-yet-loaded coordinate still consumes a slot because vanilla resolves it
    /// to `ChunkProviderClient.blankChunk` before incrementing the budget.
    pub fn tick_chunk_relight_active(
        &mut self,
        active_chunks: impl IntoIterator<Item = ChunkPos>,
        columns_per_tick: usize,
    ) -> usize {
        let cache = &mut self.active_light_chunks;
        cache.ordered.clear();
        cache.ordered.extend(active_chunks);
        cache.ordered.sort_unstable_by_key(|pos| (pos.x, pos.z));
        cache.ordered.dedup();
        cache.keyed = false;
        self.relight_active_cycle(true, columns_per_tick)
    }

    /// [`Self::tick_chunk_relight_active`] for the live `activeChunkSet`: the union of the
    /// `(2r+1)²` column squares around each player's chunk `centres` (any order, duplicates
    /// allowed), `r = radius`.
    ///
    /// Same visits and the same `previousActiveChunkSet` evolution as passing that set to
    /// [`Self::tick_chunk_relight_active`], but the set is rebuilt only when the centres or the
    /// radius change (a player crosses a chunk border, joins or leaves, or the view distance
    /// changes), without hashing, and a steady tick resumes the scan where the cycle left off
    /// instead of re-testing the consumed prefix (#6/C1-2).
    pub fn tick_chunk_relight_active_around(
        &mut self,
        centres: &[ChunkPos],
        radius: i32,
        columns_per_tick: usize,
    ) -> usize {
        let changed = self.active_light_chunks.update(centres, radius);
        self.relight_active_cycle(changed, columns_per_tick)
    }

    /// The shared `WorldClient.updateBlocks` cycle over `active_light_chunks.ordered`.
    /// `changed` is false only when `ordered` is the very set the previous call used.
    fn relight_active_cycle(&mut self, changed: bool, columns_per_tick: usize) -> usize {
        {
            let cache = &mut self.active_light_chunks;
            let previous = &mut self.previous_active_light_chunks;
            if changed {
                // java: `previousActiveChunkSet.retainAll(activeChunkSet)`. Both lists are
                // sorted by `(x, z)`, so membership is a merge walk.
                let ordered = &cache.ordered;
                let mut j = 0;
                previous.retain(|pos| {
                    while j < ordered.len() && (ordered[j].x, ordered[j].z) < (pos.x, pos.z) {
                        j += 1;
                    }
                    j < ordered.len() && ordered[j] == *pos
                });
                cache.scan_from = 0;
            }
            // With an unchanged set `retainAll` is the identity: every entry of `previous` was
            // retained against, or consumed from, this same set.
            if previous.len() == cache.ordered.len() {
                previous.clear();
                cache.scan_from = 0;
            }
            cache.picked.clear();
            if columns_per_tick == 0 {
                return 0;
            }

            // Java's HashSet order is JVM-layout-specific. Sorting supplies the same stable
            // replay policy as the older all-loaded compatibility queue without changing set
            // membership or the at-most-ten lifecycle. `previous ⊆ ordered`, both sorted, so
            // membership is again a merge walk; it starts at `scan_from` because everything
            // before it is known consumed.
            let ordered = &cache.ordered;
            let mut i = cache.scan_from;
            if let Some(first) = ordered.get(i) {
                let mut p = previous.partition_point(|q| (q.x, q.z) < (first.x, first.z));
                while i < ordered.len() && cache.picked.len() < columns_per_tick {
                    let cpos = ordered[i];
                    i += 1;
                    if previous.get(p) == Some(&cpos) {
                        p += 1;
                        continue;
                    }
                    cache.picked.push(cpos);
                }
            }
            cache.scan_from = i;
            merge_sorted_disjoint(previous, &cache.picked);
        }
        // The selection never depends on what `enqueue_relight_checks` does, so visiting after
        // choosing is the same sequence as Java's interleaved add-then-check loop.
        let picked = std::mem::take(&mut self.active_light_chunks.picked);
        for &cpos in &picked {
            if self.is_chunk_loaded(cpos) {
                self.enqueue_relight_checks(cpos);
            }
        }
        let visited = picked.len();
        self.active_light_chunks.picked = picked;
        visited
    }

    /// One complete client tick of periodic light repair.
    ///
    /// This compatibility wrapper retains the world-level API used by existing callers while
    /// making the two vanilla schedules explicit: first visit every loaded column for gap
    /// repair, then apply the caller's bounded relight budget. java: the two calls are
    /// `ChunkProviderClient.unloadQueuedChunks` followed by `WorldClient.updateBlocks`.
    /// Returns the number of columns consumed by the bounded relight pass; callers that need the
    /// gap count can call [`Self::tick_chunk_gap_light`] directly.
    pub fn tick_chunk_light(&mut self, columns_per_tick: usize) -> usize {
        self.tick_chunk_gap_light();
        self.tick_chunk_relight(columns_per_tick)
    }

    // ---- mesh invalidation --------------------------------------------

    /// java: `RenderGlobal.markBlockForUpdate` / `notifyLightSet` (`:1972`/`:1980`) — the
    /// `(p-1)..(p+1)` box, bucketed by 16.
    pub fn mark_dirty_box(&mut self, p: BlockPos) {
        self.mark_dirty_block_range(p.x - 1, p.y - 1, p.z - 1, p.x + 1, p.y + 1, p.z + 1);
    }

    /// java: `ViewFrustum.markBlocksForUpdate` (`:104`) — `MathHelper.bucketInt(v, 16)` (which
    /// is a floor-divide, i.e. `>> 4`) on both ends of each axis, then a **triple nested loop**
    /// over the resulting box.
    ///
    /// The previous six-face version missed the 12 edge and 8 corner sections, which
    /// is where smooth lighting and AO read their diagonal samples from.
    pub(crate) fn mark_dirty_block_range(
        &mut self,
        x1: i32,
        y1: i32,
        z1: i32,
        x2: i32,
        y2: i32,
        z2: i32,
    ) {
        mark_dirty_block_range_into(&mut self.dirty, x1, y1, z1, x2, y2, z2);
    }

    /// java: `WorldClient.doPreChunk`/`NetHandlerPlayClient.handleChunkData`'s
    /// `markBlockRangeForRenderUpdate(x*16, 0, z*16, x*16+15, 256, z*16+15)`, which
    /// `RenderGlobal.markBlockRangeForRenderUpdate` (`:1992`) **expands by one block on every
    /// axis** before bucketing — so it re-meshes the 3×3 column neighbourhood, not just this
    /// column and not just its four face neighbours.
    pub fn mark_column_and_neighbours_dirty(&mut self, pos: ChunkPos) {
        // java: `chunkX << 4`, `+ 15` and RenderGlobal's `-1`/`+1` are wrapping `int`
        // arithmetic. S21's chunk coordinates are raw server ints, so a column past +/-2^27
        // wraps there and here alike, in every build profile: it aliases near the origin or
        // becomes an empty (`from > to`) range. A debug build used to panic on the overflow.
        let x = pos.x.wrapping_mul(16);
        let z = pos.z.wrapping_mul(16);
        self.mark_dirty_block_range(
            x.wrapping_sub(1),
            -1,
            z.wrapping_sub(1),
            x.wrapping_add(16),
            256,
            z.wrapping_add(16),
        );
    }

    /// Apply chunk-local writes and then finish their world-wide lighting work as one batch.
    ///
    /// This keeps the same refusal rules as [`World::set_block`]: out-of-range Y, unchanged
    /// states, unloaded chunks, and air into an absent section are ignored.  The chunk writes
    /// remain in caller order, so a malformed duplicate entry still leaves the last state in
    /// place; only the expensive world-wide work is coalesced.
    pub(crate) fn apply_block_changes_lit(&mut self, changes: &[(BlockPos, BlockState)]) {
        let mut pending = Vec::with_capacity(changes.len());
        for &(position, state) in changes {
            if !(0..256).contains(&position.y) {
                continue;
            }
            let old = self.block_at(position);
            if old == state || !self.chunks.contains_key(&position.chunk_pos()) {
                continue;
            }
            let outcome = {
                let cpos = position.chunk_pos();
                let (x, z) = local_xz(position);
                let Some(chunk) = self.chunks.get_mut(&cpos).map(Arc::make_mut) else {
                    continue;
                };
                chunk.set_block_relit(x, position.y, z, state)
            };
            if matches!(outcome, ChunkSetOutcome::Changed { .. }) {
                // The chunk-local write has landed; mirror the client-visible tile-entity half
                // immediately, as the sequential `set_block` path does.
                if self.block_at(position) == state {
                    self.reconcile_tile_entity_after_block_change(position, old, state);
                }
                pending.push(PendingBlockChange {
                    position,
                    state,
                    old,
                    outcome,
                });
            }
        }
        self.finish_block_change_batch(&pending);
    }

    /// Finish a batch of already-decoded block writes with one height/sky pass per affected
    /// column and one mesher invalidation per distinct position.
    ///
    /// The client receives `S22PacketMultiBlockChange` entries together, while vanilla's
    /// `WorldClient.handleMultiBlockChange` calls `invalidateRegionAndSetBlock` once per entry.
    /// Running the chunk-local writes first lets a packet which touches a column repeatedly (or
    /// several nearby columns) avoid repeating the expensive height-map/sky-column work against
    /// intermediate states.  Final block state, tile-entity replacement, and light values remain
    /// the same as the sequential calls; every changed position still dirties its local mesh
    /// neighbourhood.  This is also the safe construction
    /// path for large fixture worlds; callers which need the exact per-write callback cadence
    /// should continue using [`World::set_block`].
    pub(crate) fn finish_block_change_batch(&mut self, pending: &[PendingBlockChange]) {
        if pending.is_empty() {
            return;
        }

        // `Chunk.generateSkylightMap` rebuilds every column from the final storage, so one call
        // per chunk supersedes all per-entry calls for sections created at the old height.
        let regenerate: BTreeSet<ChunkPos> = pending
            .iter()
            .filter_map(|change| match change.outcome {
                ChunkSetOutcome::Changed {
                    regenerate_skylight: true,
                    ..
                } => Some(change.position.chunk_pos()),
                ChunkSetOutcome::Changed {
                    regenerate_skylight: false,
                    ..
                }
                | ChunkSetOutcome::Unchanged => None,
            })
            .collect();
        for cpos in &regenerate {
            self.generate_skylight_map(*cpos);
        }

        // A relight is a column operation.  Preserve the greatest requested starting point for
        // each column; `relight_begin` then scans the final blocks down to the new top exactly
        // once.  A generated map already did this work for its chunk.
        let mut relight: HashMap<(ChunkPos, u8, u8), i32> = HashMap::new();
        for change in pending {
            let ChunkSetOutcome::Changed { relight_at, .. } = change.outcome else {
                continue;
            };
            let Some(y) = relight_at else {
                continue;
            };
            let cpos = change.position.chunk_pos();
            if regenerate.contains(&cpos) {
                continue;
            }
            let (x, z) = local_xz(change.position);
            relight
                .entry((cpos, x, z))
                .and_modify(|current| *current = (*current).max(y))
                .or_insert(y);
        }
        let mut relight: Vec<_> = relight.into_iter().collect();
        relight.sort_by_key(|((cpos, x, z), y)| (cpos.x, cpos.z, *x, *z, *y));
        for ((cpos, x, z), y) in relight {
            self.relight_block(cpos, x, y, z);
        }

        // The skylight-occlusion flag is column-scoped.  Apply each final opacity transition
        // after relight, matching the reads in `Chunk.setBlockState` while avoiding duplicate
        // flags for a packet that rewrites one column several times.
        let mut occlusion: HashSet<(ChunkPos, u8, u8)> = HashSet::new();
        for change in pending {
            let ChunkSetOutcome::Changed { .. } = change.outcome else {
                continue;
            };
            let old_props = change.old.props();
            let new_props = change.state.props();
            if old_props.light_opacity != new_props.light_opacity {
                let p = change.position;
                if new_props.light_opacity < old_props.light_opacity
                    || self.light_for(SkyBlock::Sky, p) > 0
                    || self.light_for(SkyBlock::Block, p) > 0
                {
                    let (x, z) = local_xz(p);
                    occlusion.insert((p.chunk_pos(), x, z));
                }
            }
        }
        let mut occlusion: Vec<_> = occlusion.into_iter().collect();
        occlusion.sort_by_key(|(cpos, x, z)| (cpos.x, cpos.z, *x, *z));
        for (cpos, x, z) in occlusion {
            if let Some(chunk) = self.chunks.get_mut(&cpos).map(Arc::make_mut) {
                chunk.propagate_skylight_occlusion(x, z);
            }
        }

        // Block light is still a bounded flood fill, but only entries that changed opacity or
        // emission need it.  Deduplicating positions handles duplicate packet entries without
        // changing the final state or the resulting flood-fill seed.
        let mut check_light: HashSet<BlockPos> = HashSet::new();
        for change in pending {
            let ChunkSetOutcome::Changed { .. } = change.outcome else {
                continue;
            };
            let old_props = change.old.props();
            let new_props = change.state.props();
            if old_props.light_opacity != new_props.light_opacity
                || old_props.light_emission != new_props.light_emission
            {
                check_light.insert(change.position);
            }
        }
        let mut check_light: Vec<_> = check_light.into_iter().collect();
        check_light.sort_by_key(|p| (p.x, p.y, p.z));
        for p in check_light {
            self.check_light(p);
        }

        let mut dirty: HashSet<BlockPos> = HashSet::new();
        for change in pending {
            if matches!(change.outcome, ChunkSetOutcome::Changed { .. }) {
                dirty.insert(change.position);
            }
        }
        let mut dirty: Vec<_> = dirty.into_iter().collect();
        dirty.sort_by_key(|p| (p.x, p.y, p.z));
        for p in dirty {
            self.mark_dirty_box(p);
        }
    }

    // ---- the block write ------------------------------------------------------

    /// `World.setBlockState` + `Chunk.setBlockState`, light included. java:
    /// `World.setBlockState:369` — note the `!isRemote` guard there covers only the
    /// *neighbour-notify* half, never `checkLight`.
    pub(crate) fn set_block_lit(&mut self, p: BlockPos, s: BlockState) {
        let cpos = p.chunk_pos();
        let (x, z) = local_xz(p);
        let outcome = {
            let Some(chunk) = self.chunks.get_mut(&cpos).map(Arc::make_mut) else {
                return;
            };
            chunk.set_block_relit(x, p.y, z, s)
        };
        let ChunkSetOutcome::Changed {
            old,
            regenerate_skylight,
            relight_at,
        } = outcome
        else {
            return;
        };

        if regenerate_skylight {
            self.generate_skylight_map(cpos);
        } else {
            if let Some(y) = relight_at {
                self.relight_block(cpos, x, y, z);
            }
            // java: `if (j1 != k1 && (j1 < k1 || getLightFor(SKY, pos) > 0 ||
            //        getLightFor(BLOCK, pos) > 0)) this.propagateSkylightOcclusion(i, k);`
            // (`Chunk.setBlockState:735`). The two light reads happen **after** `relightBlock`
            // has already rewritten this column, which is why they cannot live inside
            // `set_block_relit`.
            let new_opacity = s.props().light_opacity;
            let old_opacity = old.props().light_opacity;
            if new_opacity != old_opacity
                && (new_opacity < old_opacity
                    || self.light_for(SkyBlock::Sky, p) > 0
                    || self.light_for(SkyBlock::Block, p) > 0)
            {
                if let Some(chunk) = self.chunks.get_mut(&cpos).map(Arc::make_mut) {
                    chunk.propagate_skylight_occlusion(x, z);
                }
            }
        }

        // java: `if (block.getLightOpacity() != block1.getLightOpacity() ||
        //        block.getLightValue() != block1.getLightValue()) this.checkLight(pos);`
        let (np, op) = (s.props(), old.props());
        if np.light_opacity != op.light_opacity || np.light_emission != op.light_emission {
            self.check_light(p);
        }

        // java: `markBlockForUpdate(pos)` — flag 2, which every client-applied change carries.
        self.mark_dirty_box(p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// S21's chunk x/z are raw server ints, and vanilla computes this dirty
    /// box with wrapping `int` arithmetic. `chunkX = Integer.MAX_VALUE` shifts to -16, so the
    /// box is x in [-17, 0] (columns -2..=0); `chunkX = 2^27 - 1` shifts to `2^31 - 16`, whose
    /// `+ 16` wraps to `i32::MIN`, an empty range. The old `* 16` panicked in debug builds.
    #[test]
    fn a_far_out_column_dirty_box_wraps_like_java_int_arithmetic() {
        let mut world = World::new();
        world.mark_column_and_neighbours_dirty(ChunkPos::new(i32::MAX, 0));
        let columns: std::collections::BTreeSet<i32> = world
            .drain_dirty_sections()
            .into_iter()
            .map(|section| section.x)
            .collect();
        assert_eq!(columns, std::collections::BTreeSet::from([-2, -1, 0]));

        world.mark_column_and_neighbours_dirty(ChunkPos::new((1 << 27) - 1, 0));
        assert!(world.drain_dirty_sections().is_empty());
    }

    /// java: `isAreaLoaded(center, radius, false)` builds its box with wrapping
    /// `int` arithmetic. Near `Integer.MAX_VALUE` the x range wraps to `start > end`, the loop is
    /// empty and the (empty) area counts as loaded; a wrapped y range fails the height gate.
    /// The old `+`/`-` panicked in debug builds.
    #[test]
    fn is_area_loaded_wraps_its_box_like_java_int_arithmetic() {
        let world = World::new();
        assert!(
            !world.is_area_loaded(BlockPos::new(0, 64, 0), 17, false),
            "control"
        );
        assert!(world.is_area_loaded(BlockPos::new(i32::MAX, 64, 0), 17, false));
        assert!(world.is_area_loaded(BlockPos::new(0, 64, i32::MIN), 17, false));
        assert!(!world.is_area_loaded(BlockPos::new(0, i32::MAX, 0), 17, false));
        assert!(!world.is_area_loaded(BlockPos::new(0, i32::MIN, 0), 17, true));
    }

    #[test]
    fn a_new_section_rebuilds_skylight_before_writing_the_requested_nibble() {
        let mut world = World::new();
        let chunk_pos = ChunkPos::new(0, 0);
        world.load_empty_chunk(chunk_pos);

        // A leaf at y=64 creates section 4 and raises the heightmap while still allowing the
        // top-down skylight descent to continue. Section 3 remains absent until setLightFor.
        let leaves = BlockState::new(mc_data::block_by_name("leaves").expect("known block"), 0);
        world.set_block(BlockPos::new(0, 64, 0), leaves);
        assert!(world.chunk(chunk_pos).unwrap().section(3).is_none());

        let target = BlockPos::new(0, 56, 0);
        world.set_light_for(SkyBlock::Sky, target, 3);

        // generateSkylightMap computes 6 here. If the requested nibble is written before that
        // rebuild (the old Rust order), this reads 6 instead of the caller's 3.
        assert_eq!(world.light_for(SkyBlock::Sky, target), 3);
    }

    #[test]
    fn neighbour_brightness_blocks_use_the_brightest_of_vanillas_five_samples() {
        let mut world = World::new();
        world.set_has_sky_light(false);
        world.load_empty_chunk(ChunkPos::new(0, 0));
        let target = BlockPos::new(4, 64, 4);
        let slab = BlockState::new(
            mc_data::block_by_name("stone_slab").expect("known block"),
            0,
        );
        assert!(slab.props().use_neighbor_brightness);
        world.set_block(target, slab);
        world.set_light_for(SkyBlock::Block, target, 2);
        world.set_light_for(SkyBlock::Block, BlockPos::new(5, 64, 4), 12);

        assert_eq!(world.light_subtracted(target, 0), 12);
    }

    #[test]
    fn light_beyond_the_horizontal_world_border_is_full_bright_before_y_is_checked() {
        let world = World::new();
        assert_eq!(
            world.light_subtracted(BlockPos::new(30_000_000, -1, 0), 11),
            15
        );
        assert_eq!(
            world.light_subtracted(BlockPos::new(-30_000_001, 256, 0), 11),
            15
        );
    }

    /// The pristine `enqueue_relight_checks` (guard evaluated inside every `check_light`), kept
    /// as the parity oracle and bench baseline for the hoisted area guard (NF1).
    fn reference_enqueue_relight_checks(world: &mut World, cpos: ChunkPos) {
        let base_x = cpos.x << 4;
        let base_z = cpos.z << 4;
        for _ in 0..8 {
            let q = {
                let Some(chunk) = world.chunks.get_mut(&cpos).map(Arc::make_mut) else {
                    return;
                };
                if chunk.queued_light_checks >= 4096 {
                    return;
                }
                let q = chunk.queued_light_checks;
                chunk.queued_light_checks += 1;
                q
            };
            let (j, k, l) = ((q % 16) as u8, (q / 16 % 16) as u8, (q / 256) as u8);
            for i1 in 0..16u8 {
                let bp = BlockPos::new(
                    base_x + i32::from(k),
                    (i32::from(j) << 4) + i32::from(i1),
                    base_z + i32::from(l),
                );
                let edge = i1 == 0 || i1 == 15 || k == 0 || k == 15 || l == 0 || l == 15;
                let interesting = match world.chunk(cpos).and_then(|c| c.section(j)) {
                    None => edge,
                    Some(section) => {
                        section.block(k, i1, l).props().material == mc_data::Material::Air
                    }
                };
                if !interesting {
                    continue;
                }
                for (dx, dy, dz) in FACINGS {
                    let np = BlockPos::new(bp.x + dx, bp.y + dy, bp.z + dz);
                    if world.block_at(np).props().light_emission > 0 {
                        world.check_light(np);
                    }
                }
                world.check_light(bp);
            }
        }
    }

    /// NF1 parity: one `is_area_loaded` per relight slot instead of two per cell leaves every
    /// light nibble, every queue cursor and the return values exactly where the pristine
    /// per-cell `check_light` loop left them. Unlit torches and glowstone near column edges,
    /// one missing column (so some slots fail the 17-block guard) and cursors that start off
    /// the 8-slot grid.
    #[test]
    fn relight_sweep_with_hoisted_area_guard_matches_the_per_cell_guard() {
        fn build() -> World {
            let mut world = World::new();
            for cx in -1..=2 {
                for cz in -1..=1 {
                    if (cx, cz) != (2, 1) {
                        world.load_empty_chunk(ChunkPos::new(cx, cz));
                    }
                }
            }
            let mut rng = mc_math::JavaRandom::new(0x0119_0f17);
            for _ in 0..300 {
                let p = BlockPos::new(
                    rng.next_i32_bound(64) - 16,
                    56 + rng.next_i32_bound(24),
                    rng.next_i32_bound(48) - 16,
                );
                let state = match rng.next_i32_bound(4) {
                    0 => BlockState::new(mc_data::BlockId(50), 5),
                    1 => BlockState::new(mc_data::BlockId(89), 0),
                    _ => BlockState::new(mc_data::BlockId(1), 0),
                };
                // Raw storage writes: the light is left stale for the sweep to repair.
                if let Some(chunk) = world.chunks.get_mut(&p.chunk_pos()).map(Arc::make_mut) {
                    let (x, z) = local_xz(p);
                    chunk.set_block_at(x, p.y as u8, z, state);
                }
            }
            let mut columns: Vec<ChunkPos> = world.chunks.keys().copied().collect();
            columns.sort_unstable_by_key(|pos| (pos.x, pos.z));
            for (i, cpos) in columns.into_iter().enumerate() {
                let chunk = world
                    .chunks
                    .get_mut(&cpos)
                    .map(Arc::make_mut)
                    .expect("loaded");
                chunk.queued_light_checks = [0, 3, 1000, 2051, 4090][i % 5];
            }
            world
        }
        let (mut hoisted, mut reference) = (build(), build());
        let mut columns: Vec<ChunkPos> = hoisted.chunks.keys().copied().collect();
        columns.sort_unstable_by_key(|pos| (pos.x, pos.z));
        for _ in 0..10 {
            for &cpos in &columns {
                hoisted.enqueue_relight_checks(cpos);
                reference_enqueue_relight_checks(&mut reference, cpos);
            }
        }
        for &cpos in &columns {
            assert_eq!(
                hoisted.chunk(cpos).unwrap().queued_light_checks,
                reference.chunk(cpos).unwrap().queued_light_checks
            );
            for y in 0..256 {
                for x in 0..16 {
                    for z in 0..16 {
                        let p = BlockPos::new(cpos.x * 16 + x, y, cpos.z * 16 + z);
                        for kind in [SkyBlock::Sky, SkyBlock::Block] {
                            assert_eq!(
                                hoisted.light_for(kind, p),
                                reference.light_for(kind, p),
                                "{kind:?} at {p:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Micro-bench (NF1/U6): `cargo test -p mc-world --lib -- --ignored bench_relight_sweep
    /// --nocapture`. A void map: 7×7 resident columns with no sections, sweeps mid-way
    /// (`queued_light_checks = 1000`, so the visits cross edge and interior slots), ten column
    /// visits per tick as the live budget. Light is already consistent, so this is the pure probe
    /// cost of the per-cell `check_light`s, pristine vs. one area guard per slot.
    #[test]
    #[ignore = "timing bench; run explicitly"]
    fn bench_relight_sweep_area_guard_vs_per_cell() {
        use std::time::Instant;
        let build = || {
            let mut world = World::new();
            for cx in -3..=3 {
                for cz in -3..=3 {
                    world.load_empty_chunk(ChunkPos::new(cx, cz));
                    let chunk = world.chunks.get_mut(&ChunkPos::new(cx, cz));
                    Arc::make_mut(chunk.expect("loaded")).queued_light_checks = 1000;
                }
            }
            world
        };
        // The inner 5×5 columns have their whole 17-block area resident.
        let mut columns = Vec::new();
        for cx in -2..=2 {
            for cz in -2..=2 {
                columns.push(ChunkPos::new(cx, cz));
            }
        }
        let (mut hoisted, mut reference) = (build(), build());
        let rounds = 16;
        let start = Instant::now();
        for _ in 0..rounds {
            for &cpos in &columns {
                reference_enqueue_relight_checks(&mut reference, cpos);
            }
        }
        let old = start.elapsed();
        let start = Instant::now();
        for _ in 0..rounds {
            for &cpos in &columns {
                hoisted.enqueue_relight_checks(cpos);
            }
        }
        let new = start.elapsed();
        let visits = f64::from(rounds) * columns.len() as f64;
        println!(
            "bench_relight_sweep per 10-visit tick: old={:.1}us new={:.1}us",
            old.as_secs_f64() * 1e7 / visits,
            new.as_secs_f64() * 1e7 / visits,
        );
    }

    /// The gap pass is deliberately independent from the ten-column relight budget.
    ///
    /// Negative control: the pre-split implementation only examined the one column popped from
    /// `light_tick_queue` when the budget was `1`; it therefore left the low-coordinate flag set
    /// because the queue pops the sorted high-coordinate entry first.
    #[test]
    fn gap_rechecks_visit_all_loaded_columns_before_the_relight_budget() {
        let mut world = World::new();
        let low = ChunkPos::new(0, 0);
        let high = ChunkPos::new(1, 0);
        world.load_empty_chunk(low);
        world.load_empty_chunk(high);

        // This is the exact side effect of `Chunk.propagateSkylightOcclusion`: a column-level
        // flag plus the aggregate short-circuit tested by `Chunk.func_150804_b`.
        world
            .chunks
            .get_mut(&low)
            .map(Arc::make_mut)
            .expect("low column is loaded")
            .propagate_skylight_occlusion(0, 0);
        world
            .chunks
            .get_mut(&low)
            .map(Arc::make_mut)
            .expect("low column is loaded")
            .queued_light_checks = 0;
        world
            .chunks
            .get_mut(&high)
            .map(Arc::make_mut)
            .expect("high column is loaded")
            .queued_light_checks = 0;

        assert_eq!(
            world.tick_chunk_light(1),
            1,
            "relight still consumes one column"
        );
        let low_chunk = world.chunk(low).expect("low column is loaded");
        assert!(!low_chunk.update_skylight_columns[0]);
        // `only_one = true` matches the client-side `recheckGaps` call: it returns after the
        // first flagged cell, so the aggregate remains set until the next all-column visit.
        assert!(low_chunk.is_gap_lighting_updated);
        // The relight half still obeys its independent one-column budget and therefore touches
        // only the high coordinate popped from the sorted round-robin queue.
        assert_eq!(
            world.chunk(high).unwrap().queued_light_checks,
            8,
            "the relight budget is still independent"
        );
        assert_eq!(
            world.chunk(low).unwrap().queued_light_checks,
            0,
            "the low column was not consumed by the one-column relight budget"
        );
    }

    /// F-P1-client.multiplayer-02-6: inactive loaded columns must not steal the bounded relight
    /// budget. Once every active column has been seen, the Java previous-active set clears and a
    /// new cycle starts without admitting the outside column.
    #[test]
    fn active_relight_excludes_loaded_outside_columns_and_restarts_after_catch_up() {
        let mut world = World::new();
        let first = ChunkPos::new(-1, 0);
        let second = ChunkPos::new(0, 0);
        let outside = ChunkPos::new(30, 30);
        for pos in [first, second, outside] {
            world.load_empty_chunk(pos);
            world
                .chunks
                .get_mut(&pos)
                .map(Arc::make_mut)
                .expect("loaded test column")
                .queued_light_checks = 0;
        }

        assert_eq!(world.tick_chunk_relight_active([first, second], 1), 1);
        assert_eq!(world.chunk(first).unwrap().queued_light_checks, 8);
        assert_eq!(world.chunk(second).unwrap().queued_light_checks, 0);
        assert_eq!(world.chunk(outside).unwrap().queued_light_checks, 0);

        assert_eq!(world.tick_chunk_relight_active([first, second], 1), 1);
        assert_eq!(world.chunk(second).unwrap().queued_light_checks, 8);
        assert_eq!(world.chunk(outside).unwrap().queued_light_checks, 0);

        assert_eq!(world.tick_chunk_relight_active([first, second], 1), 1);
        assert_eq!(
            world.chunk(first).unwrap().queued_light_checks,
            16,
            "catching up clears previousActiveChunkSet and starts a fresh cycle"
        );
        assert_eq!(world.chunk(outside).unwrap().queued_light_checks, 0);
    }

    /// The pristine per-tick `tick_chunk_relight_active` body (fresh `HashSet`, retain, sort,
    /// scan from zero), kept as the parity oracle for the cached cycle (#6). Returns the
    /// consumed columns in visit order.
    fn reference_relight_active(
        previous: &mut HashSet<ChunkPos>,
        active: &HashSet<ChunkPos>,
        columns_per_tick: usize,
    ) -> Vec<ChunkPos> {
        previous.retain(|pos| active.contains(pos));
        if previous.len() == active.len() {
            previous.clear();
        }
        if columns_per_tick == 0 {
            return Vec::new();
        }
        let mut ordered: Vec<_> = active.iter().copied().collect();
        ordered.sort_unstable_by_key(|pos| (pos.x, pos.z));
        let mut visited = Vec::new();
        for cpos in ordered {
            if previous.contains(&cpos) {
                continue;
            }
            previous.insert(cpos);
            visited.push(cpos);
            if visited.len() >= columns_per_tick {
                break;
            }
        }
        visited
    }

    /// The pristine `active_player_chunks` loop over already-floored centres.
    fn reference_square_union(centres: &[ChunkPos], radius: i32) -> HashSet<ChunkPos> {
        let mut active = HashSet::new();
        for c in centres {
            for dx in -radius..=radius {
                for dz in -radius..=radius {
                    active.insert(ChunkPos::new(c.x.wrapping_add(dx), c.z.wrapping_add(dz)));
                }
            }
        }
        active
    }

    fn sorted(set: &HashSet<ChunkPos>) -> Vec<ChunkPos> {
        let mut v: Vec<_> = set.iter().copied().collect();
        v.sort_unstable_by_key(|pos| (pos.x, pos.z));
        v
    }

    #[test]
    fn chunk_square_union_matches_the_hashed_rebuild() {
        let mut rng = mc_math::JavaRandom::new(0x5eed);
        let (mut zs, mut out) = (Vec::new(), Vec::new());
        for case in 0..2_000 {
            let players = rng.next_i32_bound(8);
            let radius = rng.next_i32_bound(6) - 1; // includes the empty r = -1 square
            let spread = 1 + rng.next_i32_bound(40);
            let mut centres: Vec<ChunkPos> = (0..players)
                .map(|_| {
                    ChunkPos::new(
                        rng.next_i32_bound(spread) - spread / 2,
                        rng.next_i32_bound(spread) - spread / 2,
                    )
                })
                .collect();
            centres.sort_unstable_by_key(|pos| (pos.x, pos.z));
            centres.dedup();
            chunk_square_union(&centres, radius, &mut zs, &mut out);
            assert_eq!(
                out,
                sorted(&reference_square_union(&centres, radius)),
                "case {case}: centres {centres:?}, r {radius}"
            );
        }
        // Centres at the i32 edge take the literal (wrapping) construction.
        let edge = [
            ChunkPos::new(-5, i32::MIN + 1),
            ChunkPos::new(i32::MAX - 1, 0),
        ];
        chunk_square_union(&edge, 2, &mut zs, &mut out);
        assert_eq!(out, sorted(&reference_square_union(&edge, 2)));
    }

    /// #6 parity: the cached, cursor-resumed cycle consumes exactly the columns, in exactly the
    /// order, that the per-tick rebuild did, and leaves the same `previousActiveChunkSet`, while
    /// players walk across chunk borders, join, leave and the view distance changes. The
    /// iterator entry point is interleaved so both entries share one cycle state.
    #[test]
    fn cached_active_cycle_matches_the_per_tick_rebuild_over_a_scripted_walk() {
        for budget in [0usize, 1, 3, 10] {
            let mut rng = mc_math::JavaRandom::new(0xac71 + budget as i64);
            let mut world = World::new();
            let mut previous: HashSet<ChunkPos> = HashSet::new();
            let mut players = vec![
                ChunkPos::new(0, 0),
                ChunkPos::new(2, -1),
                ChunkPos::new(0, 0),
            ];
            let mut radius = 2;
            for tick in 0..1_500 {
                match rng.next_i32_bound(24) {
                    0 if players.len() < 6 => players.push(ChunkPos::new(
                        rng.next_i32_bound(13) - 6,
                        rng.next_i32_bound(13) - 6,
                    )),
                    1 if !players.is_empty() => {
                        let i = rng.next_i32_bound(players.len() as i32) as usize;
                        players.remove(i);
                    }
                    2 => radius = 1 + rng.next_i32_bound(3),
                    3..=6 if !players.is_empty() => {
                        let i = rng.next_i32_bound(players.len() as i32) as usize;
                        let step = if rng.next_bool() { 1 } else { -1 };
                        if rng.next_bool() {
                            players[i].x += step;
                        } else {
                            players[i].z += step;
                        }
                    }
                    _ => {} // a steady tick: nobody crossed a border
                }
                let active = reference_square_union(&players, radius);
                let want = reference_relight_active(&mut previous, &active, budget);
                let got = if tick % 97 == 13 {
                    world.tick_chunk_relight_active(active.iter().copied(), budget)
                } else {
                    world.tick_chunk_relight_active_around(&players, radius, budget)
                };
                assert_eq!(got, want.len(), "budget {budget}, tick {tick}");
                assert_eq!(
                    world.active_light_chunks.picked, want,
                    "budget {budget}, tick {tick}: visit order"
                );
                assert_eq!(
                    world.previous_active_light_chunks,
                    sorted(&previous),
                    "budget {budget}, tick {tick}: previousActiveChunkSet"
                );
            }
        }
    }

    /// TICK-6 parity: visiting only the sorted flagged columns repairs exactly what the pristine
    /// sort-every-loaded-column pass repaired (same light, same remaining flags, same return
    /// value), over several ticks of `only_one` rechecks, with and without sky.
    #[test]
    fn flagged_gap_pass_matches_the_sort_every_column_pass() {
        fn reference_gap_light(world: &mut World) -> usize {
            let mut all: Vec<ChunkPos> = world.loaded_chunks().map(|(pos, _)| pos).collect();
            all.sort_unstable_by_key(|pos| (pos.x, pos.z));
            let has_sky = world.has_sky_light();
            for cpos in &all {
                if has_sky
                    && world
                        .chunk(*cpos)
                        .is_some_and(|c| c.is_gap_lighting_updated)
                {
                    world.recheck_gaps(*cpos, true);
                }
            }
            all.len()
        }
        fn build(sky: bool) -> World {
            let mut world = World::new();
            if !sky {
                world.set_dimension_id(-1);
            }
            for cx in -2..=2 {
                for cz in -2..=2 {
                    world.load_empty_chunk(ChunkPos::new(cx, cz));
                }
            }
            let mut rng = mc_math::JavaRandom::new(0x6a9_5eed);
            for _ in 0..400 {
                // Columns -2..=2 span blocks -32..=47 on each axis.
                let p = BlockPos::new(
                    rng.next_i32_bound(80) - 32,
                    40 + rng.next_i32_bound(60),
                    rng.next_i32_bound(80) - 32,
                );
                let chunk = world.chunks.get_mut(&p.chunk_pos()).map(Arc::make_mut);
                let chunk = chunk.expect("inside the loaded square");
                let (x, z) = local_xz(p);
                chunk.set_block_at(x, p.y as u8, z, BlockState::new(mc_data::BlockId(1), 0));
                if rng.next_i32_bound(3) == 0 {
                    chunk.propagate_skylight_occlusion(x, z);
                }
            }
            world
        }
        for sky in [true, false] {
            let (mut fast, mut reference) = (build(sky), build(sky));
            let mut columns: Vec<ChunkPos> = fast.chunks.keys().copied().collect();
            columns.sort_unstable_by_key(|pos| (pos.x, pos.z));
            for tick in 0..6 {
                assert_eq!(
                    fast.tick_chunk_gap_light(),
                    reference_gap_light(&mut reference),
                    "sky {sky}, tick {tick}"
                );
                for &cpos in &columns {
                    let (a, b) = (fast.chunk(cpos).unwrap(), reference.chunk(cpos).unwrap());
                    assert_eq!(a.is_gap_lighting_updated, b.is_gap_lighting_updated);
                    assert_eq!(a.update_skylight_columns, b.update_skylight_columns);
                }
            }
            for &cpos in &columns {
                for y in 0..256 {
                    for x in 0..16 {
                        for z in 0..16 {
                            let p = BlockPos::new(cpos.x * 16 + x, y, cpos.z * 16 + z);
                            for kind in [SkyBlock::Sky, SkyBlock::Block] {
                                assert_eq!(
                                    fast.light_for(kind, p),
                                    reference.light_for(kind, p),
                                    "sky {sky}: {kind:?} at {p:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// TICK-6: a finished relight sweep (`queued_light_checks >= 4096`, every column's state
    /// after `Chunk::new`) must not deep-copy a column that a render snapshot still shares.
    #[test]
    fn finished_relight_sweep_does_not_unshare_a_snapshot_column() {
        let mut world = World::new();
        let pos = ChunkPos::new(0, 0);
        world.load_empty_chunk(pos);
        let snapshot = Arc::clone(world.chunks.get(&pos).expect("loaded"));
        assert_eq!(snapshot.queued_light_checks, 4096);
        world.enqueue_relight_checks(pos);
        assert!(Arc::ptr_eq(
            &snapshot,
            world.chunks.get(&pos).expect("loaded")
        ));
    }

    /// Micro-bench (#6): `cargo test -p mc-world --lib -- --ignored bench_active --nocapture`.
    /// Bookkeeping only (no loaded columns, so no `enqueueRelightChecks` work on either side):
    /// 12 players clustered on a SkyWars-sized area, r = 12, and a share of ticks on which one
    /// player crosses a chunk border.
    #[test]
    #[ignore = "timing bench; run explicitly"]
    fn bench_active_set_rebuild_vs_cached() {
        use std::hint::black_box;
        use std::time::Instant;
        for (label, move_every) in [
            ("steady", 0),
            ("1 move / 4 ticks", 4),
            ("move every tick", 1),
        ] {
            let base: Vec<ChunkPos> = (0..12)
                .map(|i| ChunkPos::new(i % 4 * 2 - 3, i / 4 * 2 - 2))
                .collect();
            let ticks = 2_000;
            let script = |tick: usize, players: &mut Vec<ChunkPos>| {
                if move_every != 0 && tick.is_multiple_of(move_every) {
                    let i = tick % players.len();
                    players[i].x += if (tick / 7).is_multiple_of(2) { 1 } else { -1 };
                }
            };
            let mut players = base.clone();
            let mut previous = HashSet::new();
            let start = Instant::now();
            for tick in 0..ticks {
                script(tick, &mut players);
                // Pristine: the client's set, then `tick_chunk_relight_active`'s second set.
                let active: HashSet<ChunkPos> =
                    reference_square_union(&players, 12).into_iter().collect();
                black_box(reference_relight_active(&mut previous, &active, 10));
            }
            let old = start.elapsed();
            let mut players = base.clone();
            let mut world = World::new();
            let start = Instant::now();
            for tick in 0..ticks {
                script(tick, &mut players);
                black_box(world.tick_chunk_relight_active_around(&players, 12, 10));
            }
            let new = start.elapsed();
            println!(
                "bench_active_set [{label}] active={} old={:.1}us/tick new={:.2}us/tick",
                world.active_light_chunks.ordered.len(),
                old.as_secs_f64() * 1e6 / ticks as f64,
                new.as_secs_f64() * 1e6 / ticks as f64,
            );
        }
    }
}
