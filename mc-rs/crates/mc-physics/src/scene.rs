//! Building synthetic worlds, for tests and offline tooling.
//!
//! [`mc_world::World::set_block`] on an unloaded position writes the client's shared blank
//! chunk (`ChunkProviderClient.blankChunk`), which every absent column reads back and which
//! leaves the column unloaded (so `moveEntityWithHeading`'s gravity pin still applies). A
//! synthetic scene must load the real chunk before writing to get an ordinary resident column.
//!
//! These helpers load the chunk first. They exist here rather than in `mc-world` because S13
//! needed them and S08 is `DONE`; the §13 cross-section request that used to live here shipped
//! as [`mc_world::World::load_empty_chunk`], which [`ensure_chunk`] now calls directly.

use std::collections::BTreeMap;

use mc_data::BlockState;
use mc_math::{BlockPos, ChunkPos};
use mc_world::{expected_chunk_data_len, Chunk, World};

/// Loads `pos` as an all-air chunk if it is not loaded already.
pub fn ensure_chunk(w: &mut World, pos: ChunkPos) {
    if w.is_chunk_loaded(pos) {
        return;
    }
    w.load_empty_chunk(pos);
}

/// [`mc_world::World::set_block`], but loading the chunk first so the write always lands.
pub fn set_block(w: &mut World, p: BlockPos, state: BlockState) {
    if !(0..256).contains(&p.y) {
        return;
    }
    ensure_chunk(w, p.chunk_pos());
    w.set_block(p, state);
}

/// Fills the inclusive box `from..=to` with `state`, loading chunks as needed.
///
/// The positions are collected in source order and sent through [`World::apply_block_changes`]
/// once.  This preserves the final fixture state while avoiding a full light relight for every
/// block in a large synthetic scene; use [`set_block`] when a test specifically needs the
/// one-write-at-a-time client path.
pub fn fill(w: &mut World, from: BlockPos, to: BlockPos, state: BlockState) {
    let (x0, x1) = (from.x.min(to.x), from.x.max(to.x));
    let (y0, y1) = (from.y.min(to.y), from.y.max(to.y));
    let (z0, z1) = (from.z.min(to.z), from.z.max(to.z));
    let mut changes = Vec::new();
    for x in x0..=x1 {
        for z in z0..=z1 {
            ensure_chunk(w, BlockPos::new(x, 0, z).chunk_pos());
            for y in y0..=y1 {
                if (0..256).contains(&y) {
                    changes.push((BlockPos::new(x, y, z), state));
                }
            }
        }
    }
    w.apply_block_changes(&changes);
}

/// Applies a batch of inclusive fixture fills through a lighting-free full-column path.
///
/// The old [`fill`] helper intentionally exercises [`World::set_block`] and is useful for
/// small behaviour tests, but that setter performs a light flood fill for every write. Physics
/// parity fixtures contain large static terrain and do not use light, so this helper preserves
/// fill order in local [`Chunk`] values, then installs each changed column with one full
/// packet-shaped [`World::apply_chunk_data`] call. Block states, section presence, biome bytes,
/// loaded-column residency, and the world's biome provider are retained; block and sky light are
/// deliberately zero-filled because static physics fixtures do not reconstruct lighting.
pub fn fill_static<I>(w: &mut World, fills: I)
where
    I: IntoIterator<Item = (BlockPos, BlockPos, BlockState)>,
{
    let mut columns = BTreeMap::<ChunkPos, Chunk>::new();

    for (from, to, state) in fills {
        let (x0, x1) = (from.x.min(to.x), from.x.max(to.x));
        let (y0, y1) = (from.y.min(to.y), from.y.max(to.y));
        let (z0, z1) = (from.z.min(to.z), from.z.max(to.z));

        // Keep the original fill's residency behaviour, including boxes wholly outside the
        // world height: it loads every column in the horizontal box before dropping writes.
        for x in x0..=x1 {
            for z in z0..=z1 {
                let position = BlockPos::new(x, 0, z).chunk_pos();
                ensure_chunk(w, position);
                for y in y0..=y1 {
                    if !(0..256).contains(&y) {
                        continue;
                    }
                    let chunk = columns
                        .entry(position)
                        .or_insert_with(|| w.chunk(position).cloned().unwrap_or_else(Chunk::empty));
                    let local_x = x.rem_euclid(16) as u8;
                    let local_z = z.rem_euclid(16) as u8;
                    // Match World::set_block's no-op for an air write into a missing section.
                    // This keeps section presence equivalent to the per-block reference path.
                    if chunk.block_at(local_x, y as u8, local_z) != state {
                        chunk.set_block_at(local_x, y as u8, local_z, state);
                    }
                }
            }
        }
    }

    let has_sky_light = w.has_sky_light();
    for (position, chunk) in columns {
        let primary_bitmask =
            chunk
                .sections()
                .iter()
                .enumerate()
                .fold(0u16, |mask, (section, value)| {
                    if value.is_some() {
                        mask | (1u16 << section)
                    } else {
                        mask
                    }
                });
        if primary_bitmask == 0 {
            continue;
        }

        let section_count = primary_bitmask.count_ones() as usize;
        let mut data = Vec::with_capacity(expected_chunk_data_len(
            primary_bitmask,
            has_sky_light,
            true,
        ));
        // ChunkSection stores cells in y,z,x order, the same order consumed by fill_from_packet.
        for section_index in 0..16u8 {
            if let Some(section) = chunk.section(section_index) {
                for y in 0..16u8 {
                    for z in 0..16u8 {
                        for x in 0..16u8 {
                            data.extend_from_slice(&section.block(x, y, z).id.to_le_bytes());
                        }
                    }
                }
            }
        }
        // Static fixtures intentionally do not reconstruct lighting. The payload still needs
        // the protocol-shaped nibble regions so apply_chunk_data can decode the full column.
        data.resize(data.len() + section_count * 2048, 0);
        if has_sky_light {
            data.resize(data.len() + section_count * 2048, 0);
        }
        data.extend_from_slice(chunk.biomes());
        w.apply_chunk_data(position, true, primary_bitmask, &data)
            .expect("static fixture chunk encoding must be accepted");
    }
}

/// A flat world: `state` filling `y = floor_y` over `radius` blocks in each horizontal
/// direction from the origin, air above.
#[must_use]
pub fn flat(state: BlockState, floor_y: i32, radius: i32) -> World {
    let mut w = World::new();
    fill(
        &mut w,
        BlockPos::new(-radius, floor_y, -radius),
        BlockPos::new(radius, floor_y, radius),
        state,
    );
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_data::block_by_name;

    #[test]
    fn writes_land_once_the_chunk_is_loaded() {
        let stone = BlockState::new(block_by_name("stone").unwrap(), 0);
        let mut w = World::new();
        let pos = BlockPos::new(500, 64, 500);

        // A direct unloaded write lands in the shared blank chunk (EmptyChunk inherits
        // `Chunk.setBlockState`) but leaves the column absent. The scene helper must turn it
        // into a resident chunk, which starts as fresh air, before writing.
        w.set_block(pos, stone);
        assert_eq!(w.block_at(pos), stone);
        assert!(w.chunk(pos.chunk_pos()).is_none());

        let glass = BlockState::new(block_by_name("glass").unwrap(), 0);
        set_block(&mut w, pos, glass);
        assert!(w.chunk(pos.chunk_pos()).is_some());
        assert_eq!(w.block_at(pos), glass);
    }

    #[test]
    fn fill_spans_chunk_boundaries() {
        let stone = BlockState::new(block_by_name("stone").unwrap(), 0);
        let mut w = World::new();
        fill(
            &mut w,
            BlockPos::new(-20, 60, -20),
            BlockPos::new(20, 61, 20),
            stone,
        );
        for p in [
            BlockPos::new(-20, 60, -20),
            BlockPos::new(0, 61, 0),
            BlockPos::new(20, 60, 20),
            BlockPos::new(-1, 61, 17),
        ] {
            assert_eq!(w.block_at(p), stone, "{p:?}");
        }
        assert_eq!(w.block_at(BlockPos::new(21, 60, 0)), BlockState::AIR);
        assert_eq!(w.block_at(BlockPos::new(0, 62, 0)), BlockState::AIR);
    }

    #[test]
    fn out_of_world_writes_are_dropped_without_loading_a_chunk() {
        let stone = BlockState::new(block_by_name("stone").unwrap(), 0);
        let mut w = World::new();
        set_block(&mut w, BlockPos::new(0, 300, 0), stone);
        assert_eq!(w.loaded_chunk_count(), 0);
    }

    #[test]
    fn static_fill_matches_reference_across_negative_boundaries_and_overlaps() {
        let stone = BlockState::new(block_by_name("stone").unwrap(), 0);
        let dirt = BlockState::new(block_by_name("dirt").unwrap(), 0);
        let fills = [
            (BlockPos::new(-17, 10, -1), BlockPos::new(16, 10, 1), stone),
            (BlockPos::new(-16, 10, -1), BlockPos::new(0, 12, 0), dirt),
            (
                BlockPos::new(-33, 255, 17),
                BlockPos::new(-16, 255, 17),
                stone,
            ),
        ];
        let mut reference = World::new();
        for &(from, to, state) in &fills {
            fill(&mut reference, from, to, state);
        }
        let mut bulk = World::new();
        fill_static(&mut bulk, fills);

        assert_eq!(bulk.loaded_chunk_count(), reference.loaded_chunk_count());
        for x in -35..=18 {
            for z in -2..=18 {
                for y in [0, 9, 10, 11, 12, 13, 254, 255] {
                    let p = BlockPos::new(x, y, z);
                    assert_eq!(
                        bulk.is_chunk_loaded(p.chunk_pos()),
                        reference.is_chunk_loaded(p.chunk_pos())
                    );
                    assert_eq!(bulk.block_at(p), reference.block_at(p), "{p:?}");
                }
            }
        }
        for (position, reference_chunk) in reference.loaded_chunks() {
            let bulk_chunk = bulk.chunk(position).expect("bulk retained residency");
            assert_eq!(
                bulk_chunk
                    .sections()
                    .iter()
                    .map(Option::is_some)
                    .collect::<Vec<_>>(),
                reference_chunk
                    .sections()
                    .iter()
                    .map(Option::is_some)
                    .collect::<Vec<_>>(),
                "section residency at {position:?}"
            );
            assert_eq!(
                bulk_chunk.biomes(),
                reference_chunk.biomes(),
                "biomes at {position:?}"
            );
        }
        // The bulk fixture's light is intentionally not compared with the relit reference path.
        assert_eq!(bulk.block_light_at(BlockPos::new(0, 10, 0)), 0);
    }
}
