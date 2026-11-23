//! Building `mc_world::World`s to mesh, without a server.
//!
//! [`mesh_section`](super::mesh_section) is a pure function of a `World`, which is the whole
//! reason this section can be verified numerically in a sandbox with no network: a test can
//! state exactly which blocks exist and then assert on exact quad counts, exact culling and
//! exact per-vertex AO. `mc_world` deliberately exposes no "insert a blank chunk" entry point
//! (a real client only ever gets chunks from a packet), so these helpers go through
//! `World::apply_chunk_data` with a synthetic full-column payload of the documented shape.
//!
//! This is test-support API. It is `pub` because `tests/s11_*.rs` need it too, not because a
//! game client should call it.

use mc_data::BlockState;
use mc_math::{BlockPos, ChunkPos};
use mc_world::World;

/// A full-column `S21PacketChunkData` payload: all sixteen sections present, every block air,
/// block light 0, sky light 15, biome id `biome` everywhere.
#[must_use]
pub fn blank_column_payload(has_sky_light: bool, biome: u8) -> Vec<u8> {
    // blocks: little-endian u16, all air (H6); then block light, all zero.
    let mut data = vec![0u8; 16 * 4096 * 2 + 16 * 2048];
    if has_sky_light {
        data.resize(data.len() + 16 * 2048, 0xFF); // sky light: full
    }
    data.resize(data.len() + 256, biome); // biomes
    data
}

/// Load a blank chunk column at `pos`. Panics only on a payload/length mismatch, which would
/// be a bug in this helper rather than in the caller.
pub fn load_blank_chunk(world: &mut World, pos: ChunkPos) {
    let payload = blank_column_payload(world.has_sky_light(), 1);
    world
        .apply_chunk_data(pos, true, 0xFFFF, &payload)
        .expect("blank column payload length");
}

/// A world with `(2 * radius + 1)^2` blank chunks loaded around the origin chunk, sky light
/// on, everything air, plains biome.
#[must_use]
pub fn blank_world(radius_chunks: i32) -> World {
    let mut w = World::new();
    w.set_has_sky_light(true);
    for cx in -radius_chunks..=radius_chunks {
        for cz in -radius_chunks..=radius_chunks {
            load_blank_chunk(&mut w, ChunkPos::new(cx, cz));
        }
    }
    w
}

/// Resolve a block by its registry name and place its default state, panicking with a useful
/// message when the name is not registered.
pub fn place(world: &mut World, pos: BlockPos, name: &str) {
    world.set_block(pos, state_of(name));
}

/// The default state of a registered block, by name.
///
/// # Panics
/// If `name` is not a registered block.
#[must_use]
pub fn state_of(name: &str) -> BlockState {
    mc_data::block_by_name(name)
        .unwrap_or_else(|| panic!("{name} is not a registered block"))
        .default_state()
}

/// Fill an axis-aligned box (inclusive) with one state.
pub fn fill(world: &mut World, from: BlockPos, to: BlockPos, state: BlockState) {
    for x in from.x.min(to.x)..=from.x.max(to.x) {
        for y in from.y.min(to.y)..=from.y.max(to.y) {
            for z in from.z.min(to.z)..=from.z.max(to.z) {
                world.set_block(BlockPos::new(x, y, z), state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_world_is_air_and_fully_sky_lit() {
        let w = blank_world(1);
        assert_eq!(w.loaded_chunk_count(), 9);
        assert!(w.block_at(BlockPos::new(0, 64, 0)).props().is_air());
        assert_eq!(w.sky_light_at(BlockPos::new(0, 64, 0)), 15);
        assert_eq!(w.block_light_at(BlockPos::new(0, 64, 0)), 0);
    }

    #[test]
    fn placed_blocks_read_back() {
        let mut w = blank_world(0);
        place(&mut w, BlockPos::new(3, 20, 5), "minecraft:stone");
        assert_eq!(
            w.block_at(BlockPos::new(3, 20, 5)),
            state_of("minecraft:stone")
        );
    }
}
