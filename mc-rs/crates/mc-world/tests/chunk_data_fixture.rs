//! Golden test: a hand-built `S21PacketChunkData`/`S26PacketMapChunkBulk`-shaped byte payload,
//! assembled independently of `mc-world`'s own unpack code from the documented wire layout
//! (see `src/chunk.rs`'s module docs and `mc-parity`'s hazard H6), decoded through
//! [`mc_world::World`], and checked against the known block ids / light / biome values that
//! went into it.
//!
//! No real packet capture is available yet (S06/`mc-protocol` has not landed), so this plays
//! the role the task list calls for in that case: a synthetic fixture with a documented byte
//! layout, verified end to end.

use mc_data::{block_by_name, BlockState};
use mc_math::{BlockPos, ChunkPos};
use mc_world::{BulkChunkEntry, World};

/// `ExtendedBlockStorage`/`NibbleArray`'s coordinate index: `y << 8 | z << 4 | x`. Duplicated
/// here (rather than calling into `mc-world`'s private helper) so the fixture is built from the
/// spec, not from the code under test.
fn idx(x: usize, y: usize, z: usize) -> usize {
    (y << 8) | (z << 4) | x
}

fn push_u16_le(buf: &mut Vec<u8>, v: u16) {
    buf.push((v & 0xFF) as u8);
    buf.push((v >> 8) as u8);
}

/// Assembles one section's worth of block data (2 bytes/cell, little-endian) from a sparse
/// fill: `default` everywhere, with `overrides` punched in at specific `(x, y, z)` cells.
fn section_blocks(default: u16, overrides: &[((usize, usize, usize), u16)]) -> [u16; 4096] {
    let mut cells = [default; 4096];
    for &((x, y, z), v) in overrides {
        cells[idx(x, y, z)] = v;
    }
    cells
}

fn append_blocks(buf: &mut Vec<u8>, cells: &[u16; 4096]) {
    for &c in cells {
        push_u16_le(buf, c);
    }
}

#[test]
fn s21_chunk_data_synthetic_fixture_decodes_to_known_states() {
    let stone = BlockState::new(block_by_name("stone").unwrap(), 0);
    let dirt = BlockState::new(block_by_name("dirt").unwrap(), 0);
    let diamond_block = BlockState::new(block_by_name("diamond_block").unwrap(), 0);

    // Section 0 (world y 0..15): air everywhere except two known cells.
    let section0 = section_blocks(
        0, // air
        &[((0, 0, 0), stone.id), ((5, 10, 7), diamond_block.id)],
    );
    // Section 1 (world y 16..31): dirt everywhere.
    let section1 = section_blocks(dirt.id, &[]);

    let bitmask: u16 = 0b0000_0000_0000_0011; // sections 0 and 1

    let mut payload = Vec::new();
    // 1. Blocks, in ascending section order.
    append_blocks(&mut payload, &section0);
    append_blocks(&mut payload, &section1);
    // 2. Block light: zero for section 0, a uniform nibble-1 for section 1.
    payload.extend(std::iter::repeat_n(0x00u8, 2048));
    payload.extend(std::iter::repeat_n(0x11u8, 2048)); // both nibbles = 1
                                                       // 3. Sky light: full brightness (both nibbles = 15) for both sections. Present because
                                                       //    this fixture models an overworld chunk (has_sky_light = true).
    payload.extend(std::iter::repeat_n(0xFFu8, 2048));
    payload.extend(std::iter::repeat_n(0xFFu8, 2048));
    // 4. Biomes: plains (id 1) everywhere.
    payload.extend(std::iter::repeat_n(1u8, 256));

    let mut world = World::new();
    assert!(
        world.has_sky_light(),
        "World::new defaults to an overworld-shaped sky"
    );

    let chunk_pos = ChunkPos::new(2, -3);
    world
        .apply_chunk_data(chunk_pos, true, bitmask, &payload)
        .expect("well-formed fixture must decode");

    // Known block IDs at known coordinates (the "Done when" criterion).
    assert_eq!(world.block_at(chunk_pos.origin(0)), stone);
    assert_eq!(
        world.block_at(BlockPos::new(
            chunk_pos.x * 16 + 5,
            10,
            chunk_pos.z * 16 + 7
        )),
        diamond_block
    );
    // A cell in section 0 that got no override is still air.
    assert_eq!(
        world.block_at(BlockPos::new(chunk_pos.x * 16 + 1, 1, chunk_pos.z * 16 + 1)),
        BlockState::AIR
    );
    // Section 1 (world y 16..31) is uniformly dirt.
    assert_eq!(
        world.block_at(BlockPos::new(chunk_pos.x * 16, 20, chunk_pos.z * 16)),
        dirt
    );
    // Above both loaded sections: no section 2 was sent, so it reads back as air.
    assert_eq!(
        world.block_at(BlockPos::new(chunk_pos.x * 16, 40, chunk_pos.z * 16)),
        BlockState::AIR
    );

    // Light: section 0 is block-light 0 / sky-light 15 -> effective 15. Section 1 is
    // block-light 1 / sky-light 15 -> still 15 (sky dominates when both are lit).
    let p0 = BlockPos::new(chunk_pos.x * 16, 0, chunk_pos.z * 16);
    assert_eq!(world.block_light_at(p0), 0);
    assert_eq!(world.sky_light_at(p0), 15);
    assert_eq!(world.light_at(p0), 15);
    let p1 = BlockPos::new(chunk_pos.x * 16, 20, chunk_pos.z * 16);
    assert_eq!(world.block_light_at(p1), 1);

    // Biomes.
    assert_eq!(world.chunk(chunk_pos).unwrap().biome(0, 0), 1);

    // Sections above the two sent ones were not touched, and are not present.
    assert!(world.chunk(chunk_pos).unwrap().section(2).is_none());
}

#[test]
fn nether_shaped_fixture_has_no_sky_light_region_and_biomes_still_land_correctly() {
    // Hazard H6 regression guard: has_sky_light = false must shrink the payload by exactly
    // the sky-light region, and the biome region (which comes *after* where sky light would
    // have been) must still be read from the right offset.
    let netherrack = BlockState::new(block_by_name("netherrack").unwrap(), 0);
    let section0 = section_blocks(netherrack.id, &[]);
    let bitmask: u16 = 0b1;

    let mut payload = Vec::new();
    append_blocks(&mut payload, &section0);
    payload.extend(std::iter::repeat_n(0x22u8, 2048)); // block light, nibble 2 throughout
                                                       // NOTE: no sky-light region here at all.
    payload.extend(std::iter::repeat_n(7u8, 256)); // biome id 7 (a placeholder) everywhere

    let mut world = World::new();
    world.set_has_sky_light(false); // e.g. after a respawn into the Nether

    let chunk_pos = ChunkPos::new(0, 0);
    world
        .apply_chunk_data(chunk_pos, true, bitmask, &payload)
        .expect("nether-shaped fixture (no sky light bytes) must decode");

    assert_eq!(world.block_at(chunk_pos.origin(0)), netherrack);
    assert_eq!(
        world.sky_light_at(chunk_pos.origin(0)),
        0,
        "no sky in this dimension"
    );
    assert_eq!(world.block_light_at(chunk_pos.origin(0)), 2);
    assert_eq!(world.light_at(chunk_pos.origin(0)), 2);
    assert_eq!(world.chunk(chunk_pos).unwrap().biome(0, 0), 7);

    // Feeding the *same* bytes back in as though sky light were present must fail loudly
    // (wrong length) rather than silently misreading the tail as sky light and shifting the
    // biome region — this is the failure mode H6 warns about.
    let mut misread_world = World::new();
    assert!(misread_world
        .apply_chunk_data(chunk_pos, true, bitmask, &payload)
        .is_err());
}

#[test]
fn s26_map_chunk_bulk_decodes_multiple_columns_and_sets_dimension_sky_flag() {
    let stone = BlockState::new(block_by_name("stone").unwrap(), 0);
    let sand = BlockState::new(block_by_name("sand").unwrap(), 0);

    let sec_a = section_blocks(stone.id, &[]);
    let sec_b = section_blocks(sand.id, &[]);

    let build = |cells: &[u16; 4096]| -> Vec<u8> {
        let mut buf = Vec::new();
        append_blocks(&mut buf, cells);
        buf.extend(std::iter::repeat_n(0u8, 2048)); // block light
        buf.extend(std::iter::repeat_n(0xFFu8, 2048)); // sky light (bulk packets always overworld here)
        buf.extend(std::iter::repeat_n(3u8, 256)); // biome
        buf
    };
    let data_a = build(&sec_a);
    let data_b = build(&sec_b);

    let entries = [
        BulkChunkEntry {
            chunk_x: 0,
            chunk_z: 0,
            primary_bitmask: 0b1,
            data: &data_a,
        },
        BulkChunkEntry {
            chunk_x: 1,
            chunk_z: 0,
            primary_bitmask: 0b1,
            data: &data_b,
        },
    ];

    let mut world = World::new();
    // The packet flag describes this payload's byte layout only. The world's dimension/provider
    // owns whether future chunks carry sky light, so one bulk packet must not overwrite it.
    world.set_has_sky_light(false);
    world
        .apply_chunk_bulk(true, &entries)
        .expect("bulk fixture must decode");

    assert!(!world.has_sky_light());
    assert_eq!(world.block_at(BlockPos::new(0, 0, 0)), stone);
    assert_eq!(world.block_at(BlockPos::new(16, 0, 0)), sand);
    assert_eq!(world.chunk(ChunkPos::new(0, 0)).unwrap().biome(0, 0), 3);
    assert_eq!(world.chunk(ChunkPos::new(1, 0)).unwrap().biome(0, 0), 3);
}
