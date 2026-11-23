//! Client-side light propagation and the mesh-invalidation family.
//!
//! Every world here is built from a **real `S21PacketChunkData` payload whose light nibbles are
//! all zero** — i.e. what a server that has lit nothing at all would send. That is deliberate:
//! it means every non-zero light level asserted below was produced by this crate's own flood
//! fill and could not have arrived off the wire. Before A12-01 these worlds stayed black.
//!
//! The differential test against the real recompiled 1.8.9 classes lives next door in
//! `light_parity.rs`; this file is the hand-built complement that runs without a JDK.

use mc_data::{block_by_name, BlockState};
use mc_math::{BlockPos, ChunkPos, ChunkSectionPos};
use mc_world::{expected_chunk_data_len, SkyBlock, World};

fn state(name: &str, meta: u8) -> BlockState {
    BlockState::new(block_by_name(name).expect("known block"), meta)
}

/// A full-column payload with every section present and every light nibble zero.
fn column_payload(blocks: &dyn Fn(u8, u8, u8) -> BlockState) -> Vec<u8> {
    let mut out = Vec::with_capacity(expected_chunk_data_len(0xFFFF, true, true));
    for s in 0..16u8 {
        for y in 0..16u8 {
            for z in 0..16u8 {
                for x in 0..16u8 {
                    let id = blocks(x, (s << 4) | y, z).id;
                    out.push((id & 0xFF) as u8);
                    out.push((id >> 8) as u8);
                }
            }
        }
    }
    out.resize(out.len() + 16 * 2048, 0); // block light
    out.resize(out.len() + 16 * 2048, 0); // sky light
    out.resize(out.len() + 256, 1); // biomes: plains
    out
}

/// A `(2r+1)²` block of columns around chunk `(0, 0)`. `r >= 2` is what makes
/// `checkLightFor`'s `isAreaLoaded(pos, 17, false)` guard pass for a position near the origin.
fn build_world(r: i32, f: impl Fn(i32, i32, i32) -> BlockState) -> World {
    let mut w = World::new();
    for cx in -r..=r {
        for cz in -r..=r {
            let payload = column_payload(&|x, y, z| {
                f(
                    (cx << 4) + i32::from(x),
                    i32::from(y),
                    (cz << 4) + i32::from(z),
                )
            });
            w.apply_chunk_data(ChunkPos::new(cx, cz), true, 0xFFFF, &payload)
                .expect("well-formed payload");
        }
    }
    w.drain_dirty_sections();
    w
}

/// Stone up to `y = 79`, with a hollow 5×5×5 room centred on `(0, 67, 0)`.
fn sealed_room() -> World {
    let stone = state("stone", 0);
    build_world(2, |x, y, z| {
        if (-2..=2).contains(&x) && (-2..=2).contains(&z) && (65..=69).contains(&y) {
            BlockState::AIR
        } else if y <= 79 {
            stone
        } else {
            BlockState::AIR
        }
    })
}

// ---- Heightmap ---------------------------------------------------------------------

#[test]
fn a_chunk_packet_builds_the_heightmap_and_can_see_sky() {
    let w = sealed_room();
    // The roof is the topmost opaque block, so the heightmap sits one above it.
    assert_eq!(w.height(0, 0), 80);
    assert!(w.can_see_sky(BlockPos::new(0, 80, 0)));
    assert!(!w.can_see_sky(BlockPos::new(0, 79, 0)));
    // Inside the sealed room nothing sees sky — the base case `getRawLight` needs.
    assert!(!w.can_see_sky(BlockPos::new(0, 67, 0)));
    assert_eq!(w.chunks_lowest_horizon(0, 0), 80);
    // An unloaded column: `EmptyChunk.getHeightValue`/`getLowestHeight` are both 0.
    assert_eq!(w.height(10_000, 10_000), 0);
    assert_eq!(w.chunks_lowest_horizon(10_000, 10_000), 0);
}

#[test]
fn precipitation_height_skips_what_rain_falls_through() {
    // Glass has light opacity 0 but its material blocks movement, so the heightmap ignores it
    // and the precipitation map does not. java: `Chunk.getPrecipitationHeight:1195`
    let stone = state("stone", 0);
    let glass = state("glass", 0);
    let mut w = build_world(1, |_x, y, _z| {
        if y == 70 {
            glass
        } else if y <= 63 {
            stone
        } else {
            BlockState::AIR
        }
    });
    assert_eq!(w.height(3, 3), 64, "glass is transparent to the heightmap");
    assert_eq!(w.precipitation_height(3, 3), 71, "rain lands on glass");
    // Breaking the glass invalidates the cached column (`Chunk.setBlockState:667`) and the
    // next read recomputes it.
    w.set_block(BlockPos::new(3, 70, 3), BlockState::AIR);
    assert_eq!(w.precipitation_height(3, 3), 64);
    // A column with nothing in it at all is vanilla's `-1`.
    let empty = build_world(1, |_, _, _| BlockState::AIR);
    assert_eq!(empty.precipitation_height(3, 3), -1);
}

/// `World.getPrecipitationHeight` delegates even for a missing client column: the provider's
/// persistent `EmptyChunk` inherits `Chunk.getPrecipitationHeight`, whose empty scan caches and
/// returns the vanilla `-1` sentinel. An absent-column `0` is a tempting but wrong synthetic
/// ground level (and changes the weather band around unloaded terrain).
#[test]
fn an_unloaded_column_keeps_vanillas_empty_precipitation_sentinel() {
    let world = World::new();

    assert_eq!(world.precipitation_height(0, 0), -1);
    // Exercise negative chunk/local coordinates as well; Java's `& 15` local mapping still
    // reaches the same shared blank column when no resident chunk exists.
    assert_eq!(world.precipitation_height(-17, 31), -1);
}

/// `Chunk.getPrecipitationHeight` stops only at a material that blocks movement or is liquid.
/// A torch has `Material.circuits`, whose `blocksMovement` is false, so it cannot act as a
/// weather roof even though it is a non-air block and emits light.
///
/// Negative control: changing the predicate to merely test for a non-air state makes this
/// return 71 and therefore kills the test.
#[test]
fn a_torch_does_not_stop_precipitation() {
    let stone = state("stone", 0);
    let torch = state("torch", 5);
    let w = build_world(1, |_x, y, _z| {
        if y == 70 {
            torch
        } else if y <= 63 {
            stone
        } else {
            BlockState::AIR
        }
    });

    assert_eq!(w.precipitation_height(3, 3), 64);
}

// ---- The biome sentinel ------------------------------------------------------------

#[test]
fn an_unset_biome_byte_resolves_to_plains() {
    let mut w = World::new();
    w.load_empty_chunk(ChunkPos::new(0, 0));
    let chunk = w.chunk(ChunkPos::new(0, 0)).expect("loaded");
    // The raw array still holds vanilla's `-1` fill...
    assert_eq!(chunk.biomes()[0], 0xFF);
    // ...but the accessor resolves it, as `Chunk.getBiome:1382` does.
    assert_eq!(chunk.biome(0, 0), 1);
}

// ---- Block light -------------------------------------------------------------------

#[test]
fn a_torch_lights_the_room_it_is_placed_in() {
    let mut w = sealed_room();
    let torch = state("torch", 5); // meta 5 = standing on the floor
    for p in [
        BlockPos::new(0, 67, 0),
        BlockPos::new(1, 67, 0),
        BlockPos::new(2, 67, 0),
    ] {
        assert_eq!(w.block_light_at(p), 0, "the packet lit nothing");
    }

    w.set_block(BlockPos::new(0, 67, 0), torch);

    // java: torch is `setLightLevel(0.9375F)` -> `(int)(0.9375 * 15) == 14`.
    assert_eq!(w.block_light_at(BlockPos::new(0, 67, 0)), 14);
    // The gradient is what proves a flood fill ran rather than a single-cell write.
    assert_eq!(w.block_light_at(BlockPos::new(1, 67, 0)), 13);
    assert_eq!(w.block_light_at(BlockPos::new(2, 67, 0)), 12);
    assert_eq!(w.block_light_at(BlockPos::new(0, 67, 2)), 12);
    // Light decays with the *Manhattan* path, not the Euclidean one, so a 2+2 diagonal is 4
    // steps from the torch, not 3.
    assert_eq!(
        w.block_light_at(BlockPos::new(2, 67, 2)),
        10,
        "diagonals too"
    );
    assert_eq!(w.block_light_at(BlockPos::new(0, 69, 0)), 12, "and upward");
    // Straight into the wall: stone absorbs everything.
    assert_eq!(w.block_light_at(BlockPos::new(4, 67, 0)), 0);
}

#[test]
fn removing_the_torch_darkens_everything_it_lit() {
    let mut w = sealed_room();
    w.set_block(BlockPos::new(0, 67, 0), state("torch", 5));
    assert_eq!(w.block_light_at(BlockPos::new(2, 67, 0)), 12);

    // The darkening half of `checkLightFor` — the branch that erases the old lit region before
    // re-brightening from its frontier.
    w.set_block(BlockPos::new(0, 67, 0), BlockState::AIR);
    for p in [
        BlockPos::new(0, 67, 0),
        BlockPos::new(1, 67, 0),
        BlockPos::new(2, 67, 0),
        BlockPos::new(2, 67, 2),
        BlockPos::new(0, 69, 0),
    ] {
        assert_eq!(w.block_light_at(p), 0, "{p:?} should be dark again");
    }
}

#[test]
fn two_torches_and_the_removal_of_one_leaves_the_others_field_intact() {
    let mut w = sealed_room();
    w.set_block(BlockPos::new(-2, 67, 0), state("torch", 5));
    w.set_block(BlockPos::new(2, 67, 0), state("torch", 5));
    assert_eq!(
        w.block_light_at(BlockPos::new(0, 67, 0)),
        12,
        "12 from both"
    );

    // This is the case a naive "erase then re-flood from the changed cell" gets wrong: the
    // surviving torch's contribution has to be rebuilt from the erased region's frontier.
    w.set_block(BlockPos::new(2, 67, 0), BlockState::AIR);
    assert_eq!(w.block_light_at(BlockPos::new(2, 67, 0)), 10);
    assert_eq!(w.block_light_at(BlockPos::new(0, 67, 0)), 12);
    assert_eq!(w.block_light_at(BlockPos::new(-2, 67, 0)), 14);
}

#[test]
fn a_wall_built_between_two_cells_darkens_the_far_side() {
    let mut w = sealed_room();
    w.set_block(BlockPos::new(-2, 67, 0), state("torch", 5));
    assert_eq!(w.block_light_at(BlockPos::new(2, 67, 0)), 10);

    // Seal the room in half. The far side loses its only light path.
    for y in 65..=69 {
        for z in -2..=2 {
            w.set_block(BlockPos::new(0, y, z), state("stone", 0));
        }
    }
    assert_eq!(
        w.block_light_at(BlockPos::new(-1, 67, 0)),
        13,
        "near side kept"
    );
    assert_eq!(
        w.block_light_at(BlockPos::new(2, 67, 0)),
        0,
        "far side sealed off"
    );
}

// ---- Sky light ---------------------------------------------------------------------

#[test]
fn mining_a_shaft_lets_daylight_down_it_and_into_a_side_tunnel() {
    let stone = state("stone", 0);
    let mut w = build_world(2, |_x, y, _z| if y <= 79 { stone } else { BlockState::AIR });
    assert_eq!(
        w.sky_light_at(BlockPos::new(0, 75, 0)),
        0,
        "solid rock is dark"
    );

    // Dig straight down. Each removal moves the heightmap, which is `relightBlock`'s job.
    for y in (70..=79).rev() {
        w.set_block(BlockPos::new(0, y, 0), BlockState::AIR);
    }
    for y in 70..=79 {
        assert_eq!(
            w.sky_light_at(BlockPos::new(0, y, 0)),
            15,
            "open shaft at y={y} should be full daylight"
        );
    }
    assert_eq!(w.height(0, 0), 70, "heightmap followed the shaft down");

    // Now a side tunnel off the bottom. Nothing in it sees sky, so every level here came from
    // `checkLightFor` spreading sideways out of the shaft.
    for x in 1..=4 {
        w.set_block(BlockPos::new(x, 70, 0), BlockState::AIR);
    }
    assert_eq!(w.sky_light_at(BlockPos::new(1, 70, 0)), 14);
    assert_eq!(w.sky_light_at(BlockPos::new(2, 70, 0)), 13);
    assert_eq!(w.sky_light_at(BlockPos::new(3, 70, 0)), 12);
    assert_eq!(w.sky_light_at(BlockPos::new(4, 70, 0)), 11);
    assert!(!w.can_see_sky(BlockPos::new(4, 70, 0)));

    // ...and plugging the shaft again takes it all back out.
    w.set_block(BlockPos::new(0, 79, 0), stone);
    assert_eq!(w.height(0, 0), 80);
    for x in 0..=4 {
        assert_eq!(
            w.sky_light_at(BlockPos::new(x, 70, 0)),
            0,
            "the tunnel is sealed again at x={x}"
        );
    }
}

#[test]
fn a_block_placed_under_open_sky_shadows_the_column_below_it() {
    let stone = state("stone", 0);
    let mut w = build_world(2, |_x, y, _z| if y <= 63 { stone } else { BlockState::AIR });
    assert_eq!(
        w.sky_light_at(BlockPos::new(0, 64, 0)),
        0,
        "packet sent no light"
    );

    // A 3×3 roof, so the centre column is genuinely two steps from open sky rather than one.
    for x in -1..=1 {
        for z in -1..=1 {
            w.set_block(BlockPos::new(x, 70, z), stone);
        }
    }
    assert_eq!(w.height(0, 0), 71);
    assert_eq!(
        w.sky_light_at(BlockPos::new(0, 71, 0)),
        15,
        "above the new block"
    );
    assert_eq!(w.sky_light_at(BlockPos::new(0, 70, 0)), 0, "inside it");
    // Under the roof: the edge of the 3×3 is one step from daylight, the centre is two.
    assert_eq!(w.sky_light_at(BlockPos::new(1, 69, 0)), 14);
    assert_eq!(w.sky_light_at(BlockPos::new(0, 69, 0)), 13);
    assert!(!w.can_see_sky(BlockPos::new(0, 69, 0)));
    assert!(w.can_see_sky(BlockPos::new(2, 69, 0)));
}

#[test]
fn the_nether_never_writes_sky_light() {
    let mut w = World::new();
    w.set_has_sky_light(false);
    let netherrack = state("netherrack", 0);
    let payload = column_payload(&|_x, y, _z| {
        if y <= 63 {
            netherrack
        } else {
            BlockState::AIR
        }
    });
    // A Nether payload has no sky-light region at all (H6).
    let nether_payload = &payload[..expected_chunk_data_len(0xFFFF, false, true)];
    for cx in -2..=2 {
        for cz in -2..=2 {
            w.apply_chunk_data(ChunkPos::new(cx, cz), true, 0xFFFF, nether_payload)
                .expect("well-formed");
        }
    }
    w.set_block(BlockPos::new(0, 70, 0), state("glowstone", 0));
    assert_eq!(
        w.block_light_at(BlockPos::new(1, 70, 0)),
        14,
        "block light still works"
    );
    assert_eq!(
        w.sky_light_at(BlockPos::new(0, 90, 0)),
        0,
        "no sky in the Nether"
    );
}

// ---- the loaded-area guard -------------------------------------------------------------------

#[test]
fn check_light_for_refuses_to_run_against_unloaded_columns() {
    // Exactly one column: the 17-block ball around its centre reaches its neighbours.
    let mut w = World::new();
    w.load_empty_chunk(ChunkPos::new(0, 0));
    assert!(!w.is_area_loaded(BlockPos::new(8, 70, 8), 17, false));
    assert!(!w.check_light_for(SkyBlock::Block, BlockPos::new(8, 70, 8)));
    // `allowEmpty = true` is unconditionally satisfied on the client — see the doc comment.
    assert!(w.is_area_loaded(BlockPos::new(8, 70, 8), 16, true));
    // ...but not outside the world's vertical extent.
    assert!(!w.is_area_loaded(BlockPos::new(8, 400, 8), 16, true));
}

// ---- Mesh invalidation --------------------------------------------------------

#[test]
fn a_block_at_a_section_corner_dirties_all_eight_sections() {
    let mut w = build_world(1, |_x, _y, _z| BlockState::AIR);
    w.drain_dirty_sections();
    // (16, 16, 16) is the (0,0,0) corner of section (1,1,1): vanilla's `(p-1)..(p+1)` box
    // buckets to sections 0..1 on every axis, i.e. eight of them.
    w.mark_dirty_box(BlockPos::new(16, 16, 16));
    let dirty = w.drain_dirty_sections();
    assert_eq!(dirty.len(), 8, "{dirty:?}");
    for sx in 0..=1 {
        for sy in 0..=1 {
            for sz in 0..=1 {
                assert!(dirty.contains(&ChunkSectionPos::new(sx, sy, sz)));
            }
        }
    }
    // A block in the middle of a section touches only that section.
    w.mark_dirty_box(BlockPos::new(24, 24, 24));
    assert_eq!(w.drain_dirty_sections().len(), 1);
}

#[test]
fn loading_a_column_dirties_its_neighbours_too() {
    let mut w = World::new();
    w.load_empty_chunk(ChunkPos::new(5, 5));
    let dirty = w.drain_dirty_sections();
    // java: `markBlockRangeForRenderUpdate(x*16, 0, z*16, x*16+15, 256, z*16+15)` expands by
    // one block on every axis, so the 3×3 column neighbourhood, 16 sections each.
    assert_eq!(dirty.len(), 9 * 16, "3x3 columns of 16 sections");
    for dx in -1..=1 {
        for dz in -1..=1 {
            assert!(dirty.contains(&ChunkSectionPos::new(5 + dx, 0, 5 + dz)));
            assert!(dirty.contains(&ChunkSectionPos::new(5 + dx, 15, 5 + dz)));
        }
    }
    // Never above or below the world.
    assert!(!dirty.iter().any(|s| !(0..16).contains(&s.y)));
}

#[test]
fn unloading_a_column_marks_it_dirty_rather_than_clearing_it() {
    let mut w = World::new();
    w.load_empty_chunk(ChunkPos::new(5, 5));
    w.drain_dirty_sections();
    w.unload_chunk(ChunkPos::new(5, 5));
    let dirty = w.drain_dirty_sections();
    // This used to `remove` these keys, leaving the renderer holding meshes for a
    // column that no longer exists.
    assert!(dirty.contains(&ChunkSectionPos::new(5, 7, 5)));
    assert_eq!(dirty.len(), 9 * 16);
}

// ---- The periodic repair pass ----------------------------------------------------------

#[test]
fn a_partial_chunk_packet_arms_the_relight_sweep_and_a_full_one_does_not() {
    let mut w = World::new();
    let full = column_payload(&|_x, _y, _z| BlockState::AIR);
    w.apply_chunk_data(ChunkPos::new(0, 0), true, 0xFFFF, &full)
        .unwrap();
    // java: `handleChunkData:768` only resets on a partial packet (or a non-surface world).
    assert_eq!(
        w.chunk(ChunkPos::new(0, 0)).unwrap().queued_light_checks(),
        4096
    );

    let partial = &full[..expected_chunk_data_len(0xFFFF, true, false)];
    w.apply_chunk_data(ChunkPos::new(0, 0), false, 0xFFFF, partial)
        .unwrap();
    assert_eq!(
        w.chunk(ChunkPos::new(0, 0)).unwrap().queued_light_checks(),
        0
    );
}

#[test]
fn the_tick_pass_is_bounded_and_visits_every_column_before_repeating() {
    let mut w = World::new();
    for cx in 0..4 {
        for cz in 0..4 {
            w.load_empty_chunk(ChunkPos::new(cx, cz));
        }
    }
    assert_eq!(w.tick_chunk_light(5), 5, "the budget is respected");
    assert_eq!(w.tick_chunk_light(5), 5);
    assert_eq!(w.tick_chunk_light(100), 6, "only 6 of 16 columns were left");
    assert_eq!(w.tick_chunk_light(0), 0);
    let mut empty = World::new();
    assert_eq!(empty.tick_chunk_light(64), 0);
}

#[test]
fn recheck_gaps_clears_the_flag_a_block_change_raised() {
    let stone = state("stone", 0);
    // A step in the terrain exactly on the x = 16 chunk boundary.
    let mut w = build_world(2, |x, y, _z| {
        let top = if x < 16 { 63 } else { 71 };
        if y <= top {
            stone
        } else {
            BlockState::AIR
        }
    });
    // `propagateSkylightOcclusion`'s guard is `opacity changed && (it *fell* || this cell
    // already has light)` (`Chunk.setBlockState:735`), so breaking a block always flags the
    // column and placing one into the dark does not.
    w.set_block(BlockPos::new(15, 64, 4), stone);
    assert!(!w
        .chunk(ChunkPos::new(0, 0))
        .unwrap()
        .is_gap_lighting_updated());

    w.set_block(BlockPos::new(15, 63, 4), BlockState::AIR);
    assert!(w
        .chunk(ChunkPos::new(0, 0))
        .unwrap()
        .is_gap_lighting_updated());
    // java: `recheckGaps(worldObj.isRemote)` returns after the *first* flagged column without
    // clearing `isGapLightingUpdated` (`Chunk.java:352-357`), so the client always needs a
    // second pass to find nothing left and drop the flag. That one-column-per-chunk-per-tick
    // trickle is exactly the budget this port keeps.
    w.tick_chunk_light(64);
    assert!(w
        .chunk(ChunkPos::new(0, 0))
        .unwrap()
        .is_gap_lighting_updated());
    w.tick_chunk_light(64);
    assert!(
        !w.chunk(ChunkPos::new(0, 0))
            .unwrap()
            .is_gap_lighting_updated(),
        "the second pass finds nothing flagged and clears it"
    );
}

// ---- The darkened query ----------------------------------------------------------------

#[test]
fn light_subtracted_darkens_only_the_sky_channel() {
    let mut w = build_world(2, |_x, y, _z| {
        if y <= 63 {
            state("stone", 0)
        } else {
            BlockState::AIR
        }
    });
    w.set_block(BlockPos::new(0, 64, 0), state("torch", 5));
    let p = BlockPos::new(1, 64, 0);
    // Sky 15 (open) and block 13 (one cell from the torch).
    assert_eq!(w.light_for(SkyBlock::Sky, p), 15);
    assert_eq!(w.light_for(SkyBlock::Block, p), 13);
    assert_eq!(w.light_subtracted(p, 0), 15);
    // Midnight subtracts 11: the sky channel drops to 4, the torch still wins at 13.
    assert_eq!(w.light_subtracted(p, 11), 13);
    // Twelve cells out the torch is down to 2, so midnight's sky-4 wins. (Still inside the
    // 17-block ball the sky fill reached when the torch's own `checkLight` ran.)
    let far = BlockPos::new(12, 64, 0);
    assert_eq!(w.light_for(SkyBlock::Block, far), 2);
    assert_eq!(w.light_subtracted(far, 11), 4);
    assert_eq!(w.light_subtracted(BlockPos::new(0, -1, 0), 0), 0);
}
