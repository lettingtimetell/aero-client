//! W-RCLICK — differential test for the block-placement predicates.
//!
//! Replays the neighbourhoods in `tests/fixtures/placement/placement.bin` through `mc-world`
//! and compares [`World::can_place_block_at`] and [`World::can_place_block_on_side`] against
//! what the **real** 1.8.9 `Block.canPlaceBlockAt` / `Block.canPlaceBlockOnSide` answered for
//! the same world.
//!
//! The fixture is produced by `javaref/PlacementRefGen.java` (see `javaref/generate.ps1`),
//! which drives a genuine `net.minecraft.world.World` with a real `ChunkProviderClient` under
//! the recompiled vanilla classes. Every one of the 29 `canPlaceBlockAt` overrides and 8
//! `canPlaceBlockOnSide` overrides is exercised, on all 256 block ids and all six faces, so
//! the comparison covers the whole registry rather than the blocks somebody thought to list.
//!
//! **What would let this pass while the code is wrong:** a divergence that no template's
//! neighbourhood can reach. The templates cover ground/side/above supports for every
//! `instanceof` arm the predicates test, plus both halves of each two-way distinction
//! (stairs/slab top vs bottom, snow depth 1 vs 7, still vs flowing water, single vs double
//! chest, occupied vs empty cell, y = 0/254/255). They do **not** cover: an unloaded chunk
//! (`World.isBlockNormalCube`'s `_default` arm — see the unit test at the bottom), or a
//! neighbourhood in which two overrides interact through a block placed by a *previous*
//! placement, which vanilla never evaluates either.

use std::path::PathBuf;

use mc_data::{BlockId, BlockState};
use mc_math::{BlockPos, ChunkPos, Facing};
use mc_world::World;

const MAGIC: &[u8; 8] = b"MCPLACE1";

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn u8(&mut self) -> u8 {
        let v = self.data[self.at];
        self.at += 1;
        v
    }
    fn u16(&mut self) -> u16 {
        let v = u16::from_be_bytes([self.data[self.at], self.data[self.at + 1]]);
        self.at += 2;
        v
    }
    fn i32(&mut self) -> i32 {
        let v = i32::from_be_bytes([
            self.data[self.at],
            self.data[self.at + 1],
            self.data[self.at + 2],
            self.data[self.at + 3],
        ]);
        self.at += 4;
        v
    }
    fn u32(&mut self) -> u32 {
        self.i32() as u32
    }
    fn bytes(&mut self, n: usize) -> &'a [u8] {
        let v = &self.data[self.at..self.at + n];
        self.at += n;
        v
    }
}

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/placement/placement.bin")
}

#[test]
fn every_block_and_face_matches_vanilla_can_place_block_at() {
    let raw = std::fs::read(fixture()).expect("placement fixture; run javaref/generate.ps1");
    let mut r = Reader { data: &raw, at: 0 };
    assert_eq!(r.bytes(8), MAGIC, "fixture magic");

    let template_count = r.u32();
    assert!(template_count > 0);

    let mut mismatches: Vec<String> = Vec::new();
    let mut checked = 0usize;
    let mut true_answers = 0usize;

    for _ in 0..template_count {
        let name_len = usize::from(r.u16());
        let name = String::from_utf8(r.bytes(name_len).to_vec()).unwrap();
        let radius = r.i32();

        // The Java side is a Nether world so no sky light exists on either side; see the
        // generator's header for why that matters to `BlockMushroom`.
        let mut world = World::new();
        world.set_has_sky_light(false);
        for cx in -radius..=radius {
            for cz in -radius..=radius {
                world.load_empty_chunk(ChunkPos::new(cx, cz));
            }
        }

        let op_count = r.u32();
        for _ in 0..op_count {
            let (x, y, z) = (r.i32(), r.i32(), r.i32());
            let state = BlockState::from_raw(r.u16());
            world.set_block(BlockPos::new(x, y, z), state);
        }

        let cell = BlockPos::new(r.i32(), r.i32(), r.i32());
        let java_light = r.u8();
        assert_eq!(
            world.light_at(cell),
            java_light,
            "{name}: block light at the placement cell diverged before any placement predicate \
             ran — this is an A12-01 light-propagation difference, not a placement one"
        );

        let case_count = r.u32();
        for _ in 0..case_count {
            let block = BlockId(r.u16());
            let side = Facing::from_index(i32::from(r.u8()));
            let java_on_side = r.u8() == 1;
            let java_at = r.u8() == 1;

            let ours_on_side = world.can_place_block_on_side(block, cell, side);
            let ours_at = world.can_place_block_at(block, cell);
            checked += 2;
            true_answers += usize::from(java_on_side) + usize::from(java_at);

            if ours_on_side != java_on_side && mismatches.len() < 40 {
                mismatches.push(format!(
                    "{name}: canPlaceBlockOnSide({}, {side:?}) java={java_on_side} ours={ours_on_side}",
                    mc_data::block_info(block).name,
                ));
            }
            if ours_at != java_at && mismatches.len() < 40 {
                mismatches.push(format!(
                    "{name}: canPlaceBlockAt({}) java={java_at} ours={ours_at}",
                    mc_data::block_info(block).name,
                ));
            }
        }
    }

    assert_eq!(r.at, raw.len(), "trailing bytes in the fixture");
    assert!(
        mismatches.is_empty(),
        "{} of {checked} answers diverged from vanilla:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );

    // A predicate that returned a constant would pass a comparison whose expected answers are
    // all the same value. They are not: both outcomes appear in bulk.
    assert!(checked > 100_000, "fixture too small to mean anything");
    assert!(
        true_answers > checked / 20 && true_answers < checked - checked / 20,
        "expected answers are lopsided ({true_answers} true of {checked}); a constant \
         predicate could pass this test"
    );
}

/// The one arm the differential cannot reach: `World.isBlockNormalCube`'s `_default`, which
/// fires for an unreceived chunk. `PlacementRefGen` loads every column it probes, and the
/// `IBlockAccess`-based `doesBlockHaveSolidTopSurface` has no such arm — so this is the
/// distinction between the two, checked directly.
///
/// java: `World.isBlockNormalCube` (`World.java:2429`) returns `_default` when
/// `chunkProvider.provideChunk(pos)` hands back an `EmptyChunk`; `BlockTorch.canPlaceAt`
/// passes `true`, which is why a wall torch may be placed against an unloaded neighbour.
#[test]
fn is_block_normal_cube_falls_back_to_its_default_in_an_unloaded_chunk() {
    let mut world = World::new();
    world.set_has_sky_light(false);
    world.load_empty_chunk(ChunkPos::new(0, 0));
    world.set_block(BlockPos::new(0, 70, 0), BlockState::new(BlockId(1), 0));

    assert!(world.is_block_normal_cube(BlockPos::new(0, 70, 0), false));
    assert!(!world.is_block_normal_cube(BlockPos::new(1, 70, 0), false));

    // One chunk over, never received.
    let far = BlockPos::new(100, 70, 0);
    assert!(world.is_block_normal_cube(far, true), "default true");
    assert!(!world.is_block_normal_cube(far, false), "default false");
    // ...where the `IBlockAccess` predicate simply reads air.
    assert!(!world.does_block_have_solid_top_surface(far));

    // And out of the world vertically, on both sides of the range.
    assert!(world.is_block_normal_cube(BlockPos::new(0, -1, 0), true));
    assert!(world.is_block_normal_cube(BlockPos::new(0, 256, 0), true));
}
