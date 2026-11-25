//! The 1.7.10 collision query and the block shapes that differ from 1.8.9.
//!
//! java17: `World.getCollidingBoundingBoxes` (`world/World.java:1302-1358`). The scan range, the
//! entity part and every block shape are 1.8.9's except:
//!
//! * there is no world border: a column outside `[-30000000, 30000000)` on either horizontal
//!   axis is **stone** at every scanned `y`, whoever is asking (1.8.9 skips those columns and
//!   puts its stone at the border instead);
//! * carpet collides as a zero-height box at its base, cake's top is `0.4375`, and a closed
//!   fence gate picks its axis from the raw metadata.
//!
//! Every other block goes through the 1.8.9 shape code: per-block collision dumps of both
//! versions (java17 and java18 `Block.addCollisionBoxesToList`) show identical boxes for every
//! vanilla metadata value.

use mc_data::BlockState;
use mc_math::{floor_f64, Aabb, BlockPos};
use mc_world::World;

use crate::collision::{
    add_block_boxes_for, append_entity_collision_boxes, EntityCollisionCandidate,
};

/// java17: the `World.getCollidingBoundingBoxes` coordinate limit, `-30000000 <= x < 30000000`.
const WORLD_LIMIT: i32 = 30_000_000;

/// java17 block ids of the three shapes that differ.
const FENCE_GATE_ID: u16 = 107;
const CAKE_ID: u16 = 92;
const CARPET_ID: u16 = 171;

/// java17: `Blocks.stone`, the substitute outside the world's horizontal limit.
const STONE: BlockState = BlockState::new(mc_data::BlockId(1), 0);

/// Every collision box overlapping `mask`, in vanilla 1.7.10's order: blocks column by column
/// (`x`, then `z`, then `y` from `floor(minY) - 1`), then the entity part.
///
/// java17: `World.getCollidingBoundingBoxes(entity, mask)`. `blockExists(x, 64, z)` is
/// `chunkExists`, a hard-coded `true` on a client (`ChunkProviderClient`), so every column is
/// scanned, including one the server never sent.
pub fn collision_boxes(
    w: &World,
    mask: &Aabb,
    entities: &[EntityCollisionCandidate],
    out: &mut Vec<Aabb>,
) {
    let x0 = floor_f64(mask.min.x);
    let x1 = floor_f64(mask.max.x + 1.0);
    let y0 = floor_f64(mask.min.y);
    let y1 = floor_f64(mask.max.y + 1.0);
    let z0 = floor_f64(mask.min.z);
    let z1 = floor_f64(mask.max.z + 1.0);

    for x in x0..x1 {
        for z in z0..z1 {
            let inside = (-WORLD_LIMIT..WORLD_LIMIT).contains(&x)
                && (-WORLD_LIMIT..WORLD_LIMIT).contains(&z);
            // One column lookup per (x, z); `World::block_at` answers an absent column.
            let chunk = if inside {
                w.chunk(BlockPos::new(x, 64, z).chunk_pos())
            } else {
                None
            };
            let (local_x, local_z) = ((x & 15) as u8, (z & 15) as u8);
            // java17: `for (int var11 = var5 - 1; var11 < var6; ++var11)`, int arithmetic.
            for y in y0.wrapping_sub(1)..y1 {
                let pos = BlockPos::new(x, y, z);
                let state = if !inside {
                    STONE
                } else {
                    match chunk {
                        Some(chunk) if (0..256).contains(&y) => {
                            chunk.block_at(local_x, y as u8, local_z)
                        }
                        Some(_) => BlockState::AIR,
                        None => w.block_at(pos),
                    }
                };
                if state != BlockState::AIR {
                    add_block_boxes(w, pos, state, mask, out);
                }
            }
        }
    }

    append_entity_collision_boxes(mask, entities, out);
}

/// One block's collision boxes in 1.7.10: the three shapes that differ, else 1.8.9's.
///
/// java17: `Block.addCollisionBoxesToList` (`block/Block.java:540-548`), which none of the
/// three classes override: their `getCollisionBoundingBoxFromPool` box is added when
/// `mask.intersectsWith(box)`.
fn add_block_boxes(w: &World, pos: BlockPos, state: BlockState, mask: &Aabb, out: &mut Vec<Aabb>) {
    match state.block().0 {
        CARPET_ID => push(out, mask, carpet_box(pos)),
        CAKE_ID => push(out, mask, cake_box(pos, state.meta())),
        FENCE_GATE_ID => {
            if let Some(a) = fence_gate_box(pos, state.meta()) {
                push(out, mask, a);
            }
        }
        _ => add_block_boxes_for(w, pos, state, mask, out),
    }
}

fn push(out: &mut Vec<Aabb>, mask: &Aabb, a: Aabb) {
    if mask.intersects_with(&a) {
        out.push(a);
    }
}

/// java17: `BlockCarpet.getCollisionBoundingBoxFromPool` (`block/BlockCarpet.java:33-38`):
///
/// ```text
/// (double)x + minX, (double)y + minY, (double)z + minZ,
/// (double)x + maxX, (double)((float)y + (float)0 * 0.0625F), (double)z + maxZ
/// ```
///
/// A box of height zero at the carpet's base: a player stands on the block below. The bounds
/// fields are the carpet's own `(0, 0, 0)..(1, 0.0625, 1)`, the only values its class ever writes
/// (`setBlockBoundsFromMeta`), so the box is `(x, y, z)..(x + 1, (float)y, z + 1)`.
#[must_use]
pub fn carpet_box(pos: BlockPos) -> Aabb {
    let (x, y, z) = (f64::from(pos.x), f64::from(pos.y), f64::from(pos.z));
    Aabb::from_bounds(x, y, z, x + 1.0, f64::from(pos.y as f32), z + 1.0)
}

/// java17: `BlockCake.getCollisionBoundingBoxFromPool` (`block/BlockCake.java:43-50`): the
/// 1.8.9 box with its top lowered by the inset, `(double)((float)y + 0.5F - 0.0625F)`.
#[must_use]
pub fn cake_box(pos: BlockPos, meta: u8) -> Aabb {
    let f = 0.0625f32;
    let f1 = (1 + i32::from(meta) * 2) as f32 / 16.0;
    let f2 = 0.5f32;
    Aabb::from_bounds(
        f64::from(pos.x as f32 + f1),
        f64::from(pos.y),
        f64::from(pos.z as f32 + f),
        f64::from(pos.x.wrapping_add(1) as f32 - f),
        f64::from(pos.y as f32 + f2 - f),
        f64::from(pos.z.wrapping_add(1) as f32 - f),
    )
}

/// java17: `BlockFenceGate.getCollisionBoundingBoxFromPool` (`block/BlockFenceGate.java:36-40`):
/// no box while open (`isFenceGateOpen`, `meta & 4`), otherwise the axis comes from the **raw**
/// metadata, `meta != 2 && meta != 0` meaning the thin-in-X box. 1.8.9 reads only `meta & 3`,
/// so the two differ at metadata 8 and 10.
#[must_use]
pub fn fence_gate_box(pos: BlockPos, meta: u8) -> Option<Aabb> {
    if meta & 4 != 0 {
        return None;
    }
    Some(if meta != 2 && meta != 0 {
        Aabb::from_bounds(
            f64::from(pos.x as f32 + 0.375),
            f64::from(pos.y),
            f64::from(pos.z),
            f64::from(pos.x as f32 + 0.625),
            f64::from(pos.y as f32 + 1.5),
            f64::from(pos.z.wrapping_add(1)),
        )
    } else {
        Aabb::from_bounds(
            f64::from(pos.x),
            f64::from(pos.y),
            f64::from(pos.z as f32 + 0.375),
            f64::from(pos.x.wrapping_add(1)),
            f64::from(pos.y as f32 + 1.5),
            f64::from(pos.z as f32 + 0.625),
        )
    })
}

/// java17: `World.isAnyLiquid` (`world/World.java:2010-2051`).
///
/// The scan is `floor(min) .. floor(max + 1.0)` exclusive, and on each axis whose minimum is
/// negative it starts **one block further** in the negative direction. 1.8.9 scans
/// `floor(min) ..= floor(max)` with no such extension, so beside a liquid at negative
/// coordinates the two disagree: the swim and lava ledge exit is refused there in 1.7.10.
#[must_use]
pub fn is_any_liquid(w: &World, bb: &Aabb) -> bool {
    let mut x0 = floor_f64(bb.min.x);
    let x1 = floor_f64(bb.max.x + 1.0);
    let mut y0 = floor_f64(bb.min.y);
    let y1 = floor_f64(bb.max.y + 1.0);
    let mut z0 = floor_f64(bb.min.z);
    let z1 = floor_f64(bb.max.z + 1.0);
    if bb.min.x < 0.0 {
        x0 = x0.wrapping_sub(1);
    }
    if bb.min.y < 0.0 {
        y0 = y0.wrapping_sub(1);
    }
    if bb.min.z < 0.0 {
        z0 = z0.wrapping_sub(1);
    }
    for x in x0..x1 {
        for y in y0..y1 {
            for z in z0..z1 {
                if mc_data::block_props(w.block_at(BlockPos::new(x, y, z)))
                    .material
                    .is_liquid()
                {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene;
    use mc_data::{block_by_name, BlockId};

    fn state(name: &str, meta: u8) -> BlockState {
        BlockState::new(block_by_name(name).expect(name), meta)
    }

    fn boxes_at(w: &World, pos: BlockPos) -> Vec<Aabb> {
        let mask = Aabb::from_bounds(
            f64::from(pos.x) - 0.5,
            f64::from(pos.y) - 0.5,
            f64::from(pos.z) - 0.5,
            f64::from(pos.x) + 1.5,
            f64::from(pos.y) + 1.5,
            f64::from(pos.z) + 1.5,
        );
        let mut out = Vec::new();
        collision_boxes(w, &mask, &[], &mut out);
        out.retain(|b| {
            b.min.x >= f64::from(pos.x)
                && b.max.x <= f64::from(pos.x + 1)
                && b.min.z >= f64::from(pos.z)
                && b.max.z <= f64::from(pos.z + 1)
                && b.min.y >= f64::from(pos.y)
                && b.max.y <= f64::from(pos.y + 2)
        });
        out
    }

    fn one(w: &World, pos: BlockPos) -> Vec<Aabb> {
        let mut out = Vec::new();
        crate::collision_boxes_in(
            w,
            &Aabb::from_bounds(
                f64::from(pos.x) - 0.5,
                f64::from(pos.y) - 0.5,
                f64::from(pos.z) - 0.5,
                f64::from(pos.x) + 1.5,
                f64::from(pos.y) + 1.5,
                f64::from(pos.z) + 1.5,
            ),
            &mut out,
        );
        out.retain(|b| {
            b.min.x >= f64::from(pos.x)
                && b.max.x <= f64::from(pos.x + 1)
                && b.min.z >= f64::from(pos.z)
                && b.max.z <= f64::from(pos.z + 1)
                && b.min.y >= f64::from(pos.y)
                && b.max.y <= f64::from(pos.y + 2)
        });
        out
    }

    #[test]
    fn the_three_differing_ids_are_the_registry_blocks() {
        assert_eq!(block_by_name("carpet"), Some(BlockId(CARPET_ID)));
        assert_eq!(block_by_name("cake"), Some(BlockId(CAKE_ID)));
        assert_eq!(block_by_name("fence_gate"), Some(BlockId(FENCE_GATE_ID)));
        assert_eq!(block_by_name("stone"), Some(STONE.block()));
    }

    /// Carpet: `[0,0,0,1,0,1]` at every metadata in 1.7.10, `[0,0,0,1,0.0625,1]` in 1.8.9.
    #[test]
    fn carpet_collides_at_its_base_with_no_height() {
        let mut w = World::new();
        let pos = BlockPos::new(3, 64, -5);
        for meta in [0, 7, 15] {
            scene::set_block(&mut w, pos, state("carpet", meta));
            assert_eq!(
                boxes_at(&w, pos),
                vec![Aabb::from_bounds(3.0, 64.0, -5.0, 4.0, 64.0, -4.0)],
                "meta {meta}"
            );
            // Negative control: the 1.8.9 query keeps the 1/16 box.
            assert_eq!(
                one(&w, pos),
                vec![Aabb::from_bounds(3.0, 64.0, -5.0, 4.0, 64.0625, -4.0)]
            );
        }
    }

    /// Cake: the top is `0.4375` for every bite count, the x/z insets are unchanged.
    #[test]
    fn cake_is_a_sixteenth_lower_than_in_1_8_9() {
        let mut w = World::new();
        let pos = BlockPos::new(1, 70, 1);
        for bites in 0..=6u8 {
            scene::set_block(&mut w, pos, state("cake", bites));
            let v5 = boxes_at(&w, pos);
            let v47 = one(&w, pos);
            assert_eq!(v5.len(), 1);
            assert_eq!(v5[0].max.y, 70.4375, "bites {bites}");
            assert_eq!(v47[0].max.y, 70.5, "bites {bites}: 1.8.9 control");
            assert_eq!(
                (v5[0].min, v5[0].max.x, v5[0].max.z),
                (v47[0].min, v47[0].max.x, v47[0].max.z)
            );
        }
    }

    /// Fence gates: identical to 1.8.9 at the metadata vanilla writes; metadata 8 and 10 (bit 3
    /// set, closed) take the thin-in-X box in 1.7.10 and the thin-in-Z box in 1.8.9.
    #[test]
    fn fence_gate_axis_follows_the_raw_metadata() {
        let mut w = World::new();
        let pos = BlockPos::new(0, 64, 0);
        for meta in 0..16u8 {
            scene::set_block(&mut w, pos, state("fence_gate", meta));
            let v5 = boxes_at(&w, pos);
            let v47 = one(&w, pos);
            if meta & 4 != 0 {
                assert!(v5.is_empty() && v47.is_empty(), "open gate {meta}");
                continue;
            }
            assert_eq!(v5.len(), 1);
            let thin_in_x = v5[0].max.x - v5[0].min.x < 0.5;
            assert_eq!(thin_in_x, meta != 0 && meta != 2, "meta {meta}");
            if meta == 8 || meta == 10 {
                assert_ne!(v5, v47, "meta {meta} differs from 1.8.9");
            } else {
                assert_eq!(v5, v47, "meta {meta} matches 1.8.9");
            }
        }
    }

    /// No world border: stone beyond the horizontal limit, at every height.
    #[test]
    fn beyond_thirty_million_is_stone_at_every_height() {
        let w = World::new();
        for y in [-5.0, 64.0, 300.0] {
            let mask = Aabb::from_bounds(29_999_999.5, y, 0.2, 30_000_000.5, y + 1.8, 0.8);
            let mut out = Vec::new();
            collision_boxes(&w, &mask, &[], &mut out);
            assert!(!out.is_empty(), "y {y}");
            assert!(
                out.iter().all(|b| b.min.x == 30_000_000.0),
                "y {y}: {out:?}"
            );
            // Negative control: 1.8.9 without a border finds nothing there.
            let mut v47 = Vec::new();
            crate::collision_boxes_in(&w, &mask, &mut v47);
            assert!(v47.is_empty());
        }
        let mask = Aabb::from_bounds(-30_000_000.5, 64.0, 0.2, -29_999_999.5, 65.8, 0.8);
        let mut out = Vec::new();
        collision_boxes(&w, &mask, &[], &mut out);
        assert!(out.iter().all(|b| b.max.x == -30_000_000.0) && !out.is_empty());
    }

    #[test]
    fn other_blocks_and_entities_match_the_1_8_9_query() {
        let mut w = World::new();
        scene::set_block(&mut w, BlockPos::new(0, 63, 0), state("stone", 0));
        scene::set_block(&mut w, BlockPos::new(1, 64, 0), state("stone_slab", 0));
        scene::set_block(&mut w, BlockPos::new(0, 64, 1), state("oak_stairs", 0));
        scene::set_block(&mut w, BlockPos::new(-1, 64, 0), state("fence", 0));
        let mask = Aabb::from_bounds(-1.5, 63.5, -1.5, 2.5, 66.0, 2.5);
        let boat = Aabb::from_bounds(0.0, 64.0, -1.0, 1.0, 64.6, 0.0);
        let entities = [EntityCollisionCandidate::new(boat, Some(boat))];
        let mut v5 = Vec::new();
        collision_boxes(&w, &mask, &entities, &mut v5);
        let mut v47 = Vec::new();
        crate::collision_boxes_in_with_entities(&w, &mask, &entities, &mut v47);
        assert_eq!(v5, v47);
        assert!(v5.contains(&boat));
    }

    /// The negative-side extension, with the 1.8.9 query as the negative control.
    #[test]
    fn negative_minimums_extend_the_liquid_scan_by_one_block() {
        let mut w = World::new();
        scene::set_block(&mut w, BlockPos::new(-3, 64, 0), state("water", 0));
        // A box from x = -1.7 covers blocks -2 and -1; 1.7.10 also scans -3.
        let bb = Aabb::from_bounds(-1.7, 64.2, 0.2, -1.1, 65.9, 0.8);
        assert!(is_any_liquid(&w, &bb));
        assert!(!crate::fluid::is_any_liquid(&w, &bb));
        // At positive coordinates the scans agree: block 3 is outside 2.1..2.7 for both and
        // inside 2.6..3.2 for both.
        let mut w = World::new();
        scene::set_block(&mut w, BlockPos::new(3, 64, 0), state("water", 0));
        let bb = Aabb::from_bounds(2.1, 64.2, 0.2, 2.7, 65.9, 0.8);
        assert!(!is_any_liquid(&w, &bb));
        assert!(!crate::fluid::is_any_liquid(&w, &bb));
        let bb = bb.offset(0.5, 0.0, 0.0);
        assert!(is_any_liquid(&w, &bb) && crate::fluid::is_any_liquid(&w, &bb));
    }
}
