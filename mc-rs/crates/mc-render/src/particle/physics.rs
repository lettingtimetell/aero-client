//! The particle collision sweep — `Entity.moveEntity` for a non-stepping entity.
//!
//! java: `entity/Entity.java` `moveEntity`. `EntityFX` never overrides it, and vanilla's generic
//! `Entity.stepHeight` is `0.0F` (only `EntityLivingBase` raises it, to `0.6`, for the player/mob
//! step-up retry that is S13's hazard H7) — so unlike player physics there is **no step-up**
//! here: a particle that would need to climb even a slab-height ledge just stops dead against it.
//! The axis order is still Y, then X, then Z, matching `Entity.moveEntity` (and H7).

use std::cell::RefCell;

use mc_math::{Aabb, DVec3};
use mc_world::World;

thread_local! {
    /// The collision-box list every [`sweep`] on this thread reuses (PART-4): a particle resting
    /// on a block used to allocate a fresh `Vec` on every tick. Cleared before each gather;
    /// `sweep` never re-enters itself, so the borrow cannot conflict.
    static SWEEP_BOXES: RefCell<Vec<Aabb>> = const { RefCell::new(Vec::new()) };
}

/// The result of one collision sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SweepResult {
    /// The bounding box after the swept, clipped move.
    pub bb: Aabb,
    /// The velocity to carry into the next tick: the axis actually travelled, or **zero** (not
    /// the clipped distance) on any axis where collision shortened the requested motion.
    /// java: `Entity.moveEntity` — `if (d0 != x) { this.motionX = 0.0D; }` (and Y, Z).
    pub motion: DVec3,
    /// java: `Entity.moveEntity` — `this.onGround = this.motionY != d1 && this.motionY < 0.0D;`
    /// (`d1` being the *applied*, pre-zeroing Y offset — a downward move that got clipped).
    pub on_ground: bool,
}

/// Sweep `bb` by `motion`, clipping against every block collision box the swept region touches.
///
/// `no_clip` particles (portal, enchantment glyph, flame) skip collision entirely — vanilla's
/// `noClip` early return in `Entity.moveEntity`, which just applies the raw motion.
#[must_use]
pub fn sweep(world: &World, bb: Aabb, motion: DVec3, no_clip: bool) -> SweepResult {
    if no_clip {
        return SweepResult {
            bb: bb.offset(motion.x, motion.y, motion.z),
            motion,
            on_ground: false,
        };
    }

    // java: `World.getCollidingBoundingBoxes(entity, entity.boundingBox.addCoord(x, y, z))` —
    // the swept region, then every block whose collision box could possibly be touched. A
    // 1-block margin covers boxes that extend beyond their own block's unit cube (open trapdoors,
    // fences) without needing per-block-type knowledge here.
    // `mc-physics` is the world-aware `getCollidingBoundingBoxes` implementation. Its query
    // recomputes connected fences/panes and stair corners from the live World (and handles
    // moving-piston tile state), whereas mc-world's extracted table is necessarily isolated.
    let swept = bb.add_coord(motion.x, motion.y, motion.z);
    SWEEP_BOXES.with(|boxes| {
        let mut boxes = boxes.borrow_mut();
        boxes.clear();
        mc_physics::collision_boxes_in(world, &swept, &mut boxes);
        clip(bb, motion, &boxes)
    })
}

/// The Y, then X, then Z clip of `bb` by `motion` against `boxes`, in list order.
fn clip(bb: Aabb, motion: DVec3, boxes: &[Aabb]) -> SweepResult {
    let mut off = motion;
    for b in boxes {
        off.y = b.calculate_y_offset(&bb, off.y);
    }
    let bb = bb.offset(0.0, off.y, 0.0);
    let y_applied = off.y;

    for b in boxes {
        off.x = b.calculate_x_offset(&bb, off.x);
    }
    let bb = bb.offset(off.x, 0.0, 0.0);

    for b in boxes {
        off.z = b.calculate_z_offset(&bb, off.z);
    }
    let bb = bb.offset(0.0, 0.0, off.z);

    let resting = DVec3::new(
        if off.x != motion.x { 0.0 } else { motion.x },
        if off.y != motion.y { 0.0 } else { motion.y },
        if off.z != motion.z { 0.0 } else { motion.z },
    );
    let on_ground = motion.y != y_applied && motion.y < 0.0;

    SweepResult {
        bb,
        motion: resting,
        on_ground,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_data::{block_by_name, BlockState};
    use mc_math::{BlockPos, ChunkPos};

    /// [`World::set_block`] is deliberately a no-op in an unloaded chunk (mirrors
    /// `ChunkProviderClient`'s blank-chunk fallback — see `mc_world::World::block_at`'s doc
    /// comment and §13's filed cross-section request). Load the chunk first, exactly as
    /// `mc-physics`'s `scene::ensure_chunk` does, so a hand-built test world's writes land.
    fn ensure_chunk(w: &mut World, pos: ChunkPos) {
        if w.is_chunk_loaded(pos) {
            return;
        }
        let len = mc_world::expected_chunk_data_len(1, w.has_sky_light(), true);
        w.apply_chunk_data(pos, true, 1, &vec![0u8; len])
            .expect("a zeroed payload of the exact expected length always decodes");
    }

    fn place(w: &mut World, p: BlockPos, state: BlockState) {
        ensure_chunk(w, p.chunk_pos());
        w.set_block(p, state);
    }

    fn stone_floor(world: &mut World, y: i32) {
        let stone = BlockState::new(block_by_name("stone").unwrap(), 0);
        for x in -2..=2 {
            for z in -2..=2 {
                place(world, BlockPos::new(x, y, z), stone);
            }
        }
    }

    #[test]
    fn free_fall_moves_unclipped_far_from_any_block() {
        let world = World::new();
        let bb = Aabb::from_bounds(-0.1, 10.0, -0.1, 0.1, 10.2, 0.1);
        let r = sweep(&world, bb, DVec3::new(0.0, -0.5, 0.0), false);
        assert!((r.bb.min.y - 9.5).abs() < 1e-9);
        assert_eq!(r.motion, DVec3::new(0.0, -0.5, 0.0));
        assert!(!r.on_ground);
    }

    #[test]
    fn landing_on_a_floor_zeroes_y_motion_and_sets_on_ground() {
        let mut world = World::new();
        stone_floor(&mut world, 5); // top surface at y = 6
        let bb = Aabb::from_bounds(-0.1, 6.5, -0.1, 0.1, 6.7, 0.1);
        let r = sweep(&world, bb, DVec3::new(0.0, -1.0, 0.0), false);
        assert!((r.bb.min.y - 6.0).abs() < 1e-9, "{:?}", r.bb);
        assert_eq!(r.motion.y, 0.0);
        assert!(r.on_ground);
    }

    #[test]
    fn a_wall_zeroes_horizontal_motion_but_not_vertical() {
        let mut world = World::new();
        let stone = BlockState::new(block_by_name("stone").unwrap(), 0);
        place(&mut world, BlockPos::new(1, 10, 0), stone);
        let bb = Aabb::from_bounds(0.4, 10.0, -0.1, 0.6, 10.2, 0.1);
        let r = sweep(&world, bb, DVec3::new(1.0, -0.1, 0.0), false);
        assert_eq!(
            r.motion.x, 0.0,
            "clipped X motion resets to zero, not the partial distance"
        );
        assert!(r.motion.y != 0.0, "unclipped Y motion survives");
        assert!(r.bb.max.x <= 1.0 + 1e-9);
    }

    #[test]
    fn no_clip_ignores_every_block() {
        let mut world = World::new();
        stone_floor(&mut world, 5);
        let bb = Aabb::from_bounds(-0.1, 6.5, -0.1, 0.1, 6.7, 0.1);
        let r = sweep(&world, bb, DVec3::new(0.0, -5.0, 0.0), true);
        assert!((r.bb.min.y - 1.5).abs() < 1e-9);
        assert!(!r.on_ground);
        assert_eq!(r.motion, DVec3::new(0.0, -5.0, 0.0));
    }

    #[test]
    fn empty_world_never_panics_at_extreme_y() {
        let world = World::new();
        let bb = Aabb::from_bounds(-0.1, -5.0, -0.1, 0.1, 0.1, 0.1);
        let r = sweep(&world, bb, DVec3::new(0.0, -1.0, 0.0), false);
        assert!(!r.on_ground);
        let _ = r.bb;
    }

    #[test]
    fn fence_particle_collision_reads_the_live_connection() {
        let mut world = World::new();
        let fence = BlockState::new(block_by_name("fence").unwrap(), 0);
        let origin = BlockPos::new(0, 10, 0);
        place(&mut world, origin, fence);
        place(&mut world, origin.east(), fence);

        // The east arm occupies x=0.375..1.0, z=0.375..0.625. The isolated table only has the
        // centre post (x=0.375..0.625), so this particle is the old false negative.
        let bb = Aabb::from_bounds(0.70, 10.5, -0.10, 0.80, 10.6, 0.10);
        let connected = sweep(&world, bb, DVec3::new(0.0, 0.0, 0.8), false);
        assert_eq!(
            connected.motion.z, 0.0,
            "the live fence arm clips the particle"
        );
        assert!(connected.bb.max.z <= 0.375 + 1e-9, "{connected:?}");

        // Mutation control: without the neighbour the arm disappears and the same particle
        // clears the isolated post. This fails if sweep regresses to table-only collision.
        place(
            &mut world,
            origin.east(),
            BlockState::new(block_by_name("air").unwrap(), 0),
        );
        let disconnected = sweep(&world, bb, DVec3::new(0.0, 0.0, 0.8), false);
        assert_eq!(disconnected.motion.z, 0.8, "{disconnected:?}");
    }

    #[test]
    fn pane_particle_collision_reads_the_live_connection() {
        let mut world = World::new();
        let pane = BlockState::new(block_by_name("glass_pane").unwrap(), 0);
        let origin = BlockPos::new(0, 10, 0);
        place(&mut world, origin, pane);
        place(&mut world, origin.north(), pane);

        // A north-only connection has no Z-axis arm behind the pane. An isolated pane has a
        // full cross, whose Z arm would incorrectly stop this particle moving along X at z=.75.
        let bb = Aabb::from_bounds(-0.10, 10.5, 0.70, 0.10, 10.6, 0.80);
        let connected = sweep(&world, bb, DVec3::new(0.8, 0.0, 0.0), false);
        assert_eq!(connected.motion.x, 0.8, "{connected:?}");

        place(
            &mut world,
            origin.north(),
            BlockState::new(block_by_name("air").unwrap(), 0),
        );
        let isolated = sweep(&world, bb, DVec3::new(0.8, 0.0, 0.0), false);
        assert_eq!(
            isolated.motion.x, 0.0,
            "the isolated cross has its Z arm: {isolated:?}"
        );
        assert!(isolated.bb.max.x <= 0.4375 + 1e-9, "{isolated:?}");
    }

    #[test]
    fn stair_particle_collision_reads_the_live_inner_corner() {
        let mut world = World::new();
        let stairs = block_by_name("stone_stairs").unwrap();
        let origin = BlockPos::new(0, 10, 0);
        // meta 0 is east-facing; the west south-facing neighbour (meta 2) fills the upper
        // x=0.0..0.5, z=0.5..1.0 inner-corner quadrant.
        place(&mut world, origin, BlockState::new(stairs, 0));
        place(&mut world, origin.west(), BlockState::new(stairs, 2));
        let bb = Aabb::from_bounds(-0.30, 10.70, 0.70, -0.10, 10.80, 0.80);
        let corner = sweep(&world, bb, DVec3::new(0.8, 0.0, 0.0), false);
        assert_eq!(corner.motion.x, 0.0, "the inner corner clips the particle");
        assert!(corner.bb.max.x <= 0.0 + 1e-9, "{corner:?}");

        place(
            &mut world,
            origin.west(),
            BlockState::new(block_by_name("air").unwrap(), 0),
        );
        let straight = sweep(&world, bb, DVec3::new(0.8, 0.0, 0.0), false);
        // The straight stair's ordinary upper step begins at x=.5, so it stops there rather
        // than at the corner's x=0 plane.
        assert_eq!(straight.motion.x, 0.0, "the base step still clips at x=.5");
        assert!(
            straight.bb.max.x >= 0.5 - 1e-9,
            "the straight stair leaves the inner quadrant open: {straight:?}"
        );
    }
}
