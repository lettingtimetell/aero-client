//! java17: `Entity.moveEntity` (`entity/Entity.java:457-806`), the 1.7.10 sweep.
//!
//! Differences from the 1.8.9 sweep in [`crate::movement`]:
//!
//! * `yOffset2 *= 0.4F` first, and `posY` re-derived from the box afterwards;
//! * the step-up is one candidate only, it steps back down by the **full** step height, and it
//!   is gated by `onGround && isSneaking() || yOffset2 < 0.05F`;
//! * motion is zeroed per axis with no `Block.onLanded` (there are no slime blocks);
//! * the walked-on block is looked up from the eye `posY`, falls back to a fence, wall or gate
//!   below even when the first block is solid, and its walking callback fires only with a
//!   footstep;
//! * collisions come from [`super::collision::collision_boxes`] (no world border; 1.7.10 shapes).

use mc_data::{block_info, block_props, BlockState, Material};
use mc_math::{floor_f64, Aabb, BlockPos, DVec3};
use mc_world::World;

use super::collision::{collision_boxes, is_any_liquid};
use super::{PlayerV5, STEP_GATE_Y_SIZE};
use crate::collision::EntityCollisionCandidate;
use crate::movement::{do_block_collisions, update_fall_state};
use crate::state::{GroundContact, PhysicsState, StepSound};

/// java17: `var21`, the step of `moveEntity`'s sneak edge loops.
const SNEAK_PROBE_STEP: f64 = 0.05;

/// java17: `0.20000000298023224D`, the walked-on block's depth below the feet.
const GROUND_PROBE: f64 = 0.200_000_002_980_232_24;

/// java17 class names of render types 11 (fence), 32 (wall) and 21 (fence gate), the blocks
/// `moveEntity` and `updateFallState` look through to (`entity/Entity.java:729-734`).
const FENCE_LIKE: [&str; 3] = [
    "net.minecraft.block.BlockFence",
    "net.minecraft.block.BlockWall",
    "net.minecraft.block.BlockFenceGate",
];

/// java17: `Entity.moveEntity(x, y, z)` for the local player. Returns the delta actually
/// applied (java17's mutated `x`/`y`/`z`).
pub(crate) fn move_entity(
    w: &World,
    s: &mut PhysicsState,
    v5: &mut PlayerV5,
    delta: DVec3,
    entities: &[EntityCollisionCandidate],
) -> DVec3 {
    // java17 `:459-465`: a `noClip` entity offsets its box and re-derives its position.
    if s.no_clip {
        let bb = s.aabb().offset(delta.x, delta.y, delta.z);
        s.set_from_aabb(&bb);
        v5.derive_pos_y(bb.min.y);
        return delta;
    }

    v5.decay();
    // java17 `:470-472`: `var7`/`var9`/`var11`, the position before the move. `var9` is the
    // eye `posY` as last written, before this move re-derives it.
    let var7 = s.pos.x;
    let var9 = v5.pos_y();
    let var11 = s.pos.z;

    let mut delta = delta;
    if s.in_web {
        s.in_web = false;
        delta.x *= 0.25;
        delta.y *= 0.050_000_000_745_058_06;
        delta.z *= 0.25;
        s.vel = DVec3::ZERO;
    }

    // java17 `:489`: `var20 = onGround && isSneaking() && this instanceof EntityPlayer`.
    let var20 = s.on_ground && s.sneaking;
    let mut bb = s.aabb();
    let r = sweep(
        w,
        &mut bb,
        delta,
        s.step_height,
        var20,
        s.on_ground,
        v5.y_size(),
        entities,
    );

    // java17 `:695-701`.
    s.set_from_aabb(&bb);
    v5.derive_pos_y(bb.min.y);
    s.collided_horizontally = r.collided_horizontally;
    s.collided_vertically = r.collided_vertically;
    s.on_ground = r.on_ground;

    // java17 `:702`: `updateFallState(y, onGround)`, `EntityLivingBase`'s override first.
    let (fall_pos, fall_block) = fall_block(w, s, v5);
    update_fall_state(w, s, r.applied.y, r.on_ground, fall_block, fall_pos);

    // java17 `:704-717`: plain zeroing, no `onLanded`.
    if r.requested.x != r.applied.x {
        s.vel.x = 0.0;
    }
    if r.requested.y != r.applied.y {
        s.vel.y = 0.0;
    }
    if r.requested.z != r.applied.z {
        s.vel.z = 0.0;
    }

    // java17 `:719-763`: `EntityPlayer.canTriggerWalking()` is `!capabilities.isFlying`.
    if !s.flying && !var20 && !s.riding {
        walk(
            w,
            s,
            v5,
            DVec3::new(s.pos.x - var7, v5.pos_y() - var9, s.pos.z - var11),
        );
    }

    do_block_collisions(w, s);
    r.applied
}

/// java17: the lookup at the top of `EntityLivingBase.updateFallState`
/// (`entity/EntityLivingBase.java:156-169`): the block `0.2` below the feet, computed from the
/// eye `posY`, or a fence, wall or gate below it when that block is air.
fn fall_block(w: &World, s: &PhysicsState, v5: &PlayerV5) -> (BlockPos, BlockState) {
    let pos = walked_on_pos(s, v5);
    let block = w.block_at(pos);
    if block_props(block).material == Material::Air {
        let below = w.block_at(pos.down());
        if is_fence_like(below) {
            return (pos, below);
        }
    }
    (pos, block)
}

/// java17: `floor(posX), floor(posY - 0.20000000298023224D - (double)yOffset), floor(posZ)`.
fn walked_on_pos(s: &PhysicsState, v5: &PlayerV5) -> BlockPos {
    BlockPos::new(
        floor_f64(s.pos.x),
        floor_f64(v5.pos_y() - GROUND_PROBE - f64::from(v5.y_offset())),
        floor_f64(s.pos.z),
    )
}

fn is_fence_like(state: BlockState) -> bool {
    FENCE_LIKE.contains(&block_info(block_props(state).block).java_class)
}

/// java17: the walking block of `Entity.moveEntity` (`entity/Entity.java:723-763`).
///
/// `moved` is `(posX - var7, posY - var9, posZ - var11)` with the eye `posY`, so its Y includes
/// this tick's `yOffset2` change; it only counts on a ladder.
fn walk(w: &World, s: &mut PhysicsState, v5: &PlayerV5, moved: DVec3) {
    let pos = walked_on_pos(s, v5);
    let mut block = w.block_at(pos);
    // java17 `:729-734`: unlike 1.8.9, the fallback does not require air above it.
    let below = w.block_at(pos.down());
    if is_fence_like(below) {
        block = below;
    }
    let dy = if block_info(block_props(block).block).name == "minecraft:ladder" {
        moved.y
    } else {
        0.0
    };

    s.distance_walked_modified = (f64::from(s.distance_walked_modified)
        + f64::from(mc_math::sqrt_d(moved.x * moved.x + moved.z * moved.z)) * 0.6)
        as f32;
    s.distance_walked_on_step_modified = (f64::from(s.distance_walked_on_step_modified)
        + f64::from(mc_math::sqrt_d(
            moved.x * moved.x + dy * dy + moved.z * moved.z,
        )) * 0.6) as f32;

    if s.distance_walked_on_step_modified > s.next_step_distance as f32
        && block_props(block).material != Material::Air
    {
        s.next_step_distance = s.distance_walked_on_step_modified as i32 + 1;

        if s.in_water {
            let mut f = mc_math::sqrt_d(
                s.vel.x * s.vel.x * 0.200_000_002_980_232_24
                    + s.vel.y * s.vel.y
                    + s.vel.z * s.vel.z * 0.200_000_002_980_232_24,
            ) * 0.35;
            if f > 1.0 {
                f = 1.0;
            }
            s.swim_sound_volume = Some(f);
        }

        // java17: `playStepSound(var39, var30, var31, var32)` (`entity/Entity.java:849-862`)
        // reads the snow layer above the *original* position, even when the sound block came
        // from the fence below it.
        let above = w.block_at(pos.up());
        if block_info(block_props(above).block).name == "minecraft:snow_layer" {
            s.step_sound = Some(StepSound { pos, state: above });
        } else if !block_props(block).material.is_liquid() {
            s.step_sound = Some(StepSound { pos, state: block });
        }

        // java17: `var32.onEntityWalking(...)`: redstone ore lights up
        // (`block/BlockRedstoneOre.java:41-45`), the app's standing-contact reaction.
        if matches!(block.block().0, 73 | 74) {
            s.ground_contact = Some(GroundContact { pos, state: block });
        }
    }
}

/// java17: `Entity.isOffsetPositionInLiquid` (`entity/Entity.java:450-455`): the offset box must
/// be free of collision boxes and contain no liquid per the 1.7.10 [`is_any_liquid`]. This is
/// the swim and lava ledge-exit test.
pub(crate) fn is_offset_position_in_liquid(
    w: &World,
    s: &PhysicsState,
    offset: DVec3,
    entities: &[EntityCollisionCandidate],
) -> bool {
    let bb = s.aabb().offset(offset.x, offset.y, offset.z);
    let mut boxes = Vec::new();
    collision_boxes(w, &bb, entities, &mut boxes);
    boxes.is_empty() && !is_any_liquid(w, &bb)
}

/// The requested and applied deltas, java17's `var13`/`var15`/`var17` and `x`/`y`/`z`.
struct Sweep {
    requested: DVec3,
    applied: DVec3,
    on_ground: bool,
    collided_horizontally: bool,
    collided_vertically: bool,
}

/// java17 `:485-691`: the sneak edge probes, the Y/X/Z sweep and the single-path step-up.
#[allow(clippy::too_many_arguments)]
fn sweep(
    w: &World,
    bb: &mut Aabb,
    delta: DVec3,
    step_height: f32,
    sneak_guard: bool,
    was_on_ground: bool,
    y_size: f32,
    entities: &[EntityCollisionCandidate],
) -> Sweep {
    let mut x = delta.x;
    let mut y = delta.y;
    let mut z = delta.z;
    let mut var13 = x;
    let var15 = y;
    let mut var17 = z;
    // java17 `:488`: `var19`, the box before the move.
    let var19 = *bb;
    let mut scratch: Vec<Aabb> = Vec::new();

    if sneak_guard {
        let d = SNEAK_PROBE_STEP;
        while x != 0.0 && {
            scratch.clear();
            collision_boxes(w, &bb.offset(x, -1.0, 0.0), entities, &mut scratch);
            scratch.is_empty()
        } {
            x = probe_step(x, d);
            var13 = x;
        }
        while z != 0.0 && {
            scratch.clear();
            collision_boxes(w, &bb.offset(0.0, -1.0, z), entities, &mut scratch);
            scratch.is_empty()
        } {
            z = probe_step(z, d);
            var17 = z;
        }
        while x != 0.0 && z != 0.0 && {
            scratch.clear();
            collision_boxes(w, &bb.offset(x, -1.0, z), entities, &mut scratch);
            scratch.is_empty()
        } {
            x = probe_step(x, d);
            z = probe_step(z, d);
            var13 = x;
            var17 = z;
        }
    }

    // java17 `:560`: one list from the swept box, used for all three axes.
    let mut list: Vec<Aabb> = Vec::new();
    collision_boxes(w, &bb.add_coord(x, y, z), entities, &mut list);
    for b in &list {
        y = b.calculate_y_offset(bb, y);
    }
    *bb = bb.offset(0.0, y, 0.0);
    // java17 `:576`: `var37 = onGround || var15 != y && var15 < 0.0D`.
    let var37 = was_on_ground || (var15 != y && var15 < 0.0);
    for b in &list {
        x = b.calculate_x_offset(bb, x);
    }
    *bb = bb.offset(x, 0.0, 0.0);
    for b in &list {
        z = b.calculate_z_offset(bb, z);
    }
    *bb = bb.offset(0.0, 0.0, z);

    // java17 `:612-691`. The `field_70135_K` tests in between are dead: the field is set
    // `true` in `Entity.<init>` (`:130`) and never written again.
    if step_height > 0.0
        && var37
        && (sneak_guard || y_size < STEP_GATE_Y_SIZE)
        && (var13 != x || var17 != z)
    {
        let var38 = x;
        let var25 = y;
        let var27 = z;
        x = var13;
        y = f64::from(step_height);
        z = var17;
        let var29 = *bb;
        *bb = var19;
        list.clear();
        collision_boxes(w, &bb.add_coord(var13, y, var17), entities, &mut list);
        for b in &list {
            y = b.calculate_y_offset(bb, y);
        }
        *bb = bb.offset(0.0, y, 0.0);
        for b in &list {
            x = b.calculate_x_offset(bb, x);
        }
        *bb = bb.offset(x, 0.0, 0.0);
        for b in &list {
            z = b.calculate_z_offset(bb, z);
        }
        *bb = bb.offset(0.0, 0.0, z);
        // java17 `:674`: down by the full step height, not by the height climbed.
        y = f64::from(-step_height);
        for b in &list {
            y = b.calculate_y_offset(bb, y);
        }
        *bb = bb.offset(0.0, y, 0.0);

        if var38 * var38 + var27 * var27 >= x * x + z * z {
            x = var38;
            y = var25;
            z = var27;
            *bb = var29;
        }
    }

    Sweep {
        requested: DVec3::new(var13, var15, var17),
        applied: DVec3::new(x, y, z),
        on_ground: var15 != y && var15 < 0.0,
        collided_horizontally: var13 != x || var17 != z,
        collided_vertically: var15 != y,
    }
}

/// One step of the sneak probe loops: zero once within a step of zero, else one step closer.
fn probe_step(v: f64, d: f64) -> f64 {
    if v < d && v >= -d {
        0.0
    } else if v > 0.0 {
        v - d
    } else {
        v + d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene;
    use mc_data::block_by_name;

    fn state(name: &str, meta: u8) -> BlockState {
        BlockState::new(block_by_name(name).expect(name), meta)
    }

    fn floor_world() -> World {
        scene::flat(state("stone", 0), 63, 8)
    }

    fn player(feet: DVec3) -> (PhysicsState, PlayerV5) {
        let mut s = PhysicsState::new_player(feet);
        s.step_height = super::super::STEP_HEIGHT;
        s.on_ground = true;
        let v5 = PlayerV5::at_feet(&s);
        (s, v5)
    }

    /// Walk into a step of `height` and report the box bottom after the move.
    fn step_onto(w: &World, step_sneaking: bool, y_size: f32) -> f64 {
        let (mut s, mut v5) = player(DVec3::new(0.5, 64.0, 0.5));
        s.sneaking = step_sneaking;
        v5.y_size = y_size;
        move_entity(w, &mut s, &mut v5, DVec3::new(0.0, -0.08, 0.3), &[]);
        s.pos.y
    }

    /// Step height 0.5: a slab (0.5) is a step, a bed (0.5625) is a wall.
    #[test]
    fn half_a_block_is_the_highest_step() {
        let mut w = floor_world();
        scene::set_block(&mut w, BlockPos::new(0, 64, 1), state("stone_slab", 0));
        assert_eq!(step_onto(&w, false, 0.0), 64.5);
        let mut w = floor_world();
        scene::set_block(&mut w, BlockPos::new(0, 64, 1), state("bed", 0));
        assert_eq!(step_onto(&w, false, 0.0), 64.0, "a bed is a wall in 1.7.10");
    }

    /// The single-path step: a 0.125 snow step whose edge is also the edge of a ceiling 2.0
    /// above the floor (the `focus_step_ceiling_snow` fixture). 1.7.10 climbs from the plain box
    /// by the full 0.5,
    /// runs into the ceiling's side and keeps the blocked move; 1.8.9's swept-box candidate
    /// climbs only 0.2 and passes under the ceiling onto the snow.
    #[test]
    fn a_ceiling_over_the_step_edge_blocks_the_single_path_step() {
        let mut w = floor_world();
        scene::set_block(&mut w, BlockPos::new(0, 64, 1), state("snow_layer", 1));
        scene::set_block(&mut w, BlockPos::new(0, 66, 1), state("stone", 0));
        let (mut s, mut v5) = player(DVec3::new(0.5, 64.0, 0.5));
        let applied = move_entity(&w, &mut s, &mut v5, DVec3::new(0.0, -0.08, 0.3), &[]);
        assert_eq!(s.pos.y, 64.0, "the step is reverted");
        assert_eq!(s.aabb().max.z, 1.0, "pressed against the snow");
        assert_eq!(applied.y, 0.0);
        assert!(s.collided_horizontally);
        assert!(s.on_ground);

        // Negative control: 1.8.9's two-path step climbs onto the snow under the ceiling.
        let mut t = PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5));
        t.on_ground = true;
        crate::move_entity_full(&w, &mut t, DVec3::new(0.0, -0.08, 0.3));
        assert_eq!(t.pos.y, 64.125);
    }

    /// While sneaking airborne (`onGround` false) the steady `yOffset2` 0.08 closes the
    /// step gate; on the ground the sneak flag opens it, and an unsneaking player's decayed
    /// offset opens it too.
    #[test]
    fn the_sneak_offset_gates_the_step() {
        let mut w = floor_world();
        scene::set_block(&mut w, BlockPos::new(0, 64, 1), state("stone_slab", 0));
        // On the ground and sneaking: var20 opens the gate (the edge guard keeps the floor).
        assert_eq!(step_onto(&w, true, 0.2), 64.5);
        // Not sneaking with a fresh 0.2 offset: 0.2 * 0.4 = 0.08 >= 0.05 closes it.
        assert_eq!(step_onto(&w, false, 0.2), 64.0);
        // 0.12 * 0.4 = 0.048 < 0.05 opens it again.
        assert_eq!(step_onto(&w, false, 0.12), 64.5);
    }

    #[test]
    fn a_move_decays_the_offset_and_rederives_the_eye() {
        let w = floor_world();
        let (mut s, mut v5) = player(DVec3::new(0.5, 64.0, 0.5));
        v5.raise_for_sneak(true);
        let before = v5.pos_y();
        move_entity(&w, &mut s, &mut v5, DVec3::new(0.0, -0.08, 0.0), &[]);
        assert_eq!(v5.y_size(), 0.2_f32 * 0.4);
        assert_eq!(
            v5.pos_y(),
            64.0 + f64::from(1.62_f32) - f64::from(0.2_f32 * 0.4)
        );
        // java17's posY within one sneak tick: 65.53999999910593, the oracle's number.
        assert_eq!(v5.pos_y(), 65.539_999_999_105_93);
        assert_ne!(before, v5.pos_y());
    }

    /// A fence below replaces a solid walked-on block too, and the footstep keeps the original
    /// position.
    #[test]
    fn the_fence_below_wins_even_under_a_solid_block() {
        let mut w = floor_world();
        scene::set_block(&mut w, BlockPos::new(0, 62, 0), state("fence", 0));
        let (mut s, mut v5) = player(DVec3::new(0.5, 64.0, 0.5));
        s.distance_walked_on_step_modified = 10.0;
        move_entity(&w, &mut s, &mut v5, DVec3::new(0.0, -0.08, 0.2), &[]);
        let sound = s.step_sound.expect("a footstep");
        assert_eq!(sound.pos, BlockPos::new(0, 63, 0));
        assert_eq!(sound.state, state("fence", 0));
    }
}
