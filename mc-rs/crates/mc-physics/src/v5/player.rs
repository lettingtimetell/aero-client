//! One 1.7.10 tick of the local player.
//!
//! java17, in order: `EntityClientPlayerMP.onUpdate` -> `EntityPlayer.onUpdate` ->
//! `EntityLivingBase.onUpdate` -> `Entity.onEntityUpdate` -> `EntityPlayerSP.onLivingUpdate`
//! -> `EntityPlayer.onLivingUpdate` -> `EntityLivingBase.onLivingUpdate` ->
//! `EntityPlayer.moveEntityWithHeading` -> `EntityLivingBase.moveEntityWithHeading` ->
//! `Entity.moveEntity`.
//!
//! The body follows [`crate::player`]'s 1.8.9 tick statement for statement; the differences are:
//!
//! * no spectator mode and no `isCurrentViewEntity()` gates;
//! * a held sneak key raises `yOffset2` to `0.2F`;
//! * double-tap sprint has no previous-sneak guard, so releasing sneak while holding forward
//!   counts as a forward tap;
//! * creative flight moves vertically by a constant `0.15` and flies horizontally at
//!   `flySpeed` with no sprint doubling;
//! * no depth strider (enchantment id 8 does not exist in 1.7.10);
//! * the walk speed is the movement attribute clamped to `[0, Double.MAX_VALUE]`, not
//!   `[0, 1024]` (java17: `SharedMonsterAttributes.java:22`);
//! * the water and lava ledge exits and the unloaded-column pin read the eye `posY`, and the
//!   exit uses 1.7.10's `isAnyLiquid`;
//! * the camera pitch sway narrows `Math.atan` before multiplying by 15.

use mc_math::DVec3;
use mc_world::World;

use super::movement::{is_offset_position_in_liquid, move_entity};
use super::{PlayerV5, FLY_VERTICAL_SPEED};
use crate::collision::EntityCollisionCandidate;
use crate::fluid;
use crate::player::{
    column_is_loaded, ground_slipperiness, is_on_ladder, jump, move_flying, push_out_of_blocks_all,
    update_render_arm, DRAG_Y, GRAVITY, GROUND_ACCEL_NUMERATOR,
};
use crate::state::{MoveInput, PhysicsState};

/// java17: `float var2 = 0.8F;`, the sprint threshold (`client/entity/EntityPlayerSP.java:169`).
const SPRINT_FORWARD_THRESHOLD: f32 = 0.8;

/// java17: `this.jumpTicks = 10;` (`entity/EntityLivingBase.java:1752`).
const JUMP_COOLDOWN: i32 = 10;

/// Advances the 1.7.10 local player by one walking tick.
///
/// `entities` is the tracker's collision candidate set, as for the 1.8.9 tick. There is no world
/// border and no view gate in 1.7.10.
pub fn tick_player(
    w: &World,
    s: &mut PhysicsState,
    v5: &mut PlayerV5,
    input: &MoveInput,
    entities: &[EntityCollisionCandidate],
) {
    // The session forks on riding before choosing this walking body.
    s.riding = false;
    begin_tick(w, s, v5, input);

    // -- EntityPlayerSP.onLivingUpdate (`client/entity/EntityPlayerSP.java:90-289`) ---------
    let prev_jump = s.prev_jump;
    let was_forward_full = s.prev_move_forward >= SPRINT_FORWARD_THRESHOLD;
    let move_forward = input.move_forward();
    let move_strafe = input.move_strafe();
    refresh_input(s, input, move_forward, move_strafe);

    // java17 `:173-178`: `isUsingItem() && !isRiding()`; the 0.2 scale is in `MoveInput`.
    if input.using_item {
        s.sprint_toggle_timer = 0;
    }
    living_prologue(w, s, v5, input, prev_jump, was_forward_full, move_forward);

    // -- EntityLivingBase.onLivingUpdate (`entity/EntityLivingBase.java:1675-1780`) ---------
    zero_small_motion(s);
    let (move_strafe, move_forward, jumping) = if s.movement_blocked {
        s.move_strafing = 0.0;
        s.move_forward = 0.0;
        s.jumping = false;
        (0.0f32, 0.0f32, false)
    } else {
        (s.move_strafing, s.move_forward, s.jumping)
    };
    jump_section(w, s, jumping);

    s.move_strafing = move_strafe * 0.98;
    s.move_forward = move_forward * 0.98;
    move_entity_with_heading_player(w, s, v5, s.move_strafing, s.move_forward, entities);

    // -- EntityPlayer.onLivingUpdate's tail (`entity/player/EntityPlayer.java:497-530`) -----
    s.jump_movement_factor = s.recomputed_jump_movement_factor();
    update_camera_sway(s);

    // -- EntityPlayerSP.onLivingUpdate's tail (`:283-287`) -----------------------------------
    if s.on_ground && s.flying {
        s.flying = false;
    }
}

/// The first half of a mounted 1.7.10 player's update, before its own travel.
///
/// java17: `Entity.updateRidden` zeroes the motion and runs `onUpdate()`, so a rider still runs
/// `EntityPlayerSP.onLivingUpdate`'s prologue (sprint, the sneak offset, flight) with the riding
/// movement input (no item-use scale). The caller follows with [`mounted_living_travel`].
pub fn mounted_on_living_update_preflight(
    w: &World,
    s: &mut PhysicsState,
    v5: &mut PlayerV5,
    input: &MoveInput,
) {
    s.riding = true;
    // java17: `Entity.updateRidden` (`entity/Entity.java`) zeroes the motion before `onUpdate`.
    s.vel = DVec3::ZERO;
    begin_tick(w, s, v5, input);

    let prev_jump = s.prev_jump;
    let was_forward_full = s.prev_move_forward >= SPRINT_FORWARD_THRESHOLD;
    let move_forward = input.move_forward_riding(true);
    let move_strafe = input.move_strafe_riding(true);
    refresh_input(s, input, move_forward, move_strafe);
    living_prologue(w, s, v5, input, prev_jump, was_forward_full, move_forward);

    if s.movement_blocked {
        s.move_strafing = 0.0;
        s.move_forward = 0.0;
        s.jumping = false;
    }
}

/// The second half of a mounted 1.7.10 player's update: the jump and travel sections and the
/// tails, after [`mounted_on_living_update_preflight`].
///
/// As in 1.8.9 the rider's own move leaves its flags, motion and the `yOffset2` decay behind,
/// and its position is then discarded: `ridingEntity.updateRiderPosition()` places the rider,
/// which the caller does with [`PlayerV5::set_position`] once the vehicle has moved. Until then
/// the rider stays exactly where it started (box and eye).
pub fn mounted_living_travel(
    w: &World,
    s: &mut PhysicsState,
    v5: &mut PlayerV5,
    entities: &[EntityCollisionCandidate],
) {
    s.riding = true;
    let seat = s.location();
    let seat_eye = v5.pos_y;

    zero_small_motion(s);
    let jumping = s.jumping;
    jump_section(w, s, jumping);
    s.move_strafing *= 0.98;
    s.move_forward *= 0.98;
    move_entity_with_heading_player(w, s, v5, s.move_strafing, s.move_forward, entities);

    s.jump_movement_factor = s.recomputed_jump_movement_factor();
    update_camera_sway(s);
    if s.on_ground && s.flying {
        s.flying = false;
    }

    s.restore_location(seat);
    v5.pos_y = seat_eye;
}

/// The per-tick signal reset, the look, and `Entity.onEntityUpdate`
/// (`entity/Entity.java:285-413`): `prevPos*`, the water probe, the lava fall halving.
fn begin_tick(w: &World, s: &mut PhysicsState, v5: &mut PlayerV5, input: &MoveInput) {
    // java17: nothing sets `noClip` for a player and there is no spectator mode, so the shared
    // 1.8.9 helpers (the ladder test reads `spectator`) must not see one either.
    s.no_clip = false;
    s.spectator = false;
    s.spectator_mode = false;
    s.predicted_fall_damage = None;
    s.ground_contact = None;
    s.step_sound = None;
    s.swim_sound_volume = None;
    s.water_entry = None;
    s.touched_cactus = false;
    s.in_portal = false;
    // java17: `EntityPlayerSP.onLivingUpdate:162-165` counts `timeUntilPortal` down before the
    // move's `doBlockCollisions` can refresh it.
    if s.portal_cooldown > 0 {
        s.portal_cooldown -= 1;
    }

    s.yaw = input.yaw;
    s.pitch = input.pitch;
    s.sneaking = input.sneak;

    s.prev_distance_walked_modified = s.distance_walked_modified;
    s.prev_pos = s.pos;
    v5.snapshot_prev();
    fluid::handle_water_movement(w, s);
    if fluid::is_in_lava(w, s) {
        s.fall_distance *= 0.5;
    }
    s.first_update = false;

    // -- EntityPlayerSP.onLivingUpdate (`:92-104`) ------------------------------------------
    if s.sprinting_ticks_left > 0 {
        s.sprinting_ticks_left -= 1;
        if s.sprinting_ticks_left == 0 {
            s.set_sprinting(false);
        }
    }
    if s.sprint_toggle_timer > 0 {
        s.sprint_toggle_timer -= 1;
    }
}

/// `MovementInputFromOptions.updatePlayerMoveState` and `EntityPlayerSP
/// .updateEntityActionState` (`client/entity/EntityPlayerSP.java:78-88`), which in 1.7.10 has
/// no view gate.
fn refresh_input(s: &mut PhysicsState, input: &MoveInput, move_forward: f32, move_strafe: f32) {
    s.prev_move_forward = move_forward;
    s.prev_sneak = input.sneak;
    s.prev_jump = input.jump;
    s.move_strafing = move_strafe;
    s.move_forward = move_forward;
    s.jumping = input.jump;
    if !s.movement_blocked {
        update_render_arm(s);
    }
}

/// java17: `EntityPlayerSP.onLivingUpdate` from the sneak offset to the flight impulse
/// (`client/entity/EntityPlayerSP.java:180-239`), then `EntityPlayer.onLivingUpdate`'s and
/// `EntityLivingBase.onLivingUpdate`'s timer prologues.
fn living_prologue(
    w: &World,
    s: &mut PhysicsState,
    v5: &mut PlayerV5,
    input: &MoveInput,
    prev_jump: bool,
    was_forward_full: bool,
    move_forward: f32,
) {
    v5.raise_for_sneak(input.sneak);
    push_out_of_blocks_all(w, s);

    // java17 `:189`: `(float)foodLevel > 6.0F || capabilities.allowFlying`.
    let has_food = s.food_level as f32 > 6.0 || s.allow_flying;
    let may_start_sprinting = !input.using_item && !s.blindness;
    // java17 `:191`: no previous-sneak term.
    if s.on_ground
        && !was_forward_full
        && move_forward >= SPRINT_FORWARD_THRESHOLD
        && !s.sprinting
        && has_food
        && may_start_sprinting
    {
        if s.sprint_toggle_timer <= 0 && !input.sprint {
            s.sprint_toggle_timer = 7;
        } else {
            s.set_sprinting(true);
        }
    }
    if !s.sprinting
        && move_forward >= SPRINT_FORWARD_THRESHOLD
        && has_food
        && may_start_sprinting
        && input.sprint
    {
        s.set_sprinting(true);
    }
    if s.sprinting
        && (move_forward < SPRINT_FORWARD_THRESHOLD || s.collided_horizontally || !has_food)
    {
        s.set_sprinting(false);
    }

    // java17 `:213-225`: the double-tap-jump flight toggle.
    if s.allow_flying && !prev_jump && input.jump {
        if s.fly_toggle_timer == 0 {
            s.fly_toggle_timer = 7;
        } else {
            s.flying = !s.flying;
            s.fly_toggle_timer = 0;
        }
    }
    // java17 `:227-239`: a constant impulse, with no view gate.
    if s.flying {
        if input.sneak {
            s.vel.y -= FLY_VERTICAL_SPEED;
        }
        if input.jump {
            s.vel.y += FLY_VERTICAL_SPEED;
        }
    }

    // -- EntityPlayer.onLivingUpdate (`entity/player/EntityPlayer.java:481-485`) -------------
    if s.fly_toggle_timer > 0 {
        s.fly_toggle_timer -= 1;
    }
    // -- EntityLivingBase.onLivingUpdate (`entity/EntityLivingBase.java:1677-1680`) ----------
    if s.jump_ticks > 0 {
        s.jump_ticks -= 1;
    }
}

/// java17: `EntityLivingBase.onLivingUpdate`'s `Math.abs(motion) < 0.005D` zeroing
/// (`entity/EntityLivingBase.java:1701-1714`). `EntityPlayerSP.isServerWorld()` is true, so the
/// interpolation damping above it never runs.
fn zero_small_motion(s: &mut PhysicsState) {
    if s.vel.x.abs() < 0.005 {
        s.vel.x = 0.0;
    }
    if s.vel.y.abs() < 0.005 {
        s.vel.y = 0.0;
    }
    if s.vel.z.abs() < 0.005 {
        s.vel.z = 0.0;
    }
}

/// java17: the `jump` section of `EntityLivingBase.onLivingUpdate`
/// (`entity/EntityLivingBase.java:1745-1763`).
fn jump_section(w: &World, s: &mut PhysicsState, jumping: bool) {
    if jumping {
        if !s.in_water && !fluid::is_in_lava(w, s) {
            if s.on_ground && s.jump_ticks == 0 {
                jump(s);
                s.jump_ticks = JUMP_COOLDOWN;
            }
        } else {
            s.vel.y += 0.039_999_999_105_930_33;
        }
    } else {
        s.jump_ticks = 0;
    }
}

/// java17: `EntityPlayer.moveEntityWithHeading` (`entity/player/EntityPlayer.java:1488-1509`).
///
/// While flying (and not riding), `jumpMovementFactor` is `capabilities.getFlySpeed()` for the
/// call, with no sprint doubling, and `motionY` is replaced by the pre-move value times `0.6`.
fn move_entity_with_heading_player(
    w: &World,
    s: &mut PhysicsState,
    v5: &mut PlayerV5,
    strafe: f32,
    forward: f32,
    entities: &[EntityCollisionCandidate],
) {
    if s.flying && !s.riding {
        let d = s.vel.y;
        move_entity_with_heading(w, s, v5, strafe, forward, s.fly_speed, entities);
        s.vel.y = d * 0.6;
    } else {
        let factor = s.jump_movement_factor;
        move_entity_with_heading(w, s, v5, strafe, forward, factor, entities);
    }
}

/// java17: `EntityLivingBase.moveEntityWithHeading` (`entity/EntityLivingBase.java:1345-1487`).
/// `air_factor` is the `jumpMovementFactor` in effect for the call.
fn move_entity_with_heading(
    w: &World,
    s: &mut PhysicsState,
    v5: &mut PlayerV5,
    strafe: f32,
    forward: f32,
    air_factor: f32,
    entities: &[EntityCollisionCandidate],
) {
    if s.in_water && !s.flying {
        // ---- water (`:1349-1363`) ----------------------------------------------------------
        let var8 = v5.pos_y;
        move_flying(s, strafe, forward, 0.02);
        move_entity(w, s, v5, s.vel, entities);
        s.vel.x *= 0.800_000_011_920_929;
        s.vel.y *= 0.800_000_011_920_929;
        s.vel.z *= 0.800_000_011_920_929;
        s.vel.y -= 0.02;
        ledge_exit(w, s, v5, var8, entities);
    } else if !s.flying && fluid::is_in_lava(w, s) {
        // ---- lava (`:1364-1378`) -----------------------------------------------------------
        let var8 = v5.pos_y;
        move_flying(s, strafe, forward, 0.02);
        move_entity(w, s, v5, s.vel, entities);
        s.vel.x *= 0.5;
        s.vel.y *= 0.5;
        s.vel.z *= 0.5;
        s.vel.y -= 0.02;
        ledge_exit(w, s, v5, var8, entities);
    } else {
        // ---- air / ground (`:1379-1473`) ---------------------------------------------------
        let mut var3 = ground_slipperiness(w, s);
        let var4 = GROUND_ACCEL_NUMERATOR / (var3 * var3 * var3);
        let var5 = if s.on_ground {
            ai_move_speed(s) * var4
        } else {
            air_factor
        };
        move_flying(s, strafe, forward, var5);
        var3 = ground_slipperiness(w, s);

        if is_on_ladder(w, s) {
            let var6 = f64::from(0.15f32);
            s.vel.x = mc_math::clamp_f64(s.vel.x, -var6, var6);
            s.vel.z = mc_math::clamp_f64(s.vel.z, -var6, var6);
            s.fall_distance = 0.0;
            if s.vel.y < -0.15 {
                s.vel.y = -0.15;
            }
            if s.sneaking && s.vel.y < 0.0 {
                s.vel.y = 0.0;
            }
        }

        move_entity(w, s, v5, s.vel, entities);

        if s.collided_horizontally && is_on_ladder(w, s) {
            s.vel.y = 0.2;
        }
        // java17 `:1454-1468`: the unloaded-column pin compares the eye `posY` with zero.
        if column_is_loaded(w, s.pos) {
            s.vel.y -= GRAVITY;
        } else if v5.pos_y > 0.0 {
            s.vel.y = -0.1;
        } else {
            s.vel.y = 0.0;
        }
        s.vel.y *= DRAG_Y;
        s.vel.x *= f64::from(var3);
        s.vel.z *= f64::from(var3);
    }

    // java17 `:1475-1486`: the limb swing, outside every branch.
    s.prev_limb_swing_amount = s.limb_swing_amount;
    let d2 = s.pos.x - s.prev_pos.x;
    let d3 = s.pos.z - s.prev_pos.z;
    let mut f = mc_math::sqrt_d(d2 * d2 + d3 * d3) * 4.0;
    if f > 1.0 {
        f = 1.0;
    }
    s.limb_swing_amount += (f - s.limb_swing_amount) * 0.4;
    s.limb_swing += s.limb_swing_amount;
}

/// java17: `EntityPlayer.getAIMoveSpeed()` (`entity/player/EntityPlayer.java:1511-1514`), the
/// movement attribute narrowed to `float`. [`PhysicsState::ai_move_speed`] computes the same
/// value with 1.8.9's range; 1.7.10's `generic.movementSpeed` is
/// `RangedAttribute(.., 0.0D, Double.MAX_VALUE)` (`entity/SharedMonsterAttributes.java:22`) and
/// `RangedAttribute.clampValue` compares with `<` and `>`, so NaN passes through, as it does
/// through `f64::clamp`.
fn ai_move_speed(s: &PhysicsState) -> f32 {
    let mut v = s.movement_speed;
    if let Some(amount) = s.sprint_modifier_amount {
        v *= 1.0 + amount;
    }
    v.clamp(0.0, f64::MAX) as f32
}

/// java17: `if (isCollidedHorizontally && isOffsetPositionInLiquid(motionX, motionY +
/// 0.6000000238418579D - posY + var8, motionZ)) motionY = 0.30000001192092896D;` with the eye
/// `posY` before (`var8`) and after the move (`entity/EntityLivingBase.java:1359`, `:1374`).
fn ledge_exit(
    w: &World,
    s: &mut PhysicsState,
    v5: &PlayerV5,
    var8: f64,
    entities: &[EntityCollisionCandidate],
) {
    if s.collided_horizontally
        && is_offset_position_in_liquid(
            w,
            s,
            DVec3::new(
                s.vel.x,
                s.vel.y + 0.600_000_023_841_857_9 - v5.pos_y + var8,
                s.vel.z,
            ),
            entities,
        )
    {
        s.vel.y = 0.300_000_011_920_928_96;
    }
}

/// java17: `EntityPlayer.onLivingUpdate`'s camera sway (`entity/player/EntityPlayer.java:511-
/// 528`): `(float)Math.atan(-motionY * 0.20000000298023224D) * 15.0F`, a float product of the
/// narrowed arctangent (1.8.9 multiplies in double before narrowing).
fn update_camera_sway(s: &mut PhysicsState) {
    s.prev_camera_yaw = s.camera_yaw;
    s.prev_camera_pitch = s.camera_pitch;

    let mut yaw = mc_math::sqrt_d(s.vel.x * s.vel.x + s.vel.z * s.vel.z);
    let mut pitch = (-s.vel.y * 0.200_000_002_980_232_24_f64).atan() as f32 * 15.0;
    if yaw > 0.1 {
        yaw = 0.1;
    }
    if !s.on_ground || s.movement_blocked {
        yaw = 0.0;
    }
    if s.on_ground || s.movement_blocked {
        pitch = 0.0;
    }
    s.camera_yaw += (yaw - s.camera_yaw) * 0.4;
    s.camera_pitch += (pitch - s.camera_pitch) * 0.8;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene;
    use mc_data::{block_by_name, BlockState};
    use mc_math::BlockPos;

    fn state(name: &str, meta: u8) -> BlockState {
        BlockState::new(block_by_name(name).expect(name), meta)
    }

    fn standing(w: &World, feet: DVec3) -> (PhysicsState, PlayerV5) {
        let mut s = PhysicsState::new_player(feet);
        s.step_height = super::super::STEP_HEIGHT;
        let mut v5 = PlayerV5::at_feet(&s);
        // Settle onto the floor.
        for _ in 0..3 {
            tick_player(w, &mut s, &mut v5, &MoveInput::default(), &[]);
        }
        assert!(s.on_ground);
        (s, v5)
    }

    fn flat() -> World {
        scene::flat(state("stone", 0), 63, 16)
    }

    fn keys(forward: bool, sneak: bool) -> MoveInput {
        MoveInput {
            forward,
            sneak,
            ..MoveInput::default()
        }
    }

    /// The oracle's numbers (`focus_sneak_ysize`): eye 65.53999999910593 while sneaking, then
    /// `yOffset2` 0.032, 0.0128, ... after release.
    #[test]
    fn the_sneak_eye_drops_at_once_and_rises_over_several_ticks() {
        let w = flat();
        let (mut s, mut v5) = standing(&w, DVec3::new(0.5, 64.0, 0.5));
        assert_eq!(v5.pos_y(), 64.0 + f64::from(1.62_f32));
        tick_player(&w, &mut s, &mut v5, &keys(false, true), &[]);
        assert_eq!(v5.pos_y(), 65.539_999_999_105_93);
        tick_player(&w, &mut s, &mut v5, &keys(false, true), &[]);
        assert_eq!(v5.pos_y(), 65.539_999_999_105_93);
        tick_player(&w, &mut s, &mut v5, &MoveInput::default(), &[]);
        assert_eq!(f64::from(v5.y_size()), f64::from(0.2_f32 * 0.4 * 0.4));
        assert_eq!(
            v5.pos_y(),
            64.0 + f64::from(1.62_f32) - f64::from(0.2_f32 * 0.4 * 0.4)
        );
        tick_player(&w, &mut s, &mut v5, &MoveInput::default(), &[]);
        assert_eq!(v5.y_size(), 0.2_f32 * 0.4 * 0.4 * 0.4);
        assert_eq!(s.pos.y, 64.0, "the box never moves");
    }

    /// Releasing sneak while holding forward (0.3 -> 1.0) is a forward tap, with the 1.8.9 tick
    /// as the negative control.
    #[test]
    fn releasing_sneak_while_holding_forward_counts_as_a_tap() {
        let w = flat();
        let (mut s, mut v5) = standing(&w, DVec3::new(0.5, 64.0, 0.5));
        let mut t = s;
        for _ in 0..5 {
            tick_player(&w, &mut s, &mut v5, &keys(true, true), &[]);
            crate::tick_player(&w, &mut t, &keys(true, true));
        }
        tick_player(&w, &mut s, &mut v5, &keys(true, false), &[]);
        crate::tick_player(&w, &mut t, &keys(true, false));
        assert_eq!(s.sprint_toggle_timer, 7, "1.7.10 arms the double tap");
        assert_eq!(t.sprint_toggle_timer, 0, "1.8.9 ignores the release");
        // A real tap inside the window starts the sprint.
        tick_player(&w, &mut s, &mut v5, &keys(false, false), &[]);
        tick_player(&w, &mut s, &mut v5, &keys(true, false), &[]);
        assert!(s.sprinting);
    }

    /// A constant 0.15 vertical impulse at any fly speed, and no sprint doubling of the
    /// horizontal flight factor.
    #[test]
    fn flight_is_a_constant_impulse_without_sprint_doubling() {
        let w = flat();
        let fly = |fly_speed: f32, sprint: bool| {
            let mut s = PhysicsState::new_player(DVec3::new(0.5, 70.0, 0.5));
            let mut v5 = PlayerV5::at_feet(&s);
            s.allow_flying = true;
            s.flying = true;
            s.fly_speed = fly_speed;
            if sprint {
                s.set_sprinting(true);
            }
            let input = MoveInput {
                jump: true,
                forward: true,
                ..MoveInput::default()
            };
            tick_player(&w, &mut s, &mut v5, &input, &[]);
            s
        };
        let slow = fly(0.05, false);
        let fast = fly(0.1, false);
        // motionY = (0 + 0.15) * 0.6 after the move, the same at both fly speeds.
        assert_eq!(slow.vel.y, 0.15 * 0.6);
        assert_eq!(fast.vel.y, 0.15 * 0.6);
        assert_eq!(slow.pos.y, 70.0 + 0.15);
        // Horizontal: flySpeed, not 2 * flySpeed, while sprinting.
        let sprinting = fly(0.05, true);
        assert_eq!(sprinting.vel.z, slow.vel.z);

        // Negative control: the 1.8.9 tick doubles the factor while sprinting.
        let mut t = PhysicsState::new_player(DVec3::new(0.5, 70.0, 0.5));
        t.allow_flying = true;
        t.flying = true;
        t.set_sprinting(true);
        crate::tick_player(
            &w,
            &mut t,
            &MoveInput {
                forward: true,
                ..MoveInput::default()
            },
        );
        assert_ne!(t.vel.z, slow.vel.z);
    }

    /// Step height through the tick: walking into a bed stops, a slab is climbed.
    #[test]
    fn walking_climbs_a_slab_but_not_a_bed() {
        let mut w = flat();
        scene::set_block(&mut w, BlockPos::new(0, 64, 3), state("stone_slab", 0));
        scene::set_block(&mut w, BlockPos::new(2, 64, 3), state("bed", 0));
        for (x, top) in [(0.5, 64.5), (2.5, 64.0)] {
            let (mut s, mut v5) = standing(&w, DVec3::new(x, 64.0, 0.5));
            let mut highest = s.pos.y;
            for _ in 0..30 {
                tick_player(&w, &mut s, &mut v5, &keys(true, false), &[]);
                highest = highest.max(s.pos.y);
            }
            assert_eq!(highest, top, "x {x}");
        }
    }

    /// 1.7.10's movement attribute has no 1024 cap; 1.8.9's does.
    #[test]
    fn the_walk_speed_is_not_capped_at_1024() {
        let mut s = PhysicsState::new_player(DVec3::ZERO);
        s.movement_speed = 2000.0;
        assert_eq!(ai_move_speed(&s), 2000.0);
        assert_eq!(s.ai_move_speed(), 1024.0, "1.8.9 control");
        s.movement_speed = -1.0;
        assert_eq!(ai_move_speed(&s), 0.0);
        s.movement_speed = f64::NAN;
        assert!(ai_move_speed(&s).is_nan());
        s.movement_speed = 0.1;
        s.set_sprinting(true);
        assert_eq!(ai_move_speed(&s), s.ai_move_speed());
    }

    /// The ledge-exit test: next to water at negative x the exit is refused.
    #[test]
    fn the_negative_side_liquid_refuses_the_ledge_boost() {
        let w = flat();
        let (s, _) = standing(&w, DVec3::new(0.5, 64.0, 0.5));
        let mut w2 = flat();
        scene::set_block(&mut w2, BlockPos::new(-2, 65, 0), state("water", 0));
        let offset = DVec3::new(-0.6, 0.2, 0.0);
        // The offset box spans x -0.4..0.2: 1.7.10 scans from block -2, 1.8.9 from -1.
        assert!(is_offset_position_in_liquid(&w, &s, offset, &[]));
        assert!(!is_offset_position_in_liquid(&w2, &s, offset, &[]));
        assert!(crate::fluid::is_offset_position_in_liquid(&w2, &s, offset));
    }
}
