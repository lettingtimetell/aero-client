//! One client tick of the local player.
//!
//! java, in order: `EntityPlayerSP.onUpdate` -> `EntityLivingBase.onUpdate` ->
//! `Entity.onEntityUpdate` -> `EntityPlayerSP.onLivingUpdate` -> `EntityPlayer.onLivingUpdate`
//! -> `EntityLivingBase.onLivingUpdate` -> `EntityLivingBase.moveEntityWithHeading` ->
//! `Entity.moveEntity`.
//!
//! Everything that cannot exist without a `Minecraft` instance — GUI, sound, network, item
//! use, horses, portals, flight toggling — is out; everything that moves the player is in.

use mc_data::{block_info, block_props, Material};
use mc_math::{cos, floor_f64, sin, sqrt_f, DVec3};
use mc_world::World;

use crate::collision::{CollisionBorder, EntityCollisionCandidate};
use crate::fluid;
use crate::movement::move_entity_full_with_context;
use crate::state::{MoveInput, PhysicsState};

/// Gravity per tick. java: `EntityLivingBase.moveEntityWithHeading:1677` — `motionY -= 0.08D`.
pub const GRAVITY: f64 = 0.08;

/// Vertical drag per tick. java: `motionY *= 0.9800000190734863D`.
///
/// There is no explicit terminal velocity in 1.8.9: `v' = (v - 0.08) * 0.98` settles at
/// `-0.08 * 0.98 / (1 - 0.98) = -3.92` blocks per tick, which is the number people quote.
pub const DRAG_Y: f64 = 0.980_000_019_073_486_3;

/// Base horizontal air/ground drag before slipperiness. java: `float f4 = 0.91F`.
pub const BASE_SLIPPERINESS: f32 = 0.91;

/// The magic ground-acceleration numerator. java: `float f = 0.16277136F / (f4 * f4 * f4);`
///
/// `0.16277136 == 0.91³`, so on default ground `f == 1.0` exactly and the acceleration is just
/// the move speed. On ice it is much smaller, which is why you accelerate slowly there.
pub const GROUND_ACCEL_NUMERATOR: f32 = 0.162_771_36;

/// Upward speed a jump imparts. java: `EntityLivingBase.getJumpUpwardsMotion` — `0.42F`.
pub const JUMP_MOTION: f32 = 0.42;

/// Horizontal boost a sprint jump adds along the look direction. java: `EntityLivingBase.jump`.
pub const SPRINT_JUMP_BOOST: f32 = 0.2;

/// Degrees-to-radians as vanilla spells it. java: `EntityLivingBase.jump` — `0.017453292F`.
pub const DEG_TO_RAD: f32 = 0.017_453_292;

/// `(float) Math.PI`, which is what `moveFlying` uses — not `std::f32::consts::PI`'s value by
/// accident, but by the same rounding.
const PI_F: f32 = std::f64::consts::PI as f32;

/// Ticks a jump locks out further jumps for. java: `EntityLivingBase.onLivingUpdate:2021`.
const JUMP_COOLDOWN: i32 = 10;

/// The `moveForward` threshold for sprinting. java: `EntityPlayerSP.onLivingUpdate` — `0.8F`.
const SPRINT_FORWARD_THRESHOLD: f32 = 0.8;

/// Advances the local player by one tick.
///
/// This is S13's frozen entry point. `input` is raw key state plus the look angles for this
/// tick; sprint state, the jump cooldown, the double-tap-forward window and the cobweb flag are
/// all carried in `s`.
pub fn tick_player(w: &World, s: &mut PhysicsState, input: &MoveInput) {
    tick_player_with_entities(w, s, input, &[]);
}

/// Advances the local player by one tick, adding nearby entity boxes to every movement sweep.
///
/// The caller owns entity tracking and supplies the exact candidate set from its current world.
/// `tick_player` remains the block-only compatibility entry point for worlds without that data.
pub fn tick_player_with_entities(
    w: &World,
    s: &mut PhysicsState,
    input: &MoveInput,
    entity_collisions: &[EntityCollisionCandidate],
) {
    tick_player_with_entities_view(w, s, input, entity_collisions, true);
}

/// Advances one local-player tick with nearby entity boxes, selecting whether the local player is
/// the current render-view entity.
///
/// java: `EntityPlayerSP.updateEntityActionState` copies `movementInput` into the persistent
/// `moveStrafing`/`moveForward`/`isJumping` fields only when `isCurrentViewEntity()` is true.
/// The ordinary [`tick_player_with_entities`] API remains view-local for compatibility; this
/// additive entry point is the seam used by a camera attached to another entity.
pub fn tick_player_with_entities_view(
    w: &World,
    s: &mut PhysicsState,
    input: &MoveInput,
    entity_collisions: &[EntityCollisionCandidate],
    view_is_local: bool,
) {
    let mut outside_border = false;
    tick_player_with_context(
        w,
        s,
        input,
        entity_collisions,
        None,
        &mut outside_border,
        view_is_local,
    );
}

/// Advances one local-player tick with a live world-border projection and persistent
/// `Entity.outsideBorder` latch.
///
/// The border is sampled by the client caller for this tick. The latch must be retained across
/// calls; vanilla changes it only at a loaded collision-query boundary and uses the one-block
/// hysteresis region to decide which side supplies the physical stone wall.
pub fn tick_player_with_entities_and_border(
    w: &World,
    s: &mut PhysicsState,
    input: &MoveInput,
    entity_collisions: &[EntityCollisionCandidate],
    border: CollisionBorder,
    outside_border: &mut bool,
) {
    tick_player_with_entities_and_border_view(
        w,
        s,
        input,
        entity_collisions,
        border,
        outside_border,
        true,
    );
}

/// Advances one local-player tick with a live world-border projection and a current-view gate.
///
/// This is the view-aware counterpart to [`tick_player_with_entities_and_border`]. It keeps the
/// physics/history tick alive for a foreign camera while leaving the entity action state at the
/// last value copied when the player was the active render view.
pub fn tick_player_with_entities_and_border_view(
    w: &World,
    s: &mut PhysicsState,
    input: &MoveInput,
    entity_collisions: &[EntityCollisionCandidate],
    border: CollisionBorder,
    outside_border: &mut bool,
    view_is_local: bool,
) {
    tick_player_with_context(
        w,
        s,
        input,
        entity_collisions,
        Some(border),
        outside_border,
        view_is_local,
    );
}

/// Shared player-tick body for the compatibility and border-aware entry points.
fn tick_player_with_context(
    w: &World,
    s: &mut PhysicsState,
    input: &MoveInput,
    entity_collisions: &[EntityCollisionCandidate],
    border: Option<CollisionBorder>,
    outside_border: &mut bool,
    view_is_local: bool,
) {
    // The session forks on `isRiding()` before it reaches this walking body, so this tick is by
    // definition one with `ridingEntity == null` (the mounted body sets the flag).
    s.riding = false;

    // -- EntityPlayer.onUpdate (`entity/player/EntityPlayer.java:268`) ------------------
    // A spectator is a ghost: the sweep is skipped entirely and `onGround` is forced false
    // *before* anything reads it, which is why a spectator never triggers step-up or friction.
    s.no_clip = s.spectator;
    if s.spectator {
        s.on_ground = false;
    }

    // Per-tick output signals. `inPortal` is vanilla's too: `BlockPortal` sets it during
    // `doBlockCollisions` and `EntityPlayerSP.onLivingUpdate:735` consumes and clears it.
    s.predicted_fall_damage = None;
    s.ground_contact = None;
    s.step_sound = None;
    s.swim_sound_volume = None;
    s.water_entry = None;
    s.touched_cactus = false;
    s.in_portal = false;
    if s.portal_cooldown > 0 {
        s.portal_cooldown -= 1;
    }

    s.yaw = input.yaw;
    s.pitch = input.pitch;
    s.sneaking = input.sneak;

    // -- Entity.onEntityUpdate (`entity/Entity.java:420`) ------------------------------
    s.prev_distance_walked_modified = s.distance_walked_modified;
    s.prev_pos = s.pos;
    fluid::handle_water_movement(w, s);
    if fluid::is_in_lava(w, s) {
        // java: Entity.onEntityUpdate:511 — `setOnFireFromLava()` is damage, this is physics.
        s.fall_distance *= 0.5;
    }
    // java: `Entity.onEntityUpdate:524` clears this as its last statement, after the water probe
    // above and before `onLivingUpdate`. The initial probe does not splash, while a later
    // air-to-water edge does — including one during this very tick's `updateFallState` (H2:
    // this used to be cleared at the end of the tick, which silenced a first-tick splash).
    s.first_update = false;

    // -- EntityPlayerSP.onLivingUpdate (prologue) --------------------------------------
    if s.sprinting_ticks_left > 0 {
        s.sprinting_ticks_left -= 1;
        if s.sprinting_ticks_left == 0 {
            set_sprinting(s, false);
        }
    }
    if s.sprint_toggle_timer > 0 {
        s.sprint_toggle_timer -= 1;
    }

    // java: `flag`/`flag1`/`flag2` are read from `movementInput` *before*
    // `updatePlayerMoveState()` overwrites it, so all three are the previous tick's.
    let prev_jump = s.prev_jump;
    let prev_sneak = s.prev_sneak;
    let was_forward_full = s.prev_move_forward >= SPRINT_FORWARD_THRESHOLD;
    let move_forward = input.move_forward();
    let move_strafe = input.move_strafe();
    s.prev_move_forward = move_forward;
    s.prev_sneak = input.sneak;
    s.prev_jump = input.jump;
    // java: `EntityPlayerSP.updateEntityActionState` (`:690-704`) only copies the refreshed
    // movement input while this player is `isCurrentViewEntity()`. A foreign camera still runs
    // the rest of `onLivingUpdate`, but travel consumes these persistent entity fields instead
    // of the newly refreshed keys.
    if view_is_local {
        s.move_strafing = move_strafe;
        s.move_forward = move_forward;
        s.jumping = input.jump;
    }
    if view_is_local && !s.movement_blocked {
        update_render_arm(s);
    }

    // java: `EntityPlayerSP.onLivingUpdate:788` — the `×0.2F` half lives in
    // `MoveInput::move_forward`/`move_strafe`; this is the other statement in the same block.
    if input.using_item {
        s.sprint_toggle_timer = 0;
    }

    push_out_of_blocks_all(w, s);

    // java: `boolean flag3 = (float)this.getFoodStats().getFoodLevel() > 6.0F ||
    // this.capabilities.allowFlying;`
    let has_food = s.food_level as f32 > 6.0 || s.allow_flying;
    // java: both sprint-*start* conditions carry `!isUsingItem() && !isPotionActive(blindness)`;
    // the sprint-*stop* condition (below) carries neither.
    let may_start_sprinting = !input.using_item && !s.blindness;
    if s.on_ground
        && !prev_sneak
        && !was_forward_full
        && move_forward >= SPRINT_FORWARD_THRESHOLD
        && !s.sprinting
        && has_food
        && may_start_sprinting
    {
        if s.sprint_toggle_timer <= 0 && !input.sprint {
            s.sprint_toggle_timer = 7;
        } else {
            set_sprinting(s, true);
        }
    }
    if !s.sprinting
        && move_forward >= SPRINT_FORWARD_THRESHOLD
        && has_food
        && may_start_sprinting
        && input.sprint
    {
        set_sprinting(s, true);
    }
    if s.sprinting
        && (move_forward < SPRINT_FORWARD_THRESHOLD || s.collided_horizontally || !has_food)
    {
        set_sprinting(s, false);
    }

    // java: `EntityPlayerSP.onLivingUpdate:823` — the double-tap-jump flight toggle. Spectator
    // mode forces flight on instead of toggling it. That branch asks the *controller*
    // (`mc.playerController.isSpectatorMode()`), not the tab-list `isSpectator()`.
    if s.allow_flying {
        if s.spectator_mode {
            s.flying = true;
        } else if !prev_jump && input.jump {
            if s.fly_toggle_timer == 0 {
                s.fly_toggle_timer = 7;
            } else {
                s.flying = !s.flying;
                s.fly_toggle_timer = 0;
            }
        }
    }

    // java: `EntityPlayerSP.onLivingUpdate:848` — `isCurrentViewEntity()` is true for the local
    // player whenever the camera is not attached to something else.
    // java: `EntityPlayerSP.onLivingUpdate:848` is independently guarded by
    // `isCurrentViewEntity()`, even though the capability toggle above is not.
    if s.flying && view_is_local {
        if input.sneak {
            s.vel.y -= f64::from(s.fly_speed * 3.0);
        }
        if input.jump {
            s.vel.y += f64::from(s.fly_speed * 3.0);
        }
    }

    // -- EntityPlayer.onLivingUpdate (prologue, `entity/player/EntityPlayer.java:599`) --
    if s.fly_toggle_timer > 0 {
        s.fly_toggle_timer -= 1;
    }

    // -- EntityLivingBase.onLivingUpdate -----------------------------------------------
    if s.jump_ticks > 0 {
        s.jump_ticks -= 1;
    }
    // java: `EntityPlayerSP.isServerWorld()` is true, so the `motion *= 0.98` interpolation
    // damping never runs for the local player.
    if s.vel.x.abs() < 0.005 {
        s.vel.x = 0.0;
    }
    if s.vel.y.abs() < 0.005 {
        s.vel.y = 0.0;
    }
    if s.vel.z.abs() < 0.005 {
        s.vel.z = 0.0;
    }

    // java: `EntityLivingBase.onLivingUpdate:1991` — `isMovementBlocked()` zeroes the *input*
    // and the jump flag. Gravity, drag and the sweep all still run, so a corpse still falls.
    let (move_strafe, move_forward, jumping) = if s.movement_blocked {
        // java: the blocked branch writes the persistent EntityLivingBase action fields too;
        // the next foreign-camera tick must therefore retain zero rather than a pre-death key.
        s.move_strafing = 0.0;
        s.move_forward = 0.0;
        s.jumping = false;
        (0.0f32, 0.0f32, false)
    } else {
        (s.move_strafing, s.move_forward, s.jumping)
    };

    let jump_ticks_before = s.jump_ticks;
    if jumping {
        if s.in_water {
            // java: EntityLivingBase.updateAITick
            s.vel.y += 0.039_999_999_105_930_33;
        } else if fluid::is_in_lava(w, s) {
            // java: EntityLivingBase.handleJumpLava
            s.vel.y += 0.039_999_999_105_930_33;
        } else if s.on_ground && jump_ticks_before == 0 {
            jump(s);
            s.jump_ticks = JUMP_COOLDOWN;
        }
    } else {
        s.jump_ticks = 0;
    }

    // java: `this.moveStrafing *= 0.98F; this.moveForward *= 0.98F;` (`:2031-2032`) — an
    // in-place write to the persistent `EntityLivingBase` fields, then
    // `moveEntityWithHeading(this.moveStrafing, this.moveForward)`. A local-view
    // tick refreshes the fields from the input next tick, so only two readers see the damped
    // value: a foreign-camera tick (which skips that refresh and so decays them by 0.98 per
    // tick) and a boat/pig, which reads its rider's fields from the previous update.
    s.move_strafing = move_strafe * 0.98;
    s.move_forward = move_forward * 0.98;
    move_entity_with_heading_player(
        w,
        s,
        s.move_strafing,
        s.move_forward,
        entity_collisions,
        border,
        outside_border,
    );

    // -- EntityPlayer.onLivingUpdate's tail (`entity/player/EntityPlayer.java:627`) -----
    // This is *after* `super.onLivingUpdate()` has already run the move, which is exactly why
    // `jump_movement_factor` is a stored field: the move above read the previous tick's value.
    s.jump_movement_factor = s.recomputed_jump_movement_factor();

    // java: `EntityLivingBase.onLivingUpdate`'s camera-sway tail. The player camera consumes
    // these damped current/previous samples at render time; raw horizontal speed would make
    // every step snap instead of easing in and out.
    update_camera_sway(s);

    // -- EntityPlayerSP.onLivingUpdate's tail (`:904`) ---------------------------------
    // java: `!this.mc.playerController.isSpectatorMode()` — the controller again.
    if s.on_ground && s.flying && !s.spectator_mode {
        s.flying = false;
    }
}

/// Run the part of `EntityPlayerSP.onLivingUpdate` that still executes before a mounted player's
/// position is written by `Entity.updateRidden`.
///
/// java: `EntityPlayerSP.onLivingUpdate:715-855`. A mounted player still decrements the sprint
/// and flight windows, refreshes movement input, pushes out of blocks, evaluates sprint and
/// flight mutations, and applies view-local flight thrust. The session's riding branch owns the
/// C05+C0C wire pair and the flight C13s around it.
///
/// This is the first half of the mounted living update. The caller must follow it with
/// [`mounted_living_travel`], the jump and travel sections `EntityLivingBase.onLivingUpdate`
/// still runs for a rider (H2: the rider's own move sets `isCollidedHorizontally`, which the next
/// tick's sprint stop reads, and the jump section resets `jumpTicks`).
///
/// `view_is_local` is the same `isCurrentViewEntity()` gate used by
/// `EntityPlayerSP.updateEntityActionState`: foreign cameras refresh the raw input edges but
/// leave the persistent entity action fields untouched.
///
/// The caller feeds the watched sprint flag in first, exactly as before a walking tick.
pub fn mounted_on_living_update_preflight(
    w: &World,
    s: &mut PhysicsState,
    input: &MoveInput,
    view_is_local: bool,
) {
    // The session calls this body only while `ridingEntity != null`.
    s.riding = true;
    // java: `Entity.updateRidden` (`:1886-1888`) — `motionX = motionY = motionZ = 0.0D`, before
    // `onUpdate` and therefore before anything can integrate them.
    s.vel = DVec3::ZERO;

    // -- EntityPlayer.onUpdate (`entity/player/EntityPlayer.java:268`) -----------------
    s.no_clip = s.spectator;
    if s.spectator {
        s.on_ground = false;
    }

    // Per-tick output signals, cleared as at the top of a walking tick. `inPortal` is one of
    // them: `BlockPortal` never sets it for a rider, so a flag a prior walking tick left behind is
    // consumed once and not again.
    s.predicted_fall_damage = None;
    s.ground_contact = None;
    s.step_sound = None;
    s.swim_sound_volume = None;
    s.water_entry = None;
    s.touched_cactus = false;
    s.in_portal = false;
    if s.portal_cooldown > 0 {
        s.portal_cooldown -= 1;
    }

    s.yaw = input.yaw;
    s.pitch = input.pitch;
    s.sneaking = input.sneak;

    // -- Entity.onEntityUpdate (`entity/Entity.java:420`) ------------------------------
    s.prev_distance_walked_modified = s.distance_walked_modified;
    s.prev_pos = s.pos;
    fluid::handle_water_movement(w, s);
    if fluid::is_in_lava(w, s) {
        s.fall_distance *= 0.5;
    }
    s.first_update = false;

    // -- EntityPlayerSP.onLivingUpdate (prologue) --------------------------------------
    if s.sprinting_ticks_left > 0 {
        s.sprinting_ticks_left -= 1;
        if s.sprinting_ticks_left == 0 {
            set_sprinting(s, false);
        }
    }
    if s.sprint_toggle_timer > 0 {
        s.sprint_toggle_timer -= 1;
    }

    // `flag`/`flag1`/`flag2` are read before updatePlayerMoveState. The effective movement values
    // here are riding values because EntityPlayerSP skips the item-use 0.2 scale while mounted.
    let prev_jump = s.prev_jump;
    let prev_sneak = s.prev_sneak;
    let was_forward_full = s.prev_move_forward >= SPRINT_FORWARD_THRESHOLD;
    let move_forward = input.move_forward_riding(true);
    let move_strafe = input.move_strafe_riding(true);
    s.prev_move_forward = move_forward;
    s.prev_sneak = input.sneak;
    s.prev_jump = input.jump;

    // java: `updateEntityActionState` is the only writer of these EntityLivingBase fields for a
    // client player. The C0C packet below reads them, so a foreign camera must not substitute the
    // current raw keys here.
    if view_is_local {
        s.move_strafing = move_strafe;
        s.move_forward = move_forward;
        s.jumping = input.jump;
    }
    if view_is_local && !s.movement_blocked {
        update_render_arm(s);
    }

    // `isUsingItem() && !isRiding()` is false on this branch, so no sprint-toggle reset here.
    push_out_of_blocks_all(w, s);

    let has_food = s.food_level as f32 > 6.0 || s.allow_flying;
    let may_start_sprinting = !input.using_item && !s.blindness;
    if s.on_ground
        && !prev_sneak
        && !was_forward_full
        && move_forward >= SPRINT_FORWARD_THRESHOLD
        && !s.sprinting
        && has_food
        && may_start_sprinting
    {
        if s.sprint_toggle_timer <= 0 && !input.sprint {
            s.sprint_toggle_timer = 7;
        } else {
            set_sprinting(s, true);
        }
    }
    if !s.sprinting
        && move_forward >= SPRINT_FORWARD_THRESHOLD
        && has_food
        && may_start_sprinting
        && input.sprint
    {
        set_sprinting(s, true);
    }
    if s.sprinting
        && (move_forward < SPRINT_FORWARD_THRESHOLD || s.collided_horizontally || !has_food)
    {
        set_sprinting(s, false);
    }

    // java: the capability mutation is not current-view gated; only the vertical motion below
    // is. Session compares the before/after flag and emits each C13 mutation before C05+C0C.
    // The forced-flight branch asks `playerController.isSpectatorMode()`, as on foot.
    if s.allow_flying {
        if s.spectator_mode {
            s.flying = true;
        } else if !prev_jump && input.jump {
            if s.fly_toggle_timer == 0 {
                s.fly_toggle_timer = 7;
            } else {
                s.flying = !s.flying;
                s.fly_toggle_timer = 0;
            }
        }
    }
    if s.flying && view_is_local {
        if input.sneak {
            s.vel.y -= f64::from(s.fly_speed * 3.0);
        }
        if input.jump {
            s.vel.y += f64::from(s.fly_speed * 3.0);
        }
    }

    // -- EntityPlayer.onLivingUpdate / EntityLivingBase.onLivingUpdate prologue --------
    if s.fly_toggle_timer > 0 {
        s.fly_toggle_timer -= 1;
    }
    if s.jump_ticks > 0 {
        s.jump_ticks -= 1;
    }
    if s.movement_blocked {
        // Java's blocked branch runs after the SP flight preflight and zeros the entity action
        // fields before the C0C packet is built.
        s.move_strafing = 0.0;
        s.move_forward = 0.0;
        s.jumping = false;
    }
}

/// The second half of a mounted player's living update: the jump and travel sections of
/// `EntityLivingBase.onLivingUpdate` and `EntityPlayer.onLivingUpdate`'s tail, run after
/// [`mounted_on_living_update_preflight`].
///
/// java: `Entity.updateRidden` (`:1880-1891`) zeroes the motion and calls the full `onUpdate()`,
/// so a rider still travels (`EntityLivingBase.onLivingUpdate:1972-2034`):
///
/// * the `0.005` motion zeroing (only the view-local flight thrust can be non-zero by then);
/// * the jump section: `jumpTicks = 0` whenever `isJumping` is false, the water/lava `+0.04`,
///   and a real `jump()` when the rider is still on the ground from its last walking tick;
/// * `moveStrafing/moveForward *= 0.98F` and `moveEntityWithHeading`. `EntityPlayer`'s flight
///   wrapper and `Entity.moveEntity`'s walking branch are both `ridingEntity == null`-gated, so
///   this is the plain living travel with no footsteps; `BlockPortal` skips riders too.
///
/// That move is what leaves `isCollidedHorizontally`, `isCollidedVertically`, `onGround`, the
/// post-travel motion (the camera pitch) and the limb swing behind. Then `EntityPlayer`'s tail
/// recomputes `jumpMovementFactor` from the sprint flag and damps the camera sway, and
/// `EntityPlayerSP`'s tail clears creative flight on landing.
///
/// The position the move produced is discarded: `Entity.updateRidden` immediately calls
/// `ridingEntity.updateRiderPosition()`, and the session writes that seat after the vehicle's
/// tick. Until then the rider stays exactly where it started (position and box).
pub fn mounted_living_travel(
    w: &World,
    s: &mut PhysicsState,
    entity_collisions: &[EntityCollisionCandidate],
    border: Option<CollisionBorder>,
    outside_border: &mut bool,
) {
    s.riding = true;
    let seat = s.location();

    // -- EntityLivingBase.onLivingUpdate (`:1972-1985`) --------------------------------
    // `isServerWorld()` is true for EntityPlayerSP, so there is no `motion *= 0.98` branch.
    if s.vel.x.abs() < 0.005 {
        s.vel.x = 0.0;
    }
    if s.vel.y.abs() < 0.005 {
        s.vel.y = 0.0;
    }
    if s.vel.z.abs() < 0.005 {
        s.vel.z = 0.0;
    }

    // java: `:2005-2026`. The preflight already applied the `isMovementBlocked()` zeroing to
    // the persistent fields, so they are exactly what this section reads.
    if s.jumping {
        if s.in_water {
            // java: EntityLivingBase.updateAITick
            s.vel.y += 0.039_999_999_105_930_33;
        } else if fluid::is_in_lava(w, s) {
            // java: EntityLivingBase.handleJumpLava
            s.vel.y += 0.039_999_999_105_930_33;
        } else if s.on_ground && s.jump_ticks == 0 {
            jump(s);
            s.jump_ticks = JUMP_COOLDOWN;
        }
    } else {
        s.jump_ticks = 0;
    }

    // java: `:2031-2034` — the in-place damping (which the C0C then reads) and the travel.
    s.move_strafing *= 0.98;
    s.move_forward *= 0.98;
    move_entity_with_heading_player(
        w,
        s,
        s.move_strafing,
        s.move_forward,
        entity_collisions,
        border,
        outside_border,
    );

    // -- EntityPlayer.onLivingUpdate's tail (`:627-650`) --------------------------------
    s.jump_movement_factor = s.recomputed_jump_movement_factor();
    // java: the camera sway reads the post-travel motion and `onGround`. `EntityPlayer
    // .updateRidden()` then snapshots and clears only the horizontal half (the session does).
    update_camera_sway(s);

    // -- EntityPlayerSP.onLivingUpdate's tail (`:904`) ---------------------------------
    // The landing clear reads the `onGround` the rider's own move produced; the caller compares
    // `flying` around this call to send its C13 after any horse-jump C0B.
    if s.on_ground && s.flying && !s.spectator_mode {
        s.flying = false;
    }

    // java: `Entity.updateRidden` -> `ridingEntity.updateRiderPosition()`.
    s.restore_location(seat);
}

fn update_camera_sway(s: &mut PhysicsState) {
    s.prev_camera_yaw = s.camera_yaw;
    s.prev_camera_pitch = s.camera_pitch;

    let mut yaw = mc_math::sqrt_d(s.vel.x * s.vel.x + s.vel.z * s.vel.z);
    // java: `EntityPlayer.onLivingUpdate:636` — `(float)(Math.atan(-motionY * 0.2..D) * 15.0D)`:
    // the `* 15` is a double product narrowed once, not an f32 product of a narrowed atan.
    let mut pitch = ((-s.vel.y * 0.200_000_002_980_232_24_f64).atan() * 15.0) as f32;
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

/// java: `EntityPlayer.moveEntityWithHeading` (`entity/player/EntityPlayer.java:1788`) — the
/// creative-flight wrapper around `EntityLivingBase`'s.
///
/// While flying, `jumpMovementFactor` is swapped for `flySpeed * (sprinting ? 2 : 1)` (handled
/// by [`PhysicsState::effective_jump_movement_factor`]) and, crucially, `motionY` is **replaced**
/// afterwards by the pre-move value times `0.6` — so the gravity step, the `0.98` vertical drag
/// and even the ladder's `0.2` push are all discarded on a flying tick.
///
/// The wrapper is `capabilities.isFlying && this.ridingEntity == null`: a flying *rider*'s own
/// travel is the plain `EntityLivingBase` one, gravity and drag included.
fn move_entity_with_heading_player(
    w: &World,
    s: &mut PhysicsState,
    strafe: f32,
    forward: f32,
    entity_collisions: &[EntityCollisionCandidate],
    border: Option<CollisionBorder>,
    outside_border: &mut bool,
) {
    if s.flying && !s.riding {
        let d3 = s.vel.y;
        move_entity_with_heading(
            w,
            s,
            strafe,
            forward,
            entity_collisions,
            border,
            outside_border,
        );
        s.vel.y = d3 * 0.6;
    } else {
        move_entity_with_heading(
            w,
            s,
            strafe,
            forward,
            entity_collisions,
            border,
            outside_border,
        );
    }
}

/// java: `EntityLivingBase.jump` + `EntityPlayer.jump`.
fn jump(s: &mut PhysicsState) {
    s.vel.y = f64::from(JUMP_MOTION);
    // java: `EntityLivingBase.jump:1567` — `isPotionActive(Potion.jump)` is *not* `isRemote`
    // gated, and the client's `activePotionsMap` is filled by `S1DPacketEntityEffect`, so Jump
    // Boost really does raise the client's own jump.
    if let Some(amplifier) = s.jump_boost {
        s.vel.y += f64::from((f32::from(amplifier) + 1.0) * 0.1);
    }
    if s.sprinting {
        let f = s.yaw * DEG_TO_RAD;
        // java: the products are f32 and widened afterwards. H1: `sin`/`cos` are the LUT.
        s.vel.x -= f64::from(sin(f) * SPRINT_JUMP_BOOST);
        s.vel.z += f64::from(cos(f) * SPRINT_JUMP_BOOST);
    }
}

/// java: `EntityPlayerSP.setSprinting` — the flag, the attribute modifier and
/// `sprintingTicksLeft` together. See [`PhysicsState::set_sprinting`].
fn set_sprinting(s: &mut PhysicsState, sprinting: bool) {
    s.set_sprinting(sprinting);
}

/// java: the arm half of `EntityPlayerSP.updateEntityActionState` (`:699-702`).
///
/// `EntityLivingBase.onLivingUpdate` (`:1991-2001`) reaches that method only while
/// `!isMovementBlocked()` (alive and not sleeping), and the method touches the arm only while
/// the player `isCurrentViewEntity()`; callers apply both gates. The float difference is widened,
/// halved and added in double precision, then narrowed once, as the Java expression is written.
fn update_render_arm(s: &mut PhysicsState) {
    s.prev_render_arm_yaw = s.render_arm_yaw;
    s.prev_render_arm_pitch = s.render_arm_pitch;
    s.render_arm_pitch =
        (f64::from(s.render_arm_pitch) + f64::from(s.pitch - s.render_arm_pitch) * 0.5) as f32;
    s.render_arm_yaw =
        (f64::from(s.render_arm_yaw) + f64::from(s.yaw - s.render_arm_yaw) * 0.5) as f32;
}

/// java: `Entity.moveFlying` (`Entity.java:1224`).
///
/// **Every arithmetic step here is `f32`**, including the trig, and only the two products are
/// widened when they land in `motionX`/`motionZ`. Doing it in `f64` produces a plausible number
/// that drifts from the server's — this is the other half of hazard H7.
pub fn move_flying(s: &mut PhysicsState, strafe: f32, forward: f32, friction: f32) {
    let mut strafe = strafe;
    let mut forward = forward;
    let mut f = strafe * strafe + forward * forward;
    if f >= 1.0e-4 {
        f = sqrt_f(f);
        if f < 1.0 {
            f = 1.0;
        }
        f = friction / f;
        strafe *= f;
        forward *= f;
        let f1 = sin(s.yaw * PI_F / 180.0);
        let f2 = cos(s.yaw * PI_F / 180.0);
        s.vel.x += f64::from(strafe * f2 - forward * f1);
        s.vel.z += f64::from(forward * f2 + strafe * f1);
    }
}

/// java: `EntityLivingBase.moveEntityWithHeading` (`EntityLivingBase.java:1602`).
fn move_entity_with_heading(
    w: &World,
    s: &mut PhysicsState,
    strafe: f32,
    forward: f32,
    entity_collisions: &[EntityCollisionCandidate],
    border: Option<CollisionBorder>,
    outside_border: &mut bool,
) {
    // java: `if (!this.isInWater() || this instanceof EntityPlayer && ((EntityPlayer)this)
    // .capabilities.isFlying)` — a flying player takes the *ground* branch even under water or
    // lava, which is why creative flight works inside an ocean.
    if !s.in_water || s.flying {
        if !fluid::is_in_lava(w, s) || s.flying {
            // ---- air / ground ---------------------------------------------------------
            let mut f4 = ground_slipperiness(w, s);
            let f = GROUND_ACCEL_NUMERATOR / (f4 * f4 * f4);
            let f5 = if s.on_ground {
                s.ai_move_speed() * f
            } else {
                s.effective_jump_movement_factor()
            };
            move_flying(s, strafe, forward, f5);

            // java: recomputed after moveFlying, before moveEntity — so it is the slipperiness
            // of the block you were standing on *before* the move, which is the whole reason
            // jumping off ice keeps ice's drag for that tick.
            f4 = ground_slipperiness(w, s);

            let on_ladder = is_on_ladder(w, s);
            if on_ladder {
                // java: `float f6 = 0.15F;` then `clamp_double(motionX, (double)(-f6), (double)f6)`
                // — the bound is the *f32* 0.15 widened, 0.15000000596046448, not 0.15. A
                // climbing player is pinned to this value for many ticks in a row, so the
                // 6e-9 difference is not a rounding curiosity: it is a persistent drift.
                let f6 = f64::from(0.15f32);
                s.vel.x = mc_math::clamp_f64(s.vel.x, -f6, f6);
                s.vel.z = mc_math::clamp_f64(s.vel.z, -f6, f6);
                s.fall_distance = 0.0;
                if s.vel.y < -0.15 {
                    s.vel.y = -0.15;
                }
                if s.sneaking && s.vel.y < 0.0 {
                    s.vel.y = 0.0;
                }
            }

            move_entity_full_with_context(w, s, s.vel, entity_collisions, border, outside_border);

            if s.collided_horizontally && is_on_ladder(w, s) {
                s.vel.y = 0.2;
            }

            // java: `EntityLivingBase.moveEntityWithHeading:1664` — on a **client** world, a
            // player whose own column is not loaded sinks at a fixed rate *instead of*
            // accelerating. This is the "drifting down while the chunks arrive" behaviour, and it
            // replaces the gravity step rather than adding to it.
            //
            // Note the coordinate: `new BlockPos((int)this.posX, 0, (int)this.posZ)` uses the
            // **int** constructor, so the cast truncates toward zero. `EntityPlayerSP.onUpdate`'s
            // outer gate one level up uses the *double* constructor and therefore floors —
            // different for negative coordinates, and both are reproduced as written.
            if column_is_loaded(w, s.pos) {
                s.vel.y -= GRAVITY;
            } else if s.pos.y > 0.0 {
                s.vel.y = -0.1;
            } else {
                s.vel.y = 0.0;
            }
            s.vel.y *= DRAG_Y;
            s.vel.x *= f64::from(f4);
            s.vel.z *= f64::from(f4);
        } else {
            // ---- lava -----------------------------------------------------------------
            let d1 = s.pos.y;
            move_flying(s, strafe, forward, 0.02);
            move_entity_full_with_context(w, s, s.vel, entity_collisions, border, outside_border);
            s.vel *= 0.5;
            s.vel.y -= 0.02;
            // The same getCollidingBoundingBoxes(this, ..) as the sweep.
            if s.collided_horizontally
                && fluid::is_offset_position_in_liquid_with(
                    w,
                    s,
                    DVec3::new(
                        s.vel.x,
                        s.vel.y + 0.600_000_023_841_857_9 - s.pos.y + d1,
                        s.vel.z,
                    ),
                    entity_collisions,
                    border,
                    outside_border,
                )
            {
                s.vel.y = 0.300_000_011_920_928_96;
            }
        }
    } else {
        // ---- water --------------------------------------------------------------------
        let d0 = s.pos.y;
        let mut f1 = 0.8f32;
        let mut f2 = 0.02f32;
        // java: EntityLivingBase.moveEntityWithHeading:1705 — Depth Strider raises *both* the
        // horizontal drag toward `0.54600006` and the acceleration toward the walk speed. The
        // level is clamped to 3 and halved while airborne (i.e. while swimming, not standing on
        // the bottom), and `0.54600006F` must stay exactly that.
        let mut f3 = s.depth_strider;
        if f3 > 3.0 {
            f3 = 3.0;
        }
        if !s.on_ground {
            f3 *= 0.5;
        }
        if f3 > 0.0 {
            f1 += (0.546_000_06 - f1) * f3 / 3.0;
            f2 += (s.ai_move_speed() * 1.0 - f2) * f3 / 3.0;
        }
        move_flying(s, strafe, forward, f2);
        move_entity_full_with_context(w, s, s.vel, entity_collisions, border, outside_border);
        s.vel.x *= f64::from(f1);
        s.vel.y *= 0.800_000_011_920_929;
        s.vel.z *= f64::from(f1);
        s.vel.y -= 0.02;
        // The same getCollidingBoundingBoxes(this, ..) as the sweep.
        if s.collided_horizontally
            && fluid::is_offset_position_in_liquid_with(
                w,
                s,
                DVec3::new(
                    s.vel.x,
                    s.vel.y + 0.600_000_023_841_857_9 - s.pos.y + d0,
                    s.vel.z,
                ),
                entity_collisions,
                border,
                outside_border,
            )
        {
            s.vel.y = 0.300_000_011_920_928_96;
        }
    }

    // java: `EntityLivingBase.moveEntityWithHeading:1737` — the last twelve lines, **outside**
    // every branch above and outside the `isServerWorld()` guard, so they run on every entity
    // every tick. This is the sole producer of the walk animation's two inputs.
    s.prev_limb_swing_amount = s.limb_swing_amount;
    let d2 = s.pos.x - s.prev_pos.x;
    let d3 = s.pos.z - s.prev_pos.z;
    // java: `MathHelper.sqrt_double` returns a **float**, so the whole expression is f32.
    let mut f7 = mc_math::sqrt_d(d2 * d2 + d3 * d3) * 4.0;
    if f7 > 1.0 {
        f7 = 1.0;
    }
    s.limb_swing_amount += (f7 - s.limb_swing_amount) * 0.4;
    s.limb_swing += s.limb_swing_amount;
}

/// `World.isBlockLoaded(new BlockPos((int)x, 0, (int)z))` and the chunk's own `isLoaded()`, as
/// `moveEntityWithHeading`'s gravity pin asks it.
///
/// On a vanilla *client* the `isBlockLoaded` half degrades to `isValid` — `ChunkProviderClient`
/// hard-codes `chunkExists` to `true` — but the second half does not: `provideChunk` hands out a
/// shared `EmptyChunk` for a column that has never arrived, and `EmptyChunk.isLoaded()` is
/// `false`. So the pin really does fire on an unsent column, and only on one.
fn column_is_loaded(w: &World, pos: DVec3) -> bool {
    // java: `(int)posX` truncates toward zero rather than flooring.
    let column = mc_math::BlockPos::new(pos.x as i32, 0, pos.z as i32).chunk_pos();
    w.is_chunk_loaded(column)
}

/// java: the `f4` expression in `moveEntityWithHeading` — `0.91F`, or the slipperiness of the
/// block one below the box's bottom, times `0.91F`, when on the ground.
fn ground_slipperiness(w: &World, s: &PhysicsState) -> f32 {
    if !s.on_ground {
        return BASE_SLIPPERINESS;
    }
    let pos = mc_math::BlockPos::new(
        floor_f64(s.pos.x),
        floor_f64(s.aabb().min.y) - 1,
        floor_f64(s.pos.z),
    );
    block_props(w.block_at(pos)).slipperiness * BASE_SLIPPERINESS
}

/// java: `EntityLivingBase.isOnLadder` (`:1140`) — the block at the *feet* is a ladder or a
/// vine, **and** the entity is not a spectating player:
/// `(block == ladder || block == vine) && (!(this instanceof EntityPlayer) || !isSpectator())`.
#[must_use]
pub fn is_on_ladder(w: &World, s: &PhysicsState) -> bool {
    if s.spectator {
        return false;
    }
    let pos = mc_math::BlockPos::new(
        floor_f64(s.pos.x),
        floor_f64(s.aabb().min.y),
        floor_f64(s.pos.z),
    );
    let name = block_info(block_props(w.block_at(pos)).block).name;
    name == "minecraft:ladder" || name == "minecraft:vine"
}

/// java: the four `pushOutOfBlocks` calls at `EntityPlayerSP.onLivingUpdate:795`.
///
/// This is public only so the session-owned mounted `updateRidden` seam can run the exact same
/// four probes before it emits C05+C0C; ordinary walking ticks call it through the player body.
pub fn push_out_of_blocks_all(w: &World, s: &mut PhysicsState) {
    let bb_min_y = s.aabb().min.y;
    let w35 = f64::from(s.width) * 0.35;
    let y = bb_min_y + 0.5;
    push_out_of_blocks(w, s, s.pos.x - w35, y, s.pos.z + w35);
    push_out_of_blocks(w, s, s.pos.x - w35, y, s.pos.z - w35);
    push_out_of_blocks(w, s, s.pos.x + w35, y, s.pos.z - w35);
    push_out_of_blocks(w, s, s.pos.x + w35, y, s.pos.z + w35);
}

/// java: `EntityPlayerSP.pushOutOfBlocks` (`EntityPlayerSP.java:437`).
///
/// It never teleports — it only sets a `±0.1` motion component toward the nearest open side —
/// which is why being stuck in a block shoves you out over several ticks.
fn push_out_of_blocks(w: &World, s: &mut PhysicsState, x: f64, y: f64, z: f64) {
    // java: `EntityPlayerSP.pushOutOfBlocks:439` — `if (this.noClip) return false;`. A spectator
    // is *meant* to be inside blocks, so nothing shoves them back out.
    if s.no_clip {
        return;
    }
    let pos = mc_math::BlockPos::new(floor_f64(x), floor_f64(y), floor_f64(z));
    let d0 = x - f64::from(pos.x);
    let d1 = z - f64::from(pos.z);

    if is_open_block_space(w, pos) {
        return;
    }

    let mut i = -1;
    let mut d2 = 9999.0f64;
    if is_open_block_space(w, pos.west()) && d0 < d2 {
        d2 = d0;
        i = 0;
    }
    if is_open_block_space(w, pos.east()) && 1.0 - d0 < d2 {
        d2 = 1.0 - d0;
        i = 1;
    }
    if is_open_block_space(w, pos.north()) && d1 < d2 {
        d2 = d1;
        i = 4;
    }
    if is_open_block_space(w, pos.south()) && 1.0 - d1 < d2 {
        d2 = 1.0 - d1;
        i = 5;
    }
    // South is the final candidate today, but Java still updates the nearest-side accumulator.
    // Keep that state observable so this parity detail cannot regress into a dead-looking write.
    if i == 5 {
        debug_assert_eq!(d2, 1.0 - d1);
    }

    let f = 0.1f32;
    match i {
        0 => s.vel.x = f64::from(-f),
        1 => s.vel.x = f64::from(f),
        4 => s.vel.z = f64::from(-f),
        5 => s.vel.z = f64::from(f),
        _ => {}
    }
}

/// java: `EntityPlayerSP.isOpenBlockSpace` — neither this block nor the one above is a normal
/// cube.
fn is_open_block_space(w: &World, pos: mc_math::BlockPos) -> bool {
    !block_props(w.block_at(pos)).normal_cube && !block_props(w.block_at(pos.up())).normal_cube
}

/// Whether the player's eyes are inside the given material — vanilla's underwater-overlay and
/// breath test. java: `Entity.isInsideOfMaterial`.
#[must_use]
pub fn is_inside_of_material(w: &World, s: &PhysicsState, material: Material) -> bool {
    let d0 = s.eye_pos().y;
    let pos = mc_math::BlockPos::new(floor_f64(s.pos.x), floor_f64(d0), floor_f64(s.pos.z));
    let state = w.block_at(pos);
    if block_props(state).material != material {
        return false;
    }
    let f = fluid::liquid_height_percent(state.meta()) - 0.111_111_11;
    let f1 = (pos.y + 1) as f32 - f;
    let flag = d0 < f64::from(f1);
    // java: `return !flag && this instanceof EntityPlayer ? false : flag;`
    flag
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_data::{block_by_name, BlockState};
    use mc_math::BlockPos;

    /// A plain big enough that 200 ticks of sprinting stays on it. A world that runs out from
    /// under the player turns a ground-speed test into an air-drag test.
    fn flat(block: &str) -> World {
        crate::scene::flat(BlockState::new(block_by_name(block).unwrap(), 0), 63, 128)
    }

    fn player() -> PhysicsState {
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5));
        s.on_ground = true;
        s
    }

    #[test]
    fn standing_still_stays_still() {
        let w = flat("stone");
        let mut s = player();
        let input = MoveInput::default();
        for _ in 0..40 {
            tick_player(&w, &mut s, &input);
        }
        assert_eq!(s.pos, DVec3::new(0.5, 64.0, 0.5));
        assert!(s.on_ground);
    }

    #[test]
    fn local_camera_sway_uses_java_caps_gates_and_damping() {
        let mut s = player();
        s.vel = DVec3::new(0.2, 0.0, 0.0);
        update_camera_sway(&mut s);
        assert_eq!(s.prev_camera_yaw, 0.0);
        assert!((s.camera_yaw - 0.04).abs() < 1e-6);
        assert_eq!(s.camera_pitch, 0.0);

        s.on_ground = false;
        s.vel = DVec3::new(0.2, 0.5, 0.0);
        update_camera_sway(&mut s);
        assert!((s.prev_camera_yaw - 0.04).abs() < 1e-6);
        assert!((s.camera_yaw - 0.024).abs() < 1e-6);
        assert!(s.camera_pitch < 0.0);

        s.movement_blocked = true;
        update_camera_sway(&mut s);
        assert!(s.camera_yaw < 0.024);
        assert!(s.camera_pitch > -1.2, "blocked sway decays toward zero");
    }

    /// `EntityPlayerSP.updateEntityActionState` (`:699-702`) moves each render-arm angle half-way
    /// to the rotation once per tick, only for the current view entity, and only while
    /// `EntityLivingBase.onLivingUpdate` (`:1991-2001`) is not movement-blocked.
    #[test]
    fn render_arm_chases_the_rotation_half_way_per_tick_under_javas_gates() {
        let w = flat("stone");
        let input = MoveInput {
            yaw: 90.0,
            pitch: 30.0,
            ..MoveInput::default()
        };

        let mut s = player();
        tick_player_with_entities_view(&w, &mut s, &input, &[], true);
        assert_eq!((s.prev_render_arm_yaw, s.render_arm_yaw), (0.0, 45.0));
        assert_eq!((s.prev_render_arm_pitch, s.render_arm_pitch), (0.0, 15.0));
        tick_player_with_entities_view(&w, &mut s, &input, &[], true);
        assert_eq!((s.prev_render_arm_yaw, s.render_arm_yaw), (45.0, 67.5));
        assert_eq!((s.prev_render_arm_pitch, s.render_arm_pitch), (15.0, 22.5));

        // `isCurrentViewEntity()` is false while spectating through another entity.
        let mut foreign = player();
        tick_player_with_entities_view(&w, &mut foreign, &input, &[], false);
        assert_eq!(
            (foreign.render_arm_yaw, foreign.render_arm_pitch),
            (0.0, 0.0)
        );

        // Dead or asleep: `isMovementBlocked()` skips `updateEntityActionState` altogether.
        let mut blocked = player();
        blocked.movement_blocked = true;
        tick_player_with_entities_view(&w, &mut blocked, &input, &[], true);
        assert_eq!(
            (blocked.render_arm_yaw, blocked.render_arm_pitch),
            (0.0, 0.0)
        );

        // A mounted player's `onLivingUpdate` still reaches the same method.
        let mut mounted = player();
        mounted_on_living_update_preflight(&w, &mut mounted, &input, true);
        assert_eq!(
            (mounted.render_arm_yaw, mounted.render_arm_pitch),
            (45.0, 15.0)
        );
    }

    /// `EntityPlayerSP.updateEntityActionState` still runs for a foreign render-view entity, but
    /// its current-view guard prevents the freshly sampled movement keys from replacing the
    /// persistent `EntityLivingBase` action fields. The later living travel phase nevertheless
    /// consumes those old fields, so the local body keeps moving under the detached camera, and
    /// its in-place `*= 0.98F` (`EntityLivingBase.onLivingUpdate:2031`) decays them every tick
    /// (the fields used to stay at their copied value forever).
    #[test]
    fn foreign_camera_preserves_action_state_but_still_runs_travel() {
        let w = flat("stone");
        let mut s = player();
        s.move_strafing = 0.25;
        s.move_forward = 1.0;
        s.jumping = false;
        let before = s.pos;
        let input = MoveInput {
            back: true,
            jump: true,
            ..MoveInput::default()
        };

        tick_player_with_entities_view(&w, &mut s, &input, &[], false);

        assert_eq!(
            s.move_strafing,
            0.25 * 0.98,
            "foreign input must not replace strafe; travel damps it in place"
        );
        assert_eq!(
            s.move_forward, 0.98,
            "foreign input must not replace forward; travel damps it in place"
        );
        tick_player_with_entities_view(&w, &mut s, &input, &[], false);
        assert_eq!(
            s.move_forward,
            0.98f32 * 0.98,
            "a second foreign tick decays the kept action again"
        );
        assert!(
            !s.jumping,
            "foreign input must not replace the jump action field"
        );
        assert!(
            s.pos.z > before.z,
            "travel still consumes the persistent forward action"
        );
        assert_eq!(
            s.prev_move_forward, -1.0,
            "raw movement edges still refresh"
        );
        assert!(
            s.prev_jump,
            "raw jump state still refreshes for the next edge"
        );
    }

    /// `EntityPlayerSP.onLivingUpdate` forces flight on (`:825`) and refuses the
    /// landing clear (`:904`) by asking `playerController.isSpectatorMode()`, while `noClip`,
    /// `onGround` and ladders use the tab-list `isSpectator()`. The two are separate fields.
    #[test]
    fn flight_branches_follow_the_controller_spectator_mode_not_the_tab_row() {
        let w = flat("stone");
        // Controller says Spectator, no tab row yet: flight is forced on, and a grounded
        // flyer is not cleared.
        let mut s = player();
        s.allow_flying = true;
        s.spectator_mode = true;
        tick_player(&w, &mut s, &MoveInput::default());
        assert!(s.flying, "isSpectatorMode() forces isFlying");
        assert!(!s.no_clip, "noClip follows the (absent) tab row");
        s.on_ground = true;
        tick_player(&w, &mut s, &MoveInput::default());
        assert!(
            s.flying,
            "no landing clear while the controller is in spectator mode"
        );

        // Tab row says Spectator, controller says Survival: no forced flight, and the landing
        // clear runs. (A tab-row spectator is noClip, so it never lands by collision; seed it.)
        let mut t = player();
        t.allow_flying = true;
        t.spectator = true;
        tick_player(&w, &mut t, &MoveInput::default());
        assert!(!t.flying, "the tab row alone does not force flight");
        t.flying = true;
        t.spectator = false;
        t.on_ground = true;
        tick_player(&w, &mut t, &MoveInput::default());
        assert!(
            !t.flying,
            "a grounded flyer in a non-spectator controller mode is cleared"
        );
    }

    /// The capability toggle is outside the `isCurrentViewEntity()` gate, while creative-flight
    /// vertical thrust is inside it. A foreign camera therefore receives the C13-driving state
    /// mutation but does not move the local body from the detached camera's jump key.
    #[test]
    fn foreign_camera_toggles_flight_but_does_not_apply_view_local_thrust() {
        let w = flat("stone");
        let mut foreign = player();
        foreign.on_ground = false;
        foreign.allow_flying = true;
        foreign.fly_toggle_timer = 1;
        foreign.prev_jump = false;

        let mut local = foreign;
        let input = MoveInput {
            jump: true,
            ..MoveInput::default()
        };

        tick_player_with_entities_view(&w, &mut foreign, &input, &[], false);
        tick_player_with_entities_view(&w, &mut local, &input, &[], true);

        assert!(
            foreign.flying && local.flying,
            "the double-tap mutation is view-independent"
        );
        assert_eq!(
            foreign.vel.y, 0.0,
            "foreign-camera flight must not consume the view-local vertical thrust"
        );
        assert!(
            local.vel.y > 0.0,
            "the local view applies jump thrust while flying"
        );
    }

    /// Mounted `EntityPlayerSP` still reaches the living preflight even though the session owns
    /// the later C05+C0C pair. The action-field copy remains current-view gated. The camera sway
    /// is the travel's tail (H2), so it reads the rider's post-travel motion and `onGround`, not
    /// the motion `updateRidden` zeroed: Java's rider settles at a small positive `cameraPitch`.
    #[test]
    fn mounted_preflight_gates_actions_and_updates_both_sway_channels() {
        let w = flat("stone");
        let input = MoveInput {
            forward: true,
            left: true,
            jump: true,
            sneak: true,
            ..MoveInput::default()
        };

        let mut foreign = player();
        foreign.move_strafing = -0.2;
        foreign.move_forward = 0.7;
        foreign.jumping = false;
        foreign.camera_yaw = 0.25;
        foreign.camera_pitch = 0.5;
        mounted_on_living_update_preflight(&w, &mut foreign, &input, false);
        assert_eq!(foreign.move_strafing, -0.2);
        assert_eq!(foreign.move_forward, 0.7);
        assert!(!foreign.jumping);
        assert_eq!(
            (foreign.camera_yaw, foreign.camera_pitch),
            (0.25, 0.5),
            "the sway belongs to the travel's tail"
        );

        let mut latch = false;
        mounted_living_travel(&w, &mut foreign, &[], None, &mut latch);
        // The rider's own move requested y == 0, so `onGround = isCollidedVertically && y < 0`
        // is false even on the floor; gravity then leaves `-0.08 * 0.98` for the camera pitch.
        assert!(!foreign.on_ground);
        let vy = (0.0 - GRAVITY) * DRAG_Y;
        assert_eq!(foreign.vel.y, vy);
        assert_eq!(foreign.prev_camera_yaw, 0.25);
        assert_eq!(foreign.camera_yaw, 0.25 + (0.0 - 0.25) * 0.4);
        assert_eq!(foreign.prev_camera_pitch, 0.5);
        let f1 = ((-vy * 0.200_000_002_980_232_24).atan() * 15.0) as f32;
        assert_eq!(foreign.camera_pitch, 0.5 + (f1 - 0.5) * 0.8);
        assert_eq!(
            foreign.pos,
            DVec3::new(0.5, 64.0, 0.5),
            "the seat, not the travel"
        );

        let mut local = player();
        mounted_on_living_update_preflight(&w, &mut local, &input, true);
        assert_eq!(local.move_strafing, 0.3);
        assert_eq!(local.move_forward, 0.3);
        assert!(local.jumping);
    }

    /// H2. `Entity.onEntityUpdate` clears `firstUpdate` as its last statement (`:524`), before
    /// `onLivingUpdate`'s travel, so a fresh player that drops into water during its very first
    /// tick splashes from `updateFallState`'s water probe. The flag used to be cleared at the end
    /// of the whole tick, which silenced that splash.
    #[test]
    fn a_first_tick_drop_into_water_splashes() {
        let mut w = flat("stone");
        let water = BlockState::new(block_by_name("water").unwrap(), 0);
        for x in -2..=2 {
            for z in -2..=2 {
                crate::scene::set_block(&mut w, BlockPos::new(x, 64, z), water);
            }
        }
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 66.0, 0.5));
        assert!(s.first_update);
        s.vel.y = -1.5;
        tick_player(&w, &mut s, &MoveInput::default());
        assert!(s.in_water);
        assert!(
            s.water_entry.is_some(),
            "the air-to-water edge inside the first tick's move resets height"
        );
    }

    /// `Entity.updateRidden` runs the rider's full `onUpdate`,
    /// so its own travel recomputes `isCollidedHorizontally` every ride tick, and the jump section
    /// zeroes `jumpTicks` while jump is up (`EntityLivingBase.onLivingUpdate:2005-2026`). A flag
    /// left by walking into the boat therefore cancels a mounted sprint for one tick only.
    #[test]
    fn a_rider_travel_refreshes_the_collision_flag_and_the_jump_cooldown() {
        let w = flat("stone");
        let mut walked_into_the_boat = player();
        walked_into_the_boat.collided_horizontally = true;
        walked_into_the_boat.jump_ticks = 7;
        let input = MoveInput {
            forward: true,
            sprint: true,
            ..MoveInput::default()
        };

        let mut s = walked_into_the_boat;
        let mut latch = false;
        mounted_on_living_update_preflight(&w, &mut s, &input, true);
        assert!(
            !s.sprinting,
            "tick 1 still reads the walking tick's collision"
        );
        mounted_living_travel(&w, &mut s, &[], None, &mut latch);
        assert!(
            !s.collided_horizontally,
            "the rider's own move found no wall"
        );
        assert_eq!(s.jump_ticks, 0, "jump up: `jumpTicks = 0`, not a countdown");
        mounted_on_living_update_preflight(&w, &mut s, &input, true);
        mounted_living_travel(&w, &mut s, &[], None, &mut latch);
        assert!(s.sprinting, "tick 2 keeps the sprint the key started");
        assert_eq!(s.move_forward, 0.98, "the C0C field is the damped one");

        // Negative control: the pre-H2 mounted tick had no travel, so the stale flag stayed
        // and stopped the sprint on every ride tick, while the cooldown only counted down.
        let mut old = walked_into_the_boat;
        for _ in 0..2 {
            mounted_on_living_update_preflight(&w, &mut old, &input, true);
        }
        assert!(old.collided_horizontally && !old.sprinting);
        assert_eq!(old.jump_ticks, 5);
    }

    /// H2. A rider's own travel takes none of the three `ridingEntity == null` paths: the flight
    /// wrapper (gravity applies, `motionY` is not replaced by `d3 * 0.6`), the walking branch (no
    /// walked distance or footstep), and `BlockPortal.setPortal`.
    #[test]
    fn a_rider_travel_skips_the_flight_wrapper_walking_branch_and_portal() {
        let mut w = flat("stone");
        crate::scene::set_block(
            &mut w,
            BlockPos::new(0, 64, 0),
            BlockState::new(block_by_name("portal").unwrap(), 1),
        );
        let mut s = player();
        s.allow_flying = true;
        s.flying = true;
        s.on_ground = false;
        s.set_position(DVec3::new(0.5, 64.5, 0.5));
        let input = MoveInput {
            forward: true,
            ..MoveInput::default()
        };
        let mut latch = false;
        mounted_on_living_update_preflight(&w, &mut s, &input, true);
        mounted_living_travel(&w, &mut s, &[], None, &mut latch);
        assert_eq!(
            s.vel.y,
            (0.0 - GRAVITY) * DRAG_Y,
            "no flight wrapper for a rider"
        );
        assert!(s.flying, "airborne, so no landing clear");
        assert_eq!(s.distance_walked_modified, 0.0);
        assert!(s.step_sound.is_none());
        assert!(!s.in_portal, "BlockPortal skips riders");
        assert_eq!(s.portal_cooldown, 0);

        // The same position on foot does set the portal flag.
        let mut walker = player();
        walker.set_position(DVec3::new(0.5, 64.5, 0.5));
        tick_player(&w, &mut walker, &MoveInput::default());
        assert!(walker.in_portal);
        assert!(!walker.riding);
    }

    #[test]
    fn portal_cooldown_decrements_before_collision_and_saturates_at_zero() {
        let stone = flat("stone");
        let mut s = player();
        s.portal_cooldown = 2;
        tick_player(&stone, &mut s, &MoveInput::default());
        assert_eq!(s.portal_cooldown, 1);
        tick_player(&stone, &mut s, &MoveInput::default());
        assert_eq!(s.portal_cooldown, 0);
        tick_player(&stone, &mut s, &MoveInput::default());
        assert_eq!(s.portal_cooldown, 0);

        let mut portal = flat("stone");
        crate::scene::set_block(
            &mut portal,
            BlockPos::new(0, 64, 0),
            BlockState::new(block_by_name("portal").unwrap(), 1),
        );
        let mut expiring = player();
        expiring.portal_cooldown = 1;
        tick_player(&portal, &mut expiring, &MoveInput::default());
        assert_eq!(expiring.portal_cooldown, 0);
        assert!(
            expiring.in_portal,
            "the tick decrements 1 to 0 before BlockPortal calls setPortal"
        );
    }

    #[test]
    fn walking_forward_at_yaw_zero_moves_plus_z() {
        let w = flat("stone");
        let mut s = player();
        let input = MoveInput {
            forward: true,
            ..MoveInput::default()
        };
        for _ in 0..40 {
            tick_player(&w, &mut s, &input);
        }
        assert_eq!(s.pos.x, 0.5, "no lateral drift");
        assert!(s.pos.z > 5.0, "got {}", s.pos.z);
        assert!(!s.sprinting);
    }

    #[test]
    fn entity_collision_candidates_reach_the_high_level_player_tick() {
        // World.getCollidingBoundingBoxes appends a boat's own AABB after the block scan.  Keep
        // the world empty here so this control can only pass when the candidate slice reaches
        // move_entity_full_with_entities; the identical block-only control must walk through.
        let w = World::new();
        let boat = mc_math::Aabb::from_bounds(-0.5, 64.0, 1.0, 0.5, 65.0, 2.0);
        let candidate = EntityCollisionCandidate::new(boat, Some(boat));
        let input = MoveInput::default();

        let mut blocked = player();
        blocked.vel = DVec3::new(0.0, 0.0, 3.0);
        tick_player_with_entities(&w, &mut blocked, &input, &[candidate]);
        assert_eq!(
            blocked.aabb().max.z,
            1.0,
            "the high-level tick forwards the boat box into its movement sweep"
        );

        let mut block_only = player();
        block_only.vel = DVec3::new(0.0, 0.0, 3.0);
        tick_player(&w, &mut block_only, &input);
        assert!(
            block_only.aabb().max.z > 3.0,
            "the negative control reaches the requested displacement without a candidate"
        );
    }

    #[test]
    fn walk_speed_settles_at_the_vanilla_figure() {
        let w = flat("stone");
        let mut s = player();
        let input = MoveInput {
            forward: true,
            ..MoveInput::default()
        };
        for _ in 0..200 {
            tick_player(&w, &mut s, &input);
        }
        let before = s.pos.z;
        tick_player(&w, &mut s, &input);
        // Vanilla's steady-state walk is 0.21585 blocks/tick (4.317 m/s).
        assert!(s.on_ground, "still on the plain");
        assert!(
            (s.pos.z - before - 0.21585).abs() < 1e-4,
            "{}",
            s.pos.z - before
        );
    }

    #[test]
    fn sprinting_is_faster_and_the_key_engages_it() {
        let w = flat("stone");
        let mut s = player();
        let input = MoveInput {
            forward: true,
            sprint: true,
            ..MoveInput::default()
        };
        for _ in 0..200 {
            tick_player(&w, &mut s, &input);
        }
        assert!(s.sprinting);
        let before = s.pos.z;
        tick_player(&w, &mut s, &input);
        // 0.28061 blocks/tick (5.612 m/s).
        assert!(s.on_ground, "still on the plain");
        assert!(
            (s.pos.z - before - 0.28061).abs() < 1e-4,
            "{}",
            s.pos.z - before
        );
    }

    #[test]
    fn a_jump_peaks_at_about_one_and_a_quarter_blocks() {
        let w = flat("stone");
        let mut s = player();
        let input = MoveInput {
            jump: true,
            ..MoveInput::default()
        };
        let mut peak = s.pos.y;
        for _ in 0..40 {
            tick_player(&w, &mut s, &input);
            peak = peak.max(s.pos.y);
        }
        // The continuous apex is 1.2522 blocks, but a tick-sampled jump peaks at 1.24919 --
        // vanilla never observes the true apex, because it only samples every 50 ms.
        assert!(
            (peak - 65.249_187_078_744_68).abs() < 1e-9,
            "peak was {peak}"
        );
        // Holding jump re-jumps the moment the 10-tick cooldown expires, so the player is
        // airborne about as often as not; what matters is that the apex is reproducible.
        assert!(s.pos.y >= 64.0);
    }

    #[test]
    fn the_jump_cooldown_is_ten_ticks() {
        let w = flat("stone");
        let mut s = player();
        let input = MoveInput {
            jump: true,
            ..MoveInput::default()
        };
        tick_player(&w, &mut s, &input);
        assert_eq!(s.jump_ticks, 10);
        assert!(s.vel.y > 0.0);
    }

    #[test]
    fn jump_boost_uses_the_signed_packet_amplifier_without_clamping() {
        // Keep the column loaded but empty. The -128 impulse is downward by design and would
        // collide with a stone floor before this test could observe the Java jump arithmetic.
        let mut w = World::new();
        crate::scene::ensure_chunk(&mut w, mc_math::ChunkPos::new(0, 0));
        for amplifier in [127_i8, -128_i8, -1_i8] {
            let mut s = player();
            s.jump_boost = Some(amplifier);
            tick_player(
                &w,
                &mut s,
                &MoveInput {
                    jump: true,
                    ..MoveInput::default()
                },
            );

            // java: `jump()` adds `(amplifier + 1) * 0.1F`, then the ordinary air tick subtracts
            // 0.08D and applies 0.98D drag. Negative signed bytes are intentional here: wire
            // 128 and 255 arrive as -128 and -1, rather than being clamped to level zero.
            let jump_term = f64::from((f32::from(amplifier) + 1.0) * 0.1);
            let expected = (f64::from(JUMP_MOTION) + jump_term - GRAVITY) * DRAG_Y;
            assert_eq!(s.vel.y, expected, "signed amplifier {amplifier}");
        }

        // Ordinary/no-effect control: no potion contributes only the base jump impulse.
        let mut s = player();
        tick_player(
            &w,
            &mut s,
            &MoveInput {
                jump: true,
                ..MoveInput::default()
            },
        );
        let expected = (f64::from(JUMP_MOTION) - GRAVITY) * DRAG_Y;
        assert_eq!(s.vel.y, expected);
    }

    #[test]
    fn ice_is_slippery() {
        let stone = flat("stone");
        let ice = flat("ice");
        let input = MoveInput {
            forward: true,
            ..MoveInput::default()
        };
        let stop = MoveInput::default();
        let mut a = player();
        let mut b = player();
        for _ in 0..60 {
            tick_player(&stone, &mut a, &input);
            tick_player(&ice, &mut b, &input);
        }
        let (a0, b0) = (a.pos.z, b.pos.z);
        for _ in 0..40 {
            tick_player(&stone, &mut a, &stop);
            tick_player(&ice, &mut b, &stop);
        }
        assert!(
            b.pos.z - b0 > 4.0 * (a.pos.z - a0),
            "ice slides much further: {} vs {}",
            b.pos.z - b0,
            a.pos.z - a0
        );
    }

    #[test]
    fn a_sprint_jump_carries_further_than_a_walk_jump() {
        let w = flat("stone");
        let sprint = MoveInput {
            forward: true,
            jump: true,
            sprint: true,
            ..MoveInput::default()
        };
        let walk = MoveInput {
            forward: true,
            jump: true,
            ..MoveInput::default()
        };
        let mut a = player();
        let mut b = player();
        for _ in 0..80 {
            tick_player(&w, &mut a, &sprint);
            tick_player(&w, &mut b, &walk);
        }
        assert!(a.sprinting);
        assert!(a.pos.z > b.pos.z * 1.2, "{} vs {}", a.pos.z, b.pos.z);
    }

    #[test]
    fn falling_accumulates_fall_distance_and_damage() {
        let w = flat("stone");
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 90.0, 0.5));
        let input = MoveInput::default();
        let mut damage = None;
        for _ in 0..200 {
            tick_player(&w, &mut s, &input);
            if s.predicted_fall_damage.is_some() {
                damage = s.predicted_fall_damage;
            }
        }
        assert_eq!(s.pos.y, 64.0);
        assert_eq!(damage, Some(23), "26 blocks of fall, minus the 3 free ones");
        assert_eq!(s.fall_distance, 0.0);
    }

    #[test]
    fn terminal_velocity_settles_near_minus_three_point_nine_two() {
        // No floor at all: `World::block_at` is air outside y=0..255, so the fall never ends.
        // The *column* still has to be loaded, or `moveEntityWithHeading`'s pin replaces gravity
        // with a flat `-0.1` — which is what `an_unloaded_column_pins_the_sink_rate` is for.
        let mut w = World::new();
        crate::scene::ensure_chunk(&mut w, mc_math::ChunkPos::new(0, 0));
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 250.0, 0.5));
        let input = MoveInput::default();
        for _ in 0..500 {
            tick_player(&w, &mut s, &input);
        }
        // `v' = (v - 0.08) * 0.98` has the fixed point -0.08 * 0.98 / 0.02 = -3.92. 1.8.9 has
        // no explicit clamp; the number everyone quotes is that fixed point.
        // Approached asymptotically at 0.98 per tick, so 500 ticks leaves ~1.6e-4.
        assert!((s.vel.y + 3.92).abs() < 1e-3, "{}", s.vel.y);
        assert!(s.fall_distance > 1000.0);
    }

    /// java: `EntityLivingBase.moveEntityWithHeading:1664` — over a column the server has
    /// not sent, a **client** replaces the `-0.08` acceleration with a flat `motionY = -0.1`, so
    /// you sink at a constant `0.098` blocks/tick instead of accelerating to terminal velocity.
    /// The `unloaded_chunk` Java fixture pins the exact trace; this pins the shape.
    #[test]
    fn an_unloaded_column_pins_the_sink_rate() {
        let mut w = World::new();
        crate::scene::ensure_chunk(&mut w, mc_math::ChunkPos::new(0, 0));
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 250.0, 0.5));
        let input = MoveInput::default();
        for _ in 0..40 {
            tick_player(&w, &mut s, &input);
        }
        let loaded_after_40 = s.vel.y;
        assert!(loaded_after_40 < -1.0, "accelerating: {loaded_after_40}");

        // The very same world one column over, which was never loaded.
        let mut s = PhysicsState::new_player(DVec3::new(200.5, 250.0, 200.5));
        for _ in 0..40 {
            tick_player(&w, &mut s, &input);
        }
        assert_eq!(s.vel.y, -0.1 * DRAG_Y);
        // java: `if (this.posY > 0.0D) motionY = -0.1 else motionY = 0.0` — below the world the
        // pin is zero, so an entity under y=0 in an unloaded column simply stops.
        let mut s = PhysicsState::new_player(DVec3::new(200.5, -5.0, 200.5));
        tick_player(&w, &mut s, &input);
        assert_eq!(s.vel.y, 0.0);
    }

    #[test]
    fn sneaking_at_a_ledge_does_not_fall_off() {
        let mut w = World::new();
        let stone = BlockState::new(block_by_name("stone").unwrap(), 0);
        for x in 0..4 {
            for z in 0..4 {
                crate::scene::set_block(&mut w, BlockPos::new(x, 63, z), stone);
            }
        }
        let mut s = PhysicsState::new_player(DVec3::new(2.0, 64.0, 2.0));
        s.on_ground = true;
        let input = MoveInput {
            forward: true,
            sneak: true,
            ..MoveInput::default()
        };
        for _ in 0..200 {
            tick_player(&w, &mut s, &input);
        }
        assert_eq!(s.pos.y, 64.0, "still on the platform");
        assert!(s.pos.z < 4.4, "did not walk into the void: {}", s.pos.z);
        assert!(s.pos.z > 3.0, "did reach the edge: {}", s.pos.z);
    }

    /// **F-P0-client.entity-01-11.** `EntityPlayerSP.pushOutOfBlocks` compares the four open
    /// sides by distance from the fractional position.  The south distance is `1 - d1`, not
    /// `d1`; use an asymmetric position so a reversion mutant visibly shoves north instead.
    #[test]
    fn push_out_of_blocks_uses_south_complement_distance_and_keeps_east_west_control() {
        let stone = BlockState::new(block_by_name("stone").unwrap(), 0);
        let mut w = World::new();
        // The center is solid, north/south are open, and west/east are solid.  All four player
        // corner probes remain in the center at x=.5,z=.7, so the live tick helper sees the same
        // asymmetric north/south distances (north=.7, south=.3).
        for pos in [
            BlockPos::new(0, 64, 0),
            BlockPos::new(-1, 64, 0),
            BlockPos::new(1, 64, 0),
        ] {
            crate::scene::set_block(&mut w, pos, stone);
        }
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.7));
        tick_player(&w, &mut s, &MoveInput::default());
        assert!(
            s.vel.z > 0.0,
            "nearest open side is south at 1-d1=.3; velocity={:?}",
            s.vel
        );

        // With north/south closed and only east open, the same helper still selects east; this
        // guards the neighboring branches while keeping the test's open-side oracle focused.
        let mut east_only = World::new();
        for pos in [
            BlockPos::new(0, 64, 0),
            BlockPos::new(-1, 64, 0),
            BlockPos::new(0, 64, -1),
            BlockPos::new(0, 64, 1),
        ] {
            crate::scene::set_block(&mut east_only, pos, stone);
        }
        let mut east_state = PhysicsState::new_player(DVec3::new(0.2, 64.0, 0.5));
        tick_player(&east_only, &mut east_state, &MoveInput::default());
        assert!(
            east_state.vel.x > 0.0,
            "only east is open, so the helper must shove east: velocity={:?}",
            east_state.vel
        );

        let mut west_only = World::new();
        for pos in [
            BlockPos::new(0, 64, 0),
            BlockPos::new(1, 64, 0),
            BlockPos::new(0, 64, -1),
            BlockPos::new(0, 64, 1),
        ] {
            crate::scene::set_block(&mut west_only, pos, stone);
        }
        let mut west_state = PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5));
        tick_player(&west_only, &mut west_state, &MoveInput::default());
        assert!(
            west_state.vel.x < 0.0,
            "only west is open, so the helper must shove west: velocity={:?}",
            west_state.vel
        );
    }

    #[test]
    fn a_ladder_can_be_climbed_and_clamps_the_descent() {
        let mut w = World::new();
        let stone = BlockState::new(block_by_name("stone").unwrap(), 0);
        let ladder = BlockState::new(block_by_name("ladder").unwrap(), 3);
        for x in -2..3 {
            for z in -2..3 {
                crate::scene::set_block(&mut w, BlockPos::new(x, 63, z), stone);
            }
        }
        for y in 64..80 {
            crate::scene::set_block(&mut w, BlockPos::new(0, y, 1), stone);
            crate::scene::set_block(&mut w, BlockPos::new(0, y, 0), ladder);
        }
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5));
        s.on_ground = true;
        let climb = MoveInput {
            forward: true,
            ..MoveInput::default()
        };
        for _ in 0..120 {
            tick_player(&w, &mut s, &climb);
        }
        assert!(s.pos.y > 70.0, "climbed to {}", s.pos.y);
        let top = s.pos.y;
        let hang = MoveInput::default();
        tick_player(&w, &mut s, &hang);
        assert!(top - s.pos.y <= 0.15 + 1e-9, "descent is clamped to 0.15");
    }

    #[test]
    fn water_slows_the_fall_and_lets_you_swim_up() {
        let mut w = World::new();
        let stone = BlockState::new(block_by_name("stone").unwrap(), 0);
        let water = BlockState::new(block_by_name("water").unwrap(), 0);
        for x in -4..5 {
            for z in -4..5 {
                crate::scene::set_block(&mut w, BlockPos::new(x, 59, z), stone);
                for y in 60..64 {
                    crate::scene::set_block(&mut w, BlockPos::new(x, y, z), water);
                }
            }
        }
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 62.0, 0.5));
        let sink = MoveInput::default();
        for _ in 0..40 {
            tick_player(&w, &mut s, &sink);
        }
        assert!(s.in_water);
        assert!(s.vel.y > -0.2, "water drag caps the sink rate: {}", s.vel.y);
        let bottom = s.pos.y;
        let swim = MoveInput {
            jump: true,
            ..MoveInput::default()
        };
        for _ in 0..60 {
            tick_player(&w, &mut s, &swim);
        }
        assert!(s.pos.y > bottom, "swam up from {bottom} to {}", s.pos.y);
    }

    #[test]
    fn soul_sand_is_slower_than_stone() {
        let stone = flat("stone");
        let soul = flat("soul_sand");
        let input = MoveInput {
            forward: true,
            sprint: true,
            ..MoveInput::default()
        };
        let mut a = player();
        let mut b = player();
        for _ in 0..100 {
            tick_player(&stone, &mut a, &input);
            tick_player(&soul, &mut b, &input);
        }
        assert!(b.pos.z < a.pos.z * 0.6, "{} vs {}", b.pos.z, a.pos.z);
    }

    #[test]
    fn a_cobweb_makes_walking_crawl() {
        let mut w = flat("stone");
        let web = BlockState::new(block_by_name("web").unwrap(), 0);
        for z in 2..8 {
            for y in 64..66 {
                crate::scene::set_block(&mut w, BlockPos::new(0, y, z), web);
            }
        }
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5));
        s.on_ground = true;
        let input = MoveInput {
            forward: true,
            sprint: true,
            ..MoveInput::default()
        };
        for _ in 0..40 {
            tick_player(&w, &mut s, &input);
        }
        assert!(s.pos.z > 2.0, "entered the web");
        assert!(s.pos.z < 5.0, "but is crawling: {}", s.pos.z);
    }

    #[test]
    fn is_inside_of_material_tracks_the_eye_not_the_feet() {
        let mut w = World::new();
        let water = BlockState::new(block_by_name("water").unwrap(), 0);
        for x in -2..3 {
            for z in -2..3 {
                for y in 60..62 {
                    crate::scene::set_block(&mut w, BlockPos::new(x, y, z), water);
                }
            }
        }
        // Feet at 60.0 -> eyes at 61.62, inside the water block at y=61.
        let s = PhysicsState::new_player(DVec3::new(0.5, 60.0, 0.5));
        assert!(is_inside_of_material(&w, &s, Material::Water));
        // Feet at 60.5 -> eyes at 62.12, in the air above the pool.
        let s = PhysicsState::new_player(DVec3::new(0.5, 60.5, 0.5));
        assert!(!is_inside_of_material(&w, &s, Material::Water));
        // Feet in the water, eyes above it: the test is on the eyes, not the feet.
        let s = PhysicsState::new_player(DVec3::new(0.5, 61.0, 0.5));
        assert!(!is_inside_of_material(&w, &s, Material::Water));
    }
}
