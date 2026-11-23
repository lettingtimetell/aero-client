//! `Entity.moveEntity` — the AABB sweep, the sneak edge-guard, and step-up.
//!
//! java: `Entity.java:598`. **Hazard H7 lives here.** The axis order is Y, then X, then Z, and
//! the step-up retry happens *after* that first sweep and keeps whichever of two candidate
//! resolutions travelled further horizontally. Reordering any of it feels fine in a flat test
//! world and desyncs the moment a real server disagrees.

use mc_data::{block_info, block_props, BlockState, Material};
use mc_math::{floor_f64, Aabb, BlockPos, DVec3};
use mc_world::World;

use crate::collision::{
    collision_boxes_in_with_border_and_entities, collision_boxes_in_with_entities, CollisionBorder,
    EntityCollisionCandidate,
};
use crate::state::{GroundContact, PhysicsState, StepSound};

/// java: the `d6` step of `moveEntity`'s sneak retry loops.
const SNEAK_PROBE_STEP: f64 = 0.05;

/// What one [`move_entity`] call resolved to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveResult {
    /// The delta actually applied, after clipping. java: the mutated `x`/`y`/`z` locals.
    pub applied: DVec3,
    /// java: `onGround` — `isCollidedVertically && requestedY < 0`.
    pub on_ground: bool,
    /// java: `isCollidedHorizontally`.
    pub collided_horizontally: bool,
    /// java: `isCollidedVertically`.
    pub collided_vertically: bool,
}

/// Sweeps `aabb` by `delta` against the world, mutating it in place.
///
/// This is S13's frozen entry point and uses the local player's parameters: a `0.6` step
/// height, not sneaking, not in a cobweb, and **not previously on the ground**. Returns
/// `(applied_delta, on_ground)`.
///
/// That last assumption is the one to know about. Vanilla gates step-up on
/// `this.onGround || d4 != y && d4 < 0.0D`, and a bare box has no `onGround` to read, so this
/// form takes the conservative half: an entity still steps up whenever its Y sweep clipped
/// downward this tick (which is the normal walking case, since gravity is applied first), but a
/// grounded entity moving with `motionY == 0` will not. Use [`move_entity_with`] when you have
/// the flag, or [`move_entity_full`] for the whole of `Entity.moveEntity` — the sneak
/// edge-guard, cobweb scaling, block-collision effects and fall-state bookkeeping included.
/// [`crate::tick_player`] uses the latter.
pub fn move_entity(w: &World, aabb: &mut Aabb, delta: DVec3) -> (DVec3, bool) {
    let r = move_entity_with(
        w,
        aabb,
        delta,
        crate::state::PLAYER_STEP_HEIGHT,
        false,
        false,
    );
    (r.applied, r.on_ground)
}

/// [`move_entity`] with World.java's caller-provided nearby-entity collision lane.
///
/// `entity_collisions` must be gathered from the current client entity tracker. The compatibility
/// entry point deliberately passes an empty slice so existing non-ECS callers keep their old
/// block-only behaviour.
pub fn move_entity_with_entities(
    w: &World,
    aabb: &mut Aabb,
    delta: DVec3,
    entity_collisions: &[EntityCollisionCandidate],
) -> (DVec3, bool) {
    let r = move_entity_with_entities_full(
        w,
        aabb,
        delta,
        crate::state::PLAYER_STEP_HEIGHT,
        false,
        false,
        entity_collisions,
    );
    (r.applied, r.on_ground)
}

/// The geometric sweep with every vanilla knob exposed: step height, the entity's previous
/// `onGround`, and whether the sneak edge-guard applies.
///
/// java: `Entity.moveEntity` minus the entity bookkeeping (motion zeroing, `updateFallState`,
/// `onLanded`, `doBlockCollisions`).
pub fn move_entity_with(
    w: &World,
    aabb: &mut Aabb,
    delta: DVec3,
    step_height: f32,
    on_ground: bool,
    sneaking: bool,
) -> MoveResult {
    move_entity_with_entities_full(w, aabb, delta, step_height, on_ground, sneaking, &[])
}

/// [`move_entity_with`] with nearby entity collision candidates.
///
/// This is the geometry-only form for remote entities and physics tools. It preserves vanilla's
/// Y/X/Z order and step-up retry while appending the supplied entity boxes to the one sweep list.
pub fn move_entity_with_entities_full(
    w: &World,
    aabb: &mut Aabb,
    delta: DVec3,
    step_height: f32,
    on_ground: bool,
    sneaking: bool,
    entity_collisions: &[EntityCollisionCandidate],
) -> MoveResult {
    let mut outside_border = false;
    let mut context = SweepContext {
        entity_collisions,
        border: None,
        entity_position: DVec3::ZERO,
        outside_border: &mut outside_border,
    };
    let r = sweep_inner(
        w,
        aabb,
        delta,
        step_height,
        on_ground && sneaking,
        on_ground,
        &mut context,
    );
    MoveResult {
        applied: r.applied,
        on_ground: r.on_ground,
        collided_horizontally: r.collided_horizontally,
        collided_vertically: r.collided_vertically,
    }
}

/// The full `Entity.moveEntity`, including everything that needs entity state.
///
/// Mutates `s` the way vanilla mutates the entity: position, `motionX`/`motionZ` zeroing on a
/// horizontal collision, the `onGround`/`isCollided*` flags, `fallDistance` via
/// `updateFallState`, `onLanded` (the slime bounce), and `doBlockCollisions` (cobweb, soul
/// sand, slime). Returns the delta actually applied.
pub fn move_entity_full(w: &World, s: &mut PhysicsState, delta: DVec3) -> DVec3 {
    move_entity_full_with_entities(w, s, delta, &[])
}

/// [`move_entity_full`] with nearby entity collision candidates from the local entity tracker.
///
/// This owns the full `Entity.moveEntity` bookkeeping just like [`move_entity_full`]; only the
/// single collision list gathered for its sweep gains the entity lane.
pub fn move_entity_full_with_entities(
    w: &World,
    s: &mut PhysicsState,
    delta: DVec3,
    entity_collisions: &[EntityCollisionCandidate],
) -> DVec3 {
    let mut outside_border = false;
    move_entity_full_with_context(w, s, delta, entity_collisions, None, &mut outside_border)
}

/// [`move_entity_full_with_entities`] with a live world-border projection and the entity's
/// caller-owned `outsideBorder` latch.
///
/// The border is sampled by the client caller for the same game tick as the movement input.
/// `outside_border` must survive across ticks for this entity; vanilla's one-block hysteresis is
/// otherwise lost at the physical edge. Passing a border-aware context changes only the collision
/// list source; all movement bookkeeping and the legacy no-border APIs remain shared.
pub fn move_entity_full_with_entities_and_border(
    w: &World,
    s: &mut PhysicsState,
    delta: DVec3,
    entity_collisions: &[EntityCollisionCandidate],
    border: CollisionBorder,
    outside_border: &mut bool,
) -> DVec3 {
    move_entity_full_with_context(w, s, delta, entity_collisions, Some(border), outside_border)
}

/// Shared implementation for the legacy no-border and border-aware movement entry points.
pub(crate) fn move_entity_full_with_context(
    w: &World,
    s: &mut PhysicsState,
    delta: DVec3,
    entity_collisions: &[EntityCollisionCandidate],
    border: Option<CollisionBorder>,
    outside_border: &mut bool,
) -> DVec3 {
    // java: Entity.moveEntity:600 — the very first branch. A `noClip` entity (only a spectator,
    // client-side) offsets its box and returns: no sweep, no flags, no `updateFallState`, no
    // `doBlockCollisions`, and not even the cobweb scaling, which lives in the `else`.
    if s.no_clip {
        let bb = s.aabb().offset(delta.x, delta.y, delta.z);
        s.set_from_aabb(&bb);
        return delta;
    }

    let mut x = delta.x;
    let mut y = delta.y;
    let mut z = delta.z;

    // java: Entity.moveEntity:607 — `d0`/`d1`/`d2`, the position before anything moves. The walk
    // accumulators at the bottom are differences against these.
    let d0 = s.pos.x;
    let d1 = s.pos.y;
    let d2 = s.pos.z;

    // java: Entity.moveEntity:612 — a web scales the *requested* delta and zeroes motion, then
    // clears the flag so it only bites for one tick.
    if s.in_web {
        s.in_web = false;
        x *= 0.25;
        y *= 0.050_000_000_745_058_06;
        z *= 0.25;
        s.vel = DVec3::ZERO;
    }

    let was_on_ground = s.on_ground;
    let sneak_guard = was_on_ground && s.sneaking;
    let mut bb = s.aabb();
    let entity_position = s.pos;
    let mut context = SweepContext {
        entity_collisions,
        border,
        entity_position,
        outside_border,
    };
    let r = sweep_inner(
        w,
        &mut bb,
        DVec3::new(x, y, z),
        s.step_height,
        sneak_guard,
        was_on_ground,
        &mut context,
    );
    s.set_from_aabb(&bb);

    s.collided_horizontally = r.collided_horizontally;
    s.collided_vertically = r.collided_vertically;
    s.on_ground = r.on_ground;

    // java: Entity.moveEntity:822 — the block the entity is standing on, with a fence/wall/gate
    // fallback one block down when the sampled block is air.
    let mut ground_pos = BlockPos::new(
        floor_f64(s.pos.x),
        floor_f64(s.pos.y - 0.200_000_002_980_232_24),
        floor_f64(s.pos.z),
    );
    let mut ground = w.block_at(ground_pos);
    if block_props(ground).material == Material::Air {
        let below = w.block_at(ground_pos.down());
        let class = block_info(block_props(below).block).java_class;
        if class == "net.minecraft.block.BlockFence"
            || class == "net.minecraft.block.BlockWall"
            || class == "net.minecraft.block.BlockFenceGate"
        {
            ground = below;
            ground_pos = ground_pos.down();
        }
    }

    update_fall_state(w, s, r.applied.y, r.on_ground, ground, ground_pos);

    if r.requested.x != r.applied.x {
        s.vel.x = 0.0;
    }
    if r.requested.z != r.applied.z {
        s.vel.z = 0.0;
    }
    if r.requested.y != r.applied.y {
        on_landed(s, ground);
    }

    // java: Entity.moveEntity:856 — `canTriggerWalking() && !flag && ridingEntity == null`,
    // where `flag` is the sneak edge-guard. `EntityPlayer.canTriggerWalking()` (`:2205`) is
    // **`!capabilities.isFlying`**, not `Entity`'s plain `true`, so a flying player accumulates
    // no walked distance, makes no footsteps, and never gets the three-argument
    // `onEntityCollidedWithBlock` — you can hover over a slime block without being damped.
    // A mounted player's own travel (`Entity.updateRidden` -> `onUpdate`) is excluded by the
    // `ridingEntity == null` term: no walked distance, footsteps or standing callback (H2).
    if !sneak_guard && !s.flying && !s.riding {
        // java: Entity.moveEntity:867 — the block you are *standing on* gets the
        // **three-argument** `onEntityCollidedWithBlock`, from inside this same branch.
        if s.on_ground {
            on_stood_on(s, ground);
            if matches!(ground.block().0, 73 | 74) {
                s.ground_contact = Some(GroundContact {
                    pos: ground_pos,
                    state: ground,
                });
            }
        }
        accumulate_walked_distance(w, s, DVec3::new(d0, d1, d2), ground, ground_pos);
    }

    do_block_collisions(w, s);

    r.applied
}

/// java: `Entity.moveEntity:858-892` — the `distanceWalked*` accumulators and the footstep they
/// trigger.
///
/// `distanceWalkedOnStepModified` is the one that fires the sound and it counts *vertical*
/// movement too, but only on a ladder (`if (block1 != Blocks.ladder) d13 = 0.0D;`), which is why
/// climbing makes footstep noises and falling does not. Both accumulate the `f32` field through
/// an `f64` addition and narrow once, exactly as vanilla writes it.
fn accumulate_walked_distance(
    w: &World,
    s: &mut PhysicsState,
    before: DVec3,
    ground: BlockState,
    ground_pos: BlockPos,
) {
    let d12 = s.pos.x - before.x;
    let mut d13 = s.pos.y - before.y;
    let d14 = s.pos.z - before.z;

    if block_info(block_props(ground).block).name != "minecraft:ladder" {
        d13 = 0.0;
    }

    s.distance_walked_modified = (f64::from(s.distance_walked_modified)
        + f64::from(mc_math::sqrt_d(d12 * d12 + d14 * d14)) * 0.6)
        as f32;
    s.distance_walked_on_step_modified = (f64::from(s.distance_walked_on_step_modified)
        + f64::from(mc_math::sqrt_d(d12 * d12 + d13 * d13 + d14 * d14)) * 0.6)
        as f32;

    if s.distance_walked_on_step_modified > s.next_step_distance as f32
        && block_props(ground).material != Material::Air
    {
        s.next_step_distance = s.distance_walked_on_step_modified as i32 + 1;

        if s.in_water {
            // java: `sqrt(mx² * 0.2F + my² + mz² * 0.2F) * 0.35F`, clamped to 1. The pitch
            // vanilla picks is RNG-driven and therefore not reproducible here; only the volume
            // is a function of state, so only the volume travels.
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

        // java: `Entity.playStepSound` (`entity/Entity.java:986`) — a snow layer on top wins,
        // and a liquid without one is silent.
        let above = w.block_at(ground_pos.up());
        if block_info(block_props(above).block).name == "minecraft:snow_layer" {
            s.step_sound = Some(StepSound {
                pos: ground_pos,
                state: above,
            });
        } else if !block_props(ground).material.is_liquid() {
            s.step_sound = Some(StepSound {
                pos: ground_pos,
                state: ground,
            });
        }
    }
}

/// java: `Block.onLanded` — zeroes `motionY`, except on a slime block, which reflects it.
fn on_landed(s: &mut PhysicsState, ground: BlockState) {
    let info = block_info(block_props(ground).block);
    if info.java_class == "net.minecraft.block.BlockSlime" {
        // java: BlockSlime.onLanded:44
        if s.sneaking {
            s.vel.y = 0.0;
        } else if s.vel.y < 0.0 {
            s.vel.y = -s.vel.y;
        }
    } else {
        s.vel.y = 0.0;
    }
}

/// java: `EntityLivingBase.updateFallState` -> `Entity.updateFallState`.
///
/// The living override re-runs `handleWaterMovement` when not already in water — that is how
/// stepping into water mid-move cancels the fall — and then falls through to the base version.
/// The particle burst in between is `!worldObj.isRemote` only, so the client never runs it.
fn update_fall_state(
    w: &World,
    s: &mut PhysicsState,
    y: f64,
    on_ground: bool,
    ground: BlockState,
    _pos: BlockPos,
) {
    if !s.in_water {
        crate::fluid::handle_water_movement(w, s);
    }

    if on_ground {
        if s.fall_distance > 0.0 {
            on_fallen_upon(s, ground);
            s.fall_distance = 0.0;
        }
    } else if y < 0.0 {
        s.fall_distance = (f64::from(s.fall_distance) - y) as f32;
    }
}

/// java: `Block.onFallenUpon` (which just calls `Entity.fall(distance, 1.0F)`), plus
/// `BlockSlime`'s override, which passes a `0.0F` multiplier so slime never hurts.
fn on_fallen_upon(s: &mut PhysicsState, ground: BlockState) {
    // java: `EntityPlayer.fall` (`entity/player/EntityPlayer.java:1929`) is the whole method
    // wrapped in `if (!this.capabilities.allowFlying)`. A creative player — or anyone a server
    // has handed the fly bit to — takes no fall damage at all, and the *client* knows it, so the
    // prediction has to know it too. Found by the `creative_fly` differential fixture, which
    // landed after a long descent and predicted 23 damage the game did not deal.
    if s.allow_flying {
        return;
    }
    let info = block_info(block_props(ground).block);
    let multiplier = if info.java_class == "net.minecraft.block.BlockSlime" && !s.sneaking {
        0.0f32
    } else {
        1.0f32
    };
    // java: `EntityLivingBase.fall` (`:1151`) subtracts `amplifier + 1` for Jump Boost. That
    // read is `isPotionActive`, not an attribute, so — unlike Speed — it really does run on the
    // client from `S1DPacketEntityEffect`'s own map.
    let jump_boost = s.jump_boost.map_or(0.0, |a| f32::from(a) + 1.0);
    let damage = fall_damage_with_jump_boost(s.fall_distance, multiplier, jump_boost);
    if damage > 0 {
        s.predicted_fall_damage = Some(damage);
    }
}

/// java: `EntityLivingBase.fall` — `MathHelper.ceiling_float_int((distance - 3.0F - f) * mult)`
/// where `f` is the Jump Boost amplifier plus one, or zero without the effect.
///
/// Potion effects are not modelled by this crate, so `f` is zero; a caller that tracks them can
/// call [`fall_damage_with_jump_boost`].
#[must_use]
pub fn fall_damage(distance: f32, damage_multiplier: f32) -> u32 {
    fall_damage_with_jump_boost(distance, damage_multiplier, 0.0)
}

/// [`fall_damage`] with an explicit Jump Boost term (`amplifier + 1`, or `0.0` for none).
#[must_use]
pub fn fall_damage_with_jump_boost(distance: f32, damage_multiplier: f32, jump_boost: f32) -> u32 {
    let value = (distance - 3.0 - jump_boost) * damage_multiplier;
    // java: MathHelper.ceiling_float_int
    let i = value as i32;
    let ceil = if value > i as f32 { i + 1 } else { i };
    u32::try_from(ceil).unwrap_or(0)
}

/// java: `Entity.doBlockCollisions` -> `Block.onEntityCollidedWithBlock` for every block the
/// (slightly shrunk) bounding box overlaps.
///
/// Only the two overlap-driven effects that change movement are modelled: cobwebs set
/// `isInWeb`, and soul sand scales horizontal motion by `0.4`. Slime is *not* here — it
/// overrides the other overload; see [`on_stood_on`]. Everything else vanilla does here (fire
/// damage, portals, pressure plates, tripwire) is either server-side or not physics.
fn do_block_collisions(w: &World, s: &mut PhysicsState) {
    let bb = s.aabb();
    let from = BlockPos::new(
        floor_f64(bb.min.x + 0.001),
        floor_f64(bb.min.y + 0.001),
        floor_f64(bb.min.z + 0.001),
    );
    let to = BlockPos::new(
        floor_f64(bb.max.x - 0.001),
        floor_f64(bb.max.y - 0.001),
        floor_f64(bb.max.z - 0.001),
    );

    // java: `Entity.doBlockCollisions:958` — the whole triple loop is inside
    // `if (this.worldObj.isAreaLoaded(blockpos, blockpos1))`.
    if !crate::is_area_loaded(w, from, to) {
        return;
    }

    for x in from.x..=to.x {
        for y in from.y..=to.y {
            for z in from.z..=to.z {
                on_collided_with(s, w.block_at(BlockPos::new(x, y, z)));
            }
        }
    }
}

/// java: the **three-argument** `Block.onEntityCollidedWithBlock(World, BlockPos, Entity)`.
///
/// `Block` declares two unrelated overloads of this name (`Block.java:859` and `Block.java:969`)
/// and **neither delegates to the other** — both base bodies are empty. Which one a block
/// overrides decides when it fires, and getting that backwards is invisible in a hand-written
/// test:
///
/// | block | overload | fires when |
/// |---|---|---|
/// | `BlockSlime` | 3-arg | you are *standing on* it |
/// | `BlockWeb`, `BlockSoulSand`, `BlockCactus` | 4-arg | your box *overlaps* it |
///
/// That is why soul sand only slows you once you have actually sunk onto its `0.875`-tall
/// collision box, and not while you are still walking on the full block beside it.
fn on_stood_on(s: &mut PhysicsState, state: BlockState) {
    if block_info(block_props(state).block).java_class == "net.minecraft.block.BlockSlime" {
        // java: BlockSlime.onEntityCollidedWithBlock:59
        if s.vel.y.abs() < 0.1 && !s.sneaking {
            let d0 = 0.4 + s.vel.y.abs() * 0.2;
            s.vel.x *= d0;
            s.vel.z *= d0;
        }
    }
}

/// java: the **four-argument** `Block.onEntityCollidedWithBlock(World, BlockPos, IBlockState,
/// Entity)`, called by `doBlockCollisions` for every block the box overlaps. See [`on_stood_on`]
/// for why the two overloads are not interchangeable.
fn on_collided_with(s: &mut PhysicsState, state: BlockState) {
    match block_info(block_props(state).block).java_class {
        // java: BlockWeb.onEntityCollidedWithBlock -> Entity.setInWeb, which also zeroes
        // `fallDistance` — a web catches you mid-fall.
        "net.minecraft.block.BlockWeb" => {
            // java: EntityPlayer.setInWeb:1979 — unlike Entity.setInWeb, a player ignores
            // webs while capabilities.isFlying. This is a player physics state, so preserve
            // both the flag and fall distance in that case.
            if !s.flying {
                s.in_web = true;
                s.fall_distance = 0.0;
            }
        }
        // java: BlockSoulSand.onEntityCollidedWithBlock:29
        "net.minecraft.block.BlockSoulSand" => {
            s.vel.x *= 0.4;
            s.vel.z *= 0.4;
        }
        // java: BlockCactus.onEntityCollidedWithBlock:121 calls `attackEntityFrom(DamageSource
        // .cactus, 1.0F)` without its own `isRemote` guard, but EntityLivingBase's client-side
        // attack path immediately returns false. Retain this contact bit for the differential
        // trace only; it must not become predicted client damage or alter cactus movement.
        "net.minecraft.block.BlockCactus" => {
            s.touched_cactus = true;
        }
        // java: BlockPortal.onEntityCollidedWithBlock:197 sets `Entity.inPortal`, even on the
        // client, which drives EntityPlayerSP's time-in-portal ramp and screen overlay.
        // BlockEndPortal deliberately does *not*: its dimension-transfer body is entirely
        // server-side (`if (!worldIn.isRemote)`).
        // `BlockPortal.onEntityCollidedWithBlock:197-202` only calls `setPortal` when
        // `ridingEntity == null && riddenByEntity == null`, so a mounted player's own travel
        // neither refreshes the cooldown nor sets the flag (H2).
        "net.minecraft.block.BlockPortal" if s.riding => {}
        "net.minecraft.block.BlockPortal" => {
            if s.portal_cooldown > 0 {
                // EntityPlayer overrides Entity.getPortalCooldown: the local player's
                // cooldown is ten ticks, not the base entity's 300-tick value.
                s.portal_cooldown = 10;
            } else {
                s.in_portal = true;
            }
        }
        _ => {}
    }
}

/// The requested-vs-applied pair the caller needs to reproduce vanilla's `d3 != x` tests.
struct Sweep {
    requested: DVec3,
    applied: DVec3,
    on_ground: bool,
    collided_horizontally: bool,
    collided_vertically: bool,
}

/// The caller-owned context shared by every collision query in one Java `moveEntity` sweep.
/// Grouping these related inputs keeps the geometric entry point readable without changing any
/// of the query ordering or the lifetime of the outside-border latch.
struct SweepContext<'a> {
    entity_collisions: &'a [EntityCollisionCandidate],
    border: Option<CollisionBorder>,
    entity_position: DVec3,
    outside_border: &'a mut bool,
}

/// Gathers one movement sweep's collision list while preserving the legacy block/entity path.
///
/// `entity_position` is intentionally separate from the mutable AABB: Java's entity position
/// fields are not reset from the box until `moveEntity` finishes, so every query in one sweep
/// sees the same position for `World.isInsideBorder`.
fn collision_boxes_for_move(
    w: &World,
    mask: &Aabb,
    context: &mut SweepContext<'_>,
    out: &mut Vec<Aabb>,
) {
    if let Some(border) = context.border {
        collision_boxes_in_with_border_and_entities(
            w,
            mask,
            border,
            context.entity_position.x,
            context.entity_position.z,
            context.outside_border,
            context.entity_collisions,
            out,
        );
    } else {
        collision_boxes_in_with_entities(w, mask, context.entity_collisions, out);
    }
}

/// The geometric core of `Entity.moveEntity`: the sneak probes, the Y/X/Z sweep, and the
/// step-up retry. Takes and returns only geometry, so it can be reused for any entity.
fn sweep_inner(
    w: &World,
    bb: &mut Aabb,
    delta: DVec3,
    step_height: f32,
    sneak_guard: bool,
    was_on_ground: bool,
    context: &mut SweepContext<'_>,
) -> Sweep {
    let mut x = delta.x;
    let mut y = delta.y;
    let mut z = delta.z;
    // java: d3/d4/d5 — the *requested* delta, captured before the sneak probes narrow it.
    let mut d3 = x;
    let d4 = y;
    let mut d5 = z;

    let mut scratch: Vec<Aabb> = Vec::new();

    if sneak_guard {
        // java: Entity.moveEntity:632 — three loops that shrink the step until the box, one
        // block lower, still overlaps something. `d3`/`d5` track the last *accepted* value,
        // which is what makes the horizontal-collision test below fire.
        let d6 = SNEAK_PROBE_STEP;

        while x != 0.0 && {
            scratch.clear();
            collision_boxes_for_move(w, &bb.offset(x, -1.0, 0.0), context, &mut scratch);
            scratch.is_empty()
        } {
            if x < d6 && x >= -d6 {
                x = 0.0;
            } else if x > 0.0 {
                x -= d6;
            } else {
                x += d6;
            }
            d3 = x;
        }

        while z != 0.0 && {
            scratch.clear();
            collision_boxes_for_move(w, &bb.offset(0.0, -1.0, z), context, &mut scratch);
            scratch.is_empty()
        } {
            if z < d6 && z >= -d6 {
                z = 0.0;
            } else if z > 0.0 {
                z -= d6;
            } else {
                z += d6;
            }
            d5 = z;
        }

        while x != 0.0 && z != 0.0 && {
            scratch.clear();
            collision_boxes_for_move(w, &bb.offset(x, -1.0, z), context, &mut scratch);
            scratch.is_empty()
        } {
            if x < d6 && x >= -d6 {
                x = 0.0;
            } else if x > 0.0 {
                x -= d6;
            } else {
                x += d6;
            }
            d3 = x;

            if z < d6 && z >= -d6 {
                z = 0.0;
            } else if z > 0.0 {
                z -= d6;
            } else {
                z += d6;
            }
            d5 = z;
        }
    }

    // java: one list, gathered once from the *swept* box, and reused for all three axes.
    let mut list1: Vec<Aabb> = Vec::new();
    collision_boxes_for_move(w, &bb.add_coord(x, y, z), context, &mut list1);
    let saved = *bb;

    // -- Y, then X, then Z. H7. --------------------------------------------------------
    for b in &list1 {
        y = b.calculate_y_offset(bb, y);
    }
    *bb = bb.offset(0.0, y, 0.0);
    // java: `boolean flag1 = this.onGround || d4 != y && d4 < 0.0D;` — the step-up gate.
    let flag1 = was_on_ground || (d4 != y && d4 < 0.0);

    for b in &list1 {
        x = b.calculate_x_offset(bb, x);
    }
    *bb = bb.offset(x, 0.0, 0.0);

    for b in &list1 {
        z = b.calculate_z_offset(bb, z);
    }
    *bb = bb.offset(0.0, 0.0, z);

    if step_height > 0.0 && flag1 && (d3 != x || d5 != z) {
        step_up(
            w,
            bb,
            saved,
            &mut x,
            &mut y,
            &mut z,
            d3,
            d5,
            step_height,
            context,
        );
    }

    let collided_horizontally = d3 != x || d5 != z;
    let collided_vertically = d4 != y;

    Sweep {
        requested: DVec3::new(d3, d4, d5),
        applied: DVec3::new(x, y, z),
        on_ground: collided_vertically && d4 < 0.0,
        collided_horizontally,
        collided_vertically,
    }
}

/// java: `Entity.moveEntity:721` — the step-up retry.
///
/// Two candidate resolutions are computed from the pre-sweep box: one that lifts by the step
/// height *before* moving horizontally against a box widened by the requested delta, and one
/// that lifts against the un-widened box. Whichever travelled further in the XZ plane wins, and
/// the whole thing is discarded if the original sweep already went at least as far.
#[allow(clippy::too_many_arguments)]
fn step_up(
    w: &World,
    bb: &mut Aabb,
    saved: Aabb,
    x: &mut f64,
    y: &mut f64,
    z: &mut f64,
    d3: f64,
    d5: f64,
    step_height: f32,
    context: &mut SweepContext<'_>,
) {
    let d11 = *x;
    let d7 = *y;
    let d8 = *z;
    let axisalignedbb3 = *bb;
    *bb = saved;
    *y = f64::from(step_height);

    let mut list: Vec<Aabb> = Vec::new();
    collision_boxes_for_move(w, &bb.add_coord(d3, *y, d5), context, &mut list);

    // -- candidate A: sweep the box *widened by the horizontal request* upward first --------
    let mut aabb4 = *bb;
    let aabb5 = aabb4.add_coord(d3, 0.0, d5);
    let mut d9 = *y;
    for b in &list {
        d9 = b.calculate_y_offset(&aabb5, d9);
    }
    aabb4 = aabb4.offset(0.0, d9, 0.0);
    let mut d15 = d3;
    for b in &list {
        d15 = b.calculate_x_offset(&aabb4, d15);
    }
    aabb4 = aabb4.offset(d15, 0.0, 0.0);
    let mut d16 = d5;
    for b in &list {
        d16 = b.calculate_z_offset(&aabb4, d16);
    }
    aabb4 = aabb4.offset(0.0, 0.0, d16);

    // -- candidate B: sweep the plain box upward first --------------------------------------
    let mut aabb14 = *bb;
    let mut d17 = *y;
    for b in &list {
        d17 = b.calculate_y_offset(&aabb14, d17);
    }
    aabb14 = aabb14.offset(0.0, d17, 0.0);
    let mut d18 = d3;
    for b in &list {
        d18 = b.calculate_x_offset(&aabb14, d18);
    }
    aabb14 = aabb14.offset(d18, 0.0, 0.0);
    let mut d19 = d5;
    for b in &list {
        d19 = b.calculate_z_offset(&aabb14, d19);
    }
    aabb14 = aabb14.offset(0.0, 0.0, d19);

    let d20 = d15 * d15 + d16 * d16;
    let d10 = d18 * d18 + d19 * d19;

    if d20 > d10 {
        *x = d15;
        *z = d16;
        *y = -d9;
        *bb = aabb4;
    } else {
        *x = d18;
        *z = d19;
        *y = -d17;
        *bb = aabb14;
    }

    for b in &list {
        *y = b.calculate_y_offset(bb, *y);
    }
    *bb = bb.offset(0.0, *y, 0.0);

    // java: keep the original sweep if it already went at least as far horizontally.
    if d11 * d11 + d8 * d8 >= *x * *x + *z * *z {
        *x = d11;
        *y = d7;
        *z = d8;
        *bb = axisalignedbb3;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_data::{block_by_name, BlockState};
    use mc_math::DVec3;

    fn flat_world(y: i32) -> World {
        let mut w = World::new();
        let stone = BlockState::new(block_by_name("stone").unwrap(), 0);
        for x in -8..8 {
            for z in -8..8 {
                crate::scene::set_block(&mut w, BlockPos::new(x, y, z), stone);
            }
        }
        w
    }

    #[test]
    fn falling_onto_a_floor_stops_at_its_surface() {
        let w = flat_world(63);
        let mut bb = Aabb::from_bounds(-0.3, 66.0, -0.3, 0.3, 67.8, 0.3);
        let (applied, on_ground) = move_entity(&w, &mut bb, DVec3::new(0.0, -4.0, 0.0));
        assert!(on_ground);
        assert_eq!(bb.min.y, 64.0);
        assert_eq!(applied.y, -2.0);
    }

    #[test]
    fn grounded_redstone_ore_reports_the_exact_post_sweep_contact() {
        let pos = BlockPos::new(0, 63, 0);
        for name in ["redstone_ore", "lit_redstone_ore"] {
            let mut w = flat_world(63);
            let ore = BlockState::new(block_by_name(name).expect("redstone ore registry entry"), 0);
            crate::scene::set_block(&mut w, pos, ore);
            let mut s = PhysicsState::new_player(DVec3::new(0.5, 66.0, 0.5));
            s.vel.y = -1.0;

            move_entity_full(&w, &mut s, DVec3::new(0.0, -3.0, 0.0));

            assert_eq!(s.pos.y, 64.0);
            assert_eq!(
                s.ground_contact,
                Some(GroundContact { pos, state: ore }),
                "the standing callback must preserve the selected position and lit state"
            );
        }

        let w = flat_world(63);
        let mut airborne = PhysicsState::new_player(DVec3::new(0.5, 66.0, 0.5));
        move_entity_full(&w, &mut airborne, DVec3::new(0.0, -0.25, 0.0));
        assert_eq!(airborne.ground_contact, None);
    }

    #[test]
    fn a_wall_clips_the_horizontal_delta_and_reports_a_collision() {
        let mut w = flat_world(63);
        let stone = BlockState::new(block_by_name("stone").unwrap(), 0);
        for y in 64..67 {
            crate::scene::set_block(&mut w, BlockPos::new(0, y, 3), stone);
        }
        let mut bb = Aabb::from_bounds(-0.3, 64.0, -0.3, 0.3, 65.8, 0.3);
        let (applied, _) = move_entity(&w, &mut bb, DVec3::new(0.0, 0.0, 5.0));
        assert!((bb.max.z - 3.0).abs() < 1e-12, "stopped at the wall face");
        assert!(applied.z < 5.0);
    }

    #[test]
    fn border_aware_move_clips_at_the_current_edge_but_legacy_move_does_not() {
        let mut w = World::new();
        crate::scene::ensure_chunk(&mut w, mc_math::ChunkPos::new(0, 0));
        let border = CollisionBorder::new(0.0, 0.0, 4.0, 29_999_984);

        let mut bounded = PhysicsState::new_player(DVec3::new(1.5, 64.0, 0.5));
        bounded.on_ground = true;
        let mut outside_border = false;
        move_entity_full_with_entities_and_border(
            &w,
            &mut bounded,
            DVec3::new(1.0, 0.0, 0.0),
            &[],
            border,
            &mut outside_border,
        );
        let expected_x = 2.0 - f64::from(bounded.width / 2.0);
        assert!((bounded.pos.x - expected_x).abs() < 1.0e-12);
        assert!((bounded.aabb().max.x - 2.0).abs() < 1.0e-12);
        assert!(
            !outside_border,
            "the entity remains inside the hysteresis region"
        );

        // Negative control: the compatibility entry point has no border and must still cross the
        // same empty-world coordinate rather than accidentally inheriting the new wall.
        let mut unbounded = PhysicsState::new_player(DVec3::new(1.5, 64.0, 0.5));
        unbounded.on_ground = true;
        move_entity_full_with_entities(&w, &mut unbounded, DVec3::new(1.0, 0.0, 0.0), &[]);
        assert!((unbounded.pos.x - 2.5).abs() < 1.0e-12);
    }

    #[test]
    fn border_aware_move_latches_outside_and_requires_contracted_reentry() {
        let mut w = World::new();
        crate::scene::ensure_chunk(&mut w, mc_math::ChunkPos::new(0, 0));
        let border = CollisionBorder::new(0.0, 0.0, 4.0, 29_999_984);
        let mut s = PhysicsState::new_player(DVec3::new(3.1, 64.0, 0.5));
        let mut outside_border = false;

        // Physical edge is x=2; x=3.1 is beyond the inside latch's expanded edge at x=3.
        move_entity_full_with_entities_and_border(
            &w,
            &mut s,
            DVec3::ZERO,
            &[],
            border,
            &mut outside_border,
        );
        assert!(outside_border);

        // The outside latch contracts the re-entry edge to x=1. Merely crossing the physical
        // edge must not clear it; entering x<1 does.
        s.set_position(DVec3::new(1.5, 64.0, 0.5));
        move_entity_full_with_entities_and_border(
            &w,
            &mut s,
            DVec3::ZERO,
            &[],
            border,
            &mut outside_border,
        );
        assert!(outside_border);
        s.set_position(DVec3::new(0.5, 64.0, 0.5));
        move_entity_full_with_entities_and_border(
            &w,
            &mut s,
            DVec3::ZERO,
            &[],
            border,
            &mut outside_border,
        );
        assert!(!outside_border);
    }

    #[test]
    fn entity_boxes_clip_the_sweep_but_self_excluded_and_empty_candidates_do_not() {
        let w = World::new();
        let mover = Aabb::from_bounds(-0.3, 64.0, -0.3, 0.3, 65.8, 0.3);
        let boat = Aabb::from_bounds(-0.5, 64.0, 1.0, 0.5, 65.0, 2.0);

        let mut blocked = mover;
        let (applied, _) = move_entity_with_entities(
            &w,
            &mut blocked,
            DVec3::new(0.0, 0.0, 3.0),
            &[EntityCollisionCandidate::new(boat, Some(boat))],
        );
        assert_eq!(
            blocked.max.z, 1.0,
            "the entity box is in the shared sweep list"
        );
        assert_eq!(applied.z, 0.7);

        for candidate in [
            EntityCollisionCandidate::new(boat, None),
            EntityCollisionCandidate::new(boat, Some(boat))
                .with_relation(crate::EntityCollisionRelation::SelfEntity),
            EntityCollisionCandidate::new(boat, Some(boat)).excluded(),
        ] {
            let mut ignored = mover;
            let (applied, _) = move_entity_with_entities(
                &w,
                &mut ignored,
                DVec3::new(0.0, 0.0, 3.0),
                &[candidate],
            );
            assert_eq!(applied.z, 3.0, "only a selected, non-null box collides");
        }
    }

    #[test]
    fn riding_and_ridden_entity_candidates_are_not_world_query_exclusions() {
        let w = World::new();
        let mover = Aabb::from_bounds(-0.3, 64.0, -0.3, 0.3, 65.8, 0.3);
        let obstacle = Aabb::from_bounds(-0.5, 64.0, 1.0, 0.5, 65.0, 2.0);

        for relation in [
            crate::EntityCollisionRelation::Riding,
            crate::EntityCollisionRelation::RiddenBy,
        ] {
            let mut bb = mover;
            let (applied, _) = move_entity_with_entities(
                &w,
                &mut bb,
                DVec3::new(0.0, 0.0, 3.0),
                &[EntityCollisionCandidate::new(obstacle, Some(obstacle)).with_relation(relation)],
            );
            assert_eq!(
                applied.z, 0.7,
                "{relation:?} remains a World.java candidate"
            );
        }
    }

    #[test]
    fn a_half_slab_is_stepped_onto_without_jumping() {
        let mut w = flat_world(63);
        let slab = BlockState::new(block_by_name("stone_slab").unwrap(), 0);
        for x in -8..8 {
            for z in 2..8 {
                crate::scene::set_block(&mut w, BlockPos::new(x, 64, z), slab);
            }
        }
        let mut bb = Aabb::from_bounds(-0.3, 64.0, -0.3, 0.3, 65.8, 0.3);
        // A downward component is required: vanilla's step-up needs the Y sweep to have
        // clipped downward this tick (`d4 != y && d4 < 0`).
        // 20 ticks of 0.2 gets from z=0 to z=4, past the slab at z=2 but well short of the
        // edge of `flat_world`'s 16x16 platform.
        for _ in 0..20 {
            move_entity(&w, &mut bb, DVec3::new(0.0, -0.0784, 0.2));
        }
        assert_eq!(bb.min.y, 64.5, "climbed onto the slab's 0.5-high surface");
        assert!(bb.min.z > 2.0, "and kept going: {}", bb.min.z);
    }

    #[test]
    fn a_full_block_is_not_stepped_onto() {
        let mut w = flat_world(63);
        let stone = BlockState::new(block_by_name("stone").unwrap(), 0);
        for x in -8..8 {
            crate::scene::set_block(&mut w, BlockPos::new(x, 64, 2), stone);
        }
        let mut bb = Aabb::from_bounds(-0.3, 64.0, -0.3, 0.3, 65.8, 0.3);
        for _ in 0..20 {
            move_entity(&w, &mut bb, DVec3::new(0.0, -0.0784, 0.2));
        }
        assert_eq!(bb.min.y, 64.0, "0.6 step height cannot climb 1.0");
        assert!(bb.max.z <= 2.0 + 1e-12);
    }

    #[test]
    fn ceiling_float_int_matches_javas_definition() {
        assert_eq!(fall_damage(3.0, 1.0), 0);
        assert_eq!(fall_damage(4.0, 1.0), 1);
        assert_eq!(fall_damage(4.5, 1.0), 2);
        assert_eq!(fall_damage(23.0, 1.0), 20);
        assert_eq!(fall_damage(100.0, 0.0), 0, "a slime block never hurts");
        assert_eq!(fall_damage_with_jump_boost(6.0, 1.0, 2.0), 1);
    }

    #[test]
    fn fall_damage_keeps_signed_jump_boost_terms_and_none_is_ordinary() {
        // java: `fall()` subtracts `amplifier + 1` as a float. A negative signed amplifier
        // therefore increases the damage distance; an unsigned/clamped mutant would produce
        // the level-I/no-effect result instead.
        let term = |amplifier: i8| f32::from(amplifier) + 1.0;
        assert_eq!(fall_damage_with_jump_boost(20.0, 1.0, term(127)), 0);
        assert_eq!(fall_damage_with_jump_boost(20.0, 1.0, term(-128)), 144);
        assert_eq!(fall_damage_with_jump_boost(20.0, 1.0, term(-1)), 17);
        assert_eq!(fall_damage(20.0, 1.0), 17, "no-effect control");
    }

    #[test]
    fn an_extending_piston_shoves_the_player_0_5625_per_tick() {
        // java: `TileEntityPiston.launchWithSlimeBlock(progress, progress - lastProgress +
        // 0.0625F)` — `0.5 + 0.0625` on each of the two moving ticks — calling
        // `entity.moveEntity(...)` on everything inside the moving block's box. It is **not**
        // `isRemote`-gated, so on a vanilla client this is what moves the local player.
        //
        // This drives the real chain: `World::apply_block_action` runs the client half of
        // `onBlockEventReceived`, `World::tick_pistons` runs `update()`, and the returned
        // `PistonPush` is applied exactly as `mc-client`'s `tick_piston_tile_entities` does.
        let mut w = flat_world(63);
        let piston = BlockPos::new(0, 64, 0);
        let pushed = BlockPos::new(1, 64, 0);
        // meta 5 is EAST with EXTENDED clear.
        crate::scene::set_block(
            &mut w,
            piston,
            BlockState::new(block_by_name("piston").unwrap(), 5),
        );
        crate::scene::set_block(
            &mut w,
            pushed,
            BlockState::new(block_by_name("stone").unwrap(), 0),
        );
        assert!(w.apply_block_action(piston, 33, 0, 5).is_some(), "it fires");

        // The player stands in the cell the stone is being pushed into.
        let mut s = PhysicsState::new_player(DVec3::new(2.5, 64.0, 0.5));
        let start = s.pos.x;

        // Tick 1: progress 0 -> 0.5. The moving block's box has swept half of x=2..3.
        let pushes = w.tick_pistons();
        let mut applied = 0usize;
        for push in pushes {
            if !s.aabb().intersects_with(&push.area) {
                continue;
            }
            applied += 1;
            let d = f64::from(push.amount);
            let (ox, oy, oz) = push.facing.offset();
            move_entity_full(
                &w,
                &mut s,
                DVec3::new(d * f64::from(ox), d * f64::from(oy), d * f64::from(oz)),
            );
        }
        assert_eq!(applied, 1, "one moving block reaches the player");
        assert!(
            (s.pos.x - start - 0.5625).abs() < 1e-9,
            "0.5 + 0.0625, from {start} to {}",
            s.pos.x
        );
    }

    #[test]
    fn a_cobweb_scales_the_delta_and_kills_motion() {
        let mut w = flat_world(63);
        let web = BlockState::new(block_by_name("web").unwrap(), 0);
        crate::scene::set_block(&mut w, BlockPos::new(0, 64, 0), web);
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5));
        s.vel = DVec3::new(0.5, 0.0, 0.5);
        // The first move enters the web and sets the flag; the second is slowed.
        move_entity_full(&w, &mut s, DVec3::new(0.0, 0.0, 0.0));
        assert!(s.in_web);
        let applied = move_entity_full(&w, &mut s, DVec3::new(0.4, 0.0, 0.0));
        assert!((applied.x - 0.1).abs() < 1e-12, "0.25x the requested delta");
        assert_eq!(
            s.vel,
            DVec3::ZERO,
            "and motion is reset every tick you are inside"
        );
        // The flag is consumed at the top of `moveEntity` and re-set by `doBlockCollisions`
        // at the bottom, for as long as the box still overlaps a web.
        assert!(s.in_web);
        s.set_position(DVec3::new(8.5, 64.0, 8.5));
        move_entity_full(&w, &mut s, DVec3::ZERO);
        assert!(!s.in_web, "the flag clears once you leave");
    }

    #[test]
    fn a_flying_player_ignores_cobwebs_until_flight_stops() {
        let mut w = flat_world(63);
        let web = BlockState::new(block_by_name("web").unwrap(), 0);
        crate::scene::set_block(&mut w, BlockPos::new(0, 64, 0), web);
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5));
        s.flying = true;
        s.fall_distance = 7.0;

        move_entity_full(&w, &mut s, DVec3::ZERO);
        assert!(!s.in_web, "EntityPlayer.setInWeb is flight-gated");
        assert_eq!(s.fall_distance, 7.0, "ignored webs do not catch a fall");

        // Negative control: the same player and overlap must take the ordinary Entity path as
        // soon as flight is disabled. This catches both a missing and an inverted flight gate.
        s.flying = false;
        move_entity_full(&w, &mut s, DVec3::ZERO);
        assert!(s.in_web);
        assert_eq!(s.fall_distance, 0.0);
    }

    #[test]
    fn soul_sand_scales_horizontal_motion() {
        let mut w = flat_world(63);
        let soul = BlockState::new(block_by_name("soul_sand").unwrap(), 0);
        crate::scene::set_block(&mut w, BlockPos::new(0, 64, 0), soul);
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5));
        s.vel = DVec3::new(0.5, 0.0, 0.25);
        move_entity_full(&w, &mut s, DVec3::ZERO);
        assert!((s.vel.x - 0.2).abs() < 1e-12);
        assert!((s.vel.z - 0.1).abs() < 1e-12);
    }

    #[test]
    fn cactus_contact_is_trace_only_and_does_not_apply_client_damage_or_slowdown() {
        let mut w = flat_world(63);
        let cactus = BlockState::new(block_by_name("cactus").unwrap(), 0);
        crate::scene::set_block(&mut w, BlockPos::new(0, 64, 0), cactus);
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5));
        s.on_ground = true;
        s.vel = DVec3::new(0.25, 0.0, -0.5);

        move_entity_full(&w, &mut s, DVec3::ZERO);

        assert!(
            s.touched_cactus,
            "the contact callback remains visible to the trace harness"
        );
        assert_eq!(
            s.predicted_fall_damage, None,
            "client attackEntityFrom is a no-op"
        );
        assert_eq!(
            s.vel,
            DVec3::new(0.25, 0.0, -0.5),
            "cactus has no slowdown callback"
        );
    }

    #[test]
    fn only_a_nether_portal_sets_the_client_portal_overlay_signal() {
        let mut w = flat_world(63);
        // `BlockPortal` only registers its X/Z axis metadata (1/2); raw meta zero decodes as
        // air, so use the real X-axis state here.
        let portal = BlockState::new(block_by_name("portal").unwrap(), 1);
        crate::scene::set_block(&mut w, BlockPos::new(0, 64, 0), portal);
        assert_eq!(
            block_info(block_props(w.block_at(BlockPos::new(0, 64, 0))).block).java_class,
            "net.minecraft.block.BlockPortal"
        );
        let mut nether = PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5));
        move_entity_full(&w, &mut nether, DVec3::ZERO);
        assert!(
            nether.in_portal,
            "BlockPortal is the sole client-side producer of Entity.inPortal"
        );

        let mut cooling_down = PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5));
        cooling_down.portal_cooldown = 12;
        move_entity_full(&w, &mut cooling_down, DVec3::ZERO);
        assert!(!cooling_down.in_portal);
        assert_eq!(
            cooling_down.portal_cooldown, 10,
            "positive-cooldown portal contact refreshes getPortalCooldown without signaling"
        );

        // Negative control: BlockEndPortal transfers dimensions only under !world.isRemote;
        // it must not start the Nether portal overlay animation on a client.
        let end = BlockState::new(block_by_name("end_portal").unwrap(), 0);
        crate::scene::set_block(&mut w, BlockPos::new(0, 64, 0), end);
        let mut end_player = PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5));
        move_entity_full(&w, &mut end_player, DVec3::ZERO);
        assert!(
            !end_player.in_portal,
            "BlockEndPortal has no client-side Entity.inPortal write"
        );
    }

    #[test]
    fn landing_on_slime_reflects_motion_and_deals_no_damage() {
        let mut w = flat_world(63);
        let slime = BlockState::new(block_by_name("slime").unwrap(), 0);
        for x in -2..2 {
            for z in -2..2 {
                crate::scene::set_block(&mut w, BlockPos::new(x, 63, z), slime);
            }
        }
        let mut s = PhysicsState::new_player(DVec3::new(0.5, 66.0, 0.5));
        s.fall_distance = 20.0;
        s.vel = DVec3::new(0.0, -1.0, 0.0);
        move_entity_full(&w, &mut s, DVec3::new(0.0, -3.0, 0.0));
        assert_eq!(s.pos.y, 64.0);
        assert!(s.vel.y > 0.0, "bounced");
        assert_eq!(s.predicted_fall_damage, None);
    }
}
