//! The client's non-living entity rendering seam.
//!
//! java: the half of `RenderManager`'s `entityRenderMap` that maps to a plain `Render` subclass
//! rather than a `RendererLivingEntity` — `RenderEntityItem`, `RenderXPOrb`, `RenderArrow`,
//! `RenderSnowball` (six item types), `RenderPotion`, `RenderFireball` (two scales),
//! `RenderTNTPrimed`, `RenderFallingBlock`, `RenderPainting`, `RenderWitherSkull`.
//!
//! # Why this is a second pass and not an arm of `upload_entities`
//!
//! `Client::upload_entities` walks the tracker, asks
//! [`mc_render::entity::renderer_for`] for a descriptor, and `continue`s when there is none. That
//! table is `RendererLivingEntity`-shaped: one [`ModelKind`](mc_render::entity::ModelKind), one
//! texture, one `preRenderCallback` scale, a death spin and a layer list — and every entity it
//! draws goes through [`mc_render::entity::entity_matrix`], which bakes in the `180 - yaw` turn,
//! the `(-1, -1, 1)` flip and the `-1.5078125` model lift that only `RendererLivingEntity`
//! applies.
//!
//! None of the renderers here fit that shape. An arrow has no model at all, a dropped item is a
//! baked *item* model drawn five times with a jitter, primed TNT is a block model with a swell
//! curve, and a painting is a tiled cube whose size comes from a 26-entry art table. Forcing them
//! through the living descriptor would mean inventing a "model" for each and then subtracting the
//! transform back out, which is A02-12 in reverse. So they get their own walk of the same
//! tracker, and the geometry lands in the same [`EntityPhase`](mc_render::entity::EntityPhase) —
//! which is exactly how vanilla is arranged too, since `RenderGlobal.renderEntities` makes one
//! pass and `RenderManager` dispatches per class inside it.
//!
//! # What has to come off the spawn packet
//!
//! Four of these renderers need a value that appears **once**, in the spawn packet, and is never
//! repeated on the wire. They are kept as small marker components on
//! the ECS entity, which is what `mc_world`'s `hecs` arena is for:
//!
//! | Component | Source | Used by |
//! |---|---|---|
//! | [`ObjectData`] | `S0EPacketSpawnObject.data` | falling-block state, splash-potion damage |
//! | [`XpValue`] | `S11PacketSpawnExperienceOrb.xp_value` | the orb's size/colour cell |
//! | [`Hanging`] | `S10PacketSpawnPainting.title`/`.facing` | the painting's art and wall |
//!
//! Two more are pure client-side tick state that the server never sends at all and that
//! [`mc_render::entity::LivingRenderState::ticks_existed`] already carries: primed TNT's fuse
//! (`80 - ticksExisted`, java `EntityTNTPrimed.fuse`) and the XP orb's colour phase
//! (`EntityXPOrb.xpColor`).

use mc_data::EntityType;
use mc_math::{Aabb, BlockPos, DVec3, Facing};
use mc_render::context::Gpu;
use mc_render::entity::objects::{self, Art, FishingLineAngler};
use mc_render::entity::tilepass::{BuiltinTileItem, TileModels};
use mc_render::entity::{
    item as dropped, AnimParams, EntityPhase, EntityVertex, LayerBlend, RenderManager,
};
use mc_render::{terrain::frustum::Frustum, Camera};
use mc_world::entity::{ItemFrameFacing, Position, Rotation, SplashPotionDamage};
use mc_world::World;

use crate::app::assets::GameAssets;
use crate::app::driver::RenderStack;
use crate::sync::entities::ItemCollectedSource;
use crate::sync::entities::{ArrowShake, TrackedEntity, TrackedKind};
use crate::sync::state::ClientState;

/// The render-only child created by `NetHandlerPlayClient.handleCollectItem`.
///
/// Java's `EntityPickupFX` keeps references to both entities while the source is removed from
/// `WorldClient`; Rust instead keeps the source values and a collector position pair.  This is
/// intentionally separate from the live tracker: the source must disappear from gameplay and
/// collision immediately, while its three render ticks remain visible.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PickupEffect {
    /// Values copied from the source before `EntityTracker::despawn`.
    pub source: ItemCollectedSource,
    /// Collector's previous/current feet positions for this effect's target lerp.
    pub target_previous: DVec3,
    /// Collector's current feet position for this effect's target lerp.
    pub target_current: DVec3,
    /// `EntityPickupFX.age`; starts at zero and is incremented once per client tick.
    pub age: u32,
}

impl PickupEffect {
    /// Construct a fresh pickup child. Java hard-codes a three-tick lifetime.
    pub(crate) fn new(
        source: ItemCollectedSource,
        target_previous: DVec3,
        target_current: DVec3,
    ) -> Self {
        Self {
            source,
            target_previous,
            target_current,
            age: 0,
        }
    }
}

/// `EntityPickupFX.onUpdate`: increment first, then kill exactly at `maxAge == 3`.
pub(crate) fn tick_pickup_effects(effects: &mut Vec<PickupEffect>) {
    for effect in effects.iter_mut() {
        effect.age = effect.age.saturating_add(1);
    }
    effects.retain(|effect| effect.age < 3);
}

/// The world-space position `EntityPickupFX.renderParticle` submits for one frame.
///
/// Java uses the source's captured position and a quadratic interpolation to the collector's
/// interpolated position, with a `+0.5F` Y lift.  The source is never sampled again after the
/// packet handler removes it.
#[must_use]
pub(crate) fn pickup_effect_position(effect: &PickupEffect, partial_ticks: f32) -> DVec3 {
    let partial = f64::from(partial_ticks);
    let target = effect.target_previous
        + (effect.target_current - effect.target_previous) * partial
        + DVec3::new(0.0, 0.5, 0.0);
    // Java stores this intermediate as a float (`float f`), then promotes `f * f` to double
    // for each source coordinate. Keep that rounding boundary; doing the whole calculation as
    // f64 moves a few ulps at non-integral partial ticks.
    let age = (effect.age as f32 + partial_ticks) / 3.0;
    let progress = f64::from(age * age);
    effect.source.position + (target - effect.source.position) * progress
}

// ---------------------------------------------------------------------------------------------
// Spawn-packet components
// ---------------------------------------------------------------------------------------------

/// `S0EPacketSpawnObject`'s type-specific `data` word.
///
/// java: `NetHandlerPlayClient.handleSpawnObject` (`:297-436`) uses `data` as a **constructor
/// argument** per object type — `70` is `Block.getStateById(data & 65535)` for the falling block,
/// `71` is `EnumFacing.getHorizontal(data)` for the item frame, `73` is the splash potion's
/// damage value, `60` the arrow's shooter id, `10` the minecart variant. Only the minecart case
/// was consumed before this existed; everything else was decoded off the wire and dropped at the
/// ECS boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectData(pub i32);

/// java: `S11PacketSpawnExperienceOrb.getXPValue` → `new EntityXPOrb(world, x, y, z, xpValue)`.
///
/// `EntityXPOrb.getTextureByXP` is the only consumer, and it is what makes a big drop visibly
/// bigger than a small one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XpValue(pub i32);

/// A painting's art and the wall it hangs on.
///
/// java: `NetHandlerPlayClient.handleSpawnPainting` (`:491`) →
/// `new EntityPainting(world, blockpos, EnumFacing.getHorizontal(dir), title)`, which resolves
/// the title against `EnumArt` and then runs `EntityHanging.updateFacingWithBoundingBox`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hanging {
    /// The resolved `EnumArt` row. See [`Art::by_title`].
    pub art: &'static Art,
    /// java: `EntityHanging.facingDirection` — always horizontal.
    pub facing: Facing,
}

/// The shared `Render.shouldRender` predicate for tracked entities.
///
/// The range test uses the entity's original box, while the frustum test uses Java's +/-2 block
/// fallback for a NaN or zero-sized box. `force_visible` is RenderGlobal's local-rider exemption
/// and the wither-skull fallback; `ignore_frustum` covers dragons and fishing bobbers.
#[must_use]
pub(crate) fn should_render_bounds(
    frustum: &Frustum,
    render_view_position: DVec3,
    position: DVec3,
    bounds: Aabb,
    render_distance_weight: f64,
    ignore_frustum: bool,
    force_visible: bool,
) -> bool {
    should_render_bounds_with_distance_base(
        frustum,
        render_view_position,
        position,
        bounds,
        render_distance_weight,
        None,
        ignore_frustum,
        force_visible,
    )
}

/// `EntityItemFrame.isInRangeToRenderDist`'s fixed-size range override at the culling boundary.
///
/// Item frames replace the `EntityHanging` 0.5 x 0.5 box with a thin 12 x 12 plate for picking,
/// frustum checks and rendering. Their Java distance override intentionally does not inspect that
/// plate: it uses a fixed `16.0 * 64.0 * renderDistanceWeight` range instead. `None` keeps the
/// ordinary `Entity.isInRangeToRenderDist` average-edge-length rule.
#[must_use]
#[allow(clippy::too_many_arguments)]
fn should_render_bounds_with_distance_base(
    frustum: &Frustum,
    render_view_position: DVec3,
    position: DVec3,
    bounds: Aabb,
    render_distance_weight: f64,
    render_distance_base: Option<f64>,
    ignore_frustum: bool,
    force_visible: bool,
) -> bool {
    if force_visible {
        return true;
    }
    let distance_squared = (position - render_view_position).length_squared();
    let in_render_range = match render_distance_base {
        Some(base) => {
            let range = base * 64.0 * render_distance_weight;
            distance_squared < range * range
        }
        None => bounds.is_in_render_range(distance_squared, render_distance_weight),
    };
    in_render_range
        && (ignore_frustum || {
            let render_box = bounds.render_box_or_fallback(position);
            frustum.contains_box(render_box.min, render_box.max)
        })
}

fn render_distance_base(entity_type: Option<EntityType>) -> Option<f64> {
    (entity_type == Some(EntityType::ItemFrame)).then_some(16.0)
}

/// `EntityOtherPlayerMP`'s constructor raises the public `Entity.renderDistanceWeight` field to
/// `10.0D` (the same field `EntityArrow` raises; see `mc-data`'s extractor note).
///
/// Players are not `EntityList` rows, so the extracted per-type weight table cannot carry this
/// value and [`TrackedKind::entity_type`] is `None` for them. Falling back to `Entity`'s `1.0`
/// culled every remote player at ~64 blocks; vanilla keeps the ordinary average-edge rule with
/// this weight, i.e. ~640 blocks for a standing player (0.6 x 1.8 x 0.6, average edge 1.0) and
/// ~128 for a sleeping one (0.2 x 0.2). In live play the server's tracking range and the terrain
/// walk's section gate are the binding limits, exactly as in vanilla.
///
/// 1.9+ moved the weight to a static and wrote the factor into an `EntityOtherPlayerMP`
/// `isInRangeToRenderDist` override (`averageEdgeLength * 10`, NaN -> 1). For every non-NaN box
/// that is the same threshold up to one ulp; a NaN box needs a NaN position, whose NaN distance
/// fails the strict `<` in both forms.
const REMOTE_PLAYER_RENDER_DISTANCE_WEIGHT: f64 = 10.0;

/// The `Entity.renderDistanceWeight` of one tracked entity.
#[must_use]
fn tracked_render_distance_weight(kind: TrackedKind) -> f64 {
    match kind {
        TrackedKind::Player => REMOTE_PLAYER_RENDER_DISTANCE_WEIGHT,
        _ => kind
            .entity_type()
            .map_or(1.0, |kind| kind.info().render_distance_weight),
    }
}

/// Whether this kind's class overrides `isInRangeToRenderDist` with the fourfold projectile rule
/// ([`projectile_render_distance_base`]).
///
/// java: the same override body sits on `EntityThrowable` (snowball, egg, ender pearl, splash
/// potion, experience bottle), `EntityFireball` (large and small fireballs, wither skull),
/// `EntityEnderEye` and `EntityFishHook`. `EntityArrow` instead raises `renderDistanceWeight`
/// (extracted), and `EntityArmorStand` has its own variant with a zero-size clause.
#[must_use]
fn uses_projectile_render_distance(kind: TrackedKind) -> bool {
    use EntityType as E;
    match kind {
        TrackedKind::FishingBobber => true,
        TrackedKind::Player => false,
        TrackedKind::Type(entity_type) => matches!(
            entity_type,
            E::Snowball
                | E::ThrownEgg
                | E::ThrownEnderpearl
                | E::ThrownPotion
                | E::ThrownExpBottle
                | E::Fireball
                | E::SmallFireball
                | E::WitherSkull
                | E::EyeOfEnderSignal
        ),
    }
}

/// The fourfold projectile override of `isInRangeToRenderDist`.
///
/// java: `d0 = getEntityBoundingBox().getAverageEdgeLength() * 4.0D; if (Double.isNaN(d0)) d0 =
/// 4.0D; d0 = d0 * 64.0D;` — no `renderDistanceWeight` factor (all of these keep `Entity`'s
/// extracted `1.0`, so the shared `base * 64 * weight` is exact). A 0.25 throwable or hook is
/// therefore drawn to 64 blocks rather than the generic 16, a 0.3125 small fireball to 80 rather
/// than 20, and a 1.0 large fireball to 256 rather than 64. Unlike the armor stand there is no
/// zero-size clause.
#[must_use]
fn projectile_render_distance_base(bounds: Aabb) -> f64 {
    let render_distance_base = bounds.average_edge_length() * 4.0;
    if render_distance_base.is_nan() {
        4.0
    } else {
        render_distance_base
    }
}

/// `EntityArmorStand.isInRangeToRenderDist`'s fourfold box-size override.
///
/// Java multiplies the box's average edge length by `4.0` before applying the shared `64.0`
/// factor. Marker stands have a zero-sized live box, so Java's zero/NaN fallback keeps them at a
/// fixed `4.0 * 64.0` range instead of letting the generic zero-box calculation cull them at once.
#[must_use]
fn armor_stand_render_distance_base(bounds: Aabb) -> f64 {
    let mut render_distance_base = bounds.average_edge_length() * 4.0;
    if render_distance_base.is_nan() || render_distance_base == 0.0 {
        render_distance_base = 4.0;
    }
    render_distance_base
}

fn tracked_render_bounds(world: &World, tracked: &TrackedEntity, position: DVec3) -> Aabb {
    let (width, height) = tracked.dimensions();
    let fallback = || crate::app::interact::size_bounding_box(position, width, height);
    match tracked.kind.entity_type() {
        Some(EntityType::LeashKnot) => crate::app::interact::leash_knot_bounding_box(position),
        Some(EntityType::ItemFrame) => world
            .ecs
            .get::<&ItemFrameFacing>(tracked.entity)
            .ok()
            .map(|facing| crate::app::interact::hanging_bounding_box(position, facing.0, 12, 12))
            .unwrap_or_else(fallback),
        Some(EntityType::Painting) => world
            .ecs
            .get::<&Hanging>(tracked.entity)
            .ok()
            .map(|hanging| {
                crate::app::interact::hanging_bounding_box(
                    position,
                    hanging.facing,
                    hanging.art.size_x,
                    hanging.art.size_y,
                )
            })
            .unwrap_or_else(fallback),
        _ => fallback(),
    }
}

/// The section whose entity list holds an entity at `position`. java: `Chunk.addEntity` clamps
/// `chunkCoordY` into the column, so an entity above or below the world is still listed.
#[must_use]
fn entity_list_section(position: DVec3) -> mc_math::ChunkSectionPos {
    let block = mc_math::BlockPos::from_vec3(position);
    mc_math::ChunkSectionPos::new(block.x >> 4, (block.y >> 4).clamp(0, 15), block.z >> 4)
}

/// Whether this frame's culling walk reached the section listing this entity.
///
/// java: `RenderGlobal.renderEntities` iterates `renderInfos` and each section's entity list, so an
/// entity in a section the walk did not reach is never considered at all. The list follows the
/// tick position, not the interpolated one. The local vehicle stays exempt: it sits around the
/// camera, and drawing it from a section the walk missed costs one entity.
#[must_use]
pub(crate) fn entity_section_reached(
    render: &RenderStack,
    world: &World,
    tracked: &TrackedEntity,
    is_local_vehicle: bool,
) -> bool {
    is_local_vehicle
        || render
            .terrain
            .borrow()
            .section_reached_by_walk(entity_list_section(tracked.render_position(world, 1.0)))
}

/// What the living-entity pass may draw for one entity, given the section gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EntityPassGate {
    /// Nothing at all.
    Skip,
    /// The spectator outline and nothing else.
    OutlineOnly,
    /// The whole entity.
    Everything,
}

/// java: `RenderGlobal.renderEntities` reads `renderInfos` for everything except its spectator
/// outline loop (`:606`), which walks every loaded entity and keeps the players. So a player in a
/// section the walk did not reach still gets an outline. That is the point of the feature: it shows
/// players through terrain.
#[must_use]
pub(crate) fn entity_pass_gate(
    section_reached: bool,
    outlines_enabled: bool,
    is_player: bool,
) -> EntityPassGate {
    if section_reached {
        EntityPassGate::Everything
    } else if outlines_enabled && is_player {
        EntityPassGate::OutlineOnly
    } else {
        EntityPassGate::Skip
    }
}

/// Apply the shared range/frustum decision to one tracked entity at its interpolated frame pose.
#[must_use]
pub(crate) fn should_render_tracked(
    frustum: &Frustum,
    render_view_position: DVec3,
    world: &World,
    tracked: &TrackedEntity,
    partial_ticks: f32,
    force_visible: bool,
) -> bool {
    let position = tracked.render_position(world, partial_ticks);
    let bounds = tracked_render_bounds(world, tracked, position);
    let ignore_frustum = tracked.kind == TrackedKind::FishingBobber
        || tracked.kind.entity_type() == Some(EntityType::EnderDragon);
    let entity_type = tracked.kind.entity_type();
    let render_distance_weight = tracked_render_distance_weight(tracked.kind);
    let render_distance_base_override = if entity_type == Some(EntityType::ArmorStand) {
        Some(armor_stand_render_distance_base(bounds))
    } else if uses_projectile_render_distance(tracked.kind) {
        Some(projectile_render_distance_base(bounds))
    } else {
        render_distance_base(entity_type)
    };
    match render_distance_base_override {
        Some(base) => should_render_bounds_with_distance_base(
            frustum,
            render_view_position,
            position,
            bounds,
            render_distance_weight,
            Some(base),
            ignore_frustum,
            force_visible,
        ),
        None => should_render_bounds(
            frustum,
            render_view_position,
            position,
            bounds,
            render_distance_weight,
            ignore_frustum,
            force_visible,
        ),
    }
}

/// Opt-in, **non-vanilla** entity render-distance caps: `playerRenderDistance` and
/// `entityRenderDistance` in `options.txt`, in blocks, `0` = off. Both default off, so unless the
/// player asks, only vanilla's own gate ([`should_render_tracked`]) decides.
///
/// A cap is applied *after* the vanilla gate (leash retention included) and only ever removes
/// entities, so it is a hard visual limit and never draws something vanilla would not. It is
/// render-only: picking, ticking and every packet read the tracker, not this gate. The caller
/// exempts the local vehicle, as vanilla's `riddenByEntity == thePlayer` clause does.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct EntityRenderCaps {
    /// Squared cap for remote players.
    player_sq: Option<f64>,
    /// Squared cap for everything else that is not [`Self::exempt`].
    other_sq: Option<f64>,
}

impl EntityRenderCaps {
    /// Read both caps from the settings; non-positive values are "off".
    #[must_use]
    pub(crate) fn from_settings(settings: &mc_render::settings::GameSettings) -> Self {
        let squared = |blocks: i32| (blocks > 0).then(|| f64::from(blocks) * f64::from(blocks));
        Self {
            player_sq: squared(settings.player_render_distance),
            other_sq: squared(settings.entity_render_distance),
        }
    }

    /// Whether the entity cap leaves this kind alone: anything that can hit the player
    /// (projectiles, the fishing hook, primed TNT) stays on vanilla's range whatever the cap.
    #[must_use]
    fn exempt(kind: TrackedKind) -> bool {
        use EntityType as E;
        match kind {
            TrackedKind::Player => false,
            TrackedKind::FishingBobber => true,
            TrackedKind::Type(entity_type) => matches!(
                entity_type,
                E::Arrow
                    | E::Snowball
                    | E::ThrownEgg
                    | E::Fireball
                    | E::SmallFireball
                    | E::ThrownEnderpearl
                    | E::ThrownPotion
                    | E::WitherSkull
                    | E::PrimedTnt
                    | E::MinecartTNT
            ),
        }
    }

    /// The squared cap that applies to this kind, if any.
    #[must_use]
    fn cap_squared(self, kind: TrackedKind) -> Option<f64> {
        match kind {
            TrackedKind::Player => self.player_sq,
            kind if Self::exempt(kind) => None,
            _ => self.other_sq,
        }
    }

    /// Whether the caps let an entity at this squared distance through (strict, like vanilla).
    #[must_use]
    fn allows_distance_squared(self, kind: TrackedKind, distance_squared: f64) -> bool {
        self.cap_squared(kind)
            .is_none_or(|cap_squared| distance_squared < cap_squared)
    }

    /// Whether the caps let this tracked entity through, measured from the same interpolated
    /// render-view position as the vanilla gate. Costs nothing while both caps are off.
    #[must_use]
    pub(crate) fn allows(
        self,
        world: &World,
        tracked: &TrackedEntity,
        partial_ticks: f32,
        render_view_position: DVec3,
    ) -> bool {
        if self.cap_squared(tracked.kind).is_none() {
            return true;
        }
        let distance_squared =
            (tracked.render_position(world, partial_ticks) - render_view_position).length_squared();
        self.allows_distance_squared(tracked.kind, distance_squared)
    }
}

/// A session-owned local player snapshot used by leash visibility. The local player is not in
/// [`EntityTracker`], so the regular remote-generation lookup cannot resolve a leash that points
/// at the local entity. Keep only the frame pose and dimensions here; the renderer still uses the
/// tracker generation token for every remote holder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LocalLeashHolder {
    pub network_id: i32,
    pub position: DVec3,
    /// The current tick's pose, which `RenderGuardian.shouldRender` samples (`partialTicks`
    /// `1.0F`) for a beam aimed at the local player.
    pub tick_position: DVec3,
    pub width: f32,
    pub height: f32,
    pub prev_yaw: f32,
    pub yaw: f32,
    pub prev_pitch: f32,
    pub pitch: f32,
    pub eye_height: f32,
}

/// `RenderLiving.shouldRender`'s leash-holder frustum retention.
///
/// Java first runs the ordinary living-render predicate. If that fails, a living entity remains
/// eligible when its *current* leash holder's bounding box intersects the camera frustum. The
/// holder is deliberately resolved by an ECS generation token: a late holder may attach once it
/// spawns, while a destroy/reuse boundary never lets the old leash follow the replacement id.
#[must_use]
pub(crate) fn should_render_tracked_with_leash(
    frustum: &Frustum,
    render_view_position: DVec3,
    state: &ClientState,
    tracked: &TrackedEntity,
    partial_ticks: f32,
    force_visible: bool,
    local_holder: Option<LocalLeashHolder>,
) -> bool {
    if should_render_tracked(
        frustum,
        render_view_position,
        &state.world,
        tracked,
        partial_ticks,
        force_visible,
    ) {
        return true;
    }
    if !tracked.kind.is_living() {
        return false;
    }
    let Some(holder_id) = tracked.leashed_to() else {
        return guardian_beam_in_frustum(frustum, state, tracked, local_holder);
    };
    if let Some(local) = local_holder.filter(|local| local.network_id == holder_id) {
        let bounds =
            crate::app::interact::size_bounding_box(local.position, local.width, local.height);
        return frustum.contains_box(bounds.min, bounds.max)
            || guardian_beam_in_frustum(frustum, state, tracked, local_holder);
    }
    let Some(holder) = state.entities.get_loaded(holder_id) else {
        return guardian_beam_in_frustum(frustum, state, tracked, local_holder);
    };
    if tracked.leash_holder_entity() != Some(holder.entity) {
        return false;
    }
    let position = holder.render_position(&state.world, partial_ticks);
    let bounds = tracked_render_bounds(&state.world, holder, position);
    let render_box = bounds.render_box_or_fallback(position);
    frustum.contains_box(render_box.min, render_box.max)
        || guardian_beam_in_frustum(frustum, state, tracked, local_holder)
}

/// `RenderGuardian.shouldRender`'s second frustum branch. It keeps a Guardian visible when the
/// body itself is outside the camera but the eye-to-target beam segment is inside. Java samples
/// both ends with `func_177110_a(entity, offset, 1.0F)`: the current tick's pose, not the
/// frame's partial tick. The target uses the same local-player snapshot / remote-generation
/// resolver as the actual emitter, so a stale watcher id cannot bring either the body or its
/// orphaned beam back into the scene.
fn guardian_beam_in_frustum(
    frustum: &Frustum,
    state: &ClientState,
    tracked: &TrackedEntity,
    local_holder: Option<LocalLeashHolder>,
) -> bool {
    if tracked.kind.entity_type() != Some(EntityType::Guardian) {
        return false;
    }
    let Some(target_id) = tracked
        .guardian_laser_state()
        .and_then(|state| state.target)
    else {
        return false;
    };
    let target_centre = if local_holder.is_some_and(|local| local.network_id == target_id) {
        let local = local_holder.expect("checked above");
        local.tick_position + DVec3::new(0.0, f64::from(local.height * 0.5), 0.0)
    } else {
        let Some(target) = state.entities.guardian_target_for(tracked) else {
            return false;
        };
        let position = target.render_position(&state.world, 1.0);
        position + DVec3::new(0.0, f64::from(target.dimensions().1 * 0.5), 0.0)
    };
    let source = tracked.render_position(&state.world, 1.0)
        + DVec3::new(0.0, f64::from(tracked.eye_height(&state.world)), 0.0);
    let beam_box = Aabb::new(
        DVec3::new(
            source.x.min(target_centre.x),
            source.y.min(target_centre.y),
            source.z.min(target_centre.z),
        ),
        DVec3::new(
            source.x.max(target_centre.x),
            source.y.max(target_centre.y),
            source.z.max(target_centre.z),
        ),
    );
    frustum.contains_box(beam_box.min, beam_box.max)
}

/// `EntityFireworkRocket.isInRangeToRenderDist`'s fixed client visibility limit.
///
/// Unlike the shared entity range calculation, rockets use a strict squared distance below
/// 4096. The origin is the same interpolated render-view feet position used by the shared gate;
/// using the F5 camera or eye position would move the boundary while changing perspective.
#[must_use]
pub(crate) fn should_render_firework(render_view_position: DVec3, rocket_position: DVec3) -> bool {
    (rocket_position - render_view_position).length_squared() < 4096.0
}

// ---------------------------------------------------------------------------------------------
// Texture keys
// ---------------------------------------------------------------------------------------------

/// The [`EntityPhase`] key the stitched block atlas is uploaded under.
///
/// java: `TextureMap.locationBlocksTexture`, which `RenderSnowball`, `RenderFireball`,
/// `RenderTNTPrimed`, `RenderFallingBlock` and `RenderEntityItem` all bind instead of an entity
/// texture, because a baked model's UVs are atlas coordinates.
///
/// The string is duplicated from `driver.rs`'s constant of the same name rather than shared:
/// both are private to their module and `driver.rs` is edited by several authors at once. The
/// *value* must match, and no test can assert that across a private item, so this comment is the
/// contract. Uploading under it twice is harmless ([`EntityPhase::upload_texture`] replaces by
/// key), and the two passes then merge into a single draw call because the batches are adjacent
/// and share both texture and blend mode.
const BLOCK_ATLAS_TEXTURE: &str = "mc-render/entity/block-atlas";

/// The texture selected by one non-living renderer.
///
/// Most object renderers bind a fixed resource-pack path.  `builtin/entity` item models are the
/// one exception: their tile renderer can select a per-banner composite key, so the descriptor
/// has to travel with the key until the entity phase has a chance to upload it.  Keeping this
/// wrapper private leaves the existing object pass API and all ordinary atlas paths unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
enum ObjectTexture {
    /// A fixed entity or atlas texture path.
    Resource(&'static str),
    /// A tile-item texture, possibly a dynamic banner composite.
    Builtin {
        /// The decoded tile-item state used to build the geometry and, for banners, the texture.
        item: BuiltinTileItem,
        /// The key returned by [`TileModels::emit_builtin_item`].
        key: String,
    },
    /// An item frame whose shell is an atlas model and whose displayed stack is a
    /// `builtin/entity` tile model.  Java submits those through two different texture binds but
    /// the object dispatcher owns one vertex scratch buffer, so retain both ranges until upload.
    ItemFrameBuiltin {
        /// The tile-item descriptor selected from the displayed stack.
        item: BuiltinTileItem,
        /// The dynamic/fixed entity texture key selected by the tile model.
        key: String,
        /// Shell vertex range in the shared object scratch buffer.
        shell_range: (usize, usize),
        /// Displayed builtin vertex range in the shared object scratch buffer.
        content_range: (usize, usize),
    },
}

/// The object-renderer shadow values inherited from Java's base `Render` class.
///
/// `RenderXPOrb` and `RenderEntityItem` both use the small `0.15` shadow with `0.75` opacity;
/// primed TNT keeps the base opacity. Paintings deliberately return `None`: their renderer never
/// assigns a positive `shadowSize`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ObjectShadow {
    size: f32,
    opacity: f32,
}

#[must_use]
const fn object_shadow(kind: EntityType) -> Option<ObjectShadow> {
    match kind {
        EntityType::XPOrb | EntityType::Item => Some(ObjectShadow {
            size: 0.15,
            opacity: 0.75,
        }),
        EntityType::PrimedTnt => Some(ObjectShadow {
            size: 0.5,
            opacity: 1.0,
        }),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------------------------

/// `RenderManager.playerViewY`/`playerViewX` for this frame, from the frame camera.
///
/// java: `RenderManager.cacheActiveRenderInfo` (`:258-267`) sets them to the view entity's
/// interpolated yaw and pitch, then `if (thirdPersonView == 2) playerViewY += 180.0F` — the yaw
/// only. The F5 front-view camera looks back at the player, so its yaw already includes that 180
/// but its pitch is the view pitch *negated*; vanilla's billboards (RenderSnowball/RenderPotion,
/// RenderFireball, RenderXPOrb) keep the un-negated pitch, so in that view their sprites tilt
/// against the camera. First person and the back view use the camera's own angles.
#[must_use]
pub(crate) fn render_manager_view(camera: &Camera, third_person_view: u8) -> (f32, f32) {
    if third_person_view == 2 {
        (camera.yaw, -camera.pitch)
    } else {
        (camera.yaw, camera.pitch)
    }
}

/// Draws every entity `upload_entities` skipped.
///
/// Runs after it, in the same frame and against the same [`EntityPhase`]: `push` after `finish`
/// is legal (`finish` only re-uploads the accumulated vertex list, it does not close the phase),
/// so this pass appends and calls `finish` again. The second upload rewrites the living-entity
/// geometry too, which costs one extra `write_buffer` of a few hundred kilobytes at most and is
/// the price of not editing the loop five other authors are inside.
///
/// Entities with no renderer here — lightning, boats/minecarts and the leash knot — are silently
/// skipped, exactly as `upload_entities` skipped these until now. Item frames are special: their
/// baked shell and ordinary fixed-item contents use the atlas batch, builtin tile contents append
/// a tile-texture batch, and filled maps append a dynamic-texture batch immediately afterwards
/// (see [`push_item_frame_map`]).
// The explicit context mirrors Java's RenderGlobal/RenderManager adapter: keeping the GPU,
// render phases, asset bundle, camera, and per-frame entity state separate makes each texture
// batch's ownership visible at the call site.
// The pickup slice is deliberately crate-owned: only the client driver can produce a valid
// captured child, while this public render entry point remains part of the app facade.
#[allow(private_interfaces)]
#[allow(clippy::too_many_arguments)]
pub fn upload_objects(
    gpu: &Gpu,
    render: &RenderStack,
    models: &mut RenderManager,
    assets: &GameAssets,
    state: &ClientState,
    camera: &Camera,
    render_view_position: DVec3,
    partial_ticks: f32,
    local_vehicle: Option<i32>,
    entity_shadows: bool,
    pickup_effects: &[PickupEffect],
    local_fishing_angler: Option<FishingLineAngler>,
) {
    // This standalone wrapper has no retained `Client` to borrow models from; the frame path
    // (`Client::frame`) passes its persistent `tile_models` instead of building one per frame.
    let mut tile_models = TileModels::new(crate::app::tiles::christmas_today());
    let finish_needed = upload_objects_unfinalized(
        gpu,
        render,
        models,
        &mut tile_models,
        assets,
        state,
        camera,
        render_view_position,
        partial_ticks,
        local_vehicle,
        entity_shadows,
        pickup_effects,
        local_fishing_angler,
        EntityRenderCaps::default(),
        // No perspective mode here: the first-person/back-view pair, which is the camera's own.
        (camera.yaw, camera.pitch),
        None,
        // No `GameSettings` here either: vanilla's defaults (fancy graphics, smooth lighting).
        mc_render::terrain::MeshSettings::default(),
        &mut ObjectScratch {
            full_gate_walk: true,
            ..ObjectScratch::default()
        },
    );
    // Preserve the standalone helper existing contract. Client::frame uses the append-only
    // entry point below and owns the final finish after every producer has appended.
    if finish_needed {
        render.entities.with(|phase| phase.finish(gpu));
    }
}

#[derive(Default)]
pub(crate) struct ObjectScratch {
    verts: Vec<EntityVertex>,
    painting_verts: Vec<objects::PaintingVertex>,
    glint_verts: Vec<EntityVertex>,
    tnt_flash_verts: Vec<EntityVertex>,
    fishing_line_verts: Vec<EntityVertex>,
    fire_verts: Vec<EntityVertex>,
    shadow_verts: Vec<EntityVertex>,
    /// Gate every entity, object renderer or not. Only the standalone
    /// [`upload_objects`] wrapper sets it: its `finish` decision counts entities that passed the
    /// gates. The frame path ignores that answer and skips them.
    full_gate_walk: bool,
}

/// `MC_NW_F1_OBJECT_PREFILTER=0`
/// gates every tracked entity in the object pass again. Read once per process.
fn nw_f1_object_prefilter() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("MC_NW_F1_OBJECT_PREFILTER").is_none_or(|value| value != "0")
    })
}

/// The kinds [`emit_object`] has an arm for; every other kind returns
/// `None` there, so the object pass draws nothing for it. Kept equal to that match by
/// `object_pass_kinds_are_exactly_emit_objects_arms`.
#[must_use]
fn object_pass_draws(kind: EntityType) -> bool {
    use EntityType as E;
    matches!(
        kind,
        E::Item
            | E::XPOrb
            | E::Arrow
            | E::Snowball
            | E::ThrownEgg
            | E::ThrownEnderpearl
            | E::EyeOfEnderSignal
            | E::ThrownExpBottle
            | E::FireworksRocketEntity
            | E::ThrownPotion
            | E::Fireball
            | E::SmallFireball
            | E::PrimedTnt
            | E::FallingSand
            | E::ItemFrame
            | E::Painting
            | E::WitherSkull
    )
}

/// Append object/pickup/debug geometry without finalizing the shared entity phase.
///
/// Client::frame uses this entry point so tile and object producers share one final upload;
/// upload_objects remains the compatibility wrapper for direct preview callers.
#[allow(private_interfaces)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn upload_objects_unfinalized(
    gpu: &Gpu,
    render: &RenderStack,
    models: &mut RenderManager,
    tile_models: &mut TileModels,
    assets: &GameAssets,
    state: &ClientState,
    camera: &Camera,
    render_view_position: DVec3,
    partial_ticks: f32,
    local_vehicle: Option<i32>,
    entity_shadows: bool,
    pickup_effects: &[PickupEffect],
    local_fishing_angler: Option<FishingLineAngler>,
    entity_render_caps: EntityRenderCaps,
    render_manager_view: (f32, f32),
    // The entity a spectator looks out of, while in first person
    // (`mods::x6_entities::first_person_view_entity`): `RenderGlobal.renderEntities` skips it.
    hidden_view_entity: Option<i32>,
    // The live graphics settings a falling block's `BlockModelRenderer.renderModel` reads:
    // `Minecraft.isAmbientOcclusionEnabled()` and the fancy-graphics leaves opacity.
    block_mesh: mc_render::terrain::MeshSettings,
    scratch: &mut ObjectScratch,
) -> bool {
    mc_core::profile_scope!("client.upload_objects");
    // java: `RenderManager.cacheActiveRenderInfo` (`:258-267`). RenderSnowball/RenderPotion,
    // RenderFireball and RenderXPOrb billboard with `playerViewY`/`playerViewX`: the view
    // entity's yaw and pitch, with only the yaw turned 180 for the F5 front view. In the front
    // view that is not the camera's own orientation (its pitch is negated), so those renderers
    // get a copy carrying `render_manager_view` = (playerViewY, playerViewX). Only yaw/pitch
    // differ: the frustum, positions and item-frame culling still read the real camera.
    let billboard_camera = Camera {
        yaw: render_manager_view.0,
        pitch: render_manager_view.1,
        ..*camera
    };
    render.entities.with(|phase| {
        if !phase.has_texture(BLOCK_ATLAS_TEXTURE) {
            phase.upload_texture(
                gpu,
                BLOCK_ATLAS_TEXTURE,
                &assets.atlas.texture,
                assets.atlas.w,
                assets.atlas.h,
            );
        }
    });

    let block_lighting = crate::app::appearance::ShadowLighting::from_environment(
        render.fog.with(|phase| phase.environment),
        partial_ticks,
    );
    // `Render.renderEntityOnFire` resolves both animation frames once before it starts its
    // per-entity stack.  This pass owns `RenderFireball`, so it needs the same pair even though
    // the living-entity pass already has one of its own.  A resource pack can omit either sprite;
    // in that case vanilla's missing-texture result is less useful than declining to stretch the
    // complete atlas over a projectile, which is the policy `sprite_uv` documents and the living
    // pass follows too.
    let fire_sprites: Option<[[f32; 4]; 2]> =
        crate::app::appearance::sprite_uv(crate::app::appearance::FIRE_SPRITES[0])
            .zip(crate::app::appearance::sprite_uv(
                crate::app::appearance::FIRE_SPRITES[1],
            ))
            .map(|(a, b)| [a, b]);
    let mut verts = std::mem::take(&mut scratch.verts);
    let mut painting_verts = std::mem::take(&mut scratch.painting_verts);
    let mut glint_verts = std::mem::take(&mut scratch.glint_verts);
    let mut tnt_flash_verts = std::mem::take(&mut scratch.tnt_flash_verts);
    let mut fishing_line_verts = std::mem::take(&mut scratch.fishing_line_verts);
    // The flame pass is a distinct GL state from a fireball's lit item billboard: unlit, but
    // still opaque/cutout with depth writes.  Keep it separate so EntityPhase selects that
    // state instead of folding the vertices into the ordinary object batch.
    let mut fire_verts = std::mem::take(&mut scratch.fire_verts);
    let mut shadow_verts = std::mem::take(&mut scratch.shadow_verts);
    let mut drawn = false;
    let mut pickup_drawn = false;
    let mut debug_drawn = false;
    // 06-entity-batch U2: `tile_models` is the caller's retained instance (`Client::tile_models`
    // on the frame path, shared with the TESR pass) rather than eight models (a 39-part horse
    // included) rebuilt and dropped every frame. These models carry no GPU state and every
    // builtin emit sets the part state it depends on before posing, so sharing is safe; the
    // Christmas flag is then sampled once per `Client`, like `TileEntityChestRenderer`'s
    // constructor, instead of once per frame.
    // F3+B is read once per pass: when it is off, `push_debug_vertices` discards everything, so
    // the per-entity debug geometry (a Vec, an ECS rotation lookup, the look vector) is skipped.
    let debug_bounding_box = render.entities.with(|phase| phase.debug_bounding_box());
    // `RenderItem.renderEffect` samples `Minecraft.getSystemTime()` for each draw.  A single
    // frame snapshot keeps every non-GUI foil coherent while remaining deterministic at the
    // explicit policy boundary used by the CPU tests.
    let glint_time_ms = dropped::item_glint_system_time_ms();
    let frustum = Frustum::from_camera(camera);
    // With F3+B off, an entity this pass has no renderer for (every
    // player and living mob) can only run the gates below and `continue`: skip it first.
    let prefilter = nw_f1_object_prefilter() && !debug_bounding_box && !scratch.full_gate_walk;
    for (network_id, tracked) in state.entities.iter() {
        verts.clear();
        painting_verts.clear();
        glint_verts.clear();
        tnt_flash_verts.clear();
        fishing_line_verts.clear();
        fire_verts.clear();
        shadow_verts.clear();
        if hidden_view_entity == Some(network_id) {
            continue;
        }
        let kind = tracked.kind.entity_type();
        if prefilter
            && tracked.kind != TrackedKind::FishingBobber
            && !kind.is_some_and(object_pass_draws)
        {
            continue;
        }
        if !entity_section_reached(
            render,
            &state.world,
            tracked,
            local_vehicle == Some(network_id),
        ) {
            continue;
        }
        let force_visible =
            local_vehicle == Some(network_id) || kind == Some(EntityType::WitherSkull);
        if !should_render_tracked(
            &frustum,
            render_view_position,
            &state.world,
            tracked,
            partial_ticks,
            force_visible,
        ) {
            continue;
        }
        // Opt-in `entityRenderDistance` (off by default); see `EntityRenderCaps`.
        if local_vehicle != Some(network_id)
            && !entity_render_caps.allows(
                &state.world,
                tracked,
                partial_ticks,
                render_view_position,
            )
        {
            continue;
        }
        if kind == Some(EntityType::FireworksRocketEntity)
            && !should_render_firework(
                render_view_position,
                tracked.render_position(&state.world, partial_ticks),
            )
        {
            continue;
        }
        // RenderManager's debug hitbox is attached to every visible non-living renderer too.
        // Keep this in the object pass so dropped items, projectiles and the fishing hook receive
        // the same range/frustum/chunk gate as their actual body, while invisible entities remain
        // a deliberate negative even when a spectator can see their body.
        let invisible = tracked
            .with_watched(&state.world, |watched| watched.is_invisible())
            .unwrap_or(false);
        if !invisible {
            debug_drawn = true;
        }
        if !invisible && debug_bounding_box {
            let position = tracked.render_position(&state.world, partial_ticks);
            let (yaw, pitch) = state
                .world
                .ecs
                .get::<&Rotation>(tracked.entity)
                .map(|rotation| (rotation.yaw, rotation.pitch))
                .unwrap_or((tracked.target_yaw, tracked.target_pitch));
            let debug_vertices = mc_render::entity::render::debug_entity_vertices(
                camera.relative(position),
                tracked.dimensions().0,
                tracked.dimensions().1,
                tracked.eye_height(&state.world),
                crate::app::driver::debug_look_vector(
                    (tracked.prev_rotation_yaw, tracked.prev_rotation_pitch),
                    (yaw, pitch),
                    partial_ticks,
                ),
                false,
            );
            render
                .entities
                .with(|phase| phase.push_debug_vertices(&debug_vertices, false));
        }
        let texture = if tracked.kind == TrackedKind::FishingBobber {
            // `RenderFish` is a direct-construction renderer, not an `EntityList` arm. Its
            // fishing line needs angler swing/sneak state from the driver seam, so only the
            // independent billboard is emitted here.
            // java: `RenderFish.doRender` (`:43-44`) billboards with `rotate(180 - playerViewY,
            // Y)` then `rotate(-playerViewX, X)`, i.e. `RenderManager`'s view angles: in the F5
            // front view the yaw is turned 180 but the pitch is *not* negated, so the sprite
            // tilts against that camera exactly as vanilla's does.
            objects::emit_fishing_bobber(
                camera.relative(tracked.render_position(&state.world, partial_ticks)),
                render_manager_view.0,
                render_manager_view.1,
                &mut verts,
            );
            let owner = tracked.fishing_owner_id().and_then(|owner_id| {
                if owner_id == state.local.entity_id {
                    local_fishing_angler
                } else {
                    let owner = state.entities.get_loaded(owner_id)?;
                    (tracked.fishing_owner_entity() == Some(owner.entity)
                        && owner.kind == TrackedKind::Player)
                        .then(|| fishing_line_angler_for_remote(&state.world, owner))
                        .flatten()
                }
            });
            if let Some(angler) = owner {
                objects::emit_fishing_line(
                    tracked.render_position(&state.world, partial_ticks),
                    camera.pos,
                    angler,
                    partial_ticks,
                    &mut fishing_line_verts,
                );
            }
            Some(ObjectTexture::Resource(objects::FISH_TEXTURE))
        } else {
            let Some(kind) = kind else {
                continue;
            };
            emit_object(
                &mut verts,
                models,
                tile_models,
                &state.world,
                Some(&state.players),
                tracked,
                kind,
                &billboard_camera,
                partial_ticks,
                block_lighting,
                &mut glint_verts,
                glint_time_ms,
                &mut painting_verts,
                block_mesh,
            )
        };
        let Some(texture) = texture else {
            continue;
        };
        prepare_tnt_flash_vertices(
            kind,
            tracked.render.ticks_existed,
            partial_ticks,
            &verts,
            &mut tnt_flash_verts,
        );
        // `RenderItemFrame.doRender` first draws the baked `item_frame#map` model, then
        // `RenderItemFrame.renderItem` invokes `MapItemRenderer`.  Keep the lookup separate from
        // `emit_object` because the map has its own dynamic texture rather than the block atlas,
        // but make the push directly after the frame body so it retains Java's depth/order.
        let frame_map = (kind == Some(EntityType::ItemFrame))
            .then(|| item_frame_map_for_entity(&state.world, tracked, camera, partial_ticks))
            .flatten();
        // java: `Entity.isBurning()`, which on a client is the shared watcher flag (bit 0) the
        // server raises for a Flame arrow or a burning item, boat or minecart. It selects both the
        // fire stack below (`Render.doRenderShadowAndFire:314`) and the full-bright lightmap
        // (`RenderManager.renderEntityStatic:325`); read it once for both.
        let burning = crate::app::appearance::is_burning(&state.world, tracked);
        // `EntityFireball.onUpdate` calls `setFire(1)` every client tick.  Its inherited
        // `Render.doRenderShadowAndFire` therefore draws `renderEntityOnFire` after
        // `RenderFireball`'s own billboard.  Fireball and SmallFireball are the only entries in
        // this pass with that contract: `EntityWitherSkull` shares the motion superclass but
        // overrides `isBurning()` to false, so looking merely at the common base would put flames
        // on wither skulls.  The entity's `fire` timer is client-local, whereas mc-rs only retains
        // the packet-visible shared watcher flag, so do not gate this on `is_burning` -- doing so
        // would make every local fireball bare until an unrelated metadata update happened.
        // Every other object takes the same stack from the watched flag, at its own size.
        let fire_size = kind
            .and_then(fireball_overlay_size)
            .or_else(|| (burning && object_can_render_on_fire(kind)).then(|| tracked.dimensions()));
        if let (Some(sprites), Some((width, height))) = (fire_sprites, fire_size) {
            crate::app::appearance::emit_fire_overlay(
                camera.relative(tracked.render_position(&state.world, partial_ticks)),
                width,
                height,
                camera.yaw,
                |i| sprites[i],
                &mut fire_verts,
            );
        }
        if verts.is_empty()
            && painting_verts.is_empty()
            && fire_verts.is_empty()
            && frame_map.is_none()
        {
            continue;
        }
        drawn = true;
        // java: object renderers submit with the packed lightmap selected by
        // `Entity.getBrightnessForRender`. Dropped items, arrows, the inherited
        // RenderSnowball/RenderPotion variants, primed TNT, and fishing bobbers sample their
        // current entity/eye cell before `RenderManager` interpolates the draw pose; Ender Eye,
        // fireballs and wither skulls are excluded because their Java entity classes explicitly
        // return the full-bright constant. Item frames retain their sample while drawing the
        // frame/content/map, while XP orbs apply their own half-bright block-light adjustment.
        // Paintings replace the inherited value with their per-cell wall samples below.
        let render_position = tracked.render_position(&state.world, partial_ticks);
        let current_position = tracked_current_position(&state.world, tracked, render_position);
        let light_position = object_light_position(&state.world, tracked, kind, render_position);
        let object_combined_light = if tracked.kind == TrackedKind::FishingBobber {
            Some(fishing_bobber_combined_light(&state.world, light_position))
        } else {
            inherited_object_light(kind, &state.world, light_position)
        };
        // `RenderManager.renderEntityStatic` replaces the concrete entity's selected packed
        // lightmap with full bright when `Entity.isBurning()` is true.  Keep that terminal
        // override after the per-kind world/emissive selection: a watched fire flag must brighten
        // an otherwise ordinary object, while explicit full-bright entities remain unchanged.
        let object_combined_light = object_combined_light.map(|combined_light| {
            if burning {
                mc_render::entity::render::FULL_BRIGHT_COMBINED_LIGHT
            } else {
                combined_light
            }
        });
        let falling_vertex_lights = (kind == Some(EntityType::FallingSand)).then(|| {
            falling_block_vertex_lights_for_entity(
                &state.world,
                tracked,
                current_position,
                block_mesh,
            )
            .unwrap_or_default()
        });
        let falling_vertex_lights = falling_vertex_lights
            .as_deref()
            .filter(|lights| lights.len() == verts.len() && !lights.is_empty());
        if painting_verts.is_empty() {
            match &texture {
                ObjectTexture::Resource(key) => {
                    if let Some(lights) = falling_vertex_lights {
                        // java: `RenderFallingBlock.doRender`
                        // calls `disableLighting()` around its draw (and the BLOCK vertex format
                        // has no normal), leaving the world pass's back-face culling on. Its only
                        // shading is the block renderer's own, already in the vertex colour; the
                        // lit pipeline's two-light diffuse made every side face darker (x0.74
                        // north/south, x0.5 east/west, x0.4 bottom) than the same block in the
                        // terrain.
                        push_geometry_lit_with_vertex_lights(
                            gpu,
                            render,
                            assets,
                            key,
                            LayerBlend::OpaqueUnlitCulled,
                            object_combined_light,
                            &verts,
                            lights,
                        );
                    } else {
                        push_geometry_lit(
                            gpu,
                            render,
                            assets,
                            key,
                            object_body_blend(kind),
                            object_combined_light,
                            &verts,
                        );
                    }
                }
                ObjectTexture::Builtin { item, key } => push_builtin_geometry(
                    gpu,
                    render,
                    assets,
                    item,
                    key,
                    LayerBlend::Opaque,
                    object_combined_light,
                    &verts,
                ),
                ObjectTexture::ItemFrameBuiltin {
                    item,
                    key,
                    shell_range,
                    content_range,
                } => {
                    let (shell_start, shell_end) = *shell_range;
                    let (content_start, content_end) = *content_range;
                    debug_assert!(shell_start <= shell_end && shell_end <= verts.len());
                    debug_assert!(
                        content_start <= content_end
                            && content_end <= verts.len()
                            && shell_end <= content_start
                    );
                    // The frame shell is emitted unculled; its pipeline culls it.
                    push_geometry_lit(
                        gpu,
                        render,
                        assets,
                        BLOCK_ATLAS_TEXTURE,
                        object_body_blend(kind),
                        object_combined_light,
                        &verts[shell_start..shell_end],
                    );
                    push_builtin_geometry(
                        gpu,
                        render,
                        assets,
                        item,
                        key,
                        LayerBlend::Opaque,
                        object_combined_light,
                        &verts[content_start..content_end],
                    );
                }
            }
        } else {
            debug_assert_eq!(texture, ObjectTexture::Resource(objects::PAINTING_TEXTURE));
            push_painting_geometry(
                gpu,
                render,
                assets,
                objects::PAINTING_TEXTURE,
                &painting_verts,
            );
        }
        if !tnt_flash_verts.is_empty() {
            // X6: java's flash keeps the lightmap stage (only texture unit 0 is disabled), so it
            // takes the entity's own packed light, as the body does.
            push_geometry_lit(
                gpu,
                render,
                assets,
                BLOCK_ATLAS_TEXTURE,
                LayerBlend::TntFlash,
                object_combined_light,
                &tnt_flash_verts,
            );
        }
        // java: `RenderManager.doRenderEntity` calls `Render.doRenderShadowAndFire` after each
        // renderer's body. XP orbs override both radius and opacity; primed TNT overrides only
        // radius. Paintings leave the inherited radius at zero and therefore never enter this
        // path. Keep the shared invisible/settings/distance gates identical to living entities.
        if entity_shadows {
            if let Some(shadow) = kind.and_then(object_shadow) {
                let invisible = tracked
                    .with_watched(&state.world, |watched| watched.is_invisible())
                    .unwrap_or(false);
                if !invisible {
                    let position = tracked.render_position(&state.world, partial_ticks);
                    let alpha = crate::app::appearance::shadow_alpha(
                        (position - camera.pos).length_squared(),
                    ) * shadow.opacity;
                    if alpha > 0.0
                        && super::driver::emit_entity_shadow_for_environment(
                            &state.world,
                            camera,
                            position,
                            shadow.size,
                            alpha,
                            render.fog.with(|phase| phase.environment),
                            partial_ticks,
                            &mut shadow_verts,
                        ) > 0
                    {
                        push_geometry(
                            gpu,
                            render,
                            assets,
                            crate::app::appearance::SHADOW_TEXTURE,
                            LayerBlend::Translucent,
                            &shadow_verts,
                        );
                    }
                }
            }
        }
        if !glint_verts.is_empty() {
            push_geometry_lit(
                gpu,
                render,
                assets,
                dropped::ENCHANTED_ITEM_GLINT_TEXTURE,
                object_glint_blend(kind),
                object_combined_light,
                &glint_verts,
            );
        }
        if let Some(map) = frame_map {
            push_item_frame_map(gpu, render, assets, map, object_combined_light, camera.view);
        }
        // `Render.doRenderShadowAndFire` runs after RenderFireball.doRender.  This call is
        // intentionally after the main object batch so the two billboards retain that order.
        push_geometry(
            gpu,
            render,
            assets,
            BLOCK_ATLAS_TEXTURE,
            LayerBlend::OpaqueUnlit,
            &fire_verts,
        );
        if !fishing_line_verts.is_empty() {
            push_geometry(
                gpu,
                render,
                assets,
                EntityPhase::SOLID_TEXTURE,
                LayerBlend::FishingLine,
                &fishing_line_verts,
            );
        }
    }
    // The opaque framed-map images queued during the loop, now one draw per
    // atlas page (and the frame shells they no longer split merged as they were pushed).
    render.entities.with(|phase| phase.flush_deferred_maps());
    // `EntityPickupFX` is submitted after the source entity has already left the live tracker.
    // Keep these children in the same entity phase so they share the ordinary item/orb renderer
    // and depth state, but never reinsert a dead source into world collision or entity counts.
    for effect in pickup_effects {
        verts.clear();
        glint_verts.clear();
        let Some(texture) = emit_pickup_effect(
            &mut verts,
            &mut glint_verts,
            glint_time_ms,
            models,
            tile_models,
            state,
            &billboard_camera,
            partial_ticks,
            effect,
        ) else {
            continue;
        };
        if verts.is_empty() {
            continue;
        }
        pickup_drawn = true;
        // `EntityPickupFX.renderParticle` draws its source through the ordinary renderer, so a
        // collected item keeps `RenderEntityItem`'s culled, blended state.
        let source_kind = effect.source.kind.entity_type();
        match texture {
            ObjectTexture::Resource(key) => push_geometry(
                gpu,
                render,
                assets,
                key,
                object_body_blend(source_kind),
                &verts,
            ),
            ObjectTexture::Builtin { item, key } => push_builtin_geometry(
                gpu,
                render,
                assets,
                &item,
                &key,
                LayerBlend::Opaque,
                None,
                &verts,
            ),
            ObjectTexture::ItemFrameBuiltin {
                item,
                key,
                shell_range,
                content_range,
            } => {
                let (shell_start, shell_end) = shell_range;
                let (content_start, content_end) = content_range;
                push_geometry(
                    gpu,
                    render,
                    assets,
                    BLOCK_ATLAS_TEXTURE,
                    LayerBlend::Opaque,
                    &verts[shell_start..shell_end],
                );
                push_builtin_geometry(
                    gpu,
                    render,
                    assets,
                    &item,
                    &key,
                    LayerBlend::Opaque,
                    None,
                    &verts[content_start..content_end],
                );
            }
        }
        if !glint_verts.is_empty() {
            push_geometry(
                gpu,
                render,
                assets,
                dropped::ENCHANTED_ITEM_GLINT_TEXTURE,
                object_glint_blend(source_kind),
                &glint_verts,
            );
        }
    }
    super::frame_scratch::recycle(&mut scratch.verts, verts);
    super::frame_scratch::recycle(&mut scratch.painting_verts, painting_verts);
    super::frame_scratch::recycle(&mut scratch.glint_verts, glint_verts);
    super::frame_scratch::recycle(&mut scratch.tnt_flash_verts, tnt_flash_verts);
    super::frame_scratch::recycle(&mut scratch.fishing_line_verts, fishing_line_verts);
    super::frame_scratch::recycle(&mut scratch.fire_verts, fire_verts);
    super::frame_scratch::recycle(&mut scratch.shadow_verts, shadow_verts);
    drawn || pickup_drawn || debug_drawn
}

/// The entity-phase state an object's body is drawn with.
///
/// java: `RenderEntityItem.doRender` (`:103-106`) blends and keeps the world pass's back-face
/// culling (see [`LayerBlend::DroppedItem`]). `RenderItemFrame` keeps that culling too, for its
/// frame model and for `renderItem(stack, FIXED)` → `renderItemModelTransform`, which also
/// blends (`RenderItem.java:316-341`); the frame textures are fully opaque, so the blend leaves
/// the frame itself unchanged. `RenderArrow` keeps the culling as well, unblended and with its
/// light in the vertex colour ([`objects::ARROW_BLEND`]); drawn two-sided, every arrow z-fought
/// with itself. The other object renderers stay on the two-sided opaque state.
fn object_body_blend(kind: Option<EntityType>) -> LayerBlend {
    match kind {
        Some(EntityType::Item | EntityType::ItemFrame) => LayerBlend::DroppedItem,
        Some(EntityType::Arrow) => objects::ARROW_BLEND,
        _ => LayerBlend::Opaque,
    }
}

/// F3+B's box and look ray for every arrow `LayerArrow` sticks in a mob, 20 vertices each (the
/// [`EntityPhase::push_debug_vertices`] non-living stream), appended to `out`.
///
/// java: `LayerArrow` draws each arrow through `RenderManager.renderEntityWithPosYaw`, i.e.
/// `doRenderEntity(..., hideDebugBox = false)`, so with F3+B on `renderDebugBoundingBox` runs
/// for the temporary `EntityArrow` (never invisible). That happens after `RenderArrow`'s
/// `popMatrix`, under the posed attachment matrix: the arrow's own 0.5 x 0.5 box sits
/// axis-aligned in the model part's frame at the sampled point, and its blue ray is the arrow's
/// `getLook` (the sampled yaw and pitch) from its eye height, two blocks long. That height is 0:
/// `EntityArrow.getEyeHeight()` is overridden to return `0.0F`. Captured on the real GL by
/// `javaref/ArrowRenderRef` (case `player_hitboxes`: four draws per stuck arrow).
///
/// Not reproduced: `renderDebugBoundingBox` ends with `enableCull()` and `enableLighting()`, so
/// in vanilla the stuck arrows after the first draw culled and lit by the ambient term alone.
pub(crate) fn stuck_arrow_debug_vertices(
    models: &mut RenderManager,
    model_kind: mc_render::entity::ModelKind,
    params: &AnimParams,
    base: mc_render::entity::Mat4,
    out: &mut Vec<mc_render::entity::render::DebugVertex>,
) {
    let info = EntityType::Arrow.info();
    for attachment in models.stuck_arrow_attachments(
        model_kind,
        params.entity.entity_id,
        params.entity.arrow_count,
        params.scale,
        base,
    ) {
        // `getLook(partialTicks)` with `prevRotation == rotation`, as LayerArrow sets them.
        let rotation = (attachment.yaw, attachment.pitch);
        let look = crate::app::driver::debug_look_vector(rotation, rotation, 1.0);
        let local = mc_render::entity::render::debug_entity_vertices(
            mc_math::Vec3::ZERO,
            info.width,
            info.height,
            // java: `EntityArrow.getEyeHeight()` returns `0.0F`, not Entity's `height * 0.85F`.
            info.eye_height,
            look,
            false,
        );
        out.extend(
            local
                .into_iter()
                .map(|vertex| mc_render::entity::render::DebugVertex {
                    pos: attachment.matrix.transform_point3(vertex.pos),
                    color: vertex.color,
                }),
        );
    }
}

/// The foil state matching [`object_body_blend`]: a dropped item's (or a framed item's)
/// `renderEffect` replay keeps its culling, so only the camera-facing quad of a coplanar
/// opposite-facing pair is foiled.
fn object_glint_blend(kind: Option<EntityType>) -> LayerBlend {
    if matches!(kind, Some(EntityType::Item | EntityType::ItemFrame)) {
        LayerBlend::DroppedItemGlint
    } else {
        LayerBlend::Glint
    }
}

/// `Entity.canRenderOnFire()` for an object whose watched burning flag is set.
///
/// java: `Render.doRenderShadowAndFire` asks every entity, so a Flame arrow (the server's
/// `setFire(100)` in `ItemBow.onPlayerStoppedUsing`), a dropped item or a boat in lava gets the
/// same two-sprite stack as a burning mob. Two object classes override the answer:
/// `EntityWitherSkull.isBurning()` is hard-coded false although the server's
/// `EntityFireball.onUpdate` keeps its flag raised, and `EntityFallingBlock.canRenderOnFire()`
/// is hard-coded false.
fn object_can_render_on_fire(kind: Option<EntityType>) -> bool {
    !matches!(
        kind,
        Some(EntityType::WitherSkull | EntityType::FallingSand)
    )
}

/// The two `EntityFireball` subclasses that inherit `Entity.isBurning()` after their per-tick
/// `setFire(1)`.  `EntityWitherSkull` is deliberately excluded: it shares `EntityFireball`'s
/// renderer-adjacent motion but its Java `isBurning()` override is hard-coded false.
///
/// Keeping this as an object-pass discriminator rather than relying on watched flag zero is
/// significant.  The fire timer that `isBurning()` reads is client-local and not a spawn value;
/// the server's eventually mirrored watcher is neither its source nor a reliable first-frame
/// substitute.
fn fireball_overlay_size(kind: EntityType) -> Option<(f32, f32)> {
    matches!(kind, EntityType::Fireball | EntityType::SmallFireball).then(|| {
        let info = kind.info();
        (info.width, info.height)
    })
}

fn prepare_tnt_flash_vertices(
    kind: Option<EntityType>,
    age: u32,
    _partial_ticks: f32,
    body: &[EntityVertex],
    out: &mut Vec<EntityVertex>,
) {
    if kind != Some(EntityType::PrimedTnt) {
        return;
    }
    let fuse = objects::TNT_FUSE_TICKS - i32::try_from(age).unwrap_or(i32::MAX);
    if !objects::tnt_flashing(fuse) {
        return;
    }
    // java's `color(1, 1, 1, f2)` is overridden by the ITEM
    // format's per-vertex white (`objects::tnt_flash_alpha`), so the flash is full strength and
    // the partial tick no longer matters.
    out.extend(body.iter().copied().map(|mut vertex| {
        vertex.color = objects::TNT_FLASH_COLOR;
        vertex
    }));
}

/// Emit one `EntityPickupFX` child at its quadratic source-to-collector position.
///
/// The real Java renderer receives the original entity object, so its ordinary renderer remains
/// type-dependent.  The source snapshot carries the values that renderer needs after the live
/// ECS entity has been removed.  Unsupported source classes still get the vanilla sound/removal
/// path; they simply contribute no geometry, matching an absent renderer.
// Java's EntityPickupFX is dispatched with the live render context plus the captured source;
// retain that shape so the three-tick child follows the ordinary object pipeline exactly.
#[allow(clippy::too_many_arguments)]
fn emit_pickup_effect(
    out: &mut Vec<EntityVertex>,
    glint_out: &mut Vec<EntityVertex>,
    glint_time_ms: u64,
    models: &mut RenderManager,
    tile_models: &mut TileModels,
    state: &ClientState,
    camera: &Camera,
    partial_ticks: f32,
    effect: &PickupEffect,
) -> Option<ObjectTexture> {
    use EntityType as E;

    let position = pickup_effect_position(effect, partial_ticks);
    let rel = camera.relative(position);
    let source = &effect.source;
    match source.kind.entity_type()? {
        E::Item => {
            let stack = source.stack.as_ref()?;
            let item = mc_data::ItemId(u16::try_from(stack.item_id).ok()?);
            item.try_props()?;
            let model = mc_assets::item_model_override_for(item, stack.damage).model;
            let builtin = crate::app::tiles::builtin_tile_item_from_stack_with_players(
                stack,
                Some(&state.players),
            );
            let item = dropped::DroppedItem {
                rel,
                age: i32::try_from(source.age).unwrap_or(i32::MAX),
                hover_start: source.hover_start,
                stack_size: stack.count,
                gui_3d: builtin.is_some() || model.gui_3d,
                ground_scale: model.transforms.ground.scale,
            };
            if let Some(builtin) = builtin {
                let key = emit_dropped_builtin_item(
                    tile_models,
                    &builtin,
                    model,
                    &item,
                    partial_ticks,
                    out,
                )?;
                Some(ObjectTexture::Builtin { item: builtin, key })
            } else {
                let base_start = out.len();
                dropped::emit_dropped_item_tinted(
                    &item,
                    model,
                    mc_render::entity::Mat4::IDENTITY,
                    partial_ticks,
                    out,
                    |tint_index| item_quad_tint(stack, tint_index),
                );
                if item_has_effect(stack) {
                    // `RenderEntityItem` draws the collected item exactly as a live one, so the
                    // `GL_EQUAL` foil replays the copies just emitted. A camera-facing
                    // (`RenderSnowball`) frame here missed the spinning model's depth entirely.
                    dropped::emit_item_glint_from_base(
                        &out[base_start..],
                        item.render_count() as usize,
                        glint_time_ms,
                        glint_out,
                    );
                }
                Some(ObjectTexture::Resource(BLOCK_ATLAS_TEXTURE))
            }
        }
        E::XPOrb => {
            objects::emit_xp_orb(
                rel,
                camera.yaw,
                camera.pitch,
                source.xp_value,
                source.age,
                partial_ticks,
                out,
            );
            Some(ObjectTexture::Resource(objects::XP_ORB_TEXTURE))
        }
        E::Arrow => {
            // java: `EntityPickupFX` draws in `renderLitParticles`, after the entity pass's
            // `disableStandardItemLighting()` and with culling still on: the same culled state
            // as a live arrow (`object_body_blend`), but unlit, so the plain white emitter.
            objects::emit_arrow(rel, source.yaw, source.pitch, 0.0, out);
            Some(ObjectTexture::Resource(objects::ARROW_TEXTURE))
        }
        E::Snowball
        | E::ThrownEgg
        | E::ThrownEnderpearl
        | E::EyeOfEnderSignal
        | E::ThrownExpBottle
        | E::FireworksRocketEntity
        | E::ThrownPotion => {
            let kind = source.kind.entity_type()?;
            let (item, damage) = thrown_item_stack(kind, None)?;
            let model = mc_assets::item_model_override_for(item, damage).model;
            emit_thrown_item_tinted(rel, camera.yaw, camera.pitch, model, out, |_| 0xFF_FF_FF);
            if thrown_item_has_effect(kind, None) {
                // The foil replays the base pass under RenderSnowball's exact item transform.
                let matrix = objects::thrown_item_matrix(rel, camera.yaw, camera.pitch, model);
                dropped::emit_baked_model_glint(model, matrix, glint_time_ms, glint_out);
            }
            Some(ObjectTexture::Resource(BLOCK_ATLAS_TEXTURE))
        }
        E::Fireball | E::SmallFireball => {
            let scale = if source.kind.entity_type()? == E::Fireball {
                objects::LARGE_FIREBALL_SCALE
            } else {
                objects::SMALL_FIREBALL_SCALE
            };
            let location = mc_assets::ResourceLocation::minecraft("textures/items/fireball");
            objects::emit_fireball(
                rel,
                camera.yaw,
                camera.pitch,
                scale,
                mc_assets::sprite(&location),
                out,
            );
            Some(ObjectTexture::Resource(BLOCK_ATLAS_TEXTURE))
        }
        _ => {
            // `RenderManager.getEntityRenderObject` returns null for classes outside the object
            // and living descriptor tables. No geometry is the correct omission for those.
            let _ = models;
            None
        }
    }
}

#[must_use]
fn falling_block_should_render(
    world: &World,
    position: mc_math::DVec3,
    state: mc_data::BlockState,
) -> bool {
    world.block_at(BlockPos::from_vec3(position)) != state
}

/// One baked falling-block quad's world-model lighting result.
///
/// `BlockModelRenderer.AmbientOcclusionFace` produces two independent per-corner arrays: the
/// primary-colour multiplier and the packed `sky << 16 | block` brightness. Keeping both arrays
/// together here prevents the entity-phase adapter from accidentally applying a flat colour to a
/// smooth-lighting corner (or from throwing the packed result away before the handoff boundary).
#[derive(Debug, Clone, Copy, PartialEq)]
struct FallingBlockQuadLighting {
    color_multiplier: [[f32; 4]; 4],
    packed_light: [u32; 4],
}

const FALLING_BLOCK_QUAD_INDICES: [usize; 6] = [0, 1, 2, 0, 2, 3];

/// Resolve one quad through Java's `BlockModelRenderer` lighting branches.
///
/// `face_bucket` is `Some` for `model.face_quads[facing]` and `None` for a general quad.  The
/// distinction matters on the flat path: face-quads use the neighbour's brightness, while
/// general quads use their own block unless their own face is flush with a block boundary.
#[must_use]
#[allow(clippy::too_many_arguments)]
fn falling_block_quad_lighting(
    world: &World,
    state: mc_data::BlockState,
    block_pos: BlockPos,
    quad: &mc_assets::BakedQuad,
    face_bucket: Option<Facing>,
    smooth: bool,
    settings: mc_render::terrain::MeshSettings,
) -> FallingBlockQuadLighting {
    let props = mc_data::block_props(state);
    let (ao_multiplier, packed_light) = if smooth {
        if let Some(face) = quad.face {
            let bounds = mc_render::terrain::ao::fill_quad_bounds(quad, face, props.full_cube);
            let ao = mc_render::terrain::ao::update_vertex_brightness(
                world,
                state,
                block_pos,
                face,
                &bounds,
                Some(&quad.verts),
                settings,
            );
            (ao.color_multiplier, ao.brightness)
        } else {
            // java: mesher's smooth path falls back to renderModelStandard for the rare
            // unresolvable-facing quad; there is no neighbour table to run AO against.
            let brightness = mc_render::terrain::light::mixed_brightness(world, state, block_pos);
            ([1.0; 4], [brightness; 4])
        }
    } else {
        let brightness = match face_bucket {
            // java: renderModelStandardQuads passes the selected face neighbour's brightness.
            Some(face) => {
                mc_render::terrain::light::mixed_brightness(world, state, block_pos.relative(face))
            }
            // java: renderModelStandardQuads(... ownBrightness == true) for general quads.
            None => match quad.face {
                Some(face) => {
                    let bounds =
                        mc_render::terrain::ao::fill_quad_bounds(quad, face, props.full_cube);
                    let sample = if bounds.flush {
                        block_pos.relative(face)
                    } else {
                        block_pos
                    };
                    mc_render::terrain::light::mixed_brightness(world, state, sample)
                }
                None => mc_render::terrain::light::mixed_brightness(world, state, block_pos),
            },
        };
        ([1.0; 4], [brightness; 4])
    };

    let tint = if quad.tint_index.is_some() {
        let tint = mc_render::terrain::tint::color_multiplier(world, state, block_pos);
        [
            ((tint >> 16) & 255) as f32 / 255.0,
            ((tint >> 8) & 255) as f32 / 255.0,
            (tint & 255) as f32 / 255.0,
        ]
    } else {
        [1.0; 3]
    };
    let shade = mc_render::terrain::ao::quad_shade(quad);
    let color_multiplier = core::array::from_fn(|vertex| {
        [
            tint[0] * shade * ao_multiplier[vertex],
            tint[1] * shade * ao_multiplier[vertex],
            tint[2] * shade * ao_multiplier[vertex],
            1.0,
        ]
    });
    FallingBlockQuadLighting {
        color_multiplier,
        packed_light,
    }
}

/// Walk a falling block's baked model in the same order as `emit_baked_model` and retain the
/// per-vertex world light that the Java block renderer computes.
///
/// The returned stream is expanded into the same triangle-list order as the emitted entity
/// geometry and reaches `EntityPhase` through its parallel packed-light buffer. Keeping the model
/// walk here means AO/tint controls remain independently testable from GPU submission.
#[must_use]
fn falling_block_model_lighting(
    world: &World,
    state: mc_data::BlockState,
    block_pos: BlockPos,
    model: &mc_assets::BakedModel,
    settings: mc_render::terrain::MeshSettings,
) -> Vec<FallingBlockQuadLighting> {
    let props = mc_data::block_props(state);
    let smooth = settings.ambient_occlusion && props.light_emission == 0 && model.ambient_occlusion;
    let mut lighting = Vec::with_capacity(
        model.face_quads.iter().map(Vec::len).sum::<usize>() + model.general_quads.len(),
    );
    for (index, quads) in model.face_quads.iter().enumerate() {
        let face = Facing::from_index(index as i32);
        for quad in quads {
            lighting.push(falling_block_quad_lighting(
                world,
                state,
                block_pos,
                quad,
                Some(face),
                smooth,
                settings,
            ));
        }
    }
    for quad in &model.general_quads {
        lighting.push(falling_block_quad_lighting(
            world, state, block_pos, quad, None, smooth, settings,
        ));
    }
    lighting
}

/// Replay a falling block through the world-model lighting path while retaining the entity
/// phase's camera-relative geometry.
///
/// java: `RenderFallingBlock` calls `BlockModelRenderer.renderModel(world, model, state,
/// blockPos, worldRenderer, false)`, not `RenderItem.renderItem`. The latter is what
/// [`dropped::emit_baked_model`] normally implements, so the geometry walk is shared first and
/// then its per-vertex primary colour is replaced with the block renderer's shade/tint/AO result.
/// The packed per-vertex values are retained by [`falling_block_model_lighting`] for the exact
/// entity-phase handoff; `EntityPhase` carries them in a parallel vertex stream so the sky and
/// block axes remain available to the lightmap sampler and rasterizer interpolation.
///
/// `settings` are the live graphics settings: `renderModel` takes the
/// smooth path only while `Minecraft.isAmbientOcclusionEnabled()`, so with smooth lighting off a
/// falling block is lit flat exactly like the same block in the terrain. The frame passes the
/// driver's `mesh_settings` (fancy graphics and smooth lighting, no mod folds).
fn emit_falling_block_model_with_settings(
    world: &World,
    state: mc_data::BlockState,
    block_pos: BlockPos,
    model: &mc_assets::BakedModel,
    transform: mc_render::entity::Mat4,
    settings: mc_render::terrain::MeshSettings,
    out: &mut Vec<EntityVertex>,
) {
    let start = out.len();
    dropped::emit_baked_model(model, transform, out);

    let lighting = falling_block_model_lighting(world, state, block_pos, model, settings);
    let mut cursor = start;
    for quad in lighting {
        for (index, vertex) in FALLING_BLOCK_QUAD_INDICES.into_iter().enumerate() {
            out[cursor + index].color = quad.color_multiplier[vertex];
        }
        cursor += 6;
    }
    debug_assert_eq!(cursor, out.len());
}

/// Reconstruct the packed light array for one live falling-block entity.
///
/// The dispatcher has already resolved the same state/model immediately before calling the
/// emitter. Repeating that tiny CPU walk at the upload boundary keeps the public object adapter
/// unchanged for every other renderer while making the per-vertex light handoff explicit here.
#[must_use]
fn falling_block_vertex_lights_for_entity(
    world: &World,
    tracked: &TrackedEntity,
    position: DVec3,
    settings: mc_render::terrain::MeshSettings,
) -> Option<Vec<u32>> {
    let data = world.ecs.get::<&ObjectData>(tracked.entity).ok()?.0;
    let state = mc_data::BlockState::from_java_state_id(data & 0xFFFF);
    (mc_data::block_props(state).render_type == mc_data::RenderType::Model).then_some(())?;
    if !falling_block_should_render(world, position, state) {
        return None;
    }
    let model = mc_assets::model_for(state);
    let lighting =
        falling_block_model_lighting(world, state, BlockPos::from_vec3(position), model, settings);
    let mut lights = Vec::with_capacity(lighting.len() * FALLING_BLOCK_QUAD_INDICES.len());
    for quad in lighting {
        for vertex in FALLING_BLOCK_QUAD_INDICES {
            lights.push(quad.packed_light[vertex]);
        }
    }
    Some(lights)
}

/// One entity's geometry, and the texture key it belongs under. `None` means "no renderer for
/// this type", which is `RenderManager.getEntityRenderObject` returning null.
// This is the RenderManager object-dispatch adapter. Its arguments are the distinct Java render
// inputs (world/tracker pose, camera, lighting, and the two scratch batches), so a context struct
// would obscure rather than simplify the parity mapping.
#[allow(clippy::too_many_arguments)]
fn emit_object(
    out: &mut Vec<EntityVertex>,
    models: &mut RenderManager,
    tile_models: &mut TileModels,
    world: &World,
    players: Option<&crate::sync::PlayerList>,
    tracked: &TrackedEntity,
    kind: EntityType,
    camera: &Camera,
    partial_ticks: f32,
    block_lighting: crate::app::appearance::ShadowLighting,
    glint_out: &mut Vec<EntityVertex>,
    glint_time_ms: u64,
    painting_out: &mut Vec<objects::PaintingVertex>,
    block_mesh: mc_render::terrain::MeshSettings,
) -> Option<ObjectTexture> {
    use EntityType as E;

    // java: `RenderManager.renderEntityStatic` draws from the interpolated position; for a
    // non-living entity `Entity.setPositionAndRotation2` snapped it, so this is the packet
    // position, which is correct and is *not* the same as reading `target_pos` directly (an
    // entity the server has not moved this tick still needs the ECS value).
    let position = tracked.render_position(world, partial_ticks);
    let current_position = tracked_current_position(world, tracked, position);
    let rel = camera.relative(position);
    let age = tracked.render.ticks_existed;

    match kind {
        // -- RenderEntityItem -------------------------------------------------------
        E::Item => {
            let stack = dropped_item_stack(world, tracked)?;
            let item = mc_data::ItemId(u16::try_from(stack.item_id).ok()?);
            item.try_props()?;
            // Resolve legacy names from the same local player
            // list as first-person items; absent rows remain the deterministic Steve fallback.
            let builtin =
                crate::app::tiles::builtin_tile_item_from_stack_with_players(&stack, players);
            let model = mc_assets::item_model_override_for(item, stack.damage).model;
            let item = dropped::DroppedItem {
                rel,
                age: i32::try_from(age).unwrap_or(i32::MAX),
                hover_start: tracked.dropped_item_hover_start,
                stack_size: stack.count,
                // java: `BuiltInModel.isGui3d()` is hard-coded true, even though these item
                // models carry no baked quads.  That means builtin tile items take the 3D
                // dropped-copy branch (including the half-scale and seeded jitter).
                gui_3d: builtin.is_some() || model.gui_3d,
                ground_scale: model.transforms.ground.scale,
            };
            if let Some(builtin) = builtin {
                let key = emit_dropped_builtin_item(
                    tile_models,
                    &builtin,
                    model,
                    &item,
                    partial_ticks,
                    out,
                )?;
                // `RenderItem.renderItem` does not enter its ordinary `stack.hasEffect()` branch
                // for `isBuiltInRenderer()` models, so builtin tile items intentionally do not
                // append a baked-model glint here.
                return Some(ObjectTexture::Builtin { item: builtin, key });
            }
            let base_start = out.len();
            dropped::emit_dropped_item_tinted(
                &item,
                model,
                mc_render::entity::Mat4::IDENTITY,
                partial_ticks,
                out,
                |tint_index| item_quad_tint(&stack, tint_index),
            );
            if item_has_effect(&stack) {
                // #22: the foil replays every copy just emitted, under the same matrices; derive
                // it from those vertices instead of re-walking the model per copy and pass.
                dropped::emit_item_glint_from_base(
                    &out[base_start..],
                    item.render_count() as usize,
                    glint_time_ms,
                    glint_out,
                );
            }
            Some(ObjectTexture::Resource(BLOCK_ATLAS_TEXTURE))
        }

        // -- RenderXPOrb ------------------------------------------------------------
        E::XPOrb => {
            let value = world.ecs.get::<&XpValue>(tracked.entity).map_or(1, |v| v.0);
            objects::emit_xp_orb(
                rel,
                camera.yaw,
                camera.pitch,
                value,
                age,
                partial_ticks,
                out,
            );
            Some(ObjectTexture::Resource(objects::XP_ORB_TEXTURE))
        }

        // -- RenderArrow ---------------------------------------------------------------------
        E::Arrow => {
            let (yaw, pitch) = arrow_rotation(world, tracked, partial_ticks);
            // `arrowShake` is absent from every packet because `EntityArrow.onUpdate` produces
            // it locally when its own block raycast hits. The collision tick writes seven and
            // decrements it once per tick; RenderArrow then subtracts the frame's partial tick.
            // A missing component is the safe settled state for entities spawned before this
            // client-local component existed.
            let shake = world
                .ecs
                .get::<&ArrowShake>(tracked.entity)
                .map(|value| value.render_value(partial_ticks))
                .unwrap_or(0.0);
            // java: the world pass draws RenderArrow lit, with its 0.05625-long normal; the
            // light goes into the vertex colour for `objects::ARROW_BLEND`.
            objects::emit_arrow_lit(rel, yaw, pitch, shake, out);
            Some(ObjectTexture::Resource(objects::ARROW_TEXTURE))
        }

        // -- RenderSnowball, six item types, and RenderPotion --------------------------------
        E::Snowball
        | E::ThrownEgg
        | E::ThrownEnderpearl
        | E::EyeOfEnderSignal
        | E::ThrownExpBottle
        | E::FireworksRocketEntity
        | E::ThrownPotion => {
            // `handleSpawnObject` gives EntityPotion its own typed constructor argument before
            // the generic ObjectData retention.  Keep this render path tied to that component:
            // a potion is not an arbitrary object whose packet word happens to be metadata.
            // It also means a future ObjectData consumer cannot accidentally alter the bottle
            // model or liquid colour.
            let potion_damage = (kind == E::ThrownPotion)
                .then(|| {
                    world
                        .ecs
                        .get::<&SplashPotionDamage>(tracked.entity)
                        .ok()
                        .map(|damage| damage.0)
                })
                .flatten();
            let (item, damage) = thrown_item_stack(kind, potion_damage)?;
            let model = mc_assets::item_model_override_for(item, damage).model;
            emit_thrown_item_tinted(rel, camera.yaw, camera.pitch, model, out, |tint_index| {
                thrown_item_quad_tint(kind, potion_damage, tint_index)
            });
            if thrown_item_has_effect(kind, potion_damage) {
                // The foil replays the base pass under RenderSnowball's exact item transform,
                // including RenderItem.preTransform's flat-model `2` (see thrown_item_matrix).
                let matrix = objects::thrown_item_matrix(rel, camera.yaw, camera.pitch, model);
                dropped::emit_baked_model_glint(model, matrix, glint_time_ms, glint_out);
            }
            Some(ObjectTexture::Resource(BLOCK_ATLAS_TEXTURE))
        }

        // -- RenderFireball, two scales ------------------------------------------------------
        E::Fireball | E::SmallFireball => {
            let scale = if kind == E::Fireball {
                objects::LARGE_FIREBALL_SCALE
            } else {
                objects::SMALL_FIREBALL_SCALE
            };
            // java: `getItemModelMesher().getParticleIcon(Items.fire_charge)` — an item model's
            // particle texture is its `layer0`, and `item/fire_charge.json`'s is `items/fireball`.
            let location = mc_assets::ResourceLocation::minecraft("textures/items/fireball");
            let sprite = mc_assets::sprite(&location);
            objects::emit_fireball(rel, camera.yaw, camera.pitch, scale, sprite, out);
            Some(ObjectTexture::Resource(BLOCK_ATLAS_TEXTURE))
        }

        // -- RenderTNTPrimed --------------------------------------------------------
        E::PrimedTnt => {
            // java: `EntityTNTPrimed`'s constructor sets `fuse = 80` and `onUpdate` decrements
            // it on the client too; the server never sends it, so it is reconstructed from the
            // entity's own age. It can go negative on a server that keeps the entity alive past
            // the explosion, which is what `tnt_scale`'s clamp is for.
            let fuse = objects::TNT_FUSE_TICKS - i32::try_from(age).unwrap_or(i32::MAX);
            let state = mc_data::block_by_name("minecraft:tnt")?.default_state();
            let model = mc_assets::model_for(state);
            crate::app::driver::emit_entity_block_model_brightness(
                state,
                model,
                objects::tnt_matrix(rel, fuse, partial_ticks),
                entity_brightness(world, current_position, block_lighting),
                out,
            );
            // `upload_objects` immediately replays these vertices through `TntFlash` when this
            // fuse is in its five-tick white phase. Keeping the clone at the submission boundary
            // preserves body-then-flash ordering without teaching this model emitter GPU state.
            Some(ObjectTexture::Resource(BLOCK_ATLAS_TEXTURE))
        }

        // -- RenderFallingBlock -----------------------------------------------------
        E::FallingSand => {
            // java: `Block.getStateById(data & 65535)`, whose packing is `id | meta << 12` — a
            // *different* packing from the `id << 4 | meta` that `S22`/`S23` use for block
            // changes. Mixing the two is a very visible bug, so this goes through
            // `BlockState::from_java_state_id` rather than `dispatch.rs`'s `block_state`.
            let data = world.ecs.get::<&ObjectData>(tracked.entity).ok()?.0;
            let state = mc_data::BlockState::from_java_state_id(data & 0xFFFF);
            // java: `if (block.getRenderType() != -1)` and `== 3` — only ordinary model blocks
            // are drawn; a tile-entity-rendered or invisible block falls silently.
            if mc_data::block_props(state).render_type != mc_data::RenderType::Model {
                return None;
            }
            // java: `RenderFallingBlock.doRender` skips the entity model while the identical
            // full block state still occupies the entity's floored world cell. The client tick
            // normally clears the source first, but the render guard also covers the spawn
            // frame and a later server-side replacement.
            if !falling_block_should_render(world, current_position, state) {
                return None;
            }
            let model = mc_assets::model_for(state);
            emit_falling_block_model_with_settings(
                world,
                state,
                BlockPos::from_vec3(current_position),
                model,
                objects::falling_block_matrix(rel),
                block_mesh,
                out,
            );
            Some(ObjectTexture::Resource(BLOCK_ATLAS_TEXTURE))
        }

        // -- RenderPainting ---------------------------------------------------------
        E::ItemFrame => {
            // `handleSpawnObject` reconstructs this hanging entity from the floor of the wire
            // position and type 71's horizontal facing. `RenderItemFrame` uses that block centre,
            // rather than the entity's wall-offset position, while its outer turn reads the
            // entity's authoritative packet yaw (which may differ from the constructor facing).
            let facing = world
                .ecs
                .get::<&ItemFrameFacing>(tracked.entity)
                .map(|value| value.0)
                .unwrap_or(Facing::South);
            let frame_position = item_frame_render_position(world, tracked, position, facing);
            let frame = item_frame_matrix(
                frame_position,
                camera,
                facing,
                item_frame_rotation_yaw(world, tracked),
            );
            let stack = watched_stack(world, tracked);
            let shell_start = out.len();
            // RenderItemFrame runs with the world renderer's ordinary back-face cull state
            // (`GL_CULL_FACE`, `GL_BACK`). The frame and its fixed item are pushed with the
            // culled `DroppedItem`/`DroppedItemGlint` states (`object_body_blend`), which cull by
            // screen-space winding as GL does. The CPU cull used here before tested only the
            // camera-space normal (an orthographic test), so it dropped the side faces that
            // perspective shows on a frame away from the view axis.
            dropped::emit_baked_model(
                mc_assets::item_frame_model(stack.as_ref().is_some_and(is_filled_map)),
                frame,
                out,
            );
            let shell_end = out.len();
            if let Some(stack) = stack.filter(|stack| !is_filled_map(stack)) {
                // `RenderItem.renderItem` asks the mesher for the stack model first.  A
                // `builtin/entity` sentinel then routes to TileEntityItemStackRenderer, which
                // cannot be represented by the empty baked-model quad list.  Keep this route
                // separate from ordinary frame contents so upload can bind the atlas for the
                // shell and the tile texture for the displayed model in Java's order.
                let builtin =
                    crate::app::tiles::builtin_tile_item_from_stack_with_players(&stack, players);
                if let Some(builtin) = builtin {
                    let content_start = out.len();
                    if let Some(key) = emit_item_frame_builtin_stack(
                        tile_models,
                        world,
                        tracked,
                        &stack,
                        frame,
                        camera.view,
                        &builtin,
                        out,
                    ) {
                        let content_end = out.len();
                        return Some(ObjectTexture::ItemFrameBuiltin {
                            item: builtin,
                            key,
                            shell_range: (shell_start, shell_end),
                            content_range: (content_start, content_end),
                        });
                    }
                } else {
                    emit_item_frame_stack(
                        world,
                        tracked,
                        &stack,
                        frame,
                        out,
                        glint_out,
                        glint_time_ms,
                    );
                }
            }
            Some(ObjectTexture::Resource(BLOCK_ATLAS_TEXTURE))
        }

        E::Painting => {
            let hanging = world.ecs.get::<&Hanging>(tracked.entity).ok()?;
            objects::emit_painting_with_lightmap(
                rel,
                objects::hanging_yaw(hanging.facing),
                current_position.to_array(),
                hanging.facing,
                hanging.art,
                |sample| {
                    painting_combined_light_at(
                        world,
                        BlockPos::new(sample[0], sample[1], sample[2]),
                    )
                },
                painting_out,
            );
            Some(ObjectTexture::Resource(objects::PAINTING_TEXTURE))
        }

        // -- RenderWitherSkull ---------------------------------------------------------------
        E::WitherSkull => {
            let (yaw, pitch) = object_rotation(
                E::WitherSkull,
                (tracked.render.prev_body_yaw, tracked.render.prev_pitch),
                rotation(world, tracked),
                partial_ticks,
            );
            // java: `ModelSkeletonHead.setRotationAngles` takes the interpolated yaw as
            // `netHeadYaw` and the pitch as `headPitch`; the outer matrix carries neither.
            let params = AnimParams {
                net_head_yaw: yaw,
                head_pitch: pitch,
                ..AnimParams::still()
            };
            models.emit_with(
                mc_render::entity::ModelKind::WitherSkull,
                objects::wither_skull_matrix(rel),
                &params,
                out,
            );
            // java: `getEntityTexture` returns the invulnerable sheet when watched index 10 is
            // exactly one. The watched decoder already exposes that state; keep texture choice
            // beside this non-living render dispatch so the blue skull cannot silently fall back.
            Some(ObjectTexture::Resource(wither_skull_texture(
                tracked
                    .with_watched(world, |watched| watched.is_skull_charged())
                    .unwrap_or(false),
            )))
        }

        _ => None,
    }
}

/// Emits every copy of a dropped `builtin/entity` stack through the same ground placement as
/// [`dropped::emit_dropped_item_tinted`], then hands each copy to the shared tile-item models.
///
/// `DroppedItem::copies` already contains Java's hover/bob, spin, stack count, 3-D half-scale and
/// seeded jitter.  The only extra operation here is the model's resolved `display.ground`
/// transform before [`TileModels::emit_builtin_item`] applies `RenderItem`'s common half-scale,
/// 180-degree turn and recenter.  Keeping those operations in this order is what makes an item
/// chest follow the ordinary dropped-item path instead of appearing as a static origin tile.
fn emit_dropped_builtin_item(
    tile_models: &mut TileModels,
    builtin: &BuiltinTileItem,
    model: &mc_assets::BakedModel,
    item: &dropped::DroppedItem,
    partial_ticks: f32,
    out: &mut Vec<EntityVertex>,
) -> Option<String> {
    let ground = objects::item_transform_matrix(&model.transforms.ground);
    let mut texture = None;
    for copy in item.copies(partial_ticks) {
        let key = tile_models.emit_builtin_item(builtin, copy.matrix * ground, out)?;
        if let Some(expected) = &texture {
            debug_assert_eq!(
                expected, &key,
                "a builtin stack changed texture between copies"
            );
        } else {
            texture = Some(key);
        }
    }
    texture
}

/// `Entity.getBrightness(partialTicks)` for an entity whose eye position equals its feet.
/// `EntityTNTPrimed.getEyeHeight()` is exactly zero, and the base method ignores partial ticks.
fn entity_brightness(
    world: &World,
    position: mc_math::DVec3,
    lighting: crate::app::appearance::ShadowLighting,
) -> f32 {
    let pos = mc_math::BlockPos::new(
        position.x.floor() as i32,
        position.y.floor() as i32,
        position.z.floor() as i32,
    );
    if !world.is_chunk_loaded(pos.chunk_pos()) {
        return 0.0;
    }
    let level = world.light_subtracted(pos, lighting.skylight_subtracted);
    mc_render::sky::lightmap::light_brightness_table(lighting.ambient_light)
        [usize::from(level.min(15))]
}

/// java: `RenderWitherSkull.getEntityTexture`.
fn wither_skull_texture(charged: bool) -> &'static str {
    if charged {
        objects::WITHER_SKULL_INVULNERABLE_TEXTURE
    } else {
        objects::WITHER_SKULL_TEXTURE
    }
}

/// The `ItemStack` in one watched slot, if the server has sent one.
fn watched_stack(world: &World, tracked: &TrackedEntity) -> Option<mc_protocol::ItemStack> {
    tracked
        .with_watched(world, |watched| watched.frame_item().cloned())
        .flatten()
}

/// Java's `EntityItem.getEntityItem` has two deliberately different fallbacks.
///
/// A missing/null watcher 10 returns a real one-count stone stack, while a present stack whose
/// item id is not registered remains a present invalid stack and therefore has no renderer. Do
/// not route the latter through [`mc_data::ItemId::props`], whose compatibility sentinel would
/// fabricate missing-cube geometry.
fn dropped_item_stack(world: &World, tracked: &TrackedEntity) -> Option<mc_protocol::ItemStack> {
    let watched = tracked
        .with_watched(world, |watched| watched.dropped_item().cloned())
        .flatten();

    if let Some(stack) = watched {
        let item = mc_data::ItemId(u16::try_from(stack.item_id).ok()?);
        item.try_props()?;
        return Some(stack);
    }

    let stone = mc_data::item_by_name("minecraft:stone")?;
    Some(mc_protocol::ItemStack {
        item_id: stone.0 as i16,
        count: 1,
        damage: 0,
        nbt: None,
    })
}

/// `RenderItemFrame`'s map branch is not a normal `RenderItem` call.
fn is_filled_map(stack: &mc_protocol::ItemStack) -> bool {
    mc_data::item_by_name("minecraft:filled_map").is_some_and(|map| stack.item_id == map.0 as i16)
}

/// The CPU-side handoff for the dynamic image inside a filled item frame.
///
/// `RenderItemFrame` does not draw a filled map through `RenderItem`: after it has drawn the
/// `item_frame#map` model it asks `MapItemRenderer` for this separate 128-by-128 quad.  The
/// caller owns the dynamic-texture cache and GPU upload; keeping the map record and already
/// transformed vertices together here means that upload path cannot quietly reimplement either
/// the map-id lookup or the frame's unusually specific matrix stack.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ItemFrameMapRender<'a> {
    /// The live `map_<id>` record selected by the displayed stack's metadata.
    pub map: &'a mc_world::MapData,
    /// Two triangles for `MapItemRenderer.Instance.render`'s image quad.
    pub vertices: [EntityVertex; 6],
    /// The frame marker(s) that `MapItemRenderer.renderMap` retains when
    /// `noOverlayRendering` is true. These are a distinct texture/batch so they must stay out of
    /// the dynamic image vertices above.
    pub icon_vertices: Vec<EntityVertex>,
}

impl ItemFrameMapRender<'_> {
    /// Converts the current `MapData.colors` bytes to the 128x128 RGBA8 upload payload, or
    /// `None` for a colour byte Java's palette lacks.
    ///
    /// Called only when the map atlas lacks this `MapData::revision`, and
    /// straight into the heap (a `MapTexture` would move 64 KiB through the stack several
    /// times). A malformed colour never replaces a previously good image: the atlas keeps it.
    pub fn rgba(&self) -> Option<Vec<u8>> {
        let mut rgba = Vec::with_capacity(mc_ui::map::MAP_TEXTURE_BYTES);
        for (pixel, &color) in self.map.colors.iter().enumerate() {
            rgba.extend_from_slice(&mc_ui::map::map_texel(pixel, color).ok()?);
        }
        Some(rgba)
    }
}

/// Builds the non-GPU half of `RenderItemFrame.renderItem`'s filled-map branch.
///
/// `rotation` is `EntityItemFrame.getRotation()` (the watched byte reduced to `0..8`). Java
/// first folds it through `rotation % 4 * 2`, then performs its usual eighth-turn rotation; this
/// is why an item frame's eight click positions yield only four map orientations. `frame` is the
/// outer [`item_frame_matrix`] shared with the baked frame model.
pub(crate) fn item_frame_map_render<'a>(
    world: &'a World,
    stack: &mc_protocol::ItemStack,
    frame: mc_render::entity::Mat4,
    rotation: i32,
) -> Option<ItemFrameMapRender<'a>> {
    // java: `Items.filled_map.getMapData(entityitem.getEntityItem(), itemFrame.worldObj)` reads
    // `ItemStack.getMetadata()` and returns null until an S34 has created `map_<metadata>`.
    let map = world.map_data(i32::from(stack.damage))?;
    let matrix = item_frame_map_matrix(frame, rotation);
    let corners = [
        // MapItemRenderer: (0, 128), (128, 128), (128, 0), (0, 0), all at z = -0.01.
        (mc_math::Vec3::new(0.0, 128.0, -0.01), [0.0, 1.0]),
        (mc_math::Vec3::new(128.0, 128.0, -0.01), [1.0, 1.0]),
        (mc_math::Vec3::new(128.0, 0.0, -0.01), [1.0, 0.0]),
        (mc_math::Vec3::new(0.0, 0.0, -0.01), [0.0, 0.0]),
    ];
    let normal = matrix
        .transform_vector3(mc_math::Vec3::NEG_Z)
        .normalize_or_zero();
    let vertices = core::array::from_fn(|index| {
        let corner = [0usize, 1, 2, 0, 2, 3][index];
        let (position, uv) = corners[corner];
        EntityVertex {
            pos: matrix.transform_point3(position).into(),
            uv,
            normal: normal.into(),
            color: EntityVertex::WHITE,
        }
    });
    // `RenderItemFrame` calls `MapItemRenderer.renderMap(map, true)`: that boolean hides all
    // normal player overlays but deliberately keeps the type-1 frame marker.  It is not part of
    // the 128x128 DynamicTexture; MapItemRenderer draws it as a later quad from map_icons.png.
    let mut icon_vertices = Vec::new();
    for (icon_index, icon) in map
        .decorations
        .iter()
        .filter(|icon| icon.icon_type == 1)
        .enumerate()
    {
        let icon_matrix = matrix
            * mc_render::entity::Mat4::from_translation(
                [
                    f32::from(icon.x) / 2.0 + 64.0,
                    f32::from(icon.z) / 2.0 + 64.0,
                    -0.02,
                ]
                .into(),
            )
            * mc_render::entity::Mat4::from_rotation_z(
                f32::from(icon.direction) * 360.0_f32.to_radians() / 16.0,
            )
            * mc_render::entity::Mat4::from_scale([4.0, 4.0, 3.0].into())
            * mc_render::entity::Mat4::from_translation([-0.125, 0.125, 0.0].into());
        let normal = icon_matrix
            .transform_vector3(mc_math::Vec3::NEG_Z)
            .normalize_or_zero();
        let col = f32::from(icon.icon_type % 4) / 4.0;
        let row = f32::from(icon.icon_type / 4) / 4.0;
        for corner in [0usize, 1, 2, 0, 2, 3] {
            let (position, uv) = [
                (
                    mc_math::Vec3::new(-1.0, 1.0, -(icon_index as f32) * 0.001),
                    [col, row],
                ),
                (
                    mc_math::Vec3::new(1.0, 1.0, -(icon_index as f32) * 0.001),
                    [col + 0.25, row],
                ),
                (
                    mc_math::Vec3::new(1.0, -1.0, -(icon_index as f32) * 0.001),
                    [col + 0.25, row + 0.25],
                ),
                (
                    mc_math::Vec3::new(-1.0, -1.0, -(icon_index as f32) * 0.001),
                    [col, row + 0.25],
                ),
            ][corner];
            icon_vertices.push(EntityVertex {
                pos: icon_matrix.transform_point3(position).into(),
                uv,
                normal: normal.into(),
                color: EntityVertex::WHITE,
            });
        }
    }
    Some(ItemFrameMapRender {
        map,
        vertices,
        icon_vertices,
    })
}

/// Build the map-only half of an item frame from its live watcher records.
///
/// This stays next to [`item_frame_map_render`] so its item id, signed metadata and rotation
/// lookup cannot diverge from the `RenderItemFrame` geometry it feeds.
fn item_frame_map_for_entity<'a>(
    world: &'a World,
    tracked: &TrackedEntity,
    camera: &Camera,
    partial_ticks: f32,
) -> Option<ItemFrameMapRender<'a>> {
    let stack = watched_stack(world, tracked)?;
    if !is_filled_map(&stack) {
        return None;
    }
    let facing = world
        .ecs
        .get::<&ItemFrameFacing>(tracked.entity)
        .map(|value| value.0)
        .unwrap_or(Facing::South);
    let position = item_frame_render_position(
        world,
        tracked,
        tracked.render_position(world, partial_ticks),
        facing,
    );
    let rotation = tracked
        .with_watched(world, |watched| watched.frame_rotation())
        .unwrap_or(0);
    item_frame_map_render(
        world,
        &stack,
        item_frame_matrix(
            position,
            camera,
            facing,
            item_frame_rotation_yaw(world, tracked),
        ),
        rotation,
    )
}

/// The front-facing triangles of a map image quad, without allocating.
///
/// The same test as [`mc_render::entity::item::retain_front_facing_triangles`] (the eye must be
/// on the side its winding normal points to), on the fixed two-triangle quad: returns the kept
/// vertices, in order, and how many there are (0, 3 or 6).
fn front_facing_map_quad(
    view: mc_render::entity::Mat4,
    quad: &[EntityVertex; 6],
) -> ([EntityVertex; 6], usize) {
    let mut kept = *quad;
    let mut len = 0;
    for triangle in quad.chunks_exact(3) {
        let points: [mc_math::Vec3; 3] = core::array::from_fn(|index| {
            view.transform_point3(mc_math::Vec3::from(triangle[index].pos))
        });
        let normal = (points[1] - points[0]).cross(points[2] - points[0]);
        let length_squared = normal.length_squared();
        if length_squared.is_finite()
            && length_squared > f32::EPSILON
            && normal.dot(points[0]) < 0.0
        {
            kept[len..len + 3].copy_from_slice(triangle);
            len += 3;
        }
    }
    (kept, len)
}

/// Resolve the packed lightmap used by object renderers that draw in world space.
///
/// Java's `RenderItem`, `RenderItemFrame`, and `RenderTNTPrimed` keep the value from
/// `Entity.getBrightnessForRender` active for their dropped/fixed/body passes. Falling blocks
/// inherit the ordinary entity sample while their model adds block-renderer AO/tint, and
/// `EntityXPOrb` uses the same sampled world value before adding its half-bright block-light
/// boost. The sample is taken from the current entity pose, including `getEyeHeight`, before the
/// render pose is interpolated. Keeping the discriminator pure makes it difficult for a future
/// object caller to accidentally turn every full-bright compatibility path into a dark-world
/// sample.
#[must_use]
fn object_light_position(
    world: &World,
    tracked: &TrackedEntity,
    kind: Option<EntityType>,
    render_position: DVec3,
) -> DVec3 {
    let inherited_world_light = matches!(
        kind,
        Some(
            EntityType::Item
                | EntityType::XPOrb
                | EntityType::Arrow
                | EntityType::Snowball
                | EntityType::ThrownEgg
                | EntityType::ThrownEnderpearl
                | EntityType::ThrownPotion
                | EntityType::ThrownExpBottle
                | EntityType::FireworksRocketEntity
                | EntityType::PrimedTnt
                | EntityType::FallingSand
                | EntityType::ItemFrame
                | EntityType::Painting
        )
    ) || tracked.kind == TrackedKind::FishingBobber;
    if !inherited_world_light {
        return render_position;
    }

    // java: Entity.getBrightnessForRender samples the current posX/Y/Z plus getEyeHeight(),
    // before RenderManager interpolates the pose passed to doRender. This includes the TNT
    // zero-height override, the falling block's ordinary 0.85*height eye and EntityFishHook's
    // 0.25F constructor height (0.2125F eye).
    let feet = tracked_current_position(world, tracked, render_position);
    feet + DVec3::new(0.0, f64::from(tracked.eye_height(world)), 0.0)
}

/// The current Java entity position used by `Entity.getBrightnessForRender` and the world-model
/// falling-block renderer. `render_position` remains the draw pose; this helper intentionally
/// reads the ECS position before that second interpolation stage.
#[must_use]
fn tracked_current_position(
    world: &World,
    tracked: &TrackedEntity,
    render_position: DVec3,
) -> DVec3 {
    world
        .ecs
        .get::<&Position>(tracked.entity)
        .ok()
        .map(|position| position.0)
        .unwrap_or(render_position)
}

/// `EntityFishHook`'s inherited `Entity.getBrightnessForRender` world-light sample.
///
/// RenderFish has no renderer-specific brightness override: RenderManager selects this packed
/// lightmap before calling the bobber renderer. Keep the loaded/valid-world guard in the shared
/// driver helper so an unloaded hook is dark instead of inheriting the object pass's full-bright
/// compatibility value.
#[must_use]
fn fishing_bobber_combined_light(world: &mc_world::World, position: DVec3) -> u32 {
    let sample = BlockPos::from_f64(position.x, position.y, position.z);
    super::driver::combined_light_at_loaded_sample(world, sample)
}

#[must_use]
fn inherited_object_light(
    kind: Option<EntityType>,
    world: &mc_world::World,
    position: DVec3,
) -> Option<u32> {
    match kind {
        // java: RenderSnowball/RenderPotion/RenderTNTPrimed inherit Render.getBrightnessForRender
        // through the concrete entity. Paintings also start with the inherited value before
        // RenderPainting replaces it per cell. Ender Eye, fireballs and wither skulls are
        // deliberately absent: EntityEnderEye/EntityFireball return 15728880, which remains the
        // object's explicit full-bright route below.
        Some(
            EntityType::Item
            | EntityType::FallingSand
            | EntityType::PrimedTnt
            | EntityType::Arrow
            | EntityType::Snowball
            | EntityType::ThrownEgg
            | EntityType::ThrownEnderpearl
            | EntityType::ThrownPotion
            | EntityType::ThrownExpBottle
            | EntityType::FireworksRocketEntity
            | EntityType::Painting,
        ) => {
            let sample = BlockPos::from_f64(position.x, position.y, position.z);
            Some(super::driver::combined_light_at_loaded_sample(
                world, sample,
            ))
        }
        Some(EntityType::ItemFrame) => {
            let sample = BlockPos::from_f64(position.x, position.y, position.z);
            Some(super::driver::combined_light_at_loaded_sample(
                world, sample,
            ))
        }
        Some(EntityType::XPOrb) => Some(xp_orb_combined_light(world, position)),
        _ => None,
    }
}

/// `EntityXPOrb`'s `Entity.getBrightnessForRender` override.
///
/// Java's `super.getBrightnessForRender` has already selected the current entity position plus
/// `Entity.getEyeHeight` (0.425F for the 0.5F orb). This helper receives that eye sample, adds
/// `0.5F * 15F * 16F == 120` to the packed block-light component, clamps that component to `240`,
/// and leaves the packed sky-light component untouched. An unloaded/invalid sample returns zero
/// before the adjustment, matching `Entity.getBrightnessForRender`'s early return when
/// `World.getCombinedLight` is unavailable.
#[must_use]
fn xp_orb_combined_light(world: &mc_world::World, position: DVec3) -> u32 {
    let sample = BlockPos::from_f64(position.x, position.y, position.z);
    // `combined_light_at_loaded_sample` intentionally returns the same zero for a loaded black
    // cell and for an unloaded/invalid cell.  XP's Java override must distinguish those cases:
    // `super.getBrightnessForRender` returns zero for an unloaded cell (and exits before the
    // boost), but a loaded black cell is still raised by the exact +120 block-light adjustment.
    if !(-30_000_000..30_000_000).contains(&sample.x)
        || !(-30_000_000..30_000_000).contains(&sample.z)
        || !(0..256).contains(&sample.y)
        || !world.is_chunk_loaded(sample.chunk_pos())
    {
        return 0;
    }
    let combined = super::driver::combined_light_at_loaded_sample(world, sample);
    let block = (combined & 0xFFFF).saturating_add(120).min(240);
    (combined & 0xFFFF_0000) | block
}

/// Queue one live item-frame map after its baked frame model.
///
/// (map art lag.) The image lives
/// in the entity phase's shared map atlas (`mc_render::entity::map_atlas`) and is uploaded only
/// when `MapData::revision` moves, i.e. after an S34 changed its pixels: vanilla's
/// `MapItemRenderer.Instance.updateMapTexture` cadence. It used to be converted, given a fresh
/// GPU texture and bind group and uploaded every frame for every visible frame. A malformed
/// palette byte still cannot evict an older good upload (it stays), and a map that never
/// converted still draws the phase's missing texture.
///
/// A fully opaque image (every pixel explored) is queued and appended after the object loop
/// ([`EntityPhase::push_map_deferred`]): under the map blend it ignores what is behind it and it
/// writes depth, so only the draw count changes. Consecutive images then merge into one draw per
/// atlas page and the frame shells between them merge too. Images with unexplored (translucent)
/// pixels draw in place, as before. The icon batch comes after the image's position in the loop
/// and has the additional `-0.02` local depth nudge from vanilla, so it stays in front either way.
fn push_item_frame_map(
    gpu: &Gpu,
    render: &RenderStack,
    assets: &GameAssets,
    map: ItemFrameMapRender<'_>,
    combined_light: Option<u32>,
    view: mc_render::entity::Mat4,
) {
    let (mut image_vertices, image_len) = front_facing_map_quad(view, &map.vertices);
    let light = combined_light.unwrap_or(mc_render::entity::render::FULL_BRIGHT_COMBINED_LIGHT);
    if image_len > 0 {
        render.entities.with(|phase| {
            // `rgba` runs only when the atlas lacks this revision.
            let image = phase.map_atlas_slot(gpu, map.map.id, map.map.revision, || map.rgba());
            let image_vertices = &mut image_vertices[..image_len];
            match image {
                Some(image) => {
                    for vertex in image_vertices.iter_mut() {
                        vertex.uv = image.slot.uv(vertex.uv);
                    }
                    let blend = LayerBlend::MapTranslucentDepthWrite;
                    if image.opaque {
                        phase.push_map_deferred(image.slot, blend, light, image_vertices);
                    } else {
                        let texture = image.slot.texture_key();
                        phase.push_blended_lit(texture, blend, light, image_vertices);
                    }
                }
                None => phase.push_blended_lit(
                    EntityPhase::FALLBACK_TEXTURE,
                    LayerBlend::MapTranslucentDepthWrite,
                    light,
                    image_vertices,
                ),
            }
        });
    }
    if map.icon_vertices.is_empty() {
        return;
    }
    let mut icon_vertices = map.icon_vertices;
    mc_render::entity::item::retain_front_facing_triangles(view, &mut icon_vertices);
    // `map_icons.png` is a normal resource-pack texture, not a DynamicTexture. MapItemRenderer
    // has already restored alpha testing and disabled blending before this loop, so these use the
    // ordinary opaque/cutout pipeline. `push_geometry` preserves their post-image ordering while
    // using the EntityPhase's regular lazy loader.
    push_geometry(
        gpu,
        render,
        assets,
        mc_ui::map::MAP_ICONS_TEXTURE,
        LayerBlend::Opaque,
        &icon_vertices,
    );
}

/// The matrix immediately before `MapItemRenderer.Instance.render` emits its texture quad.
///
/// java: `RenderItemFrame.renderItem`: translate `(0, 0, .4375)`, rotate the collapsed item
/// frame rotation, rotate 180 degrees around Z, scale by `1 / 128`, translate `(-64, -64, -1)`.
fn item_frame_map_matrix(frame: mc_render::entity::Mat4, rotation: i32) -> mc_render::entity::Mat4 {
    let map_rotation = rotation.rem_euclid(4) * 2;
    frame
        * mc_render::entity::Mat4::from_translation([0.0, 0.0, 0.4375].into())
        * mc_render::entity::Mat4::from_rotation_z(map_rotation as f32 * 45.0_f32.to_radians())
        * mc_render::entity::Mat4::from_rotation_z(core::f32::consts::PI)
        * mc_render::entity::Mat4::from_scale([1.0 / 128.0; 3].into())
        * mc_render::entity::Mat4::from_translation([-64.0, -64.0, -1.0].into())
}

/// The outer matrix at `RenderItemFrame.doRender`'s model-render call.
fn item_frame_render_position(
    world: &World,
    tracked: &TrackedEntity,
    position: mc_math::DVec3,
    facing: Facing,
) -> mc_math::DVec3 {
    // A real type-71 dispatch always attaches `ItemFrameFacing`, and `TrackedEntity` has already
    // reconstructed its canonical wall-offset pose. Keep the old floor-derived constructor
    // fallback for synthetic/malformed records without that marker; it preserves the Java
    // constructor's block-centre transform instead of treating a raw wire corner as a pose.
    if world.ecs.get::<&ItemFrameFacing>(tracked.entity).is_ok() {
        return position;
    }
    let anchor = BlockPos::from_vec3(position);
    let (dx, _, dz) = facing.offset();
    mc_math::DVec3::new(
        f64::from(anchor.x) + 0.5 - f64::from(dx) * 0.46875,
        f64::from(anchor.y) + 0.5,
        f64::from(anchor.z) + 0.5 - f64::from(dz) * 0.46875,
    )
}

/// The outer matrix at `RenderItemFrame.doRender`'s model-render call.
fn item_frame_matrix(
    position: mc_math::DVec3,
    camera: &Camera,
    facing: Facing,
    rotation_yaw: f32,
) -> mc_render::entity::Mat4 {
    // java: `RenderItemFrame.doRender` computes
    // `blockPos + 0.5 - entity.pos + renderPos`; with a canonical 12x12 hanging pose this is
    // exactly the anchor centre plus the interpolated pose delta. Rebuilding it from the pose
    // keeps the first packet frame and a moving anchor coherent without flooring away motion.
    let (dx, _, dz) = facing.offset();
    let hanging_block_centre =
        position + mc_math::DVec3::new(f64::from(dx) * 0.46875, 0.0, f64::from(dz) * 0.46875);
    let rel = camera.relative(hanging_block_centre);
    mc_render::entity::Mat4::from_translation(rel)
        * mc_render::entity::Mat4::from_rotation_y((180.0 - rotation_yaw).to_radians())
}

/// `RenderItemFrame` reads `Entity.rotationYaw`, which `handleSpawnObject` overwrites from the
/// packet after the item-frame constructor has set it from `data`'s facing.
fn item_frame_rotation_yaw(world: &World, tracked: &TrackedEntity) -> f32 {
    world
        .ecs
        .get::<&Rotation>(tracked.entity)
        .map(|rotation| rotation.yaw)
        .unwrap_or(tracked.target_yaw)
}

/// java: `RenderItemFrame.renderItem`'s non-map `RenderItem(..., FIXED)` path.
// RenderItemFrame's fixed-item adapter carries both frame transforms and the two optional glint
// batches; keeping the Java call shape avoids hiding texture/state ordering in a temporary object.
#[allow(clippy::too_many_arguments)]
fn emit_item_frame_stack(
    world: &World,
    tracked: &TrackedEntity,
    stack: &mc_protocol::ItemStack,
    frame: mc_render::entity::Mat4,
    out: &mut Vec<EntityVertex>,
    glint_out: &mut Vec<EntityVertex>,
    glint_time_ms: u64,
) {
    let item = mc_data::ItemId(stack.item_id as u16);
    let Some(props) = mc_data::try_item_props(item) else {
        return;
    };
    let selection = mc_assets::item_model_override_for(props.id, stack.damage);
    let model = selection.model;
    let rotation = tracked
        .with_watched(world, |watched| watched.frame_rotation())
        .unwrap_or(0);
    let rotation = item_frame_content_rotation(selection.property, rotation);
    let mut matrix = frame
        * mc_render::entity::Mat4::from_translation([0.0, 0.0, 0.4375].into())
        * mc_render::entity::Mat4::from_rotation_z(rotation as f32 * 45.0_f32.to_radians())
        * mc_render::entity::Mat4::from_scale([0.5; 3].into());
    // `RenderItem.renderItemModelTransform` applies the flat-model pre-transform before the
    // selected fixed display transform.  Its unconditional common half-scale follows that
    // display transform; `emit_baked_model_tinted` supplies only the final recenter.
    matrix *= mc_render::entity::objects::item_model_pre_transform(model.gui_3d)
        * objects::item_transform_matrix(&model.transforms.fixed)
        * mc_render::entity::Mat4::from_scale([objects::RENDER_ITEM_SCALE; 3].into());
    if item_frame_item_needs_y_flip(model, item) {
        matrix *= mc_render::entity::Mat4::from_rotation_y(core::f32::consts::PI);
    }
    // The model and its foil are culled on the GPU (`object_body_blend`/`object_glint_blend`).
    // java: `renderItemModelTransform` switches to `cullFace(GL_FRONT)` while the FIXED scale
    // has an odd number of negative components (`RenderItem.java:329-335`, `:348-351`); that
    // mirror flips every emitted winding, so reversing the triangles keeps the same faces under
    // the pipelines' `GL_BACK`. No vanilla model has one; resource packs can.
    let scale = model.transforms.fixed.scale;
    let mirrored = (scale[0] < 0.0) ^ (scale[1] < 0.0) ^ (scale[2] < 0.0);
    let base_start = out.len();
    dropped::emit_baked_model_tinted(model, matrix, out, |tint| item_quad_tint(stack, tint));
    let glint_start = glint_out.len();
    if item_has_effect(stack) {
        dropped::emit_baked_model_glint(model, matrix, glint_time_ms, glint_out);
    }
    if mirrored {
        for triangle in out[base_start..]
            .chunks_exact_mut(3)
            .chain(glint_out[glint_start..].chunks_exact_mut(3))
        {
            triangle.swap(1, 2);
        }
    }
}

/// Emit a `builtin/entity` stack displayed by an item frame.
///
/// The frame renderer has already applied the frame translation, item-frame rotation and
/// Java's `0.5` item scale.  [`TileModels::emit_builtin_item`] owns the subsequent
/// `RenderItem.renderItem` sentinel transform (`180°` Y and `(-.5, -.5, -.5)` recenter) and the
/// concrete banner/chest/skull TESR.  Returning only the texture key lets the caller retain the
/// atlas-backed frame shell and tile-backed content as adjacent batches.
// The builtin-item TESR adapter intentionally keeps the frame shell/content ranges and both
// transforms explicit: Java binds these textures in sequence and the caller must preserve it.
#[allow(clippy::too_many_arguments)]
fn emit_item_frame_builtin_stack(
    tile_models: &mut TileModels,
    world: &World,
    tracked: &TrackedEntity,
    stack: &mc_protocol::ItemStack,
    frame: mc_render::entity::Mat4,
    view: mc_render::entity::Mat4,
    builtin: &BuiltinTileItem,
    out: &mut Vec<EntityVertex>,
) -> Option<String> {
    let item = mc_data::ItemId(stack.item_id as u16);
    // Resolve the item through the same registry gate as ordinary item-frame stacks.  The
    // builtin descriptor has already selected the sentinel; this lookup only rejects malformed
    // wire ids before the tile renderer is called.
    let _props = mc_data::try_item_props(item)?;
    let model = mc_assets::item_model_override_for(item, stack.damage).model;
    let rotation = tracked
        .with_watched(world, |watched| watched.frame_rotation())
        .unwrap_or(0);
    let parent = frame
        * mc_render::entity::Mat4::from_translation([0.0, 0.0, 0.4375].into())
        * mc_render::entity::Mat4::from_rotation_z(rotation as f32 * 45.0_f32.to_radians())
        * mc_render::entity::Mat4::from_scale([0.5; 3].into())
        * mc_render::entity::objects::item_model_pre_transform(model.gui_3d)
        * objects::item_transform_matrix(&model.transforms.fixed);
    // `emit_builtin_item` intentionally returns `None` for an uninitialised item banner, just as
    // TileEntityItemStackRenderer has no drawable tile until setItemValues completes.  The
    // caller then leaves the already-emitted frame shell intact.
    let mut content = Vec::new();
    let key = tile_models.emit_builtin_item(builtin, parent, &mut content)?;
    // `TileEntityItemStackRenderer` keeps culling enabled for chests/banners, but explicitly
    // disables it for skulls. Keep that one vanilla exception while filtering the other builtin
    // model families at the same frame-specific seam as ordinary baked content.
    if !matches!(
        builtin,
        BuiltinTileItem::Skull { .. } | BuiltinTileItem::PlayerSkull { .. }
    ) {
        mc_render::entity::item::retain_front_facing_triangles(view, &mut content);
    }
    out.extend(content);
    Some(key)
}

/// Whether `RenderItemFrame.renderItem` applies its extra 180-degree Y turn before `FIXED`.
///
/// java: `!RenderItem.shouldRenderItemIn3D(stack) || item instanceof ItemSkull`.  The skull
/// exception is independent of the baked model's `isGui3d` bit, so keeping it explicit prevents
/// a future asset-pack model from silently reversing a player head in a frame.
#[must_use]
fn item_frame_item_needs_y_flip(model: &mc_assets::BakedModel, item: mc_data::ItemId) -> bool {
    !model.gui_3d || item.0 == mc_render::entity::tilepass::SKULL_ITEM_ID
}

/// The compass exception in `RenderItemFrame.renderItem`.
///
/// A frame rotates ordinary fixed items by its eight-step watched rotation.  Vanilla updates a
/// compass sprite with the frame position/facing while it is drawn, so that temporary dynamic
/// sprite orientation cancels that frame rotation.  The live atlas owns one shared compass frame
/// for all consumers; suppressing the geometry rotation here is the equivalent per-object
/// override and keeps a framed compass pointing at spawn instead of spinning with the frame.
#[must_use]
fn item_frame_content_rotation(property: mc_assets::ItemModelProperty, frame_rotation: i32) -> i32 {
    if property == mc_assets::ItemModelProperty::Compass {
        0
    } else {
        frame_rotation
    }
}

/// The entity's current rotation, straight off the ECS.
fn rotation(world: &World, tracked: &TrackedEntity) -> (f32, f32) {
    world
        .ecs
        .get::<&Rotation>(tracked.entity)
        .map_or((tracked.target_yaw, tracked.target_pitch), |r| {
            (r.yaw, r.pitch)
        })
}

/// `RenderArrow.doRender`'s interpolated yaw and pitch.
///
/// `EntityArrow.onUpdate` first moves each `prevRotation*` sample by full turns until it is
/// within one half-turn of the current angle, then `RenderArrow` does a plain lerp. Keeping the
/// loop form matters: the unwrapped answer is `-180`, not `180`, for a `170 -> -170` turn. The
/// two orientations look equivalent in ideal arithmetic, but are not bit-for-bit equivalent
/// once the angle reaches the vertex transforms.
fn arrow_rotation(world: &World, tracked: &TrackedEntity, partial_ticks: f32) -> (f32, f32) {
    let current = rotation(world, tracked);
    object_rotation(
        EntityType::Arrow,
        (tracked.render.prev_body_yaw, tracked.render.prev_pitch),
        current,
        partial_ticks,
    )
}

/// Object renderers that consume `Entity` rotation each frame have renderer-specific interpolation
/// rules. `RenderArrow` receives the previous sample after `EntityArrow.onUpdate` unwraps it;
/// `RenderWitherSkull` performs its own shortest-arc yaw helper in `RenderWitherSkull.doRender`.
/// Other object renderers retain their current ECS rotation.
fn object_rotation(
    kind: EntityType,
    previous: (f32, f32),
    current: (f32, f32),
    partial_ticks: f32,
) -> (f32, f32) {
    match kind {
        EntityType::Arrow => interpolate_arrow_rotation(previous, current, partial_ticks),
        EntityType::WitherSkull => {
            interpolate_wither_skull_rotation(previous, current, partial_ticks)
        }
        _ => current,
    }
}

/// Java's `EntityArrow.onUpdate` unwrap followed by `RenderArrow.doRender`'s plain lerp.
fn interpolate_arrow_rotation(
    previous: (f32, f32),
    current: (f32, f32),
    partial_ticks: f32,
) -> (f32, f32) {
    (
        interpolate_arrow_angle(previous.0, current.0, partial_ticks),
        interpolate_arrow_angle(previous.1, current.1, partial_ticks),
    )
}

fn interpolate_arrow_angle(mut previous: f32, current: f32, partial_ticks: f32) -> f32 {
    while current - previous < -180.0 {
        previous -= 360.0;
    }
    while current - previous >= 180.0 {
        previous += 360.0;
    }
    previous + (current - previous) * partial_ticks
}

/// Java's `RenderWitherSkull.func_82400_a` shortest-arc yaw helper followed by its plain pitch
/// lerp. Unlike `EntityArrow.onUpdate`, the renderer normalises a local delta and leaves the
/// previous sample unchanged; that preserves the sign of the exact `170 -> -170` crossing
/// (`180` at half-tick rather than the arrow's equivalent `-180`).
fn interpolate_wither_skull_rotation(
    previous: (f32, f32),
    current: (f32, f32),
    partial_ticks: f32,
) -> (f32, f32) {
    let mut yaw_delta = current.0 - previous.0;
    while yaw_delta < -180.0 {
        yaw_delta += 360.0;
    }
    while yaw_delta >= 180.0 {
        yaw_delta -= 360.0;
    }
    (
        previous.0 + partial_ticks * yaw_delta,
        previous.1 + (current.1 - previous.1) * partial_ticks,
    )
}

/// The item id and model-selecting damage a thrown-entity renderer draws.
///
/// java: `RenderManager`'s constructor (`:177-183`) — `RenderSnowball` is registered six times
/// with a different `Item`, and `RenderPotion` is a seventh that additionally carries the
/// potion's damage value. `ItemStack` clamps negative constructor damage to zero; bit 14 is the
/// potion `ItemMeshDefinition`'s only model selector, so it is canonicalised before narrowing.
fn thrown_item_stack(
    kind: EntityType,
    potion_damage: Option<i32>,
) -> Option<(mc_data::ItemId, i16)> {
    use EntityType as E;
    let (name, damage) = match kind {
        E::Snowball => ("minecraft:snowball", 0),
        E::ThrownEgg => ("minecraft:egg", 0),
        E::ThrownEnderpearl => ("minecraft:ender_pearl", 0),
        E::EyeOfEnderSignal => ("minecraft:ender_eye", 0),
        E::ThrownExpBottle => ("minecraft:experience_bottle", 0),
        E::FireworksRocketEntity => ("minecraft:fireworks", 0),
        // java: `RenderPotion.func_177082_d` builds `new ItemStack(Items.potionitem, 1,
        // entity.getPotionDamage())`; RenderItem's ItemMeshDefinition then tests bit 14.
        E::ThrownPotion => {
            let data = potion_damage.unwrap_or(0).max(0);
            ("minecraft:potion", (data & 0x4000) as i16)
        }
        _ => return None,
    };
    Some((mc_data::item_by_name(name)?, damage))
}

fn item_quad_tint(stack: &mc_protocol::ItemStack, tint_index: u8) -> u32 {
    mc_ui::screens::container::render::item_quad_color_rgb(stack, Some(tint_index))
}

/// The entity-side equivalent of `RenderItem.renderItem`'s `ItemStack.hasEffect()` dispatch.
///
/// `mc-ui` owns the shared NBT/item-class predicate used by GUI stacks.  The two classes that
/// are renderer-only in this port (`ItemEditableBook` and `ItemSimpleFoiled`) are added here just
/// as the GUI path does, so dropped and item-frame contents do not silently lose their foil.
fn item_has_effect(stack: &mc_protocol::ItemStack) -> bool {
    let class_effect = u16::try_from(stack.item_id)
        .ok()
        .map(mc_data::ItemId)
        .is_some_and(|id| {
            matches!(
                mc_data::item_props(id).name,
                "minecraft:written_book" | "minecraft:nether_star"
            )
        });
    mc_ui::screens::container::tooltip::has_effect(stack) || class_effect
}

/// RenderSnowball/RenderPotion construct a fresh stack before calling `RenderItem`.  Only the
/// experience bottle is always foiled; splash potions use `ItemPotion.hasEffect` on their raw
/// damage, while the other snowball-family stacks are plain by construction.
fn thrown_item_has_effect(kind: EntityType, potion_damage: Option<i32>) -> bool {
    match kind {
        EntityType::ThrownExpBottle => true,
        EntityType::ThrownPotion => potion_damage
            .map(|damage| !mc_data::potion_effects_from_damage(damage.max(0)).is_empty())
            .unwrap_or(false),
        _ => false,
    }
}

/// Build the generation-owned `RenderFish` angler sample for a remote player. The owner handle
/// has already been checked by the bobber loop; this helper only reads the live ECS pose and the
/// tracker-held tick samples needed by Java's hand/body branch.
fn fishing_line_angler_for_remote(
    world: &World,
    owner: &TrackedEntity,
) -> Option<FishingLineAngler> {
    if owner.kind != TrackedKind::Player {
        return None;
    }
    let position = world.ecs.get::<&Position>(owner.entity).ok()?.0;
    let rotation = world
        .ecs
        .get::<&Rotation>(owner.entity)
        .ok()
        .map(|rotation| (rotation.yaw, rotation.pitch))
        .unwrap_or((owner.target_yaw, owner.target_pitch));
    Some(FishingLineAngler {
        prev_pos: owner.last_tick_pos,
        pos: position,
        prev_yaw: owner.prev_rotation_yaw,
        yaw: rotation.0,
        prev_pitch: owner.prev_rotation_pitch,
        pitch: rotation.1,
        prev_body_yaw: owner.render.prev_body_yaw,
        body_yaw: owner.render.body_yaw,
        eye_height: owner.eye_height(world),
        sneaking: owner
            .with_watched(world, |watched| watched.is_sneaking())
            .unwrap_or(false),
        prev_swing_progress: owner.render.prev_swing_progress,
        swing_progress: owner.render.swing_progress,
        is_local: false,
        third_person: true,
    })
}

fn thrown_item_quad_tint(kind: EntityType, potion_damage: Option<i32>, tint_index: u8) -> u32 {
    if kind == EntityType::ThrownPotion && tint_index == 0 {
        // java: ItemPotion.getColorFromItemStack -> getColorFromDamage.  Thrown potions have no
        // stack NBT, so the raw constructor damage is the entire input; ItemStack clamps a
        // negative damage before ItemPotion sees it.
        mc_data::potion_liquid_color_from_damage(potion_damage.unwrap_or(0).max(0))
    } else {
        0xFF_FF_FF
    }
}

fn emit_thrown_item_tinted(
    rel: mc_math::Vec3,
    view_yaw: f32,
    view_pitch: f32,
    model: &mc_assets::BakedModel,
    out: &mut Vec<EntityVertex>,
    tint: impl FnMut(u8) -> u32,
) {
    objects::emit_thrown_item_tinted(rel, view_yaw, view_pitch, model, out, tint);
}

/// Pushes one batch, loading its texture on first use.
///
/// java: `Render.bindEntityTexture` → `TextureManager.bindTexture`. Mirrors `driver.rs`'s
/// `push_entity_geometry`, including its rule that a texture which fails to load is not an error:
/// the batch goes under [`EntityPhase::FALLBACK_TEXTURE`] and the entity is still drawn.
fn push_geometry(
    gpu: &Gpu,
    render: &RenderStack,
    assets: &GameAssets,
    texture: &str,
    blend: LayerBlend,
    verts: &[EntityVertex],
) {
    push_geometry_lit(gpu, render, assets, texture, blend, None, verts);
}

/// Push one object batch while optionally retaining the entity's packed world lightmap.
///
/// `None` preserves the established full-bright compatibility route for object renderers that
/// do not yet sample `Entity.getBrightnessForRender`; item frames pass `Some` because Java's
/// `RenderItemFrame` draws the frame, content and map while the entity lightmap is active.
fn push_geometry_lit(
    gpu: &Gpu,
    render: &RenderStack,
    assets: &GameAssets,
    texture: &str,
    blend: LayerBlend,
    combined_light: Option<u32>,
    verts: &[EntityVertex],
) {
    render.entities.with(|phase| {
        let key = if texture == BLOCK_ATLAS_TEXTURE
            || texture == EntityPhase::SOLID_TEXTURE
            || phase.load_texture(gpu, &assets.resources, texture)
        {
            texture
        } else {
            EntityPhase::FALLBACK_TEXTURE
        };
        phase.push_blended_lit(
            key,
            blend,
            combined_light.unwrap_or(mc_render::entity::render::FULL_BRIGHT_COMBINED_LIGHT),
            verts,
        );
    });
}

/// Push one object batch with Java's packed sky/block lightmap coordinate per vertex.
///
/// Falling blocks are the one object renderer whose `BlockModelRenderer` supplies corner
/// brightness instead of inheriting a single `RenderManager` lightmap state. The parallel stream
/// is kept separate from [`EntityVertex`] so every existing object/layer ABI and draw ordering
/// remains unchanged.
#[allow(clippy::too_many_arguments)]
fn push_geometry_lit_with_vertex_lights(
    gpu: &Gpu,
    render: &RenderStack,
    assets: &GameAssets,
    texture: &str,
    blend: LayerBlend,
    combined_light: Option<u32>,
    verts: &[EntityVertex],
    packed_lights: &[u32],
) {
    render.entities.with(|phase| {
        let key = if texture == BLOCK_ATLAS_TEXTURE
            || texture == EntityPhase::SOLID_TEXTURE
            || phase.load_texture(gpu, &assets.resources, texture)
        {
            texture
        } else {
            EntityPhase::FALLBACK_TEXTURE
        };
        phase.push_blended_lit_with_vertex_lights(
            key,
            blend,
            combined_light.unwrap_or(mc_render::entity::render::FULL_BRIGHT_COMBINED_LIGHT),
            verts,
            packed_lights,
        );
    });
}

/// Submit one painting cell per lightmap state, matching `RenderPainting.setLightmap`'s state
/// change immediately before that cell's six quads. 06-entity-batch U1: [`EntityPhase`] writes
/// each cell's packed value into its vertices' light stream, so adjacent cells merge into one
/// draw whatever their block/sky values, with every cell keeping its own coordinate.
/// `RenderPainting.setLightmap` calls `World.getCombinedLight` directly. Unlike
/// `Entity.getBrightnessForRender`, that path does not preflight `isBlockLoaded`: unloaded or
/// invalid overworld samples receive `EnumSkyBlock.SKY.defaultLightValue == 15`, while no-sky
/// dimensions receive zero. Negative Y is clamped to zero before those checks.
fn painting_combined_light_at(world: &mc_world::World, sample: BlockPos) -> u32 {
    let sample = if sample.y < 0 {
        BlockPos::new(sample.x, 0, sample.z)
    } else {
        sample
    };
    let valid = (-30_000_000..30_000_000).contains(&sample.x)
        && (-30_000_000..30_000_000).contains(&sample.z)
        && (0..256).contains(&sample.y);
    if !valid || !world.is_chunk_loaded(sample.chunk_pos()) {
        return if world.has_sky_light() { 15 << 20 } else { 0 };
    }
    super::driver::combined_light_at_loaded_sample(world, sample)
}

fn push_painting_geometry(
    gpu: &Gpu,
    render: &RenderStack,
    assets: &GameAssets,
    texture: &str,
    verts: &[objects::PaintingVertex],
) {
    render.entities.with(|phase| {
        let key = if phase.load_texture(gpu, &assets.resources, texture) {
            texture
        } else {
            EntityPhase::FALLBACK_TEXTURE
        };
        for_each_painting_cell(verts, |combined_light, cell| {
            phase.push_blended_lit(key, LayerBlend::Opaque, combined_light, cell);
        });
    });
}

fn for_each_painting_cell(
    verts: &[objects::PaintingVertex],
    mut emit: impl FnMut(u32, &[EntityVertex]),
) {
    const VERTICES_PER_CELL: usize = 6 * 6;
    let mut cells = verts.chunks_exact(VERTICES_PER_CELL);
    let mut geometry = [EntityVertex::default(); VERTICES_PER_CELL];
    for cell in &mut cells {
        let lightmap = cell[0].lightmap;
        debug_assert!(cell.iter().all(|vertex| vertex.lightmap == lightmap));
        for (out, vertex) in geometry.iter_mut().zip(cell) {
            *out = vertex.vertex;
        }
        let combined_light = u32::from(lightmap.block) | (u32::from(lightmap.sky) << 16);
        emit(combined_light, &geometry);
    }
    debug_assert!(cells.remainder().is_empty());
}

/// Pushes a dropped builtin tile-item batch, uploading a banner composite into the entity phase
/// when its deterministic key is first encountered.
///
/// Fixed chest/skull sheets use the ordinary resource-pack loader.  Banners are different: their
/// `LayeredColorMaskTexture` is a CPU composite, and the tile phase's texture cache cannot bind an
/// `EntityVertex` batch.  Reusing [`mc_render::entity::tilepass::banner_texture_pixels`] keeps the
/// source/mask decoding, dye arithmetic and sixteen-pattern cap identical across world and GUI
/// callers while retaining the entity pass's ordering and glint pipeline.
// This upload adapter forwards the already-separated Java texture/state inputs to EntityPhase;
// collapsing them would make the atlas/builtin/light ordering less auditable.
#[allow(clippy::too_many_arguments)]
fn push_builtin_geometry(
    gpu: &Gpu,
    render: &RenderStack,
    assets: &GameAssets,
    item: &BuiltinTileItem,
    texture: &str,
    blend: LayerBlend,
    combined_light: Option<u32>,
    verts: &[EntityVertex],
) {
    render.entities.with(|phase| {
        let resident = if phase.has_texture(texture) {
            true
        } else {
            match item {
                BuiltinTileItem::Banner { state } => {
                    match mc_render::entity::tilepass::banner_texture_pixels(
                        &assets.resources,
                        state,
                    ) {
                        Some((key, pixels, width, height)) if key == texture => {
                            phase.upload_texture(gpu, &key, &pixels, width, height);
                            true
                        }
                        _ => false,
                    }
                }
                BuiltinTileItem::Chest { .. }
                | BuiltinTileItem::EnderChest
                | BuiltinTileItem::Skull { .. } => {
                    phase.load_texture(gpu, &assets.resources, texture)
                }
                BuiltinTileItem::PlayerSkull { owner, .. } if texture == owner.cache_key => {
                    // A completed item-profile request is published by the ordinary
                    // AppearanceState cache.  Keep the dropped-item phase on the same normalized
                    // pixels; before completion emit_builtin_item selected the UUID-derived
                    // Steve/Alex resource path and reaches the fixed loader above instead.
                    if let Some(skin) = mc_render::entity::skin::shared_skin(&owner.cache_key) {
                        phase.upload_texture(
                            gpu,
                            texture,
                            &skin.rgba,
                            mc_render::entity::SkinTexture::WIDTH,
                            mc_render::entity::SkinTexture::HEIGHT,
                        );
                        true
                    } else {
                        false
                    }
                }
                BuiltinTileItem::PlayerSkull { .. } => {
                    phase.load_texture(gpu, &assets.resources, texture)
                }
            }
        };
        let key = if resident {
            texture
        } else {
            EntityPhase::FALLBACK_TEXTURE
        };
        phase.push_blended_lit(
            key,
            blend,
            combined_light.unwrap_or(mc_render::entity::render::FULL_BRIGHT_COMBINED_LIGHT),
            verts,
        );
    });
}

#[cfg(test)]
mod entity_list_section_tests {
    use super::*;

    #[test]
    fn an_unreached_player_keeps_only_its_outline_and_only_while_outlines_are_on() {
        use EntityPassGate::{Everything, OutlineOnly, Skip};
        assert_eq!(entity_pass_gate(true, false, false), Everything);
        assert_eq!(entity_pass_gate(true, true, true), Everything);
        assert_eq!(entity_pass_gate(false, true, true), OutlineOnly);
        assert_eq!(
            entity_pass_gate(false, true, false),
            Skip,
            "outlines are players only"
        );
        assert_eq!(entity_pass_gate(false, false, true), Skip);
        assert_eq!(entity_pass_gate(false, false, false), Skip);
    }

    #[test]
    fn the_listing_section_floors_and_clamps_the_way_chunk_add_entity_does() {
        let section = |x, y, z| entity_list_section(DVec3::new(x, y, z));
        assert_eq!(
            section(0.0, 0.0, 0.0),
            mc_math::ChunkSectionPos::new(0, 0, 0)
        );
        assert_eq!(
            section(15.9, 15.9, 15.9),
            mc_math::ChunkSectionPos::new(0, 0, 0)
        );
        assert_eq!(
            section(16.0, 16.0, 16.0),
            mc_math::ChunkSectionPos::new(1, 1, 1)
        );
        assert_eq!(
            section(-0.1, 64.0, -16.0),
            mc_math::ChunkSectionPos::new(-1, 4, -1)
        );
        assert_eq!(
            section(-16.1, 64.0, -32.1),
            mc_math::ChunkSectionPos::new(-2, 4, -3)
        );
        assert_eq!(
            section(0.0, -20.0, 0.0),
            mc_math::ChunkSectionPos::new(0, 0, 0)
        );
        assert_eq!(
            section(0.0, 300.0, 0.0),
            mc_math::ChunkSectionPos::new(0, 15, 0)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_nbt::{Nbt, NbtCompound};
    use mc_world::entity::{Metadata, MetadataValue};

    // These watcher indices belong to the metadata fixtures below. Production reads them through
    // the typed `Watched` accessors, which intentionally own the Java slot mapping.
    const DROPPED_ITEM_SLOT: u8 = 10;
    const ITEM_FRAME_ITEM_SLOT: u8 = 8;

    fn pickup_source(kind: TrackedKind, position: DVec3) -> ItemCollectedSource {
        ItemCollectedSource {
            kind,
            position,
            yaw: 12.0,
            pitch: -8.0,
            age: 0,
            hover_start: 0.0,
            stack: None,
            xp_value: 3,
        }
    }

    #[test]
    fn pickup_fx_uses_three_tick_quadratic_trajectory_and_expires_on_tick_three() {
        let mut effect = PickupEffect::new(
            pickup_source(TrackedKind::Type(EntityType::XPOrb), DVec3::ZERO),
            DVec3::new(3.0, 4.0, 5.0),
            DVec3::new(6.0, 10.0, 14.0),
        );
        assert_eq!(
            pickup_effect_position(&effect, 0.0),
            DVec3::ZERO,
            "EntityPickupFX starts at the source"
        );
        let at_half = pickup_effect_position(&effect, 0.5);
        let target_half = DVec3::new(4.5, 7.5, 9.5);
        let expected_half = target_half / 36.0;
        assert!((at_half - expected_half).length_squared() < 1.0e-12);

        effect.age = 1;
        let at_one = pickup_effect_position(&effect, 0.0);
        let expected_one = DVec3::new(3.0, 4.5, 5.0) / 9.0;
        assert!((at_one - expected_one).length_squared() < 1.0e-12);

        let mut effects = vec![effect];
        tick_pickup_effects(&mut effects);
        assert_eq!(effects.len(), 1, "age 2 remains visible");
        tick_pickup_effects(&mut effects);
        assert!(effects.is_empty(), "age 3 is dead before the next frame");
    }

    #[test]
    fn pickup_fx_has_an_end_to_end_producer_and_omitted_unknown_path() {
        let driver = include_str!("driver.rs");
        let dispatch = include_str!("../sync/dispatch.rs");
        assert!(dispatch.contains("capture_item_collected_source"));
        assert!(driver.contains("ClientEvent::ItemCollected {"));
        assert!(driver.contains("self.play_item_collected_sound"));
        assert!(driver.contains("PickupEffect::new"));
        assert!(driver.contains("&self.pickup_effects"));
        assert!(driver.contains("let local_fallback = ||"));
        assert!(driver.contains("target.kind.is_living()"));
    }

    #[test]
    fn tnt_flash_clones_the_live_body_only_during_the_white_fuse_phase() {
        let body = [
            EntityVertex {
                pos: [1.0, 2.0, 3.0],
                uv: [0.25, 0.5],
                normal: [0.0, 1.0, 0.0],
                color: [0.2, 0.3, 0.4, 1.0],
            },
            EntityVertex {
                pos: [-1.0, 0.0, 4.0],
                uv: [0.75, 0.125],
                normal: [1.0, 0.0, 0.0],
                color: [0.8, 0.7, 0.6, 1.0],
            },
        ];

        let mut flash = Vec::new();
        prepare_tnt_flash_vertices(Some(EntityType::PrimedTnt), 6, 0.5, &body, &mut flash); // fuse 74: visible
        assert_eq!(flash.len(), body.len());
        // X6: the ITEM format's per-vertex white overrides java's `color(1, 1, 1, f2)`.
        for (body_vertex, flash_vertex) in body.iter().zip(&flash) {
            assert_eq!(flash_vertex.pos, body_vertex.pos);
            assert_eq!(flash_vertex.uv, body_vertex.uv);
            assert_eq!(flash_vertex.normal, body_vertex.normal);
            assert_eq!(flash_vertex.color, objects::TNT_FLASH_COLOR);
        }

        flash.clear();
        prepare_tnt_flash_vertices(Some(EntityType::PrimedTnt), 5, 0.5, &body, &mut flash); // fuse 75: hidden
        assert!(flash.is_empty());

        prepare_tnt_flash_vertices(Some(EntityType::Zombie), 6, 0.5, &body, &mut flash);
        assert!(
            flash.is_empty(),
            "non-TNT geometry never receives the flash pass"
        );

        for fuse in (0..=9).rev() {
            flash.clear();
            let age = u32::try_from(objects::TNT_FUSE_TICKS - fuse).unwrap();
            prepare_tnt_flash_vertices(Some(EntityType::PrimedTnt), age, 0.5, &body, &mut flash);
            assert_eq!(!flash.is_empty(), objects::tnt_flashing(fuse));
            // Full strength for the whole fuse: java's rising `f2` never reaches the fragment.
            for vertex in &flash {
                assert_eq!(vertex.color, objects::TNT_FLASH_COLOR);
            }
        }

        let source = include_str!("entrender.rs");
        let (_, upload) = source
            .split_once("pub fn upload_objects(")
            .expect("live object uploader remains present");
        let (upload, _) = upload
            .split_once("fn fireball_overlay_size(")
            .expect("object uploader ends before its helper");
        let body_push = upload
            .find("match &texture")
            .expect("body push remains live");
        let flash_push = upload
            .find("LayerBlend::TntFlash")
            .expect("TNT flash push remains live");
        assert!(
            body_push < flash_push,
            "body must be submitted before flash"
        );
    }

    /// java: `RenderManager.cacheActiveRenderInfo` (`:258-267`) turns only `playerViewY` for the
    /// F5 front view, and RenderSnowball billboards with `rotate(-playerViewY, Y)` then
    /// `rotate(playerViewX, X)`. The front-view camera (yaw + 180, pitch negated) must therefore
    /// hand the object billboards the *un-negated* view pitch; the other two views pass the
    /// camera's own angles through.
    #[test]
    fn object_billboards_use_render_manager_view_in_every_perspective() {
        let view_yaw = 30.0_f32;
        let view_pitch = 25.0_f32;
        let front = Camera {
            yaw: view_yaw + 180.0,
            pitch: -view_pitch,
            ..Camera::default()
        };
        assert_eq!(
            render_manager_view(&front, 2),
            (view_yaw + 180.0, view_pitch)
        );
        let behind = Camera {
            yaw: view_yaw,
            pitch: view_pitch,
            ..Camera::default()
        };
        assert_eq!(render_manager_view(&behind, 1), (view_yaw, view_pitch));
        assert_eq!(render_manager_view(&behind, 0), (view_yaw, view_pitch));

        // The thrown-item transform then tilts by the view pitch in front view: for a sprite
        // above the item origin, the front-view result differs from billboarding with the
        // camera's own (negated) pitch.
        let model = mc_assets::BakedModel::default();
        let (yaw, pitch) = render_manager_view(&front, 2);
        let vanilla =
            mc_render::entity::objects::thrown_item_matrix(mc_math::Vec3::ZERO, yaw, pitch, &model);
        let camera_facing = mc_render::entity::objects::thrown_item_matrix(
            mc_math::Vec3::ZERO,
            front.yaw,
            front.pitch,
            &model,
        );
        let top = mc_math::Vec3::new(0.0, 0.5, 0.0);
        assert!(
            (vanilla.transform_point3(top) - camera_facing.transform_point3(top)).length() > 0.01
        );
    }

    #[test]
    fn shared_entity_gate_combines_range_frustum_fallback_and_exemptions() {
        let camera = Camera::default();
        let frustum = Frustum::from_camera(&camera);
        let front_position = DVec3::new(0.0, 0.0, 10.0);
        let front = Aabb::from_bounds(-1.0, -1.0, 9.0, 1.0, 1.0, 11.0);
        assert!(should_render_bounds(
            &frustum,
            camera.pos,
            front_position,
            front,
            1.0,
            false,
            false,
        ));

        let behind_position = DVec3::new(0.0, 0.0, -10.0);
        let behind = Aabb::from_bounds(-1.0, -1.0, -11.0, 1.0, 1.0, -9.0);
        assert!(!should_render_bounds(
            &frustum,
            camera.pos,
            behind_position,
            behind,
            1.0,
            false,
            false,
        ));
        assert!(should_render_bounds(
            &frustum,
            camera.pos,
            behind_position,
            behind,
            1.0,
            true,
            false,
        ));

        let far_position = DVec3::new(0.0, 0.0, 1_000.0);
        let far = Aabb::from_bounds(-1.0, -1.0, 999.0, 1.0, 1.0, 1_001.0);
        assert!(
            !should_render_bounds(&frustum, camera.pos, far_position, far, 1.0, true, false,),
            "ignoreFrustumCheck must not bypass Entity.isInRangeToRenderDist",
        );
        assert!(should_render_bounds(
            &frustum,
            camera.pos,
            far_position,
            far,
            1.0,
            false,
            true,
        ));

        let nan = Aabb {
            min: DVec3::new(f64::NAN, 0.0, 0.0),
            max: DVec3::new(1.0, 1.0, 1.0),
        };
        assert!(should_render_bounds(
            &frustum,
            camera.pos,
            front_position,
            nan,
            1.0,
            false,
            false,
        ));

        let production = include_str!("entrender.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("production entity renderer");
        assert!(
            production.contains("let render_box = bounds.render_box_or_fallback(position);"),
            "the live shared gate must call the tested NaN/zero-size fallback helper",
        );
    }

    #[test]
    fn extracted_entity_weight_changes_the_shared_range_boundary() {
        let camera = Camera::default();
        let frustum = Frustum::from_camera(&camera);
        let unit = Aabb::from_bounds(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
        let arrow_weight = EntityType::Arrow.info().render_distance_weight;
        let ordinary_weight = EntityType::Zombie.info().render_distance_weight;
        assert_eq!(arrow_weight, 10.0);
        assert_eq!(ordinary_weight, 1.0);

        let beyond_ordinary = DVec3::new(100.0, 0.0, 0.0);
        assert!(!should_render_bounds(
            &frustum,
            camera.pos,
            beyond_ordinary,
            unit,
            ordinary_weight,
            true,
            false,
        ));
        assert!(should_render_bounds(
            &frustum,
            camera.pos,
            beyond_ordinary,
            unit,
            arrow_weight,
            true,
            false,
        ));
        assert!(!should_render_bounds(
            &frustum,
            camera.pos,
            DVec3::new(640.0, 0.0, 0.0),
            unit,
            arrow_weight,
            true,
            false,
        ));

        let production = include_str!("entrender.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("production entity renderer");
        assert!(production.contains("kind.info().render_distance_weight"));
        assert!(!production.contains("bounds.is_in_render_range(distance_squared, 1.0)"));
    }

    #[test]
    fn item_frame_range_uses_fixed_entity_size_and_keeps_thin_plate_geometry() {
        let camera = Camera::default();
        let frustum = Frustum::from_camera(&camera);
        let frame_weight = EntityType::ItemFrame.info().render_distance_weight;
        let plate = crate::app::interact::hanging_bounding_box(DVec3::ZERO, Facing::South, 12, 12);

        // EntityHanging's 12x12 plate remains the frustum/render box: 0.75 x 0.75 x 0.0625.
        assert_eq!(plate.max - plate.min, DVec3::new(0.75, 0.75, 0.0625));
        assert_eq!(
            render_distance_base(Some(EntityType::ItemFrame)),
            Some(16.0)
        );
        assert_eq!(16.0 * 64.0 * frame_weight, 1024.0);

        // The ordinary AABB rule rejects this thin plate already at ~33 blocks. The Java
        // EntityItemFrame override keeps it visible through 1024 blocks, with a strict edge.
        assert!(!should_render_bounds(
            &frustum,
            camera.pos,
            DVec3::new(40.0, 0.0, 0.0),
            plate,
            frame_weight,
            true,
            false,
        ));
        assert!(should_render_bounds_with_distance_base(
            &frustum,
            camera.pos,
            DVec3::new(40.0, 0.0, 0.0),
            plate,
            frame_weight,
            render_distance_base(Some(EntityType::ItemFrame)),
            true,
            false,
        ));
        assert!(should_render_bounds_with_distance_base(
            &frustum,
            camera.pos,
            DVec3::new(60.0, 0.0, 0.0),
            plate,
            frame_weight,
            render_distance_base(Some(EntityType::ItemFrame)),
            true,
            false,
        ));
        assert!(should_render_bounds_with_distance_base(
            &frustum,
            camera.pos,
            DVec3::new(1023.999, 0.0, 0.0),
            plate,
            frame_weight,
            render_distance_base(Some(EntityType::ItemFrame)),
            true,
            false,
        ));
        assert!(!should_render_bounds_with_distance_base(
            &frustum,
            camera.pos,
            DVec3::new(1024.0, 0.0, 0.0),
            plate,
            frame_weight,
            render_distance_base(Some(EntityType::ItemFrame)),
            true,
            false,
        ));
    }

    #[test]
    fn item_frame_range_override_does_not_change_item_or_painting_ranges() {
        let camera = Camera::default();
        let frustum = Frustum::from_camera(&camera);
        let item_bounds = Aabb::from_bounds(-0.125, -0.125, -0.125, 0.125, 0.125, 0.125);
        let painting_bounds =
            crate::app::interact::hanging_bounding_box(DVec3::ZERO, Facing::South, 16, 16);

        // Ordinary dropped items retain their 0.25 cube's 16-block range.
        assert_eq!(render_distance_base(Some(EntityType::Item)), None);
        assert!(!should_render_bounds_with_distance_base(
            &frustum,
            camera.pos,
            DVec3::new(40.0, 0.0, 0.0),
            item_bounds,
            EntityType::Item.info().render_distance_weight,
            render_distance_base(Some(EntityType::Item)),
            true,
            false,
        ));

        // Paintings continue to use their art-sized hanging box (~44 blocks for a 16x16 art),
        // rather than inheriting the item frame's fixed 1024-block range.
        assert_eq!(render_distance_base(Some(EntityType::Painting)), None);
        assert!(should_render_bounds_with_distance_base(
            &frustum,
            camera.pos,
            DVec3::new(40.0, 0.0, 0.0),
            painting_bounds,
            EntityType::Painting.info().render_distance_weight,
            render_distance_base(Some(EntityType::Painting)),
            true,
            false,
        ));
        assert!(!should_render_bounds_with_distance_base(
            &frustum,
            camera.pos,
            DVec3::new(60.0, 0.0, 0.0),
            painting_bounds,
            EntityType::Painting.info().render_distance_weight,
            render_distance_base(Some(EntityType::Painting)),
            true,
            false,
        ));
    }

    #[test]
    fn armor_stand_range_uses_fourfold_normal_box_at_a_strict_boundary() {
        let camera = Camera::default();
        let frustum = Frustum::from_camera(&camera);
        let normal = Aabb::from_bounds(-0.25, 0.0, -0.25, 0.25, 1.975, 0.25);
        let range = armor_stand_render_distance_base(normal) * 64.0;

        assert_eq!(
            armor_stand_render_distance_base(normal),
            normal.average_edge_length() * 4.0
        );
        assert!(should_render_bounds_with_distance_base(
            &frustum,
            camera.pos,
            DVec3::new(range - 0.001, 0.0, 0.0),
            normal,
            1.0,
            Some(armor_stand_render_distance_base(normal)),
            true,
            false,
        ));
        assert!(!should_render_bounds_with_distance_base(
            &frustum,
            camera.pos,
            DVec3::new(range, 0.0, 0.0),
            normal,
            1.0,
            Some(armor_stand_render_distance_base(normal)),
            true,
            false,
        ));
    }

    #[test]
    fn armor_stand_marker_range_uses_four_block_fallback_at_a_strict_boundary() {
        let camera = Camera::default();
        let frustum = Frustum::from_camera(&camera);
        let marker = Aabb::from_bounds(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        assert_eq!(armor_stand_render_distance_base(marker), 4.0);
        assert!(should_render_bounds_with_distance_base(
            &frustum,
            camera.pos,
            DVec3::new(4.0 * 64.0 - 0.001, 0.0, 0.0),
            marker,
            1.0,
            Some(armor_stand_render_distance_base(marker)),
            true,
            false,
        ));
        assert!(!should_render_bounds_with_distance_base(
            &frustum,
            camera.pos,
            DVec3::new(4.0 * 64.0, 0.0, 0.0),
            marker,
            1.0,
            Some(armor_stand_render_distance_base(marker)),
            true,
            false,
        ));
    }

    /// §7 correctness #2: a remote player used `Entity`'s 1.0 weight (no `EntityList` row), so
    /// every player beyond ~64 blocks vanished — the opposite island in SkyWars. java:
    /// `EntityOtherPlayerMP`'s constructor sets `renderDistanceWeight = 10.0D`.
    #[test]
    fn remote_players_use_the_entity_other_player_mp_weight_at_a_strict_boundary() {
        let camera = Camera::default();
        let frustum = Frustum::from_camera(&camera);
        assert_eq!(tracked_render_distance_weight(TrackedKind::Player), 10.0);
        assert_eq!(
            tracked_render_distance_weight(TrackedKind::Type(EntityType::Zombie)),
            1.0
        );
        assert_eq!(
            tracked_render_distance_weight(TrackedKind::Type(EntityType::Arrow)),
            10.0,
            "extracted weights still reach the shared gate"
        );
        assert_eq!(
            tracked_render_distance_weight(TrackedKind::FishingBobber),
            1.0
        );

        // The box sits away from the origin so its edges carry Java's float-half-width rounding;
        // the range half reads only its average edge, so distances are measured from zero to
        // keep the strict boundary exact.
        let standing = crate::app::interact::size_bounding_box(
            DVec3::new(1_000.25, 70.0, -2_000.75),
            0.6,
            1.8,
        );
        let range = standing.average_edge_length() * 64.0 * REMOTE_PLAYER_RENDER_DISTANCE_WEIGHT;
        assert!((range - 640.0).abs() < 1e-6, "{range}");
        let feet = DVec3::ZERO;
        let at = |distance: f64| DVec3::new(distance, 0.0, 0.0);
        for (distance, expected) in [
            (63.0, true),
            (100.0, true),
            (range - 0.001, true),
            (range, false),
            (range + 0.001, false),
        ] {
            assert_eq!(
                should_render_bounds(&frustum, at(distance), feet, standing, 10.0, true, false),
                expected,
                "{distance}"
            );
            // The 1.9+ spelling (override `averageEdgeLength * 10`, static weight 1.0) agrees
            // everywhere off the last-ulp boundary.
            if distance != range {
                let override_form = standing.average_edge_length() * 10.0 * 64.0;
                assert_eq!(
                    (feet - at(distance)).length_squared() < override_form * override_form,
                    expected,
                    "{distance}"
                );
            }
        }
        // The old fallback weight culled the same player at 100 blocks.
        assert!(!should_render_bounds(
            &frustum,
            at(100.0),
            feet,
            standing,
            1.0,
            true,
            false
        ));

        // A sleeping player's 0.2 x 0.2 box keeps the same weight: ~128 blocks.
        let sleeping = crate::app::interact::size_bounding_box(
            DVec3::new(1_000.25, 70.0, -2_000.75),
            0.2,
            0.2,
        );
        assert!(should_render_bounds(
            &frustum,
            at(127.9),
            feet,
            sleeping,
            10.0,
            true,
            false
        ));
        assert!(!should_render_bounds(
            &frustum,
            at(128.1),
            feet,
            sleeping,
            10.0,
            true,
            false
        ));

        let production = include_str!("entrender.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("production entity renderer");
        assert!(production.contains(
            "let render_distance_weight = tracked_render_distance_weight(tracked.kind);"
        ));
        assert!(production.contains("TrackedKind::Player => REMOTE_PLAYER_RENDER_DISTANCE_WEIGHT,"));
    }

    /// java: `EntityThrowable`, `EntityFireball`, `EntityEnderEye` and `EntityFishHook` override
    /// `isInRangeToRenderDist` with `averageEdgeLength * 4` (NaN -> 4) times 64 and no weight.
    /// The generic rule culled pearls, snowballs, eggs, potions and the hook at 16 blocks, small
    /// fireballs at 20 and BedWars' large fireballs at 64.
    #[test]
    fn projectiles_use_the_fourfold_override_at_a_strict_boundary() {
        let camera = Camera::default();
        let frustum = Frustum::from_camera(&camera);
        let nan = Aabb {
            min: DVec3::new(f64::NAN, 0.0, 0.0),
            max: DVec3::new(1.0, 1.0, 1.0),
        };
        assert_eq!(projectile_render_distance_base(nan), 4.0);
        let degenerate = Aabb::from_bounds(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(
            projectile_render_distance_base(degenerate),
            0.0,
            "no zero-size clause, unlike EntityArmorStand"
        );

        let cases = [
            (TrackedKind::FishingBobber, 0.25, 64.0),
            (TrackedKind::Type(EntityType::Snowball), 0.25, 64.0),
            (TrackedKind::Type(EntityType::ThrownEgg), 0.25, 64.0),
            (TrackedKind::Type(EntityType::ThrownEnderpearl), 0.25, 64.0),
            (TrackedKind::Type(EntityType::ThrownPotion), 0.25, 64.0),
            (TrackedKind::Type(EntityType::ThrownExpBottle), 0.25, 64.0),
            (TrackedKind::Type(EntityType::EyeOfEnderSignal), 0.25, 64.0),
            (TrackedKind::Type(EntityType::SmallFireball), 0.3125, 80.0),
            (TrackedKind::Type(EntityType::WitherSkull), 0.3125, 80.0),
            (TrackedKind::Type(EntityType::Fireball), 1.0, 256.0),
        ];
        for (kind, size, range) in cases {
            assert!(uses_projectile_render_distance(kind), "{kind:?}");
            if let Some(entity_type) = kind.entity_type() {
                assert_eq!(f64::from(entity_type.info().width), size, "{kind:?}");
                assert_eq!(f64::from(entity_type.info().height), size, "{kind:?}");
            }
            let half = size / 2.0;
            let bounds = Aabb::from_bounds(-half, 0.0, -half, half, size, half);
            let gate = |distance: f64| {
                should_render_bounds_with_distance_base(
                    &frustum,
                    camera.pos,
                    camera.pos + DVec3::new(distance, 0.0, 0.0),
                    bounds,
                    tracked_render_distance_weight(kind),
                    Some(projectile_render_distance_base(bounds)),
                    true,
                    false,
                )
            };
            assert!(gate(range - 0.001), "{kind:?}");
            assert!(!gate(range), "{kind:?}");
            // The generic average-edge rule stopped at a quarter of that.
            assert!(
                !should_render_bounds(
                    &frustum,
                    camera.pos,
                    camera.pos + DVec3::new(range / 4.0, 0.0, 0.0),
                    bounds,
                    1.0,
                    true,
                    false,
                ),
                "{kind:?}"
            );
        }
        for kind in [
            TrackedKind::Player,
            TrackedKind::Type(EntityType::Arrow),
            TrackedKind::Type(EntityType::Item),
            TrackedKind::Type(EntityType::PrimedTnt),
            TrackedKind::Type(EntityType::ArmorStand),
            TrackedKind::Type(EntityType::Zombie),
        ] {
            assert!(!uses_projectile_render_distance(kind), "{kind:?}");
        }

        let production = include_str!("entrender.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("production entity renderer");
        assert!(production.contains("Some(projectile_render_distance_base(bounds))"));
    }

    #[test]
    fn firework_visibility_uses_the_strict_sixty_four_block_boundary() {
        let view = DVec3::new(12.0, -4.0, 30.0);
        assert!(should_render_firework(
            view,
            view + DVec3::new(63.999, 0.0, 0.0),
        ));
        assert!(!should_render_firework(
            view,
            view + DVec3::new(64.0, 0.0, 0.0),
        ));
        assert!(!should_render_firework(
            view,
            view + DVec3::new(65.0, 0.0, 0.0),
        ));

        let production = include_str!("entrender.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("production object pass");
        assert_eq!(production.matches("&& !should_render_firework(").count(), 1);
    }

    /// The opt-in caps are off by default (vanilla), strict at their boundary, keep players and
    /// everything else apart, and never cap something that can hit the player.
    #[test]
    fn entity_render_caps_are_off_by_default_and_exempt_projectiles() {
        let off = EntityRenderCaps::from_settings(&mc_render::settings::GameSettings::default());
        assert_eq!(off, EntityRenderCaps::default());
        for kind in [
            TrackedKind::Player,
            TrackedKind::FishingBobber,
            TrackedKind::Type(EntityType::ArmorStand),
            TrackedKind::Type(EntityType::Item),
        ] {
            assert_eq!(off.cap_squared(kind), None, "{kind:?}");
            assert!(off.allows_distance_squared(kind, f64::MAX), "{kind:?}");
        }

        let caps = EntityRenderCaps::from_settings(&mc_render::settings::GameSettings {
            player_render_distance: 96,
            entity_render_distance: 32,
            ..mc_render::settings::GameSettings::default()
        });
        assert!(caps.allows_distance_squared(TrackedKind::Player, 95.999 * 95.999));
        assert!(!caps.allows_distance_squared(TrackedKind::Player, 96.0 * 96.0));
        let stand = TrackedKind::Type(EntityType::ArmorStand);
        assert!(caps.allows_distance_squared(stand, 31.999 * 31.999));
        assert!(!caps.allows_distance_squared(stand, 32.0 * 32.0));
        assert!(!caps.allows_distance_squared(TrackedKind::Type(EntityType::Item), 40.0 * 40.0));
        for kind in [
            TrackedKind::FishingBobber,
            TrackedKind::Type(EntityType::Arrow),
            TrackedKind::Type(EntityType::Fireball),
            TrackedKind::Type(EntityType::SmallFireball),
            TrackedKind::Type(EntityType::Snowball),
            TrackedKind::Type(EntityType::ThrownEgg),
            TrackedKind::Type(EntityType::ThrownEnderpearl),
            TrackedKind::Type(EntityType::ThrownPotion),
            TrackedKind::Type(EntityType::WitherSkull),
            TrackedKind::Type(EntityType::PrimedTnt),
            TrackedKind::Type(EntityType::MinecartTNT),
        ] {
            assert_eq!(caps.cap_squared(kind), None, "{kind:?} keeps vanilla range");
        }

        // Only a player cap: everything else stays vanilla.
        let players_only = EntityRenderCaps::from_settings(&mc_render::settings::GameSettings {
            player_render_distance: 64,
            entity_render_distance: -5,
            ..mc_render::settings::GameSettings::default()
        });
        assert_eq!(players_only.cap_squared(stand), None);
        assert_eq!(players_only.cap_squared(TrackedKind::Player), Some(4096.0));

        // Applied after the vanilla gate at all three call sites, never to the local vehicle.
        let objects = include_str!("entrender.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("production object pass");
        assert_eq!(objects.matches("&& !entity_render_caps.allows(").count(), 1);
        let driver = include_str!("driver.rs");
        assert_eq!(driver.matches("&& !entity_render_caps.allows(").count(), 2);
        assert_eq!(
            driver
                .matches("if local_vehicle != Some(network_id)\n                    && !entity_render_caps.allows(")
                .count(),
            2
        );
    }

    /// java: `RenderGuardian.shouldRender` keeps a guardian whose body is out of view while its
    /// beam's box is in view, and builds that box with `func_177110_a(entity, offset, 1.0F)`:
    /// the current tick's pose of both ends, never the frame's partial tick.
    #[test]
    fn guardian_beam_frustum_branch_samples_the_current_tick() {
        let camera = Camera::default();
        let frustum = Frustum::from_camera(&camera);
        let ahead = DVec3::new(0.0, 0.5, 5.0);
        let behind = DVec3::new(0.0, 0.5, -5.0);
        assert!(frustum.contains_box(ahead, ahead + DVec3::ONE));
        assert!(!frustum.contains_box(behind, behind + DVec3::ONE));

        let mut state = ClientState::default();
        let mut metadata = Metadata::new();
        metadata.set(17, MetadataValue::Int(7));
        // The guardian floats behind the camera, its target too, last tick.
        state.entities.spawn(
            &mut state.world,
            6,
            TrackedKind::Type(EntityType::Guardian),
            None,
            [0, 0, -5 * 32],
            0.0,
            0.0,
            DVec3::ZERO,
            metadata,
        );
        let target = state.entities.spawn(
            &mut state.world,
            7,
            TrackedKind::Type(EntityType::Zombie),
            None,
            [0, 0, -3 * 32],
            0.0,
            0.0,
            DVec3::ZERO,
            Metadata::new(),
        );
        let guardian = state.entities.get(6).expect("guardian");
        assert!(!should_render_tracked(
            &frustum,
            camera.pos,
            &state.world,
            guardian,
            0.0,
            false
        ));
        assert!(!guardian_beam_in_frustum(&frustum, &state, guardian, None));

        // This tick the target moved in front of the camera: at the start of the frame's
        // interpolation (partial 0) the beam is still behind, at the tick's pose it crosses
        // the view.
        state
            .world
            .ecs
            .get::<&mut Position>(target)
            .expect("target position")
            .0 = DVec3::new(0.0, 0.0, 10.0);
        let guardian = state.entities.get(6).expect("guardian");
        let tracked_target = state.entities.get(7).expect("target");
        assert_eq!(
            tracked_target.render_position(&state.world, 0.0),
            DVec3::new(0.0, 0.0, -3.0)
        );
        assert!(guardian_beam_in_frustum(&frustum, &state, guardian, None));

        // The local player as the target: its tick pose decides, not the frame pose.
        let local = |position: DVec3, tick_position: DVec3| LocalLeashHolder {
            network_id: 7,
            position,
            tick_position,
            width: 0.6,
            height: 1.8,
            prev_yaw: 0.0,
            yaw: 0.0,
            prev_pitch: 0.0,
            pitch: 0.0,
            eye_height: 1.62,
        };
        let back = DVec3::new(0.0, 0.0, -3.0);
        let front = DVec3::new(0.0, 0.0, 10.0);
        assert!(guardian_beam_in_frustum(
            &frustum,
            &state,
            guardian,
            Some(local(back, front))
        ));
        assert!(!guardian_beam_in_frustum(
            &frustum,
            &state,
            guardian,
            Some(local(front, back))
        ));
    }

    #[test]
    fn frame_routes_body_nameplates_and_objects_through_the_shared_gate() {
        let driver = include_str!("driver.rs");
        assert_eq!(
            driver
                .matches("crate::app::entrender::should_render_tracked_with_leash(")
                .count(),
            2,
            "body/layers and nameplates must both use the shared leash-aware gate",
        );
        assert!(driver.contains("session.riding(),"));

        let (_, frame) = driver
            .split_once(
                "pub fn frame(&mut self, gpu: &Gpu, camera: &mut Camera, partial_ticks: f32) {",
            )
            .expect("live frame producer");
        let (frame, _) = frame
            .split_once("fn place_camera(")
            .expect("frame ends before camera helper");
        assert!(frame.contains("self.render_view_position(camera, partial_ticks)"));
        assert!(frame.contains("render_view_position,"));

        let (_, entity_pass) = driver
            .split_once(
                "fn upload_entities(&mut self, gpu: &Gpu, camera: &Camera, partial_ticks: f32) {",
            )
            .expect("live entity pass");
        let (entity_pass, _) = entity_pass
            .split_once("fn drive_atmosphere(")
            .expect("entity pass ends before atmosphere driver");
        assert!(
            entity_pass.contains("self.render_view_position(camera, partial_ticks)"),
            "the entity pass must use the shared render-view origin producer",
        );
        assert!(
            entity_pass.matches("render_view_position,").count() >= 2,
            "both living bodies and nameplates must receive render-view feet",
        );

        let (_, view_position) = driver
            .split_once("fn render_view_position(")
            .expect("render-view feet producer");
        let (view_position, _) = view_position
            .split_once("fn corrections(")
            .expect("render-view producer ends before correction accessor");
        assert!(
            view_position.contains("interpolated_feet(prev, &predictor.state, partial_ticks)"),
            "the shared origin must be interpolated feet, never eye height or F5 camera position",
        );

        let object_pass = include_str!("entrender.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("production object pass");
        assert_eq!(object_pass.matches("if !should_render_tracked(").count(), 1);
        assert!(object_pass.contains("kind == Some(EntityType::WitherSkull)"));
    }

    #[test]
    fn primed_tnt_brightness_samples_its_loaded_feet_position() {
        let position = mc_math::DVec3::new(0.75, 64.9, 0.75);
        let lighting = crate::app::appearance::ShadowLighting {
            skylight_subtracted: 11,
            ambient_light: 0.0,
        };
        let mut world = World::new();
        assert_eq!(entity_brightness(&world, position, lighting), 0.0);

        let pos = mc_math::BlockPos::new(0, 64, 0);
        world.load_empty_chunk(pos.chunk_pos());
        world.set_light_for(mc_world::SkyBlock::Sky, pos, 15);
        world.set_light_for(mc_world::SkyBlock::Block, pos, 2);
        let table = mc_render::sky::lightmap::light_brightness_table(0.0);
        assert_eq!(entity_brightness(&world, position, lighting), table[4]);

        world.set_light_for(mc_world::SkyBlock::Block, pos, 12);
        assert_eq!(entity_brightness(&world, position, lighting), table[12]);
    }

    #[test]
    fn falling_block_hides_only_over_the_identical_source_state() {
        let position = mc_math::DVec3::new(0.75, 64.9, 0.75);
        let pos = BlockPos::from_vec3(position);
        let stone = mc_data::block_by_name("minecraft:stone")
            .expect("stone")
            .default_state();
        let dirt = mc_data::block_by_name("minecraft:dirt")
            .expect("dirt")
            .default_state();
        let mut world = World::new();
        world.load_empty_chunk(pos.chunk_pos());

        assert!(falling_block_should_render(&world, position, stone));
        world.set_block(pos, stone);
        assert!(!falling_block_should_render(&world, position, stone));
        world.set_block(pos, dirt);
        assert!(falling_block_should_render(&world, position, stone));
        assert!(falling_block_should_render(
            &world,
            position + mc_math::DVec3::X,
            stone
        ));

        let production = include_str!("entrender.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("production entity renderer");
        let (_, falling_arm) = production
            .split_once("E::FallingSand => {")
            .expect("live falling-block renderer arm");
        let (falling_arm, _) = falling_arm
            .split_once("// -- RenderPainting")
            .expect("falling-block arm ends before painting rendering");
        let gate = falling_arm
            .find("if !falling_block_should_render(world, current_position, state)")
            .expect("the live arm gates identical source state");
        let emit = falling_arm
            .find("emit_falling_block_model_with_settings(")
            .expect("the live arm still emits visible falling blocks");
        assert!(
            gate < emit,
            "source-state suppression must precede geometry"
        );
        // java: `renderModel` reads `Minecraft.isAmbientOcclusionEnabled()` live, so the live arm
        // and its packed-light twin take the frame's settings, never the vanilla defaults.
        assert!(
            falling_arm.contains("block_mesh,") && !falling_arm.contains("MeshSettings::default()"),
            "the live falling-block arm must light with the live graphics settings"
        );
        let (_, lights_call) = production
            .split_once("falling_block_vertex_lights_for_entity(")
            .expect("the live packed-light call precedes its definition");
        let (lights_call, _) = lights_call.split_once(')').expect("the call ends");
        assert!(
            lights_call.contains("block_mesh"),
            "the packed per-vertex light twin must use the same live settings: {lights_call}"
        );
        // java: `RenderFallingBlock` draws with `disableLighting()` and the world pass's culling,
        // so its submission takes the unlit, back-face-culled pipeline, never the lit `Opaque`
        // one (`tests/w18_pa_pictures.rs` measures the difference on a real frame).
        let (_, falling_submit) = production
            .split_once("if let Some(lights) = falling_vertex_lights {")
            .expect("the live falling-block submission");
        let (falling_submit, _) = falling_submit
            .split_once("} else {")
            .expect("the submission ends");
        assert!(
            falling_submit.contains("LayerBlend::OpaqueUnlitCulled")
                && !falling_submit.contains("LayerBlend::Opaque,"),
            "the falling block must be drawn unlit: {falling_submit}"
        );
        assert!(
            production.contains("falling_block_vertex_lights_for_entity")
                && production.contains("push_geometry_lit_with_vertex_lights")
                && !production.contains("apply_falling_block_vertex_lightmap"),
            "the live falling-block arm must retain its packed per-vertex light handoff"
        );
    }

    #[test]
    fn object_light_samples_dropped_items_without_leaking_into_other_entities() {
        let position = DVec3::new(0.25, 64.25, 0.25);
        let sample = BlockPos::from_f64(position.x, position.y, position.z);
        let mut world = World::new();
        world.set_has_sky_light(true);
        world.load_empty_chunk(sample.chunk_pos());
        world.set_light_for(mc_world::SkyBlock::Block, sample, 3);
        world.set_light_for(mc_world::SkyBlock::Sky, sample, 9);

        assert_eq!(
            inherited_object_light(Some(EntityType::FallingSand), &world, position),
            Some((3_u32 << 4) | (9_u32 << 20)),
            "falling blocks must carry their loaded entity lightmap into the object batch"
        );
        assert_eq!(
            inherited_object_light(Some(EntityType::Item), &world, position),
            Some((3_u32 << 4) | (9_u32 << 20)),
            "ordinary dropped items must carry Entity.getBrightnessForRender's packed sample"
        );
        assert_eq!(
            inherited_object_light(Some(EntityType::Arrow), &world, position),
            Some((3_u32 << 4) | (9_u32 << 20)),
            "arrows must carry the base Entity.getBrightnessForRender packed sample"
        );
        assert_eq!(
            inherited_object_light(Some(EntityType::ItemFrame), &world, position),
            Some((3_u32 << 4) | (9_u32 << 20)),
            "item frames keep their independent world-light contract"
        );
        assert_eq!(
            inherited_object_light(Some(EntityType::FallingSand), &World::new(), position),
            Some(0),
            "an unloaded falling block must not read the blank-column skylight default"
        );
        assert_eq!(
            inherited_object_light(Some(EntityType::XPOrb), &World::new(), position),
            Some(0),
            "an unloaded XP orb must retain the zero world sample rather than inventing light"
        );
        assert_eq!(
            inherited_object_light(Some(EntityType::ItemFrame), &World::new(), position),
            Some(0),
            "an unloaded item frame must retain the zero world sample"
        );
        assert_eq!(
            inherited_object_light(Some(EntityType::Cow), &world, position),
            None,
            "ordinary living entities are lit by the living upload path, not this object seam"
        );

        // Dark and torch-lit dropped items must select different packed light values. The
        // ordinary object fallback remains full bright only when this helper returns `None`.
        world.set_light_for(mc_world::SkyBlock::Block, sample, 0);
        world.set_light_for(mc_world::SkyBlock::Sky, sample, 0);
        assert_eq!(
            inherited_object_light(Some(EntityType::Item), &world, position),
            Some(0),
            "a dropped item in an unloaded-light cell must render dark"
        );
        world.set_light_for(mc_world::SkyBlock::Block, sample, 12);
        assert_eq!(
            inherited_object_light(Some(EntityType::Item), &world, position),
            Some(12_u32 << 4),
            "a dropped item over torch light must carry the block-light nibble"
        );
        assert_eq!(
            inherited_object_light(Some(EntityType::Arrow), &world, position),
            Some(12_u32 << 4),
            "a lit arrow must carry the live torch-light nibble"
        );
        assert_eq!(
            inherited_object_light(Some(EntityType::Arrow), &World::new(), position),
            Some(0),
            "an arrow in an unloaded chunk must render dark, not full-bright"
        );
    }

    #[test]
    fn primed_tnt_body_and_flash_inherit_world_light_and_shadow_is_unchanged() {
        let position = DVec3::new(0.25, 64.25, 0.25);
        let sample = BlockPos::from_f64(position.x, position.y, position.z);
        let mut world = World::new();
        world.set_has_sky_light(true);
        world.load_empty_chunk(sample.chunk_pos());
        world.set_light_for(mc_world::SkyBlock::Block, sample, 0);
        world.set_light_for(mc_world::SkyBlock::Sky, sample, 0);
        let dark = inherited_object_light(Some(EntityType::PrimedTnt), &world, position);
        assert_eq!(dark, Some(0), "a loaded dark TNT body must stay dark");

        world.set_light_for(mc_world::SkyBlock::Block, sample, 5);
        world.set_light_for(mc_world::SkyBlock::Sky, sample, 12);
        let lit = inherited_object_light(Some(EntityType::PrimedTnt), &world, position);
        assert_eq!(lit, Some((5_u32 << 4) | (12_u32 << 20)));
        assert_ne!(
            dark, lit,
            "the TNT body must vary with the sampled world light"
        );

        // RenderTNTPrimed's flashing replay disables lighting and texture unit 0 and supplies
        // white geometry, but the lightmap stage stays on (X6, `javaref/TntFlashRef.java`): the
        // upload seam submits the overlay with the entity's own packed light, like the body.
        let body = [EntityVertex {
            pos: [0.0, 0.0, 0.0],
            uv: [0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            color: [0.2, 0.3, 0.4, 1.0],
        }];
        let mut flash = Vec::new();
        prepare_tnt_flash_vertices(Some(EntityType::PrimedTnt), 6, 0.0, &body, &mut flash);
        assert_eq!(flash[0].color, objects::TNT_FLASH_COLOR);
        // The checkout may use CRLF; the assertions below are about the code, not line endings.
        let source = include_str!("entrender.rs").replace("\r\n", "\n");
        let (_, upload) = source
            .split_once("pub fn upload_objects(")
            .expect("object uploader");
        let (upload, _) = upload
            .split_once("fn fireball_overlay_size(")
            .expect("object uploader end");
        assert!(upload.contains("object_combined_light,\n                    &verts"));
        assert!(upload.contains(
            "LayerBlend::TntFlash,\n                object_combined_light,\n                &tnt_flash_verts"
        ));
        assert_eq!(
            object_shadow(EntityType::PrimedTnt),
            Some(ObjectShadow {
                size: 0.5,
                opacity: 1.0,
            }),
            "lighting changes must not alter TNT's inherited shadow contract"
        );
    }

    #[test]
    fn thrown_renderers_inherit_world_light_except_ender_eye_full_bright() {
        use EntityType as E;

        let position = DVec3::new(0.25, 64.25, 0.25);
        let sample = BlockPos::from_f64(position.x, position.y, position.z);
        let inherited = [
            E::Snowball,
            E::ThrownEgg,
            E::ThrownEnderpearl,
            E::ThrownExpBottle,
            E::ThrownPotion,
            E::FireworksRocketEntity,
        ];
        let mut world = World::new();

        // RenderManager selects Entity.getBrightnessForRender before dispatching each of these
        // RenderSnowball/RenderPotion bodies. A loaded dark cell must therefore stay dark rather
        // than taking the object's FULL_BRIGHT compatibility fallback.
        world.set_has_sky_light(true);
        world.load_empty_chunk(sample.chunk_pos());
        world.set_light_for(mc_world::SkyBlock::Block, sample, 0);
        world.set_light_for(mc_world::SkyBlock::Sky, sample, 0);
        for kind in inherited {
            assert_eq!(
                inherited_object_light(Some(kind), &world, position),
                Some(0),
                "{kind:?} must inherit a loaded dark world sample"
            );
        }

        // Keep the two Java packed channels independently observable: the block nibble occupies
        // bits 4..7 and the sky nibble occupies bits 20..23.
        world.set_light_for(mc_world::SkyBlock::Block, sample, 5);
        world.set_light_for(mc_world::SkyBlock::Sky, sample, 12);
        let packed = (5_u32 << 4) | (12_u32 << 20);
        for kind in inherited {
            assert_eq!(
                inherited_object_light(Some(kind), &world, position),
                Some(packed),
                "{kind:?} must preserve packed block and sky light"
            );
        }

        // EntityEnderEye.getBrightnessForRender is an entity-specific Java 15728880 override;
        // keeping None here preserves its full-bright renderer fallback instead of sampling this
        // world's dark/block/sky value.
        assert_eq!(
            inherited_object_light(Some(E::EyeOfEnderSignal), &world, position),
            None,
            "Ender Eye must retain its Java full-bright exception"
        );
        for kind in [E::Fireball, E::SmallFireball, E::WitherSkull] {
            assert_eq!(
                inherited_object_light(Some(kind), &world, position),
                None,
                "{kind:?} inherits EntityFireball's explicit full-bright exception"
            );
        }
        assert_eq!(
            inherited_object_light(Some(E::Snowball), &World::new(), position),
            Some(0),
            "an unloaded inherited RenderSnowball sample must be dark"
        );
    }

    #[test]
    fn thrown_renderers_sample_current_eye_cell_not_interpolated_draw_pose() {
        use EntityType as E;

        let current = DVec3::new(0.25, 63.75, 0.25);
        let draw_pose = DVec3::new(-1.25, 63.75, -1.25);
        let current_sample = BlockPos::new(0, 63, 0);
        let draw_sample = BlockPos::new(-2, 63, -2);
        let mut world = World::new();
        world.set_has_sky_light(true);
        world.load_empty_chunk(current_sample.chunk_pos());
        world.load_empty_chunk(draw_sample.chunk_pos());
        world.set_light_for(mc_world::SkyBlock::Block, current_sample, 4);
        world.set_light_for(mc_world::SkyBlock::Sky, current_sample, 9);
        world.set_light_for(mc_world::SkyBlock::Block, draw_sample, 12);
        world.set_light_for(mc_world::SkyBlock::Sky, draw_sample, 2);

        // RenderManager computes the packed value from Entity.pos + getEyeHeight before
        // interpolating lastTickPos into the draw arguments.  All RenderSnowball registrations
        // and RenderPotion must retain that order, even when the frame pose crosses a block.
        for (network_id, kind) in [
            (101, E::Snowball),
            (102, E::ThrownEgg),
            (103, E::ThrownEnderpearl),
            (104, E::ThrownExpBottle),
            (105, E::ThrownPotion),
            (106, E::FireworksRocketEntity),
        ] {
            let mut tracker = crate::sync::entities::EntityTracker::new();
            tracker.spawn(
                &mut world,
                network_id,
                TrackedKind::Type(kind),
                None,
                [8, 2040, 8],
                0.0,
                0.0,
                current,
                Metadata::new(),
            );
            let tracked = tracker.get(network_id).expect("spawned thrown object");
            let light_position = object_light_position(&world, tracked, Some(kind), draw_pose);
            assert_eq!(
                BlockPos::from_f64(light_position.x, light_position.y, light_position.z),
                current_sample,
                "{kind:?} light sample must use current entity eye coordinates"
            );
            assert_eq!(
                inherited_object_light(Some(kind), &world, light_position),
                Some((4_u32 << 4) | (9_u32 << 20)),
                "{kind:?} must preserve the current cell's packed light"
            );
            assert_ne!(
                inherited_object_light(Some(kind), &world, draw_pose),
                inherited_object_light(Some(kind), &world, light_position),
                "{kind:?} negative control: interpolated draw pose is a different light cell"
            );
        }

        // EntityEnderEye overrides getBrightnessForRender with Java's 15728880 constant.  Its
        // renderer therefore deliberately takes the full-bright fallback rather than this cell.
        let mut tracker = crate::sync::entities::EntityTracker::new();
        tracker.spawn(
            &mut world,
            107,
            TrackedKind::Type(E::EyeOfEnderSignal),
            None,
            [8, 2040, 8],
            0.0,
            0.0,
            current,
            Metadata::new(),
        );
        let tracked = tracker.get(107).expect("spawned ender eye");
        assert_eq!(
            inherited_object_light(
                Some(E::EyeOfEnderSignal),
                &world,
                object_light_position(&world, tracked, Some(E::EyeOfEnderSignal), draw_pose),
            ),
            None,
            "Ender Eye must keep the explicit full-bright renderer path"
        );
    }

    #[test]
    fn fishing_bobber_world_light_is_dark_or_packed_block_sample() {
        // RenderManager feeds RenderFish EntityFishHook.getBrightnessForRender's packed world
        // value. Keep this oracle intentionally small: an unloaded/dark hook must be zero, while
        // a torch-lit hook must expose the block-light nibble instead of the old full-bright path.
        let position = DVec3::new(0.25, 64.25, 0.25);
        let sample = BlockPos::from_f64(position.x, position.y, position.z);
        let mut world = World::new();
        assert_eq!(fishing_bobber_combined_light(&world, position), 0);

        world.set_has_sky_light(true);
        world.load_empty_chunk(sample.chunk_pos());
        world.set_light_for(mc_world::SkyBlock::Sky, sample, 0);
        world.set_light_for(mc_world::SkyBlock::Block, sample, 0);
        assert_eq!(
            fishing_bobber_combined_light(&world, position),
            0,
            "a loaded dark bobber must not use the full-bright compatibility value"
        );

        world.set_light_for(mc_world::SkyBlock::Block, sample, 11);
        assert_eq!(
            fishing_bobber_combined_light(&world, position),
            11_u32 << 4,
            "a torch-lit bobber carries the live block-light nibble"
        );

        world.set_light_for(mc_world::SkyBlock::Sky, sample, 8);
        assert_eq!(
            fishing_bobber_combined_light(&world, position),
            (11_u32 << 4) | (8_u32 << 20),
            "a bobber preserves both packed block- and sky-light channels"
        );
    }

    #[test]
    fn arrow_light_samples_current_eye_pose_while_geometry_keeps_interpolation() {
        let mut world = World::new();
        let mut tracker = crate::sync::entities::EntityTracker::new();
        let current = DVec3::new(0.5, 32.0, 0.5);
        tracker.spawn(
            &mut world,
            93,
            TrackedKind::Type(EntityType::Arrow),
            None,
            [16, 1024, 16],
            0.0,
            0.0,
            current,
            Metadata::new(),
        );
        let tracked = tracker.get(93).expect("spawned arrow");
        let eye = current + DVec3::new(0.0, f64::from(tracked.eye_height(&world)), 0.0);
        let sample = BlockPos::from_f64(eye.x, eye.y, eye.z);
        world.load_empty_chunk(sample.chunk_pos());

        // A deliberately unrelated interpolated draw pose must not move the light sample.
        let interpolated = DVec3::new(9.5, 32.0, 9.5);
        assert_eq!(
            object_light_position(&world, tracked, Some(EntityType::Arrow), interpolated),
            eye,
            "RenderArrow geometry interpolation must not alter Entity.getBrightnessForRender's current-eye sample"
        );
        assert_ne!(eye, interpolated);

        // Item keeps the same base-Eye rule, while a living entity remains on the ordinary
        // render-pose route as a negative control for this narrow object seam.
        assert_eq!(
            object_light_position(&world, tracked, Some(EntityType::Item), interpolated),
            eye
        );
        assert_eq!(
            object_light_position(&world, tracked, Some(EntityType::Cow), interpolated),
            interpolated
        );
    }

    /// Vanilla-default-settings wrapper (fancy graphics, smooth lighting) for the CPU tests.
    fn emit_falling_block_model(
        world: &World,
        state: mc_data::BlockState,
        block_pos: BlockPos,
        model: &mc_assets::BakedModel,
        transform: mc_render::entity::Mat4,
        out: &mut Vec<EntityVertex>,
    ) {
        emit_falling_block_model_with_settings(
            world,
            state,
            block_pos,
            model,
            transform,
            mc_render::terrain::MeshSettings::default(),
            out,
        );
    }

    #[test]
    fn falling_block_model_applies_position_tint_and_ambient_occlusion() {
        use mc_assets::{BakedModel, BakedQuad, Vertex};

        let quad = BakedQuad {
            verts: [
                Vertex {
                    pos: [0.0, 1.0, 0.0],
                    uv: [0.0, 0.0],
                },
                Vertex {
                    pos: [1.0, 1.0, 0.0],
                    uv: [1.0, 0.0],
                },
                Vertex {
                    pos: [1.0, 1.0, 1.0],
                    uv: [1.0, 1.0],
                },
                Vertex {
                    pos: [0.0, 1.0, 1.0],
                    uv: [0.0, 1.0],
                },
            ],
            face: Some(Facing::Up),
            shade_face: Some(Facing::Up),
            tint_index: Some(0),
            shade: false,
        };
        let mut model = BakedModel::default();
        model.general_quads.push(quad);
        let pos = BlockPos::new(0, 64, 0);
        let waterlily = mc_data::block_by_name("waterlily")
            .expect("waterlily")
            .default_state();
        let mut world = World::new();
        world.load_empty_chunk(pos.chunk_pos());

        let mut tint_vertices = Vec::new();
        emit_falling_block_model(
            &world,
            waterlily,
            pos,
            &model,
            mc_render::entity::Mat4::IDENTITY,
            &mut tint_vertices,
        );
        let tint = mc_render::terrain::tint::color_multiplier(&world, waterlily, pos);
        let expected_tint = [
            ((tint >> 16) & 255) as f32 / 255.0,
            ((tint >> 8) & 255) as f32 / 255.0,
            (tint & 255) as f32 / 255.0,
            1.0,
        ];
        assert_eq!(tint_vertices.len(), 6);
        assert!(tint_vertices
            .iter()
            .all(|vertex| vertex.color == expected_tint));
        assert_ne!(
            expected_tint,
            EntityVertex::WHITE,
            "tinted falling-block quads must not remain white"
        );

        let stone = mc_data::block_by_name("stone")
            .expect("stone")
            .default_state();
        let mut plain_model = model.clone();
        plain_model.general_quads[0].tint_index = None;
        let mut open = World::new();
        open.load_empty_chunk(pos.chunk_pos());
        let mut open_vertices = Vec::new();
        emit_falling_block_model(
            &open,
            stone,
            pos,
            &plain_model,
            mc_render::entity::Mat4::IDENTITY,
            &mut open_vertices,
        );

        let mut occluded = World::new();
        occluded.load_empty_chunk(pos.chunk_pos());
        for dx in -1..=1 {
            for dz in -1..=1 {
                occluded.set_block(pos.up().add(dx, 0, dz), stone);
            }
        }
        let mut occluded_vertices = Vec::new();
        emit_falling_block_model(
            &occluded,
            stone,
            pos,
            &plain_model,
            mc_render::entity::Mat4::IDENTITY,
            &mut occluded_vertices,
        );
        assert!(
            open_vertices[0].color[0] > occluded_vertices[0].color[0],
            "opaque neighbours must lower the smooth-lighting multiplier"
        );
        assert!(
            occluded_vertices.iter().any(|vertex| vertex.color[0] < 1.0),
            "the AO path must be observable in emitted entity colours"
        );
    }

    /// The live object dispatcher lights a falling block with the
    /// settings the frame hands it. java: `RenderFallingBlock.doRender` ->
    /// `BlockModelRenderer.renderModel`, whose smooth path needs
    /// `Minecraft.isAmbientOcclusionEnabled()`: with smooth lighting off, the sand is lit flat
    /// like the same block in the terrain. Previously, the dispatcher always used the
    /// vanilla defaults, so a player with smooth lighting off still saw AO-darkened falling
    /// blocks; the negative control shows this scene's AO is visible. Needs the vanilla jar
    /// (skips without it; `MC_JARS_DIR`).
    #[test]
    fn falling_block_dispatch_uses_the_live_smooth_lighting_setting() {
        let Some(jar) = crate::app::config::discover_jars_dir()
            .map(|dir| dir.join("versions").join("1.8.9").join("1.8.9.jar"))
            .filter(|jar| jar.is_file())
        else {
            eprintln!("skip: no vanilla jar in this checkout (set MC_JARS_DIR)");
            return;
        };
        let mut manager = mc_assets::ResourceManager::new();
        manager.push_pack(Box::new(
            mc_assets::ZipResourcePack::open(&jar).expect("open vanilla jar"),
        ));
        let (assets, report) = mc_assets::bake_all(&manager, 2_048).expect("bake vanilla models");
        assert!(report.is_clean(), "{:?}", report.failures);
        mc_assets::model::reinstall(assets);

        let sand = mc_data::block_by_name("sand")
            .expect("sand")
            .default_state();
        let stone = mc_data::block_by_name("stone")
            .expect("stone")
            .default_state();
        let cell = BlockPos::new(2, 64, 3);
        let mut world = World::new();
        world.load_empty_chunk(cell.chunk_pos());
        // Stone north-west of the cell above darkens the top face's smooth corners there; a
        // torch-lit east sample and a sky-lit south one make the smooth corners' packed light
        // differ from the flat (dark) centre. (`getAoBrightness` replaces a zero sample with the
        // centre, so dark samples alone could not tell the two paths apart.)
        for offset in [(-1, 0), (0, -1), (-1, -1)] {
            world.set_block(cell.up().add(offset.0, 0, offset.1), stone);
        }
        world.set_light_for(mc_world::SkyBlock::Block, cell.up().east(), 15);
        world.set_light_for(mc_world::SkyBlock::Sky, cell.up().south(), 15);
        let mut tracker = crate::sync::entities::EntityTracker::new();
        let entity = tracker.spawn(
            &mut world,
            7,
            TrackedKind::Type(EntityType::FallingSand),
            None,
            [cell.x * 32 + 16, cell.y * 32, cell.z * 32 + 16],
            0.0,
            0.0,
            mc_math::DVec3::ZERO,
            Metadata::new(),
        );
        world
            .ecs
            .insert_one(entity, ObjectData(sand.to_java_state_id()))
            .expect("object data");
        let tracked = tracker.get(7).expect("spawned falling block");
        let position = mc_math::DVec3::new(
            f64::from(cell.x) + 0.5,
            f64::from(cell.y),
            f64::from(cell.z) + 0.5,
        );

        let emit = |settings: mc_render::terrain::MeshSettings| {
            let mut models = RenderManager::new();
            let mut tile_models = TileModels::new(false);
            let (mut out, mut glint, mut painting) = (Vec::new(), Vec::new(), Vec::new());
            let texture = emit_object(
                &mut out,
                &mut models,
                &mut tile_models,
                &world,
                None,
                tracked,
                EntityType::FallingSand,
                &Camera::default(),
                0.0,
                crate::app::appearance::ShadowLighting::default(),
                &mut glint,
                0,
                &mut painting,
                settings,
            );
            assert_eq!(texture, Some(ObjectTexture::Resource(BLOCK_ATLAS_TEXTURE)));
            out.iter().map(|vertex| vertex.color).collect::<Vec<_>>()
        };
        let model = mc_assets::model_for(sand);
        let expected = |settings: mc_render::terrain::MeshSettings| {
            falling_block_model_lighting(&world, sand, cell, model, settings)
                .into_iter()
                .flat_map(|quad| {
                    FALLING_BLOCK_QUAD_INDICES.map(|vertex| quad.color_multiplier[vertex])
                })
                .collect::<Vec<_>>()
        };
        let expected_lights = |settings: mc_render::terrain::MeshSettings| {
            falling_block_model_lighting(&world, sand, cell, model, settings)
                .into_iter()
                .flat_map(|quad| FALLING_BLOCK_QUAD_INDICES.map(|vertex| quad.packed_light[vertex]))
                .collect::<Vec<_>>()
        };
        let smooth_settings = mc_render::terrain::MeshSettings::default();
        let flat_settings = mc_render::terrain::MeshSettings {
            ambient_occlusion: false,
            ..smooth_settings
        };

        let smooth = emit(smooth_settings);
        let flat = emit(flat_settings);
        assert!(!flat.is_empty(), "the sand model emits geometry");
        assert_eq!(
            flat,
            expected(flat_settings),
            "smooth lighting off lights flat"
        );
        assert_eq!(
            smooth,
            expected(smooth_settings),
            "smooth lighting on keeps AO"
        );
        assert_ne!(flat, smooth, "negative control: this scene's AO is visible");

        let lights = |settings| {
            falling_block_vertex_lights_for_entity(&world, tracked, position, settings)
                .expect("a model falling block has per-vertex light")
        };
        assert_eq!(lights(flat_settings), expected_lights(flat_settings));
        assert_eq!(lights(smooth_settings), expected_lights(smooth_settings));
        assert_ne!(
            lights(flat_settings),
            lights(smooth_settings),
            "negative control: the packed light follows the same switch"
        );
    }

    #[test]
    fn falling_block_lighting_retains_mixed_packed_corners_and_ao_toggle() {
        use mc_assets::{BakedModel, BakedQuad, Vertex};

        let quad = BakedQuad {
            verts: [
                Vertex {
                    pos: [0.0, 1.0, 0.0],
                    uv: [0.0, 0.0],
                },
                Vertex {
                    pos: [1.0, 1.0, 0.0],
                    uv: [1.0, 0.0],
                },
                Vertex {
                    pos: [1.0, 1.0, 1.0],
                    uv: [1.0, 1.0],
                },
                Vertex {
                    pos: [0.0, 1.0, 1.0],
                    uv: [0.0, 1.0],
                },
            ],
            face: Some(Facing::Up),
            shade_face: Some(Facing::Up),
            tint_index: None,
            shade: true,
        };
        let mut model = BakedModel::default();
        model.general_quads.push(quad);
        let state = mc_data::block_by_name("stone")
            .expect("stone")
            .default_state();
        let pos = BlockPos::new(0, 64, 0);
        let mut world = World::new();
        world.load_empty_chunk(pos.chunk_pos());
        world.set_has_sky_light(true);

        for dx in -1..=1 {
            for dz in -1..=1 {
                world.set_block(pos.up().add(dx, 0, dz), state);
            }
        }

        // Clear the surrounding sky/block samples first so the two channels below are the only
        // sources. This makes the oracle sensitive to packed-channel swaps, not just brightness.
        for dx in -2..=2 {
            for dz in -2..=2 {
                let sample = pos.up().add(dx, 0, dz);
                world.set_light_for(mc_world::SkyBlock::Sky, sample, 0);
                world.set_light_for(mc_world::SkyBlock::Block, sample, 0);
            }
        }

        // The UP face is flush, so its four AO corners read the four horizontal samples one
        // block above the falling block. One corner is torch-lit, one is sky-lit, and two remain
        // dark.
        world.set_light_for(mc_world::SkyBlock::Block, pos.up().east(), 15);
        world.set_light_for(mc_world::SkyBlock::Sky, pos.up().south(), 15);

        let smooth = falling_block_model_lighting(
            &world,
            state,
            pos,
            &model,
            mc_render::terrain::MeshSettings::default(),
        );
        assert_eq!(smooth.len(), 1);
        let packed = smooth[0].packed_light;
        let block_bytes: Vec<u8> = packed
            .into_iter()
            .map(mc_render::terrain::light::block_byte)
            .collect();
        assert!(
            block_bytes.contains(&0) && block_bytes.iter().any(|&light| light >= 60),
            "mixed dark/bright samples must survive as per-vertex packed block light: {block_bytes:?}"
        );
        assert!(packed
            .iter()
            .any(|&light| mc_render::terrain::light::sky_byte(light) > 0));
        assert!(
            packed.iter().any(|&light| {
                let block = mc_render::terrain::light::block_byte(light);
                let sky = mc_render::terrain::light::sky_byte(light);
                block > 0 && sky != block
            }),
            "block and sky channels must remain independently packed"
        );

        let flat = falling_block_model_lighting(
            &world,
            state,
            pos,
            &model,
            mc_render::terrain::MeshSettings {
                ambient_occlusion: false,
                ..mc_render::terrain::MeshSettings::default()
            },
        );
        assert!(
            flat[0]
                .color_multiplier
                .iter()
                .all(|color| color == &[1.0, 1.0, 1.0, 1.0]),
            "AO-off must take Java's flat colour path"
        );
        assert!(
            smooth[0]
                .color_multiplier
                .iter()
                .any(|color| color[0] < 1.0),
            "AO-on must darken at least one corner"
        );
    }

    fn item_stack(name: &str, damage: i16) -> mc_protocol::ItemStack {
        mc_protocol::ItemStack {
            item_id: mc_data::item_by_name(name).expect("registered item").0 as i16,
            count: 1,
            damage,
            nbt: None,
        }
    }

    /// java: `RenderManager:177-183` — the six `RenderSnowball` registrations plus the potion,
    /// each resolving to a real item id.
    #[test]
    fn every_thrown_entity_resolves_to_its_vanilla_item() {
        use EntityType as E;
        for kind in [
            E::Snowball,
            E::ThrownEgg,
            E::ThrownEnderpearl,
            E::EyeOfEnderSignal,
            E::ThrownExpBottle,
            E::FireworksRocketEntity,
            E::ThrownPotion,
        ] {
            assert!(thrown_item_stack(kind, None).is_some(), "{kind:?}");
        }
        // Everything else has no `RenderSnowball` registration.
        assert!(thrown_item_stack(E::Arrow, None).is_none());
        assert!(thrown_item_stack(E::Creeper, None).is_none());
    }

    /// `RenderItemFrame.doRender` uses the floor-derived hanging block centre, while its outer
    /// turn reads the entity yaw that the spawn packet writes after construction.
    #[test]
    fn item_frame_facing_selects_the_hanging_plane_and_block_centre() {
        let camera = Camera::default();
        let anchor = BlockPos::new(10, 64, -4);
        let centre = mc_math::Vec3::new(10.5, 64.5, -3.5);
        for (facing, normal) in [
            (Facing::South, mc_math::Vec3::NEG_Z),
            (Facing::West, mc_math::Vec3::X),
            (Facing::North, mc_math::Vec3::Z),
            (Facing::East, mc_math::Vec3::NEG_X),
        ] {
            let (dx, _, dz) = facing.offset();
            let position = mc_math::DVec3::new(
                f64::from(anchor.x) + 0.5 - f64::from(dx) * 0.46875,
                f64::from(anchor.y) + 0.5,
                f64::from(anchor.z) + 0.5 - f64::from(dz) * 0.46875,
            );
            let yaw = f32::from(facing.horizontal_index().unwrap()) * 90.0;
            let matrix = item_frame_matrix(position, &camera, facing, yaw);
            assert!(
                matrix
                    .transform_point3(mc_math::Vec3::ZERO)
                    .abs_diff_eq(centre, 1e-5),
                "{facing:?}"
            );
            assert!(
                matrix
                    .transform_vector3(mc_math::Vec3::Z)
                    .abs_diff_eq(normal, 1e-5),
                "{facing:?}"
            );
        }
        // `handleSpawnObject` overwrites the constructor yaw from the packet after creating the
        // facing-specific anchor. A malformed/mixed packet must therefore rotate the frame by
        // the packet yaw while keeping that same wall plane.
        let position = mc_math::DVec3::new(10.5, 64.5, -3.03125);
        let mismatched = item_frame_matrix(position, &camera, Facing::South, 90.0);
        assert!(
            mismatched
                .transform_vector3(mc_math::Vec3::Z)
                .abs_diff_eq(mc_math::Vec3::X, 1e-5),
            "outer item-frame rotation follows packet yaw, not constructor facing"
        );
    }

    #[test]
    fn item_frame_world_light_is_packed_at_the_frame_sample_and_scoped_to_frames() {
        let sample = mc_math::DVec3::new(0.25, 64.25, 0.25);
        let mut world = World::new();
        world.set_has_sky_light(true);
        world.load_empty_chunk(mc_math::ChunkPos::new(0, 0));
        let block = BlockPos::from_f64(sample.x, sample.y, sample.z);
        world.set_light_for(mc_world::SkyBlock::Block, block, 4);
        world.set_light_for(mc_world::SkyBlock::Sky, block, 7);
        assert_eq!(
            inherited_object_light(Some(EntityType::ItemFrame), &world, sample),
            Some((4_u32 << 4) | (7_u32 << 20)),
            "frame content must inherit Java's packed block/sky lightmap"
        );
        assert_eq!(
            inherited_object_light(Some(EntityType::Item), &world, sample),
            Some((4_u32 << 4) | (7_u32 << 20)),
            "ordinary dropped items use their own packed world-light sample"
        );
        assert_eq!(
            inherited_object_light(Some(EntityType::ItemFrame), &World::new(), sample),
            Some(0),
            "a frame in an unloaded chunk must not inherit blank-column skylight"
        );
    }

    #[test]
    fn xp_orb_world_light_adds_java_half_bright_block_boost_and_keeps_sky() {
        // Entity.getBrightnessForRender samples the current feet position plus
        // Entity.getEyeHeight (0.85 * 0.5 = 0.425). This y deliberately differs from the feet
        // cell so sampling the raw position would hit the wrong light entry.
        let position = mc_math::DVec3::new(0.25, 63.75, 0.25);
        let sample = BlockPos::from_f64(
            position.x,
            position.y + f64::from(EntityType::XPOrb.info().eye_height),
            position.z,
        );
        let mut world = World::new();
        world.set_has_sky_light(true);
        world.load_empty_chunk(sample.chunk_pos());
        world.set_light_for(mc_world::SkyBlock::Sky, sample, 4);

        // Java's EntityXPOrb override adds 0.5 * 15 * 16 = 120 to the packed block-light axis.
        world.set_light_for(mc_world::SkyBlock::Block, sample, 0);
        assert_eq!(
            inherited_object_light(
                Some(EntityType::XPOrb),
                &world,
                position + DVec3::new(0.0, f64::from(EntityType::XPOrb.info().eye_height), 0.0),
            ),
            Some(120 | (4_u32 << 20)),
            "dark-world orbs still receive the exact half-bright boost"
        );

        // A loaded cell whose two channels are both zero is not the same thing as an unloaded
        // cell. Java still applies the orb's +120 block-light override to the former.
        world.set_light_for(mc_world::SkyBlock::Sky, sample, 0);
        assert_eq!(
            inherited_object_light(
                Some(EntityType::XPOrb),
                &world,
                position + DVec3::new(0.0, f64::from(EntityType::XPOrb.info().eye_height), 0.0),
            ),
            Some(120),
            "a loaded black cell must still receive XP's half-bright boost"
        );
        assert_eq!(
            inherited_object_light(
                Some(EntityType::XPOrb),
                &World::new(),
                position + DVec3::new(0.0, f64::from(EntityType::XPOrb.info().eye_height), 0.0),
            ),
            Some(0),
            "an unloaded XP sample must remain zero before the override"
        );

        world.set_light_for(mc_world::SkyBlock::Block, sample, 7);
        world.set_light_for(mc_world::SkyBlock::Sky, sample, 4);
        assert_eq!(
            inherited_object_light(
                Some(EntityType::XPOrb),
                &world,
                position + DVec3::new(0.0, f64::from(EntityType::XPOrb.info().eye_height), 0.0),
            ),
            Some((7_u32 << 4) + 120 + (4_u32 << 20)),
            "block light is boosted without changing sky light"
        );

        world.set_light_for(mc_world::SkyBlock::Block, sample, 15);
        assert_eq!(
            inherited_object_light(
                Some(EntityType::XPOrb),
                &world,
                position + DVec3::new(0.0, f64::from(EntityType::XPOrb.info().eye_height), 0.0),
            ),
            Some(240 + (4_u32 << 20)),
            "the boosted block-light component clamps at 240"
        );

        // Ordinary living entities retain the compatibility full-bright route, dropped items
        // keep their unadjusted sampled light, and item frames do not inherit the XP-orb boost.
        assert_eq!(
            inherited_object_light(Some(EntityType::Cow), &world, position),
            None
        );
        let frame_position = mc_math::DVec3::new(0.25, 64.25, 0.25);
        assert_eq!(
            inherited_object_light(Some(EntityType::Item), &world, frame_position),
            Some(240 + (4_u32 << 20))
        );
        assert_eq!(
            inherited_object_light(Some(EntityType::ItemFrame), &world, frame_position),
            Some((15_u32 << 4) | (4_u32 << 20))
        );
    }

    #[test]
    fn filled_maps_select_the_map_frame_path_not_the_generic_item_path() {
        assert!(is_filled_map(&item_stack("filled_map", 0)));
        assert!(!is_filled_map(&item_stack("map", 0)));
        assert!(!is_filled_map(&item_stack("compass", 0)));
    }

    /// `RenderItemFrame.renderItem` keeps the dynamic map content outside the baked
    /// `item_frame#map` model. The content is exactly one 128-square, point-sampled dynamic
    /// texture quad, centred by the `-64` translation and pulled infinitesimally in front of the
    /// model by the two negative-Z translations.
    #[test]
    fn filled_item_frame_exposes_its_live_map_pixels_and_exact_quad() {
        use mc_protocol::play::s2c::{MapPixelUpdate, Maps};

        let mut world = World::new();
        world
            .apply_map_packet(&Maps {
                map_id: 27,
                scale: 0,
                icons: vec![],
                update: Some(MapPixelUpdate {
                    width: 1,
                    height: 1,
                    min_x: 5,
                    min_y: 6,
                    // Palette eight is white; shade one makes the output 220, not a generic
                    // full-white fallback. This proves the render handoff uses live MapData.
                    data: vec![8 * 4 + 1],
                }),
            })
            .unwrap();
        let render = item_frame_map_render(
            &world,
            &item_stack("filled_map", 27),
            mc_render::entity::Mat4::IDENTITY,
            0,
        )
        .expect("S34-created map id is rendered");

        assert_eq!(render.map.id, 27);
        let pixels = render.rgba().expect("valid map palette byte");
        assert_eq!(pixels.len(), mc_ui::map::MAP_TEXTURE_BYTES);
        assert_eq!(
            &pixels[(5 + 6 * mc_world::MAP_SIZE) * 4..][..4],
            &[220, 220, 220, 255]
        );
        // The heap conversion is the `MapTexture` conversion, byte for byte,
        // and an invalid palette byte still yields nothing.
        let texture = mc_ui::map::MapTexture::from_data(render.map).unwrap();
        assert_eq!(pixels.as_slice(), texture.rgba.as_slice());
        let mut broken = render.map.clone();
        broken.colors[9] = 36 * 4;
        let broken_render = ItemFrameMapRender {
            map: &broken,
            vertices: render.vertices,
            icon_vertices: Vec::new(),
        };
        assert!(broken_render.rgba().is_none());
        assert_eq!(render.vertices.len(), 6);
        let positions = render
            .vertices
            .map(|vertex| mc_math::Vec3::from(vertex.pos));
        assert!(positions[0].abs_diff_eq(mc_math::Vec3::new(0.5, -0.5, 0.429_609_4), 1e-6));
        assert!(positions[2].abs_diff_eq(mc_math::Vec3::new(-0.5, 0.5, 0.429_609_4), 1e-6));
        assert!(render.vertices.iter().all(|vertex| {
            mc_math::Vec3::from(vertex.normal).abs_diff_eq(mc_math::Vec3::NEG_Z, 1e-6)
        }));
        assert_eq!(
            render.vertices.map(|vertex| vertex.uv),
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0]
            ]
        );
    }

    #[test]
    fn filled_item_frame_needs_the_stack_metadata_map_and_collapses_rotation_to_quarters() {
        let world = World::new();
        assert!(item_frame_map_render(
            &world,
            &item_stack("filled_map", 27),
            mc_render::entity::Mat4::IDENTITY,
            0,
        )
        .is_none());

        // Java's `i = i % 4 * 2` makes click positions one and five draw at the same map
        // orientation, unlike ordinary fixed-item geometry which retains all eight turns.
        let one = item_frame_map_matrix(mc_render::entity::Mat4::IDENTITY, 1);
        let five = item_frame_map_matrix(mc_render::entity::Mat4::IDENTITY, 5);
        let two = item_frame_map_matrix(mc_render::entity::Mat4::IDENTITY, 2);
        assert!(one.abs_diff_eq(five, 1e-6));
        assert!(!one.abs_diff_eq(two, 1e-6));
    }

    #[test]
    fn filled_item_frame_keeps_only_the_live_frame_marker_after_its_map_image() {
        use mc_protocol::play::s2c::{MapIcon, Maps};

        let mut world = World::new();
        world
            .apply_map_packet(&Maps {
                map_id: -3,
                scale: 0,
                icons: vec![
                    MapIcon {
                        icon_type: 0,
                        x: 0,
                        z: 0,
                        direction: 0,
                    },
                    MapIcon {
                        icon_type: 1,
                        x: 8,
                        z: -4,
                        direction: 4,
                    },
                ],
                update: None,
            })
            .unwrap();

        let render = item_frame_map_render(
            &world,
            &item_stack("filled_map", -3),
            mc_render::entity::Mat4::IDENTITY,
            0,
        )
        .unwrap();
        // `RenderItemFrame` calls `renderMap(..., true)`: type zero is suppressed, but type one
        // is the one exception. It is a second 2-triangle batch from map_icons.png.
        assert_eq!(render.icon_vertices.len(), 6);
        assert!(render.icon_vertices.iter().all(|vertex| {
            mc_math::Vec3::from(vertex.normal).abs_diff_eq(mc_math::Vec3::NEG_Z, 1e-6)
        }));
        // The image itself no longer has a per-map texture key: it lives in
        // the shared atlas (`map_atlas_world_path_*` tests below).
    }

    /// The allocation-free quad filter keeps exactly the triangles the
    /// shared `retain_front_facing_triangles` keeps, in the same order, from every side.
    #[test]
    fn front_facing_map_quad_matches_the_shared_triangle_filter() {
        let base = item_frame_map_matrix(mc_render::entity::Mat4::IDENTITY, 0);
        for yaw in (0..360).step_by(15) {
            for (x, z) in [(0.0_f32, 2.0_f32), (0.3, -2.0), (-1.5, 0.2), (0.0, 0.0)] {
                let matrix = mc_render::entity::Mat4::from_translation([x, 0.1, z].into())
                    * mc_render::entity::Mat4::from_rotation_y((yaw as f32).to_radians())
                    * base;
                let quad: [EntityVertex; 6] = core::array::from_fn(|index| {
                    let corner = [
                        [0.0_f32, 128.0],
                        [128.0, 128.0],
                        [128.0, 0.0],
                        [0.0, 128.0],
                        [128.0, 0.0],
                        [0.0, 0.0],
                    ][index];
                    EntityVertex {
                        pos: matrix
                            .transform_point3(mc_math::Vec3::new(corner[0], corner[1], -0.01))
                            .into(),
                        uv: [corner[0] / 128.0, corner[1] / 128.0],
                        normal: [0.0, 0.0, -1.0],
                        color: EntityVertex::WHITE,
                    }
                });
                for view_yaw in [0.0_f32, 90.0, 180.0, 270.0] {
                    let view = mc_render::entity::Mat4::from_rotation_y(view_yaw.to_radians());
                    let mut shared = quad.to_vec();
                    mc_render::entity::item::retain_front_facing_triangles(view, &mut shared);
                    let (kept, len) = front_facing_map_quad(view, &quad);
                    assert_eq!(&kept[..len], shared.as_slice(), "yaw {yaw} at ({x}, {z})");
                }
            }
        }
    }

    /// Java oracle: `RenderPotion.func_177082_d` retains the spawn damage and RenderItem's
    /// potion ItemMeshDefinition calls `ItemPotion.isSplash(meta)`, `(meta & 16384) != 0`.
    /// Non-potion RenderSnowball registrations always construct metadata-zero stacks.
    #[test]
    fn thrown_potion_model_damage_is_exactly_the_nonnegative_splash_bit() {
        use EntityType as E;
        let damage = |kind, data| thrown_item_stack(kind, data).unwrap().1;

        assert_eq!(damage(E::ThrownPotion, None), 0);
        assert_eq!(damage(E::ThrownPotion, Some(-1)), 0);
        assert_eq!(damage(E::ThrownPotion, Some(8194)), 0);
        assert_eq!(damage(E::ThrownPotion, Some(16_384)), 16_384);
        assert_eq!(damage(E::ThrownPotion, Some(16_386)), 16_384);
        assert_eq!(damage(E::ThrownPotion, Some(32_767)), 16_384);
        assert_eq!(damage(E::ThrownPotion, Some(0x1_4000)), 16_384);

        assert_eq!(damage(E::Snowball, Some(16_384)), 0);
        assert_eq!(damage(E::ThrownExpBottle, Some(16_384)), 0);

        // Biting controls for dropping ObjectData and retaining the whole potion metadata.
        assert_ne!(damage(E::ThrownPotion, Some(16_386)), 0);
        assert_ne!(damage(E::ThrownPotion, Some(16_386)), 16_386);
    }

    #[test]
    fn dropped_and_thrown_potion_tint_only_layer_zero() {
        let speed = item_stack("potion", 8194);
        assert_eq!(item_quad_tint(&speed, 0), 0x7C_AF_C6);
        assert_eq!(item_quad_tint(&speed, 1), 0xFF_FF_FF);
        assert_eq!(item_quad_tint(&item_stack("apple", 0), 0), 0xFF_FF_FF);

        assert_eq!(
            thrown_item_quad_tint(EntityType::ThrownPotion, Some(8196), 0),
            0x4E_93_31
        );
        assert_eq!(
            thrown_item_quad_tint(EntityType::ThrownPotion, Some(8196), 1),
            0xFF_FF_FF
        );
        assert_eq!(
            thrown_item_quad_tint(EntityType::Snowball, Some(8196), 0),
            0xFF_FF_FF
        );
    }

    #[test]
    fn entity_item_glint_predicate_matches_renderitem_has_effect_classes() {
        let plain = item_stack("apple", 0);
        assert!(!item_has_effect(&plain));

        // ItemStack.isItemEnchanted checks only for a TAG_List named `ench`; an empty list still
        // foils in Java.  A numeric replacement is deliberately not accepted by that predicate.
        let mut root = NbtCompound::new();
        root.insert("ench", Nbt::List(Vec::new().into()));
        let enchanted = mc_protocol::ItemStack {
            nbt: Some(root),
            ..item_stack("diamond_sword", 0)
        };
        assert!(item_has_effect(&enchanted));

        assert!(item_has_effect(&item_stack("written_book", 0)));
        assert!(item_has_effect(&item_stack("nether_star", 0)));
        assert!(!item_has_effect(&item_stack("golden_apple", 0)));
        assert!(item_has_effect(&item_stack("golden_apple", 1)));
        assert!(!item_has_effect(&item_stack("potion", 0)));
        assert!(item_has_effect(&item_stack("potion", 8_194)));

        let source = include_str!("entrender.rs");
        // `emit_pickup_effect` has its own item arm before the live object dispatcher.  Scope
        // this source contract to `emit_object`, whose RenderItem branch is the one under test.
        let live_objects = source
            .split_once("fn emit_object(")
            .expect("live object dispatcher")
            .1;
        let item_branch = live_objects
            .split_once("        E::Item => {")
            .expect("dropped-item render branch")
            .1
            .split_once("        E::XPOrb => {")
            .expect("bounded dropped-item render branch")
            .0;
        assert!(item_branch.contains("if item_has_effect(&stack) {"));
        assert!(item_branch.contains("dropped::emit_item_glint_from_base("));

        // Negative control: an arbitrary item with the same item id but no NBT does not inherit
        // the foil, so this cannot be implemented as a blanket entity-item overlay.
        assert_ne!(item_has_effect(&plain), item_has_effect(&enchanted));
    }

    #[test]
    fn thrown_renderitem_glint_predicate_preserves_raw_potion_damage() {
        use EntityType as E;

        assert!(thrown_item_has_effect(E::ThrownExpBottle, None));
        assert!(thrown_item_has_effect(E::ThrownPotion, Some(8_194)));
        assert!(!thrown_item_has_effect(E::ThrownPotion, Some(0)));
        assert!(!thrown_item_has_effect(E::Snowball, Some(8_194)));

        // The model selector intentionally keeps only bit 14, but hasEffect reads all potion
        // metadata.  This pair catches accidentally testing the canonicalised model damage.
        assert_eq!(
            thrown_item_stack(E::ThrownPotion, Some(8_194)).unwrap().1,
            0
        );
        assert!(thrown_item_has_effect(E::ThrownPotion, Some(8_194)));
    }

    #[test]
    fn dropped_firework_star_tints_only_overlay_layer_one() {
        let mut explosion = NbtCompound::new();
        explosion.insert(
            "Colors",
            Nbt::IntArray(vec![0xFF_0000, 0x00_FF00, 0x00_00FF]),
        );
        let mut root = NbtCompound::new();
        root.insert("Explosion", Nbt::Compound(explosion));
        let mut star = item_stack("firework_charge", 0);
        star.nbt = Some(root);

        assert_eq!(item_quad_tint(&star, 0), 0xFF_FF_FF);
        assert_eq!(item_quad_tint(&star, 1), 0x55_55_55);
        assert_eq!(item_quad_tint(&star, 2), 0xFF_FF_FF);
        assert_eq!(
            item_quad_tint(&item_stack("firework_charge", 0), 1),
            0x8A_8A_8A,
            "missing Colors retains ItemFireworkCharge's neutral overlay"
        );
        assert_eq!(
            item_quad_tint(&item_stack("apple", 0), 1),
            0xFF_FF_FF,
            "the tint-index match alone cannot colour another item"
        );
    }

    /// Dropped stacks use the GUI's stack-aware resolver, rather than a second, partial copy of
    /// `Item.getColorFromItemStack`.  These controls cover the metadata/NBT branches that the
    /// old potion/firework-only closure could never reach.
    #[test]
    fn dropped_spawn_eggs_and_leather_use_the_shared_item_tint_policy() {
        let mut egg = item_stack("spawn_egg", 54); // Zombie: EntityList egg (44975, 7969893).
        assert_eq!(item_quad_tint(&egg, 0), 0x00_AF_AF);
        assert_eq!(item_quad_tint(&egg, 7), 0x79_9C_65);
        egg.damage = 53; // Giant has an entity entry, but no EntityList.entityEggs entry.
        assert_eq!(item_quad_tint(&egg, 0), 0xFF_FF_FF);
        egg.damage = 300;
        assert_eq!(item_quad_tint(&egg, 1), 0xFF_FF_FF);

        let mut display = NbtCompound::new();
        display.insert("color", Nbt::Int(0x12_3456));
        let mut root = NbtCompound::new();
        root.insert("display", Nbt::Compound(display));
        let mut leather = item_stack("leather_helmet", 0);
        leather.nbt = Some(root);
        assert_eq!(item_quad_tint(&leather, 0), 0x12_3456);
        assert_eq!(item_quad_tint(&leather, 1), 0xFF_FF_FF);

        let mut display = NbtCompound::new();
        display.insert("color", Nbt::Byte(0x12));
        let mut root = NbtCompound::new();
        root.insert("display", Nbt::Compound(display));
        let mut wrong_tag_type = item_stack("leather_chestplate", 0);
        wrong_tag_type.nbt = Some(root);
        assert_eq!(item_quad_tint(&wrong_tag_type, 0), 0xA0_65_40);
    }

    /// The no-double-draw invariant.
    ///
    /// This pass and `Client::upload_entities` walk the *same* tracker in the same frame, and the
    /// only thing keeping them from both drawing an entity is that `renderer_for` returns `None`
    /// for everything handled here. If someone later adds one of these types to the
    /// `RendererLivingEntity` dispatch table, the entity is drawn twice — once correctly and once
    /// through the living matrix, a block and a half out of place. This is that alarm.
    #[test]
    fn nothing_this_pass_draws_is_also_in_the_living_dispatch_table() {
        use EntityType as E;
        for kind in [
            E::Item,
            E::XPOrb,
            E::Arrow,
            E::Snowball,
            E::ThrownEgg,
            E::ThrownEnderpearl,
            E::EyeOfEnderSignal,
            E::ThrownExpBottle,
            E::FireworksRocketEntity,
            E::ThrownPotion,
            E::Fireball,
            E::SmallFireball,
            E::PrimedTnt,
            E::FallingSand,
            E::ItemFrame,
            E::Painting,
            E::WitherSkull,
        ] {
            assert!(
                mc_render::entity::renderer_for(kind).is_none(),
                "{kind:?} is now in both dispatch paths and will be drawn twice"
            );
        }
    }

    #[test]
    fn dropped_item_xp_and_tnt_shadows_keep_java_sizes_and_paintings_stay_shadowless() {
        assert_eq!(
            object_shadow(EntityType::XPOrb),
            Some(ObjectShadow {
                size: 0.15,
                opacity: 0.75,
            })
        );
        assert_eq!(
            object_shadow(EntityType::Item),
            Some(ObjectShadow {
                size: 0.15,
                opacity: 0.75,
            })
        );
        assert_eq!(
            object_shadow(EntityType::PrimedTnt),
            Some(ObjectShadow {
                size: 0.5,
                opacity: 1.0,
            })
        );
        assert_eq!(object_shadow(EntityType::Painting), None);
    }

    /// The table above is only useful if the production object pass consumes it under the same
    /// settings/invisibility/distance gates as living shadows. Keep this source oracle beside the
    /// pure constants so dropping the live bridge is a biting mutation rather than dead code.
    #[test]
    fn object_shadow_descriptors_reach_the_live_shared_shadow_emitter() {
        let source = include_str!("entrender.rs");
        let production = source
            .split("#[cfg(test)]")
            .next()
            .expect("production object renderer");
        assert!(production.contains("if entity_shadows {"));
        assert!(production.contains("kind.and_then(object_shadow)"));
        assert!(production.contains("watched.is_invisible()"));
        assert!(production.contains("* shadow.opacity"));
        assert!(production.contains("emit_entity_shadow_for_environment("));
        assert!(production.contains("SHADOW_TEXTURE"));

        let driver = include_str!("driver.rs");
        assert!(driver.contains("session.riding(),\n                self.settings.entity_shadows,"));
    }

    #[test]
    fn charged_wither_skulls_select_the_blue_texture() {
        assert_eq!(
            wither_skull_texture(true),
            objects::WITHER_SKULL_INVULNERABLE_TEXTURE
        );
        assert_eq!(wither_skull_texture(false), objects::WITHER_SKULL_TEXTURE);
    }

    /// java: `EntityArrow.onUpdate` unwraps its *previous* samples, then `RenderArrow` performs
    /// a normal partial-tick lerp. In particular, `170 -> -170` becomes `-190 -> -170`, not the
    /// numerically different but geometrically equivalent `170 -> 190` formulation.
    #[test]
    fn arrow_rotation_lerps_the_unwrapped_short_arc_for_yaw_and_pitch() {
        let previous = (170.0, -170.0);
        let current = (-170.0, 170.0);

        assert_eq!(
            object_rotation(EntityType::Arrow, previous, current, 0.0),
            (-190.0, 190.0)
        );
        assert_eq!(
            object_rotation(EntityType::Arrow, previous, current, 0.5),
            (-180.0, 180.0)
        );
        assert_eq!(
            object_rotation(EntityType::Arrow, previous, current, 1.0),
            current
        );
    }

    /// `RenderWitherSkull` uses its renderer-local shortest-arc yaw helper and plain pitch lerp,
    /// while an unrelated non-living object still receives its current rotation unchanged.
    #[test]
    fn wither_skull_rotation_uses_shortest_yaw_but_plain_pitch() {
        let previous = (170.0, -170.0);
        let current = (-170.0, 170.0);

        assert_eq!(
            object_rotation(EntityType::WitherSkull, previous, current, 0.5),
            (180.0, 0.0)
        );
        assert_eq!(
            object_rotation(EntityType::WitherSkull, previous, current, 0.0),
            previous
        );
        assert_eq!(
            object_rotation(EntityType::WitherSkull, previous, current, 1.0),
            (190.0, 170.0)
        );

        // Negative control: only the WitherSkull renderer opts into this interpolation. A
        // snowball's Entity.setPositionAndRotation2 path remains an immediate current-angle read.
        assert_eq!(
            object_rotation(EntityType::Snowball, previous, current, 0.5),
            current
        );
    }

    /// The live `RenderWitherSkull` caller must use the projectile's no-argument
    /// `ModelSkeletonHead` (64x64, V offset 35), while ordinary tile skulls retain the explicit
    /// 64x32 constructor. Calling the object dispatcher makes a regression back to the tile key
    /// observable through both the cache count and the emitted UV sheet.
    #[test]
    fn wither_skull_dispatch_selects_projectile_model_and_preserves_tile_model() {
        let mut world = World::new();
        let mut tracker = crate::sync::entities::EntityTracker::new();
        tracker.spawn(
            &mut world,
            91,
            TrackedKind::Type(EntityType::WitherSkull),
            None,
            [0, 0, 0],
            0.0,
            0.0,
            mc_math::DVec3::ZERO,
            Metadata::new(),
        );
        let tracked = tracker.get(91).expect("spawned wither skull");
        let camera = Camera::default();
        let mut models = RenderManager::new();
        let mut tile_models = TileModels::new(false);
        let mut wither = Vec::new();
        let mut wither_glint = Vec::new();
        let mut painting = Vec::new();
        let texture = emit_object(
            &mut wither,
            &mut models,
            &mut tile_models,
            &world,
            None,
            tracked,
            EntityType::WitherSkull,
            &camera,
            0.5,
            crate::app::appearance::ShadowLighting::default(),
            &mut wither_glint,
            0,
            &mut painting,
            mc_render::terrain::MeshSettings::default(),
        );

        assert_eq!(
            texture,
            Some(ObjectTexture::Resource(objects::WITHER_SKULL_TEXTURE))
        );
        assert_eq!(
            models.built_models(),
            1,
            "the wither dispatch builds one model"
        );
        assert!(
            wither
                .iter()
                .any(|vertex| vertex.uv == [8.0 / 64.0, 43.0 / 64.0]),
            "wither skull must emit from the 64x64, V-offset-35 sheet"
        );

        let mut tile = Vec::new();
        models.emit_with(
            mc_render::entity::ModelKind::SkeletonHead,
            mc_render::entity::Mat4::IDENTITY,
            &AnimParams::still(),
            &mut tile,
        );
        assert_eq!(
            models.built_models(),
            2,
            "tile skull keeps its own cache entry"
        );
        assert!(
            tile.iter()
                .any(|vertex| vertex.uv == [8.0 / 64.0, 8.0 / 32.0]),
            "ordinary tile skull must retain its 64x32 UV sheet"
        );
        let wither_uvs: Vec<_> = wither.iter().map(|vertex| vertex.uv).collect();
        let tile_uvs: Vec<_> = tile.iter().map(|vertex| vertex.uv).collect();
        assert_ne!(
            wither_uvs, tile_uvs,
            "projectile and tile skulls must not share the tile model layout"
        );
    }

    /// `RenderArrow` receives the live counter after a client-side block collision and subtracts
    /// partial ticks before the existing `arrow_matrix` wobble guard. A settled (or missing)
    /// counter is intentionally the identity control.
    #[test]
    fn arrow_shake_reaches_the_wobble_with_partial_tick_precision() {
        let impact = ArrowShake(ArrowShake::BLOCK_IMPACT_TICKS);
        let shake = impact.render_value(0.25);
        assert_eq!(shake.to_bits(), 6.75_f32.to_bits());

        let settled = objects::arrow_matrix(Default::default(), 10.0, 20.0, 0.0);
        let wobbling = objects::arrow_matrix(Default::default(), 10.0, 20.0, shake);
        assert_ne!(wobbling, settled, "a post-impact arrow must rattle");
        assert_eq!(ArrowShake::default().render_value(0.75), -0.75);
    }

    /// Both world-item consumers use the baked bit, so a generated `ItemBlock` cannot regress to
    /// the old registry-class proxy while ordinary block models remain three-dimensional.
    #[test]
    fn world_item_gui_policy_is_asset_derived() {
        let generated_item_block = mc_assets::BakedModel::default();
        let ordinary_block = mc_assets::BakedModel {
            gui_3d: true,
            ..mc_assets::BakedModel::default()
        };
        assert!(!generated_item_block.gui_3d);
        assert!(ordinary_block.gui_3d);

        let production = include_str!("entrender.rs")
            .split_once("#[cfg(test)]")
            .expect("entity renderer keeps inline tests")
            .0;
        assert!(production.contains("gui_3d: builtin.is_some() || model.gui_3d"));
        assert!(production.contains("item_model_pre_transform(model.gui_3d)"));
        assert!(!production.contains("fn is_gui_3d("));
    }

    #[test]
    fn item_frame_fixed_pose_turns_flat_models_and_skulls_but_not_ordinary_3d_models() {
        let flat = mc_assets::BakedModel::default();
        let block = mc_assets::BakedModel {
            gui_3d: true,
            ..mc_assets::BakedModel::default()
        };
        assert!(item_frame_item_needs_y_flip(&flat, mc_data::ItemId(1),));
        assert!(!item_frame_item_needs_y_flip(&block, mc_data::ItemId(1)));
        // The Java condition names ItemSkull, not the model's GUI bit.  This synthetic 3-D
        // model is the biting control for an implementation that only checks `gui_3d`.
        assert!(item_frame_item_needs_y_flip(
            &block,
            mc_data::ItemId(mc_render::entity::tilepass::SKULL_ITEM_ID),
        ));
    }

    #[test]
    fn item_frame_compass_suppresses_watched_rotation_but_other_properties_do_not() {
        assert_eq!(
            item_frame_content_rotation(mc_assets::ItemModelProperty::Compass, 7),
            0,
            "the live compass sprite supplies its own world-bearing orientation"
        );
        assert_eq!(
            item_frame_content_rotation(mc_assets::ItemModelProperty::Clock, 7),
            7,
            "clock is dynamic but has no compass frame exception"
        );
        assert_eq!(
            item_frame_content_rotation(mc_assets::ItemModelProperty::None, -3),
            -3,
            "ordinary fixed items retain all eight frame rotations"
        );
    }

    #[test]
    fn item_frame_flat_pre_transform_and_common_half_cancel_like_java() {
        let display = mc_assets::model::ItemTransform {
            rotation: [0.0, 23.0, 0.0],
            translation: [0.125, -0.25, 0.0625],
            scale: [1.25, 0.75, 1.5],
        };
        let flat = mc_render::entity::objects::item_model_pre_transform(false)
            * mc_render::entity::objects::item_transform_matrix(&display)
            * mc_render::entity::Mat4::from_scale([0.5; 3].into());
        let gui_3d = mc_render::entity::objects::item_model_pre_transform(true)
            * mc_render::entity::objects::item_transform_matrix(&display)
            * mc_render::entity::Mat4::from_scale([0.5; 3].into());
        assert_ne!(
            flat, gui_3d,
            "flat fixed models must retain RenderItem.preTransform"
        );
        // The flat branch's `2 * display * .5` preserves display's dimensions; omitting either
        // the pre-transform or common half-scale changes every non-identity extent.
        let point = mc_render::entity::Mat4::IDENTITY.transform_point3(
            mc_render::entity::Mat4::from_scale([1.0; 3].into())
                .transform_point3([0.2, -0.4, 0.8].into()),
        );
        let transformed = flat.transform_point3(point);
        let expected = mc_render::entity::objects::item_transform_matrix(&display)
            .transform_point3(point * 0.5)
            * 2.0;
        for (got, want) in transformed.to_array().into_iter().zip(expected.to_array()) {
            assert!((got - want).abs() < 1e-6, "{transformed:?} != {expected:?}");
        }
    }

    /// `BuiltInModel.isGui3d()` sends tile items through `RenderEntityItem`'s 3-D copy loop,
    /// even though its baked model has no quads.  This is a CPU-only reachability control for the
    /// dropped caller: two copies must become two chest models, and the returned sheet must be
    /// the normal chest texture rather than the atlas/fallback path.
    #[test]
    fn dropped_builtin_item_uses_ground_copies_bob_and_tile_texture() {
        let model =
            mc_assets::item_model_for_damage(mc_data::ItemId(mc_world::tileent::CHEST_ID), 0);
        let item = dropped::DroppedItem {
            rel: mc_math::Vec3::new(4.0, 65.0, -2.0),
            age: 17,
            hover_start: 0.25,
            stack_size: 17,
            // This is the value vanilla's BuiltInModel reports, not the ordinary ItemBlock
            // predicate (which would also be true for stone but false for banner/skull).
            gui_3d: true,
            ground_scale: model.transforms.ground.scale,
        };
        let copies = item.copies(0.5);
        assert_eq!(copies.len(), 3, "17 items use vanilla's three-copy step");
        assert_ne!(
            copies[0].matrix, copies[1].matrix,
            "3-D copies after the first consume the seeded jitter stream"
        );

        let builtin = BuiltinTileItem::Chest { trapped: false };
        let mut tile_models = TileModels::new(false);
        let mut vertices = Vec::new();
        let texture =
            emit_dropped_builtin_item(&mut tile_models, &builtin, model, &item, 0.5, &mut vertices)
                .expect("normal chest item emits");
        assert_eq!(texture, "textures/entity/chest/normal.png");
        assert_eq!(
            vertices.len(),
            copies.len() * 3 * 6 * 6,
            "each dropped copy emits the complete three-part chest"
        );
        assert_ne!(
            vertices[0].pos,
            vertices[3 * 6 * 6].pos,
            "the second copy retains bob/spin plus its deterministic jitter"
        );
    }

    /// Every vanilla `builtin/entity` family must reach the same dropped helper.  Banner state is
    /// deliberately synthetic here: texture pixels are an upload concern, while descriptor/key
    /// selection and geometry reachability are fully CPU-verifiable.
    #[test]
    fn dropped_builtin_item_families_emit_their_descriptor_geometry() {
        let model =
            mc_assets::item_model_for_damage(mc_data::ItemId(mc_world::tileent::CHEST_ID), 0);
        let item = dropped::DroppedItem {
            stack_size: 1,
            gui_3d: true,
            ground_scale: model.transforms.ground.scale,
            ..dropped::DroppedItem::default()
        };
        let cases = [
            (
                BuiltinTileItem::Chest { trapped: false },
                "textures/entity/chest/normal.png",
            ),
            (
                BuiltinTileItem::Chest { trapped: true },
                "textures/entity/chest/trapped.png",
            ),
            (
                BuiltinTileItem::EnderChest,
                "textures/entity/chest/ender.png",
            ),
            (
                BuiltinTileItem::Banner {
                    state: mc_world::BannerState {
                        base_color: 14,
                        patterns: vec![("cr".to_owned(), 1)],
                        initialized: true,
                    },
                },
                "mc-render/banner/b14cr1",
            ),
            (
                BuiltinTileItem::Skull { skull_type: 2 },
                "textures/entity/zombie/zombie.png",
            ),
        ];
        for (builtin, expected_texture) in cases {
            let mut tile_models = TileModels::new(false);
            let mut vertices = Vec::new();
            let texture = emit_dropped_builtin_item(
                &mut tile_models,
                &builtin,
                model,
                &item,
                0.0,
                &mut vertices,
            )
            .expect("initialized builtin family emits");
            assert_eq!(texture, expected_texture);
            assert!(
                !vertices.is_empty(),
                "{builtin:?} must not be an empty sentinel"
            );
        }
    }

    #[test]
    fn dropped_player_skull_switches_to_the_shared_profile_texture_without_http() {
        let model =
            mc_assets::item_model_for_damage(mc_data::ItemId(mc_world::tileent::CHEST_ID), 0);
        let item = dropped::DroppedItem {
            stack_size: 1,
            gui_3d: true,
            ground_scale: model.transforms.ground.scale,
            ..dropped::DroppedItem::default()
        };
        let key = "mc-render/entity/skin/dropped-player-skull-test";
        let owner = mc_render::entity::tilepass::BuiltinSkullOwner::new(
            key,
            mc_render::entity::SkinModel::Slim,
            mc_render::entity::SkinModel::Classic,
        );
        let builtin = BuiltinTileItem::PlayerSkull {
            skull_type: 3,
            owner,
        };

        let mut fallback_models = TileModels::new(false);
        let mut fallback_vertices = Vec::new();
        let fallback_key = emit_dropped_builtin_item(
            &mut fallback_models,
            &builtin,
            model,
            &item,
            0.0,
            &mut fallback_vertices,
        )
        .expect("fallback player skull emits");
        assert_eq!(fallback_key, "textures/entity/steve.png");
        assert!(!fallback_vertices.is_empty());

        mc_render::entity::skin::publish_shared_skin(
            key,
            &mc_render::entity::SkinTexture {
                rgba: vec![255; 64 * 64 * 4].into(),
                layout: mc_render::entity::SkinLayout::Modern64x64,
                model: mc_render::entity::SkinModel::Slim,
            },
        );
        let mut ready_models = TileModels::new(false);
        let mut ready_vertices = Vec::new();
        let ready_key = emit_dropped_builtin_item(
            &mut ready_models,
            &builtin,
            model,
            &item,
            0.0,
            &mut ready_vertices,
        )
        .expect("cached player skull emits");
        assert_eq!(ready_key, key);
        assert_eq!(ready_vertices.len(), fallback_vertices.len());
    }

    /// Java's missing/null stone fallback is distinct from a present unregistered stack. The
    /// latter must not leak through `ItemId::props` as fabricated missing-cube geometry.
    #[test]
    fn dropped_item_getter_uses_stone_only_for_missing_or_null_watcher() {
        fn resolve(value: Option<MetadataValue>) -> Option<mc_protocol::ItemStack> {
            let mut world = World::new();
            let mut metadata = Metadata::new();
            if let Some(value) = value {
                metadata.set(DROPPED_ITEM_SLOT, value);
            }
            let mut tracker = crate::sync::entities::EntityTracker::new();
            tracker.spawn(
                &mut world,
                96,
                TrackedKind::Type(EntityType::Item),
                None,
                [0, 0, 0],
                0.0,
                0.0,
                mc_math::DVec3::ZERO,
                metadata,
            );
            dropped_item_stack(&world, tracker.get(96).expect("dropped item spawned"))
        }

        let stone = item_stack("stone", 0);
        assert_eq!(resolve(None), Some(stone.clone()));
        assert_eq!(
            resolve(Some(MetadataValue::ItemStack(None))),
            Some(stone.clone())
        );
        assert_eq!(
            resolve(Some(MetadataValue::ItemStack(Some(stone.clone())))),
            Some(stone)
        );

        for item_id in [0_i16, 8] {
            assert_eq!(
                resolve(Some(MetadataValue::ItemStack(Some(
                    mc_protocol::ItemStack {
                        item_id,
                        count: 1,
                        damage: 0,
                        nbt: None,
                    }
                )))),
                None,
                "a present unregistered id {item_id} must not become the missing stone fallback"
            );
        }
    }

    #[test]
    fn unregistered_item_frame_contents_emit_no_item_geometry() {
        let mut world = World::new();
        let mut tracker = crate::sync::entities::EntityTracker::new();
        tracker.spawn(
            &mut world,
            97,
            TrackedKind::Type(EntityType::ItemFrame),
            None,
            [0, 0, 0],
            0.0,
            0.0,
            mc_math::DVec3::ZERO,
            Metadata::new(),
        );
        let tracked = tracker.get(97).expect("item frame spawned");

        for raw_id in [8_u16, 60_000] {
            let stack = mc_protocol::ItemStack {
                item_id: raw_id as i16,
                count: 1,
                damage: 0,
                nbt: None,
            };
            let mut vertices = Vec::new();
            let mut glint = Vec::new();
            emit_item_frame_stack(
                &world,
                tracked,
                &stack,
                mc_render::entity::Mat4::IDENTITY,
                &mut vertices,
                &mut glint,
                0,
            );
            assert!(vertices.is_empty(), "unregistered item id {raw_id}");
            assert!(glint.is_empty(), "unregistered item id {raw_id}");
        }
    }

    /// Exercise the live object dispatcher, not just the reusable geometry helper: a watched
    /// chest stack must select `ObjectTexture::Builtin` and never fall through to the empty baked
    /// `builtin/entity` sentinel.
    #[test]
    fn dropped_object_dispatch_routes_watched_builtin_stack_to_tile_models() {
        let mut world = World::new();
        let mut metadata = Metadata::new();
        metadata.set(
            DROPPED_ITEM_SLOT,
            MetadataValue::ItemStack(Some(item_stack("chest", 0))),
        );
        let mut tracker = crate::sync::entities::EntityTracker::new();
        tracker.spawn(
            &mut world,
            97,
            TrackedKind::Type(EntityType::Item),
            None,
            [0, 0, 0],
            0.0,
            0.0,
            mc_math::DVec3::new(3.0, 64.0, 2.0),
            metadata,
        );
        let tracked = tracker.get(97).expect("dropped item spawned");
        let mut models = RenderManager::new();
        let mut tile_models = TileModels::new(false);
        let mut vertices = Vec::new();
        let mut glint = Vec::new();
        let mut painting = Vec::new();
        let texture = emit_object(
            &mut vertices,
            &mut models,
            &mut tile_models,
            &world,
            None,
            tracked,
            EntityType::Item,
            &Camera::default(),
            0.5,
            crate::app::appearance::ShadowLighting::default(),
            &mut glint,
            0,
            &mut painting,
            mc_render::terrain::MeshSettings::default(),
        )
        .expect("watched chest stack renders");

        assert_eq!(
            texture,
            ObjectTexture::Builtin {
                item: BuiltinTileItem::Chest { trapped: false },
                key: "textures/entity/chest/normal.png".to_owned(),
            }
        );
        assert_eq!(
            vertices.len(),
            3 * 6 * 6,
            "one chest stack is one tile model"
        );
        assert!(
            glint.is_empty(),
            "builtin models skip RenderItem's baked glint branch"
        );
    }

    /// Item frames use the same `builtin/entity` sentinel as dropped items, but their baked
    /// shell remains on the block atlas.  A single texture batch would bind the chest sheet over
    /// the shell (or the atlas over the chest); retaining the two ranges is the biting routing
    /// control for that Java two-bind sequence.
    #[test]
    fn item_frame_dispatch_splits_atlas_shell_from_builtin_content() {
        let mut world = World::new();
        let mut metadata = Metadata::new();
        metadata.set(
            ITEM_FRAME_ITEM_SLOT,
            MetadataValue::ItemStack(Some(item_stack("chest", 0))),
        );
        let mut tracker = crate::sync::entities::EntityTracker::new();
        tracker.spawn(
            &mut world,
            98,
            TrackedKind::Type(EntityType::ItemFrame),
            None,
            [0, 0, 0],
            0.0,
            0.0,
            mc_math::DVec3::new(3.0, 64.0, 2.0),
            metadata,
        );
        let tracked = tracker.get(98).expect("item frame spawned");
        let mut models = RenderManager::new();
        let mut tile_models = TileModels::new(false);
        let mut vertices = Vec::new();
        let mut glint = Vec::new();
        let mut painting = Vec::new();
        let texture = emit_object(
            &mut vertices,
            &mut models,
            &mut tile_models,
            &world,
            None,
            tracked,
            EntityType::ItemFrame,
            &Camera::default(),
            0.5,
            crate::app::appearance::ShadowLighting::default(),
            &mut glint,
            0,
            &mut painting,
            mc_render::terrain::MeshSettings::default(),
        )
        .expect("item frame shell renders");
        let ObjectTexture::ItemFrameBuiltin {
            item,
            key,
            shell_range,
            content_range,
        } = texture
        else {
            panic!("item frame chest must route through TileEntityItemStackRenderer");
        };
        assert_eq!(item, BuiltinTileItem::Chest { trapped: false });
        assert_eq!(key, "textures/entity/chest/normal.png");
        // The global baked registry is intentionally empty in a cold unit-test process.  The
        // live dispatcher must preserve that no-geometry result (rather than inventing a shell),
        // while an installed vanilla resource pack exposes all 22 quads of `item_frame.json`
        // (the backing panel's two faces, six for each horizontal bar, four for each upright).
        // The shell is emitted unculled: its `DroppedItem` pipeline applies `RenderItemFrame`'s
        // GL_BACK state on the GPU, by screen-space winding.
        assert!(shell_range.0 <= shell_range.1);
        assert!(content_range.0 < content_range.1);
        assert_eq!(shell_range.1, content_range.0);
        let shell_vertices = shell_range.1 - shell_range.0;
        assert!(
            shell_vertices == 0 || shell_vertices == 22 * 6,
            "frame model is empty before asset install or its 22 quads: {shell_vertices}"
        );
        // The builtin chest is emitted by `TileEntityItemStackRenderer` while the frame's
        // world back-face state is still active, culled on the CPU: of its three boxes' 18
        // quads only those whose plane faces the eye survive. GL's facing is the projected
        // winding, i.e. the eye's side of each plane (`entity::item`'s
        // `cpu_cull_matches_the_projected_winding_off_the_view_axis`); with this frame 64 blocks
        // above the camera that includes the chest's underside.
        let content = &vertices[content_range.0..content_range.1];
        assert_eq!(content.len() % 6, 0, "whole quads only");
        assert!(
            !content.is_empty() && content.len() < 18 * 6,
            "builtin chest content follows the frame's Java GL_BACK state: {}",
            content.len()
        );
        let view = Camera::default().view;
        for triangle in content.chunks_exact(3) {
            let p: [_; 3] = core::array::from_fn(|i| view.transform_point3(triangle[i].pos.into()));
            let winding = (p[1] - p[0]).cross(p[2] - p[0]);
            assert!(
                winding.dot(p[0]) < 0.0,
                "a retained chest triangle faces away: {p:?}"
            );
        }
        assert!(
            glint.is_empty(),
            "builtin item frames do not use baked glint"
        );
    }

    /// java: `EntityXPOrb.getTextureByXP` is fed the *spawn packet's* value, so the component
    /// has to survive spawn. This pins the component shape the dispatcher writes.
    #[test]
    fn the_spawn_components_are_plain_copyable_markers() {
        let d = ObjectData(70);
        assert_eq!(d, ObjectData(70));
        let x = XpValue(2477);
        assert_eq!(x.0, 2477);
        let h = Hanging {
            art: Art::by_title("Pigscene"),
            facing: Facing::North,
        };
        assert_eq!(h.art.title, "Pigscene");
        assert_eq!(h.facing, Facing::North);
    }

    // -- the spawn-packet plumbing ------------------------

    use mc_protocol::play::{s2c, S2CPlay};
    use mc_protocol::Angle;

    fn spawn(state: &mut ClientState, packet: S2CPlay) {
        let mut events = Vec::new();
        crate::sync::apply(state, packet, &mut events).expect("apply");
    }

    fn object(entity_id: i32, object_type: i8, data: i32) -> S2CPlay {
        S2CPlay::SpawnObject(s2c::SpawnObject {
            entity_id,
            object_type,
            x: 0,
            y: 0,
            z: 0,
            pitch: Angle::from_degrees(0.0),
            yaw: Angle::from_degrees(0.0),
            data,
            velocity: Some((8000, 8000, 8000)),
        })
    }

    /// Type 71's `data` chooses the `EntityHanging` wall anchor, while the packet yaw written
    /// after construction is the renderer's outer turn. Keep those two authorities distinct in
    /// the actual dispatch path, including a deliberately mismatched packet yaw.
    #[test]
    fn item_frame_spawn_packet_keeps_anchor_facing_and_packet_yaw_distinct() {
        let mut state = ClientState::default();
        let anchor = BlockPos::new(3, 64, -2);
        let packet = S2CPlay::SpawnObject(s2c::SpawnObject {
            entity_id: 71,
            object_type: 71,
            x: anchor.x * 32 + 1,
            y: anchor.y * 32 + 16,
            z: anchor.z * 32 + 16,
            pitch: Angle::from_degrees(0.0),
            yaw: Angle::from_degrees(90.0),
            data: 3,
            velocity: Some((8000, 8000, 8000)),
        });
        spawn(&mut state, packet);

        let tracked = state.entities.get(71).expect("item-frame generation");
        let facing = state
            .world
            .ecs
            .get::<&ItemFrameFacing>(tracked.entity)
            .expect("type 71 facing marker")
            .0;
        assert_eq!(facing, Facing::East);
        let expected = mc_math::DVec3::new(3.03125, 64.5, -1.5);
        assert_eq!(tracked.render_position(&state.world, 1.0), expected);

        let frame = item_frame_matrix(expected, &Camera::default(), facing, 90.0);
        assert!(frame
            .transform_point3(mc_math::Vec3::ZERO)
            .abs_diff_eq(mc_math::Vec3::new(3.5, 64.5, -1.5), 1e-5));
        assert!(
            frame
                .transform_vector3(mc_math::Vec3::Z)
                .abs_diff_eq(mc_math::Vec3::X, 1e-5),
            "the packet yaw controls the outer turn even when data says East"
        );
    }

    /// `data` used to be read for the minecart variant and then dropped. A falling block
    /// carries its `Block.getStateById` there and cannot be textured without it.
    #[test]
    fn a_falling_block_keeps_its_block_state_from_the_spawn_packet() {
        let mut state = ClientState::default();
        // Object type 70, `getStateById(stone)` = `id | meta << 12` = 1.
        spawn(&mut state, object(11, 70, 1));
        let tracked = state.entities.get(11).expect("tracked");
        let data = state
            .world
            .ecs
            .get::<&ObjectData>(tracked.entity)
            .expect("the falling block's ObjectData component");
        assert_eq!(data.0, 1);
        assert_eq!(
            mc_data::BlockState::from_java_state_id(data.0),
            mc_data::block_by_name("minecraft:stone")
                .expect("stone")
                .default_state()
        );
    }

    #[test]
    fn a_thrown_potion_keeps_typed_spawn_damage_for_model_and_liquid_tint() {
        let mut state = ClientState::default();
        spawn(&mut state, object(12, 73, 16_386));
        let tracked = state.entities.get(12).expect("tracked");
        let data = state
            .world
            .ecs
            .get::<&SplashPotionDamage>(tracked.entity)
            .expect("the potion's typed damage component");
        assert_eq!(data.0, 16_386);
        assert_eq!(
            thrown_item_stack(EntityType::ThrownPotion, Some(data.0))
                .expect("potion item")
                .1,
            16_384
        );
        assert_eq!(
            thrown_item_quad_tint(EntityType::ThrownPotion, Some(data.0), 0),
            mc_data::potion_liquid_color_from_damage(16_386)
        );
    }

    /// `handleSpawnObject` calls `packetIn.func_149002_g(0)` in nine of its branches
    /// *before* the `func_149009_m() > 0` velocity guard, so those nine spawn with zero motion no
    /// matter what `data` held. Evaluating the guard against the original `data` gave a falling
    /// block (whose `data` is a block-state id) the packet's velocity and made it drift.
    #[test]
    fn the_nine_zeroed_object_types_spawn_without_the_packet_velocity() {
        for (i, object_type) in [63i8, 64, 66, 70, 71, 73, 75, 77].iter().enumerate() {
            let mut state = ClientState::default();
            let id = 100 + i as i32;
            spawn(&mut state, object(id, *object_type, 4096));
            // Type 90 is owner-gated and has its own direct-construction test above.
            let tracked = state.entities.get(id).expect("tracked");
            let velocity = state
                .world
                .ecs
                .get::<&mc_world::entity::Velocity>(tracked.entity)
                .expect("velocity");
            assert_eq!(
                velocity.0,
                mc_math::DVec3::ZERO,
                "object type {object_type}"
            );
        }
        // A type that is *not* on the list still takes it: an arrow (60) with a nonzero shooter.
        let mut state = ClientState::default();
        spawn(&mut state, object(200, 60, 4096));
        let tracked = state.entities.get(200).expect("tracked");
        let velocity = state
            .world
            .ecs
            .get::<&mc_world::entity::Velocity>(tracked.entity)
            .expect("velocity");
        assert_ne!(velocity.0, mc_math::DVec3::ZERO);
    }

    /// The orb's XP value decides `getTextureByXP`'s cell and is on the wire exactly once.
    #[test]
    fn an_xp_orb_keeps_its_value_and_reaches_the_right_texture_cell() {
        let mut state = ClientState::default();
        spawn(
            &mut state,
            S2CPlay::SpawnExperienceOrb(s2c::SpawnExperienceOrb {
                entity_id: 5,
                x: 0,
                y: 0,
                z: 0,
                xp_value: 1237,
            }),
        );
        let tracked = state.entities.get(5).expect("tracked");
        let value = state
            .world
            .ecs
            .get::<&XpValue>(tracked.entity)
            .expect("the orb's XpValue component");
        assert_eq!(value.0, 1237);
        assert_eq!(objects::xp_orb_texture_index(value.0), 9);
    }

    /// The art name and the facing were both dropped, and the position was a bare block
    /// centre. `EntityHanging.updateBoundingBox` backs the entity `0.46875` off its wall and
    /// offsets even-sized art by half a block.
    #[test]
    fn a_painting_lands_off_its_wall_with_the_right_art() {
        let mut state = ClientState::default();
        spawn(
            &mut state,
            S2CPlay::SpawnPainting(s2c::SpawnPainting {
                entity_id: 9,
                title: "DonkeyKong".to_owned(),
                position: mc_math::BlockPos::new(10, 64, 20),
                facing: Facing::North,
            }),
        );
        let tracked = state.entities.get(9).expect("tracked");
        let hanging = state
            .world
            .ecs
            .get::<&Hanging>(tracked.entity)
            .expect("the painting's Hanging component");
        assert_eq!(hanging.art.title, "DonkeyKong");
        assert_eq!(hanging.facing, Facing::North);

        // 64 texels wide is a multiple of 32, so it slides half a block along rotateYCCW(north)
        // = west (-x); 48 tall is not, so there is no vertical offset. North's front offset is
        // z = -1, so the entity backs off to +z.
        let position = tracked.target_pos;
        assert!((position.x - (10.5 - 0.5)).abs() < 1e-6, "{position:?}");
        assert!((position.y - 64.5).abs() < 1e-6, "{position:?}");
        assert!(
            (position.z - (20.5 + 0.468_75)).abs() < 1e-6,
            "{position:?}"
        );
        // java: `rotationYaw = facingDirection.getHorizontalIndex() * 90` — north is index 2.
        assert!((tracked.target_yaw - 180.0).abs() < 1e-6);
    }

    #[test]
    fn live_painting_dispatch_preserves_each_cells_world_light_batch() {
        let mut state = ClientState::default();
        spawn(
            &mut state,
            S2CPlay::SpawnPainting(s2c::SpawnPainting {
                entity_id: 19,
                title: "Match".to_owned(),
                position: mc_math::BlockPos::new(10, 64, 20),
                facing: Facing::North,
            }),
        );

        let tracked = state.entities.get(19).expect("tracked painting");
        let position = tracked.render_position(&state.world, 0.5);
        let hanging = *state
            .world
            .ecs
            .get::<&Hanging>(tracked.entity)
            .expect("painting state");
        assert_eq!((hanging.art.size_x, hanging.art.size_y), (32, 32));

        let samples =
            [(-8.0, -8.0), (-8.0, 8.0), (8.0, -8.0), (8.0, 8.0)].map(|(cell_x, cell_y)| {
                let [x, y, z] = objects::painting_cell_light_position(
                    position.to_array(),
                    hanging.facing,
                    cell_x,
                    cell_y,
                );
                BlockPos::new(x, y, z)
            });
        let mut loaded_chunks = Vec::new();
        for sample in samples {
            let chunk = sample.chunk_pos();
            if !loaded_chunks.contains(&chunk) {
                state.world.load_empty_chunk(chunk);
                loaded_chunks.push(chunk);
            }
        }

        let mut expected = Vec::new();
        for (index, sample) in samples.into_iter().enumerate() {
            let block = u8::try_from(index + 1).unwrap();
            let sky = u8::try_from(15 - index).unwrap();
            state
                .world
                .set_light_for(mc_world::SkyBlock::Block, sample, block);
            state
                .world
                .set_light_for(mc_world::SkyBlock::Sky, sample, sky);
            expected.push((u32::from(block) << 4) | (u32::from(sky) << 20));
        }

        let tracked = state.entities.get(19).expect("tracked painting");
        let mut models = RenderManager::new();
        let mut tile_models = TileModels::new(false);
        let mut ordinary = Vec::new();
        let mut glint = Vec::new();
        let mut painting = Vec::new();
        let texture = emit_object(
            &mut ordinary,
            &mut models,
            &mut tile_models,
            &state.world,
            None,
            tracked,
            EntityType::Painting,
            &Camera::default(),
            0.5,
            crate::app::appearance::ShadowLighting::default(),
            &mut glint,
            0,
            &mut painting,
            mc_render::terrain::MeshSettings::default(),
        );

        assert_eq!(
            texture,
            Some(ObjectTexture::Resource(objects::PAINTING_TEXTURE))
        );
        assert!(ordinary.is_empty());
        assert!(glint.is_empty());
        assert_eq!(painting.len(), 4 * 36);
        let mut batches = Vec::new();
        for_each_painting_cell(&painting, |combined_light, vertices| {
            batches.push((combined_light, vertices.len()));
        });
        assert_eq!(
            batches,
            expected
                .into_iter()
                .map(|combined_light| (combined_light, 36))
                .collect::<Vec<_>>()
        );

        let production = include_str!("entrender.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("production object pass");
        assert!(production.contains("push_painting_geometry("));
        assert!(production.contains("phase.push_blended_lit("));
    }

    /// java: `RenderPainting.setLightmap` calls `World.getCombinedLight` without the loaded-block
    /// guard used by `Entity.getBrightnessForRender`. `World.getLightFromNeighborsFor` therefore
    /// returns the sky channel's default 15 for an unloaded overworld cell and zero in a no-sky
    /// dimension. A wide painting crossing a chunk boundary can reach this path normally.
    #[test]
    fn painting_unloaded_cell_light_uses_the_dimension_sky_default() {
        let sample = BlockPos::new(16, 64, 0);
        let mut world = mc_world::World::new();
        assert!(!world.is_chunk_loaded(sample.chunk_pos()));
        assert_eq!(painting_combined_light_at(&world, sample), 15 << 20);

        world.set_has_sky_light(false);
        assert_eq!(painting_combined_light_at(&world, sample), 0);
    }

    /// java: `EntityPainting`'s titled constructor first runs the no-title wall candidate scan.
    /// With this empty client world no candidate is valid, so that scan leaves the final enum
    /// value (`DONKEY_KONG`) in place when the title does not match; the entity still renders.
    #[test]
    fn an_unknown_painting_title_still_spawns() {
        let mut state = ClientState::default();
        spawn(
            &mut state,
            S2CPlay::SpawnPainting(s2c::SpawnPainting {
                entity_id: 3,
                title: "NotAPainting".to_owned(),
                position: mc_math::BlockPos::new(0, 0, 0),
                facing: Facing::South,
            }),
        );
        let tracked = state.entities.get(3).expect("tracked");
        let hanging = state
            .world
            .ecs
            .get::<&Hanging>(tracked.entity)
            .expect("Hanging");
        assert_eq!(hanging.art.title, "DonkeyKong");
    }

    /// java: an unknown title is not a synonym for `DONKEY_KONG` when the client world contains
    /// a valid wall. With one support block, only the seven 16x16 arts fit, so the constructor's
    /// random candidate must be small even though the packet title is not in `EnumArt`.
    #[test]
    fn an_unknown_painting_title_chooses_from_the_wall_valid_candidates() {
        let mut state = ClientState::default();
        let anchor = mc_math::BlockPos::new(4, 64, 4);
        state.world.load_empty_chunk(anchor.chunk_pos());
        state.world.set_block(
            anchor.south(),
            mc_data::block_by_name("stone")
                .expect("vanilla stone is registered")
                .default_state(),
        );
        spawn(
            &mut state,
            S2CPlay::SpawnPainting(s2c::SpawnPainting {
                entity_id: 31,
                title: "NotAPainting".to_owned(),
                position: anchor,
                facing: Facing::North,
            }),
        );
        let tracked = state.entities.get(31).expect("tracked");
        let hanging = state
            .world
            .ecs
            .get::<&Hanging>(tracked.entity)
            .expect("Hanging");
        assert_eq!(
            (hanging.art.size_x, hanging.art.size_y),
            (16, 16),
            "unknown title must use the valid one-block-wall candidate set"
        );
        assert_ne!(hanging.art.title, "DonkeyKong");
    }

    /// `EntityFireball.onUpdate` sets its client-local fire timer every tick, so both concrete
    /// fireball registrations get the shared Render fire stack.  The closely related skull is
    /// the important negative control: its explicit `isBurning() == false` override means it
    /// must retain just its skull billboard.
    #[test]
    fn only_large_and_small_fireballs_receive_the_two_layer_fire_stack() {
        use EntityType as E;

        assert_eq!(fireball_overlay_size(E::Fireball), Some((1.0, 1.0)));
        assert_eq!(
            fireball_overlay_size(E::SmallFireball),
            Some((0.3125, 0.3125))
        );
        for control in [E::WitherSkull, E::Arrow, E::Snowball, E::PrimedTnt] {
            assert_eq!(
                fireball_overlay_size(control),
                None,
                "{control:?} must not inherit EntityFireball's flame stack"
            );
        }

        for kind in [E::Fireball, E::SmallFireball] {
            let (width, height) = fireball_overlay_size(kind).unwrap();
            let mut verts = Vec::new();
            crate::app::appearance::emit_fire_overlay(
                mc_math::Vec3::new(0.0, 0.0, 2.0),
                width,
                height,
                90.0,
                |i| {
                    if i == 0 {
                        [0.0, 0.0, 0.5, 1.0]
                    } else {
                        [0.5, 0.0, 1.0, 1.0]
                    }
                },
                &mut verts,
            );
            // height / (width * 1.4) is 0.714... for both fireball hitboxes, so the Java
            // 0.45-step loop emits two alternating layers (six vertices each).  A yaw of 90
            // also exercises that these use Render's camera-facing transform, not the fireball
            // sprite's 180 - viewYaw transform.
            assert_eq!(verts.len(), 12, "{kind:?}");
            assert_ne!(
                verts[0].pos, verts[6].pos,
                "{kind:?} alternates and offsets layers"
            );
        }
    }

    /// java: `Render.doRenderShadowAndFire` draws `renderEntityOnFire` for any entity whose
    /// `canRenderOnFire()` holds, so a Flame arrow (its watched flag bit 0) gets the same stack
    /// as a burning mob, sized by its own 0.5 x 0.5 box. `JAVA` is the real 1.8.9 draw captured
    /// by `javaref/ArrowRenderRef` (case `burning`: the eye looking south, the arrow at
    /// (0, -0.125, 1.625), 1920x1080, FOV 90), as window x, window y (down) and eye depth.
    #[test]
    fn a_burning_arrow_gets_vanillas_fire_stack_at_its_own_size() {
        use EntityType as E;

        const JAVA: [[f64; 3]; 8] = [
            [826.4310, 587.7032, 1.415000],
            [1093.5689, 587.7032, 1.415000],
            [1093.5689, 213.7102, 1.415000],
            [826.4310, 213.7102, 1.415000],
            [841.5459, 468.5515, 1.436000],
            [1078.4540, 468.5515, 1.436000],
            [1078.4540, 100.0278, 1.436000],
            [841.5459, 100.0278, 1.436000],
        ];
        for kind in [
            Some(E::Arrow),
            Some(E::Item),
            Some(E::Boat),
            Some(E::PrimedTnt),
            None,
        ] {
            assert!(object_can_render_on_fire(kind), "{kind:?}");
        }
        // `EntityWitherSkull.isBurning()` and `EntityFallingBlock.canRenderOnFire()`.
        for kind in [Some(E::WitherSkull), Some(E::FallingSand)] {
            assert!(!object_can_render_on_fire(kind), "{kind:?}");
        }
        // The fireballs' client-local fire keeps its own, unconditional route.
        assert_eq!(fireball_overlay_size(E::Arrow), None);

        let info = E::Arrow.info();
        assert_eq!((info.width, info.height), (0.5, 0.5));
        let mut camera = Camera {
            fov: 90.0,
            ..Camera::default()
        };
        camera.set_viewport(1920, 1080);
        camera.set_view_distance_chunks(8);
        camera.update();
        let mut verts = Vec::new();
        crate::app::appearance::emit_fire_overlay(
            mc_math::Vec3::new(0.0, -0.125, 1.625),
            info.width,
            info.height,
            camera.yaw,
            |_| [0.0, 0.0, 1.0, 1.0],
            &mut verts,
        );
        assert_eq!(verts.len(), 12, "Java drew two layers");
        for (layer, java) in JAVA.chunks_exact(4).enumerate() {
            for (corner, &index) in [0usize, 1, 2, 5].iter().enumerate() {
                let p = mc_math::Vec3::from(verts[layer * 6 + index].pos);
                let ndc = camera.view_proj().project_point3(p);
                let x = (f64::from(ndc.x) * 0.5 + 0.5) * 1920.0;
                let y = (1.0 - (f64::from(ndc.y) * 0.5 + 0.5)) * 1080.0;
                let depth = -f64::from(camera.view.transform_point3(p).z);
                let [jx, jy, jz] = java[corner];
                assert!(
                    (x - jx).hypot(y - jy) < 0.01 && (depth - jz).abs() < 1e-5,
                    "layer {layer} corner {corner}: ({x:.4}, {y:.4}, {depth:.6}) vs Java \
                     ({jx}, {jy}, {jz})"
                );
            }
        }
    }

    /// `RenderArrow` keeps the world pass's back-face culling: every arrow quad has a coplanar
    /// twin wound the other way, and two-sided they z-fight. Its fixed-function light (normal
    /// `0.05625` long) is folded into the vertex colour by the live dispatch.
    #[test]
    fn a_live_arrow_is_drawn_culled_with_its_fixed_function_light() {
        assert_eq!(
            object_body_blend(Some(EntityType::Arrow)),
            objects::ARROW_BLEND
        );
        assert_eq!(
            object_body_blend(Some(EntityType::Snowball)),
            LayerBlend::Opaque
        );
        assert_eq!(
            object_body_blend(Some(EntityType::Item)),
            LayerBlend::DroppedItem
        );

        let mut world = World::new();
        let mut tracker = crate::sync::entities::EntityTracker::new();
        tracker.spawn(
            &mut world,
            92,
            TrackedKind::Type(EntityType::Arrow),
            None,
            [0, 0, 64],
            0.0,
            0.0,
            mc_math::DVec3::ZERO,
            Metadata::new(),
        );
        let tracked = tracker.get(92).expect("spawned arrow");
        let camera = Camera::default();
        let mut models = RenderManager::new();
        let mut tile_models = TileModels::new(false);
        let (mut verts, mut glint, mut painting) = (Vec::new(), Vec::new(), Vec::new());
        let texture = emit_object(
            &mut verts,
            &mut models,
            &mut tile_models,
            &world,
            None,
            tracked,
            EntityType::Arrow,
            &camera,
            0.0,
            crate::app::appearance::ShadowLighting::default(),
            &mut glint,
            0,
            &mut painting,
            mc_render::terrain::MeshSettings::default(),
        );
        assert_eq!(
            texture,
            Some(ObjectTexture::Resource(objects::ARROW_TEXTURE))
        );
        assert_eq!(verts.len(), 36, "six quads");
        for vertex in &verts {
            let light = objects::arrow_light(mc_math::Vec3::from(vertex.normal));
            assert_eq!(vertex.color, [light, light, light, 1.0]);
            assert!((0.4..0.455).contains(&light), "{light}");
        }
    }

    /// java: with F3+B on, `LayerArrow`'s `renderEntityWithPosYaw` boxes every stuck arrow.
    /// `JAVA` is the real draw of `javaref/ArrowRenderRef`'s `player_hitboxes` case (Steve,
    /// entity id 4321, three arrows, 37 ticks + 0.25, facing the eye 2.5 blocks away, FOV 90 at
    /// 1920x1080) as window x, window y (down) and eye depth. Per arrow: the bottom strip, the
    /// top strip, the four verticals, then the two-block look ray.
    #[test]
    fn stuck_arrow_hitboxes_match_vanillas_debug_draws() {
        #[rustfmt::skip]
        const JAVA: [[f64; 3]; 60] = [
            [901.8001, 532.5635, 2.040606], [1025.8441, 532.5635, 2.040606],
            [1013.5444, 533.9527, 2.509356], [912.6719, 533.9527, 2.509356],
            [901.8001, 532.5635, 2.040606], [901.8001, 656.6076, 2.040606],
            [1025.8441, 656.6076, 2.040606], [1013.5444, 634.8252, 2.509356],
            [912.6719, 634.8252, 2.509356], [901.8001, 656.6076, 2.040606],
            [901.8001, 532.5635, 2.040606], [901.8001, 656.6076, 2.040606],
            [1025.8441, 532.5635, 2.040606], [1025.8441, 656.6076, 2.040606],
            [1013.5444, 533.9527, 2.509356], [1013.5444, 634.8252, 2.509356],
            [912.6719, 533.9527, 2.509356], [912.6719, 634.8252, 2.509356],
            [963.4283, 533.3297, 2.274981], [977.5315, 530.7671, 4.145634],
            [846.1848, 706.2684, 2.383833], [952.3688, 706.3859, 2.383833],
            [953.6252, 676.3846, 2.852371], [864.8833, 676.2864, 2.852371],
            [846.1848, 706.2684, 2.383833], [846.7375, 810.8019, 2.397938],
            [952.2968, 810.9188, 2.397938], [953.5589, 763.9787, 2.866476],
            [865.2536, 763.8810, 2.866476], [846.7375, 810.8019, 2.397938],
            [846.1848, 706.2684, 2.383833], [846.7375, 810.8019, 2.397938],
            [952.3688, 706.3859, 2.383833], [952.2968, 810.9188, 2.397938],
            [953.6252, 676.3846, 2.852371], [953.5589, 763.9787, 2.866476],
            [864.8833, 676.2864, 2.852371], [865.2536, 763.8810, 2.866476],
            [904.7119, 689.9895, 2.618102], [1299.9400, 1165.6066, 1.281360],
            [1007.0566, 595.3373, 2.289387], [1117.6211, 595.3373, 2.289387],
            [1090.8331, 585.9327, 2.758137], [999.0592, 585.9327, 2.758137],
            [1007.0566, 595.3373, 2.289387], [1007.0566, 705.9019, 2.289387],
            [1117.6211, 705.9019, 2.289387], [1090.8331, 677.7066, 2.758137],
            [999.0592, 677.7066, 2.758137], [1007.0566, 705.9019, 2.289387],
            [1007.0566, 595.3373, 2.289387], [1007.0566, 705.9019, 2.289387],
            [1117.6211, 595.3373, 2.289387], [1117.6211, 705.9019, 2.289387],
            [1090.8331, 585.9327, 2.758137], [1090.8331, 677.7066, 2.758137],
            [999.0592, 585.9327, 2.758137], [999.0592, 677.7066, 2.758137],
            [1052.8349, 590.1983, 2.523762], [1338.4291, 235.6320, 2.203305],
        ];
        let mut camera = Camera {
            fov: 90.0,
            ..Camera::default()
        };
        camera.set_viewport(1920, 1080);
        camera.set_view_distance_chunks(8);
        camera.update();
        // `UUID(0, 0)` gave the Java player Steve's classic arms.
        let renderer =
            mc_render::entity::renderer_for_player(mc_render::entity::SkinModel::Classic);
        let mut models = RenderManager::new();
        models.set_player_model_parts(renderer.model, 0x7f);
        let params = AnimParams {
            age_in_ticks: 37.25,
            partial_ticks: 0.25,
            entity: mc_render::entity::EntityAnimState {
                entity_id: 4321,
                arrow_count: 3,
                ticks_existed: 37,
                ..mc_render::entity::EntityAnimState::default()
            },
            ..AnimParams::still()
        };
        models.pose_with(renderer.model, &params);
        let base = mc_render::entity::entity_matrix(&mc_render::entity::render::EntityTransform {
            rel: mc_math::Vec3::new(0.0, -1.625, 2.5),
            body_yaw: 180.0,
            scale: renderer.scale,
            ..mc_render::entity::render::EntityTransform::default()
        });
        let mut boxes = Vec::new();
        stuck_arrow_debug_vertices(&mut models, renderer.model, &params, base, &mut boxes);
        assert_eq!(
            boxes.len(),
            JAVA.len(),
            "twenty debug vertices per stuck arrow"
        );
        for (i, (vertex, java)) in boxes.iter().zip(JAVA).enumerate() {
            let ndc = camera.view_proj().project_point3(vertex.pos);
            let x = (f64::from(ndc.x) * 0.5 + 0.5) * 1920.0;
            let y = (1.0 - (f64::from(ndc.y) * 0.5 + 0.5)) * 1080.0;
            let depth = -f64::from(camera.view.transform_point3(vertex.pos).z);
            // The look ray's far end: Java adds `getLook() * 2` as is, and the lookup-table
            // vector is up to ~1e-4 off unit length, while the shared `debug_entity_vertices`
            // normalises it (for every entity's ray). That moves the end by ~0.05 px here.
            let (pixels, depth_slack) = if i % 20 == 19 {
                (0.1, 1e-4)
            } else {
                (0.01, 1e-5)
            };
            assert!(
                (x - java[0]).hypot(y - java[1]) < pixels && (depth - java[2]).abs() < depth_slack,
                "vertex {i}: ({x:.4}, {y:.4}, {depth:.6}) vs Java {java:?}"
            );
        }
        for arrow in boxes.chunks_exact(20) {
            assert!(arrow[..18].iter().all(|v| v.color == [1.0, 1.0, 1.0, 1.0]));
            assert!(arrow[18..].iter().all(|v| v.color == [0.0, 0.0, 1.0, 1.0]));
        }
    }
}

#[cfg(test)]
mod nw_f1_object_prefilter_tests {
    use super::*;

    /// The `E::` kinds named in `head`'s body up to `end`; with `indent`, only on lines at exactly
    /// that many spaces (arm heads, not nested matches).
    fn arm_kinds(head: &str, end: &str, indent: Option<usize>) -> Vec<String> {
        let source = include_str!("entrender.rs");
        let body = source.split_once(head).expect("the function").1;
        let body = body.split_once(end).expect("the end of the match").0;
        let mut kinds: Vec<String> = Vec::new();
        for line in body.lines() {
            let rest = match indent {
                Some(indent) => match line.strip_prefix(&" ".repeat(indent)) {
                    Some(rest) if !rest.starts_with(' ') => rest,
                    _ => continue,
                },
                None => line,
            };
            for part in rest.split("E::").skip(1) {
                let name: String = part
                    .chars()
                    .take_while(char::is_ascii_alphanumeric)
                    .collect();
                if !name.is_empty() {
                    kinds.push(name);
                }
            }
        }
        kinds.sort();
        kinds.dedup();
        kinds
    }

    #[test]
    fn object_pass_kinds_are_exactly_emit_objects_arms() {
        // Arm heads in `emit_object`'s `match kind` sit at eight spaces, up to its fallthrough.
        let arms = arm_kinds("\nfn emit_object(", "\n        _ => None,", Some(8));
        assert_eq!(arms.len(), 17, "{arms:?}");
        // The prefilter's `matches!` names nothing else.
        let accepted = arm_kinds("\nfn object_pass_draws(", "\n    )\n}", None);
        assert_eq!(accepted, arms);
        for name in &arms {
            let kind = mc_data::entity_by_name(name)
                .unwrap_or_else(|| panic!("{name} is not an entity name"));
            assert_eq!(&format!("{kind:?}"), name);
            assert!(object_pass_draws(kind), "{name}");
        }
    }

    #[test]
    fn players_and_mobs_are_skipped_but_objects_are_not() {
        assert!(!object_pass_draws(EntityType::Zombie));
        assert!(!object_pass_draws(EntityType::ArmorStand));
        assert!(object_pass_draws(EntityType::Item));
        assert!(object_pass_draws(EntityType::Arrow));
        // Players have no `EntityType` at all; the prefilter skips `kind == None` unless the
        // entity is the fishing hook, whose arm sits outside `emit_object`.
        let source = include_str!("entrender.rs");
        assert!(source.contains(
            "&& tracked.kind != TrackedKind::FishingBobber\n            && !kind.is_some_and(object_pass_draws)"
        ));
    }
}
