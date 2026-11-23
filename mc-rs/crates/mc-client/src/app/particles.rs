//! The client's particle half: the atlas installs `ParticlePhase` needs, the `IconLookup` that
//! resolves block/item crack textures, and the `ClientEvent` handlers that spawn into
//! `ParticleSystem`.
//!
//! java: `RenderGlobal.spawnParticle` and `EffectRenderer`'s input side.
//!
//! Split out of [`super::driver`] so this seam has a file of its own.
//!
//! # Why the entry point is a free function, not an `impl Client` method
//!
//! [`apply_particle_packet`] needs `&mut Client::particles` **and** `&Client::session`'s world at
//! once. Rust splits borrows across struct *fields*, not across `&self` methods, so a
//! `fn(&mut self)` here could not hold both (`Client::session` is private to [`super::driver`],
//! so even `Client::session()` — a `&self` method — would conflict with `&mut self.particles`).
//! Taking the two pieces as arguments also makes the whole spawn path unit-testable against a
//! bare [`World`], with no `Session` and no socket.
//!
//! # What this closes
//!
//! `mc_render::particle::packet::apply` was a complete,
//! tested `S2APacketParticles` applicator with **no caller**, `ClientEvent::Particles` had **no
//! reader**, and [`mc_render::particle::IconLookup`] had **no implementor** — so even a wired-up
//! `apply` would have dropped every `BLOCK_CRACK`/`BLOCK_DUST`/`ITEM_CRACK`. All three are here.

use mc_assets::model::{BakedModel, BakedQuad};
use mc_assets::sprite::UvRect;
use mc_data::block::ActualBlockState;
use mc_data::{BlockId, BlockState, ItemId};
use mc_math::{DVec3, Facing, Vec3};
use mc_protocol::play::s2c;
use mc_render::particle::packet::{self, S2APacketParticles};
use mc_render::particle::{IconLookup, ParticleDensity, ParticleIcon, ParticleSystem};
use mc_world::World;

use crate::app::driver::Client;

// -------------------------------------------------------------------------------------------
// IconLookup
// -------------------------------------------------------------------------------------------

/// Resolves a block state or item stack to its particle sprite's rect in the stitched block
/// atlas — the atlas `ParticlePhase::install_block_atlas` uploaded, since baked quads already
/// carry atlas-space UVs.
///
/// java: `EntityDiggingFX`'s ctor calls `blockModelShapes.getTexture(state)` and
/// `EntityBreakingFX`'s calls `itemModelMesher.getParticleIcon(item, meta)`; both end at some
/// baked model's `getParticleTexture()`.
///
/// A unit struct on purpose: S09 publishes one bake process-globally
/// ([`mc_assets::model_for`]/[`mc_assets::item_model_for`]) and `RenderStack::install` uploads
/// *that* bake's atlas, so there is nothing per-instance to carry. Before a bake is installed —
/// during a deferred asset load — the global lookups return the empty model and every method
/// here answers `None`, which [`packet::apply`] treats as "drop this particle", matching
/// vanilla's own `catch (Throwable)` around a factory that may legitimately return `null`.
#[derive(Debug, Clone, Copy, Default)]
pub struct AtlasIcons;

/// UV bounding box of one baked quad's four corners, as a [`ParticleIcon`].
fn quad_icon(quad: &BakedQuad) -> ParticleIcon {
    let mut min_u = f32::INFINITY;
    let mut max_u = f32::NEG_INFINITY;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for vertex in &quad.verts {
        min_u = min_u.min(vertex.uv[0]);
        max_u = max_u.max(vertex.uv[0]);
        min_v = min_v.min(vertex.uv[1]);
        max_v = max_v.max(vertex.uv[1]);
    }
    ParticleIcon {
        min_u,
        max_u,
        min_v,
        max_v,
    }
}

fn uv_icon(uv: UvRect) -> ParticleIcon {
    ParticleIcon {
        min_u: uv.min_u,
        max_u: uv.max_u,
        min_v: uv.min_v,
        max_v: uv.max_v,
    }
}

/// The quad whose UVs stand in for vanilla's resolved `"particle"` texture.
///
/// java: `BlockModelShapes.getTexture(state)` is `getModelForState(state).getParticleTexture()`,
/// i.e. the model JSON's `"particle"` entry resolved up the parent chain.
///
/// [`BakedModel::particle_uv`] is now authoritative. This compatibility fallback remains for
/// programmatically constructed models that leave the additive field absent, preferring a
/// **side** face because vanilla's workhorse parents resolve `"particle"` to exactly that:
///
/// | model | `"particle"` | side face |
/// |---|---|---|
/// | `block/cube_all` (stone, dirt, planks, wool, ...) | `#all` | `#all` ✓ |
/// | `block/cube_bottom_top` (sandstone, TNT, ...) | `#side` | `#side` ✓ |
/// | `block/cube_column` (logs, quartz pillar, ...) | `#side` | `#side` ✓ |
/// | `block/half_slab` | `#side` | `#side` ✓ (but see below) |
/// | `block/cross` (flowers, saplings) | `#cross` | `#cross` ✓ |
/// | `block/grass_normal` | `blocks/dirt` | `grass_side` ✗ |
///
/// Those fallback guesses remain intentionally approximate; production bakes retain the exact
/// particle UV even when it differs from every face.
fn particle_quad(model: &BakedModel) -> Option<&BakedQuad> {
    // Sides first (see the table above), then the caps, then anything uncullfaced — the order a
    // non-cube model (`cross`, torches, fences) actually populates.
    for face in [
        Facing::North,
        Facing::South,
        Facing::West,
        Facing::East,
        Facing::Down,
        Facing::Up,
    ] {
        if let Some(quad) = model.face_quads[face.index() as usize].first() {
            return Some(quad);
        }
    }
    model.general_quads.first()
}

/// Resolve the particle sprite carried by a model.
///
/// A model produced by `mc_assets::bake_all` always carries the result of Java's
/// `ModelBlock.resolveTextureName("particle")` in [`BakedModel::particle_uv`].  That field is
/// authoritative even when the particle texture is not used by any face (and even when it is the
/// stitched missing-texture sprite).  In particular, looking at a face first would be wrong for
/// a parent-chain model whose inherited `particle` entry is different from its visible faces.
///
/// `BakedModel` is also used as a small programmatic fixture type by client code.  Such models
/// predate/skip the bake metadata and leave `particle_uv` absent, so retain the geometry guess as
/// a compatibility escape hatch for that case only.  Keeping the escape hatch behind the
/// `None` branch prevents it from silently overriding a real parent-chain resolution.
fn synthetic_particle_icon(model: &BakedModel) -> Option<ParticleIcon> {
    // Never infer from geometry when bake metadata exists.  This guard is intentionally kept in
    // the fallback helper as well as at its call site: it makes the synthetic-only boundary
    // explicit if another caller is added later.
    if model.particle_uv.is_some() {
        return None;
    }
    particle_quad(model).map(quad_icon)
}

fn model_particle_icon(model: &BakedModel) -> Option<ParticleIcon> {
    model
        .particle_uv
        .map(uv_icon)
        .or_else(|| synthetic_particle_icon(model))
}

/// The atlas rectangle `BlockModelShapes.getTexture(state)` exposes to first-person block
/// overlays. This intentionally shares the digging-particle resolver: both Java callers ask the
/// baked model for its particle texture, including the same built-in-renderer fallbacks.
pub(crate) fn block_particle_uv(state: BlockState) -> Option<UvRect> {
    let icon = AtlasIcons.block_icon(state)?;
    Some(UvRect {
        min_u: icon.min_u,
        max_u: icon.max_u,
        min_v: icon.min_v,
        max_v: icon.max_v,
    })
}

/// `BlockModelShapes.getTexture`'s explicit fallback table for blocks registered as built-in
/// renderers. Their model lookup intentionally resolves to the missing model, but break/dust
/// particles still use these named atlas sprites.
///
/// The three registered built-ins the table does not name (air,
/// `piston_extension`, `end_portal`) fall through to the missing model's
/// `getParticleTexture()`: the `missingno` checkerboard. So a `BLOCK_CRACK` packet for any of
/// them, or a 2001 break of the last two, throws checkerboard debris in vanilla (`BLOCK_DUST`,
/// hit and running particles skip render type -1 before they reach this). mc-rs dropped those
/// particles before this change.
#[must_use]
fn builtin_block_particle_texture(block: BlockId) -> Option<&'static str> {
    match block.0 {
        // chest, standing/wall sign, trapped chest, standing/wall banner
        54 | 63 | 68 | 146 | 176 | 177 => Some("textures/blocks/planks_oak"),
        130 => Some("textures/blocks/obsidian"), // ender chest
        10 | 11 => Some("textures/blocks/lava_still"),
        8 | 9 => Some("textures/blocks/water_still"),
        144 => Some("textures/blocks/soul_sand"), // skull
        166 => Some("textures/items/barrier"),
        // air, piston_extension, end_portal: the missing model's particle sprite
        0 | 36 | 119 => Some("missingno"),
        _ => None,
    }
}

#[must_use]
fn named_sprite_icon(path: &'static str) -> Option<ParticleIcon> {
    // Preserve this module's pre-bake contract: `model_for` returns an empty model before assets
    // are installed, and a named fallback must not turn that temporary state into a full-atlas
    // placeholder particle.
    if mc_assets::model::installed_generation() == 0 {
        return None;
    }
    let uv = mc_assets::model::sprite(&mc_assets::ResourceLocation::minecraft(path)).uv;
    Some(ParticleIcon {
        min_u: uv.min_u,
        max_u: uv.max_u,
        min_v: uv.min_v,
        max_v: uv.max_v,
    })
}

/// Selects the block-state metadata used by an `ITEM_CRACK` block item.
///
/// `ItemModelMesher.getParticleIcon(item, meta)` constructs an `ItemStack`, whose constructor
/// clamps negative damage to zero, and then looks up the **unmodified** metadata in the item
/// model table.  Block-item state metadata has only the four-bit range represented by
/// [`BlockState`]; silently masking a larger value aliases (for example) metadata 16 to the
/// metadata-zero model instead of taking the missing-model path.  Keep the check here, before
/// `BlockState::new`'s storage mask, so malformed server arguments cannot select another variant.
fn block_item_particle_state(block: BlockId, damage: i32) -> Option<BlockState> {
    let metadata = u8::try_from(damage.max(0)).ok()?;
    if metadata > 0x0F {
        return None;
    }
    Some(BlockState::new(block, metadata))
}

fn actual_block_model(state: &ActualBlockState) -> &'static BakedModel {
    // java: `EntityDiggingFX` -> `BlockModelShapes.getTexture` -> `getModelForState` uses the
    // base model. Unlike ordinary terrain, destroy particles do not choose a position-weighted
    // alternative.
    mc_assets::model::model_for_actual(state)
}

/// World-aware icon lookup used only by `EffectRenderer.addBlockDestroyEffects`.
/// Packet-driven `BLOCK_CRACK`/`BLOCK_DUST` deliberately keep using [`AtlasIcons`]: Java's
/// particle factory receives a state id but no world/position from which to derive actual state.
struct ActualAtlasIcons<'a> {
    state: &'a ActualBlockState,
}

impl IconLookup for ActualAtlasIcons<'_> {
    fn block_icon(&self, state: BlockState) -> Option<ParticleIcon> {
        if state == self.state.stored() {
            let model = actual_block_model(self.state);
            model_particle_icon(model)
        } else {
            AtlasIcons.block_icon(state)
        }
    }

    fn item_icon(&self, item_id: i32, damage: i32) -> Option<ParticleIcon> {
        AtlasIcons.item_icon(item_id, damage)
    }
}

impl IconLookup for AtlasIcons {
    fn block_icon(&self, state: BlockState) -> Option<ParticleIcon> {
        let model = mc_assets::model_for(state);
        model_particle_icon(model)
            .or_else(|| builtin_block_particle_texture(state.block()).and_then(named_sprite_icon))
    }

    fn item_icon(&self, item_id: i32, damage: i32) -> Option<ParticleIcon> {
        let id = ItemId(u16::try_from(item_id).ok()?);
        let props = mc_data::item_props(id);
        if !props.registered {
            return None;
        }
        // java: `ItemModelMesher.getItemModel` hands a block item straight to the *block* model
        // (`ModelBakery` registers `ItemBlock`s against their blockstate variants), so its
        // particle texture is the block's. Routing through `block_icon` also lets the stack's
        // damage select the right variant — wool colour, log type, stone brick type — which
        // `item_model_for` cannot, since it only bakes each item's subtype-0 appearance.
        if let Some(block) = props.block {
            let state = block_item_particle_state(block, damage)?;
            if let Some(icon) = self.block_icon(state) {
                return Some(icon);
            }
            // …except when the block has no model to borrow. `Blocks.barrier` is the case that
            // matters (`Barrier.Factory` asks for exactly this icon): its `getRenderType()` is
            // `-1`, so it bakes to nothing, while `models/item/barrier.json` is an ordinary
            // `builtin/generated` icon that `RenderItem.registerBlock(Blocks.barrier, "barrier")`
            // binds. Falling through to the item bake rather than returning `None` is what makes
            // the `BARRIER` particle visible at all.
        }
        // `builtin/generated` items: vanilla's `"particle"` for those is `layer0`, and
        // `expand_builtin_generated` stamps each layer's index into `tint_index`, so layer 0's
        // quads are exactly the ones tagged `Some(0)`. Taking their union rather than one quad
        // matters here: the flat icon is a core plane *plus* one extruded rim quad per pixel
        // span, and only the union spans the whole sprite.
        let damage = i16::try_from(damage).ok()?;
        let model = mc_assets::item_model_for_damage(id, damage);
        if let Some(uv) = model.particle_uv {
            return Some(uv_icon(uv));
        }
        let mut icon: Option<ParticleIcon> = None;
        for quad in &model.general_quads {
            if quad.tint_index != Some(0) {
                continue;
            }
            let rect = quad_icon(quad);
            icon = Some(match icon {
                None => rect,
                Some(acc) => ParticleIcon {
                    min_u: acc.min_u.min(rect.min_u),
                    max_u: acc.max_u.max(rect.max_u),
                    min_v: acc.min_v.min(rect.min_v),
                    max_v: acc.max_v.max(rect.max_v),
                },
            });
        }
        icon.or_else(|| particle_quad(model).map(quad_icon))
    }
}

// -------------------------------------------------------------------------------------------
// S2APacketParticles
// -------------------------------------------------------------------------------------------

/// `mc-protocol`'s decoded packet → `mc-render`'s.
///
/// The two structs exist separately on purpose: `mc-render` does not depend on `mc-protocol`
/// (see [`packet`]'s module docs), so this crate — which depends on both — is the only place the
/// join can be made.
///
/// java: `S2APacketParticles.readPacketData:51-56` substitutes `EnumParticleTypes.BARRIER` for an
/// id `getParticleFromId` does not know. `mc-protocol` now performs that normalization at its
/// packet boundary; this defensive lookup still keeps manually constructed packets on the same
/// live-render path instead of letting [`packet::apply`]'s `type_by_id` early-return drop them.
#[must_use]
pub fn to_render_packet(packet: &s2c::Particles) -> S2APacketParticles {
    let particle_id = if packet::type_by_id(packet.particle_id).is_some() {
        packet.particle_id
    } else {
        packet::BARRIER
    };
    S2APacketParticles {
        particle_id,
        long_distance: packet.long_distance,
        pos: DVec3::new(
            f64::from(packet.x),
            f64::from(packet.y),
            f64::from(packet.z),
        ),
        offset: Vec3::new(packet.offset_x, packet.offset_y, packet.offset_z),
        speed: packet.particle_speed,
        count: packet.particle_count.min(MAX_BURST_PARTICLES),
        args: packet.extra_data.clone(),
    }
}

/// The most particles one `S2APacketParticles` spawns.
///
/// java: `handleParticles` runs `for (k = 0; k < count; ++k)`, six Gaussian draws and a spawn
/// attempt each, so one packet with a count of `Integer.MAX_VALUE` freezes a vanilla client for
/// minutes; it is a known way to knock 1.8 players off a server. Here a burst stops at 65,536,
/// sixteen times the 4000 particles a layer keeps (`EffectRenderer` evicts the oldest past that),
/// so every count up to it behaves exactly as before and a larger one still fills the layer.
pub const MAX_BURST_PARTICLES: i32 = 1 << 16;

/// Apply one `S2APacketParticles` to the live particle set.
///
/// java: `NetHandlerPlayClient.handleParticles` (`:2003`) → `WorldClient.spawnParticle` →
/// `RenderGlobal.spawnParticle` (`:2069`). Everything after the type/argument conversion is
/// [`packet::apply`], which already implements both the `count == 0` exact-velocity branch and
/// the `count > 0` Gaussian scatter, plus the range gate.
///
/// `view_pos` is the render-view entity's **feet** position, matching
/// `RenderGlobal.spawnParticle`'s `this.mc.getRenderViewEntity().posX/posY/posZ` — not the camera
/// eye, and not the third-person orbit point.
///
/// `particles_setting` is [`mc_render::GameSettings::particles`] read fresh on every packet, the
/// way `RenderGlobal.spawnParticle:2073` re-reads `gameSettings.particleSetting` on every spawn
/// attempt rather than caching it. That is what makes the video option live — the
/// demotion coin-flip inside [`ParticleDensity::Decreased`] has to be re-rolled per attempt, so
/// caching it at construction would be wrong as well as stale.
pub fn apply_particle_packet(
    system: &mut ParticleSystem,
    world: &World,
    view_pos: DVec3,
    particles_setting: i32,
    packet: &s2c::Particles,
) {
    system.set_density(ParticleDensity::from_raw(particles_setting));
    // (server S2A particles; every spawn path ends in
    // `mc_render::particle::ParticleSystem::spawn`)
    packet::apply(
        system,
        world,
        view_pos,
        &to_render_packet(packet),
        &AtlasIcons,
    );
}

// -------------------------------------------------------------------------------------------
// The client-generated spawn path (`RenderGlobal.spawnParticle`, not the packet)
// -------------------------------------------------------------------------------------------

impl Client {
    /// One client-generated particle, at an exact position and velocity.
    ///
    /// java: `RenderGlobal.spawnParticle(EnumParticleTypes, double...)` (`:2064`) — the private
    /// overload every `playAuxSFX` case and every `randomDisplayTick` body goes through, which
    /// takes `ignoreRange` from the type table rather than from a packet flag.
    pub(crate) fn spawn_effect_particle(
        &mut self,
        particle_id: i32,
        pos: DVec3,
        speed: DVec3,
        args: &[i32],
    ) {
        self.spawn_effect_particle_with(particle_id, pos, speed, args, |_| {});
    }

    /// [`Client::spawn_effect_particle`] with a hook that runs on the constructed particle before
    /// it enters the arena. java: the `playAuxSFX` case that keeps `spawnEntityFX`'s return value
    /// and calls `setRBGColorF`/`multiplyVelocity` on it (`RenderGlobal:2318-2325`).
    ///
    /// The three `self.` field reads are written out rather than gathered by a helper method on
    /// purpose: `&mut self.particles` and `&self.session` are disjoint *field* borrows, which a
    /// `fn(&self)` returning a borrow of the world would collapse into one whole-`self` borrow.
    /// Same reason [`apply_particle_packet`] is a free function — see this module's header.
    pub(crate) fn spawn_effect_particle_with(
        &mut self,
        particle_id: i32,
        pos: DVec3,
        speed: DVec3,
        args: &[i32],
        adjust: impl FnMut(&mut mc_render::particle::Particle),
    ) {
        let Some(session) = self.session.as_ref() else {
            return;
        };
        // java: `RenderGlobal.spawnParticle` culls against `getRenderViewEntity()`'s **feet**.
        let view_pos = session
            .predictor()
            .map_or(DVec3::ZERO, |predictor| predictor.state.pos);
        let world = &session.state().world;
        let ignore_range = packet::type_by_id(particle_id).is_some_and(|t| t.ignore_range);
        // java: `gameSettings.particleSetting` is re-read on every spawn attempt (`:2073`).
        self.particles
            .set_density(ParticleDensity::from_raw(self.settings.particles));
        // (client-generated particles: `particle_id` is the vanilla
        // EnumParticleTypes id; `adjust` sees the constructed particle). The particles mod's
        // amount/size table lives on `self.particles` and applies inside `spawn_one_with`, which
        // may run `adjust` once per copy it builds.
        packet::spawn_one_with(
            &mut self.particles,
            world,
            view_pos,
            particle_id,
            ignore_range,
            pos,
            speed,
            args,
            &AtlasIcons,
            adjust,
        );
    }

    /// Insert a particle that vanilla created directly through `EffectRenderer.addEffect`.
    ///
    /// `EntityFirework.StarterFX` is the important caller: its sparks and overlay are already
    /// fully constructed and therefore bypass `RenderGlobal.spawnParticle`'s packet factory,
    /// range test, and particle-density setting. Keeping this separate from
    /// [`Self::spawn_effect_particle`] prevents those client-local effects from being culled or
    /// reconstructed as the wrong packet particle.
    pub(crate) fn spawn_entity_particle(&mut self, particle: mc_render::particle::Particle) {
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let world = &session.state().world;
        self.particles.spawn(world, particle);
    }

    /// The 64-particle burst a broken block leaves. java:
    /// `EffectRenderer.addBlockDestroyEffects` (`:369`), reached from `playAuxSFX` case 2001 and
    /// from `PlayerControllerMP.onPlayerDestroyBlock`.
    ///
    /// Ungated on purpose — see [`packet::add_block_destroy_effects`].
    pub(crate) fn add_block_destroy_effects(&mut self, pos: mc_math::BlockPos, state: BlockState) {
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let world = &session.state().world;
        // java: `EffectRenderer.addBlockDestroyEffects:373` resolves `getActualState` once,
        // before constructing the 64 `EntityDiggingFX` instances.
        let actual = mc_render::terrain::mesher::actual_state(world, state, pos);
        // java: `EntityDiggingFX.setBlockPos(pos)` — the *biome-sampled* `colorMultiplier`, not
        // `getRenderColor`. This is the one place the two diverge visibly: leaves broken in a
        // jungle throw jungle-green debris here, but a `BLOCK_CRACK` packet for the same state
        // throws the flat "basic" green.
        let tint = mc_render::terrain::tint::digging_tint_from_color_multiplier(world, state, pos);
        packet::add_block_destroy_effects(
            &mut self.particles,
            world,
            pos,
            state,
            tint,
            &ActualAtlasIcons { state: &actual },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_data::block_by_name;

    /// A packet for a zero-argument type, at the origin, with `count = 0` (one particle, exact
    /// velocity) — the shape a server sends for a single crit or flame.
    fn packet_of(particle_id: i32, extra_data: Vec<i32>) -> s2c::Particles {
        s2c::Particles {
            particle_id,
            long_distance: false,
            x: 0.0,
            y: 90.0,
            z: 0.0,
            offset_x: 0.0,
            offset_y: 0.0,
            offset_z: 0.0,
            particle_speed: 0.0,
            particle_count: 0,
            extra_data,
        }
    }

    fn at_the_spawn_point() -> DVec3 {
        DVec3::new(0.0, 90.0, 0.0)
    }

    // ---- the conversion --------------------------------------------------------------------

    #[test]
    fn wire_fields_survive_the_conversion_unchanged() {
        let wire = s2c::Particles {
            particle_id: packet::BLOCK_CRACK,
            long_distance: true,
            x: 1.5,
            y: 64.0,
            z: -2.25,
            offset_x: 0.1,
            offset_y: 0.2,
            offset_z: 0.3,
            particle_speed: 0.5,
            particle_count: 7,
            extra_data: vec![300],
        };
        let render = to_render_packet(&wire);
        assert_eq!(render.particle_id, packet::BLOCK_CRACK);
        assert!(render.long_distance);
        assert!((render.pos.x - 1.5).abs() < 1e-6);
        assert!((render.pos.y - 64.0).abs() < 1e-6);
        assert!((render.pos.z + 2.25).abs() < 1e-6);
        assert!((render.offset.x - 0.1).abs() < 1e-6);
        assert!((render.speed - 0.5).abs() < 1e-6);
        assert_eq!(render.count, 7);
        assert_eq!(render.args, vec![300]);
    }

    /// java: `readPacketData` falls back to `BARRIER` for an unknown id. The wire decoder now
    /// performs that substitution; this conversion remains defensive for hand-built packets.
    #[test]
    fn an_unknown_particle_id_becomes_barrier_rather_than_being_dropped() {
        let render = to_render_packet(&packet_of(9999, vec![]));
        assert_eq!(render.particle_id, packet::BARRIER);

        // The spawn half is asserted in `mc-render`
        // (`packet::tests::every_wired_particle_type_now_reaches_its_own_family`), which hands
        // `build` an explicit icon, **not** here.
        //
        // Why not here: `BARRIER` used to fall through `build`'s catch-all to `spawn::generic`,
        // which needs no texture. It is now `Barrier.Factory`, which asks `AtlasIcons` for the
        // barrier item's particle sprite — and `AtlasIcons` reads `mc_assets::model_for`, a
        // **process-global** bake. Whether that bake is installed depends on whether some other
        // test in the same binary has already built a real `Client`, so a spawn/no-spawn
        // assertion here is decided by test *ordering*. It was asserted (as `1`) before this change and briefly as `0` during it; both were order-dependent, which is why neither is
        // asserted now. What this test is actually about — that an unknown id is *rewritten*
        // rather than dropped by the id lookup — is the assertion above.
        let world = World::new();
        let mut system = ParticleSystem::with_seed(1);
        apply_particle_packet(
            &mut system,
            &world,
            at_the_spawn_point(),
            0,
            &packet_of(9999, vec![]),
        );
        assert!(
            system.len() <= 1,
            "at most one particle from a count-0 packet, whatever the icon resolved to"
        );
    }

    // ---- the spawn path --------------------------------------------------------------------

    /// The headline of A14-03/A09-03: a server packet has to end up in the set the debug
    /// overlay's `P:` counts. `ParticleSystem::len` is literally what `driver.rs` formats there.
    #[test]
    fn a_server_packet_puts_particles_into_the_system() {
        let world = World::new();
        let mut system = ParticleSystem::with_seed(2);
        assert_eq!(system.len(), 0, "P: 0 — the state this row was filed about");

        let mut burst = packet_of(packet::FLAME, vec![]);
        burst.particle_count = 12;
        burst.offset_x = 0.5;
        burst.offset_y = 0.5;
        burst.offset_z = 0.5;
        burst.particle_speed = 0.05;
        apply_particle_packet(&mut system, &world, at_the_spawn_point(), 0, &burst);

        assert_eq!(system.len(), 12);
    }

    /// java: `RenderGlobal.spawnParticle:2073` — `particleSetting == 2` drops every particle
    /// whose type does not set `shouldIgnoreRange`. `set_density` had no caller at all,
    /// so "Minimal" did nothing.
    #[test]
    fn the_particles_video_setting_is_read_on_every_packet() {
        let world = World::new();

        let mut minimal = ParticleSystem::with_seed(3);
        apply_particle_packet(
            &mut minimal,
            &world,
            at_the_spawn_point(),
            2,
            &packet_of(packet::FLAME, vec![]),
        );
        assert_eq!(minimal.len(), 0, "'minimal' must drop a ranged particle");
        assert_eq!(minimal.density(), ParticleDensity::Minimal);

        // ... but not one whose type ignores range: java `getShouldIgnoreRange()`.
        apply_particle_packet(
            &mut minimal,
            &world,
            at_the_spawn_point(),
            2,
            &packet_of(packet::EXPLOSION_HUGE, vec![]),
        );
        assert_eq!(minimal.len(), 1);

        let mut all = ParticleSystem::with_seed(3);
        apply_particle_packet(
            &mut all,
            &world,
            at_the_spawn_point(),
            0,
            &packet_of(packet::FLAME, vec![]),
        );
        assert_eq!(all.len(), 1, "'all' must keep it");
        assert_eq!(all.density(), ParticleDensity::All);
    }

    /// java: `RenderGlobal.spawnParticle`'s `d0*d0 + d1*d1 + d2*d2 > 256.0` check, measured from
    /// the render-view entity.
    #[test]
    fn the_16_block_range_check_is_measured_from_the_view_position() {
        let world = World::new();
        let mut near = ParticleSystem::with_seed(4);
        apply_particle_packet(
            &mut near,
            &world,
            DVec3::new(0.0, 90.0, 10.0),
            0,
            &packet_of(packet::FLAME, vec![]),
        );
        assert_eq!(near.len(), 1);

        let mut far = ParticleSystem::with_seed(4);
        apply_particle_packet(
            &mut far,
            &world,
            DVec3::new(0.0, 90.0, 100.0),
            0,
            &packet_of(packet::FLAME, vec![]),
        );
        assert_eq!(far.len(), 0);
    }

    // ---- IconLookup ------------------------------------------------------------------------
    //
    // These run against whatever bake is installed process-globally. In a bare `cargo test -p
    // mc-client --lib` run nothing has installed one, so `model_for` returns the empty model and
    // every lookup answers `None` — which is itself the behaviour worth pinning (it must be
    // `None`, not a panic and not a bogus full-atlas rect). The *populated* case is covered by
    // `tests/particles.rs`, which builds a real `Client` and therefore a real bake.

    /// `F-P2-client.particle-01-3`: exact transcription of the groups inside
    /// `BlockModelShapes.getTexture`'s missing-model branch. All other blocks must continue to
    /// use their baked model's particle texture instead of being captured by this table.
    #[test]
    fn builtin_block_particle_fallback_table_matches_java() {
        for (ids, texture) in [
            (
                &[54, 63, 68, 146, 176, 177][..],
                "textures/blocks/planks_oak",
            ),
            (&[130][..], "textures/blocks/obsidian"),
            (&[10, 11][..], "textures/blocks/lava_still"),
            (&[8, 9][..], "textures/blocks/water_still"),
            (&[144][..], "textures/blocks/soul_sand"),
            (&[166][..], "textures/items/barrier"),
            // The remaining registered built-ins (air, piston_extension, end_portal) fall through
            // to the missing model's particle texture.
            (&[0, 36, 119][..], "missingno"),
        ] {
            for &id in ids {
                assert_eq!(
                    builtin_block_particle_texture(BlockId(id)),
                    Some(texture),
                    "block id {id}"
                );
            }
        }

        for id in [1, 34, 145, 167, 175] {
            assert_eq!(
                builtin_block_particle_texture(BlockId(id)),
                None,
                "ordinary/non-fallback block id {id}"
            );
        }
    }

    #[test]
    fn block_icon_is_none_rather_than_panicking_before_a_bake_is_installed() {
        let stone = BlockState::new(block_by_name("stone").expect("stone"), 0);
        // Total either way: `Some` if some other test in this binary installed a bake first,
        // `None` if not. What must never happen is a panic or a NaN rect. Air has no model; once
        // a bake exists it takes the missing model's `missingno` particle, as in vanilla
        // (`builtins_without_a_named_sprite_use_the_missing_model_particle` pins that).
        for state in [stone, BlockState::AIR] {
            if let Some(icon) = AtlasIcons.block_icon(state) {
                assert!(icon.min_u.is_finite() && icon.max_u.is_finite());
                assert!(icon.min_u <= icon.max_u && icon.min_v <= icon.max_v);
            }
        }
    }

    fn quad_with_uv(u: f32, v: f32) -> BakedQuad {
        BakedQuad {
            verts: [mc_assets::model::Vertex {
                uv: [u, v],
                ..Default::default()
            }; 4],
            face: Some(Facing::North),
            shade_face: Some(Facing::North),
            tint_index: None,
            shade: true,
        }
    }

    #[test]
    fn authoritative_particle_uv_wins_over_geometry_even_for_a_parent_style_model() {
        // A parent-chain model can inherit a particle entry which is not used by any visible
        // quad.  Java's baked model keeps that sprite separately; a face-first resolver would
        // silently select the unrelated geometry UV instead.
        let retained = UvRect {
            min_u: 0.1,
            max_u: 0.2,
            min_v: 0.3,
            max_v: 0.4,
        };
        let mut model = BakedModel {
            particle_uv: Some(retained),
            ..BakedModel::default()
        };
        model.general_quads.push(quad_with_uv(0.7, 0.8));

        assert_eq!(model_particle_icon(&model), Some(uv_icon(retained)));
    }

    #[test]
    fn missing_model_keeps_its_authoritative_missing_sprite() {
        // The missing cube has ordinary geometry, but Java still returns the model's dedicated
        // missingno particle sprite.  This fixture bites a resolver that starts at a face and
        // also proves that a degenerate-looking sprite is not treated as absent.
        let missing = UvRect {
            min_u: 0.01,
            max_u: 0.02,
            min_v: 0.03,
            max_v: 0.04,
        };
        let mut model = BakedModel {
            particle_uv: Some(missing),
            ..BakedModel::default()
        };
        model.face_quads[Facing::North.index() as usize].push(quad_with_uv(0.7, 0.8));

        assert_eq!(model_particle_icon(&model), Some(uv_icon(missing)));
    }

    #[test]
    fn synthetic_model_without_particle_metadata_uses_the_geometry_escape_hatch() {
        // Programmatic BakedModel fixtures do not have the resolved model JSON's particle entry.
        // They retain the old deterministic side/cap/general inference, but only because the
        // authoritative field is absent.
        let mut model = BakedModel::default();
        model.general_quads.push(quad_with_uv(0.7, 0.8));

        assert_eq!(
            model_particle_icon(&model),
            Some(ParticleIcon {
                min_u: 0.7,
                max_u: 0.7,
                min_v: 0.8,
                max_v: 0.8,
            })
        );

        model.particle_uv = Some(UvRect {
            min_u: 0.1,
            max_u: 0.2,
            min_v: 0.3,
            max_v: 0.4,
        });
        assert_eq!(
            model_particle_icon(&model),
            model.particle_uv.map(uv_icon),
            "adding authoritative metadata disables the synthetic geometry guess"
        );
    }

    #[test]
    fn destroy_effect_lookup_reaches_a_neighbour_derived_model() {
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
        let (assets, report) = mc_assets::bake_all(&manager, 2048).expect("bake vanilla models");
        assert!(report.is_clean(), "{:?}", report.failures);
        mc_assets::model::reinstall(assets);

        // These built-in renderer blocks have no baked model, but Java supplies a named particle
        // sprite for them. Compare the live IconLookup result with the exact stitched sprite rect.
        for (name, texture) in [
            ("chest", "textures/blocks/planks_oak"),
            ("standing_sign", "textures/blocks/planks_oak"),
            ("wall_sign", "textures/blocks/planks_oak"),
            ("trapped_chest", "textures/blocks/planks_oak"),
            ("standing_banner", "textures/blocks/planks_oak"),
            ("wall_banner", "textures/blocks/planks_oak"),
            ("ender_chest", "textures/blocks/obsidian"),
            ("flowing_lava", "textures/blocks/lava_still"),
            ("lava", "textures/blocks/lava_still"),
            ("flowing_water", "textures/blocks/water_still"),
            ("water", "textures/blocks/water_still"),
            ("skull", "textures/blocks/soul_sand"),
            ("barrier", "textures/items/barrier"),
        ] {
            let state = block_by_name(name).unwrap_or_else(|| panic!("missing block {name}"));
            assert_eq!(
                AtlasIcons.block_icon(state.default_state()),
                named_sprite_icon(texture),
                "fallback sprite for {name}"
            );
        }

        let stored = block_by_name("fence").expect("fence").default_state();
        let mut connected = ActualBlockState::from_stored(stored);
        assert!(connected.set("east", "true"));

        let signature = |model: &BakedModel| {
            model
                .general_quads
                .iter()
                .chain(model.face_quads.iter().flatten())
                .flat_map(|quad| quad.verts)
                .flat_map(|vertex| vertex.pos.map(f32::to_bits))
                .collect::<Vec<_>>()
        };
        let raw = signature(mc_assets::model_for(stored));
        let actual = signature(actual_block_model(&connected));
        assert_ne!(
            actual, raw,
            "negative control: the packed lookup selects the isolated fence model"
        );

        assert!(
            ActualAtlasIcons { state: &connected }
                .block_icon(stored)
                .is_some(),
            "the widened model still supplies a digging-particle icon"
        );
    }

    /// java: `BlockModelShapes.getTexture` for the registered
    /// built-ins without a named fallback (air, `piston_extension`, `end_portal`) returns the
    /// missing model's particle sprite, the stitched `missingno` checkerboard. A `BLOCK_CRACK`
    /// for them therefore spawns checkerboard debris and a 2001 break of a moving piston or an
    /// end portal throws the 64-particle burst; mc-rs dropped both before this change.
    /// `BLOCK_DUST` keeps rejecting render type -1, as `EntityBlockDustFX.Factory` does. Needs
    /// the vanilla jar (skips without it; `MC_JARS_DIR`).
    #[test]
    fn builtins_without_a_named_sprite_use_the_missing_model_particle() {
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
        let (assets, report) = mc_assets::bake_all(&manager, 2048).expect("bake vanilla models");
        assert!(report.is_clean(), "{:?}", report.failures);
        mc_assets::model::reinstall(assets);

        // The stitched checkerboard's own rect, not the full-atlas pre-install placeholder.
        let missingno = mc_assets::ResourceLocation::minecraft("missingno");
        let missing = mc_assets::model::sprite(&missingno).uv;
        assert_ne!(
            missing,
            UvRect::FULL,
            "the bake stitches a real missingno sprite"
        );
        let missing_icon = Some(uv_icon(missing));
        for name in ["air", "piston_extension", "end_portal"] {
            let state = block_by_name(name)
                .unwrap_or_else(|| panic!("missing block {name}"))
                .default_state();
            assert_eq!(AtlasIcons.block_icon(state), missing_icon, "{name}");
        }
        // Negative control: an ordinary model block keeps its own particle sprite.
        let stone = block_by_name("stone").expect("stone").default_state();
        assert!(AtlasIcons
            .block_icon(stone)
            .is_some_and(|icon| Some(icon) != missing_icon));

        let piston_extension = block_by_name("piston_extension")
            .expect("b36")
            .default_state();
        let id = piston_extension.to_java_state_id();
        let world = World::new();
        let spawned = |particle_id| {
            let mut system = ParticleSystem::with_seed(3);
            let packet = S2APacketParticles {
                particle_id,
                long_distance: true,
                pos: DVec3::new(0.0, 64.0, 0.0),
                offset: Vec3::ZERO,
                speed: 0.0,
                count: 0,
                args: vec![id],
            };
            packet::apply(&mut system, &world, DVec3::ZERO, &packet, &AtlasIcons);
            system.len()
        };
        assert_eq!(
            spawned(packet::BLOCK_CRACK),
            1,
            "EntityDiggingFX has no render-type gate"
        );
        assert_eq!(
            spawned(packet::BLOCK_DUST),
            0,
            "EntityBlockDustFX rejects render type -1"
        );

        let mut system = ParticleSystem::with_seed(9);
        let added = packet::add_block_destroy_effects(
            &mut system,
            &world,
            mc_math::BlockPos::new(0, 64, 0),
            piston_extension,
            mc_render::particle::spawn::DEFAULT_DIGGING_TINT,
            &AtlasIcons,
        );
        assert_eq!(added, 64, "a broken moving piston throws the 4x4x4 burst");
    }

    #[test]
    fn item_icon_rejects_ids_outside_the_registry() {
        assert!(AtlasIcons.item_icon(-1, 0).is_none(), "negative id");
        assert!(AtlasIcons.item_icon(70_000, 0).is_none(), "beyond u16");
        assert!(AtlasIcons.item_icon(60_000, 0).is_none(), "unregistered id");
    }

    /// `ItemStack(Item, 1, meta)` clamps negative damage, while the item-model key keeps a
    /// block-item's valid four-bit metadata unchanged.  Values above that range must not be
    /// narrowed through `BlockState::new`, whose storage representation necessarily masks its
    /// input.  The explicit old-mask controls below would pass if the resolver regressed to
    /// `(damage & 0x0F)`, so this test bites both the high-value alias and negative-value wrap.
    #[test]
    fn block_item_particle_metadata_matches_item_model_mesher_without_wrapping() {
        let block = block_by_name("wool").expect("wool block");
        let state = |damage| block_item_particle_state(block, damage);

        assert_eq!(state(-1), Some(BlockState::new(block, 0)));
        assert_eq!(state(0), Some(BlockState::new(block, 0)));
        assert_eq!(state(14), Some(BlockState::new(block, 14)));
        assert_eq!(state(15), Some(BlockState::new(block, 15)));

        assert_eq!(
            state(16),
            None,
            "metadata 16 must take the missing-model path"
        );
        assert_eq!(
            state(i32::MAX),
            None,
            "hostile metadata must not truncate to a variant"
        );

        // Negative controls for the removed implementation: both values used to alias a valid
        // four-bit state under `damage & 0x0F`.
        let negative_damage = -1_i32;
        assert_ne!(
            state(-1),
            Some(BlockState::new(block, (negative_damage & 0x0F) as u8))
        );
        assert_ne!(
            state(16),
            Some(BlockState::new(block, (16_i32 & 0x0F) as u8))
        );
    }

    #[test]
    fn quad_icon_takes_the_uv_bounding_box_of_all_four_corners() {
        let vertex = |u: f32, v: f32| mc_assets::model::Vertex {
            pos: [0.0, 0.0, 0.0],
            uv: [u, v],
        };
        let quad = BakedQuad {
            verts: [
                vertex(0.25, 0.75),
                vertex(0.25, 0.5),
                vertex(0.5, 0.5),
                vertex(0.5, 0.75),
            ],
            face: Some(Facing::North),
            shade_face: Some(Facing::North),
            tint_index: None,
            shade: true,
        };
        let icon = quad_icon(&quad);
        assert!((icon.min_u - 0.25).abs() < 1e-6);
        assert!((icon.max_u - 0.5).abs() < 1e-6);
        assert!((icon.min_v - 0.5).abs() < 1e-6);
        assert!((icon.max_v - 0.75).abs() < 1e-6);
    }

    /// The side-face preference from [`particle_quad`]'s table: a cube model whose six faces all
    /// carry different UVs must yield the north one, not the down one.
    #[test]
    fn particle_quad_prefers_a_side_face_over_the_caps() {
        let quad_at = |u: f32| BakedQuad {
            verts: [
                mc_assets::model::Vertex {
                    pos: [0.0, 0.0, 0.0],
                    uv: [u, 0.0],
                },
                mc_assets::model::Vertex {
                    pos: [0.0, 0.0, 0.0],
                    uv: [u, 0.1],
                },
                mc_assets::model::Vertex {
                    pos: [0.0, 0.0, 0.0],
                    uv: [u + 0.1, 0.1],
                },
                mc_assets::model::Vertex {
                    pos: [0.0, 0.0, 0.0],
                    uv: [u + 0.1, 0.0],
                },
            ],
            face: None,
            shade_face: None,
            tint_index: None,
            shade: true,
        };
        let mut model = BakedModel::default();
        model.face_quads[Facing::Down.index() as usize].push(quad_at(0.0));
        model.face_quads[Facing::Up.index() as usize].push(quad_at(0.2));
        model.face_quads[Facing::North.index() as usize].push(quad_at(0.4));
        let icon = quad_icon(particle_quad(&model).expect("a quad"));
        assert!((icon.min_u - 0.4).abs() < 1e-6, "north wins: {icon:?}");

        // With no cullfaced quads at all (a `cross`, a torch), the uncullfaced list is used.
        let mut loose = BakedModel::default();
        loose.general_quads.push(quad_at(0.6));
        let icon = quad_icon(particle_quad(&loose).expect("a quad"));
        assert!((icon.min_u - 0.6).abs() < 1e-6);

        assert!(
            particle_quad(&BakedModel::default()).is_none(),
            "an empty model has no particle texture"
        );
    }
}
