//! Billboard quad generation — `EntityFX.renderParticle`, run on the CPU once per particle per
//! frame (vanilla immediate-mode `Tessellator` style, just building a vertex buffer instead of
//! issuing GL calls).
//!
//! java: `client/particle/EntityFX.java` `renderParticle`, `client/renderer/ActiveRenderInfo.java`
//! `updateRenderInfo` (the five rotation scalars).
//!
//! # Camera-facing math, and a call-site trap in the original
//!
//! `ActiveRenderInfo.updateRenderInfo` computes five named fields from the view entity's
//! yaw/pitch: `rotationX = cos(yaw)`, `rotationZ = sin(yaw)`, `rotationXZ = cos(pitch)`,
//! `rotationYZ = -rotationZ * sin(pitch)`, `rotationXY = rotationX * sin(pitch)`. It is tempting
//! (and wrong) to read those five names straight into `EntityFX.renderParticle`'s five
//! same-named parameters. **`EffectRenderer.renderParticles` does not call it that way**:
//!
//! ```text
//! float f = ActiveRenderInfo.getRotationX();    // cos(yaw)
//! float f1 = ActiveRenderInfo.getRotationZ();   // sin(yaw)
//! float f2 = ActiveRenderInfo.getRotationYZ();  // -sin(yaw)*sin(pitch)
//! float f3 = ActiveRenderInfo.getRotationXY();  // cos(yaw)*sin(pitch)
//! float f4 = ActiveRenderInfo.getRotationXZ();  // cos(pitch)
//! entityfx.renderParticle(worldrenderer, entityIn, partialTicks, f, f4, f1, f2, f3);
//! //                                                              ^   ^   ^   ^   ^
//! //                                              rotationX ------'   |   |   |   |
//! //                                              rotationZ ----------'   |   |   |
//! //                                              rotationYZ --------------'   |   |
//! //                                              rotationXY ------------------'   |
//! //                                              rotationXZ ----------------------'
//! ```
//!
//! So inside `renderParticle`, the parameter *named* `rotationZ` actually holds `cos(pitch)`, the
//! parameter *named* `rotationYZ` holds `sin(yaw)`, and so on — every one of the last four
//! arguments is shifted by one slot. Reproducing the five *field* formulas and wiring them into
//! the vertex formula by name (as this file's first draft did) silently builds a **different,
//! degenerate** transform: at yaw = 0 (a very ordinary case, not an edge case) the "up" edge of
//! the quad collapses to a vector with zero `Y` component, so every particle facing exactly north
//! or south at eye height vanishes to a zero-height sliver. [`quad`]'s own tests would not have
//! caught this — `facing_south_billboard_lies_in_the_xy_plane` only checked the five scalars in
//! isolation, not the actual `Y`-collapse — it took a GPU readback test placing a particle
//! straight ahead to surface it as a fully invisible pixel.
//!
//! Working out what the scrambled call actually binds gives the two basis vectors vanilla's
//! renderer really uses, and they are exactly the standard camera-facing billboard basis:
//!
//! ```text
//! U (billboard "right", yaw only)      = ( cos(yaw),               0,          sin(yaw)          )
//! V (billboard "up", yaw and pitch)    = (-sin(yaw) * sin(pitch),  cos(pitch),  cos(yaw) * sin(pitch))
//! ```
//!
//! [`rotation_vectors`] computes `U`/`V` directly in this corrected form — using
//! [`mc_math::sin`]/[`mc_math::cos`]'s 65,536-entry LUT (H1), not `f32::sin`, for the same reason
//! terrain and physics do — rather than deriving the basis from [`Camera::view`]'s columns, which
//! would be a different (if visually similar) construction. `ActiveRenderInfo`'s
//! `(1 - i*2)` third-person sign factor is exposed through [`ParticleCameraMode`] and the
//! frame-local [`particle_camera_handoff`]. It is intentionally not process-global: the driver
//! hands the local entity's raw yaw/pitch and the current F5 mode to the upload call. The layer-3
//! lit-particle pass uses the raw basis in Java (`renderLitParticles`), while layers 0..2 use the
//! front-view sign flip from `updateRenderInfo`.

use glam::Vec3;

use crate::frame::Camera;

use super::types::{Particle, ParticleKind};

/// One particle billboard vertex.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleVertex {
    /// Camera-relative position (see [`Camera::relative`]).
    pub pos: [f32; 3],
    /// Atlas UV.
    pub uv: [f32; 2],
    /// `[r, g, b, a]`.
    pub color: [f32; 4],
    /// `[sky, block]`, after the particle family's `getBrightnessForRender` override, normalised
    /// to `0.0..=1.0`.
    pub light: [f32; 2],
    /// Java render-path selector. `0.0` samples the shared dynamic lightmap from `light`;
    /// `1.0` is the `EntityFootStepFX` scalar-colour path, which deliberately does not bind
    /// packed lightmap coordinates at all.
    pub lightmap_mode: f32,
}

impl ParticleVertex {
    /// Bytes per vertex.
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// The `wgpu` attribute layout, for `@location(0..=4)`.
    pub const ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x2,
        2 => Float32x4,
        3 => Float32x2,
        4 => Float32,
    ];

    /// The vertex-buffer layout to hand to `wgpu::VertexState`.
    #[must_use]
    pub const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: Self::SIZE as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

/// The billboard's camera-facing basis, already corrected for `EffectRenderer.renderParticles`'s
/// call-site argument scramble (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotationVectors {
    /// The quad's "right" edge direction, yaw only.
    pub u: Vec3,
    /// The quad's "up" edge direction, yaw and pitch.
    pub v: Vec3,
}

/// The camera mode that Java hands to `ActiveRenderInfo.updateRenderInfo`.
///
/// Vanilla's rear third-person camera has the same particle basis as first person. Only the
/// front F5 view passes `true`, which multiplies `rotationX`, `rotationZ`, `rotationYZ`, and
/// `rotationXY` by `-1` (the latter two contain the factor twice and therefore land unchanged).
/// `rotationXZ` is never sign-flipped. The enum is carried per upload through
/// [`particle_camera_handoff`], rather than through a static/global render flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleCameraMode {
    /// `GameSettings.thirdPersonView == 0`.
    FirstPerson,
    /// `GameSettings.thirdPersonView == 1`.
    ThirdPersonRear,
    /// `GameSettings.thirdPersonView == 2`; Java's sign-flipped `ActiveRenderInfo` path.
    ThirdPersonFront,
}

impl ParticleCameraMode {
    /// Convert the live Java setting to the three supported render modes. Java cycles values
    /// `0,1,2`; treating malformed values as first person keeps this helper total for replay and
    /// headless callers.
    #[must_use]
    pub const fn from_third_person_view(view: u8) -> Self {
        match view {
            1 => Self::ThirdPersonRear,
            2 => Self::ThirdPersonFront,
            _ => Self::FirstPerson,
        }
    }

    const fn front_sign(self) -> f32 {
        match self {
            Self::ThirdPersonFront => -1.0,
            Self::FirstPerson | Self::ThirdPersonRear => 1.0,
        }
    }

    /// A frame-local marker used only on the `Camera` copy handed to particle CPU upload.
    /// Rotation-only camera matrices have zero `x_axis.w`; these exact values are therefore an
    /// unambiguous sideband without changing the GPU camera that records the phase later.
    const fn handoff_tag(self) -> f32 {
        match self {
            Self::FirstPerson => 0.0,
            Self::ThirdPersonRear => 1.0,
            Self::ThirdPersonFront => -1.0,
        }
    }

    const fn from_handoff_tag(tag: f32) -> Self {
        if tag == -1.0 {
            Self::ThirdPersonFront
        } else if tag == 1.0 {
            Self::ThirdPersonRear
        } else {
            Self::FirstPerson
        }
    }
}

/// Prepare the camera copy used by [`super::phase::ParticlePhase::upload`].
///
/// `Camera` predates the particle phase and cannot grow a public render-mode field without
/// widening every renderer call. The upload-only copy therefore carries a typed mode through an
/// affine-matrix sideband (`x_axis.w`); the phase never records this copy, and the original GPU
/// camera remains untouched. `yaw_deg`/`pitch_deg` are the raw local-player angles Java gives to
/// `ActiveRenderInfo`, not the already-orbited F5 camera angles.
#[must_use]
pub fn particle_camera_handoff(
    camera: &Camera,
    yaw_deg: f32,
    pitch_deg: f32,
    mode: ParticleCameraMode,
) -> Camera {
    let mut handoff = *camera;
    handoff.yaw = yaw_deg;
    handoff.pitch = pitch_deg;
    handoff.view.x_axis.w = mode.handoff_tag();
    handoff
}

fn particle_camera_mode(camera: &Camera) -> ParticleCameraMode {
    ParticleCameraMode::from_handoff_tag(camera.view.x_axis.w)
}

/// `U`/`V` as vanilla's scrambled call actually binds them — see the module docs for the full
/// derivation. This no-mode convenience is the first-person basis; callers with an explicit F5
/// state use [`rotation_vectors_for_mode`].
#[must_use]
pub fn rotation_vectors(yaw_deg: f32, pitch_deg: f32) -> RotationVectors {
    let ry = yaw_deg.to_radians();
    let rp = pitch_deg.to_radians();
    let (sin_yaw, cos_yaw) = (mc_math::sin(ry), mc_math::cos(ry));
    let (sin_pitch, cos_pitch) = (mc_math::sin(rp), mc_math::cos(rp));
    RotationVectors {
        u: Vec3::new(cos_yaw, 0.0, sin_yaw),
        v: Vec3::new(-sin_yaw * sin_pitch, cos_pitch, cos_yaw * sin_pitch),
    }
}

/// Compute the `ActiveRenderInfo` billboard basis for one explicit camera mode.
///
/// Java applies its front-view factor to all four named horizontal billboard components. In the
/// corrected `U`/`V` basis this is exactly a sign on `U`; `V` is unchanged because Java multiplies
/// `rotationYZ` and `rotationXY` by the factor once when constructing the yaw component and once
/// again at assignment. Keeping this as a separate pure helper makes the mode transition
/// testable without a GPU or a static render-info singleton.
#[must_use]
pub fn rotation_vectors_for_mode(
    yaw_deg: f32,
    pitch_deg: f32,
    mode: ParticleCameraMode,
) -> RotationVectors {
    let mut vectors = rotation_vectors(yaw_deg, pitch_deg);
    vectors.u *= mode.front_sign();
    vectors
}

/// java: `EntityFX.renderParticle`'s render-time scale — for kinds vanilla ramps
/// (`peak_scale * clamp((age+partial)/maxAge*32, 0, 1)` for smoke/crit/cloud/reddust, a quadratic
/// ease for flame/lava, a squared ease for portal). Every other kind renders at
/// [`Particle::scale`] unmodified.
#[must_use]
fn effective_scale(p: &Particle, partial_ticks: f32) -> f32 {
    let age_f = p.age as f32 + partial_ticks;
    let max_age = p.max_age.max(1) as f32;
    match p.kind {
        ParticleKind::Smoke { peak_scale }
        | ParticleKind::Crit { peak_scale }
        | ParticleKind::Cloud { peak_scale }
        | ParticleKind::Reddust { peak_scale }
        // java: `EntityHeartFX`, `EntityNoteFX` and `EntitySnowShovelFX` each carry a
        // byte-for-byte copy of `EntityReddustFX.renderParticle`'s ramp.
        | ParticleKind::Heart { peak_scale }
        | ParticleKind::Note { peak_scale }
        | ParticleKind::SnowShovel { peak_scale } => {
            let f = (age_f / max_age * 32.0).clamp(0.0, 1.0);
            peak_scale * f
        }
        // java: `Barrier.renderParticle` writes `0.5F` straight into the vertex offsets, with no
        // `0.1F * particleScale` factor — the same shape as `EntityLargeExplodeFX` below. `quad`
        // multiplies by `0.1`, so `5.0` reproduces a half-block-wide sprite exactly.
        ParticleKind::Barrier => 5.0,
        ParticleKind::Flame { peak_scale } => {
            let f = age_f / max_age;
            peak_scale * (1.0 - f * f * 0.5)
        }
        ParticleKind::Lava { peak_scale } => {
            let f = age_f / max_age;
            peak_scale * (1.0 - f * f)
        }
        ParticleKind::Portal { .. } => {
            // java: `f = age/maxAge; f = 1-f; f = f*f; f = 1-f;` — portalParticleScale is
            // `Particle::scale` itself (spawn-time, never mutated by `tick`).
            let raw = age_f / max_age;
            let inv = 1.0 - raw;
            p.scale * (1.0 - inv * inv)
        }
        // java: `EntityFirework.OverlayFX.renderParticle` writes this half-extent directly,
        // bypassing both `particleScale` and the base renderer's `0.1F` factor. `quad` applies
        // that factor below, so return ten times Java's `7.1F * sin(...)` here.
        ParticleKind::FireworkOverlay => {
            71.0 * mc_math::sin(
                (age_f - 1.0) * 0.25 * core::f32::consts::PI,
            )
        }
        // java: `EntityLargeExplodeFX.renderParticle:49` — `f4 = 2.0F * field_70582_as` is the
        // quad's **half-width in blocks**, written straight into the vertex positions with no
        // `0.1` factor. Every other family goes through `EntityFX.renderParticle`'s
        // `f4 = 0.1F * particleScale`, which is the `0.1 *` in `quad` below, so the equivalent
        // return here is `20.0 * scale_factor` (`0.1 * 20 * s == 2 * s`). Ignoring this arm made
        // every explosion render at the base `Particle::scale`, ~20x too small.
        ParticleKind::LargeExplode { scale_factor, .. } => 20.0 * scale_factor,
        _ => p.scale,
    }
}

/// Add one of vanilla's `0..=240` packed-light ramps to a world-light level and clamp it to the
/// last lightmap texel. The integer truncation is intentional: Java casts the ramp to `int`
/// before adding it to the packed byte.
#[must_use]
fn add_light_ramp(level: f32, ramp: f32) -> f32 {
    let base = level.clamp(0.0, 15.0) * 16.0;
    let addition = (ramp.clamp(0.0, 1.0) * 15.0 * 16.0) as u16;
    ((base + f32::from(addition)).min(240.0)) / 240.0
}

/// Reproduces each particle subclass's `getBrightnessForRender` packed-light override as
/// `[sky, block]` fractions. Vanilla's low packed coordinate is block light and its high packed
/// coordinate is sky light; importantly, most overrides change only one of them.
#[must_use]
fn render_light(p: &Particle, partial_ticks: f32, sky_level: f32, block_level: f32) -> [f32; 2] {
    let sky = (sky_level / 15.0).clamp(0.0, 1.0);
    let block = (block_level / 15.0).clamp(0.0, 1.0);
    let max_age = p.max_age.max(1) as f32;
    match p.kind {
        // EntityFlameFX raises low byte `j`, i.e. block light, and includes partial ticks.
        ParticleKind::Flame { .. } => {
            let age = (p.age as f32 + partial_ticks) / max_age;
            [sky, add_light_ramp(block_level, age)]
        }
        // java: `EntityFirework.SparkFX.getBrightnessForRender` returns the constant `15728880`
        // (`240 << 16 | 240` — maximum sky *and* block light) and `getBrightness` returns `1.0F`.
        // Falling through to world light took the ambient sample instead, so a firework burst at
        // night or high in the sky was dim where vanilla's blazes.
        // `F-P2-client.particle-02-4`, `F-P2-client.particle-SWEEP-7`.
        ParticleKind::FireworkSpark { .. } => [1.0, 1.0],
        // EntityLavaFX forces low byte `j` to 240 but preserves high sky byte `k`.
        ParticleKind::Lava { .. } => [sky, 1.0],
        // Non-water EntityDropParticleFX returns 257: sky coordinate 0 and block coordinate just
        // beyond 240, which clamps to the final block-light texel.
        ParticleKind::Drip {
            is_water: false, ..
        } => [0.0, 1.0],
        // Its custom renderer writes `.lightmap(0, 240)` on each vertex. The separately-defined
        // `getBrightnessForRender() == 61680` is not consulted by that renderer.
        ParticleKind::LargeExplode { .. } => [0.0, 1.0],
        // Portal and enchantment glyph raise only high sky byte `k`, using age without partials.
        ParticleKind::Portal { .. } | ParticleKind::EnchantGlyph { .. } => {
            let age = (p.age as f32 / max_age).clamp(0.0, 1.0);
            let sq = age * age;
            [add_light_ramp(sky_level, sq * sq), block]
        }
        _ => [sky, block],
    }
}

/// `EntityLargeExplodeFX`'s 4x4 animation grid inside `textures/entity/explosion.png`.
/// java: `EntityLargeExplodeFX.renderParticle:40-48`.
///
/// Split out of [`uv_rect`] so the frame arithmetic can be unit-tested against the Java line by
/// line without building a whole quad.
#[must_use]
fn explosion_frame_rect(age: u32, max: u32, partial_ticks: f32) -> (f32, f32, f32, f32) {
    // java: `int i = (int)(((float)field_70581_a + partialTicks) * 15.0F / (float)field_70584_aq)`
    // — `field_70581_a` is the tick counter this port reuses `Particle::age` for (see `tick.rs`'s
    // `LargeExplode` arm). Vanilla guards with `if (i <= 15)` and draws *nothing* above that;
    // that can only happen once the particle is already dead (`age == max` sets `dead`), so the
    // clamp here is defensive rather than a behaviour change.
    let max = max.max(1) as f32;
    let i = (((age as f32 + partial_ticks) * 15.0 / max) as i32).clamp(0, 15);
    // java: `f = (i % 4) / 4.0F; f1 = f + 0.24975F; f2 = (i / 4) / 4.0F; f3 = f2 + 0.24975F;`
    // — 0.24975, not 0.25: vanilla insets by half a texel to avoid bleeding the next cell.
    const CELL_INSET: f32 = 0.24975;
    let u0 = (i % 4) as f32 / 4.0;
    let v0 = (i / 4) as f32 / 4.0;
    (u0, u0 + CELL_INSET, v0, v0 + CELL_INSET)
}

/// `(min_u, max_u, min_v, max_v)`, java: `renderParticle`'s `f, f1, f2, f3`.
#[must_use]
fn uv_rect(p: &Particle, partial_ticks: f32) -> (f32, f32, f32, f32) {
    // java: `OverlayFX.renderParticle` spans a fixed 4x4-cell region of particles.png rather
    // than reading `particleTextureIndexX/Y` through the base 16x16-cell path.
    if matches!(p.kind, ParticleKind::FireworkOverlay) {
        return (0.25, 0.5, 0.125, 0.375);
    }
    // java: `Barrier.renderParticle` reads `getMinU()/getMaxU()/getMinV()/getMaxV()` directly —
    // the whole sprite, not `EntityDiggingFX`'s quarter-sprite jitter window. Without this arm
    // the barrier icon would render as one randomly-chosen sixteenth of itself.
    if let (ParticleKind::Barrier, Some(icon)) = (&p.kind, p.icon) {
        return (icon.min_u, icon.max_u, icon.min_v, icon.max_v);
    }
    if let Some(icon) = p.icon {
        let u0 = icon.interpolated_u(p.jitter_x / 4.0 * 16.0);
        let u1 = icon.interpolated_u((p.jitter_x + 1.0) / 4.0 * 16.0);
        let v0 = icon.interpolated_v(p.jitter_y / 4.0 * 16.0);
        let v1 = icon.interpolated_v((p.jitter_y + 1.0) / 4.0 * 16.0);
        (u0, u1, v0, v1)
    } else if let ParticleKind::LargeExplode { max, .. } = p.kind {
        // This family draws from `explosion.png`, not the 16x16 misc grid, and it never
        // sets `tex_x`/`tex_y` — so without this arm it rendered cell (0,0) of the explosion
        // sheet at 1/4 the width, frozen, for its whole life.
        explosion_frame_rect(p.age, max, partial_ticks)
    } else {
        // java: `f = texX/16.0; f1 = f + 0.0624375; f2 = texY/16.0; f3 = f2 + 0.0624375;`
        const CELL_INSET: f32 = 0.0624375;
        let u0 = p.tex_x as f32 / 16.0;
        let v0 = p.tex_y as f32 / 16.0;
        (u0, u0 + CELL_INSET, v0, v0 + CELL_INSET)
    }
}

/// Per-frame colour overrides applied by `renderParticle` implementations.
#[must_use]
fn render_color(p: &Particle, partial_ticks: f32) -> [f32; 4] {
    let mut color = p.color;
    if matches!(p.kind, ParticleKind::FireworkOverlay) {
        // java: `particleAlpha = 0.6F - (age + partialTicks - 1) * 0.25F * 0.5F`.
        color[3] = 0.6 - (p.age as f32 + partial_ticks - 1.0) * 0.25 * 0.5;
    }
    if matches!(p.kind, ParticleKind::Footstep) {
        // java: `EntityFootStepFX.renderParticle`: `f = ((age + partial) / 200)^2`, then
        // `f1 = min(1, 2 - 2*f) * 0.2`. The clamp is observationally redundant during the
        // particle's 0..199 live frames, but retained exactly for synthetic callers.
        let f = (p.age as f32 + partial_ticks) / p.max_age.max(1) as f32;
        color[3] = (2.0 - 2.0 * f * f).min(1.0) * 0.2;
    }
    color
}

/// `EntityFootStepFX.renderParticle`'s standalone ground-aligned quad. It deliberately does
/// not use [`rotation_vectors`]: vanilla ignores every camera rotation parameter for footprints.
/// The texture is bound by [`super::phase::ParticlePhase`] as a separate draw state.
#[must_use]
fn footstep_quad(
    p: &Particle,
    cam: &Camera,
    partial_ticks: f32,
    sky_level: f32,
    block_level: f32,
) -> [ParticleVertex; 4] {
    // `EntityFootStepFX.renderParticle` subtracts `interpPosX/Y/Z` directly from its current
    // `posX/Y/Z`; unlike EntityFX's billboard path it does not interpolate prevPos -> pos.
    // Footsteps normally never move, but retaining that distinction keeps synthetic/replayed
    // particles and a future constructor change on the Java path too.
    let center = cam.relative(p.pos);
    let color = render_color(p, partial_ticks);
    let light = render_light(p, partial_ticks, sky_level, block_level);
    let mk = |x: f32, z: f32, u: f32, v: f32| ParticleVertex {
        pos: [center.x + x, center.y, center.z + z],
        uv: [u, v],
        color,
        light,
        lightmap_mode: 0.0,
    };
    // java order: (-x,+z)/(0,1), (+x,+z)/(1,1), (+x,-z)/(1,0), (-x,-z)/(0,0).
    [
        mk(-0.125, 0.125, 0.0, 1.0),
        mk(0.125, 0.125, 1.0, 1.0),
        mk(0.125, -0.125, 1.0, 0.0),
        mk(-0.125, -0.125, 0.0, 0.0),
    ]
}

/// Builds one particle's camera-facing quad. `sky_level`/`block_level` are the world light at
/// the particle's position, `0.0..=15.0` (java: `World.getSkyBlockTypeBrightness`-shaped values,
/// pre-extraction — see [`super::phase`] for where they come from).
///
/// Returns four vertices in the exact order `EntityFX.renderParticle` emits them: this is not a
/// triangle strip or fan, [`super::phase`] draws every four vertices as two triangles
/// (`0,1,2` and `0,2,3`).
#[must_use]
pub fn quad(
    p: &Particle,
    cam: &Camera,
    partial_ticks: f32,
    sky_level: f32,
    block_level: f32,
) -> [ParticleVertex; 4] {
    quad_with_basis(
        p,
        cam,
        partial_ticks,
        sky_level,
        block_level,
        &QuadBasis::new(cam),
    )
}

/// The two billboard bases one upload can need, computed once per frame instead of once per
/// particle (PART-2). Both are pure functions of the handoff camera's raw yaw/pitch and mode
/// tag, so hoisting them is bit-exact.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct QuadBasis {
    /// Layers 0..2: `ActiveRenderInfo`'s basis with the F5-front sign applied.
    normal: RotationVectors,
    /// Layer 3 (`renderLitParticles`): the raw first-person basis.
    lit: RotationVectors,
    /// `|U| + |V|` of each basis, for [`Placement::radius`].
    normal_extent: f32,
    lit_extent: f32,
}

impl QuadBasis {
    #[must_use]
    pub(crate) fn new(cam: &Camera) -> Self {
        let normal = rotation_vectors_for_mode(cam.yaw, cam.pitch, particle_camera_mode(cam));
        let lit = rotation_vectors_for_mode(cam.yaw, cam.pitch, ParticleCameraMode::FirstPerson);
        Self {
            normal,
            lit,
            normal_extent: normal.u.length() + normal.v.length(),
            lit_extent: lit.u.length() + lit.v.length(),
        }
    }

    fn for_particle(&self, p: &Particle) -> (RotationVectors, f32) {
        // Java's layer-3 `renderLitParticles` reconstructs the basis directly from the local
        // entity's raw yaw/pitch and does not consume ActiveRenderInfo's front-view sign. The
        // driver supplies those same raw angles on the upload handoff; all ordinary layers
        // consume the explicit mode factor instead.
        if matches!(p.layer, super::types::FxLayer::LitExplosion) {
            (self.lit, self.lit_extent)
        } else {
            (self.normal, self.normal_extent)
        }
    }
}

/// Where one particle's quad goes this frame: the centre and half-size [`quad_at`] builds
/// from, plus the radius of a sphere around `center` containing all four corners (for the
/// phase's conservative frustum test). Computing it once lets the phase cull and then build
/// without evaluating the interpolation and scale ramp twice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Placement {
    pub(crate) center: Vec3,
    half: f32,
    pub(crate) radius: f32,
}

/// [`Placement`] for `p`: exactly the centre and half-size arithmetic [`quad`] has always used.
/// The radius is `|half| * (|U| + |V|)`, an upper bound on `|±U·half ± V·half|` by the triangle
/// inequality, so it holds for the LUT basis as well; footprints use `0.25` (corners at
/// `(±0.125, 0, ±0.125)` around their raw `pos`).
#[must_use]
pub(crate) fn placement(
    p: &Particle,
    cam: &Camera,
    partial_ticks: f32,
    basis: &QuadBasis,
) -> Placement {
    if matches!(p.kind, ParticleKind::Footstep) {
        return Placement {
            center: cam.relative(p.pos),
            half: 0.0,
            radius: 0.25,
        };
    }
    let (_, extent) = basis.for_particle(p);
    let center = cam.relative(p.interpolated_pos(partial_ticks));
    let half = 0.1 * effective_scale(p, partial_ticks);
    Placement {
        center,
        half,
        radius: half.abs() * extent,
    }
}

/// `(centre, radius)` of [`placement`], for tests of the frustum test.
#[cfg(test)]
#[must_use]
pub(crate) fn cull_sphere(
    p: &Particle,
    cam: &Camera,
    partial_ticks: f32,
    basis: &QuadBasis,
) -> (Vec3, f32) {
    let placed = placement(p, cam, partial_ticks, basis);
    (placed.center, placed.radius)
}

/// [`quad`] with the per-frame basis supplied by the caller. Bit-identical to [`quad`].
#[must_use]
pub(crate) fn quad_with_basis(
    p: &Particle,
    cam: &Camera,
    partial_ticks: f32,
    sky_level: f32,
    block_level: f32,
    basis: &QuadBasis,
) -> [ParticleVertex; 4] {
    quad_at(
        p,
        cam,
        partial_ticks,
        sky_level,
        block_level,
        basis,
        &placement(p, cam, partial_ticks, basis),
    )
}

/// [`quad_with_basis`] from an already computed [`placement`] of the same particle, camera,
/// partial tick and basis. Bit-identical to [`quad`].
#[must_use]
pub(crate) fn quad_at(
    p: &Particle,
    cam: &Camera,
    partial_ticks: f32,
    sky_level: f32,
    block_level: f32,
    basis: &QuadBasis,
    placed: &Placement,
) -> [ParticleVertex; 4] {
    if matches!(p.kind, ParticleKind::Footstep) {
        return footstep_quad(p, cam, partial_ticks, sky_level, block_level);
    }
    let (rot, _) = basis.for_particle(p);
    let center = placed.center;
    let half = placed.half;
    let (u0, u1, v0, v1) = uv_rect(p, partial_ticks);
    let light = render_light(p, partial_ticks, sky_level, block_level);
    let color = render_color(p, partial_ticks);

    let u_vec = rot.u;
    let v_vec = rot.v;

    let mk = |su: f32, sv: f32, u: f32, v: f32| ParticleVertex {
        pos: (center + u_vec * (su * half) + v_vec * (sv * half)).into(),
        uv: [u, v],
        color,
        light,
        lightmap_mode: 0.0,
    };

    // java: the base `EntityFX.renderParticle` emits `(f1,f3), (f1,f2), (f,f2), (f,f3)` — `maxU`
    // on the `-U` corner. `EntityDiggingFX` (`:96-99`) and `EntityBreakingFX` (`:68-71`) both
    // **override** `renderParticle` and emit the mirror image, `(f,f3), (f,f2), (f1,f2), (f1,f3)`.
    // `Barrier` also overrides it but keeps the base order, so this keys on the two families
    // rather than on `icon.is_some()`. Using the base order for icon particles flipped every
    // block-break and item-break shard in U — invisible on stone, obvious on logs and rails.
    // `F-P2-client.particle-SWEEP-9`.
    let (u_left, u_right) = if matches!(
        p.kind,
        ParticleKind::Digging { .. } | ParticleKind::Breaking
    ) {
        (u0, u1)
    } else {
        (u1, u0)
    };

    [
        mk(-1.0, -1.0, u_left, v1),
        mk(-1.0, 1.0, u_left, v0),
        mk(1.0, 1.0, u_right, v0),
        mk(1.0, -1.0, u_right, v1),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use mc_math::JavaRandom;

    fn particle() -> Particle {
        let mut r = JavaRandom::new(1);
        super::super::spawn::generic(&mut r, DVec3::new(0.0, 70.0, 0.0), DVec3::ZERO)
    }

    #[test]
    fn footstep_is_a_fixed_ground_quad_with_its_own_full_texture_and_fade() {
        let mut rng = JavaRandom::new(12);
        let mut footstep = super::super::spawn::footstep(&mut rng, DVec3::new(3.0, 70.0, -4.0));
        // A bogus sprite/camera orientation catches both accidental billboard routing and an
        // accidental misc-atlas cell: footprints always use the whole separate PNG.
        footstep.tex_x = 9;
        footstep.tex_y = 11;
        footstep.age = 100;
        // A footprint is normally stationary, but Java still deliberately reads raw `pos`, not
        // the base billboard's prev->current interpolation.  Split them to make that rule bite.
        footstep.prev_pos = DVec3::new(-90.0, -80.0, -70.0);
        let camera = Camera {
            pos: DVec3::new(1.0, 60.0, 2.0),
            yaw: 123.0,
            pitch: -67.0,
            ..Camera::default()
        };
        let q = quad(&footstep, &camera, 0.5, 15.0, 0.0);
        assert_eq!(
            q.map(|v| v.uv),
            [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
        );
        assert!(q.iter().all(|v| (v.pos[1] - 10.0).abs() < 1e-6));
        assert_eq!(q[0].pos, [1.875, 10.0, -5.875]);
        assert_eq!(q[2].pos, [2.125, 10.0, -6.125]);
        let f = 100.5_f32 / 200.0;
        let alpha = (2.0 - 2.0 * f * f).min(1.0) * 0.2;
        assert!((q[0].color[3] - alpha).abs() < 1e-6, "{:?}", q[0].color);
    }

    #[test]
    fn facing_south_billboard_lies_in_the_xy_plane() {
        // yaw 0 = facing south (+Z); the billboard should face the camera, i.e. lie in a plane
        // whose normal is +Z: U = +X (right), V = +Y (up), both with zero Z component.
        let rot = rotation_vectors(0.0, 0.0);
        assert!(
            (rot.u - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-4,
            "{:?}",
            rot.u
        );
        assert!(
            (rot.v - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-4,
            "{:?}",
            rot.v
        );
    }

    #[test]
    fn active_render_info_front_mode_flips_the_four_horizontal_components_only() {
        let yaw = 37.0;
        let pitch = -23.0;
        let first = rotation_vectors_for_mode(yaw, pitch, ParticleCameraMode::FirstPerson);
        let rear = rotation_vectors_for_mode(yaw, pitch, ParticleCameraMode::ThirdPersonRear);
        let front = rotation_vectors_for_mode(yaw, pitch, ParticleCameraMode::ThirdPersonFront);

        // updateRenderInfo's boolean is `thirdPersonView == 2`, not `> 0`: rear F5 is the same
        // basis as first person. Its factor is applied to rotationX/Z and to the yaw components
        // used by rotationYZ/XY; rotationXZ (the pitch cosine) remains positive.
        assert_eq!(rear, first);
        assert!(
            (front.u + first.u).length() < 1.0e-5,
            "front U={:?}",
            front.u
        );
        assert!(
            (front.v - first.v).length() < 1.0e-5,
            "front V={:?}",
            front.v
        );
        assert!(front.v.y > 0.9, "rotationXZ must not flip: {:?}", front.v);
    }

    #[test]
    fn particle_camera_handoff_is_explicit_and_frame_local() {
        let camera = Camera::default();
        let first = particle_camera_handoff(&camera, 37.0, -23.0, ParticleCameraMode::FirstPerson);
        let rear =
            particle_camera_handoff(&camera, 37.0, -23.0, ParticleCameraMode::ThirdPersonRear);
        let front =
            particle_camera_handoff(&camera, 37.0, -23.0, ParticleCameraMode::ThirdPersonFront);

        assert_eq!(
            particle_camera_mode(&camera),
            ParticleCameraMode::FirstPerson
        );
        assert_eq!(
            particle_camera_mode(&first),
            ParticleCameraMode::FirstPerson
        );
        assert_eq!(
            particle_camera_mode(&rear),
            ParticleCameraMode::ThirdPersonRear
        );
        assert_eq!(
            particle_camera_mode(&front),
            ParticleCameraMode::ThirdPersonFront
        );
        assert_eq!(camera.yaw, Camera::default().yaw);
        assert_eq!(camera.pitch, Camera::default().pitch);
        assert_eq!((front.yaw, front.pitch), (37.0, -23.0));
        assert_eq!((rear.yaw, rear.pitch), (37.0, -23.0));
    }

    #[test]
    fn quad_switches_front_f5_for_ordinary_particles_but_not_lit_particles() {
        let p = particle();
        let camera = Camera {
            pos: DVec3::new(0.0, 70.0, -5.0),
            ..Camera::default()
        };
        let first = particle_camera_handoff(&camera, 37.0, -23.0, ParticleCameraMode::FirstPerson);
        let front =
            particle_camera_handoff(&camera, 37.0, -23.0, ParticleCameraMode::ThirdPersonFront);
        let ordinary_first = quad(&p, &first, 0.0, 15.0, 15.0);
        let ordinary_front = quad(&p, &front, 0.0, 15.0, 15.0);
        // q[2]-q[1] is the positive-U edge, so only its U component changes sign under Java's
        // front-view factor; the V edge remains identical.
        let ordinary_u_first =
            Vec3::from(ordinary_first[2].pos) - Vec3::from(ordinary_first[1].pos);
        let ordinary_u_front =
            Vec3::from(ordinary_front[2].pos) - Vec3::from(ordinary_front[1].pos);
        assert!((ordinary_u_front + ordinary_u_first).length() < 1.0e-5);

        let mut lit = p.clone();
        lit.layer = super::super::types::FxLayer::LitExplosion;
        let lit_first = quad(&lit, &first, 0.0, 15.0, 15.0);
        let lit_front = quad(&lit, &front, 0.0, 15.0, 15.0);
        assert_eq!(
            lit_first, lit_front,
            "renderLitParticles keeps raw yaw/pitch in F5 front"
        );
    }

    #[test]
    fn looking_straight_down_the_billboard_lies_flat_in_the_xz_plane() {
        // Pitch = 90 (straight down): U stays +X (yaw-only), V rotates into +X... no — V must
        // rotate to lie in the horizontal plane too, spanning Z, so the billboard is still a
        // proper non-degenerate square (this is the case the pre-fix formula collapsed to a
        // line for, at yaw = 0).
        let rot = rotation_vectors(0.0, 90.0);
        assert!(rot.u.length() > 0.9, "{:?}", rot.u);
        assert!(rot.v.length() > 0.9, "{:?}", rot.v);
        // U and V must not be (anti)parallel, or the quad has zero area.
        let cross = rot.u.cross(rot.v);
        assert!(
            cross.length() > 0.9,
            "U and V must not collapse: u={:?} v={:?}",
            rot.u,
            rot.v
        );
    }

    #[test]
    fn u_and_v_never_collapse_across_a_full_orbit() {
        // The bug this guards against only showed up at specific yaw/pitch combinations
        // (yaw = 0 with any pitch, in the pre-fix version) — sweep both broadly.
        for yaw in (0..360).step_by(15) {
            for pitch in (-90..=90).step_by(15) {
                let rot = rotation_vectors(yaw as f32, pitch as f32);
                let area = rot.u.cross(rot.v).length();
                assert!(
                    area > 0.9,
                    "yaw={yaw} pitch={pitch}: u={:?} v={:?} area={area}",
                    rot.u,
                    rot.v
                );
            }
        }
    }

    #[test]
    fn quad_is_centred_on_the_camera_relative_particle_position() {
        let p = particle();
        let cam = Camera {
            pos: DVec3::new(0.0, 70.0, -5.0),
            ..Camera::default()
        };
        let mut cam = cam;
        cam.update();
        let verts = quad(&p, &cam, 0.0, 15.0, 15.0);
        let centroid = verts.iter().fold(Vec3::ZERO, |a, v| a + Vec3::from(v.pos)) / 4.0;
        let expected = cam.relative(p.pos);
        assert!(
            (centroid - expected).length() < 1e-4,
            "{centroid:?} vs {expected:?}"
        );
    }

    #[test]
    fn quad_half_width_matches_vanillas_tenth_scale_constant() {
        let mut p = particle();
        p.scale = 2.0;
        let mut cam = Camera::default();
        cam.update();
        let verts = quad(&p, &cam, 0.0, 15.0, 15.0);
        // Adjacent corners along the "U" edge (0 and 3) differ by `2 * half = 2 * 0.1 * scale`.
        let d = (Vec3::from(verts[0].pos) - Vec3::from(verts[3].pos)).length();
        assert!((d - 0.4).abs() < 1e-4, "{d}");
    }

    #[test]
    fn uv_order_matches_java_vertex_order() {
        let mut p = particle();
        p.tex_x = 2;
        p.tex_y = 3;
        let cam = Camera::default();
        let verts = quad(&p, &cam, 0.0, 15.0, 15.0);
        let (u0, u1, v0, v1) = uv_rect(&p, 0.0);
        assert_eq!(verts[0].uv, [u1, v1]);
        assert_eq!(verts[1].uv, [u1, v0]);
        assert_eq!(verts[2].uv, [u0, v0]);
        assert_eq!(verts[3].uv, [u0, v1]);
    }

    #[test]
    fn full_world_light_with_no_boost_is_full_brightness() {
        let p = particle();
        let cam = Camera::default();
        let verts = quad(&p, &cam, 0.0, 15.0, 15.0);
        assert!((verts[0].light[0] - 1.0).abs() < 1e-5);
        assert!((verts[0].light[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn zero_world_light_still_shows_full_bright_kinds() {
        let mut r = JavaRandom::new(2);
        let p = super::super::spawn::lava(&mut r, DVec3::ZERO);
        let cam = Camera::default();
        let verts = quad(&p, &cam, 0.0, 0.0, 0.0);
        assert!(
            (verts[0].light[1] - 1.0).abs() < 1e-5,
            "lava is always full bright"
        );
    }

    #[test]
    fn flame_light_boost_ramps_from_zero_to_one() {
        let mut r = JavaRandom::new(3);
        let mut p = super::super::spawn::flame(&mut r, DVec3::ZERO, DVec3::ZERO);
        p.max_age = 10;
        let cam = Camera::default();
        let early = quad(&p, &cam, 0.0, 0.0, 0.0)[0].light[1];
        p.age = 9;
        let late = quad(&p, &cam, 0.9, 0.0, 0.0)[0].light[1];
        assert!(
            late > early,
            "boost should ramp up with age: {early} -> {late}"
        );
    }

    /// `F-P2-client.particle-SWEEP-7`: Flame raises only Java's low packed byte (block light),
    /// while Portal and EnchantmentTable raise only the high byte (sky light). The old scalar
    /// blend changed both channels and used lerp arithmetic rather than Java's integer addition.
    #[test]
    fn brightness_ramps_change_only_the_java_packed_light_channel() {
        let mut r = JavaRandom::new(31);
        let cam = Camera::default();

        let mut flame = super::super::spawn::flame(&mut r, DVec3::ZERO, DVec3::ZERO);
        flame.max_age = 10;
        flame.age = 2;
        let light = quad(&flame, &cam, 0.5, 3.0, 6.0)[0].light;
        assert!(
            (light[0] - 0.2).abs() < 1e-6,
            "flame preserves sky: {light:?}"
        );
        // block byte = 6*16 + int((2+0.5)/10 * 240) = 96 + 60 = 156.
        assert!(
            (light[1] - 156.0 / 240.0).abs() < 1e-6,
            "flame adds to block: {light:?}"
        );

        for kind in [
            ParticleKind::Portal {
                origin: DVec3::ZERO,
            },
            ParticleKind::EnchantGlyph {
                origin: DVec3::ZERO,
            },
        ] {
            let mut p = particle();
            p.kind = kind;
            p.max_age = 10;
            p.age = 5;
            let at_tick = quad(&p, &cam, 0.0, 3.0, 6.0)[0].light;
            let between_ticks = quad(&p, &cam, 0.9, 3.0, 6.0)[0].light;
            // sky byte = 3*16 + int((5/10)^4 * 240) = 48 + 15 = 63.
            assert!((at_tick[0] - 63.0 / 240.0).abs() < 1e-6, "{at_tick:?}");
            assert!(
                (at_tick[1] - 0.4).abs() < 1e-6,
                "preserves block: {at_tick:?}"
            );
            assert_eq!(
                at_tick, between_ticks,
                "these Java overrides ignore partial ticks"
            );
        }

        // Negative control: base EntityFX preserves both sampled world-light channels.
        let base = quad(&particle(), &cam, 0.5, 3.0, 6.0)[0].light;
        assert!((base[0] - 0.2).abs() < 1e-6 && (base[1] - 0.4).abs() < 1e-6);
    }

    /// Constant packed-light overrides are deliberately distinct. This catches both a return to
    /// the old one-scalar model and accidental grouping of all glowing particles as `[1, 1]`.
    #[test]
    fn constant_brightness_families_keep_their_exact_sky_and_block_coordinates() {
        let mut r = JavaRandom::new(32);
        let cam = Camera::default();

        let lava = super::super::spawn::lava(&mut r, DVec3::ZERO);
        assert_eq!(quad(&lava, &cam, 0.0, 3.0, 6.0)[0].light, [0.2, 1.0]);

        let lava_drip = super::super::spawn::drip(&mut r, DVec3::ZERO, false);
        assert_eq!(quad(&lava_drip, &cam, 0.0, 3.0, 6.0)[0].light, [0.0, 1.0]);

        let large = super::super::spawn::explode_large(&mut r, DVec3::ZERO, 0.0);
        assert_eq!(quad(&large, &cam, 0.0, 3.0, 6.0)[0].light, [0.0, 1.0]);

        // Negative control: water drips inherit EntityFX world light.
        let water_drip = super::super::spawn::drip(&mut r, DVec3::ZERO, true);
        assert_eq!(quad(&water_drip, &cam, 0.0, 3.0, 6.0)[0].light, [0.2, 0.4]);
    }

    #[test]
    fn firework_overlay_uses_its_raw_sine_extent_fixed_uvs_and_frame_alpha() {
        let mut r = JavaRandom::new(4);
        let mut p = super::super::spawn::firework_overlay(&mut r, DVec3::ZERO);
        p.age = 2;
        p.scale = 123.0; // OverlayFX ignores particleScale completely.
        p.color = [0.2, 0.4, 0.8, 0.99];
        let partial = 0.5;

        let angle = (p.age as f32 + partial - 1.0) * 0.25 * core::f32::consts::PI;
        let expected_half = 7.1 * mc_math::sin(angle);
        assert!((0.1 * effective_scale(&p, partial) - expected_half).abs() < 1e-5);
        assert_eq!(uv_rect(&p, partial), (0.25, 0.5, 0.125, 0.375));
        let expected_alpha = 0.6 - (p.age as f32 + partial - 1.0) * 0.25 * 0.5;
        assert_eq!(render_color(&p, partial), [0.2, 0.4, 0.8, expected_alpha]);

        let mut cam = Camera::default();
        cam.update();
        let q = quad(&p, &cam, partial, 15.0, 15.0);
        let width = (Vec3::from(q[0].pos) - Vec3::from(q[3].pos)).length();
        assert!(
            (width - 2.0 * expected_half.abs()).abs() < 1e-4,
            "width {width}"
        );
        assert!(q.iter().all(|v| v.color[3] == expected_alpha));
        assert_eq!(q[0].uv, [0.5, 0.375]);
        assert_eq!(q[2].uv, [0.25, 0.125]);

        // Negative control: the base renderer still uses particleScale, its selected misc cell,
        // and the particle's stored alpha rather than any of OverlayFX's three overrides.
        let mut base = particle();
        base.scale = 2.0;
        base.tex_x = 2;
        base.tex_y = 3;
        base.color[3] = 0.42;
        assert_eq!(effective_scale(&base, partial), 2.0);
        assert_ne!(uv_rect(&base, partial), (0.25, 0.5, 0.125, 0.375));
        assert_eq!(render_color(&base, partial)[3], 0.42);
    }

    // ---- EXPLOSION_LARGE's 4x4 explosion.png animation --------------------------

    /// Transcribes `EntityLargeExplodeFX.renderParticle:40-48` for a hand-picked set of
    /// `(age, max, partialTicks)` and checks the Rust against the arithmetic done by hand.
    #[test]
    fn large_explode_walks_the_4x4_grid_row_major() {
        // max = 15 makes `i` equal to `age` exactly, so the grid walk is readable.
        for (age, want_u0, want_v0) in [
            (0u32, 0.0f32, 0.0f32),
            (1, 0.25, 0.0),
            (3, 0.75, 0.0),
            (4, 0.0, 0.25),
            (7, 0.75, 0.25),
            (12, 0.0, 0.75),
            (15, 0.75, 0.75),
        ] {
            let (u0, u1, v0, v1) = explosion_frame_rect(age, 15, 0.0);
            assert!((u0 - want_u0).abs() < 1e-6, "age {age}: u0 {u0}");
            assert!((v0 - want_v0).abs() < 1e-6, "age {age}: v0 {v0}");
            // java: `f1 = f + 0.24975F`, `f3 = f2 + 0.24975F`.
            assert!((u1 - (want_u0 + 0.24975)).abs() < 1e-6);
            assert!((v1 - (want_v0 + 0.24975)).abs() < 1e-6);
        }
    }

    /// A short-lived explosion still covers the whole sheet: `i` scales by `15 / maxAge`, so a
    /// `max = 6` particle (the low end of vanilla's `6 + rand.nextInt(4)`) steps 0, 2, 5, 7, 10,
    /// 12 rather than 0..5.
    #[test]
    fn large_explode_frame_index_scales_with_its_own_lifetime() {
        let frames: Vec<i32> = (0..6)
            .map(|age| {
                let (u0, _, v0, _) = explosion_frame_rect(age, 6, 0.0);
                (v0 * 4.0).round() as i32 * 4 + (u0 * 4.0).round() as i32
            })
            .collect();
        assert_eq!(frames, vec![0, 2, 5, 7, 10, 12]);
    }

    /// `partial_ticks` must advance the frame *within* a tick — the whole reason `uv_rect` takes
    /// it. At `max = 6` one tick is 2.5 frames, so a nearly-complete frame is already two cells
    /// along: `(0 + 0.99) * 15 / 6 = 2.475 -> i = 2 -> u0 = 0.5`.
    #[test]
    fn large_explode_frame_advances_within_a_tick() {
        let (a, _, _, _) = explosion_frame_rect(0, 6, 0.0);
        let (b, _, _, _) = explosion_frame_rect(0, 6, 0.99);
        assert!((a - 0.0).abs() < 1e-6, "{a}");
        assert!((b - 0.5).abs() < 1e-6, "{b}");
    }

    /// java: `f4 = 2.0F * field_70582_as` is a raw half-width, where every other family gets
    /// `f4 = 0.1F * particleScale`. `quad`'s own `0.1 *` therefore needs `20.0 * scale_factor`.
    #[test]
    fn large_explode_half_width_is_twice_its_scale_factor() {
        let mut r = JavaRandom::new(6);
        // `explode_large`'s `scale_input` feeds `scale_factor = 1.0 - input * 0.5`; 0.0 -> 1.0.
        let p = super::super::spawn::explode_large(&mut r, DVec3::ZERO, 0.0);
        let mut cam = Camera::default();
        cam.update();
        let verts = quad(&p, &cam, 0.0, 15.0, 15.0);
        let width = (Vec3::from(verts[0].pos) - Vec3::from(verts[3].pos)).length();
        assert!(
            (width - 4.0).abs() < 1e-4,
            "half-width 2.0 * 1.0 -> full width 4.0 blocks, got {width}"
        );
    }

    /// The regression A14-18 describes: `LargeExplode` must not fall through to the 16x16
    /// misc-grid branch, whose cells are 0.0624375 wide.
    #[test]
    fn large_explode_does_not_use_the_misc_grid_cell_size() {
        let mut r = JavaRandom::new(7);
        let p = super::super::spawn::explode_large(&mut r, DVec3::ZERO, 0.0);
        let (u0, u1, _, _) = uv_rect(&p, 0.0);
        assert!(
            (u1 - u0 - 0.24975).abs() < 1e-6,
            "expected a quarter-sheet cell, got {}",
            u1 - u0
        );
    }

    #[test]
    fn vertex_layout_size_is_stable() {
        assert_eq!(ParticleVertex::SIZE, 48);
        assert_eq!(ParticleVertex::SIZE % 4, 0);
    }

    /// `F-P2-client.particle-SWEEP-9`: `EntityDiggingFX`/`EntityBreakingFX` override
    /// `renderParticle` and emit `minU` on the `-U` corner; the base `EntityFX` (and `Barrier`,
    /// which also overrides but keeps the base order) emits `maxU` there.
    ///
    /// Asserting all three families is what makes this bite: flipping every particle, or none,
    /// fails.
    #[test]
    fn only_the_two_overriding_icon_families_mirror_their_u_coordinates() {
        let mut r = JavaRandom::new(1);
        let icon = super::super::types::ParticleIcon {
            min_u: 0.25,
            max_u: 0.75,
            min_v: 0.0,
            max_v: 1.0,
        };
        let cam = Camera::default();

        let digging = super::super::spawn::digging(
            &mut r,
            DVec3::new(0.0, 70.0, 0.0),
            DVec3::ZERO,
            icon,
            super::super::spawn::DEFAULT_DIGGING_TINT,
            false,
            mc_math::BlockPos::new(0, 70, 0),
        );
        let q = quad(&digging, &cam, 0.0, 15.0, 15.0);
        assert!(
            q[0].uv[0] < q[2].uv[0],
            "digging: minU belongs on the -U corner, got {:?} then {:?}",
            q[0].uv,
            q[2].uv
        );

        let breaking = super::super::spawn::breaking_item(&mut r, DVec3::new(0.0, 70.0, 0.0), icon);
        let q = quad(&breaking, &cam, 0.0, 15.0, 15.0);
        assert!(q[0].uv[0] < q[2].uv[0], "breaking: same override");

        // Negative control 1: `Barrier` is icon-based but keeps the base order.
        let barrier = super::super::spawn::barrier(&mut r, DVec3::new(0.0, 70.0, 0.0), icon);
        let q = quad(&barrier, &cam, 0.0, 15.0, 15.0);
        assert!(
            q[0].uv[0] > q[2].uv[0],
            "Barrier.renderParticle keeps maxU on the -U corner"
        );

        // Negative control 2: a plain misc-atlas particle keeps the base order too.
        let generic = particle();
        let q = quad(&generic, &cam, 0.0, 15.0, 15.0);
        assert!(q[0].uv[0] > q[2].uv[0], "base EntityFX order");
    }

    /// `F-P2-client.particle-02-4`: `SparkFX.getBrightnessForRender` is the constant `15728880`
    /// (full sky and block light). Taking the world light made a night-time burst dim.
    ///
    /// Negative control: a smoke particle in the same pitch-dark spot stays dark.
    #[test]
    fn firework_sparks_render_full_bright_in_the_dark() {
        let mut r = JavaRandom::new(1);
        let cam = Camera::default();
        let spark = super::super::spawn::firework_spark(
            &mut r,
            DVec3::new(0.0, 70.0, 0.0),
            DVec3::ZERO,
            [1.0, 1.0, 1.0],
            None,
            false,
            false,
        );
        let q = quad(&spark, &cam, 0.0, 0.0, 0.0);
        assert_eq!(
            q[0].light,
            [1.0, 1.0],
            "a spark ignores world light entirely"
        );

        let smoke =
            super::super::spawn::smoke(&mut r, DVec3::new(0.0, 70.0, 0.0), DVec3::ZERO, 1.0);
        let q = quad(&smoke, &cam, 0.0, 0.0, 0.0);
        assert_eq!(q[0].light, [0.0, 0.0], "smoke takes the world light");
    }

    /// java: `Barrier.renderParticle` writes `0.5F` into the vertex offsets with no
    /// `0.1F * particleScale`, and reads the sprite's whole UV rect rather than the 4-way jitter
    /// window every other icon family uses.
    #[test]
    fn a_barrier_particle_is_half_a_block_wide_and_uses_the_whole_sprite() {
        let mut r = JavaRandom::new(1);
        let icon = super::super::types::ParticleIcon {
            min_u: 0.25,
            max_u: 0.75,
            min_v: 0.125,
            max_v: 0.375,
        };
        let barrier = super::super::spawn::barrier(&mut r, DVec3::new(0.0, 70.0, 0.0), icon);
        let cam = Camera::default();
        let q = quad(&barrier, &cam, 0.0, 15.0, 15.0);
        // The quad's U extent is `2 * half`; `half` must be exactly 0.5.
        let width = (Vec3::from(q[2].pos) - Vec3::from(q[0].pos)).length();
        assert!(
            (width - (2.0f32 * 0.5 * 2.0f32.sqrt())).abs() < 1e-4,
            "diagonal of a 1.0 x 1.0 quad, got {width}"
        );
        let us = [q[0].uv[0], q[1].uv[0], q[2].uv[0], q[3].uv[0]];
        assert!(
            us.contains(&0.25) && us.contains(&0.75),
            "whole sprite in U: {us:?}"
        );
        let vs = [q[0].uv[1], q[1].uv[1], q[2].uv[1], q[3].uv[1]];
        assert!(
            vs.contains(&0.125) && vs.contains(&0.375),
            "whole sprite in V: {vs:?}"
        );
    }
}
