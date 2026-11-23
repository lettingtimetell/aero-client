//! The Underwater shader pack: the world as if under the sea.
//!
//! Two render phases, both driven by the client glue (`mc-client/src/app/mods/tw_underwater.rs`):
//!
//! | phase | order | what it does |
//! |---|---|---|
//! | [`UnderwaterPhase`] | [`WORLD_ORDER`] (341, right after AO) | light shafts at half or quarter resolution, then one full-resolution composite over the opaque world and the sky |
//! | [`UnderwaterSnowPhase`] | [`SNOW_ORDER`] (641, before TAA) | marine snow: GPU-generated, depth-tested specks |
//!
//! # Why the composite sits at 341
//!
//! At 341 the frame holds exactly the sky and the opaque world (terrain, entities, AO). Everything
//! drawn later keeps its vanilla look: hitboxes (342), nametags (345), the block outline and cracks
//! (350), particles (400: crits, hit sparks), weather (450), translucent terrain (500, fogged per
//! draw with the pack's water colour by the glue) and the hand (650). So the pack can replace the
//! sky outright (every depth-1 pixel at 341 is sky; at 641 a crit particle over the sky would be
//! erased), and players' names, hits and particles stay crisp. Everything before 643 is resolved
//! by TAA when it runs, which also cleans the shafts' jitter.
//!
//! # What the composite does, per pixel (linear light)
//!
//! * **Caustics** on surfaces facing the (refracted) light, from a baked photon-splat texture
//!   ([`super::caustics`]) projected along the light from the water surface; faint near the eye
//!   and on walls (players are mostly walls), gone at grazing angles and far away.
//! * **Depth darkening** of what lies well below the eye, never darker than a floor.
//! * **Absorption and in-scatter**: per-channel `exp(-sigma * d)` beyond a clear range, red
//!   first, into the water's glow (brighter looking up, darker looking down).
//! * **Far fade** into the open water at the render distance (the pack owns the distance fade,
//!   so the glue turns vanilla's distance fog off while the pack draws).
//! * **The sky** (depth 1): the underside of the water surface, rippling with the waves that
//!   made the caustics. Inside Snell's window it shows the real sky, refracted (the frame itself
//!   where the refracted direction is on screen, vanilla's colours elsewhere); outside it, total
//!   internal reflection of the depths. Below the horizon: open water.
//! * **Light shafts** added on top: the caustic pattern, blurred, extruded along the refracted
//!   sun and integrated along each view ray.
//! * **Refraction wobble** (off by default) offsets the frame lookup.
//!
//! # In shadow
//!
//! The glue keeps a small **light-occlusion map** ([`UnderwaterPhase::occlusion_map`],
//! [`OCCLUSION_SIZE`] block columns around the player, `R8Uint`): per column, one above the
//! highest block that keeps sunlight out (water, ice and glass let it through). A surface point
//! below its column's top, or below the tops a few blocks up the light, is in shadow: no
//! caustics there. Every step of the shaft march tests the same map, so the shafts come down
//! through the gaps between islands and stop under them. The map is refreshed a slice per tick
//! and uploaded only when it changed; it costs one or three texel loads per lookup.
//!
//! # Under real water
//!
//! With the eye in water the same passes run with the "under water" clarity, the real surface
//! height, no far fade and no depth darkening, and every channel's absorption is clamped to
//! vanilla's own fog density ([`UnderwaterFrame::vanilla_density`]): the pack is never harder to
//! see through than vanilla (`exp(-sigma * max(d - r, 0)) >= exp(-rho * d)` when `sigma <= rho`).
//!
//! # Off costs nothing
//!
//! A new phase owns no GPU object. Pipelines, the caustic texture and the shaft target are built
//! on the first active frame and all dropped by [`UnderwaterPhase::set_enabled`]`(false)`. While
//! the mod is on but the pack is not drawing (Only under water, eyes in air) the phases report
//! themselves disabled and the frame is exactly vanilla's; the textures stay so the next dive
//! does not allocate. With MSAA above 1x both phases stay off, like every post effect.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{DVec3, Vec3};

use super::caustics::{self, Caustics};
use crate::post::view_geometry;
use crate::{Camera, DepthAttachment, Gpu, PhaseOrder, RenderPhase, SceneTextures};

/// The world phase's order: right after ambient occlusion, before hitboxes and nametags.
pub const WORLD_ORDER: PhaseOrder = PhaseOrder(PhaseOrder::AMBIENT_OCCLUSION.0 + 1);
/// The marine snow's order: after translucent terrain and the fog, before anti-aliasing.
pub const SNOW_ORDER: PhaseOrder = PhaseOrder(crate::sky::fog::FogPhase::ORDER.0 + 1);

/// One caustic tile, in blocks (at caustic size 1x). Holds about eight cells across.
pub const CAUSTIC_TILE: f64 = 12.0;
/// One tile of the light-shaft pattern, in blocks.
pub const SHAFT_TILE: f64 = 30.0;
/// One loop of the waves at 100% speed, in seconds.
pub const LOOP_SECONDS: f64 = 16.0;
/// The marine snow box, in blocks: specks wrap around the eye inside it.
pub const SNOW_BOX: f64 = 36.0;
/// Specks at 100% snow amount.
pub const MAX_SNOW: u32 = 4000;
/// The water surface's index of refraction.
pub const WATER_ETA: f32 = 1.3333;
/// How much of the caustics survives right next to the eye.
const CAUSTIC_NEAR_KEEP: f32 = 0.3;
/// The water surface's RMS tilt as seen from below (the sky's ripples), as a slope.
const SKY_RIPPLE_SLOPE: f32 = 0.2;
/// The light-occlusion map's edge, in block columns: a window of this many columns centred on
/// the player's chunk (12 chunks: 96 blocks each way, past the shafts' and caustics' reach).
pub const OCCLUSION_SIZE: u32 = 192;

/// When the pack draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnderwaterMode {
    /// The whole world looks under the sea (Overworld only).
    #[default]
    Everywhere,
    /// Nothing changes until the eyes are in water.
    OnlyUnderWater,
}

/// How finely the light shafts are traced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnderwaterQuality {
    /// Quarter resolution, 8 steps, 32 blocks.
    Low,
    /// Quarter resolution, 14 steps, 48 blocks.
    #[default]
    Medium,
    /// Half resolution, 24 steps, 64 blocks.
    High,
}

impl UnderwaterQuality {
    /// From the `quality` choice index.
    #[must_use]
    pub fn from_index(index: usize) -> UnderwaterQuality {
        match index {
            0 => UnderwaterQuality::Low,
            1 => UnderwaterQuality::Medium,
            _ => UnderwaterQuality::High,
        }
    }

    /// Full-resolution pixels per shaft texel along each axis.
    #[must_use]
    pub fn shaft_scale(self) -> u32 {
        match self {
            UnderwaterQuality::Low | UnderwaterQuality::Medium => 4,
            UnderwaterQuality::High => 2,
        }
    }

    /// March steps per shaft texel.
    #[must_use]
    pub fn shaft_steps(self) -> u32 {
        match self {
            UnderwaterQuality::Low => 8,
            UnderwaterQuality::Medium => 14,
            UnderwaterQuality::High => 24,
        }
    }

    /// How far the shafts are traced, in blocks.
    #[must_use]
    pub fn shaft_range(self) -> f32 {
        match self {
            UnderwaterQuality::Low => 32.0,
            UnderwaterQuality::Medium => 48.0,
            UnderwaterQuality::High => 64.0,
        }
    }
}

/// The pack's settings, as the mod's options give them (percentages as fractions).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnderwaterSettings {
    /// When the pack draws.
    pub mode: UnderwaterMode,
    /// The colour far things fade into, display-encoded `0..1`.
    pub water_color: [f32; 3],
    /// Blocks (beyond the clear range) until green light falls to 1/e.
    pub clarity: f32,
    /// Blocks around the eye with no absorption at all.
    pub clear_range: f32,
    /// How much faster red is absorbed than green, `0..1`.
    pub red_fade: f32,
    /// Depth darkening, `0..1`.
    pub depth_darkening: f32,
    /// Replace the sky with the water surface seen from below.
    pub surface_sky: bool,
    /// Caustics on.
    pub caustics: bool,
    /// Caustic strength, `0..1`.
    pub caustic_strength: f32,
    /// Caustic pattern scale, `0.5..2`.
    pub caustic_size: f32,
    /// Wave speed, `0..2` (1 = 100%). Applied by the glue to [`UnderwaterFrame::wave_clock`].
    pub wave_speed: f32,
    /// Light shafts on.
    pub light_shafts: bool,
    /// Light shaft strength, `0..1`.
    pub shaft_strength: f32,
    /// Marine snow on.
    pub marine_snow: bool,
    /// Marine snow amount, `0..1`.
    pub snow_amount: f32,
    /// Refraction wobble on.
    pub refraction: bool,
    /// Refraction wobble strength, `0..1`.
    pub refraction_strength: f32,
    /// Clarity while the eye is in water, blocks (never below vanilla's: see the module docs).
    pub underwater_clarity: f32,
    /// Shaft quality.
    pub quality: UnderwaterQuality,
}

impl Default for UnderwaterSettings {
    /// The mod's defaults (the Lagoon preset).
    fn default() -> Self {
        UnderwaterSettings {
            mode: UnderwaterMode::Everywhere,
            water_color: rgb_from_argb(0xFF2E_A9B4),
            clarity: 128.0,
            clear_range: 8.0,
            red_fade: 0.35,
            depth_darkening: 0.25,
            surface_sky: true,
            caustics: true,
            caustic_strength: 0.5,
            caustic_size: 1.0,
            wave_speed: 1.0,
            light_shafts: true,
            shaft_strength: 0.4,
            marine_snow: true,
            snow_amount: 0.3,
            refraction: false,
            refraction_strength: 0.3,
            underwater_clarity: 48.0,
            quality: UnderwaterQuality::Medium,
        }
    }
}

/// `0xAARRGGBB` to display-encoded `[r, g, b]` in `0..1`.
#[must_use]
pub fn rgb_from_argb(argb: u32) -> [f32; 3] {
    [
        ((argb >> 16) & 0xFF) as f32 / 255.0,
        ((argb >> 8) & 0xFF) as f32 / 255.0,
        (argb & 0xFF) as f32 / 255.0,
    ]
}

fn finite_or(v: f32, fallback: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        fallback
    }
}

impl UnderwaterSettings {
    /// Every number finite and in range.
    #[must_use]
    pub fn sanitized(self) -> UnderwaterSettings {
        let d = UnderwaterSettings::default();
        let unit = |v: f32, fallback: f32| finite_or(v, fallback).clamp(0.0, 1.0);
        UnderwaterSettings {
            water_color: self.water_color.map(|c| unit(c, 0.5)),
            clarity: finite_or(self.clarity, d.clarity).clamp(8.0, 1024.0),
            clear_range: finite_or(self.clear_range, d.clear_range).clamp(0.0, 64.0),
            red_fade: unit(self.red_fade, d.red_fade),
            depth_darkening: unit(self.depth_darkening, d.depth_darkening),
            caustic_strength: unit(self.caustic_strength, d.caustic_strength),
            caustic_size: finite_or(self.caustic_size, 1.0).clamp(0.25, 4.0),
            wave_speed: finite_or(self.wave_speed, 1.0).clamp(0.0, 4.0),
            shaft_strength: unit(self.shaft_strength, d.shaft_strength),
            snow_amount: unit(self.snow_amount, d.snow_amount),
            refraction_strength: unit(self.refraction_strength, d.refraction_strength),
            underwater_clarity: finite_or(self.underwater_clarity, d.underwater_clarity)
                .clamp(8.0, 1024.0),
            ..self
        }
    }
}

/// What the client tells the pack every frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnderwaterFrame {
    /// The camera's eye, world coordinates.
    pub eye: DVec3,
    /// The eye is really in water (vanilla's `Material.water` at the camera).
    pub eye_in_water: bool,
    /// The water surface above the eye, blocks: the real one under water, the pack's eased
    /// virtual one in Everywhere mode.
    pub surface_height: f32,
    /// Wave time in seconds, already scaled by the wave speed (the glue integrates it).
    pub wave_clock: f64,
    /// `Environment::celestial_angle`, `0..1` (0 = noon).
    pub celestial_angle: f32,
    /// `Environment::sun_brightness`, `0.2..1` (rain and thunder included).
    pub sun_brightness: f32,
    /// Rain strength, `0..1`.
    pub rain: f32,
    /// Thunder strength, `0..1`.
    pub thunder: f32,
    /// Vanilla's sky colour overhead (`World.getSkyColor`), display-encoded.
    pub sky_color: [f32; 3],
    /// Vanilla's fog colour for air (the sky at the horizon), display-encoded.
    pub fog_color: [f32; 3],
    /// The render distance in blocks (`chunks * 16`).
    pub render_distance: f32,
    /// Vanilla's exponential fog density for this eye in water (`0.1 - 0.03 * respiration`, or
    /// `0.01` with water breathing). The pack's absorption never exceeds it under water.
    pub vanilla_density: f32,
    /// The light-occlusion map's north-west corner (world block x and z), when the glue keeps
    /// one ([`UnderwaterPhase::occlusion_map`]). `None`: everything is in the open.
    pub occlusion_origin: Option<(i32, i32)>,
}

impl Default for UnderwaterFrame {
    fn default() -> Self {
        UnderwaterFrame {
            eye: DVec3::ZERO,
            eye_in_water: false,
            surface_height: 24.0,
            wave_clock: 0.0,
            celestial_angle: 0.0,
            sun_brightness: 1.0,
            rain: 0.0,
            thunder: 0.0,
            sky_color: [0.47, 0.65, 1.0],
            fog_color: [0.75, 0.85, 1.0],
            render_distance: 128.0,
            vanilla_density: 0.1,
            occlusion_origin: None,
        }
    }
}

/// Smooth Hermite step, as WGSL's `smoothstep`.
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn to_linear(c: [f32; 3]) -> Vec3 {
    Vec3::new(c[0].powf(2.2), c[1].powf(2.2), c[2].powf(2.2))
}

fn to_display(c: Vec3) -> Vec3 {
    let inv = 1.0 / 2.2;
    Vec3::new(
        c.x.max(0.0).powf(inv),
        c.y.max(0.0).powf(inv),
        c.z.max(0.0).powf(inv),
    )
}

/// The light the water gets this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaterLight {
    /// Toward the sun or the moon, in the air (unit).
    pub air_dir: Vec3,
    /// Toward the light inside the water: `air_dir` refracted through a flat surface (unit, y > 0).
    pub water_dir: Vec3,
    /// The light's colour (linear, about 1).
    pub color: Vec3,
    /// How strong it is: 1 at noon, about 0.15 under a full moon, less in rain.
    pub strength: f32,
    /// The ambient level of the water's glow: 1 by day, about 0.07 at night.
    pub ambient: f32,
    /// The sun or moon disc's brightness through the window.
    pub disc: f32,
}

impl WaterLight {
    /// The light for a frame. java: `RenderGlobal.renderSky`'s celestial rotation
    /// (`Ry(-90) * Rx(angle * 360)`) puts the sun at `(-sin, cos, 0)` and the moon opposite.
    #[must_use]
    pub fn new(frame: &UnderwaterFrame) -> WaterLight {
        let theta = frame.celestial_angle * std::f32::consts::TAU;
        let sun = Vec3::new(-theta.sin(), theta.cos(), 0.0);
        let weather =
            (1.0 - 0.65 * frame.rain.clamp(0.0, 1.0)) * (1.0 - 0.4 * frame.thunder.clamp(0.0, 1.0));
        let sun_up = smoothstep(-0.06, 0.22, sun.y);
        let moon_up = smoothstep(-0.06, 0.22, -sun.y) * 0.15;
        let (air_dir, color, strength, disc) = if sun_up >= moon_up {
            let warm = Vec3::new(1.0, 0.62, 0.36);
            let white = Vec3::new(1.0, 0.96, 0.9);
            let color = warm.lerp(white, smoothstep(0.0, 0.45, sun.y));
            (sun, color, sun_up * weather, sun_up)
        } else {
            (-sun, Vec3::new(0.55, 0.66, 0.95), moon_up * weather, 0.35)
        };
        let lit = Vec3::new(air_dir.x, air_dir.y.max(0.05), air_dir.z).normalize();
        let water_dir = -refract(-lit, Vec3::Y, 1.0 / WATER_ETA);
        let day = ((frame.sun_brightness - 0.2) / 0.8).clamp(0.0, 1.0);
        WaterLight {
            air_dir,
            water_dir: water_dir.normalize(),
            color,
            strength,
            ambient: 0.07 + 0.93 * day,
            disc,
        }
    }
}

/// The water's own glow at eye level (linear): the water colour lit by the ambient light, with a
/// moonlit blue tint as the day fades. What far things fade into, and what the glue gives the
/// translucent terrain's per-draw fog.
#[must_use]
pub fn water_glow(settings: &UnderwaterSettings, frame: &UnderwaterFrame) -> Vec3 {
    let light = WaterLight::new(frame);
    let day = ((light.ambient - 0.07) / 0.93).clamp(0.0, 1.0);
    let moonlit = Vec3::new(0.62, 0.74, 1.0).lerp(Vec3::ONE, day);
    to_linear(settings.water_color) * light.ambient * moonlit
}

/// GLSL/WGSL `refract`: `incident` and `normal` unit, `eta` = n1 / n2. Zero on total reflection.
fn refract(incident: Vec3, normal: Vec3, eta: f32) -> Vec3 {
    let d = normal.dot(incident);
    let k = 1.0 - eta * eta * (1.0 - d * d);
    if k < 0.0 {
        Vec3::ZERO
    } else {
        eta * incident - (eta * d + k.sqrt()) * normal
    }
}

/// Per-channel absorption per block for these settings and this eye: `1 / clarity` for green,
/// red faster, blue a little slower; under water every channel is clamped to vanilla's density.
#[must_use]
pub fn absorption(settings: &UnderwaterSettings, frame: &UnderwaterFrame) -> Vec3 {
    let clarity = if frame.eye_in_water {
        settings.underwater_clarity
    } else {
        settings.clarity
    };
    let base = 1.0 / clarity.max(1.0);
    let rf = settings.red_fade;
    let sigma = Vec3::new(base * (1.0 + 2.5 * rf), base, base * (1.0 - 0.2 * rf));
    if frame.eye_in_water {
        sigma.min(Vec3::splat(frame.vanilla_density.max(0.0)))
    } else {
        sigma
    }
}

/// `x` modulo `period`, in `0..period`, computed in double precision.
fn wrap(x: f64, period: f64) -> f32 {
    x.rem_euclid(period) as f32
}

/// `world_params`' uniform, matching `underwater.wgsl`'s `UwParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct WorldParams {
    light_dir: [f32; 4],
    light_color: [f32; 4],
    water: [f32; 4],
    sigma: [f32; 4],
    depth: [f32; 4],
    caustic: [f32; 4],
    wrap: [f32; 4],
    shaft_u: [f32; 4],
    shaft_v: [f32; 4],
    sky: [f32; 4],
    horizon: [f32; 4],
    sun: [f32; 4],
    flags: [f32; 4],
    grid: [f32; 4],
    occl: [f32; 4],
}

impl WorldParams {
    /// The uniform for one frame. Pure: everything it needs is in the arguments.
    #[must_use]
    pub fn new(
        settings: &UnderwaterSettings,
        frame: &UnderwaterFrame,
        frame_index: u32,
        shaft_size: (u32, u32),
    ) -> WorldParams {
        let light = WaterLight::new(frame);
        let water = water_glow(settings, frame);
        let sigma = absorption(settings, frame);
        let in_water = frame.eye_in_water;
        let (far_start, far_end) = if in_water {
            // Vanilla has no far fade under water (only its exponential fog): none here either.
            (1.0e8, 1.0e9)
        } else {
            (frame.render_distance * 0.72, frame.render_distance * 0.98)
        };
        let (dark_k, dark_floor) = if in_water {
            (0.0, 1.0)
        } else {
            (
                settings.depth_darkening * 0.035,
                1.0 - 0.65 * settings.depth_darkening,
            )
        };
        let down_glow = 1.0 - 0.75 * settings.depth_darkening;
        let tile = CAUSTIC_TILE * f64::from(settings.caustic_size);
        let loop_phase = (frame.wave_clock / LOOP_SECONDS).rem_euclid(1.0) as f32;
        let l = light.water_dir;
        let helper = if l.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
        let u = l.cross(helper).normalize();
        let v = l.cross(u).normalize();
        let eye = frame.eye;
        let on = |dir: Vec3| -> f32 {
            let d = eye.x * f64::from(dir.x) + eye.y * f64::from(dir.y) + eye.z * f64::from(dir.z);
            wrap(d, SHAFT_TILE) / SHAFT_TILE as f32
        };
        let quality = settings.quality;
        let shafts = if settings.light_shafts {
            settings.shaft_strength
        } else {
            0.0
        };
        let caustics = if settings.caustics {
            settings.caustic_strength * 0.9
        } else {
            0.0
        };
        let wobble = if settings.refraction {
            settings.refraction_strength * 6.0
        } else {
            0.0
        };
        WorldParams {
            light_dir: [l.x, l.y, l.z, light.strength],
            light_color: [light.color.x, light.color.y, light.color.z, shafts],
            water: [water.x, water.y, water.z, settings.clear_range],
            sigma: [sigma.x, sigma.y, sigma.z, far_start],
            depth: [far_end, dark_k, dark_floor, frame.surface_height.max(0.5)],
            caustic: [
                caustics,
                (1.0 / tile) as f32,
                loop_phase,
                (1.0 / SHAFT_TILE) as f32,
            ],
            wrap: [
                wrap(eye.x, tile) / tile as f32,
                wrap(eye.z, tile) / tile as f32,
                on(u),
                on(v),
            ],
            shaft_u: [u.x, u.y, u.z, quality.shaft_range()],
            shaft_v: [v.x, v.y, v.z, quality.shaft_steps() as f32],
            sky: {
                let s = to_linear(frame.sky_color);
                [s.x, s.y, s.z, if settings.surface_sky { 1.0 } else { 0.0 }]
            },
            horizon: {
                let h = to_linear(frame.fog_color);
                [h.x, h.y, h.z, SKY_RIPPLE_SLOPE / caustics::slope_rms()]
            },
            sun: [
                light.air_dir.x,
                light.air_dir.y,
                light.air_dir.z,
                light.disc,
            ],
            flags: [
                if in_water { 1.0 } else { 0.0 },
                wobble,
                (frame_index % 64) as f32,
                quality.shaft_scale() as f32,
            ],
            grid: [
                shaft_size.0 as f32,
                shaft_size.1 as f32,
                down_glow,
                CAUSTIC_NEAR_KEEP,
            ],
            occl: match frame.occlusion_origin {
                // The eye relative to the map's corner (small, so exact in f32), the eye's
                // height, and the switch.
                Some((x, z)) => [
                    (eye.x - f64::from(x)) as f32,
                    (eye.z - f64::from(z)) as f32,
                    eye.y as f32,
                    1.0,
                ],
                None => [0.0; 4],
            },
        }
    }
}

/// `underwater_snow.wgsl`'s `SnowParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct SnowParams {
    eye: [f32; 4],
    drift: [f32; 4],
    color: [f32; 4],
    sigma: [f32; 4],
    size: [f32; 4],
}

impl SnowParams {
    /// The snow uniform for one frame.
    #[must_use]
    pub fn new(
        settings: &UnderwaterSettings,
        frame: &UnderwaterFrame,
        screen_height: u32,
    ) -> SnowParams {
        let light = WaterLight::new(frame);
        let glow = water_glow(settings, frame);
        let speck =
            to_display(glow * 2.6 + light.color * (0.22 * light.strength) + Vec3::splat(0.02));
        let sigma = absorption(settings, frame);
        let t = frame.wave_clock;
        let eye = frame.eye;
        SnowParams {
            eye: [
                wrap(eye.x, SNOW_BOX),
                wrap(eye.y, SNOW_BOX),
                wrap(eye.z, SNOW_BOX),
                SNOW_BOX as f32,
            ],
            drift: [
                wrap(0.05 * t, SNOW_BOX),
                wrap(-0.11 * t, SNOW_BOX),
                wrap(0.03 * t, SNOW_BOX),
                (t / LOOP_SECONDS).rem_euclid(1.0) as f32,
            ],
            color: [speck.x, speck.y, speck.z, 0.6],
            sigma: [sigma.x, sigma.y, sigma.z, settings.clear_range],
            // A speck is never drawn smaller than 1.5 px (1080p; at least 1.2 px at any size),
            // so its soft disc always covers a few pixels; smaller ones are drawn fainter.
            size: [
                0.024,
                (1.5 * screen_height.max(1) as f32 / 1080.0).max(1.2),
                3.5,
                0.65,
            ],
        }
    }
}

/// How many specks the settings ask for.
#[must_use]
pub fn snow_count(settings: &UnderwaterSettings) -> u32 {
    if settings.marine_snow {
        (settings.snow_amount * MAX_SNOW as f32).round() as u32
    } else {
        0
    }
}

/// The world passes' module source: the view-geometry helpers, the wave constants, the passes.
#[must_use]
pub fn world_source() -> String {
    format!(
        "{}\n{}\n{}",
        view_geometry::COMMON_WGSL,
        caustics::wave_wgsl(),
        include_str!("underwater.wgsl")
    )
}

/// The marine snow module source.
#[must_use]
pub fn snow_source() -> String {
    format!(
        "{}\n{}",
        view_geometry::COMMON_WGSL,
        include_str!("underwater_snow.wgsl")
    )
}

fn entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

fn texture_entry(
    binding: u32,
    sample_type: wgpu::TextureSampleType,
    dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: dimension,
            multisampled: false,
        },
        count: None,
    }
}

fn depth_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    texture_entry(
        binding,
        wgpu::TextureSampleType::Depth,
        wgpu::TextureViewDimension::D2,
    )
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

/// The shaft target's format: `r` = shaft light, `g` = the texel's distance (blocks).
pub const SHAFT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

/// Pipelines, layouts, the uniform and the sampler: built on the first active frame.
struct WorldPipelines {
    format: wgpu::TextureFormat,
    shaft_layout: wgpu::BindGroupLayout,
    blur_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    shafts: wgpu::RenderPipeline,
    blur: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    params: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

impl WorldPipelines {
    fn new(gpu: &Gpu) -> WorldPipelines {
        let device = &gpu.device;
        let format = gpu.config.format;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaderpack/underwater"),
            source: wgpu::ShaderSource::Wgsl(world_source().into()),
        });
        let float = wgpu::TextureSampleType::Float { filterable: true };
        let unfiltered = wgpu::TextureSampleType::Float { filterable: false };
        let uint = wgpu::TextureSampleType::Uint;
        let d2 = wgpu::TextureViewDimension::D2;
        let d3 = wgpu::TextureViewDimension::D3;
        let shaft_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shaderpack/underwater shafts"),
            entries: &[
                view_geometry::uniform_entry(0),
                depth_entry(1),
                texture_entry(3, float, d3),
                sampler_entry(4),
                texture_entry(6, uint, d2),
            ],
        });
        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shaderpack/underwater shaft blur"),
            entries: &[
                view_geometry::uniform_entry(0),
                texture_entry(5, unfiltered, d2),
            ],
        });
        let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shaderpack/underwater composite"),
            entries: &[
                view_geometry::uniform_entry(0),
                depth_entry(1),
                texture_entry(2, float, d2),
                texture_entry(3, float, d3),
                sampler_entry(4),
                texture_entry(5, unfiltered, d2),
                texture_entry(6, uint, d2),
            ],
        });
        let build = |label: &str, group: &wgpu::BindGroupLayout, entry: &str, target| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(gpu.camera_layout()), Some(group)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let shafts = build(
            "shaderpack/underwater shafts",
            &shaft_layout,
            "fs_shafts",
            SHAFT_FORMAT,
        );
        let blur = build(
            "shaderpack/underwater shaft blur",
            &blur_layout,
            "fs_shaft_blur",
            SHAFT_FORMAT,
        );
        let composite = build(
            "shaderpack/underwater composite",
            &composite_layout,
            "fs_composite",
            format,
        );
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shaderpack/underwater params"),
            size: std::mem::size_of::<WorldParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shaderpack/underwater linear repeat"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        WorldPipelines {
            format,
            shaft_layout,
            blur_layout,
            composite_layout,
            shafts,
            blur,
            composite,
            params,
            sampler,
        }
    }
}

/// The caustic 3D texture: a 1x1x1 neutral placeholder until the bake is ready, then the bake.
struct CausticTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    baked: bool,
}

impl CausticTexture {
    fn new(gpu: &Gpu, baked: Option<&Arc<Vec<u8>>>) -> CausticTexture {
        let (size, depth, bytes): (u32, u32, &[u8]) = match baked {
            Some(bytes) => (
                caustics::SIZE as u32,
                caustics::FRAMES as u32,
                bytes.as_slice(),
            ),
            None => (1, 1, &[]),
        };
        let placeholder = caustics::placeholder_texel();
        let data = if baked.is_some() {
            bytes
        } else {
            &placeholder[..]
        };
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shaderpack/underwater caustics"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: depth,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: caustics::FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * caustics::TEXEL_BYTES as u32),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: depth,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        CausticTexture {
            _texture: texture,
            view,
            baked: baked.is_some(),
        }
    }
}

/// The light-occlusion map on the GPU (`R8Uint`, [`OCCLUSION_SIZE`] square).
struct OcclusionTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl OcclusionTexture {
    fn new(gpu: &Gpu) -> OcclusionTexture {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shaderpack/underwater occlusion"),
            size: wgpu::Extent3d {
                width: OCCLUSION_SIZE,
                height: OCCLUSION_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        OcclusionTexture { texture, view }
    }

    fn upload(&self, gpu: &Gpu, map: &[u8]) {
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            map,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(OCCLUSION_SIZE),
                rows_per_image: Some(OCCLUSION_SIZE),
            },
            wgpu::Extent3d {
                width: OCCLUSION_SIZE,
                height: OCCLUSION_SIZE,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// The low-resolution shaft targets: the march's raw output and its blurred copy.
struct ShaftTarget {
    frame: (u32, u32),
    scale: u32,
    size: (u32, u32),
    raw: wgpu::TextureView,
    clean: wgpu::TextureView,
}

/// The three bind groups, cached against what they point at.
struct WorldBindings {
    depth: wgpu::Texture,
    scene: wgpu::TextureView,
    generation: u64,
    shafts: wgpu::BindGroup,
    blur: wgpu::BindGroup,
    composite: wgpu::BindGroup,
}

/// The Underwater pack's world phase: light shafts, then the composite. See the [module docs](self).
pub struct UnderwaterPhase {
    settings: UnderwaterSettings,
    frame: UnderwaterFrame,
    enabled: bool,
    active: bool,
    msaa: bool,
    frame_index: u32,
    caustics: Caustics,
    pipelines: Option<WorldPipelines>,
    caustic_texture: Option<CausticTexture>,
    shaft_target: Option<ShaftTarget>,
    occlusion_texture: Option<OcclusionTexture>,
    /// The CPU copy of the light-occlusion map the glue fills ([`Self::occlusion_map`]).
    occlusion: Vec<u8>,
    occlusion_dirty: bool,
    bindings: Option<WorldBindings>,
    /// Bumped whenever the caustic texture, the shaft target or the occlusion map is replaced.
    generation: u64,
    uploaded: Option<WorldParams>,
    passes: u64,
}

impl std::fmt::Debug for UnderwaterPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnderwaterPhase")
            .field("enabled", &self.enabled)
            .field("active", &self.active)
            .field("msaa", &self.msaa)
            .field("allocated", &self.allocated())
            .field("passes", &self.passes)
            .finish_non_exhaustive()
    }
}

impl UnderwaterPhase {
    /// The phase's name in `PhaseList`, GPU timings and captures.
    pub const NAME: &'static str = "shaderpack/underwater";

    /// An idle phase for a frame with `sample_count` MSAA samples. Allocates nothing.
    #[must_use]
    pub fn new(sample_count: u32) -> UnderwaterPhase {
        UnderwaterPhase {
            settings: UnderwaterSettings::default(),
            frame: UnderwaterFrame::default(),
            enabled: false,
            active: false,
            msaa: sample_count > 1,
            frame_index: 0,
            caustics: Caustics::default(),
            pipelines: None,
            caustic_texture: None,
            shaft_target: None,
            occlusion_texture: None,
            occlusion: Vec::new(),
            occlusion_dirty: false,
            bindings: None,
            generation: 0,
            uploaded: None,
            passes: 0,
        }
    }

    /// The light-occlusion map for the glue to fill: [`OCCLUSION_SIZE`] rows (north to south) of
    /// [`OCCLUSION_SIZE`] columns (west to east), each the world height one above the highest
    /// block that keeps sunlight out (0 = open down to the void). Water, ice and glass let the
    /// light through. Allocated on first use (36 KiB), freed with the rest when switched off.
    /// Call [`Self::mark_occlusion_changed`] after writing.
    pub fn occlusion_map(&mut self) -> &mut [u8] {
        let len = (OCCLUSION_SIZE * OCCLUSION_SIZE) as usize;
        if self.occlusion.len() != len {
            self.occlusion = vec![0; len];
        }
        &mut self.occlusion
    }

    /// The occlusion map changed: upload it before the next frame draws.
    pub fn mark_occlusion_changed(&mut self) {
        self.occlusion_dirty = true;
    }

    /// The mod's switch. Off drops every GPU object.
    pub fn set_enabled(&mut self, on: bool) {
        if on != self.enabled {
            self.enabled = on;
            if !on {
                self.active = false;
                self.release();
            }
        }
    }

    /// Whether the mod is on.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Whether the pack draws this frame (the glue decides: mode, medium, dimension, world).
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    /// The settings.
    pub fn set_settings(&mut self, settings: UnderwaterSettings) {
        self.settings = settings.sanitized();
    }

    /// The settings in effect.
    #[must_use]
    pub fn settings(&self) -> UnderwaterSettings {
        self.settings
    }

    /// This frame's inputs.
    pub fn set_frame(&mut self, frame: UnderwaterFrame) {
        self.frame = frame;
    }

    /// Whether the phase records this frame.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.enabled && self.active && !self.msaa
    }

    /// Whether any GPU object exists.
    #[must_use]
    pub fn allocated(&self) -> bool {
        self.pipelines.is_some()
            || self.caustic_texture.is_some()
            || self.shaft_target.is_some()
            || self.occlusion_texture.is_some()
            || !self.occlusion.is_empty()
            || self.bindings.is_some()
    }

    /// Whether the baked caustics (not the placeholder) are bound.
    #[must_use]
    pub fn caustics_ready(&self) -> bool {
        self.caustic_texture.as_ref().is_some_and(|c| c.baked)
    }

    /// Frames recorded so far.
    #[must_use]
    pub fn passes_recorded(&self) -> u64 {
        self.passes
    }

    /// Block until the caustic bake is done (screenshot tools and tests; the client never waits).
    pub fn wait_for_caustics(&mut self) {
        let _ = self.caustics.wait();
    }

    /// Drop every GPU object.
    pub fn release(&mut self) {
        self.pipelines = None;
        self.caustic_texture = None;
        self.shaft_target = None;
        self.occlusion_texture = None;
        self.occlusion = Vec::new();
        self.occlusion_dirty = false;
        self.bindings = None;
        self.uploaded = None;
    }

    fn prepare(&mut self, gpu: &Gpu, scene: &SceneTextures<'_>) {
        if self
            .pipelines
            .as_ref()
            .is_none_or(|p| p.format != gpu.config.format)
        {
            self.pipelines = Some(WorldPipelines::new(gpu));
            self.bindings = None;
            self.uploaded = None;
        }
        // The caustics: the placeholder first, the bake as soon as it is ready.
        if self.caustic_texture.as_ref().is_none_or(|c| !c.baked) {
            let baked = self.caustics.poll();
            if self.caustic_texture.is_none() || baked.is_some() {
                self.caustic_texture = Some(CausticTexture::new(gpu, baked.as_ref()));
                self.generation += 1;
            }
        }
        let frame = (gpu.config.width.max(1), gpu.config.height.max(1));
        let scale = self.settings.quality.shaft_scale();
        if self
            .shaft_target
            .as_ref()
            .is_none_or(|t| t.frame != frame || t.scale != scale)
        {
            let size = view_geometry::scaled_size(frame.0, frame.1, scale);
            let target =
                |label| view_geometry::target_view(gpu, label, SHAFT_FORMAT, size.0, size.1);
            self.shaft_target = Some(ShaftTarget {
                frame,
                scale,
                size,
                raw: target("shaderpack/underwater shafts"),
                clean: target("shaderpack/underwater shafts blurred"),
            });
            self.generation += 1;
        }
        if self.occlusion_texture.is_none() {
            self.occlusion_texture = Some(OcclusionTexture::new(gpu));
            self.occlusion_dirty = true;
            self.generation += 1;
        }
        if self.occlusion_dirty {
            // Allocates the CPU copy (zeros: all open) if the glue has not filled one yet.
            let _ = self.occlusion_map();
            if let Some(texture) = &self.occlusion_texture {
                texture.upload(gpu, &self.occlusion);
            }
            self.occlusion_dirty = false;
        }
        let (Some(pipelines), Some(caustic), Some(shafts), Some(occlusion)) = (
            &self.pipelines,
            &self.caustic_texture,
            &self.shaft_target,
            &self.occlusion_texture,
        ) else {
            return;
        };
        if self.bindings.as_ref().is_none_or(|b| {
            b.depth != *gpu.depth_texture()
                || b.scene != *scene.color
                || b.generation != self.generation
        }) {
            let depth_view = gpu
                .depth_texture()
                .create_view(&wgpu::TextureViewDescriptor::default());
            let view = wgpu::BindingResource::TextureView;
            let shaft_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shaderpack/underwater shafts"),
                layout: &pipelines.shaft_layout,
                entries: &[
                    entry(0, pipelines.params.as_entire_binding()),
                    entry(1, view(&depth_view)),
                    entry(3, view(&caustic.view)),
                    entry(4, wgpu::BindingResource::Sampler(&pipelines.sampler)),
                    entry(6, view(&occlusion.view)),
                ],
            });
            let blur_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shaderpack/underwater shaft blur"),
                layout: &pipelines.blur_layout,
                entries: &[
                    entry(0, pipelines.params.as_entire_binding()),
                    entry(5, view(&shafts.raw)),
                ],
            });
            let composite_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shaderpack/underwater composite"),
                layout: &pipelines.composite_layout,
                entries: &[
                    entry(0, pipelines.params.as_entire_binding()),
                    entry(1, view(&depth_view)),
                    entry(2, view(scene.color)),
                    entry(3, view(&caustic.view)),
                    entry(4, wgpu::BindingResource::Sampler(&pipelines.sampler)),
                    entry(5, view(&shafts.clean)),
                    entry(6, view(&occlusion.view)),
                ],
            });
            self.bindings = Some(WorldBindings {
                depth: gpu.depth_texture().clone(),
                scene: scene.color.clone(),
                generation: self.generation,
                shafts: shaft_group,
                blur: blur_group,
                composite: composite_group,
            });
        }
        let params = WorldParams::new(&self.settings, &self.frame, self.frame_index, shafts.size);
        if self.uploaded != Some(params) {
            gpu.write_buffer_staged(&pipelines.params, 0, bytemuck::bytes_of(&params));
            self.uploaded = Some(params);
        }
    }
}

impl RenderPhase for UnderwaterPhase {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn enabled(&self) -> bool {
        self.is_active()
    }

    fn needs_scene_color(&self) -> bool {
        self.is_active()
    }

    fn resize(&mut self, _gpu: &Gpu, _width: u32, _height: u32) {
        self.shaft_target = None;
        self.bindings = None;
    }

    fn record(
        &mut self,
        _gpu: &Gpu,
        _enc: &mut wgpu::CommandEncoder,
        _view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        // Only reached if a caller ignores `needs_scene_color`: without the frame there is
        // nothing to composite, so draw nothing.
    }

    fn record_scene(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        if !self.is_active() {
            return;
        }
        // The shafts' dither moves on every frame only while TAA averages it away (its jitter is
        // never exactly zero); without TAA it holds still, or the shafts shimmer on a still scene.
        if cam.jitter != glam::Vec2::ZERO || !super::shadow::steady_shading() {
            self.frame_index = self.frame_index.wrapping_add(1);
        }
        self.prepare(gpu, scene);
        let (Some(pipelines), Some(shafts), Some(bindings)) =
            (&self.pipelines, &self.shaft_target, &self.bindings)
        else {
            return;
        };
        if self.settings.light_shafts && self.settings.shaft_strength > 0.0 {
            let passes = [
                (&shafts.raw, &pipelines.shafts, &bindings.shafts, "shafts"),
                (&shafts.clean, &pipelines.blur, &bindings.blur, "shaft blur"),
            ];
            for (target, pipeline, group, label) in passes {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(label),
                    color_attachments: &[Some(view_geometry::overwrite(target))],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
                pass.set_bind_group(1, group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        let mut pass = gpu.begin_pass(enc, view, Self::NAME, DepthAttachment::None);
        pass.set_pipeline(&pipelines.composite);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.set_bind_group(1, &bindings.composite, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        self.passes += 1;
    }
}

/// The marine snow pipeline and its uniform.
struct SnowGpu {
    format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    params: wgpu::Buffer,
    group: wgpu::BindGroup,
    uploaded: Option<SnowParams>,
}

impl SnowGpu {
    fn new(gpu: &Gpu) -> SnowGpu {
        let device = &gpu.device;
        let format = gpu.config.format;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaderpack/underwater snow"),
            source: wgpu::ShaderSource::Wgsl(snow_source().into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shaderpack/underwater snow"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shaderpack/underwater snow"),
            bind_group_layouts: &[Some(gpu.camera_layout()), Some(&layout)],
            immediate_size: 0,
        });
        let premultiplied = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shaderpack/underwater snow"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_snow"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: gpu.sample_count(),
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_snow"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(premultiplied),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shaderpack/underwater snow params"),
            size: std::mem::size_of::<SnowParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shaderpack/underwater snow"),
            layout: &layout,
            entries: &[entry(0, params.as_entire_binding())],
        });
        SnowGpu {
            format,
            pipeline,
            params,
            group,
            uploaded: None,
        }
    }
}

/// The Underwater pack's marine snow. See the [module docs](self).
pub struct UnderwaterSnowPhase {
    settings: UnderwaterSettings,
    frame: UnderwaterFrame,
    enabled: bool,
    active: bool,
    msaa: bool,
    gpu: Option<SnowGpu>,
    passes: u64,
}

impl std::fmt::Debug for UnderwaterSnowPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnderwaterSnowPhase")
            .field("enabled", &self.enabled)
            .field("active", &self.active)
            .field("allocated", &self.gpu.is_some())
            .field("passes", &self.passes)
            .finish_non_exhaustive()
    }
}

impl UnderwaterSnowPhase {
    /// The phase's name in `PhaseList`, GPU timings and captures.
    pub const NAME: &'static str = "shaderpack/underwater-snow";

    /// An idle phase. Allocates nothing.
    #[must_use]
    pub fn new(sample_count: u32) -> UnderwaterSnowPhase {
        UnderwaterSnowPhase {
            settings: UnderwaterSettings::default(),
            frame: UnderwaterFrame::default(),
            enabled: false,
            active: false,
            msaa: sample_count > 1,
            gpu: None,
            passes: 0,
        }
    }

    /// The mod's switch. Off drops the pipeline.
    pub fn set_enabled(&mut self, on: bool) {
        if on != self.enabled {
            self.enabled = on;
            if !on {
                self.active = false;
                self.gpu = None;
            }
        }
    }

    /// Whether the pack draws this frame.
    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    /// The settings.
    pub fn set_settings(&mut self, settings: UnderwaterSettings) {
        self.settings = settings.sanitized();
    }

    /// This frame's inputs.
    pub fn set_frame(&mut self, frame: UnderwaterFrame) {
        self.frame = frame;
    }

    /// Whether the phase records this frame.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.enabled && self.active && !self.msaa && snow_count(&self.settings) > 0
    }

    /// Whether the pipeline exists.
    #[must_use]
    pub fn allocated(&self) -> bool {
        self.gpu.is_some()
    }

    /// Frames recorded so far.
    #[must_use]
    pub fn passes_recorded(&self) -> u64 {
        self.passes
    }
}

impl RenderPhase for UnderwaterSnowPhase {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn enabled(&self) -> bool {
        self.is_active()
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        if !self.is_active() {
            return;
        }
        if self
            .gpu
            .as_ref()
            .is_none_or(|g| g.format != gpu.config.format)
        {
            self.gpu = Some(SnowGpu::new(gpu));
        }
        let Some(snow) = self.gpu.as_mut() else {
            return;
        };
        let params = SnowParams::new(&self.settings, &self.frame, gpu.config.height);
        if snow.uploaded != Some(params) {
            gpu.write_buffer_staged(&snow.params, 0, bytemuck::bytes_of(&params));
            snow.uploaded = Some(params);
        }
        let mut pass = gpu.begin_pass(enc, view, Self::NAME, DepthAttachment::Shared);
        pass.set_pipeline(&snow.pipeline);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.set_bind_group(1, &snow.group, &[]);
        pass.draw(0..6, 0..snow_count(&self.settings));
        drop(pass);
        self.passes += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(source: &str) {
        let module = wgpu::naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
    }

    /// `cargo check` never parses WGSL; this runs the front end and validator wgpu runs at
    /// pipeline creation.
    #[test]
    fn both_shaders_validate() {
        validate(&world_source());
        validate(&snow_source());
    }

    #[test]
    fn the_orders_bracket_the_world_the_way_the_docs_say() {
        assert!(WORLD_ORDER > PhaseOrder::AMBIENT_OCCLUSION);
        assert!(WORLD_ORDER < crate::entity::hitbox::HITBOX_PHASE_ORDER);
        assert!(WORLD_ORDER < PhaseOrder::BLOCK_OVERLAY);
        assert!(WORLD_ORDER < PhaseOrder::PARTICLES);
        assert!(WORLD_ORDER < crate::sky::fog::FogPhase::BEFORE_TRANSLUCENT_ORDER);
        assert!(SNOW_ORDER > PhaseOrder::TERRAIN_TRANSLUCENT);
        assert!(SNOW_ORDER < PhaseOrder::TEMPORAL_AA);
    }

    #[test]
    fn a_new_phase_is_off_and_owns_nothing() {
        let phase = UnderwaterPhase::new(1);
        assert!(!phase.enabled());
        assert!(!phase.needs_scene_color());
        assert!(!phase.allocated());
        let snow = UnderwaterSnowPhase::new(1);
        assert!(!snow.enabled());
        assert!(!snow.allocated());
    }

    #[test]
    fn it_runs_only_when_switched_on_and_active_and_never_with_msaa() {
        let mut phase = UnderwaterPhase::new(1);
        phase.set_enabled(true);
        assert!(!phase.is_active(), "the glue has not said it draws");
        phase.set_active(true);
        assert!(phase.is_active());
        assert!(phase.needs_scene_color());
        phase.set_enabled(false);
        assert!(!phase.is_active());
        assert!(!phase.needs_scene_color());

        let mut msaa = UnderwaterPhase::new(4);
        msaa.set_enabled(true);
        msaa.set_active(true);
        assert!(!msaa.is_active(), "post effects switch off with MSAA");

        let mut snow = UnderwaterSnowPhase::new(1);
        snow.set_enabled(true);
        snow.set_active(true);
        assert!(snow.is_active());
        snow.set_settings(UnderwaterSettings {
            marine_snow: false,
            ..UnderwaterSettings::default()
        });
        assert!(!snow.is_active(), "no snow, no pass");
    }

    #[test]
    fn under_water_the_pack_never_sees_less_far_than_vanilla() {
        let settings = UnderwaterSettings {
            underwater_clarity: 8.0,
            red_fade: 1.0,
            ..UnderwaterSettings::default()
        }
        .sanitized();
        for (density, label) in [
            (0.1, "plain"),
            (0.07, "respiration 1"),
            (0.01, "water breathing"),
        ] {
            let frame = UnderwaterFrame {
                eye_in_water: true,
                vanilla_density: density,
                ..UnderwaterFrame::default()
            };
            let sigma = absorption(&settings, &frame);
            for d in [1.0f32, 5.0, 10.0, 20.0, 40.0, 100.0] {
                let vanilla = (-density * d).exp();
                let travelled = (d - settings.clear_range).max(0.0);
                for c in [sigma.x, sigma.y, sigma.z] {
                    assert!(
                        (-c * travelled).exp() >= vanilla - 1e-6,
                        "{label}: channel {c} at {d} blocks"
                    );
                }
            }
            let params = WorldParams::new(&settings, &frame, 0, (1, 1));
            assert!(
                params.sigma[3] >= 1.0e8,
                "no far fade under water ({label})"
            );
            assert_eq!(params.depth[1], 0.0, "no depth darkening under water");
        }
    }

    #[test]
    fn red_fades_first_and_clarity_sets_green() {
        let settings = UnderwaterSettings::default();
        let frame = UnderwaterFrame::default();
        let sigma = absorption(&settings, &frame);
        assert!(sigma.x > sigma.y && sigma.y > sigma.z);
        assert!((sigma.y - 1.0 / settings.clarity).abs() < 1e-6);
        let flat = absorption(
            &UnderwaterSettings {
                red_fade: 0.0,
                ..settings
            },
            &frame,
        );
        assert_eq!(flat.x, flat.y);
        assert_eq!(flat.y, flat.z);
    }

    #[test]
    fn the_default_keeps_players_on_the_next_island_visible() {
        // A player 60 blocks away keeps more than half of its green and blue contrast.
        let settings = UnderwaterSettings::default();
        let sigma = absorption(&settings, &UnderwaterFrame::default());
        let travelled = 60.0 - settings.clear_range;
        assert!((-sigma.y * travelled).exp() > 0.6);
        assert!((-sigma.z * travelled).exp() > 0.6);
        assert!((-sigma.x * travelled).exp() > 0.3, "red stays readable");
        // Inside the clear range nothing changes at all.
        assert_eq!(
            (-sigma.x * (5.0f32 - settings.clear_range).max(0.0)).exp(),
            1.0
        );
    }

    /// Why the default water is a mid-bright turquoise and not a deep blue: what keeps a player
    /// readable at range is the brightness difference to the water behind it, and a background
    /// in the middle of the range keeps bright armour (brighter than it) and dark skins (darker)
    /// both apart from it. At 30 blocks every kind of player keeps at least 0.15 of luminance
    /// contrast, at 60 at least 0.12.
    #[test]
    fn the_default_water_keeps_bright_and_dark_players_readable() {
        let settings = UnderwaterSettings::default();
        let frame = UnderwaterFrame::default();
        let glow = water_glow(&settings, &frame);
        let sigma = absorption(&settings, &frame);
        let luma = |c: Vec3| c.dot(Vec3::new(0.2126, 0.7152, 0.0722));
        let players = [
            ("diamond armour", 0xFF73_D9CF),
            ("iron armour", 0xFFC8_C8C8),
            ("dark skin", 0xFF5A_3C28),
            ("red team", 0xFFB0_2020),
            ("blue team", 0xFF20_30B0),
        ];
        for (distance, floor) in [(30.0f32, 0.15f32), (60.0, 0.12)] {
            let travelled = (distance - settings.clear_range).max(0.0);
            let trans = Vec3::new(
                (-sigma.x * travelled).exp(),
                (-sigma.y * travelled).exp(),
                (-sigma.z * travelled).exp(),
            );
            for (name, argb) in players {
                let own = to_linear(rgb_from_argb(argb));
                let seen = own * trans + glow * (Vec3::ONE - trans);
                let contrast = (luma(seen) - luma(glow)).abs();
                assert!(contrast >= floor, "{name} at {distance}: {contrast}");
            }
        }
    }

    #[test]
    fn the_light_follows_the_day() {
        let noon = WaterLight::new(&UnderwaterFrame::default());
        assert!(noon.air_dir.y > 0.99);
        assert!(noon.water_dir.y > 0.99);
        assert!(noon.strength > 0.99);
        assert!(noon.ambient > 0.99);
        // Sunset: the sun sits on the western horizon; under water the light is still steep.
        let sunset = WaterLight::new(&UnderwaterFrame {
            celestial_angle: 0.25,
            sun_brightness: 0.55,
            ..UnderwaterFrame::default()
        });
        assert!(sunset.air_dir.x < -0.99);
        let critical = (1.0 / WATER_ETA).asin();
        assert!(
            sunset.water_dir.y >= critical.cos() - 0.05,
            "{:?}",
            sunset.water_dir
        );
        assert!(sunset.color.z < sunset.color.x, "warm at sunset");
        // Midnight: the moon, weak and blue.
        let night = WaterLight::new(&UnderwaterFrame {
            celestial_angle: 0.5,
            sun_brightness: 0.2,
            ..UnderwaterFrame::default()
        });
        assert!(night.air_dir.y > 0.99, "the moon is overhead");
        assert!(night.strength < 0.2);
        assert!(night.ambient < 0.1);
        assert!(night.color.z > night.color.x);
    }

    #[test]
    fn wrapped_coordinates_stay_small_far_from_the_origin() {
        let frame = UnderwaterFrame {
            eye: DVec3::new(29_999_000.5, 70.0, -12_345_678.25),
            ..UnderwaterFrame::default()
        };
        let params = WorldParams::new(&UnderwaterSettings::default(), &frame, 0, (1, 1));
        for v in params.wrap {
            assert!((0.0..=1.0).contains(&v), "{v}");
        }
        let snow = SnowParams::new(&UnderwaterSettings::default(), &frame, 1080);
        for v in &snow.eye[..3] {
            assert!((0.0..SNOW_BOX as f32 + 1e-3).contains(v), "{v}");
        }
    }

    #[test]
    fn settings_are_sanitized() {
        let s = UnderwaterSettings {
            clarity: f32::NAN,
            caustic_strength: 9.0,
            water_color: [2.0, -1.0, f32::INFINITY],
            ..UnderwaterSettings::default()
        }
        .sanitized();
        assert_eq!(s.clarity, UnderwaterSettings::default().clarity);
        assert_eq!(s.caustic_strength, 1.0);
        assert_eq!(s.water_color, [1.0, 0.0, 0.5]);
    }

    // ------------------------------------------------------------------ GPU: a synthetic world

    use std::cell::RefCell;
    use std::rc::Rc;

    /// A phase that paints `underwater_test_scene.wgsl`'s world, colour and depth.
    struct TestScene {
        pipeline: wgpu::RenderPipeline,
    }

    impl TestScene {
        fn new(gpu: &Gpu) -> TestScene {
            let source = format!(
                "{}\n{}",
                view_geometry::COMMON_WGSL,
                include_str!("underwater_test_scene.wgsl")
            );
            let shader = gpu
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("underwater test scene"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            let layout = gpu
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("underwater test scene"),
                    bind_group_layouts: &[Some(gpu.camera_layout())],
                    immediate_size: 0,
                });
            let pipeline = gpu
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("underwater test scene"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: crate::DEPTH_FORMAT,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::Always),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_scene"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.config.format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                });
            TestScene { pipeline }
        }
    }

    impl RenderPhase for TestScene {
        fn name(&self) -> &'static str {
            "test scene"
        }
        fn record(
            &mut self,
            gpu: &Gpu,
            enc: &mut wgpu::CommandEncoder,
            view: &wgpu::TextureView,
            _cam: &Camera,
        ) {
            let mut pass = gpu.begin_pass(enc, view, "test scene", DepthAttachment::Shared);
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Lets the test reach a phase the `PhaseList` owns.
    struct Handle<T>(Rc<RefCell<T>>);

    impl<T: RenderPhase> RenderPhase for Handle<T> {
        fn name(&self) -> &'static str {
            self.0.borrow().name()
        }
        fn record(
            &mut self,
            gpu: &Gpu,
            enc: &mut wgpu::CommandEncoder,
            view: &wgpu::TextureView,
            cam: &Camera,
        ) {
            self.0.borrow_mut().record(gpu, enc, view, cam);
        }
        fn enabled(&self) -> bool {
            self.0.borrow().enabled()
        }
        fn needs_scene_color(&self) -> bool {
            self.0.borrow().needs_scene_color()
        }
        fn record_scene(
            &mut self,
            gpu: &Gpu,
            enc: &mut wgpu::CommandEncoder,
            view: &wgpu::TextureView,
            cam: &Camera,
            scene: &SceneTextures<'_>,
        ) {
            self.0.borrow_mut().record_scene(gpu, enc, view, cam, scene);
        }
    }

    struct Synthetic {
        gpu: Gpu,
        phases: crate::PhaseList,
        world: Rc<RefCell<UnderwaterPhase>>,
        snow: Rc<RefCell<UnderwaterSnowPhase>>,
        cam: Camera,
        width: u32,
        height: u32,
    }

    impl Synthetic {
        fn new(width: u32, height: u32) -> Option<Synthetic> {
            let gpu = match Gpu::headless(
                width,
                height,
                crate::GpuOptions {
                    vsync: false,
                    sample_count: 1,
                    low_power: false,
                },
            ) {
                Ok(gpu) => gpu,
                Err(error) => {
                    eprintln!("SKIP: no GPU ({error})");
                    return None;
                }
            };
            let world = Rc::new(RefCell::new(UnderwaterPhase::new(1)));
            let snow = Rc::new(RefCell::new(UnderwaterSnowPhase::new(1)));
            let mut phases = crate::PhaseList::new();
            phases.insert(PhaseOrder::SKY, Box::new(TestScene::new(&gpu)));
            phases.insert(WORLD_ORDER, Box::new(Handle(world.clone())));
            phases.insert(SNOW_ORDER, Box::new(Handle(snow.clone())));
            let mut cam = Camera::default();
            cam.set_viewport(width, height);
            cam.pos = DVec3::new(-3891.5, 64.62, 2059.5);
            cam.yaw = 180.0;
            cam.pitch = 6.0;
            cam.update();
            Some(Synthetic {
                gpu,
                phases,
                world,
                snow,
                cam,
                width,
                height,
            })
        }

        fn frame(&self) -> UnderwaterFrame {
            UnderwaterFrame {
                eye: self.cam.pos,
                surface_height: 24.0,
                wave_clock: 3.0,
                ..UnderwaterFrame::default()
            }
        }

        fn set(&self, on: bool, settings: UnderwaterSettings, frame: UnderwaterFrame) {
            let mut world = self.world.borrow_mut();
            world.set_enabled(on);
            world.set_active(on);
            world.set_settings(settings);
            world.set_frame(frame);
            let mut snow = self.snow.borrow_mut();
            snow.set_enabled(on);
            snow.set_active(on);
            snow.set_settings(settings);
            snow.set_frame(frame);
        }

        fn render(&mut self) -> Vec<u8> {
            let frame = self.gpu.acquire().expect("acquire");
            self.phases.render(&self.gpu, frame.view(), &self.cam, 0.0);
            let pixels = frame.read_pixels(&self.gpu).expect("read back");
            frame.present(&self.gpu.queue);
            pixels
        }

        /// The pixel a camera-relative point lands on.
        fn pixel(&self, pixels: &[u8], rel: Vec3) -> [f32; 3] {
            let clip = self.cam.view_proj() * rel.extend(1.0);
            let ndc = clip.truncate() / clip.w;
            let x = ((ndc.x * 0.5 + 0.5) * self.width as f32) as usize;
            let y = ((0.5 - ndc.y * 0.5) * self.height as f32) as usize;
            let i = (y.min(self.height as usize - 1) * self.width as usize
                + x.min(self.width as usize - 1))
                * 4;
            [
                f32::from(pixels[i]) / 255.0,
                f32::from(pixels[i + 1]) / 255.0,
                f32::from(pixels[i + 2]) / 255.0,
            ]
        }

        fn save(&self, pixels: &[u8], name: &str) {
            if let Some(dir) = std::env::var_os("MC_TW_SC_PNG_DIR") {
                let path = std::path::Path::new(&dir).join(format!("{name}.png"));
                image::save_buffer(
                    &path,
                    pixels,
                    self.width,
                    self.height,
                    image::ColorType::Rgba8,
                )
                .expect("write png");
                eprintln!("wrote {}", path.display());
            }
        }
    }

    fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    /// The pack on a synthetic island world through a real `PhaseList` (scene colour, depth,
    /// both phases): it changes the sky and the distance, keeps the red player at 5 blocks red
    /// and the blue one at 30 blue, and switching it off gives the untouched frame back, byte
    /// for byte.
    #[test]
    fn the_pack_on_a_synthetic_world_keeps_players_readable_and_off_is_exact() {
        let Some(mut s) = Synthetic::new(320, 180) else {
            return;
        };
        let defaults = UnderwaterSettings::default();
        s.set(false, defaults, s.frame());
        let off = s.render();
        s.world.borrow_mut().wait_for_caustics();
        s.set(true, defaults, s.frame());
        s.render();
        let on = s.render();
        assert!(s.world.borrow().caustics_ready());
        assert_eq!(s.world.borrow().passes_recorded(), 2);
        assert!(s.snow.borrow().passes_recorded() >= 2);

        let red = Vec3::new(0.0, -0.9, -5.0);
        let blue = Vec3::new(5.0, -0.9, -30.0);
        let near_ground = Vec3::new(1.5, -1.62, -4.0);
        let far_ground = Vec3::new(-6.0, -1.62, -36.0);
        let sky = Vec3::new(0.0, 30.0, -100.0);
        let (r_off, r_on) = (s.pixel(&off, red), s.pixel(&on, red));
        let (b_on, sky_off, sky_on) = (s.pixel(&on, blue), s.pixel(&off, sky), s.pixel(&on, sky));
        assert!(
            r_off[0] > r_off[1] && r_off[0] > r_off[2],
            "scene sanity {r_off:?}"
        );
        assert!(
            r_on[0] > r_on[1] + 0.1 && r_on[0] > r_on[2] + 0.1,
            "the red player at 5 blocks stays red: {r_on:?}"
        );
        assert!(
            b_on[2] > b_on[0] + 0.1 && b_on[2] > b_on[1],
            "the blue player at 30 blocks stays blue: {b_on:?}"
        );
        assert!(
            distance(sky_off, sky_on) > 0.05,
            "the sky becomes the water surface"
        );
        // Absorption alone (no caustics, shafts or snow): the farther ground is closer to the
        // water's colour, and the ground inside the clear range does not change at all.
        let plain = UnderwaterSettings {
            caustics: false,
            light_shafts: false,
            marine_snow: false,
            ..defaults
        };
        s.set(true, plain, s.frame());
        let absorbed = s.render();
        let water = to_display(to_linear(defaults.water_color));
        let water = [water.x, water.y, water.z];
        let near = distance(s.pixel(&absorbed, near_ground), water);
        let far = distance(s.pixel(&absorbed, far_ground), water);
        assert!(
            far < near,
            "far ground fades toward the water ({far} vs {near})"
        );
        assert!(
            distance(s.pixel(&absorbed, near_ground), s.pixel(&off, near_ground)) < 0.01,
            "nothing changes inside the clear range"
        );

        // Inactive (Only under water, eyes in air) and switched off: the untouched frame.
        s.world.borrow_mut().set_active(false);
        s.snow.borrow_mut().set_active(false);
        assert_eq!(s.render(), off, "inactive is byte-identical to off");
        s.set(false, defaults, s.frame());
        assert_eq!(s.render(), off, "off is byte-identical to never on");
        assert!(!s.world.borrow().allocated() && !s.snow.borrow().allocated());
    }

    /// The light-occlusion map at work: with a roof's footprint marked in it, the ground under
    /// the roof gets no caustics (byte-identical to caustics off), the open ground beside it
    /// still does, and without the map the shaded ground would have them too.
    #[test]
    fn the_occlusion_map_takes_the_caustics_out_of_a_roofs_shadow() {
        let Some(mut s) = Synthetic::new(320, 180) else {
            return;
        };
        s.world.borrow_mut().wait_for_caustics();
        let caustics_only = UnderwaterSettings {
            light_shafts: false,
            marine_snow: false,
            ..UnderwaterSettings::default()
        };
        let plain = UnderwaterSettings {
            caustics: false,
            ..caustics_only
        };
        // The island's ground is world y 63 everywhere; the roof (y 65..66) covers x -3889..-3885,
        // z 2050..2054. The corner is the glue's: six chunks west and north of the eye's chunk.
        let origin = (-4000, 1952);
        {
            let mut world = s.world.borrow_mut();
            let map = world.occlusion_map();
            map.fill(63);
            for x in -3889..-3885 {
                for z in 2050..2054 {
                    let (col, row) = ((x - origin.0) as usize, (z - origin.1) as usize);
                    map[row * OCCLUSION_SIZE as usize + col] = 66;
                }
            }
            world.mark_occlusion_changed();
        }
        let masked = UnderwaterFrame {
            occlusion_origin: Some(origin),
            ..s.frame()
        };
        let unmasked = UnderwaterFrame {
            occlusion_origin: None,
            ..masked
        };
        let mut shoot = |settings, frame| {
            s.set(true, settings, frame);
            s.render();
            s.render()
        };
        let base = shoot(plain, masked);
        let lit = shoot(caustics_only, masked);
        let bare = shoot(caustics_only, unmasked);
        let patch = |px: &[u8], x0: f32| -> f32 {
            let mut sum = 0.0;
            for dx in [0.0, 1.0, 2.0] {
                for z in [-8.5, -7.5, -6.5] {
                    let p = Vec3::new(x0 + dx, -1.62, z);
                    sum += distance(s.pixel(px, p), s.pixel(&base, p));
                }
            }
            sum
        };
        let under_roof = (patch(&lit, 3.5), patch(&bare, 3.5));
        let open = patch(&lit, -3.5);
        assert_eq!(under_roof.0, 0.0, "no caustics in the roof's shadow");
        assert!(
            under_roof.1 > 0.05,
            "without the map they would be there: {under_roof:?}"
        );
        assert!(open > 0.05, "the open ground keeps its caustics: {open}");
    }

    /// The marine snow is really on screen at the default amount (a speck below a pixel would
    /// vanish), yet sparse: it changes a few hundred pixels of a 640x360 frame, never a big share.
    #[test]
    fn the_default_marine_snow_is_visible_but_sparse() {
        let Some(mut s) = Synthetic::new(640, 360) else {
            return;
        };
        s.world.borrow_mut().wait_for_caustics();
        // The shafts' jitter changes every frame; leave them out so only the snow differs.
        let defaults = UnderwaterSettings {
            light_shafts: false,
            ..UnderwaterSettings::default()
        };
        let no_snow = UnderwaterSettings {
            marine_snow: false,
            ..defaults
        };
        s.set(true, no_snow, s.frame());
        s.render();
        let without = s.render();
        s.set(true, defaults, s.frame());
        s.render();
        let with = s.render();
        let changed = without
            .chunks_exact(4)
            .zip(with.chunks_exact(4))
            .filter(|(a, b)| a[..3] != b[..3])
            .count();
        let total = 640 * 360;
        eprintln!("marine snow changed {changed} of {total} pixels");
        assert!(changed >= 60, "specks on screen: {changed} pixels");
        assert!(changed * 50 < total, "sparse: {changed} of {total} pixels");
    }

    /// GPU time of the pack's passes at 1080p on the synthetic world, from timestamp queries:
    /// `MC_GPU_TIMING=1 cargo test -p mc-render --lib synthetic_gpu_timing -- --ignored
    /// --nocapture`. The composite's cost does not depend on the scene; the shafts' does a
    /// little (the march stops at the geometry). Provisional while other work shares the GPU.
    #[test]
    #[ignore = "GPU timing run; needs MC_GPU_TIMING=1"]
    fn synthetic_gpu_timing() {
        if !crate::context::gpu_timing_requested() {
            eprintln!("SKIP: set MC_GPU_TIMING=1");
            return;
        }
        let Some(mut s) = Synthetic::new(1920, 1080) else {
            return;
        };
        s.world.borrow_mut().wait_for_caustics();
        use crate::pipeline::GpuScopeStats;
        let run = |s: &mut Synthetic, label: &str, on: bool, settings: UnderwaterSettings| {
            s.set(on, settings, s.frame());
            for _ in 0..30 {
                s.render();
            }
            let start = s.phases.gpu_timings();
            let frames = 300;
            for _ in 0..frames {
                s.render();
            }
            let end = s.phases.gpu_timings();
            let mean = |a: GpuScopeStats, b: GpuScopeStats| {
                (b.total_ns - a.total_ns) as f64 / (b.calls - a.calls).max(1) as f64 / 1.0e6
            };
            eprintln!(
                "[{label}] gpu.frame {:.4} ms",
                mean(start.frame_ns, end.frame_ns)
            );
            for (name, stats) in &end.scopes {
                let before = start
                    .scopes
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, s)| *s)
                    .unwrap_or_default();
                if stats.calls > before.calls {
                    eprintln!("[{label}]   {name:<36} {:.4} ms", mean(before, *stats));
                }
            }
        };
        let defaults = UnderwaterSettings::default();
        run(&mut s, "off", false, defaults);
        run(&mut s, "on (defaults)", true, defaults);
        let high = UnderwaterSettings {
            quality: UnderwaterQuality::High,
            ..defaults
        };
        run(&mut s, "on (High)", true, high);
        let low = UnderwaterSettings {
            quality: UnderwaterQuality::Low,
            marine_snow: false,
            ..defaults
        };
        run(&mut s, "on (Low, no snow)", true, low);
    }

    /// The synthetic world at 960x540 in several looks, for a person to look at:
    /// `MC_TW_SC_PNG_DIR=<dir> cargo test -p mc-render --lib synthetic_pictures -- --ignored`.
    #[test]
    #[ignore = "writes PNGs; run by hand"]
    fn synthetic_pictures() {
        if std::env::var_os("MC_TW_SC_PNG_DIR").is_none() {
            return;
        }
        let Some(mut s) = Synthetic::new(960, 540) else {
            return;
        };
        let defaults = UnderwaterSettings::default();
        s.set(false, defaults, s.frame());
        let off = s.render();
        s.save(&off, "synthetic_off");
        s.world.borrow_mut().wait_for_caustics();
        let looks = [
            ("noon", 0.0, 1.0),
            ("sunset", 0.235, 0.62),
            ("night", 0.5, 0.2),
        ];
        for (name, angle, brightness) in looks {
            let frame = UnderwaterFrame {
                celestial_angle: angle,
                sun_brightness: brightness,
                ..s.frame()
            };
            s.set(true, defaults, frame);
            s.render();
            let on = s.render();
            s.save(&on, &format!("synthetic_{name}"));
        }
        s.cam.pitch = -60.0;
        s.cam.update();
        s.set(true, defaults, s.frame());
        s.render();
        let up = s.render();
        s.save(&up, "synthetic_noon_up");
    }

    #[test]
    fn the_defaults_match_the_mod_descriptor() {
        // `UnderwaterSettings::default` is what a fresh install draws: keep it equal to the
        // option defaults the menu shows (mirrored here, since mc-render cannot see mc-mods).
        let d = UnderwaterSettings::default();
        assert_eq!(d.water_color, rgb_from_argb(0xFF2E_A9B4));
        assert_eq!(
            (d.clarity, d.clear_range, d.underwater_clarity),
            (128.0, 8.0, 48.0)
        );
        assert!(!d.refraction);
        assert_eq!(snow_count(&d), 1200);
    }
}
