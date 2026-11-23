//! [`PackFrame`]: the per-frame inputs every shader pack reads.
//!
//! The client glue (`mc-client/src/app/mods/tw_shaders.rs`) builds one [`PackFrame`] per frame
//! from the live session (sun and moon, weather, fog, eye, camera) and hands it to the shared
//! [`ShaderCore`](super::ShaderCore), which uploads its GPU mirror ([`PackFrameUniform`], WGSL
//! `PackFrame` in [`super::COMMON_WGSL`]) once per frame while a pack runs. Pack phases read the
//! CPU copy with [`ShaderCore::frame`](super::ShaderCore::frame) and bind the buffer from
//! [`ShaderCore::frame_buffer`](super::ShaderCore::frame_buffer).
//!
//! # Conventions
//!
//! * World space is Minecraft's: +X east, +Y up, +Z south. Directions are unit vectors pointing
//!   **toward** the light.
//! * Colours named `*_display` are vanilla's display-encoded values (what the 8-bit frame holds);
//!   colours without the suffix are linear light. Convert with `pk_to_linear` / `pk_to_display`.
//! * World-anchored noise must tile every 1024 blocks (or a divisor): the GPU only gets the eye
//!   position wrapped into `[0, 1024)` (`PackFrame.eye` in WGSL), which keeps `f32` exact.

use glam::{DVec3, Mat4, Vec2, Vec3, Vec4};

use crate::sky::environment::{Dimension, Environment};
use crate::sky::fog::{FogDistance, FogMode};

/// Which shader pack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackId {
    /// The traditional shader-pack look (`shaders_classic`).
    Classic,
    /// Atmospheric and dreamy (`shaders_dreamy`).
    Dreamy,
    /// Under the sea (`shaders_underwater`).
    Underwater,
}

impl PackId {
    /// Every pack, in the order exclusivity keeps one when several are on at load.
    pub const ALL: [PackId; 3] = [PackId::Classic, PackId::Dreamy, PackId::Underwater];

    /// The pack's mod id in the catalog and the settings file.
    #[must_use]
    pub const fn mod_id(self) -> &'static str {
        match self {
            PackId::Classic => "shaders_classic",
            PackId::Dreamy => "shaders_dreamy",
            PackId::Underwater => "shaders_underwater",
        }
    }
}

/// What the camera's eye is inside.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EyeMedium {
    /// Air (including a cloud).
    #[default]
    Air,
    /// A water block.
    Water,
    /// A lava block.
    Lava,
}

impl EyeMedium {
    fn code(self) -> f32 {
        match self {
            EyeMedium::Air => 0.0,
            EyeMedium::Water => 1.0,
            EyeMedium::Lava => 2.0,
        }
    }
}

/// The edge of the cell the GPU's eye position is wrapped into, in blocks.
pub const EYE_WRAP: f64 = 1024.0;

/// Everything a pack needs to know about this frame. `Copy`; read it with
/// [`ShaderCore::frame`](super::ShaderCore::frame).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PackFrame {
    /// Whether a world is on screen (a session past the terrain screen). Packs run only then.
    pub in_world: bool,
    /// Vanilla's `World.getCelestialAngle(partialTicks)`: 0 noon, ~0.25 sunset, 0.5 midnight,
    /// ~0.75 sunrise. Fixed at 0.5 in the Nether and 0 in the End.
    pub celestial_angle: f32,
    /// Unit vector toward the sun as the pack lights the world: where `sky/dome` draws it, turned
    /// by the running pack's sun-path tilt ([`super::CoreConfig::sun_tilt`]; none by default).
    pub sun_dir: Vec3,
    /// Unit vector toward the sun disc as the sky draws it (vanilla's path). Screen-space effects
    /// anchored on the visible sun (light rays, glints) use this one.
    pub sun_disc: Vec3,
    /// Unit vector toward the moon (`-sun_dir`).
    pub moon_dir: Vec3,
    /// Toward the light that casts shadows now: the sun while it is above the horizon, else the
    /// moon.
    pub light_dir: Vec3,
    /// `true` while [`Self::light_dir`] is the sun.
    pub light_is_sun: bool,
    /// How strong the direct light is, 0..1: fades to 0 as the light reaches the horizon (so the
    /// sun-to-moon switch never pops) and with rain and thunder. The moon's peak is
    /// [`MOON_STRENGTH`].
    pub light_strength: f32,
    /// 0 at night .. 1 by day, from the sun's height (smooth through dusk and dawn).
    pub day: f32,
    /// 0..1, strongest when the sun is low: the golden-hour factor (sunrise and sunset).
    pub golden: f32,
    /// World time of day in ticks, `0..24000` (+ partial ticks): 6000 noon, 18000 midnight.
    pub time_of_day: f32,
    /// Vanilla `World.getSunBrightness`: the sky-light dimming curve, 0.2..1.
    pub sun_brightness: f32,
    /// Vanilla `World.getStarBrightness`, 0..0.5.
    pub star_brightness: f32,
    /// Rain strength, 0..1.
    pub rain: f32,
    /// Thunder strength, 0..1.
    pub thunder: f32,
    /// The lightning flash, 0..1.
    pub lightning: f32,
    /// The dimension.
    pub dimension: Dimension,
    /// The camera's eye, absolute world position (f64).
    pub eye: DVec3,
    /// What the eye is inside.
    pub eye_medium: EyeMedium,
    /// Vanilla's fog colour this frame (also the clear colour), display-encoded.
    pub fog_color_display: Vec3,
    /// Vanilla's sky colour (`World.getSkyColor`), display-encoded.
    pub sky_color_display: Vec3,
    /// Vanilla's sunrise/sunset glow colour (rgb, display-encoded) and its strength (a); zero
    /// outside dawn and dusk.
    pub sunrise_display: Vec4,
    /// Vanilla's world fog falloff.
    pub fog: FogMode,
    /// Vanilla's fog distance metric.
    pub fog_distance: FogDistance,
    /// Render distance in blocks (`chunks * 16`).
    pub render_distance: f32,
    /// Seconds of render clock (monotonic, continuous; drives animation).
    pub seconds: f32,
    /// The frame's partial tick.
    pub partial_ticks: f32,
    /// Frames since the client started (for dithering patterns that rotate per frame).
    pub frame_index: u32,
    /// Rotation-only view matrix (`Camera::view`).
    pub view: Mat4,
    /// The projection without TAA jitter (`Camera::proj`).
    pub proj: Mat4,
    /// TAA's sub-pixel jitter this frame, in NDC units (zero without TAA).
    pub jitter: Vec2,
    /// Effective vertical field of view, degrees.
    pub fov: f32,
    /// Near and far planes.
    pub near: f32,
    /// Far plane.
    pub far: f32,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

impl Default for PackFrame {
    fn default() -> Self {
        PackFrame {
            in_world: false,
            celestial_angle: 0.0,
            sun_dir: Vec3::Y,
            sun_disc: Vec3::Y,
            moon_dir: -Vec3::Y,
            light_dir: Vec3::Y,
            light_is_sun: true,
            light_strength: 1.0,
            day: 1.0,
            golden: 0.0,
            time_of_day: 6000.0,
            sun_brightness: 1.0,
            star_brightness: 0.0,
            rain: 0.0,
            thunder: 0.0,
            lightning: 0.0,
            dimension: Dimension::Overworld,
            eye: DVec3::ZERO,
            eye_medium: EyeMedium::Air,
            fog_color_display: Vec3::new(0.75, 0.85, 1.0),
            sky_color_display: Vec3::new(0.47, 0.65, 1.0),
            sunrise_display: Vec4::ZERO,
            fog: FogMode::CLEAR,
            fog_distance: FogDistance::Planar,
            render_distance: 128.0,
            seconds: 0.0,
            partial_ticks: 0.0,
            frame_index: 0,
            view: Mat4::IDENTITY,
            proj: Mat4::IDENTITY,
            jitter: Vec2::ZERO,
            fov: 70.0,
            near: 0.05,
            far: 256.0,
            width: 1,
            height: 1,
        }
    }
}

/// The moon's direct light relative to the sun's.
pub const MOON_STRENGTH: f32 = 0.35;

/// The sun's direction for a celestial angle: `sky/dome`'s `Ry(-90°) * Rx(angle * 360°)` applied
/// to the sun quad's `(0, 100, 0)`. Noon is straight up; the sun sets in the west (-X).
#[must_use]
pub fn sun_direction(celestial_angle: f32) -> Vec3 {
    let theta = celestial_angle * std::f32::consts::TAU;
    Vec3::new(-theta.sin(), theta.cos(), 0.0)
}

/// The rotation that tilts the sun's path by `degrees` toward the south (+Z) about the east-west
/// axis; the same matrix `sky::dome::SkyPhase::celestial_tilt` applies to the sun and moon quads.
/// Sunrise stays exactly east and sunset exactly west.
#[must_use]
pub fn sun_path_tilt(degrees: f32) -> glam::Mat3 {
    glam::Mat3::from_rotation_x(degrees.to_radians())
}

/// The world-and-camera half of a [`PackFrame`], as the glue has it.
#[derive(Debug, Clone, Copy)]
pub struct FrameInputs<'a> {
    /// The sky environment (time, weather, dimension), as `sky::fog`'s phase holds it.
    pub environment: &'a Environment,
    /// The biome temperature at the view entity's feet (for vanilla's sky colour).
    pub biome_temperature: f32,
    /// The final fog colour (`FogState::color`).
    pub fog_color: Vec3,
    /// The fog falloff (`FogState::mode`).
    pub fog: FogMode,
    /// The fog metric.
    pub fog_distance: FogDistance,
    /// What the eye is in.
    pub eye_medium: EyeMedium,
    /// Render distance in chunks.
    pub render_distance_chunks: u32,
    /// The camera.
    pub camera: &'a crate::Camera,
    /// Frame size in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// The frame's partial tick.
    pub partial_ticks: f32,
    /// The world clock is held: the Time Changer shows a fixed time, or the server stopped the
    /// daylight cycle. The sun and moon then stand still instead of following the partial tick.
    pub clock_held: bool,
    /// Render-clock seconds.
    pub seconds: f32,
    /// Frame counter.
    pub frame_index: u32,
    /// Whether a world is on screen.
    pub in_world: bool,
    /// The running pack's sun-path tilt, degrees ([`super::CoreConfig::sun_tilt`]).
    pub sun_tilt: f32,
}

impl PackFrame {
    /// Build a frame from the live inputs.
    #[must_use]
    pub fn compute(inputs: &FrameInputs<'_>) -> PackFrame {
        let env = inputs.environment;
        let pt = inputs.partial_ticks;
        // A held clock's sun stays where the clock holds it: vanilla's angle adds the partial
        // tick to a time that never advances, which swings the sun back and forth by one tick
        // twenty times a second (invisible on vanilla's sky, but shadows, light shafts and rays
        // follow it).
        let sun_pt = if inputs.clock_held && super::shadow::steady_shading() {
            0.0
        } else {
            pt
        };
        let celestial_angle = env.celestial_angle(sun_pt);
        let sun_disc = sun_direction(celestial_angle);
        let mut sun_dir = sun_disc;
        if inputs.sun_tilt != 0.0 && env.dimension == Dimension::Overworld {
            sun_dir = (sun_path_tilt(inputs.sun_tilt) * sun_dir).normalize_or(sun_dir);
        }
        let moon_dir = -sun_dir;
        let light_is_sun = sun_dir.y >= 0.0;
        let light_dir = if light_is_sun { sun_dir } else { moon_dir };
        let weather = 1.0 - 0.85 * env.weather.rain_strength.clamp(0.0, 1.0);
        let weather = weather * (1.0 - 0.5 * env.weather.thunder_strength.clamp(0.0, 1.0));
        let horizon_fade = smoothstep(0.0, 0.12, light_dir.y);
        let light_strength = match env.dimension {
            Dimension::Overworld => {
                horizon_fade * weather * if light_is_sun { 1.0 } else { MOON_STRENGTH }
            }
            Dimension::Nether | Dimension::End => 0.0,
        };
        let day = match env.dimension {
            Dimension::Overworld => smoothstep(-0.18, 0.18, sun_dir.y),
            Dimension::Nether | Dimension::End => 0.0,
        };
        let golden = match env.dimension {
            Dimension::Overworld => {
                // Strongest with the sun just above the horizon, gone by ~35 degrees up and a
                // little below it.
                let up = 1.0 - smoothstep(0.02, 0.55, sun_dir.y.abs());
                up * up
            }
            Dimension::Nether | Dimension::End => 0.0,
        };
        let sunrise_display = env
            .sunrise_sunset_color(pt)
            .map_or(Vec4::ZERO, |[r, g, b, a]| Vec4::new(r, g, b, a));
        let cam = inputs.camera;
        PackFrame {
            in_world: inputs.in_world,
            celestial_angle,
            sun_dir,
            sun_disc,
            moon_dir,
            light_dir,
            light_is_sun,
            light_strength,
            day,
            golden,
            time_of_day: (env.world_time.rem_euclid(24_000)) as f32 + pt,
            sun_brightness: env.sun_brightness(pt),
            star_brightness: env.star_brightness(pt),
            rain: env.weather.rain_strength,
            thunder: env.weather.thunder_strength,
            lightning: env.lightning_flash(pt),
            dimension: env.dimension,
            eye: cam.pos,
            eye_medium: inputs.eye_medium,
            fog_color_display: inputs.fog_color,
            sky_color_display: env.sky_color(inputs.biome_temperature, pt),
            sunrise_display,
            fog: inputs.fog,
            fog_distance: inputs.fog_distance,
            render_distance: inputs.render_distance_chunks as f32 * 16.0,
            seconds: inputs.seconds,
            partial_ticks: pt,
            frame_index: inputs.frame_index,
            view: cam.view,
            proj: cam.proj,
            jitter: cam.jitter,
            fov: cam.effective_fov(),
            near: cam.near,
            far: cam.far,
            width: inputs.width.max(1),
            height: inputs.height.max(1),
        }
    }

    /// The eye wrapped into `[0, EYE_WRAP)` per axis, as the GPU sees it.
    #[must_use]
    pub fn eye_wrapped(&self) -> Vec3 {
        let wrap = |v: f64| (v - (v / EYE_WRAP).floor() * EYE_WRAP) as f32;
        Vec3::new(wrap(self.eye.x), wrap(self.eye.y), wrap(self.eye.z))
    }

    /// Where the sun disc is on screen ([`Self::sun_disc`]): UV (0..1, y down) and whether it is
    /// in front of the camera (`z` = 1) or behind (`z` = 0). Uses the unjittered projection.
    #[must_use]
    pub fn sun_screen(&self) -> Vec3 {
        project_direction(self.view, self.proj, self.sun_disc)
    }

    /// [`Self::sun_screen`] for the moon disc.
    #[must_use]
    pub fn moon_screen(&self) -> Vec3 {
        project_direction(self.view, self.proj, -self.sun_disc)
    }

    /// The GPU mirror. `light` gives the direct and ambient colours (linear).
    #[must_use]
    pub fn uniform(&self, light: &LightColors) -> PackFrameUniform {
        let fog_mode = match self.fog {
            FogMode::Linear { start, end } => [start, end, 0.0, 0.0],
            FogMode::Exp { density } => [0.0, 0.0, density, 1.0],
        };
        let radial = match self.fog_distance {
            FogDistance::Planar => 0.0,
            FogDistance::Radial => 1.0,
        };
        let dimension = match self.dimension {
            Dimension::Overworld => 0.0,
            Dimension::Nether => -1.0,
            Dimension::End => 1.0,
        };
        let eye = self.eye_wrapped();
        let sun = self.sun_screen();
        let moon = self.moon_screen();
        let w = self.width as f32;
        let h = self.height as f32;
        PackFrameUniform {
            sun_dir: [
                self.sun_dir.x,
                self.sun_dir.y,
                self.sun_dir.z,
                self.celestial_angle,
            ],
            light_dir: [
                self.light_dir.x,
                self.light_dir.y,
                self.light_dir.z,
                self.light_strength,
            ],
            sun_color: [light.direct.x, light.direct.y, light.direct.z, self.day],
            ambient_color: [
                light.ambient.x,
                light.ambient.y,
                light.ambient.z,
                self.golden,
            ],
            fog_color: [
                self.fog_color_display.x,
                self.fog_color_display.y,
                self.fog_color_display.z,
                fog_mode[3],
            ],
            fog_params: [fog_mode[0], fog_mode[1], fog_mode[2], radial],
            sky_color: [
                self.sky_color_display.x,
                self.sky_color_display.y,
                self.sky_color_display.z,
                self.rain,
            ],
            eye: [eye.x, eye.y, eye.z, self.eye.y as f32],
            time: [self.seconds, self.time_of_day, self.thunder, self.lightning],
            misc: [
                self.eye_medium.code(),
                dimension,
                self.sun_brightness,
                self.star_brightness,
            ],
            screen: [w, h, 1.0 / w, 1.0 / h],
            sun_screen: [
                sun.x,
                sun.y,
                sun.z,
                if self.light_is_sun { 1.0 } else { 0.0 },
            ],
            moon_screen: [moon.x, moon.y, moon.z, self.render_distance],
            sunrise: self.sunrise_display.to_array(),
            sun_disc: [self.sun_disc.x, self.sun_disc.y, self.sun_disc.z, 0.0],
        }
    }
}

fn project_direction(view: Mat4, proj: Mat4, dir: Vec3) -> Vec3 {
    let v = view * dir.extend(0.0);
    if v.z >= -1e-4 {
        return Vec3::new(0.5, 0.5, 0.0);
    }
    let clip = proj * Vec4::new(v.x, v.y, v.z, 1.0);
    let ndc = clip.truncate() / clip.w;
    Vec3::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, 1.0)
}

/// The direct (sun or moon) and ambient (sky) light colours the core derives from the time of
/// day, linear. Unit luminance at full strength: they tint and scale vanilla's own light levels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightColors {
    /// Direct light: warm by day, golden at sunrise and sunset, blue under the moon; times
    /// [`PackFrame::light_strength`].
    pub direct: Vec3,
    /// Sky light: a cool tint, warmer at golden hour, blue at night.
    pub ambient: Vec3,
}

impl LightColors {
    /// White light: what a pack without sun shading gets.
    pub const NEUTRAL: LightColors = LightColors {
        direct: Vec3::ONE,
        ambient: Vec3::ONE,
    };

    /// The colours for a frame. `warmth` scales the golden-hour colour (0 = none, 1 = default),
    /// `night_blue` the moonlit tint (0 = grey, 1 = default).
    #[must_use]
    pub fn compute(frame: &PackFrame, warmth: f32, night_blue: f32) -> LightColors {
        if frame.dimension != Dimension::Overworld {
            // No sun, no moon, no sky dome: vanilla's own light, untinted.
            return LightColors {
                direct: Vec3::ZERO,
                ambient: Vec3::ONE,
            };
        }
        let warmth = warmth.clamp(0.0, 2.0);
        let night_blue = night_blue.clamp(0.0, 2.0);
        let noon = Vec3::new(1.0, 0.93, 0.80);
        let gold = Vec3::new(1.0, 0.62, 0.32);
        let moon = Vec3::new(0.62, 0.74, 1.0);
        let direct = if frame.light_is_sun {
            let g = (frame.golden * warmth).clamp(0.0, 1.0);
            normalize_luma(noon.lerp(gold, g))
        } else {
            normalize_luma(Vec3::ONE.lerp(moon, night_blue.min(1.0)))
        };
        let sky_day = Vec3::new(0.74, 0.88, 1.20);
        let sky_gold = Vec3::new(1.04, 0.90, 0.84);
        let sky_night = Vec3::new(0.70, 0.80, 1.10);
        let day_sky = sky_day.lerp(sky_gold, (frame.golden * warmth).clamp(0.0, 1.0));
        let night_sky = Vec3::ONE.lerp(sky_night, night_blue.min(1.0));
        let ambient = normalize_luma(night_sky.lerp(day_sky, frame.day));
        LightColors {
            direct: direct * frame.light_strength,
            ambient,
        }
    }
}

fn normalize_luma(c: Vec3) -> Vec3 {
    let luma = c.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    if luma > 1e-6 {
        c / luma
    } else {
        Vec3::ONE
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// GPU mirror of [`PackFrame`]: WGSL `PackFrame` in [`super::COMMON_WGSL`]. 15 `vec4<f32>`s.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PackFrameUniform {
    /// xyz toward the sun, w celestial angle.
    pub sun_dir: [f32; 4],
    /// xyz toward the shadow-casting light, w its strength.
    pub light_dir: [f32; 4],
    /// rgb direct light colour (linear, times strength), w day factor.
    pub sun_color: [f32; 4],
    /// rgb ambient sky tint (linear), w golden-hour factor.
    pub ambient_color: [f32; 4],
    /// rgb vanilla fog colour (display), w 0 linear / 1 exp.
    pub fog_color: [f32; 4],
    /// x start, y end, z density, w 1 = radial.
    pub fog_params: [f32; 4],
    /// rgb vanilla sky colour (display), w rain.
    pub sky_color: [f32; 4],
    /// xyz eye wrapped to `[0, 1024)`, w absolute eye y.
    pub eye: [f32; 4],
    /// x seconds, y time of day (ticks), z thunder, w lightning.
    pub time: [f32; 4],
    /// x medium (0 air, 1 water, 2 lava), y dimension (0, -1 nether, 1 end), z sun brightness,
    /// w star brightness.
    pub misc: [f32; 4],
    /// x width, y height, z 1/width, w 1/height.
    pub screen: [f32; 4],
    /// xy sun UV (y down), z 1 = in front of the camera, w 1 = the shadow light is the sun.
    pub sun_screen: [f32; 4],
    /// xy moon UV, z 1 = in front, w render distance in blocks.
    pub moon_screen: [f32; 4],
    /// rgb vanilla sunrise/sunset glow (display), a its strength.
    pub sunrise: [f32; 4],
    /// xyz toward the sun disc as drawn (vanilla's path; `sun_dir` may be tilted), w unused.
    pub sun_disc: [f32; 4],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sun_rises_in_the_east_and_sets_in_the_west() {
        let noon = sun_direction(0.0);
        assert!((noon - Vec3::Y).length() < 1e-6, "{noon}");
        let sunset = sun_direction(0.25);
        assert!(sunset.x < -0.99, "sets in the west: {sunset}");
        let sunrise = sun_direction(0.75);
        assert!(sunrise.x > 0.99, "rises in the east: {sunrise}");
        let midnight = sun_direction(0.5);
        assert!(midnight.y < -0.99);
    }

    #[test]
    fn the_uniform_is_sixteen_byte_aligned() {
        assert_eq!(std::mem::size_of::<PackFrameUniform>() % 16, 0);
        assert_eq!(std::mem::size_of::<PackFrameUniform>(), 15 * 16);
    }

    #[test]
    fn the_light_switches_to_the_moon_without_a_pop() {
        let env_at = |ticks: i64| Environment {
            world_time: ticks,
            ..Environment::default()
        };
        let camera = crate::Camera::default();
        let frame_at = |ticks| {
            let env = env_at(ticks);
            PackFrame::compute(&FrameInputs {
                environment: &env,
                biome_temperature: 0.8,
                fog_color: Vec3::ONE,
                fog: FogMode::CLEAR,
                fog_distance: FogDistance::Planar,
                eye_medium: EyeMedium::Air,
                render_distance_chunks: 8,
                camera: &camera,
                width: 64,
                height: 64,
                partial_ticks: 0.0,
                clock_held: false,
                seconds: 0.0,
                frame_index: 0,
                in_world: true,
                sun_tilt: 0.0,
            })
        };
        let noon = frame_at(6000);
        assert!(noon.light_is_sun && noon.light_strength > 0.99);
        let midnight = frame_at(18000);
        assert!(!midnight.light_is_sun);
        assert!((midnight.light_strength - MOON_STRENGTH).abs() < 1e-3);
        // Around sunset the strength runs down to zero before the switch.
        let mut last = noon.light_strength;
        let mut switched = false;
        for t in (11_000..14_000).step_by(20) {
            let f = frame_at(t);
            if !f.light_is_sun && !switched {
                switched = true;
                assert!(last < 0.02, "strength before the switch: {last}");
                assert!(f.light_strength < 0.02, "after: {}", f.light_strength);
            }
            last = f.light_strength;
        }
        assert!(switched);
    }

    #[test]
    fn light_colours_are_warm_at_sunset_and_blue_at_night() {
        let sunset = PackFrame {
            golden: 1.0,
            ..PackFrame::default()
        };
        let gold = LightColors::compute(&sunset, 1.0, 1.0);
        assert!(gold.direct.x > gold.direct.z * 2.0);
        let night = PackFrame {
            golden: 0.0,
            light_is_sun: false,
            day: 0.0,
            ..PackFrame::default()
        };
        let night = LightColors::compute(&night, 1.0, 1.0);
        assert!(night.direct.z > night.direct.x);
        assert!(night.ambient.z > night.ambient.x);
    }
}
