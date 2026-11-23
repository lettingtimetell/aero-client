//! Fog colour and falloff. java: `EntityRenderer.updateFogColor` / `EntityRenderer.setupFog`
//! (lines 1763-2033 of `EntityRenderer.java`).
//!
//! [`FogState::compute`] is the pure, tested transcription of vanilla's fog *state* — colour,
//! mode, density/start/end. [`FogPhase`] is what puts it on the screen: a full-screen pass that
//! reads the shared depth buffer, reconstructs each fragment's eye-space distance, and blends
//! [`FogState::color`] over the frame by [`FogMode`]'s own falloff. See `fog.wgsl` for the maths
//! and for why one screen-space pass is the right rewrite of `GL_FOG` without fixed-function GL.
//!
//! # What this is not
//!
//! Vanilla fogs *per draw*, so a translucent surface is fogged at its own depth and again through
//! whatever is behind it. A single screen-space pass fogs only the nearest depth-writing surface,
//! so translucent geometry drawn with `depth_write_enabled: false` (water, stained glass, rain)
//! is fogged at the depth of whatever is behind it rather than at its own. Matching vanilla
//! exactly needs a fog term inside `terrain/**`'s, `entity/**`'s and `particle/**`'s own WGSL,
//! which this module does not own. Everything needed from this side is
//! public: [`FogState`], [`FogMode::factor`], and [`FogPhase`]'s uniform layout.
//!
//! Done for terrain: the
//! client registers this pass at [`FogPhase::BEFORE_TRANSLUCENT_ORDER`] and the TRANSLUCENT
//! terrain layer fogs itself per draw
//! ([`crate::terrain::phase::TerrainRenderer::set_translucent_fog`]), so water, stained glass and
//! ice are fogged at their own depth. Rain still takes this pass's value at the depth behind it.
//!
//! The other two GPU-side fog effects that were already real before [`FogPhase`] existed:
//!
//! * The sky dome ([`crate::sky::dome`]) paints [`FogState::color`] at the horizon, which is
//!   vanilla's own trick for far-plane fog — terrain beyond render distance fades into the sky,
//!   not into a separately-computed fog gradient.
//! * [`crate::post::overlay`] applies the water/lava/void *colour* as a full-screen tint when the
//!   camera is inside that medium.

use glam::Vec3;

use crate::sky::environment::Environment;
use crate::{Camera, DepthAttachment, Gpu, PhaseOrder, RenderPhase};

/// Tick-rate state behind vanilla's `EntityRenderer.fogColor1` / `fogColor2` pair.
///
/// `EntityRenderer.updateRenderer` snapshots `fogColor1` into `fogColor2`, then eases the
/// current value 10% toward the eye block's `World.getLightBrightness` value. The target has a
/// render-distance floor: at a short distance even a completely dark eye block retains a little
/// fog colour. `updateFogColor` samples the two values with `partialTicks` before it multiplies
/// the completed fog colour by that sample.
///
/// This deliberately stores only the pure renderer state. The client owns the live-world part:
/// it must query the *subtracted* light at the view entity's eye, turn that level through the
/// dimension's light-brightness table, and call [`Self::tick`] exactly once per client tick.
/// Keeping that query outside `mc-render` prevents a render phase from silently using raw
/// sky/block light or advancing this state once per frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FogLightSmoothing {
    previous: f32,
    current: f32,
}

impl Default for FogLightSmoothing {
    fn default() -> Self {
        // Java's two instance fields have the JVM's zero initial value. In particular, the first
        // rendered tick fades in from black rather than pretending the eye is already at full
        // light.
        Self {
            previous: 0.0,
            current: 0.0,
        }
    }
}

impl FogLightSmoothing {
    /// Vanilla's target before the 10%-per-tick ease.
    ///
    /// java: `f2 * (1 - f3) + f3`, where `f2` is `World.getLightBrightness(eyePos)` and
    /// `f3 = renderDistanceChunks / 32.0F`. Keep the raw expression rather than clamping it:
    /// `EntityRenderer.updateRenderer` does not clamp this intermediate, and Java accepts a
    /// value loaded directly from `options.txt` even when it lies outside the normal slider
    /// range. The client normally supplies 2..32, but this helper is the renderer's exact state
    /// transition, not the settings widget's validation policy.
    #[must_use]
    pub fn target(eye_light_brightness: f32, render_distance_chunks: u32) -> f32 {
        let distance_floor = render_distance_chunks as f32 / 32.0;
        eye_light_brightness * (1.0 - distance_floor) + distance_floor
    }

    /// Advance once for one client tick.
    ///
    /// This is deliberately not a frame method: advancing it per frame makes fog darken faster
    /// on high-refresh displays. The assignment order is material: `previous` receives the old
    /// `current` before the new 10% approach is calculated.
    pub fn tick(&mut self, eye_light_brightness: f32, render_distance_chunks: u32) {
        self.previous = self.current;
        let target = Self::target(eye_light_brightness, render_distance_chunks);
        self.current += (target - self.current) * 0.1;
    }

    /// `fogColor2 + (fogColor1 - fogColor2) * partialTicks`.
    #[must_use]
    pub fn sample(&self, partial_ticks: f32) -> f32 {
        self.previous + (self.current - self.previous) * partial_ticks
    }

    /// Most recent tick-rate value (`fogColor1`), useful for diagnostics and tests.
    #[must_use]
    pub const fn current(&self) -> f32 {
        self.current
    }
}

/// `GL_LINEAR` vs `GL_EXP` fog falloff. java: `GlStateManager.setFog` mode constants
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FogMode {
    /// Linear ramp between `start` and `end`, in blocks.
    Linear {
        /// Fully clear.
        start: f32,
        /// Fully fogged.
        end: f32,
    },
    /// Exponential: `factor = exp(-density * distance)`.
    Exp {
        /// java: `GL11.glFogf(GL_FOG_DENSITY, density)`
        density: f32,
    },
}

impl FogMode {
    /// No fog: a linear ramp that starts a billion blocks out (the Fog mod off, W4).
    ///
    /// It is an ordinary linear mode on purpose, so every consumer handles it with the maths it
    /// already has and none needs a special case: the screen-space pass, the clouds, terrain's
    /// per-draw term and [`Self::factor`] all compute `(end - d) / (end - start)` and clamp it
    /// to exactly `1.0` for anything the renderer draws (the farthest is the cloud layer's far
    /// plane, `4 * farPlaneDistance`, 2,048 blocks at 32 chunks). Both ends are exact `f32`s.
    /// [`FogPhase`] goes further and skips its pass outright for it.
    pub const CLEAR: Self = Self::Linear {
        start: 1.0e9,
        end: 2.0e9,
    };

    /// Whether this is [`Self::CLEAR`], the no-fog mode.
    #[must_use]
    pub fn is_clear(&self) -> bool {
        *self == Self::CLEAR
    }

    /// `0.0` (fully fogged) `..1.0` (fully clear) at `distance` blocks from the camera.
    #[must_use]
    pub fn factor(&self, distance: f32) -> f32 {
        match *self {
            FogMode::Linear { start, end } => {
                if end <= start {
                    return 1.0;
                }
                ((end - distance) / (end - start)).clamp(0.0, 1.0)
            }
            FogMode::Exp { density } => (-density * distance).exp().clamp(0.0, 1.0),
        }
    }

    /// GPU parameters shared by phases that apply vanilla fog during their own fragment draw.
    /// The screen-space [`FogPhase`] packs the same values into its private uniform; clouds need
    /// the values before that later pass because their depth is intentionally reset afterward.
    pub(crate) fn cloud_uniform_params(self, distance: FogDistance) -> ([f32; 4], f32) {
        let distance_flag = distance.uniform_flag();
        match self {
            FogMode::Linear { start, end } => ([start, end, 0.0, distance_flag], 0.0),
            FogMode::Exp { density } => ([0.0, 0.0, density, distance_flag], 1.0),
        }
    }
}

/// Which eye-space distance metric the GPU fog pass uses.
///
/// Vanilla's fixed-function fog normally consumes the eye-space z coordinate, but the Java
/// reference on this machine enables `GL_NV_fog_distance` with `GL_EYE_RADIAL_NV`.  Keep the
/// metric explicit at the pass boundary so a parity capture can select radial distance without
/// changing the colour/falloff state or silently changing the existing analytical controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FogDistance {
    /// Absolute eye-space z (`-z_eye`), the portable/default metric.
    #[default]
    Planar,
    /// Full eye-to-fragment ray length reconstructed from depth and the projection matrix.
    Radial,
}

impl FogDistance {
    /// Value packed into `FogUniform.params.w` and consumed by `sky/fog.wgsl`.
    #[must_use]
    const fn uniform_flag(self) -> f32 {
        match self {
            Self::Planar => 0.0,
            Self::Radial => 1.0,
        }
    }
}

/// What medium the camera's eye is currently in — the branch `setupFog` and `updateFogColor`
/// switch on. java: `Material.water`/`Material.lava` checks plus the `cloudFog`/blindness flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CameraMedium {
    /// Ordinary air. java: the `else` branch of both methods.
    #[default]
    Air,
    /// Eye position is inside a cloud (Overworld only). java: `EntityRenderer.cloudFog`,
    /// set from `RenderGlobal.hasCloudFog`.
    Cloud,
    /// Eye position is inside a water block.
    Water,
    /// Eye position is inside a lava block.
    Lava,
}

/// Per-frame, camera-local inputs `FogState::compute` needs beyond [`Environment`].
#[derive(Debug, Clone, Copy)]
pub struct FogInputs {
    /// What the camera's eye is standing in.
    pub medium: CameraMedium,
    /// java: `GameSettings.renderDistanceChunks`.
    pub render_distance_chunks: u32,
    /// `sin(lookVec . sunHorizonVec)`-style dot product, `>0` when looking toward the
    /// rising/setting sun; drives the sunset tint blended into fog colour. `0.0` disables it
    /// (looking away from the sun, or render distance `<4` chunks per vanilla's own gate).
    pub sunset_look_dot: f32,
    /// java: `EnchantmentHelper.getRespiration(entity)`, `0..=3`.
    pub respiration_level: i32,
    /// java: `entity.isPotionActive(Potion.waterBreathing)`.
    pub water_breathing: bool,
    /// java: potion-effect-driven blindness radius shrink, `0.0` = not blinded, `1.0` = fully.
    pub blindness: f32,
    /// Camera eye Y at this frame's partial tick. Vanilla multiplies it by
    /// `WorldProvider::getVoidFogYFactor` before its below-horizon fog-colour blackout.
    /// This is a `f64` because the Java calculation remains in double precision until it
    /// multiplies the three float fog channels.
    pub interpolated_eye_y: f64,
    /// java: `WorldProvider::getVoidFogYFactor()`. Normal dimensions return `1.0 / 32.0`;
    /// a flat world supplies `1.0`, so the client must provide the active provider's value.
    pub void_fog_y_factor: f64,
    /// Remaining blindness-effect duration when it is active. This is deliberately separate
    /// from [`Self::blindness`]: `setupFog` needs the normalized radius value, while
    /// `updateFogColor` needs the original integer duration to black out the colour (and to
    /// ramp it during the final second). `None` means no blindness effect is active.
    pub blindness_duration_ticks: Option<i32>,
    /// The already-interpolated `EntityRenderer.bossColorModifier` sample, when vanilla's
    /// outer `bossColorModifier > 0.0F` guard was true.  `None` is materially different from a
    /// numeric zero: after the final decay tick Java skips this whole block, even though its
    /// previous value could still make an interpolation non-zero.
    pub boss_color_modifier: Option<f32>,
    /// `EntityRenderer.getNightVisionBrightness(entity, partialTicks)` when the effect is
    /// active. `None` means the effect is absent; `Some` must retain Java's un-clamped value
    /// (normally `0.4..=1.0`) because it is a blend factor, not a boolean.
    pub night_vision_brightness: Option<f32>,
    /// `world.provider.doesXZShowFog(x, z)` — the Nether's always-foggy-at-the-edges rule.
    pub dimension_always_fogs: bool,
    /// `EntityRenderer.fogColor1/2` interpolated: a slowly-smoothed light-level darkening term
    /// vanilla recomputes once per tick from `world.getLightBrightness(eyePos)`. `1.0` = no
    /// extra darkening.
    pub light_smoothing: f32,
    /// The eye column's biome temperature (height-adjusted), feeding
    /// [`Environment::sky_color`]'s biome tint — this is what makes fog "biome-tinted": fog
    /// colour is blended toward the sky colour (see [`FogState::compute`]), and the sky colour
    /// itself is a function of biome temperature via `BiomeGenBase.getSkyColorByTemp`. Vanilla's
    /// own default (a temperate plains-like value) is `0.8`.
    pub biome_temperature: f32,
    /// `GameSettings.anaglyph3d` / `EntityRenderer.updateFogColor`'s final colour transform.
    ///
    /// This is deliberately a frame input rather than a texture/upload concern: Java applies
    /// the weighted RGB mix after every other fog-colour stage and sends that completed colour
    /// both to `GlStateManager.clearColor` and `GL_FOG_COLOR`. A caller may change it between
    /// frames without rebuilding the phase or touching any resident resources.
    pub anaglyph: bool,
    /// The Fog mod's switch (`mc_mods` id `fog`); `true`, the
    /// default, is vanilla. `false` removes the render-distance fog from the world pass:
    /// [`world_pass_fog_mode`] turns `setupFog`'s air branch (the Nether/End `doesXZShowFog`
    /// variant included) into [`FogMode::CLEAR`], so terrain, entities, particles and clouds
    /// stay unfogged out to the render distance. Blindness, in-cloud, water and lava fog keep
    /// their vanilla falloff, and so does the sky pass ([`sky_pass_fog_mode`]): its horizon fade
    /// hides nothing, and without it the sky planes would end in hard seams. The fog *colour*
    /// (and with it the clear colour) never changes.
    pub distance_fog: bool,
    /// The Fog mod's "Clear water and lava" option (W4); `false`, the default, is vanilla.
    /// `true` gives an eye in water or lava the air falloff instead of their thick exponential
    /// fog, in both passes (none at all in the world pass while [`Self::distance_fog`] is off).
    /// The water and lava fog colours stay.
    pub clear_liquids: bool,
}

impl FogInputs {
    /// The medium whose falloff `setupFog` applies: [`Self::medium`], except that water and
    /// lava fall back to air under [`Self::clear_liquids`]. The fog colour keeps using
    /// [`Self::medium`].
    #[must_use]
    fn falloff_medium(&self) -> CameraMedium {
        match self.medium {
            CameraMedium::Water | CameraMedium::Lava if self.clear_liquids => CameraMedium::Air,
            medium => medium,
        }
    }
}

impl Default for FogInputs {
    fn default() -> Self {
        Self {
            medium: CameraMedium::Air,
            render_distance_chunks: 8,
            sunset_look_dot: 0.0,
            respiration_level: 0,
            water_breathing: false,
            blindness: 0.0,
            // A normal above-void camera location, so a default input leaves colour unchanged.
            interpolated_eye_y: 64.0,
            void_fog_y_factor: 1.0 / 32.0,
            blindness_duration_ticks: None,
            boss_color_modifier: None,
            night_vision_brightness: None,
            dimension_always_fogs: false,
            light_smoothing: 1.0,
            biome_temperature: 0.8,
            anaglyph: false,
            distance_fog: true,
            clear_liquids: false,
        }
    }
}

/// Colour + falloff for one frame's fog. java: the combined result of `updateFogColor` (colour)
/// and `setupFog` (mode/density/start/end).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FogState {
    /// Final blended fog colour, `0.0..1.0` per channel — also vanilla's screen clear colour.
    pub color: Vec3,
    /// Falloff mode and parameters for the main (non-sky-pass) render.
    pub mode: FogMode,
}

impl FogState {
    /// Compute this frame's fog. `far_plane` is `render_distance_chunks * 16`, matching
    /// [`crate::Camera::set_view_distance_chunks`]'s undiagonalised distance (vanilla's
    /// `farPlaneDistance`, before the `* sqrt(2)` the projection's far plane applies).
    #[must_use]
    pub fn compute(env: &Environment, inputs: FogInputs, partial_ticks: f32) -> Self {
        let sky = env.sky_color(inputs.biome_temperature, partial_ticks);
        let mut color = env.base_fog_color(partial_ticks);

        if inputs.sunset_look_dot > 0.0 {
            if let Some([r, g, b, a]) = env.sunrise_sunset_color(partial_ticks) {
                let f5 = inputs.sunset_look_dot * a;
                color = color * (1.0 - f5) + Vec3::new(r, g, b) * f5;
            }
        }

        // Blend toward sky colour by a render-distance-derived factor. Java evaluates the
        // exponent as `Math.pow((double)f, 0.25D)` and narrows only the result; using f32's
        // `powf` happens to agree for the normal 2..32 slider values but diverges for valid
        // renderer state loaded from a wider raw options value.
        let f = sky_blend_factor(inputs.render_distance_chunks);
        color += (sky - color) * f;

        let rain = env.weather.rain_strength;
        if rain > 0.0 {
            color.x *= 1.0 - rain * 0.5;
            color.y *= 1.0 - rain * 0.5;
            color.z *= 1.0 - rain * 0.4;
        }
        let thunder = env.weather.thunder_strength;
        if thunder > 0.0 {
            color *= 1.0 - thunder * 0.5;
        }

        match inputs.medium {
            CameraMedium::Cloud => color = env.cloud_color(partial_ticks),
            CameraMedium::Water => {
                let mut f12 = inputs.respiration_level as f32 * 0.2;
                if inputs.water_breathing {
                    f12 = f12 * 0.3 + 0.6;
                }
                color = Vec3::new(0.02 + f12, 0.02 + f12, 0.2 + f12);
            }
            CameraMedium::Lava => color = Vec3::new(0.6, 0.1, 0.0),
            CameraMedium::Air => {}
        }

        // java: `EntityRenderer.updateFogColor:1856-1860` — `f13` is applied **after** the
        // cloud/water/lava override, not before. Underwater and in lava the override is an
        // absolute colour that vanilla then darkens by the light term; multiplying first and
        // then overwriting throws the multiply away entirely.
        color *= inputs.light_smoothing;

        // java: `EntityRenderer.updateFogColor:1861-1883`. The void factor darkens the clear
        // colour below y=32 in ordinary worlds, and blindness forces it to black except during
        // its final 20 ticks, where the same duration ramp is applied. Keep the position/factor
        // supplied by the client: mc-render deliberately has no world-provider dependency.
        let mut void_factor = inputs.interpolated_eye_y * inputs.void_fog_y_factor;
        if let Some(duration) = inputs.blindness_duration_ticks {
            if duration < 20 {
                // Java evaluates `1.0F - (float)duration / 20.0F` as f32, then promotes that
                // result for the f64 `d1` multiply. Preserve that narrowing rather than doing
                // the seemingly equivalent division in f64.
                let fade = 1.0_f32 - duration as f32 / 20.0_f32;
                void_factor *= f64::from(fade);
            } else {
                void_factor = 0.0;
            }
        }
        color = apply_void_fog(color, void_factor);

        // java: `EntityRenderer.updateFogColor:1889-1897`.  This must remain before night
        // vision: a full-strength night-vision effect normalizes the *already boss-darkened*
        // colour, which is observably different from darkening a normalized colour afterward.
        if let Some(modifier) = inputs.boss_color_modifier {
            color = apply_boss_color_modifier(color, modifier);
        }

        // java: `EntityRenderer.updateFogColor:1899-1915`.  Do not simplify this into a clamp
        // or per-channel normalization: Java chooses the reciprocal of the largest channel, then
        // blends every channel toward that common scale.  Its placement after void fog also means
        // the f32 divides see the post-blackout channels.
        if let Some(brightness) = inputs.night_vision_brightness {
            color = apply_night_vision(color, brightness);
        }

        // java: `EntityRenderer.updateFogColor:1917-1924`. This is the final colour operation:
        // its result is immediately passed to both `GlStateManager.clearColor` and the fog
        // colour state. Keep it after weather, medium, light/void, boss and night-vision stages;
        // moving it earlier changes the channel weights whenever any of those stages is active.
        if inputs.anaglyph {
            color = apply_anaglyph(color);
        }

        Self {
            color,
            mode: world_pass_fog_mode(inputs),
        }
    }
}

/// Java's render-distance sky blend, including its f32/f64 boundary.
///
/// java: `EntityRenderer.updateFogColor:1767-1768`: the base `f` is assembled from float
/// operations, then `Math.pow((double)f, 0.25D)` runs in double precision and is narrowed before
/// the final float subtraction. Keeping this as a helper makes the precision boundary explicit
/// and gives the CPU seam a direct raw-bit oracle.
#[inline]
fn sky_blend_factor(render_distance_chunks: u32) -> f32 {
    let x = 0.25_f32 + 0.75_f32 * render_distance_chunks as f32 / 32.0_f32;
    1.0_f32 - (f64::from(x).powf(0.25_f64) as f32)
}

/// Apply Java's below-void colour blackout with its f64 multiply / f32 narrowing boundary.
///
/// Bytecode for `EntityRenderer.updateFogColor` (`bfk.i(F)`, offsets 827-870) squares `d1` as a
/// double, widens each fog channel with `f2d`, multiplies, then narrows with `d2f`.  Casting the
/// squared factor to `f32` first changes real colours by an ULP near the cave boundary.
fn apply_void_fog(mut color: Vec3, mut factor: f64) -> Vec3 {
    if factor < 1.0 {
        factor = factor.max(0.0);
        factor *= factor;
        color.x = (f64::from(color.x) * factor) as f32;
        color.y = (f64::from(color.y) * factor) as f32;
        color.z = (f64::from(color.z) * factor) as f32;
    }
    color
}

/// Java's three separate boss-colour expressions, preserving their f32 operation order.
fn apply_boss_color_modifier(mut color: Vec3, modifier: f32) -> Vec3 {
    color.x = color.x * (1.0 - modifier) + color.x * 0.7 * modifier;
    color.y = color.y * (1.0 - modifier) + color.y * 0.6 * modifier;
    color.z = color.z * (1.0 - modifier) + color.z * 0.6 * modifier;
    color
}

/// Java's common-scale night-vision fog brightening.
fn apply_night_vision(mut color: Vec3, brightness: f32) -> Vec3 {
    let mut scale = 1.0 / color.x;
    if scale > 1.0 / color.y {
        scale = 1.0 / color.y;
    }
    if scale > 1.0 / color.z {
        scale = 1.0 / color.z;
    }
    color.x = color.x * (1.0 - brightness) + color.x * scale * brightness;
    color.y = color.y * (1.0 - brightness) + color.y * scale * brightness;
    color.z = color.z * (1.0 - brightness) + color.z * scale * brightness;
    color
}

/// Java's final fog/clear-colour anaglyph transform.
///
/// java: `EntityRenderer.updateFogColor:1919-1924`. The coefficients are integer weights, but
/// the source channels are `float`, so each multiply/add/divide remains an f32 operation and
/// must not be replaced with the upload-time integer `TextureUtil.anaglyphColor` transform.
fn apply_anaglyph(color: Vec3) -> Vec3 {
    Vec3::new(
        (color.x * 30.0 + color.y * 59.0 + color.z * 11.0) / 100.0,
        (color.x * 30.0 + color.y * 70.0) / 100.0,
        (color.x * 30.0 + color.z * 70.0) / 100.0,
    )
}

/// The falloff the **world pass** runs with: the mode of [`FogState::compute`], consumed by
/// [`FogPhase`] (terrain, entities, particles, weather) and by the clouds.
///
/// java: `EntityRenderer.setupFog(0, partialTicks)` (`EntityRenderer.java:1936`), called before
/// the clouds, every terrain layer, entities, particles, rain and the world border
/// (`renderWorldPass:1361`-`:1500`). [`sky_pass_fog_mode`] is its `setupFog(-1, ..)` sibling.
///
/// This is the one place the Fog mod changes the falloff:
/// with [`FogInputs::distance_fog`] off the air branch, Nether/End variant included, is
/// [`FogMode::CLEAR`]; [`FogInputs::clear_liquids`] gives water and lava the air branch. With
/// both at their defaults this is vanilla's exact expression tree.
#[must_use]
pub fn world_pass_fog_mode(inputs: FogInputs) -> FogMode {
    let far = inputs.render_distance_chunks as f32 * 16.0;
    if inputs.blindness > 0.0 {
        let base = 5.0 + (far - 5.0) * (1.0 - inputs.blindness);
        return FogMode::Linear {
            start: base * 0.25,
            end: base,
        };
    }
    match inputs.falloff_medium() {
        CameraMedium::Cloud => FogMode::Exp { density: 0.1 },
        CameraMedium::Water => FogMode::Exp {
            density: if inputs.water_breathing {
                0.01
            } else {
                // java: `setFogDensity(0.1F - respiration * 0.03F)` has no lower clamp. Legal
                // vanilla enchantments stay non-negative, but retaining the raw expression
                // matters for over-level/custom NBT (and for the exact GL state a caller
                // observes) instead of silently inventing a saturation rule.
                0.1 - inputs.respiration_level as f32 * 0.03
            },
        },
        CameraMedium::Lava => FogMode::Exp { density: 2.0 },
        // The Fog mod is off: no render-distance fog in any dimension.
        CameraMedium::Air if !inputs.distance_fog => FogMode::CLEAR,
        CameraMedium::Air if inputs.dimension_always_fogs => FogMode::Linear {
            start: far * 0.05,
            end: far.min(192.0) * 0.5,
        },
        CameraMedium::Air => FogMode::Linear {
            start: far * 0.75,
            end: far,
        },
    }
}

/// The falloff the **sky pass** runs with, which is not the same as the world pass's.
///
/// java: `EntityRenderer.setupFog(-1, partialTicks)` (`EntityRenderer.java:1936`), called at
/// `EntityRenderer.java:1348` immediately before `RenderGlobal.renderSky`. `startCoords == -1`
/// changes only the two linear branches — the fog starts at the eye instead of at `0.75 * far`:
/// blinded, `0.0 .. radius * 0.8`; otherwise `0.0 .. farPlaneDistance`.
///
/// This is what fades vanilla's flat ±384-block sky plane into the fog colour as it approaches
/// the horizon, instead of leaving the sky one flat colour that stops dead at the plane's rim.
/// [`crate::sky::dome::SkyPhase`] applies it to the sky and horizon planes and to nothing else,
/// matching the `enableFog`/`disableFog` bracketing in `RenderGlobal.renderSky`
/// (`:1231`/`:1248`, then `:1349` again for the horizon plane) — the sun, moon, stars and
/// sunrise fan are all drawn inside the un-fogged window.
///
/// The Fog mod (W4) leaves this pass alone on purpose: [`FogInputs::distance_fog`] is ignored
/// here. The sky plane ends ~2.4° above the horizon and the horizon plane is a flat dark
/// colour, so without this fade the sky would show hard seams (in a void map, a flat dark-blue
/// floor meeting a bright band at the horizon), and nothing in the world is hidden by it.
/// [`FogInputs::clear_liquids`] does apply, so the sky seen from clear water fades like air.
#[must_use]
pub fn sky_pass_fog_mode(inputs: FogInputs) -> FogMode {
    let far = inputs.render_distance_chunks as f32 * 16.0;
    if inputs.blindness > 0.0 {
        let base = 5.0 + (far - 5.0) * (1.0 - inputs.blindness);
        return FogMode::Linear {
            start: 0.0,
            end: base * 0.8,
        };
    }
    match inputs.falloff_medium() {
        CameraMedium::Cloud => FogMode::Exp { density: 0.1 },
        CameraMedium::Water => FogMode::Exp {
            density: if inputs.water_breathing {
                0.01
            } else {
                // java: `setFogDensity(0.1F - respiration * 0.03F)` has no lower clamp;
                // preserve the same expression for custom over-level enchantments.
                0.1 - inputs.respiration_level as f32 * 0.03
            },
        },
        CameraMedium::Lava => FogMode::Exp { density: 2.0 },
        // java: `EntityRenderer.java:2023` — `doesXZShowFog` overrides both branches alike.
        CameraMedium::Air if inputs.dimension_always_fogs => FogMode::Linear {
            start: far * 0.05,
            end: far.min(192.0) * 0.5,
        },
        CameraMedium::Air => FogMode::Linear {
            start: 0.0,
            end: far,
        },
    }
}

// ---------------------------------------------------------------- the GPU pass

/// `std140`-compatible mirror of [`FogState`], matching `fog.wgsl`'s `FogUniform`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct FogUniform {
    /// `rgb` = [`FogState::color`], `a` = 0 for [`FogMode::Linear`], 1 for [`FogMode::Exp`].
    color: [f32; 4],
    /// `x` = linear start, `y` = linear end, `z` = exp density, `w` = [`FogDistance`] flag.
    params: [f32; 4],
}

impl FogUniform {
    fn from_state(state: &FogState, distance: FogDistance) -> Self {
        let (kind, params) = match state.mode {
            FogMode::Linear { start, end } => (0.0, [start, end, 0.0, distance.uniform_flag()]),
            FogMode::Exp { density } => (1.0, [0.0, 0.0, density, distance.uniform_flag()]),
        };
        Self {
            color: [state.color.x, state.color.y, state.color.z, kind],
            params,
        }
    }
}

/// Full-screen distance fog, driven by [`FogState::compute`] and the shared depth buffer.
///
/// Register it at [`FogPhase::ORDER`], between [`crate::PhaseOrder::TERRAIN_TRANSLUCENT`] and
/// [`crate::PhaseOrder::FIRST_PERSON`], so everything with depth (terrain, entities, particles,
/// weather) is behind it, while the hand and the screen overlays ([`crate::post::overlay`]) are
/// still in front:
///
/// ```no_run
/// # use mc_render::{Gpu, PhaseList};
/// # use mc_render::sky::fog::FogPhase;
/// # fn wire(gpu: &Gpu, phases: &mut PhaseList) {
/// phases.insert(FogPhase::ORDER, Box::new(FogPhase::new(gpu)));
/// # }
/// ```
///
/// [`FogPhase::environment`], [`FogPhase::inputs`] and [`FogPhase::partial_ticks`] are driven
/// from outside, the same externally-fed pattern every other S20 phase uses; the phase recomputes
/// [`FogState`] from them each frame rather than caching, since `compute` is a few dozen flops.
///
/// It does **not** opt into [`crate::SceneTextures`]: depth needs no snapshot, because this pass
/// binds the depth buffer as a texture while attaching no depth
/// ([`DepthAttachment::None`]), so the frame never reads and writes it at once. Registering a
/// `FogPhase` therefore costs nothing beyond its own pass.
pub struct FogPhase {
    /// Time of day, weather and dimension. Feeds [`FogState::compute`].
    pub environment: Environment,
    /// Camera-local fog inputs. Feeds [`FogState::compute`].
    pub inputs: FogInputs,
    /// Render interpolation, `0.0..1.0`.
    pub partial_ticks: f32,
    /// Draw nothing when `false`. Vanilla has no "fog off" option, so this
    /// exists for tests and for the F3 `chunk borders`-style debug toggles.
    pub visible: bool,
    /// Eye-space distance metric used by the screen-space fog shader.
    ///
    /// This is deliberately phase-local: Java's fog colour and falloff parameters are unchanged
    /// when `GL_NV_fog_distance` switches from planar to radial distance.  Keeping the selector
    /// here also avoids changing [`FogState`] literals used by the terrain per-draw fallback.
    pub distance: FogDistance,

    pipeline: wgpu::RenderPipeline,
    /// `pipeline` with a depth state (compare `Always`, no write), for the
    /// shared read-only-depth pass ([`RenderPhase::shares_depth_read_pass`]).
    depth_read_pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    uniform_write: crate::context::ChangedWrite,
    bind_group: wgpu::BindGroup,
    /// The depth view the current `bind_group` points at is invalidated by
    /// [`Gpu::resize`](crate::Gpu::resize), which rebuilds the depth texture.
    bound_size: (u32, u32),
}

impl FogPhase {
    /// Where to register this phase: after [`crate::PhaseOrder::WEATHER`] (450) and
    /// [`crate::PhaseOrder::TERRAIN_TRANSLUCENT`] (500), and strictly before
    /// [`crate::PhaseOrder::FIRST_PERSON`] (650).
    ///
    /// It must not share 650 with the hand: `PhaseList::insert` is stable, the client registers
    /// the hand first, and the hand pass opens with `DepthAttachment::Clear`. Recorded after it,
    /// this pass read a depth buffer of 1.0 everywhere except the hand and discarded every world
    /// pixel — no distance fog in first person, for the price of a full-screen pass. java:
    /// `EntityRenderer.renderWorldPass` fogs every world draw and only then calls
    /// `GlStateManager.disableFog()` + `clear(GL_DEPTH_BUFFER_BIT)` + `renderHand`, so the hand is
    /// never fogged and the world always is.
    pub const ORDER: PhaseOrder = PhaseOrder(640);

    /// Where the client registers its world fog: after [`crate::PhaseOrder::WEATHER`] (450), before
    /// [`crate::PhaseOrder::TERRAIN_TRANSLUCENT`] (500).
    ///
    /// Translucent terrain writes no depth, so a screen-space pass after it fogs water and stained
    /// glass at the depth of whatever is behind them (and not at all against the sky). Before it,
    /// this pass fogs exactly the depth-writing world, and the translucent layer fogs itself per
    /// draw at each fragment's own distance
    /// ([`crate::terrain::phase::TerrainRenderer::set_translucent_fog`]): vanilla's order, where
    /// `GL_FOG` is applied to each draw as it happens. Pixels with no translucent surface come
    /// out exactly as at [`Self::ORDER`]: the same pass over the same depth and colour.
    /// [`Self::ORDER`] stays as it was, because other phases place themselves relative to it.
    pub const BEFORE_TRANSLUCENT_ORDER: PhaseOrder = PhaseOrder(480);

    /// Build the phase with Java's final fog/clear-colour anaglyph transform selected.
    ///
    /// The flag is stored in [`Self::inputs`] and can be changed for a live frame with
    /// [`Self::set_anaglyph`]; no GPU resource rebuild is needed because the colour is part of the
    /// per-frame state/uniform.
    #[must_use]
    pub fn new_with_anaglyph(gpu: &Gpu, anaglyph: bool) -> Self {
        let mut phase = Self::new(gpu);
        phase.inputs.anaglyph = anaglyph;
        phase
    }

    /// Build the phase.
    #[must_use]
    pub fn new(gpu: &Gpu) -> Self {
        // One source, two depth-texture spellings: a multisampled depth buffer cannot be bound as
        // an ordinary `texture_depth_2d`. See `Gpu::depth_texture`'s docs.
        let source = if gpu.sample_count() > 1 {
            include_str!("fog.wgsl").replace("texture_depth_2d", "texture_depth_multisampled_2d")
        } else {
            include_str!("fog.wgsl").to_owned()
        };
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("sky fog"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });

        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sky fog group1"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: gpu.sample_count() > 1,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sky fog layout"),
                bind_group_layouts: &[Some(gpu.camera_layout()), Some(&layout)],
                immediate_size: 0,
            });

        let make_pipeline = |label: &str, depth_stencil: Option<wgpu::DepthStencilState>| {
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil,
                    multisample: wgpu::MultisampleState {
                        count: gpu.sample_count(),
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.config.format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        };
        let pipeline = make_pipeline("sky fog", None);
        // The same pipeline for the shared read-only-depth pass, where the
        // depth buffer is attached: a test that always passes and no depth write draw exactly the
        // fragments the attachment-less pipeline draws.
        let depth_read_pipeline = make_pipeline(
            "sky fog (read-only depth pass)",
            Some(wgpu::DepthStencilState {
                format: crate::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
        );

        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky fog uniform"),
            size: std::mem::size_of::<FogUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = Self::make_bind_group(gpu, &layout, &uniform);

        Self {
            environment: Environment::default(),
            inputs: FogInputs::default(),
            partial_ticks: 0.0,
            visible: true,
            distance: FogDistance::default(),
            pipeline,
            depth_read_pipeline,
            layout,
            uniform,
            uniform_write: crate::context::ChangedWrite::default(),
            bind_group,
            bound_size: (gpu.config.width, gpu.config.height),
        }
    }

    fn make_bind_group(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let depth = gpu
            .depth_texture()
            .create_view(&wgpu::TextureViewDescriptor::default());
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky fog group1"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth),
                },
            ],
        })
    }

    /// This frame's fog, as [`FogState::compute`] would return it for the current fields. Exposed
    /// so a caller can reuse it for the clear colour (vanilla clears to the fog colour) without
    /// recomputing.
    #[must_use]
    pub fn state(&self) -> FogState {
        FogState::compute(&self.environment, self.inputs, self.partial_ticks)
    }

    /// The active eye-space distance metric for the screen-space pass.
    #[must_use]
    pub const fn distance(&self) -> FogDistance {
        self.distance
    }

    /// Change the eye-space distance metric without rebuilding any GPU resources.
    ///
    /// The portable default remains [`FogDistance::Planar`].  A caller reproducing a Java run
    /// that reports `GL_EYE_RADIAL_NV` can opt into [`FogDistance::Radial`] for that frame/pass.
    pub fn set_distance(&mut self, distance: FogDistance) {
        self.distance = distance;
    }

    /// Whether the final Java fog/clear-colour anaglyph transform is selected.
    #[must_use]
    pub const fn anaglyph_enabled(&self) -> bool {
        self.inputs.anaglyph
    }

    /// Change the final Java fog/clear-colour anaglyph transform for subsequent frames.
    ///
    /// This setter is intentionally CPU-only and live-safe: unlike texture anaglyph toggles,
    /// fog has no resident resource bytes to replay.
    pub fn set_anaglyph(&mut self, enabled: bool) {
        self.inputs.anaglyph = enabled;
    }
}

impl RenderPhase for FogPhase {
    fn name(&self) -> &'static str {
        "sky/fog"
    }

    fn enabled(&self) -> bool {
        // With the Fog mod off (W4) an air eye has no world fog at all, and the pass would
        // discard every pixel: skip it. With the mod on this is the one bool test.
        self.visible && (self.inputs.distance_fog || !world_pass_fog_mode(self.inputs).is_clear())
    }

    fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        // `Gpu::resize` has already replaced the depth texture, so the cached bind group points
        // at a dead view.
        self.bind_group = Self::make_bind_group(gpu, &self.layout, &self.uniform);
        self.bound_size = (width, height);
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        self.prepare_frame(gpu);
        let mut pass = gpu.begin_pass(enc, view, "sky/fog", DepthAttachment::None);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    // The pass reads the depth buffer and writes none, so it can share one
    // read-only-depth pass with the translucent terrain right after it (`PhaseList`).
    fn shares_depth_read_pass(&self) -> bool {
        true
    }

    fn record_into_depth_read_pass(
        &mut self,
        gpu: &Gpu,
        pass: &mut wgpu::RenderPass<'static>,
        _cam: &Camera,
    ) {
        self.prepare_frame(gpu);
        pass.set_pipeline(&self.depth_read_pipeline);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

impl FogPhase {
    /// This frame's bind group and uniform, written before the draw (`record` and the shared
    /// read-only-depth pass alike).
    fn prepare_frame(&mut self, gpu: &Gpu) {
        if (gpu.config.width, gpu.config.height) != self.bound_size {
            // A caller that resized the `Gpu` without calling `PhaseList::resize` would otherwise
            // submit a bind group whose depth view is the wrong size.
            self.bind_group = Self::make_bind_group(gpu, &self.layout, &self.uniform);
            self.bound_size = (gpu.config.width, gpu.config.height);
        }

        let uniform = FogUniform::from_state(&self.state(), self.distance);
        self.uniform_write
            .write(gpu, &self.uniform, bytemuck::bytes_of(&uniform));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// `cargo check` never parses WGSL; this does, with the same naga front end and validator
    /// wgpu runs at pipeline creation, so a shader mistake fails a unit test instead of a
    /// player's first frame.
    #[test]
    fn fog_wgsl_parses_and_validates_in_both_depth_spellings() {
        let single = include_str!("fog.wgsl").to_owned();
        let multi = single.replace("texture_depth_2d", "texture_depth_multisampled_2d");
        for source in [single, multi] {
            let module = wgpu::naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .expect("fog.wgsl validates");
        }
    }

    /// Original report §7.1 / R1-10: the hand pass clears depth, so the screen-space fog pass
    /// must record before it or it sees no world depth at all.
    #[test]
    fn fog_records_after_weather_and_before_the_hand_clears_depth() {
        assert!(FogPhase::ORDER > PhaseOrder::WEATHER);
        assert!(FogPhase::ORDER > PhaseOrder::TERRAIN_TRANSLUCENT);
        assert!(FogPhase::ORDER < PhaseOrder::FIRST_PERSON);
        assert!(FogPhase::ORDER < PhaseOrder::POST);
        // The client's slot: after everything that writes depth or is
        // fogged by it (weather included), before the self-fogging translucent terrain.
        assert!(FogPhase::BEFORE_TRANSLUCENT_ORDER > PhaseOrder::WEATHER);
        assert!(FogPhase::BEFORE_TRANSLUCENT_ORDER > PhaseOrder::PARTICLES);
        assert!(FogPhase::BEFORE_TRANSLUCENT_ORDER > PhaseOrder::VOLUMETRIC_CLOUDS);
        assert!(FogPhase::BEFORE_TRANSLUCENT_ORDER < PhaseOrder::TERRAIN_TRANSLUCENT);
    }

    /// The early `discard` in `fog.wgsl` is exact: at `factor == 1` the pass emits alpha
    /// `1 - 1 == 0`, and straight alpha blending with source alpha 0 returns the destination
    /// unchanged in every channel, so skipping the write is the same image.
    #[test]
    fn fog_shader_discards_only_where_the_blend_is_an_identity() {
        let source = include_str!("fog.wgsl");
        assert!(source.contains("if (factor >= 1.0) {\n        discard;\n    }"));
        assert!(source.contains("return vec4<f32>(fog.color.rgb, 1.0 - factor);"));
    }

    #[test]
    fn linear_fog_is_clear_at_start_and_opaque_past_end() {
        let mode = FogMode::Linear {
            start: 10.0,
            end: 20.0,
        };
        assert_eq!(mode.factor(0.0), 1.0);
        assert_eq!(mode.factor(10.0), 1.0);
        assert_eq!(mode.factor(20.0), 0.0);
        assert_eq!(mode.factor(30.0), 0.0);
        assert!((mode.factor(15.0) - 0.5).abs() < 1e-4);
    }

    #[test]
    fn exp_fog_decays_toward_zero() {
        let mode = FogMode::Exp { density: 0.1 };
        assert_eq!(mode.factor(0.0), 1.0);
        assert!(mode.factor(50.0) < mode.factor(10.0));
        assert!(mode.factor(1000.0) < 0.01);
    }

    #[test]
    fn water_density_drops_with_respiration_and_bottoms_out() {
        let base = FogState::compute(
            &Environment::default(),
            FogInputs {
                medium: CameraMedium::Water,
                respiration_level: 0,
                ..Default::default()
            },
            0.0,
        );
        let enchanted = FogState::compute(
            &Environment::default(),
            FogInputs {
                medium: CameraMedium::Water,
                respiration_level: 3,
                ..Default::default()
            },
            0.0,
        );
        let FogMode::Exp { density: base_d } = base.mode else {
            panic!("expected exp fog")
        };
        let FogMode::Exp { density: ench_d } = enchanted.mode else {
            panic!("expected exp fog")
        };
        assert!((base_d - 0.1).abs() < 1e-4);
        assert!(ench_d < base_d);
        assert!(ench_d >= 0.0);
    }

    /// Java's water branch emits the raw `0.1F - respiration * 0.03F` density. Enchantment
    /// levels normally stop at three, but custom NBT can carry a higher level; clamping that
    /// value in the renderer changes the fog state instead of preserving vanilla's no-clamp
    /// expression (`EntityRenderer.setupFog:1980-1987`).
    #[test]
    fn water_density_keeps_java_raw_value_for_overlevel_respiration() {
        let inputs = FogInputs {
            medium: CameraMedium::Water,
            respiration_level: 4,
            ..FogInputs::default()
        };
        let FogMode::Exp { density } = FogState::compute(&Environment::default(), inputs, 0.0).mode
        else {
            panic!("expected exp fog")
        };
        assert_eq!(density.to_bits(), 0xBCA3_D708);

        // Keep the sky-pass helper on the same Java branch; a one-sided fix would make the two
        // fog calls disagree whenever the camera is in an over-level respiration medium.
        let FogMode::Exp {
            density: sky_density,
        } = sky_pass_fog_mode(inputs)
        else {
            panic!("expected exp fog")
        };
        assert_eq!(sky_density.to_bits(), density.to_bits());
    }

    #[test]
    fn water_breathing_uses_the_fixed_low_density() {
        let s = FogState::compute(
            &Environment::default(),
            FogInputs {
                medium: CameraMedium::Water,
                water_breathing: true,
                respiration_level: 3,
                ..Default::default()
            },
            0.0,
        );
        let FogMode::Exp { density } = s.mode else {
            panic!("expected exp fog")
        };
        assert!((density - 0.01).abs() < 1e-4);
    }

    #[test]
    fn lava_is_thick_and_orange() {
        let s = FogState::compute(
            &Environment::default(),
            FogInputs {
                medium: CameraMedium::Lava,
                ..Default::default()
            },
            0.0,
        );
        let FogMode::Exp { density } = s.mode else {
            panic!("expected exp fog")
        };
        assert!((density - 2.0).abs() < 1e-4);
        assert!(s.color.x > s.color.y && s.color.y > s.color.z);
    }

    #[test]
    fn air_fog_is_linear_and_scales_with_render_distance() {
        let near = FogState::compute(
            &Environment::default(),
            FogInputs {
                render_distance_chunks: 4,
                ..Default::default()
            },
            0.0,
        );
        let far = FogState::compute(
            &Environment::default(),
            FogInputs {
                render_distance_chunks: 32,
                ..Default::default()
            },
            0.0,
        );
        let FogMode::Linear { end: near_end, .. } = near.mode else {
            panic!("expected linear fog")
        };
        let FogMode::Linear { end: far_end, .. } = far.mode else {
            panic!("expected linear fog")
        };
        assert!(far_end > near_end);
    }

    #[test]
    fn dimension_always_fogs_caps_the_end_distance() {
        let s = FogState::compute(
            &Environment::default(),
            FogInputs {
                render_distance_chunks: 32,
                dimension_always_fogs: true,
                ..Default::default()
            },
            0.0,
        );
        let FogMode::Linear { end, .. } = s.mode else {
            panic!("expected linear fog")
        };
        assert!(end <= 96.0 + 1e-3, "{end}");
    }

    #[test]
    fn blindness_shrinks_the_fog_radius_and_grows_back() {
        let full = FogState::compute(
            &Environment::default(),
            FogInputs {
                blindness: 1.0,
                ..Default::default()
            },
            0.0,
        );
        let fading = FogState::compute(
            &Environment::default(),
            FogInputs {
                blindness: 0.1,
                ..Default::default()
            },
            0.0,
        );
        let FogMode::Linear { end: full_end, .. } = full.mode else {
            panic!("linear")
        };
        let FogMode::Linear {
            end: fading_end, ..
        } = fading.mode
        else {
            panic!("linear")
        };
        assert!(fading_end > full_end);
    }

    #[test]
    fn void_fog_squares_the_clamped_eye_height_factor() {
        let env = Environment::default();
        let above_void = FogState::compute(
            &env,
            FogInputs {
                interpolated_eye_y: 32.0,
                ..Default::default()
            },
            0.0,
        );
        let underground = FogState::compute(
            &env,
            FogInputs {
                interpolated_eye_y: 16.0,
                ..Default::default()
            },
            0.0,
        );
        // `d1 = y / 32`, then Java squares d1 before multiplying all three channels.
        let expected = above_void.color * 0.25;
        assert!((underground.color - expected).abs().max_element() < 1e-6);

        let below_world = FogState::compute(
            &env,
            FogInputs {
                interpolated_eye_y: -1.0,
                ..Default::default()
            },
            0.0,
        );
        assert_eq!(below_world.color, Vec3::ZERO);
    }

    #[test]
    fn blindness_void_fog_blackout_and_final_second_ramp_are_duration_exact() {
        let env = Environment::default();
        let clear = FogState::compute(
            &env,
            FogInputs {
                interpolated_eye_y: 32.0,
                ..Default::default()
            },
            0.0,
        );
        let blinded = FogState::compute(
            &env,
            FogInputs {
                blindness: 1.0,
                blindness_duration_ticks: Some(20),
                interpolated_eye_y: 32.0,
                ..Default::default()
            },
            0.0,
        );
        assert_eq!(blinded.color, Vec3::ZERO);

        let expiring = FogState::compute(
            &env,
            FogInputs {
                blindness: 0.5,
                blindness_duration_ticks: Some(10),
                interpolated_eye_y: 32.0,
                ..Default::default()
            },
            0.0,
        );
        // At 10 ticks Java turns d1=1 into 1*(1-10/20)=0.5, then squares it.
        assert!((expiring.color - clear.color * 0.25).abs().max_element() < 1e-6);
    }

    /// Bytecode (`bfk.i(F):827-870`) widens the channel before multiplying the double void
    /// factor. This witness is deliberately off the tidy y/16 points: casting the factor first
    /// gives a different red bit, so an apparently harmless `Vec3 * factor as f32` rewrite bites.
    #[test]
    fn void_fog_keeps_the_java_f64_channel_multiply_boundary() {
        let input = Vec3::new(0.123_456_79, 0.02, 0.7);
        let factor = 13.37_f64 / 32.0;
        let exact = apply_void_fog(input, factor);
        let expected = (f64::from(input.x) * factor * factor) as f32;
        assert_eq!(exact.x.to_bits(), expected.to_bits());

        let narrowed_factor_first = input.x * (factor * factor) as f32;
        assert_ne!(
            exact.x.to_bits(),
            narrowed_factor_first.to_bits(),
            "negative control: this sample must distinguish Java's f2d/dmul/d2f from f32 factor math"
        );
    }

    /// `updateFogColor:1889-1915` first applies the asymmetric boss tint and only then performs
    /// the common-scale night-vision blend. Both ordering and f32 expression shape matter.
    #[test]
    fn boss_tint_then_night_vision_matches_the_java_colour_stage_order() {
        let env = Environment::default();
        let base = FogState::compute(
            &env,
            FogInputs {
                interpolated_eye_y: 16.0,
                ..FogInputs::default()
            },
            0.0,
        );
        let modifier = 0.5;
        let brightness = 1.0;
        let actual = FogState::compute(
            &env,
            FogInputs {
                interpolated_eye_y: 16.0,
                boss_color_modifier: Some(modifier),
                night_vision_brightness: Some(brightness),
                ..FogInputs::default()
            },
            0.0,
        );
        let expected =
            apply_night_vision(apply_boss_color_modifier(base.color, modifier), brightness);
        assert_eq!(
            actual.color.to_array().map(f32::to_bits),
            expected.to_array().map(f32::to_bits)
        );

        let night_vision_before_boss =
            apply_boss_color_modifier(apply_night_vision(base.color, brightness), modifier);
        assert_ne!(
            actual.color.to_array().map(f32::to_bits),
            night_vision_before_boss.to_array().map(f32::to_bits),
            "negative control: reversing the Java stages must visibly change this low-void sample"
        );
    }

    #[test]
    fn boss_tint_keeps_java_f32_expression_order() {
        let input = Vec3::new(0.123_456_79, 0.02, 0.7);
        let tinted = apply_boss_color_modifier(input, 0.5);
        let exact_red = input.x * (1.0 - 0.5) + input.x * 0.7 * 0.5;
        assert_eq!(tinted.x.to_bits(), exact_red.to_bits());

        let simplified_red = input.x * (1.0 - 0.3 * 0.5);
        assert_ne!(
            tinted.x.to_bits(),
            simplified_red.to_bits(),
            "negative control: algebraically folding the Java expression loses one f32 ULP"
        );
    }

    /// Java's fog branch uses weighted f32 channels, not the upload-time integer pixel helper.
    /// Keep a non-gray bit oracle so coefficient swaps, integer truncation, and accidental
    /// channel reordering cannot hide behind an approximately gray screenshot.
    #[test]
    fn anaglyph_fog_uses_java_weighted_float_channels() {
        let input = Vec3::new(0.123_456_79, 0.345_678_9, 0.876_543_2);
        let actual = apply_anaglyph(input);
        assert!(actual.x != actual.y && actual.y != actual.z);

        // These are the Java-shaped f32 expression's raw bits, not a tolerance-based check.
        assert_eq!(
            actual.to_array().map(f32::to_bits),
            [0x3EAC_C0A8, 0x3E8E_DAB2, 0x3F26_8EDB]
        );
    }

    /// Disabled anaglyph must be an exact identity, while enabled mode must affect both the
    /// computed clear colour and the bytes copied into the final fog uniform.
    #[test]
    fn anaglyph_fog_is_disabled_identity_and_reaches_final_uniform_colour() {
        let env = Environment::default();
        let plain = FogState::compute(&env, FogInputs::default(), 0.0);
        let explicitly_disabled = FogState::compute(
            &env,
            FogInputs {
                anaglyph: false,
                ..FogInputs::default()
            },
            0.0,
        );
        assert_eq!(
            plain.color.to_array().map(f32::to_bits),
            explicitly_disabled.color.to_array().map(f32::to_bits),
            "disabled anaglyph must preserve the pre-existing fog colour exactly"
        );
        let enabled = FogState::compute(
            &env,
            FogInputs {
                anaglyph: true,
                ..FogInputs::default()
            },
            0.0,
        );
        assert_eq!(
            enabled.color.to_array().map(f32::to_bits),
            apply_anaglyph(plain.color).to_array().map(f32::to_bits)
        );
        assert_ne!(
            enabled.color.to_array().map(f32::to_bits),
            plain.color.to_array().map(f32::to_bits),
            "non-gray fog must visibly change when anaglyph is enabled"
        );

        let uniform = FogUniform::from_state(&enabled, FogDistance::Planar);
        assert_eq!(
            [
                uniform.color[0].to_bits(),
                uniform.color[1].to_bits(),
                uniform.color[2].to_bits(),
            ],
            enabled.color.to_array().map(f32::to_bits),
            "the transformed state must be the final uniform/clear-colour source"
        );
    }

    #[test]
    fn fog_distance_is_an_explicit_uniform_selector() {
        let state = FogState::compute(&Environment::default(), FogInputs::default(), 0.0);
        let planar = FogUniform::from_state(&state, FogDistance::Planar);
        let radial = FogUniform::from_state(&state, FogDistance::Radial);

        assert_eq!(planar.params[3].to_bits(), 0.0_f32.to_bits());
        assert_eq!(radial.params[3].to_bits(), 1.0_f32.to_bits());
        assert_eq!(planar.params[..3], radial.params[..3]);
        assert_eq!(FogDistance::default(), FogDistance::Planar);
    }

    /// `updateFogColor` applies anaglyph after night vision. Reversing the stages is a decisive
    /// negative control because night vision's common scale changes all three channels.
    #[test]
    fn anaglyph_fog_runs_after_night_vision() {
        let env = Environment::default();
        let before_night_vision = FogState::compute(
            &env,
            FogInputs {
                night_vision_brightness: None,
                ..FogInputs::default()
            },
            0.0,
        );
        let actual = FogState::compute(
            &env,
            FogInputs {
                anaglyph: true,
                night_vision_brightness: Some(1.0),
                ..FogInputs::default()
            },
            0.0,
        );
        let expected = apply_anaglyph(apply_night_vision(before_night_vision.color, 1.0));
        assert_eq!(
            actual.color.to_array().map(f32::to_bits),
            expected.to_array().map(f32::to_bits)
        );
        let wrong_order = apply_night_vision(apply_anaglyph(before_night_vision.color), 1.0);
        assert_ne!(
            actual.color.to_array().map(f32::to_bits),
            wrong_order.to_array().map(f32::to_bits),
            "negative control: anaglyph must be the final colour stage"
        );
    }

    #[test]
    fn fog_colour_is_biome_tinted() {
        let cold = FogState::compute(
            &Environment::default(),
            FogInputs {
                biome_temperature: 0.0,
                ..Default::default()
            },
            0.0,
        );
        let hot = FogState::compute(
            &Environment::default(),
            FogInputs {
                biome_temperature: 2.0,
                ..Default::default()
            },
            0.0,
        );
        assert_ne!(cold.color, hot.color, "{cold:?} vs {hot:?}");
    }

    #[test]
    fn rain_desaturates_fog_colour_less_on_blue() {
        let clear = FogState::compute(&Environment::default(), FogInputs::default(), 0.0);
        let mut env = Environment::default();
        env.weather.rain_strength = 1.0;
        let rainy = FogState::compute(&env, FogInputs::default(), 0.0);
        assert!(rainy.color.x <= clear.color.x + 1e-4);
        assert!(rainy.color.z <= clear.color.z + 1e-4);
    }

    /// `EntityRenderer.updateRenderer`: the render-distance floor is intentional. It is easy
    /// to mistake it for a gamma setting and remove it, but then a pitch-dark cave at 8 chunks
    /// incorrectly fades all the way to black instead of settling at 0.25.
    #[test]
    fn fog_light_target_keeps_the_java_render_distance_floor() {
        assert_eq!(FogLightSmoothing::target(0.0, 0), 0.0);
        assert!((FogLightSmoothing::target(0.0, 8) - 0.25).abs() < 1e-6);
        assert!((FogLightSmoothing::target(0.0, 12) - 0.375).abs() < 1e-6);
        assert_eq!(FogLightSmoothing::target(0.0, 32), 1.0);
        assert_eq!(FogLightSmoothing::target(1.0, 8), 1.0);
    }

    /// `updateRenderer` does not clamp the distance floor before easing. The normal options
    /// slider ends at 32, so a legal-range-only test would let a Rust-only `.clamp(0, 1)` survive
    /// forever. A raw value loaded from Java's `options.txt` parser reaches the renderer as-is;
    /// at 64 chunks the target for a dark eye is 2.0, not the clamped 1.0.
    #[test]
    fn fog_light_target_keeps_java_raw_distance_floor_outside_slider_range() {
        let raw = FogLightSmoothing::target(0.0, 64);
        assert_eq!(raw.to_bits(), 2.0_f32.to_bits());

        // Negative control for the old implementation: this is exactly the value `.clamp(0, 1)`
        // would have produced, and must no longer be accepted as the Java target.
        assert_ne!(raw.to_bits(), 1.0_f32.to_bits());
    }

    /// Java's `Math.pow` receives a double even though the distance base is a float. This sample
    /// is deliberately outside the 2..32 UI range because all ordinary slider values happen to
    /// agree with Rust's f32 `powf`; the raw options path still reaches this pure renderer seam.
    /// The expected bits are from Temurin's 1.8.9 `EntityRenderer.updateFogColor` expression.
    #[test]
    fn sky_blend_factor_keeps_java_double_pow_boundary() {
        // Keep the witness runtime-valued. LLVM can constant-fold the f32 libm path differently
        // from its runtime implementation, which would make the mutant control vacuous.
        let chunks = std::hint::black_box(35_715_u32);
        let exact = sky_blend_factor(chunks);
        assert_eq!(exact.to_bits(), 0xC08C_22F0);

        let x = 0.25_f32 + 0.75_f32 * chunks as f32 / 32.0_f32;
        let narrowed_pow = 1.0_f32 - x.powf(0.25_f32);
        assert_eq!(narrowed_pow.to_bits(), 0xC08C_22EF);
        assert_ne!(
            exact.to_bits(),
            narrowed_pow.to_bits(),
            "negative control: f32 powf must differ from Java Math.pow(double) at this witness"
        );

        // State the covered normal range explicitly: changing the precision boundary does not
        // perturb any integer distance exposed by the vanilla 2..32 slider.
        for chunks in 2..=32 {
            let x = 0.25_f32 + 0.75_f32 * chunks as f32 / 32.0_f32;
            let old = 1.0_f32 - x.powf(0.25_f32);
            assert_eq!(
                sky_blend_factor(chunks).to_bits(),
                old.to_bits(),
                "chunks={chunks}"
            );
        }
    }

    /// The snapshot must precede the ease, and the frame sample must be between the two tick
    /// values. This catches each of the tempting but visibly wrong mutants: omitting the
    /// snapshot, doing the approach before the snapshot, or sampling `current` directly.
    #[test]
    fn fog_light_smoothing_eases_once_per_tick_and_interpolates_per_frame() {
        let mut smoothing = FogLightSmoothing::default();
        smoothing.tick(1.0, 0);
        assert!((smoothing.current() - 0.1).abs() < 1e-6);
        assert_eq!(smoothing.sample(0.0), 0.0);
        assert!((smoothing.sample(0.5) - 0.05).abs() < 1e-6);
        assert!((smoothing.sample(1.0) - 0.1).abs() < 1e-6);

        smoothing.tick(0.0, 0);
        assert!((smoothing.current() - 0.09).abs() < 1e-6);
        assert!((smoothing.sample(0.5) - 0.095).abs() < 1e-6);
    }

    /// The multiplier belongs after the medium override. Keeping this state test adjacent to
    /// the smoother pins the full `fogColor1/2 -> updateFogColor` composition, including the
    /// easily missed cave-water case.
    #[test]
    fn interpolated_light_smoothing_darkens_the_completed_medium_colour() {
        let mut smoothing = FogLightSmoothing::default();
        smoothing.tick(1.0, 0); // current 0.1, previous 0.0
        let light = smoothing.sample(0.5);
        assert!((light - 0.05).abs() < 1e-6);

        let water = FogState::compute(
            &Environment::default(),
            FogInputs {
                medium: CameraMedium::Water,
                light_smoothing: light,
                ..FogInputs::default()
            },
            0.0,
        );
        let expected = Vec3::new(0.02, 0.02, 0.2) * light;
        assert!((water.color - expected).length() < 1e-6, "{water:?}");
    }

    // ---- the Fog mod ---------------------------------

    /// Bit pattern of a mode, so comparisons cannot hide behind float tolerance.
    fn mode_bits(mode: FogMode) -> (u8, [u32; 2]) {
        match mode {
            FogMode::Linear { start, end } => (0, [start.to_bits(), end.to_bits()]),
            FogMode::Exp { density } => (1, [density.to_bits(), 0]),
        }
    }

    /// The pre-W4 mode expression of `FogState::compute`, verbatim, as the oracle.
    fn pre_w4_world_mode(inputs: FogInputs) -> FogMode {
        let far = inputs.render_distance_chunks as f32 * 16.0;
        if inputs.blindness > 0.0 {
            let base = 5.0 + (far - 5.0) * (1.0 - inputs.blindness);
            FogMode::Linear {
                start: base * 0.25,
                end: base,
            }
        } else {
            match inputs.medium {
                CameraMedium::Cloud => FogMode::Exp { density: 0.1 },
                CameraMedium::Water => FogMode::Exp {
                    density: if inputs.water_breathing {
                        0.01
                    } else {
                        0.1 - inputs.respiration_level as f32 * 0.03
                    },
                },
                CameraMedium::Lava => FogMode::Exp { density: 2.0 },
                CameraMedium::Air if inputs.dimension_always_fogs => FogMode::Linear {
                    start: far * 0.05,
                    end: far.min(192.0) * 0.5,
                },
                CameraMedium::Air => FogMode::Linear {
                    start: far * 0.75,
                    end: far,
                },
            }
        }
    }

    /// Every combination of the inputs `setupFog` branches on.
    fn input_sweep() -> Vec<FogInputs> {
        let mut out = Vec::new();
        for render_distance_chunks in [2, 4, 8, 12, 16, 32, 64] {
            for blindness in [0.0, 0.05, 0.5, 1.0] {
                for medium in [
                    CameraMedium::Air,
                    CameraMedium::Cloud,
                    CameraMedium::Water,
                    CameraMedium::Lava,
                ] {
                    for dimension_always_fogs in [false, true] {
                        for (water_breathing, respiration_level) in
                            [(false, 0), (false, 3), (false, 4), (true, 3)]
                        {
                            out.push(FogInputs {
                                render_distance_chunks,
                                blindness,
                                medium,
                                dimension_always_fogs,
                                water_breathing,
                                respiration_level,
                                ..FogInputs::default()
                            });
                        }
                    }
                }
            }
        }
        out
    }

    /// With the mod's defaults the world-pass falloff is vanilla's expression bit for bit, on
    /// every branch; moving it out of `FogState::compute` changed nothing.
    #[test]
    fn world_pass_mode_at_the_mod_defaults_is_the_pre_w4_expression_bit_for_bit() {
        let defaults = FogInputs::default();
        assert!(defaults.distance_fog && !defaults.clear_liquids);
        for inputs in input_sweep() {
            let expected = mode_bits(pre_w4_world_mode(inputs));
            assert_eq!(
                mode_bits(world_pass_fog_mode(inputs)),
                expected,
                "{inputs:?}"
            );
            let state = FogState::compute(&Environment::default(), inputs, 0.25);
            assert_eq!(mode_bits(state.mode), expected, "{inputs:?}");
        }
    }

    /// Fog off clears exactly the air branch (both dimension variants); blindness, in-cloud,
    /// water and lava keep vanilla's bits, and the colour never changes.
    #[test]
    fn fog_off_clears_only_the_air_branch_and_never_touches_the_colour() {
        let env = Environment::default();
        let mut cleared = 0;
        for inputs in input_sweep() {
            let off = FogInputs {
                distance_fog: false,
                ..inputs
            };
            let vanilla = FogState::compute(&env, inputs, 0.25);
            let state = FogState::compute(&env, off, 0.25);
            assert_eq!(
                state.color.to_array().map(f32::to_bits),
                vanilla.color.to_array().map(f32::to_bits),
                "{inputs:?}"
            );
            if inputs.blindness == 0.0 && inputs.medium == CameraMedium::Air {
                assert!(state.mode.is_clear(), "{inputs:?}");
                cleared += 1;
            } else {
                assert_eq!(mode_bits(state.mode), mode_bits(vanilla.mode), "{inputs:?}");
            }
        }
        assert!(cleared > 0);
    }

    /// "Clear water and lava" gives water and lava the air falloff, in both passes, whatever
    /// the main switch says; it keeps their colour, and never touches air, clouds or blindness.
    #[test]
    fn clear_liquids_gives_water_and_lava_the_air_falloff_and_keeps_their_colour() {
        let env = Environment::default();
        for inputs in input_sweep() {
            for distance_fog in [true, false] {
                let base = FogInputs {
                    distance_fog,
                    ..inputs
                };
                let clear = FogInputs {
                    clear_liquids: true,
                    ..base
                };
                let as_air = FogInputs {
                    medium: CameraMedium::Air,
                    ..base
                };
                let liquid = matches!(inputs.medium, CameraMedium::Water | CameraMedium::Lava);
                let world = if liquid && inputs.blindness == 0.0 {
                    world_pass_fog_mode(as_air)
                } else {
                    world_pass_fog_mode(base)
                };
                assert_eq!(
                    mode_bits(world_pass_fog_mode(clear)),
                    mode_bits(world),
                    "{clear:?}"
                );
                let sky = if liquid && inputs.blindness == 0.0 {
                    sky_pass_fog_mode(as_air)
                } else {
                    sky_pass_fog_mode(base)
                };
                assert_eq!(
                    mode_bits(sky_pass_fog_mode(clear)),
                    mode_bits(sky),
                    "{clear:?}"
                );
                assert_eq!(
                    FogState::compute(&env, clear, 0.0)
                        .color
                        .to_array()
                        .map(f32::to_bits),
                    FogState::compute(&env, base, 0.0)
                        .color
                        .to_array()
                        .map(f32::to_bits),
                    "{clear:?}"
                );
            }
        }
    }

    /// The sky pass ignores the switch: its horizon fade stays exactly vanilla's.
    #[test]
    fn the_sky_pass_ignores_the_fog_switch() {
        for inputs in input_sweep() {
            let off = FogInputs {
                distance_fog: false,
                ..inputs
            };
            assert_eq!(
                mode_bits(sky_pass_fog_mode(off)),
                mode_bits(sky_pass_fog_mode(inputs)),
                "{inputs:?}"
            );
        }
    }

    /// `FogMode::CLEAR` is an exact identity in every consumer's own formula, for every
    /// distance anything is drawn at, with no special case: [`FogMode::factor`], the
    /// screen-space pass, the clouds and terrain (`(end - d) / (end - start)` behind an
    /// `end <= start` guard), and the sky dome (`(end - d) / max(end - start, 1e-4)`).
    #[test]
    fn the_clear_mode_is_an_identity_in_every_consumer_formula() {
        let FogMode::Linear { start, end } = FogMode::CLEAR else {
            panic!("CLEAR is linear")
        };
        assert_eq!(start, 1_000_000_000.0_f32);
        assert_eq!(f64::from(start), 1.0e9, "exactly representable");
        assert_eq!(f64::from(end), 2.0e9, "exactly representable");
        assert!(FogMode::CLEAR.is_clear());
        assert!(!pre_w4_world_mode(FogInputs::default()).is_clear());
        for d in [
            0.0_f32, 0.05, 1.0, 96.0, 128.0, 181.02, 512.0, 724.08, 2048.0, 1.0e5,
        ] {
            assert_eq!(FogMode::CLEAR.factor(d), 1.0, "{d}");
            assert!(end > start);
            assert_eq!(((end - d) / (end - start)).clamp(0.0, 1.0), 1.0, "{d}");
            assert_eq!(
                ((end - d) / (end - start).max(1e-4)).clamp(0.0, 1.0),
                1.0,
                "{d}"
            );
        }
        let state = FogState {
            color: Vec3::new(0.7, 0.8, 1.0),
            mode: FogMode::CLEAR,
        };
        let uniform = FogUniform::from_state(&state, FogDistance::Radial);
        assert_eq!(uniform.params, [1.0e9, 2.0e9, 0.0, 1.0]);
        assert_eq!(uniform.color[3], 0.0, "linear");
    }
}
