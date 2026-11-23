//! The 256-entry block/sky-light lightmap texture, gamma, and night vision.
//!
//! java: `EntityRenderer.updateLightmap` (lines ~922-1067 of the 2,050-line file), plus
//! `WorldProvider.generateLightBrightnessTable` (`WorldProviderHell`'s override for the Nether)
//! and `EntityRenderer.getNightVisionBrightness`/`updateTorchFlicker`.
//!
//! Vanilla looks up combined light (`skyLight << 4 | blockLight`, `0..255`) in this texture once
//! per fragment to get the final vertex/fragment colour multiplier for every textured quad in
//! the world. [`Lightmap::generate`] reproduces that 256-entry table exactly (gamma curve, night
//! vision, lightning-flash bypass, the End's flat colour override, all included); wiring the
//! *consumer* side into `terrain.wgsl`'s `light_curve` placeholder is S11's cross-section request
//! and out of S20's ownership (`terrain/**` belongs to S11) — see this crate's
//! `sky::mod` docs for the seam this leaves for a future pass.

use mc_math::{clamp_f32, sin, BlockPos, DVec3};

use crate::sky::environment::Dimension;

/// `WorldProvider.generateLightBrightnessTable`: the 16-entry base brightness curve for one raw
/// light level (`0..=15`), before day/night or torch-flicker scaling.
///
/// `ambient` is [`Dimension::ambient_light`] — `0.0` in the Overworld/End, `0.1` in the Nether.
/// java: `terrain.wgsl`'s `light_curve` comment already documents the shape of this curve; this
/// is the same formula, table form.
#[must_use]
pub fn light_brightness_table(ambient: f32) -> [f32; 16] {
    std::array::from_fn(|i| {
        let f1 = 1.0 - i as f32 / 15.0;
        (1.0 - f1) / (f1 * 3.0 + 1.0) * (1.0 - ambient) + ambient
    })
}

/// Samples the scalar light level used by `Entity.getBrightness`.
///
/// `Entity.java:1256` samples `(posX, posY + getEyeHeight(), posZ)`, floors that point into a
/// block position, and returns zero when that block's chunk is not loaded. The caller supplies
/// the effective eye height because several subclasses override the base `height * 0.85`. The
/// latter guard matters: [`mc_world::World::light_at`] faithfully exposes an unloaded
/// overworld column's blank-chunk skylight value (`15`), while vanilla's entity method must not
/// turn that provisional value into a fully bright HUD vignette.
///
/// `ambient_light` is the current provider's `generateLightBrightnessTable` floor: `0.0` in the
/// Overworld/End and `0.1` in the Nether.  A HUD caller can use this value directly as
/// `GuiIngame.renderVignette`'s brightness input.
/// java: `Entity.getBrightness(float)` / `World.getLightBrightness(BlockPos)`.
#[must_use]
pub fn entity_brightness(
    world: &mc_world::World,
    position: DVec3,
    eye_height: f32,
    ambient_light: f32,
) -> f32 {
    let sample = entity_light_sample(position, eye_height);
    if !world.is_chunk_loaded(sample.chunk_pos()) {
        return 0.0;
    }
    light_brightness_table(ambient_light)[usize::from(world.light_at(sample))]
}

/// Samples the packed lightmap coordinate returned by `Entity.getBrightnessForRender`.
///
/// This intentionally uses the same `posX`/`posY + getEyeHeight()` block as
/// [`entity_brightness`].  `World.getCombinedLight(pos, 0)` stores the two raw nibbles in the
/// OpenGL lightmap-coordinate positions (`sky << 20 | block << 4`); it is *not* the `sky << 4 |
/// block` index used to look up [`Lightmap::rgba`].  An unloaded chunk returns literal zero
/// before either blank-chunk light default can leak through.
/// java: `Entity.getBrightnessForRender(float)` / `World.getCombinedLight(BlockPos, int)`.
#[must_use]
pub fn entity_brightness_for_render(
    world: &mc_world::World,
    position: DVec3,
    eye_height: f32,
) -> u32 {
    let sample = entity_light_sample(position, eye_height);
    if !world.is_chunk_loaded(sample.chunk_pos()) {
        return 0;
    }
    u32::from(world.sky_light_at(sample)) << 20 | u32::from(world.block_light_at(sample)) << 4
}

fn entity_light_sample(position: DVec3, eye_height: f32) -> BlockPos {
    BlockPos::new(
        position.x.floor() as i32,
        (position.y + f64::from(eye_height)).floor() as i32,
        position.z.floor() as i32,
    )
}

/// `0.4..1.0`, oscillating as a night-vision potion is about to expire so the player gets a
/// visible warning flicker. java: `EntityRenderer.getNightVisionBrightness`
///
/// `ticks_remaining` counts down to `0`; `partial_ticks` interpolates within the current tick.
#[must_use]
pub fn night_vision_brightness(ticks_remaining: i32, partial_ticks: f32) -> f32 {
    if ticks_remaining > 200 {
        1.0
    } else {
        0.7 + sin((ticks_remaining as f32 - partial_ticks) * std::f32::consts::PI * 0.2) * 0.3
    }
}

/// A random-walk flicker applied to torchlight, updated once per client tick.
/// java: `EntityRenderer.updateTorchFlicker`
#[derive(Debug, Clone, Copy, Default)]
pub struct TorchFlicker {
    velocity: f32,
    value: f32,
}

impl TorchFlicker {
    /// Advance the flicker by one tick. `random_a`/`random_b`/`random_c` are three independent
    /// draws in `0.0..1.0`; callers own their RNG source (vanilla uses `Math.random()`, which has
    /// no seed/parity requirement here — flicker is cosmetic, not simulation state).
    ///
    /// # The third line is an assignment, not an accumulation
    ///
    /// java: `EntityRenderer.updateTorchFlicker:918` reads
    /// `this.torchFlickerX += (this.torchFlickerDX - this.torchFlickerX) * 1.0F;`, and with the
    /// literal `* 1.0F` that collapses to `torchFlickerX = torchFlickerDX` — an odd way to write
    /// it, but that *is* what it computes. It is transcribed as the assignment it is, because the
    /// difference is not cosmetic:
    ///
    /// `velocity` is a damped (`* 0.9` every tick) random walk, so it is stationary with a
    /// standard deviation near `0.34` and stays inside roughly `±1`. Writing the third line as
    /// `value += velocity` instead makes `value` the *running sum* of that walk, whose variance
    /// grows without bound — about `1.5 * sqrt(steps)`. Since `step` is called once per **tick**
    /// (`Client::tick`), its cadence is fixed at 20 Hz instead of changing with frame rate, and
    /// [`block_light_multiplier`](TorchFlicker::block_light_multiplier) turns it into a
    /// block-light scale of `value * 0.1 + 1.5` that swings tens of units either side of zero
    /// rather than sitting in `1.4..1.6`.
    ///
    /// A **negative** multiplier is what makes that visible rather than merely wrong.
    /// [`Lightmap::generate`] feeds the block-light term `f3` through `r = f4 + f3`,
    /// `g = f5 + f3 * (0.36 * f3 + 0.64)` and `b = f2 + f3 * (0.6 * f3² + 0.4)` — quadratic in
    /// green and cubic in blue. At `f3 < 0` the red term clamps to zero while green's `0.36 * f3²`
    /// stays positive and blue's `0.6 * f3³` goes sharply negative, so a torch stops looking like
    /// a torch and radiates green, or blue-green once sky light adds back into `b`. The walk
    /// crossing zero in both directions is what makes it flicker between that and normal.
    pub fn step(&mut self, random_a: f32, random_b: f32, random_c: f32) {
        self.velocity += (random_a - random_b) * random_c * random_c;
        self.velocity *= 0.9;
        self.value = self.velocity;
    }

    /// The multiplier applied to the block-light table lookup, roughly `1.4..1.6`.
    /// java: `EntityRenderer.updateLightmap`'s `(torchFlickerX * 0.1F + 1.5F)`
    #[must_use]
    pub fn block_light_multiplier(&self) -> f32 {
        self.value * 0.1 + 1.5
    }
}

/// The two samples behind `EntityRenderer.bossColorModifier`.
///
/// `EntityRenderer.updateRenderer` advances this once per client tick *after* boss renderers had
/// their opportunity to request the colour effect.  It owns the one-tick request's consumption:
/// a renderer sets `has_color_modifier`, and this method clears it after reading it, exactly as
/// vanilla clears `BossStatus.hasColorModifier`.  The render-frame consumer then calls
/// [`Self::interpolated`] with the frame's `partial_ticks`.
///
/// The type deliberately lives next to [`LightmapParams`], rather than in a particular client
/// status store: renderers own the transient request, while any client loop can own these two
/// scalar samples and feed the resulting value to the lightmap.
/// java: `EntityRenderer.updateRenderer:369-385`, `updateLightmap:955-961`.
#[derive(Debug, Clone, Copy, Default)]
pub struct BossColorModifier {
    previous: f32,
    current: f32,
}

impl BossColorModifier {
    /// Advance the modifier by one client tick and consume this tick's boss-render request.
    ///
    /// This is intentionally an `&mut bool`: callers cannot accidentally leave a successful
    /// boss draw latched forever.  The rise (`0.05`) and fall (`0.0125`) are frame-independent
    /// *tick* steps from vanilla, not rates to multiply by elapsed time.
    pub fn tick(&mut self, has_color_modifier: &mut bool) {
        self.previous = self.current;
        if *has_color_modifier {
            self.current = (self.current + 0.05).min(1.0);
            *has_color_modifier = false;
        } else if self.current > 0.0 {
            self.current = (self.current - 0.0125).max(0.0);
        }
    }

    /// The render-frame value that belongs in [`LightmapParams::boss_darkening`].
    ///
    /// Kept as the source expression rather than a clamped lerp: vanilla receives a normal
    /// `partialTicks` value and performs `prev + (current - prev) * partialTicks` directly.
    #[must_use]
    pub fn interpolated(self, partial_ticks: f32) -> f32 {
        self.previous + (self.current - self.previous) * partial_ticks
    }

    /// The current tick sample, useful when a caller needs to expose status diagnostics.
    #[must_use]
    pub fn current(self) -> f32 {
        self.current
    }
}

/// `updateLightmap`'s per-channel multiplier for an interpolated boss modifier.
///
/// At `1.0`, vanilla darkens red to `0.7` and green/blue to `0.6`; retaining the three entries
/// makes the unequal red channel hard to accidentally turn into an ordinary uniform fade.
#[must_use]
fn boss_color_multiplier(modifier: f32) -> [f32; 3] {
    [
        1.0 - 0.3 * modifier,
        1.0 - 0.4 * modifier,
        1.0 - 0.4 * modifier,
    ]
}

/// Everything [`Lightmap::generate`] needs, gathered in one place so the function itself stays a
/// pure transcription of `updateLightmap`'s body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightmapParams {
    /// java: `world.getSunBrightness(1.0F)` — [`crate::sky::environment::Environment::sun_brightness`].
    pub sun_brightness: f32,
    /// Which dimension's lightmap rules apply (Nether's ambient floor, the End's flat override).
    pub dimension: Dimension,
    /// `gameSettings.gammaSetting`, `0.0..=1.0`.
    pub gamma: f32,
    /// [`night_vision_brightness`]'s result, or `0.0` when the potion is not active.
    pub night_vision: f32,
    /// [`TorchFlicker::block_light_multiplier`]'s result.
    pub torch_flicker: f32,
    /// `true` on the frame(s) right after nearby lightning: skips the sky-light dimming term for
    /// a full-brightness flash. java: `world.getLastLightningBolt() > 0`
    pub lightning_flash: bool,
    /// `0.0..1.0` wither-boss darkening. java: `EntityRenderer.bossColorModifier`
    pub boss_darkening: f32,
    /// The client's `fullbright` option — **not vanilla**, default off. When set,
    /// [`Lightmap::generate`] returns [`Lightmap::full_bright`] and every other input is ignored.
    ///
    /// Only this lookup table changes. Light propagation, `World::light_at` and everything that
    /// reads light outside a draw (the mushroom placement gate before C08 above all) are
    /// untouched, so the option cannot alter what the client sends.
    pub fullbright: bool,
}

impl Default for LightmapParams {
    fn default() -> Self {
        Self {
            sun_brightness: 1.0,
            dimension: Dimension::Overworld,
            gamma: 0.0,
            night_vision: 0.0,
            torch_flicker: 1.5,
            lightning_flash: false,
            boss_darkening: 0.0,
            fullbright: false,
        }
    }
}

/// The 256-entry (16 sky x 16 block) lightmap, packed RGBA8, ready to upload as a 16x16 texture.
/// java: `EntityRenderer.lightmapColors` / `lightmapTexture`
#[derive(Debug, Clone)]
pub struct Lightmap {
    /// `256 * 4` bytes, index `(sky << 4 | block) * 4`.
    pub rgba: Vec<u8>,
}

impl Lightmap {
    /// Every texel at full intensity, `[255, 255, 255, 255]`: the `fullbright` option's table.
    ///
    /// This is the saturated form of the familiar `gamma:100` trick, without its quirk: through
    /// `updateLightmap`'s curve a channel already clamped to exactly `1.0` gains nothing from the
    /// gamma lerp and lands on `0.99 * 255 = 252` after the final `* 0.96 + 0.03`, while every
    /// darker channel overshoots and clamps to 255.
    #[must_use]
    pub fn full_bright() -> Self {
        Self {
            rgba: vec![u8::MAX; 256 * 4],
        }
    }

    /// Build the lightmap for one frame. java: `EntityRenderer.updateLightmap`
    #[must_use]
    pub fn generate(params: LightmapParams) -> Self {
        if params.fullbright {
            return Self::full_bright();
        }
        let table = light_brightness_table(params.dimension.ambient_light());
        let f = params.sun_brightness;
        let f1 = f * 0.95 + 0.05;

        let mut rgba = vec![0u8; 256 * 4];
        for i in 0..256usize {
            let sky_level = i / 16;
            let block_level = i % 16;

            let mut f2 = table[sky_level] * f1;
            let f3 = table[block_level] * params.torch_flicker;

            if params.lightning_flash {
                f2 = table[sky_level];
            }

            let f4 = f2 * (f * 0.65 + 0.35);
            let f5 = f2 * (f * 0.65 + 0.35);
            let f6 = f3 * ((f3 * 0.6 + 0.4) * 0.6 + 0.4);
            let f7 = f3 * (f3 * f3 * 0.6 + 0.4);

            let mut r = f4 + f3;
            let mut g = f5 + f6;
            let mut b = f2 + f7;
            r = r * 0.96 + 0.03;
            g = g * 0.96 + 0.03;
            b = b * 0.96 + 0.03;

            if params.boss_darkening > 0.0 {
                let [r_mul, g_mul, b_mul] = boss_color_multiplier(params.boss_darkening);
                r *= r_mul;
                g *= g_mul;
                b *= b_mul;
            }

            if params.dimension == Dimension::End {
                r = 0.22 + f3 * 0.75;
                g = 0.28 + f6 * 0.75;
                b = 0.25 + f7 * 0.75;
            }

            if params.night_vision > 0.0 {
                let mut inv = 1.0 / r.max(1e-6);
                inv = inv.min(1.0 / g.max(1e-6));
                inv = inv.min(1.0 / b.max(1e-6));
                let f15 = params.night_vision;
                r = r * (1.0 - f15) + r * inv * f15;
                g = g * (1.0 - f15) + g * inv * f15;
                b = b * (1.0 - f15) + b * inv * f15;
            }

            r = r.min(1.0);
            g = g.min(1.0);
            b = b.min(1.0);

            // Gamma slider: lerp toward an "inverse-quartic brightened" curve.
            let gamma = params.gamma;
            let gr = 1.0 - (1.0 - r).powi(4);
            let gg = 1.0 - (1.0 - g).powi(4);
            let gb = 1.0 - (1.0 - b).powi(4);
            r = r * (1.0 - gamma) + gr * gamma;
            g = g * (1.0 - gamma) + gg * gamma;
            b = b * (1.0 - gamma) + gb * gamma;
            r = r * 0.96 + 0.03;
            g = g * 0.96 + 0.03;
            b = b * 0.96 + 0.03;

            r = clamp_f32(r, 0.0, 1.0);
            g = clamp_f32(g, 0.0, 1.0);
            b = clamp_f32(b, 0.0, 1.0);

            let idx = i * 4;
            rgba[idx] = (r * 255.0) as u8;
            rgba[idx + 1] = (g * 255.0) as u8;
            rgba[idx + 2] = (b * 255.0) as u8;
            rgba[idx + 3] = 255;
        }

        Self { rgba }
    }

    /// Combined-light index (`skyLight << 4 | blockLight`) -> RGBA8. java:
    /// `EntityRenderer.lightmapColors[combinedLight]` read at draw time.
    #[must_use]
    pub fn sample(&self, sky_light: u8, block_light: u8) -> [u8; 4] {
        let idx = ((sky_light.min(15) as usize) << 4 | block_light.min(15) as usize) * 4;
        [
            self.rgba[idx],
            self.rgba[idx + 1],
            self.rgba[idx + 2],
            self.rgba[idx + 3],
        ]
    }

    /// [`Self::generate`], memoized on the last parameters this thread generated for.
    ///
    /// GPURES-7 / SKY-5: terrain, particles, tile entities and entities each own a lightmap
    /// texture and regenerate on the same `LightmapParams` change (every tick, because torch
    /// flicker steps per tick; every frame while night vision or the boss tint fades). The first
    /// owner computes the table and the other three clone its 1 KiB result. `generate` is a pure
    /// function of the parameters, and the owners already treat `PartialEq`-equal parameters as
    /// the same table (their `uploaded_lightmap_params` skip), so this is bit-identical.
    #[must_use]
    pub fn generate_cached(params: LightmapParams) -> Self {
        thread_local! {
            static LAST: std::cell::RefCell<Option<(LightmapParams, Lightmap)>> =
                const { std::cell::RefCell::new(None) };
        }
        LAST.with(|last| {
            let mut last = last.borrow_mut();
            if let Some((cached, table)) = last.as_ref() {
                if *cached == params {
                    return table.clone();
                }
            }
            let table = Self::generate(params);
            *last = Some((params, table.clone()));
            table
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_math::ChunkPos;
    use mc_world::{SkyBlock, World};

    /// GPURES-7: the memoized table is the pure table for every parameter sequence the four
    /// owners can produce, including repeats, returns to an older value and NaN parameters.
    #[test]
    fn cached_lightmap_matches_generate_for_repeated_and_alternating_params() {
        let base = LightmapParams::default();
        let sequence = [
            base,
            base,
            LightmapParams {
                torch_flicker: 1.43,
                ..base
            },
            LightmapParams {
                torch_flicker: 1.43,
                ..base
            },
            base,
            LightmapParams {
                night_vision: 0.37,
                gamma: 1.0,
                ..base
            },
            LightmapParams {
                sun_brightness: f32::NAN,
                ..base
            },
            LightmapParams {
                sun_brightness: f32::NAN,
                ..base
            },
            LightmapParams {
                dimension: Dimension::Nether,
                boss_darkening: 0.5,
                ..base
            },
            LightmapParams {
                lightning_flash: true,
                ..base
            },
            base,
        ];
        for params in sequence {
            for _owner in 0..4 {
                assert_eq!(
                    Lightmap::generate_cached(params).rgba,
                    Lightmap::generate(params).rgba
                );
            }
        }
    }

    #[test]
    fn brightness_table_is_monotonic_and_bounded() {
        let table = light_brightness_table(0.0);
        assert!((table[0] - 0.0).abs() < 1e-6, "{}", table[0]);
        assert!((table[15] - 1.0).abs() < 1e-6, "{}", table[15]);
        for w in table.windows(2) {
            assert!(w[1] >= w[0], "{table:?}");
        }
    }

    #[test]
    fn entity_brightness_returns_zero_for_an_unloaded_chunk_not_blank_chunk_skylight() {
        let world = World::new();
        let position = DVec3::new(-0.01, 64.0, -16.01);

        // `World::light_at` quite correctly sees the blank chunk's sky=15.  Entity.java's
        // `isBlockLoaded` guard is intentionally stricter before the HUD turns this into a
        // vignette brightness.
        assert_eq!(world.light_at(BlockPos::new(-1, 65, -17)), 15);
        assert_eq!(entity_brightness(&world, position, 1.53, 0.0), 0.0);
    }

    #[test]
    fn entity_brightness_uses_the_effective_eye_height_and_provider_curve() {
        let mut world = World::new();
        world.set_has_sky_light(false);
        world.load_empty_chunk(ChunkPos::new(-1, -2));

        // Position floors to x=-1/z=-17; `64.5 + 1.53` floors to y=66. Put distinct light
        // values at adjacent cells so the historical `height * .66` sample at y=65 is detected.
        world.set_light_for(SkyBlock::Block, BlockPos::new(-1, 65, -17), 7);
        world.set_light_for(SkyBlock::Block, BlockPos::new(-1, 66, -17), 12);

        let actual = entity_brightness(&world, DVec3::new(-0.01, 64.5, -16.01), 1.53, 0.1);
        let expected = light_brightness_table(0.1)[12];
        assert_eq!(actual, expected);
        assert_ne!(actual, light_brightness_table(0.1)[7]);
    }

    #[test]
    fn entity_brightness_for_render_packs_sky_and_block_channels_in_their_gl_offsets() {
        let mut world = World::new();
        world.load_empty_chunk(ChunkPos::new(0, 0));
        let sample = BlockPos::new(3, 65, 4);
        world.set_light_for(SkyBlock::Sky, sample, 9);
        world.set_light_for(SkyBlock::Block, sample, 6);

        assert_eq!(
            entity_brightness_for_render(&world, DVec3::new(3.9, 64.0, 4.9), 1.0),
            0x0090_0060,
            "Java packs sky at bit 20 and block at bit 4, not into lightmap-table nibbles"
        );
        assert_eq!(
            entity_brightness_for_render(&World::new(), DVec3::new(3.9, 64.0, 4.9), 1.0),
            0,
            "the unloaded-chunk guard must run before blank-chunk sky light"
        );
    }

    #[test]
    fn nether_ambient_floor_lifts_the_darkest_level() {
        let overworld = light_brightness_table(0.0);
        let nether = light_brightness_table(0.1);
        assert_eq!(overworld[0], 0.0);
        assert!((nether[0] - 0.1).abs() < 1e-6);
        assert!(nether[15] >= overworld[15]);
    }

    #[test]
    fn full_sky_full_block_at_noon_is_bright() {
        let lm = Lightmap::generate(LightmapParams {
            sun_brightness: 1.0,
            ..Default::default()
        });
        let [r, g, b, a] = lm.sample(15, 15);
        assert!(r > 200 && g > 200 && b > 200, "{r} {g} {b}");
        assert_eq!(a, 255);
    }

    #[test]
    fn dark_underground_at_midnight_is_dim() {
        let lm = Lightmap::generate(LightmapParams {
            sun_brightness: 0.2,
            ..Default::default()
        });
        let [r, g, b, _] = lm.sample(0, 0);
        // Never literally zero: vanilla's `*0.96+0.03` floor keeps a touch of ambient light.
        assert!(r < 40 && g < 40 && b < 40, "{r} {g} {b}");
        assert!(r > 0);
    }

    #[test]
    fn block_light_alone_still_lights_a_dark_room() {
        let lm = Lightmap::generate(LightmapParams {
            sun_brightness: 0.0,
            ..Default::default()
        });
        let dark = lm.sample(0, 0);
        let torchlit = lm.sample(0, 14);
        assert!(torchlit[0] > dark[0], "{torchlit:?} vs {dark:?}");
    }

    #[test]
    fn gamma_brightens_dark_levels() {
        let low_gamma = Lightmap::generate(LightmapParams {
            sun_brightness: 0.2,
            gamma: 0.0,
            ..Default::default()
        });
        let high_gamma = Lightmap::generate(LightmapParams {
            sun_brightness: 0.2,
            gamma: 1.0,
            ..Default::default()
        });
        let a = low_gamma.sample(0, 0);
        let b = high_gamma.sample(0, 0);
        assert!(b[0] >= a[0], "{b:?} vs {a:?}");
    }

    #[test]
    fn night_vision_flattens_the_channels() {
        let normal = Lightmap::generate(LightmapParams {
            sun_brightness: 0.0,
            ..Default::default()
        });
        let nv = Lightmap::generate(LightmapParams {
            sun_brightness: 0.0,
            night_vision: 1.0,
            ..Default::default()
        });
        let [nr, ng, nb, _] = normal.sample(0, 0);
        let [vr, vg, vb, _] = nv.sample(0, 0);
        // Night vision brings every channel up toward the brightest one, so the darkest
        // combined-light entry should be strictly brighter under night vision.
        assert!(vr >= nr && vg >= ng && vb >= nb, "{nv:?} {normal:?}");
        let _ = (nr, ng, nb, vr, vg, vb);
    }

    #[test]
    fn lightning_flash_bypasses_night_dimming() {
        let midnight = Lightmap::generate(LightmapParams {
            sun_brightness: 0.0,
            ..Default::default()
        });
        let flash = Lightmap::generate(LightmapParams {
            sun_brightness: 0.0,
            lightning_flash: true,
            ..Default::default()
        });
        let dark = midnight.sample(15, 0);
        let lit = flash.sample(15, 0);
        assert!(lit[2] > dark[2], "{lit:?} vs {dark:?}");
    }

    #[test]
    fn the_end_overrides_to_its_fixed_tint() {
        let overworld = Lightmap::generate(LightmapParams {
            dimension: Dimension::Overworld,
            sun_brightness: 1.0,
            ..Default::default()
        });
        let end = Lightmap::generate(LightmapParams {
            dimension: Dimension::End,
            sun_brightness: 1.0,
            ..Default::default()
        });
        assert_ne!(overworld.sample(15, 0), end.sample(15, 0));
    }

    #[test]
    fn boss_color_modifier_consumes_the_draw_request_and_uses_vanillas_tick_ramp() {
        let mut modifier = BossColorModifier::default();
        let mut requested = true;

        modifier.tick(&mut requested);
        assert!(
            !requested,
            "the one-tick BossStatus request must be consumed"
        );
        assert!((modifier.current() - 0.05).abs() < f32::EPSILON);
        assert!((modifier.interpolated(0.0) - 0.0).abs() < f32::EPSILON);
        assert!((modifier.interpolated(1.0) - 0.05).abs() < f32::EPSILON);
        assert!((modifier.interpolated(0.5) - 0.025).abs() < f32::EPSILON);

        // A fresh request on each tick is what keeps the visible-withering ramp climbing.
        for _ in 0..19 {
            requested = true;
            modifier.tick(&mut requested);
        }
        assert!((modifier.current() - 1.0).abs() < f32::EPSILON);

        // No request is a much slower decay, matching `-= 0.0125F` rather than a reset.
        modifier.tick(&mut requested);
        assert!((modifier.current() - 0.9875).abs() < f32::EPSILON);
        assert!((modifier.interpolated(0.25) - 0.996_875).abs() < f32::EPSILON);
    }

    #[test]
    fn boss_color_vector_keeps_red_less_darkened_than_green_and_blue() {
        assert_eq!(boss_color_multiplier(0.0), [1.0, 1.0, 1.0]);
        assert_eq!(boss_color_multiplier(1.0), [0.7, 0.6, 0.6]);
        assert_eq!(boss_color_multiplier(0.5), [0.85, 0.8, 0.8]);

        // Negative control: a uniform fade is visually plausible but is not the source vector.
        assert_ne!(boss_color_multiplier(1.0), [0.6, 0.6, 0.6]);
    }

    #[test]
    fn full_boss_modifier_darkens_the_generated_lightmap_with_the_source_channel_order() {
        let ordinary = Lightmap::generate(LightmapParams {
            sun_brightness: 0.2,
            torch_flicker: 0.0,
            ..Default::default()
        });
        let darkened = Lightmap::generate(LightmapParams {
            sun_brightness: 0.2,
            torch_flicker: 0.0,
            boss_darkening: 1.0,
            ..Default::default()
        });
        let base = ordinary.sample(12, 0);
        let boss = darkened.sample(12, 0);
        assert!(boss[0] < base[0] && boss[1] < base[1] && boss[2] < base[2]);
        // The 0.7 red multiplier leaves proportionally more red than the 0.6 G/B multipliers.
        assert!(boss[0] as f32 / base[0] as f32 > boss[1] as f32 / base[1] as f32);
        assert_eq!(boss[3], 255);
    }

    /// A cheap LCG, so the draws are noise-like rather than the short repeating cycle a
    /// `(i * 7) % 101` pattern produces. The old form of `step` is a random *walk*, and a walk
    /// driven by a periodic sequence with zero mean does not drift — which is exactly why the
    /// previous version of this test held while the bug was live.
    fn draws(seed: &mut u32) -> (f32, f32, f32) {
        let mut next = || {
            *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (*seed >> 8) as f32 / (1u32 << 24) as f32
        };
        (next(), next(), next())
    }

    #[test]
    fn torch_flicker_walks_but_stays_bounded() {
        // The bound is cadence-independent; 200,000 deterministic steps explore far beyond the
        // point where the old accumulating form becomes visibly unbounded.
        let mut flicker = TorchFlicker::default();
        let mut seed = 0x5eed_1234u32;
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for _ in 0..200_000 {
            let (a, b, c) = draws(&mut seed);
            flicker.step(a, b, c);
            let m = flicker.block_light_multiplier();
            assert!(m.is_finite(), "{m}");
            // `velocity` is `(velocity + e) * 0.9` with `|e| <= 1`, whose fixed point is
            // `|velocity| <= 0.9 / (1 - 0.9) = 9`, so the multiplier cannot leave `1.5 +- 0.9`.
            assert!(
                (0.6..=2.4).contains(&m),
                "torch flicker left the band vanilla keeps it in: {m}"
            );
            lo = lo.min(m);
            hi = hi.max(m);
        }
        // Not merely bounded: it has to actually move, or a `step` that did nothing would pass.
        assert!(hi - lo > 0.05, "flicker never moved: {lo}..{hi}");
        // And it stays near 1.5 in practice, not just inside the hard bound.
        assert!(
            lo > 1.0 && hi < 2.0,
            "unexpectedly wide in practice: {lo}..{hi}"
        );
    }

    /// The control for the test above: the accumulating form this module shipped before
    /// (`value += velocity` rather than `value = velocity`) must *fail* the same assertion on the
    /// same draws. Without this, "the multiplier stays in `0.6..=2.4`" could be true of any
    /// implementation and would not be evidence the defect is gone.
    #[test]
    fn the_accumulating_form_this_replaced_would_fail_that_bound() {
        let (mut velocity, mut value) = (0.0f32, 0.0f32);
        let mut seed = 0x5eed_1234u32;
        let mut escaped = false;
        let mut worst = 0.0f32;
        for _ in 0..200_000 {
            let (a, b, c) = draws(&mut seed);
            velocity += (a - b) * c * c;
            velocity *= 0.9;
            value += velocity; // <- the defect
            let m = value * 0.1 + 1.5;
            if !(0.6..=2.4).contains(&m) {
                escaped = true;
            }
            if m.abs() > worst.abs() {
                worst = m;
            }
        }
        assert!(
            escaped,
            "the old form was supposed to drift out of the band; it did not, so the test above \
             proves nothing"
        );
        // It does not merely leave the band, it goes negative — which is what turned torchlight
        // green and blue-green through `Lightmap::generate`'s quadratic and cubic block terms.
        assert!(worst < 0.0, "worst multiplier reached was {worst}");
    }

    /// Not a second regression test for [`TorchFlicker::step`] — this is the evidence that the
    /// *symptom* reported ("light sources randomly switching between normal lighting and
    /// radiating dark blue and dark green light") is the downstream consequence of a negative
    /// block-light multiplier, i.e. that the one-line fix above is aimed at the right thing.
    ///
    /// Both reported colours fall out of [`Lightmap::generate`], and they are different regimes
    /// of the same sign flip:
    ///
    /// * **green**, when `f3` is negative and large — `g`'s `0.36 * f3²` term outgrows its
    ///   `0.64 * f3` term at `f3 < -1.78`, so green climbs while red and blue clamp to zero;
    /// * **dark blue**, when `f3` is negative and small — every channel loses block light, but
    ///   `b` is built from the *full* sky term `f2` where `r`/`g` get only `f2 * (0.65f + 0.35)`,
    ///   so blue is the last channel left standing.
    #[test]
    fn a_negative_block_light_multiplier_is_what_inverts_the_torch_colour() {
        let warm = |flicker: f32, sun: f32, sky: u8, block: u8| {
            Lightmap::generate(LightmapParams {
                sun_brightness: sun,
                torch_flicker: flicker,
                ..Default::default()
            })
            .sample(sky, block)
        };

        // Control: vanilla's multiplier at a mid block level in an unlit room. Warm, red-first.
        // (A *high* block level saturates all three channels to white, which is why the level
        // matters here.)
        let [r, g, b, _] = warm(1.5, 0.0, 0, 7);
        assert!(r > g && g > b, "vanilla torchlight is warm: {r} {g} {b}");

        // Green regime: a large negative multiplier at full block light.
        let [gr, gg, gb, _] = warm(-20.0, 0.0, 0, 15);
        assert!(
            gg > gr && gg > gb,
            "a large negative multiplier should radiate green: {gr} {gg} {gb}"
        );

        // Dark-blue regime: a small negative multiplier with sky light present.
        let [br, bg, bb, _] = warm(-1.0, 0.3, 12, 7);
        assert!(
            bb > br && bb > bg,
            "a small negative multiplier should leave only blue: {br} {bg} {bb}"
        );

        // And in both regimes the torch is dimmer in red than it should be — the "light source
        // stopped looking like a light source" half of the report.
        assert!(
            gr < r && br < r,
            "red should collapse: {gr} and {br} vs {r}"
        );
    }

    #[test]
    fn night_vision_brightness_is_full_strength_with_time_to_spare() {
        assert_eq!(night_vision_brightness(300, 0.0), 1.0);
    }

    #[test]
    fn night_vision_brightness_flickers_as_it_expires() {
        let v = night_vision_brightness(10, 0.0);
        assert!((0.4..=1.0).contains(&v), "{v}");
    }

    #[test]
    fn lightmap_alpha_is_always_opaque() {
        let lm = Lightmap::generate(LightmapParams::default());
        for chunk in lm.rgba.chunks(4) {
            assert_eq!(chunk[3], 255);
        }
        assert_eq!(lm.rgba.len(), 256 * 4);
    }

    /// The non-vanilla `fullbright` option is off unless asked for, and off is the vanilla path.
    #[test]
    fn fullbright_defaults_off_and_off_is_the_vanilla_table() {
        assert!(!LightmapParams::default().fullbright);
        let vanilla = Lightmap::generate(LightmapParams {
            sun_brightness: 0.2,
            ..Default::default()
        });
        assert!(vanilla.rgba.iter().any(|&channel| channel < 255));
        let explicit_off = Lightmap::generate(LightmapParams {
            sun_brightness: 0.2,
            fullbright: false,
            ..Default::default()
        });
        assert_eq!(vanilla.rgba, explicit_off.rgba);
    }

    /// With `fullbright` every texel is 255 whatever the time, dimension, potion, boss, flash or
    /// gamma — i.e. the table a shader samples is constant white.
    #[test]
    fn fullbright_saturates_every_texel_whatever_the_other_inputs() {
        let cases = [
            LightmapParams::default(),
            LightmapParams {
                sun_brightness: 0.0,
                torch_flicker: -3.0,
                ..Default::default()
            },
            LightmapParams {
                dimension: Dimension::Nether,
                gamma: 1.0,
                ..Default::default()
            },
            LightmapParams {
                dimension: Dimension::End,
                boss_darkening: 1.0,
                ..Default::default()
            },
            LightmapParams {
                night_vision: 0.5,
                lightning_flash: true,
                ..Default::default()
            },
        ];
        for params in cases {
            let lm = Lightmap::generate(LightmapParams {
                fullbright: true,
                ..params
            });
            assert_eq!(lm.rgba, vec![255u8; 256 * 4], "{params:?}");
            assert_eq!(lm.rgba, Lightmap::full_bright().rgba);
        }
    }

    /// Why the option is an override rather than a large gamma: through vanilla's curve a channel
    /// already at exactly 1.0 ends at 252 however high gamma goes, so `gamma:100` is not uniform.
    #[test]
    fn huge_gamma_is_not_uniformly_full_bright() {
        let lm = Lightmap::generate(LightmapParams {
            gamma: 100.0,
            ..Default::default()
        });
        assert_eq!(lm.sample(15, 15)[0], 252, "{:?}", lm.sample(15, 15));
        assert_eq!(lm.sample(0, 0)[0], 255, "{:?}", lm.sample(0, 0));
    }
}
