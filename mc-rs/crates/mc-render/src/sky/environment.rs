//! World time, weather, and dimension state — the shared inputs every other `sky` module reads.
//!
//! java: `World.java` (`getCelestialAngle`, `getSkyColor`, `getStarBrightness`,
//! `getSunBrightness`, `getMoonPhase`, `getRainStrength`, `getThunderStrength`) and
//! `WorldProvider`/`WorldProviderSurface` (`calculateCelestialAngle`, `getFogColor`,
//! `calcSunriseSunsetColors`, `getMoonPhase`, `generateLightBrightnessTable`).
//!
//! **Neither file is in S20's assigned Java input list** (`EntityRenderer.java`,
//! `RenderGlobal.java`, `client/shader/`) — the formulas below are transcribed from documented,
//! stable vanilla behaviour (materially unchanged across 1.7-1.12) rather than read line-by-line
//! from this repository's copy of the source. Treat them as "known-good approximation", not a
//! verified port, and check against `net/minecraft/world/World.java` /
//! `net/minecraft/world/WorldProvider*.java` if bit-for-bit fidelity is ever required.
//!
//! `mc-world::World` has no notion of "the current time of day" (S08's scope stops at
//! block/light storage) — so [`Environment`] is the small piece of state a caller (S21,
//! eventually) drives every tick from `S03PacketTimeUpdate` / the rain/thunder packets, exactly
//! the way [`crate::terrain::phase::TerrainRenderer::sky_brightness`] is externally driven today.

use glam::Vec3;
use mc_math::{clamp_f32, cos, hsv_to_rgb, sin};

/// Which sky a dimension gets. java: `WorldProvider.getDimensionId` / `isSurfaceWorld`,
/// `WorldProviderHell`, `WorldProviderEnd`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Dimension {
    /// Full day/night cycle: sun, moon, stars, clouds, rain/snow. java: dimension id `0`.
    #[default]
    Overworld,
    /// No sky geometry at all — `RenderGlobal.renderSky`'s `isSurfaceWorld()` branch is skipped
    /// entirely, so there is no sun, moon, star or void-plane draw. java: dimension id `-1`.
    Nether,
    /// A static, starless, dark cube skybox (`renderSkyEnd`), fixed vertex colour `(40,40,40)`.
    /// java: dimension id `1`.
    End,
}

impl Dimension {
    /// Whether this dimension draws a sky dome / sun / moon / stars / void plane at all.
    /// java: `RenderGlobal.renderSky` — `provider.isSurfaceWorld()`
    #[must_use]
    pub fn has_sky_dome(self) -> bool {
        matches!(self, Dimension::Overworld)
    }

    /// Whether clouds render. java: `RenderGlobal.renderClouds` — `provider.isSurfaceWorld()`
    #[must_use]
    pub fn has_clouds(self) -> bool {
        matches!(self, Dimension::Overworld)
    }

    /// Whether weather (rain/snow/lightning) renders. Vanilla drives this from
    /// `WorldProvider.getWeatherRenderer`/`canDoRainSnowIce`-adjacent checks; the Nether and End
    /// never have precipitation.
    #[must_use]
    pub fn has_weather(self) -> bool {
        matches!(self, Dimension::Overworld)
    }

    /// The ambient light floor added to every light level before the day/night curve.
    /// java: `WorldProvider.generateLightBrightnessTable`'s `f` (`0.0` overworld/End, `0.1`
    /// Nether) — see [`crate::sky::lightmap::light_brightness_table`].
    #[must_use]
    pub fn ambient_light(self) -> f32 {
        match self {
            Dimension::Nether => 0.1,
            Dimension::Overworld | Dimension::End => 0.0,
        }
    }
}

/// Rain/thunder intensity, both `0.0..=1.0`, and the last-lightning countdown.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Weather {
    /// java: `World.getRainStrength(partialTicks)` — already interpolated by the caller.
    pub rain_strength: f32,
    /// java: `World.getThunderStrength(partialTicks)`.
    pub thunder_strength: f32,
    /// "How recently did lightning strike", java: `World.lastLightningBolt`: `0` = no recent
    /// strike, counts down from a caller-chosen value (vanilla sets it to `2` on a nearby
    /// strike). Drives [`Environment::lightning_flash`].
    pub last_lightning_bolt: u8,
}

/// Time of day, weather, and dimension: the S20 inputs a caller drives every tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Environment {
    /// java: `World.getWorldTime()` — ticks since world creation. Wraps every 24000 (one day);
    /// negative values (the client-side frozen-cycle convention can expose one, and direct
    /// callers/tests may exercise the full signed `long` range) are accepted with Java's
    /// truncating `%` and one-sided float wrap, exactly as
    /// `WorldProvider.calculateCelestialAngle` does. Do not replace that sequence with
    /// `rem_euclid`: for a negative remainder near one full day Java intentionally leaves the
    /// post-wrap fraction negative.
    pub world_time: i64,
    /// Which sky/weather/lightmap rules apply.
    pub dimension: Dimension,
    /// Rain/thunder/lightning state.
    pub weather: Weather,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            world_time: 6000, // vanilla's default new-world spawn time: mid-morning
            dimension: Dimension::Overworld,
            weather: Weather::default(),
        }
    }
}

impl Environment {
    /// `0.0..1.0`: `0.0` = noon (the sun sits at local `(0, 100, 0)`, on the Y axis, which
    /// [`crate::sky::dome::celestial_rotation`]'s `Ry(-90°)` leaves invariant, so an unrotated
    /// sun — angle `0.0` — is directly overhead), `~0.25` = sunset, `0.5` = midnight, `~0.75` =
    /// sunrise. java: `WorldProvider.calculateCelestialAngle`
    ///
    /// Easy to get backwards (an earlier draft of this port did): the `-0.25` offset in the
    /// formula is *not* "angle 0 is a quarter-day before noon", it re-centres a "ticks since
    /// dawn" fraction so that noon (`worldTime == 6000`, a quarter of the way through the day)
    /// lands on angle `0.0`.
    #[must_use]
    pub fn celestial_angle(&self, partial_ticks: f32) -> f32 {
        match self.dimension {
            // java: `WorldProviderHell.calculateCelestialAngle` returns a flat `0.5F` and
            // `WorldProviderEnd.calculateCelestialAngle` a flat `0.0F` — neither dimension has a
            // day. Without this the Nether's fog and lightmap brighten and darken on the
            // overworld's clock.
            Dimension::Nether => 0.5,
            Dimension::End => 0.0,
            Dimension::Overworld => {
                // java: `(int)(worldTime % 24000L)` followed by float arithmetic. The remainder
                // is already in i32 range, but keeping `%` (rather than `rem_euclid`) matters for
                // negative clocks because Java adds one only after the first `f < 0` check.
                let day_ticks = (self.world_time % 24_000) as f32;
                let mut f = (day_ticks + partial_ticks) / 24_000.0 - 0.25;
                if f < 0.0 {
                    f += 1.0;
                }
                if f > 1.0 {
                    f -= 1.0;
                }
                // A slow-in/slow-out ease around noon/midnight rather than a linear sweep.
                //
                // java: `1.0F - (float)((Math.cos((double)f * Math.PI) + 1.0D) / 2.0D)` — the
                // **double** `Math.cos`, not `MathHelper.cos`'s 65536-entry sine table. Verified
                // in the bytecode (`javap -c anm.class`, `invokestatic java/lang/Math.cos:(D)D`)
                // because the decompiled source of this method is lossy: it prints the last line as
                // `f = f + (f - f) / 3.0F`, having collapsed the `f1` temporary.
                let f1 = 1.0 - ((f64::from(f) * std::f64::consts::PI).cos() + 1.0) / 2.0;
                let f1 = f1 as f32;
                f + (f1 - f) / 3.0
            }
        }
    }

    /// `0.0..1.0` daylight scalar used to dim the lightmap's sky-light column.
    /// java: `World.getSunBrightness`
    #[must_use]
    pub fn sun_brightness(&self, partial_ticks: f32) -> f32 {
        let angle = self.celestial_angle(partial_ticks);
        let mut f1 = 1.0 - (cos(angle * std::f32::consts::PI * 2.0) * 2.0 + 0.2);
        f1 = clamp_f32(f1, 0.0, 1.0);
        f1 = 1.0 - f1;
        // java: `World.java:1424-1425` — both multiplies widen to `double` and narrow back, the
        // same shape `skylight_subtracted` below already reproduces.
        f1 = (f64::from(f1) * (1.0 - f64::from(self.weather.rain_strength * 5.0) / 16.0)) as f32;
        f1 = (f64::from(f1) * (1.0 - f64::from(self.weather.thunder_strength * 5.0) / 16.0)) as f32;
        f1 * 0.8 + 0.2
    }

    /// The sun-brightness sample consumed by `EntityRenderer.updateLightmap`.
    ///
    /// This is intentionally a tick-end sample: vanilla calls `World.getSunBrightness(1.0F)`
    /// while rebuilding the lightmap, even though `updateLightmap` itself receives the current
    /// frame's `partialTicks`. Keep the fixed input behind a zero-argument API so a render caller
    /// cannot accidentally feed its frame interpolation into the lightmap's sun term.
    /// java: `EntityRenderer.updateLightmap:927-932`.
    #[must_use]
    pub fn sun_brightness_for_lightmap(&self) -> f32 {
        self.sun_brightness(1.0)
    }

    /// `0.0..1.0`, the "how far below the horizon is the sun" scalar used to blend the sky
    /// colour toward black at night and to fade the sunset/sunrise glow disc.
    /// java: shared sub-expression of `World.getSkyColor`/`getStarBrightness`
    fn horizon_factor(&self, partial_ticks: f32) -> f32 {
        let angle = self.celestial_angle(partial_ticks);
        clamp_f32(
            cos(angle * std::f32::consts::PI * 2.0) * 2.0 + 0.5,
            0.0,
            1.0,
        )
    }

    /// `0.0..1.0`: stars fade in at dusk. java: `World.getStarBrightness`
    /// (`World.java:1596-1602`) — `f1 * f1 * 0.5F`, and **no rain term**. The rain fade is
    /// applied once by the caller, `RenderGlobal.renderSky:1325`'s
    /// `getStarBrightness(partialTicks) * f16` where `f16 = 1 - getRainStrength`. Folding it in
    /// here as well squared it, so light rain dimmed the stars roughly twice as fast as vanilla
    /// and full rain still reached zero, which is why it read as plausible.
    #[must_use]
    pub fn star_brightness(&self, partial_ticks: f32) -> f32 {
        let angle = self.celestial_angle(partial_ticks);
        let mut f1 = 1.0 - (cos(angle * std::f32::consts::PI * 2.0) * 2.0 + 0.25);
        f1 = clamp_f32(f1, 0.0, 1.0);
        f1 * f1 * 0.5
    }

    /// How much is subtracted from the **sky** channel before a light query answers, `0..=11`.
    /// java: `World.calculateSkylightSubtracted` (`world/World.java:1403`), which `World.tick`
    /// stores into `skylightSubtracted` (`:2456-2460`) and `Chunk.getLightSubtracted` (`:818`)
    /// applies.
    ///
    /// Deliberately **not** used by the terrain mesher: that reads `getCombinedLight`, which is
    /// undarkened, and [`crate::sky::lightmap::Lightmap`] applies the day/night curve on the GPU
    /// instead. This is for the queries that ask "how dark is it *here*" the way vanilla's
    /// gameplay code does — `World.getLight(pos, true)`, `getLightBrightness`, the F3 readout —
    /// and its consumer is [`mc_world::World::light_subtracted`].
    ///
    /// Note the shape differs from [`Environment::sun_brightness`] by more than the constants:
    /// this one clamps `1 - (cos*2 + 0.5)` (not `+ 0.2`), inverts *twice*, and truncates rather
    /// than rounding.
    #[must_use]
    pub fn skylight_subtracted(&self, partial_ticks: f32) -> u8 {
        let f = self.celestial_angle(partial_ticks);
        let mut f1 = 1.0 - (cos(f * std::f32::consts::PI * 2.0) * 2.0 + 0.5);
        f1 = clamp_f32(f1, 0.0, 1.0);
        f1 = 1.0 - f1;
        // java: these two are `double` multiplies on a `float` accumulator, and the widening is
        // observable at the `(int)` truncation below.
        f1 = (f64::from(f1) * (1.0 - f64::from(self.weather.rain_strength * 5.0) / 16.0)) as f32;
        f1 = (f64::from(f1) * (1.0 - f64::from(self.weather.thunder_strength * 5.0) / 16.0)) as f32;
        f1 = 1.0 - f1;
        (f1 * 11.0) as u8
    }

    /// `0..8`. java: `WorldProvider.getMoonPhase` — `(int)(worldTime / 24000L % 8L + 8L) % 8`,
    /// with Java's **truncating** division and remainder, not Euclidean ones. They agree for
    /// every non-negative time; `div_euclid`/`rem_euclid` disagree below zero.
    #[must_use]
    pub fn moon_phase(&self) -> u8 {
        ((self.world_time / 24_000 % 8 + 8) % 8) as u8
    }

    /// UV rect `(u0, v0, u1, v1)` of this moon phase within `moon_phases.png`'s 4x2 grid.
    /// java: `RenderGlobal.renderSky` moon-phase tiling
    #[must_use]
    pub fn moon_phase_uv(&self) -> (f32, f32, f32, f32) {
        let phase = i32::from(self.moon_phase());
        let col = phase % 4;
        let row = phase / 4 % 2;
        (
            col as f32 / 4.0,
            row as f32 / 2.0,
            (col + 1) as f32 / 4.0,
            (row + 1) as f32 / 2.0,
        )
    }

    /// `0.0..1.0`: how brightened everything looks the instant after nearby lightning.
    /// java: `World.getSkyColor`/`updateLightmap`'s `lastLightningBolt` checks.
    #[must_use]
    pub fn lightning_flash(&self, partial_ticks: f32) -> f32 {
        if self.weather.last_lightning_bolt == 0 {
            return 0.0;
        }
        // java: `f12 = (float)lastLightningBolt - partialTicks; if (f12 > 1.0F) f12 =
        // 1.0F; f12 *= 0.45F`. There is deliberately no lower clamp: preserving a negative
        // post-window sample is observable for a caller that supplies a partial tick outside
        // the renderer's usual `[0, 1]` interval, and mirrors the JVM branch exactly.
        let mut f = f32::from(self.weather.last_lightning_bolt) - partial_ticks;
        if f > 1.0 {
            f = 1.0;
        }
        f * 0.45
    }

    /// Base sky colour before biome tint, `0.0..1.0` per channel.
    /// java: `BiomeGenBase.getSkyColorByTemp` (`hsvToRGB(0.62222224 - t/3*0.05, 0.5 + t/3*0.1, 1.0)`)
    #[must_use]
    pub fn biome_sky_color(temperature: f32) -> Vec3 {
        let t = clamp_f32(temperature / 3.0, -1.0, 1.0);
        let packed = hsv_to_rgb(0.622_222_24 - t * 0.05, 0.5 + t * 0.1, 1.0);
        unpack_rgb(packed)
    }

    /// The sky's own colour: biome-tinted, darkened toward the horizon, rain/thunder/lightning
    /// adjusted. java: `World.getSkyColor` / `BiomeGenEnd.getSkyColorByTemp`
    #[must_use]
    pub fn sky_color(&self, biome_temperature: f32, partial_ticks: f32) -> Vec3 {
        // java: `BiomeGenEnd.getSkyColorByTemp` is the sole biome override: it returns packed
        // RGB `0`, before `World.getSkyColor` applies its horizon/weather/lightning stages.
        // Keep that at this call site instead of making the dimension-independent
        // `biome_sky_color` lie about BiomeGenBase's HSV formula.
        let mut c = match self.dimension {
            Dimension::End => Vec3::ZERO,
            Dimension::Overworld | Dimension::Nether => {
                Self::biome_sky_color(biome_temperature) * self.horizon_factor(partial_ticks)
            }
        };

        let rain = self.weather.rain_strength;
        if rain > 0.0 {
            let luma = (c.x * 0.3 + c.y * 0.59 + c.z * 0.11) * 0.6;
            let keep = 1.0 - rain * 0.75;
            c = c * keep + Vec3::splat(luma) * (1.0 - keep);
        }
        let thunder = self.weather.thunder_strength;
        if thunder > 0.0 {
            let luma = (c.x * 0.3 + c.y * 0.59 + c.z * 0.11) * 0.2;
            let keep = 1.0 - thunder * 0.75;
            c = c * keep + Vec3::splat(luma) * (1.0 - keep);
        }
        let flash = self.lightning_flash(partial_ticks);
        if flash > 0.0 {
            c = c * (1.0 - flash) + Vec3::new(0.8, 0.8, 1.0) * flash;
        }
        c
    }

    /// The world-wide fog base colour before any camera-local override (water/lava/void).
    ///
    /// java: `WorldProvider.getFogColor` (`WorldProvider.java:177`),
    /// `WorldProviderHell.getFogColor` (`:26` — a flat dark red) and
    /// `WorldProviderEnd.getFogColor` (`:50` — `0xA080A0 * 0.15`, day-independent).
    ///
    /// This is also the colour vanilla clears the framebuffer to every frame
    /// (`EntityRenderer.updateFogColor` → `GlStateManager.clearColor`), which is why
    /// [`crate::sky::dome::SkyPhase`] paints it as a full-screen backdrop: the sky and horizon
    /// planes do not quite meet at the horizon, and in vanilla it is the clear colour that shows
    /// through the seam.
    #[must_use]
    pub fn base_fog_color(&self, partial_ticks: f32) -> Vec3 {
        match self.dimension {
            // `0.20000000298023224` / `0.029999999329447746` are the `double` literals in
            // `WorldProviderHell.getFogColor`; as `f32` they are exactly 0.2 and 0.03.
            Dimension::Nether => Vec3::new(0.2, 0.03, 0.03),
            Dimension::End => Vec3::new(
                0xA0 as f32 / 255.0 * 0.15,
                0x80 as f32 / 255.0 * 0.15,
                0xA0 as f32 / 255.0 * 0.15,
            ),
            Dimension::Overworld => {
                let f = self.horizon_factor(partial_ticks);
                // java: `WorldProvider.getFogColor:181-183` — `0.7529412F`, `0.84705883F`,
                // `1.0F`. The green literal has to be written out to 9 significant figures:
                // `0.847_058_8f32` rounds to `0x3F58D8D8` and vanilla's is `0x3F58D8D9`.
                Vec3::new(
                    0.752_941_2 * (f * 0.94 + 0.06),
                    0.847_058_83 * (f * 0.94 + 0.06),
                    1.0 * (f * 0.91 + 0.09),
                )
            }
        }
    }

    /// The sunrise/sunset glow colour and its blend-in factor (`alpha`), or `None` outside the
    /// dawn/dusk window. java: `WorldProviderSurface.calcSunriseSunsetColors`
    #[must_use]
    pub fn sunrise_sunset_color(&self, partial_ticks: f32) -> Option<[f32; 4]> {
        // java: `WorldProviderEnd.calcSunriseSunsetColors` returns `null`. This matters to
        // `EntityRenderer.updateFogColor`, which otherwise invokes the base provider's sunrise
        // blend even though the End does not have a sunrise.
        if self.dimension == Dimension::End {
            return None;
        }
        let angle = self.celestial_angle(partial_ticks);
        let band = 0.4_f32;
        let f1 = cos(angle * std::f32::consts::PI * 2.0);
        if !(-band..=band).contains(&f1) {
            return None;
        }
        let f3 = (f1 / band) * 0.5 + 0.5;
        let mut f4 = 1.0 - (1.0 - sin(f3 * std::f32::consts::PI)) * 0.99;
        f4 *= f4;
        Some([f3 * 0.3 + 0.7, f3 * f3 * 0.7 + 0.2, f3 * f3 * 0.0 + 0.2, f4])
    }

    /// Cloud tint: the base cloud colour (`World.cloudColour`, `0xFFFFFF` unless a mod overrides
    /// it), greyed by rain, **dimmed by the day/night curve**, then greyed again by thunder — in
    /// that order. java: `World.getCloudColour` (`World.java:1520`)
    ///
    /// The day/night term (`World.java:1539-1541`) is the one an earlier pass of this port left
    /// out entirely, which left clouds glowing full white at midnight; it is what makes clouds go
    /// dark grey at night and it is deliberately applied *between* the rain and thunder greys,
    /// exactly where vanilla puts it.
    #[must_use]
    pub fn cloud_color(&self, partial_ticks: f32) -> Vec3 {
        let mut c = Vec3::splat(1.0);
        let rain = self.weather.rain_strength;
        if rain > 0.0 {
            let luma = (c.x * 0.3 + c.y * 0.59 + c.z * 0.11) * 0.6;
            let keep = 1.0 - rain * 0.95;
            c = c * keep + Vec3::splat(luma) * (1.0 - keep);
        }
        // java: World.java:1539 — `f2 * (f1 * 0.9F + 0.1F)`, blue on a slightly flatter ramp.
        let f1 = self.horizon_factor(partial_ticks);
        c.x *= f1 * 0.9 + 0.1;
        c.y *= f1 * 0.9 + 0.1;
        c.z *= f1 * 0.85 + 0.15;
        let thunder = self.weather.thunder_strength;
        if thunder > 0.0 {
            let luma = (c.x * 0.3 + c.y * 0.59 + c.z * 0.11) * 0.2;
            let keep = 1.0 - thunder * 0.95;
            c = c * keep + Vec3::splat(luma) * (1.0 - keep);
        }
        c
    }
}

fn unpack_rgb(packed: i32) -> Vec3 {
    let r = ((packed >> 16) & 0xFF) as f32 / 255.0;
    let g = ((packed >> 8) & 0xFF) as f32 / 255.0;
    let b = (packed & 0xFF) as f32 / 255.0;
    Vec3::new(r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn celestial_angle_is_zero_at_noon() {
        // Noon is when the sun is unrotated (directly overhead) — angle 0.0, not 0.25. See
        // `celestial_angle`'s doc comment for why this is the easy-to-flip direction.
        let env = Environment {
            world_time: 6000,
            ..Environment::default()
        };
        assert!(
            env.celestial_angle(0.0).abs() < 1e-3,
            "{}",
            env.celestial_angle(0.0)
        );
    }

    /// Vanilla's `calculateSkylightSubtracted` is `0` in broad daylight and `11` at
    /// midnight, and rain/thunder push it toward the dark end during the day.
    #[test]
    fn skylight_subtracted_runs_from_zero_at_noon_to_eleven_at_midnight() {
        let noon = Environment {
            world_time: 6_000,
            ..Environment::default()
        };
        assert_eq!(noon.skylight_subtracted(0.0), 0);

        let midnight = Environment {
            world_time: 18_000,
            ..Environment::default()
        };
        assert_eq!(midnight.skylight_subtracted(0.0), 11);

        // A thunderstorm at noon costs 5/16 of the daylight twice over.
        let storm = Environment {
            world_time: 6_000,
            weather: Weather {
                rain_strength: 1.0,
                thunder_strength: 1.0,
                ..Weather::default()
            },
            ..Environment::default()
        };
        assert_eq!(storm.skylight_subtracted(0.0), 5);

        // Never out of range, at any time of day or weather.
        for t in (0..24_000).step_by(97) {
            let env = Environment {
                world_time: t,
                ..Environment::default()
            };
            assert!(env.skylight_subtracted(0.0) <= 11, "t={t}");
        }
    }

    #[test]
    fn celestial_angle_is_a_quarter_turn_at_sunset() {
        let env = Environment {
            world_time: 12_000,
            ..Environment::default()
        };
        assert!(
            (env.celestial_angle(0.0) - 0.25).abs() < 0.05,
            "{}",
            env.celestial_angle(0.0)
        );
    }

    #[test]
    fn celestial_angle_wraps_across_midnight() {
        let env = Environment {
            world_time: 23_999,
            ..Environment::default()
        };
        let a = env.celestial_angle(0.0);
        let env2 = Environment {
            world_time: 0,
            ..Environment::default()
        };
        let b = env2.celestial_angle(0.0);
        // Both sit right at the wrap point, so they should be close together mod 1.0.
        assert!((a - b).abs() < 0.01 || (a - b - 1.0).abs() < 0.01 || (a - b + 1.0).abs() < 0.01);
    }

    /// `WorldProvider.calculateCelestialAngle` uses Java's signed `long %` and then performs
    /// only one `f += 1.0F` wrap. A Euclidean remainder looks tidier but changes the angle for
    /// negative clocks near the end of a day; these raw bits came from the deobfuscated JVM
    /// method (`WorldProvider.java:115-143`, Temurin 21 against the 1.8.9 class files).
    #[test]
    fn celestial_angle_keeps_java_signed_remainder_for_negative_clocks() {
        let near_day_end = Environment {
            world_time: -23_999,
            ..Environment::default()
        };
        assert_eq!(near_day_end.celestial_angle(0.0).to_bits(), 0xBDF1_5572);
        // Negative-control mutant: `rem_euclid(24_000)` yields the positive-day angle instead.
        assert_ne!(near_day_end.celestial_angle(0.0).to_bits(), 0x3F48_D8FD);

        let min_clock = Environment {
            world_time: i64::MIN,
            ..Environment::default()
        };
        assert_eq!(min_clock.celestial_angle(0.0).to_bits(), 0x3ED2_4756);
    }

    #[test]
    fn sun_is_bright_at_noon_and_dark_at_midnight() {
        let noon = Environment {
            world_time: 6000,
            ..Environment::default()
        };
        let midnight = Environment {
            world_time: 18_000,
            ..Environment::default()
        };
        assert!(
            noon.sun_brightness(0.0) > 0.9,
            "{}",
            noon.sun_brightness(0.0)
        );
        assert!(
            midnight.sun_brightness(0.0) < 0.3,
            "{}",
            midnight.sun_brightness(0.0)
        );
    }

    #[test]
    fn sun_brightness_is_dimmed_by_rain_and_thunder() {
        let clear = Environment {
            world_time: 6000,
            ..Environment::default()
        };
        let stormy = Environment {
            world_time: 6000,
            weather: Weather {
                rain_strength: 1.0,
                thunder_strength: 1.0,
                ..Weather::default()
            },
            ..Environment::default()
        };
        assert!(stormy.sun_brightness(0.0) < clear.sun_brightness(0.0));
    }

    #[test]
    fn lightmap_sun_brightness_uses_vanillas_tick_end_sample() {
        // At this point in the daylight curve, one partial tick is enough to produce a distinct
        // CPU value. That is the negative control: using the frame's `partial_ticks` would bite
        // here instead of accidentally agreeing with the fixed lightmap seam.
        let env = Environment {
            world_time: 12_000,
            ..Environment::default()
        };
        let lightmap_sample = env.sun_brightness_for_lightmap();
        assert_eq!(lightmap_sample.to_bits(), env.sun_brightness(1.0).to_bits());
        assert_ne!(
            lightmap_sample.to_bits(),
            env.sun_brightness(0.0).to_bits(),
            "the sampled daylight curve must distinguish 1.0 from a frame-start partial tick"
        );
    }

    #[test]
    fn stars_are_out_at_night_and_hidden_by_day() {
        let noon = Environment {
            world_time: 6000,
            ..Environment::default()
        };
        let midnight = Environment {
            world_time: 18_000,
            ..Environment::default()
        };
        assert_eq!(noon.star_brightness(0.0), 0.0);
        assert!(midnight.star_brightness(0.0) > 0.0);
    }

    /// java: `World.getStarBrightness` (`World.java:1596-1602`) carries **no** rain term; the
    /// fade is `RenderGlobal.renderSky:1325`'s `getStarBrightness(partialTicks) * f16`, where
    /// `f16 = 1.0F - getRainStrength(partialTicks)`. This used to be folded in here as well, so
    /// the caller squared it.
    #[test]
    fn rain_does_not_dim_the_stars_here_the_caller_does() {
        let storm = |rain: f32| Environment {
            world_time: 18_000,
            weather: Weather {
                rain_strength: rain,
                ..Weather::default()
            },
            ..Environment::default()
        };
        // Rain is invisible to this function.
        let clear = storm(0.0).star_brightness(0.0);
        assert!(clear > 0.0);
        assert_eq!(storm(1.0).star_brightness(0.0), clear);
        assert_eq!(storm(0.5).star_brightness(0.0), clear);
        // And `renderSky`'s own `* (1 - rainStrength)` is what takes it to zero. Applied twice
        // (the old shape) a half-strength shower would give `0.25 * clear`, not `0.5 * clear`.
        assert_eq!(storm(1.0).star_brightness(0.0) * (1.0 - 1.0), 0.0);
        assert!(
            (storm(0.5).star_brightness(0.0) * (1.0 - 0.5) - clear * 0.5).abs() < 1e-7,
            "one fade, not two"
        );
    }

    #[test]
    fn moon_phase_cycles_over_eight_days() {
        let env = |days: i64| Environment {
            world_time: days * 24_000,
            ..Environment::default()
        };
        let phases: Vec<u8> = (0..10).map(|d| env(d).moon_phase()).collect();
        assert_eq!(phases, vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 1]);
    }

    #[test]
    fn moon_phase_uses_java_truncating_days_at_boundaries_and_negative_times() {
        // WorldProvider.getMoonPhase is deliberately *not* a floor division.  In particular,
        // `-1 / 24000L` is zero in Java, so it remains the full moon; changing this to
        // `div_euclid` would select phase seven for every frozen/negative-clock control below.
        let phase = |world_time| {
            Environment {
                world_time,
                ..Environment::default()
            }
            .moon_phase()
        };

        assert_eq!(phase(-1), 0);
        assert_eq!(phase(-23_999), 0);
        assert_eq!(phase(-24_000), 7);
        assert_eq!(phase(-24_001), 7);
        assert_eq!(phase(-2 * 24_000), 6);

        // The change happens only at a complete 24,000-tick day boundary, not at a daily
        // time-of-day wrap.  These are also the controls that catch accidentally feeding the
        // renderer `world_time % 24000`, which would pin every night to phase zero.
        assert_eq!(phase(7 * 24_000 + 23_999), 7);
        assert_eq!(phase(8 * 24_000), 0);
        assert_eq!(phase(8 * 24_000 + 1), 0);
    }

    #[test]
    fn moon_phase_handles_the_full_i64_clock_range() {
        let phase = |world_time| {
            Environment {
                world_time,
                ..Environment::default()
            }
            .moon_phase()
        };

        // Keep the exact Java `long` quotient/remainder shape exercised near both ends of the
        // representable clock range.  `i64::MIN` is especially important because callers that
        // model the S03 freeze sign use wrapping negation for that wire value.
        assert_eq!(phase(i64::MIN), 6);
        assert_eq!(phase(i64::MAX), 2);
    }

    #[test]
    fn moon_phase_uv_covers_a_four_by_two_grid() {
        let env = Environment {
            world_time: 5 * 24_000,
            ..Environment::default()
        };
        assert_eq!(env.moon_phase(), 5);
        let (u0, v0, u1, v1) = env.moon_phase_uv();
        // phase 5 -> col 1, row 1
        assert!((u0 - 0.25).abs() < 1e-6);
        assert!((v0 - 0.5).abs() < 1e-6);
        assert!((u1 - 0.5).abs() < 1e-6);
        assert!((v1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn biome_sky_color_is_bluer_in_cold_biomes() {
        let cold = Environment::biome_sky_color(0.0);
        let hot = Environment::biome_sky_color(2.0);
        assert!(cold.z >= hot.z, "cold {cold:?} hot {hot:?}");
    }

    #[test]
    fn sky_color_darkens_toward_midnight() {
        let env_noon = Environment {
            world_time: 6000,
            ..Environment::default()
        };
        let env_midnight = Environment {
            world_time: 18_000,
            ..Environment::default()
        };
        let noon = env_noon.sky_color(0.8, 0.0);
        let midnight = env_midnight.sky_color(0.8, 0.0);
        assert!(noon.length() > midnight.length());
    }

    /// java: `BiomeGenEnd.getSkyColorByTemp` returns packed RGB `0`, not the inherited
    /// `BiomeGenBase` HSV blue. `World.getSkyColor` then leaves that black in every normal End
    /// frame (the End cannot rain, thunder, or receive lightning).
    #[test]
    fn end_sky_color_is_black_independent_of_temperature_clock_and_weather() {
        for (world_time, temperature, rain, thunder) in [
            (0, -20.0, 0.0, 0.0),
            (6_000, 0.5, 1.0, 0.0),
            (18_000, 3.0, 0.0, 1.0),
        ] {
            let end = Environment {
                world_time,
                dimension: Dimension::End,
                weather: Weather {
                    rain_strength: rain,
                    thunder_strength: thunder,
                    ..Weather::default()
                },
            };
            assert_eq!(end.sky_color(temperature, 0.5), Vec3::ZERO);
        }

        // Negative control: its registered biome temperature would be a conspicuous inherited
        // blue without BiomeGenEnd's override.
        let overworld = Environment {
            dimension: Dimension::Overworld,
            ..Environment::default()
        };
        assert_ne!(overworld.sky_color(0.5, 0.0), Vec3::ZERO);
    }

    #[test]
    fn rain_greys_out_the_sky() {
        let env = |rain: f32| Environment {
            world_time: 6000,
            weather: Weather {
                rain_strength: rain,
                ..Weather::default()
            },
            ..Environment::default()
        };
        let clear = env(0.0).sky_color(0.8, 0.0);
        let stormy = env(1.0).sky_color(0.8, 0.0);
        // Grey-out should shrink the spread between channels.
        let spread = |c: Vec3| (c.x - c.z).abs();
        assert!(spread(stormy) < spread(clear), "{stormy:?} vs {clear:?}");
    }

    #[test]
    fn lightning_flash_decays_to_zero() {
        // `last_lightning_bolt: 1` keeps `f = bolt - partial_ticks` inside the un-clamped 0..1
        // range across the sampled partial_ticks, so the decay is actually visible rather than
        // saturating at the `clamp(.., 0.0, 1.0)` ceiling for both samples.
        let env = Environment {
            weather: Weather {
                last_lightning_bolt: 1,
                ..Weather::default()
            },
            ..Environment::default()
        };
        assert!(env.lightning_flash(0.0) > env.lightning_flash(1.0));
        assert_eq!(env.lightning_flash(1.0), 0.0);
        let none = Environment::default();
        assert_eq!(none.lightning_flash(0.0), 0.0);
    }

    /// The JVM only caps a lightning flash above one; it does not clamp negative partial-tick
    /// samples back to zero. The normal renderer supplies `0..=1`, but this control protects the
    /// shared CPU helper from silently changing Java's post-window arithmetic.
    #[test]
    fn lightning_flash_preserves_java_upper_only_cap() {
        let env = Environment {
            weather: Weather {
                last_lightning_bolt: 1,
                ..Weather::default()
            },
            ..Environment::default()
        };
        assert_eq!(env.lightning_flash(2.0).to_bits(), (-0.45_f32).to_bits());
        assert_eq!(env.lightning_flash(-2.0).to_bits(), 0.45_f32.to_bits());
    }

    #[test]
    fn sunrise_glow_is_none_at_noon_and_present_near_dawn() {
        let noon = Environment {
            world_time: 6000,
            ..Environment::default()
        };
        assert!(noon.sunrise_sunset_color(0.0).is_none());

        // World time 0 sits at the sunrise boundary (celestial_angle ~0).
        let dawn = Environment {
            world_time: 0,
            ..Environment::default()
        };
        assert!(dawn.sunrise_sunset_color(0.0).is_some());
    }

    #[test]
    fn end_has_no_sunrise_sunset_fog_tint() {
        let end = Environment {
            dimension: Dimension::End,
            ..Environment::default()
        };
        // `WorldProviderEnd.calculateCelestialAngle` is zero, where the base provider would
        // otherwise report a sunrise glow. Its own `calcSunriseSunsetColors` returns null.
        assert!(end.sunrise_sunset_color(0.0).is_none());
    }

    #[test]
    fn cloud_color_greys_out_in_a_storm() {
        let clear = Environment::default().cloud_color(0.0);
        let storm = Environment {
            weather: Weather {
                rain_strength: 1.0,
                thunder_strength: 1.0,
                ..Weather::default()
            },
            ..Environment::default()
        }
        .cloud_color(0.0);
        assert!(storm.length() < clear.length());
    }

    #[test]
    fn dimension_gates_sky_geometry() {
        assert!(Dimension::Overworld.has_sky_dome());
        assert!(Dimension::Overworld.has_clouds());
        assert!(Dimension::Overworld.has_weather());
        assert!(!Dimension::Nether.has_sky_dome());
        assert!(!Dimension::Nether.has_clouds());
        assert!(!Dimension::End.has_sky_dome());
        assert_eq!(Dimension::Nether.ambient_light(), 0.1);
        assert_eq!(Dimension::Overworld.ambient_light(), 0.0);
    }
}
