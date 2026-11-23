//! [`CoreConfig`]: the core features a pack switches on.
//!
//! A pack is a set of its own phases plus one [`CoreConfig`]. The running pack's client glue
//! hands its config to [`ShaderCore::set_config`](super::ShaderCore::set_config) whenever its
//! settings change; `None` in a field switches that feature off and frees its GPU resources.
//! Every value is sanitized on the way in ([`CoreConfig::sanitized`]), so a hand-edited settings
//! file can never reach the GPU as a NaN or an absurd size.

/// Sun (or moon) shadows from a shadow map.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowSettings {
    /// Half the side of the shadowed square around the player, in blocks (32..=256).
    pub distance: f32,
    /// Shadow-map edge in texels: 1024, 2048 or 4096.
    pub resolution: u32,
    /// Penumbra width in shadow-map texels (0 = hard, up to 6).
    pub softness: f32,
    /// How much of the sky light a shadow takes away, 0..=1. Torches, lava and every other block
    /// light are never darkened by a shadow.
    pub strength: f32,
    /// Alpha-tested blocks (leaves, grass, flowers, glass panes) cast shadows.
    pub cutout_casters: bool,
    /// Players, mobs and dropped items cast shadows onto the world (they never receive them:
    /// entities keep vanilla's lighting, so a player in shade stays as easy to see).
    pub entity_casters: bool,
}

impl Default for ShadowSettings {
    fn default() -> Self {
        ShadowSettings {
            distance: 96.0,
            resolution: 2048,
            softness: 1.5,
            strength: 0.6,
            cutout_casters: true,
            entity_casters: true,
        }
    }
}

/// Direction-aware sun shading in place of vanilla's fixed per-face shading.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightingSettings {
    /// The share of vanilla's daylight that comes straight from the sun (the rest is sky light
    /// from every direction), 0..=1. Higher = stronger contrast between lit and unlit sides.
    pub sun_share: f32,
    /// Golden sunrise and sunset colour, 0..=2 (1 = default).
    pub warmth: f32,
    /// Blue moonlight at night, 0..=2 (1 = default).
    pub night_blue: f32,
    /// The darkest a face turned away from the light gets, as a fraction of its sunlit
    /// brightness (0.2..=1). PvP: keep it high enough that shaded walls stay readable.
    pub shade_floor: f32,
}

impl Default for LightingSettings {
    fn default() -> Self {
        LightingSettings {
            sun_share: 0.55,
            warmth: 1.0,
            night_blue: 1.0,
            shade_floor: 0.55,
        }
    }
}

/// Shader water: wave normals, fresnel, reflections, a clearer colour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaterSettings {
    /// Wave normal strength, 0..=2 (0 = flat mirror).
    pub waves: f32,
    /// Screen-space reflections (else the sky colour only).
    pub reflections: bool,
    /// How see-through the water is, 0..=1 (0 = vanilla opacity, 1 = clearest).
    pub clarity: f32,
    /// Water tint (linear rgb), multiplied with the biome-tinted texture.
    pub tint: [f32; 3],
}

impl Default for WaterSettings {
    fn default() -> Self {
        WaterSettings {
            waves: 1.0,
            reflections: true,
            clarity: 0.45,
            tint: [1.0, 1.0, 1.0],
        }
    }
}

/// Plants and leaves sway in the wind (vertex shader).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WavingSettings {
    /// Grass, flowers, crops and other plants.
    pub plants: bool,
    /// Leaves and vines.
    pub leaves: bool,
    /// Amplitude multiplier, 0..=2 (1 = default, already subtle).
    pub strength: f32,
    /// Speed multiplier, 0.25..=3.
    pub speed: f32,
}

impl Default for WavingSettings {
    fn default() -> Self {
        WavingSettings {
            plants: true,
            leaves: true,
            strength: 1.0,
            speed: 1.0,
        }
    }
}

/// Bloom: bright areas glow (dual-filter, half resolution down to 1/32).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BloomSettings {
    /// How much glow is added, 0..=1.
    pub strength: f32,
    /// Brightness (linear luminance, 0..=1) where glow starts.
    pub threshold: f32,
    /// How far the glow spreads, 0..=1 (more blur levels).
    pub radius: f32,
}

impl Default for BloomSettings {
    fn default() -> Self {
        BloomSettings {
            strength: 0.12,
            threshold: 0.72,
            radius: 0.6,
        }
    }
}

/// The tone curve of [`TonemapSettings`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToneCurve {
    /// A soft filmic curve (shoulder and toe, hue preserving). The Classic default.
    #[default]
    Filmic,
    /// ACES-like (a fitted curve): punchier, a little warmer in the highlights.
    Aces,
    /// No curve: exposure, contrast and colour only.
    Linear,
}

/// Tone mapping and colour: the last step before the HUD.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TonemapSettings {
    /// The curve.
    pub curve: ToneCurve,
    /// Exposure in stops, -2..=2.
    pub exposure: f32,
    /// Contrast, 0.5..=1.5 (1 = unchanged).
    pub contrast: f32,
    /// Saturation, 0..=2 (1 = unchanged).
    pub saturation: f32,
    /// Vibrance: saturates muted colours more than vivid ones, -1..=1.
    pub vibrance: f32,
}

impl Default for TonemapSettings {
    fn default() -> Self {
        TonemapSettings {
            curve: ToneCurve::Filmic,
            exposure: 0.0,
            contrast: 1.05,
            saturation: 1.05,
            vibrance: 0.15,
        }
    }
}

/// Which core features a pack uses. `None` = off.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CoreConfig {
    /// Tilt of the sun's and moon's path toward the south, degrees (-45..=45, 0 = vanilla's path
    /// straight overhead). A tilted path gives noon shadows a direction and lights south walls;
    /// the sky draws the sun and moon on the same path while the pack runs.
    pub sun_tilt: f32,
    /// Sun and moon shadows.
    pub shadows: Option<ShadowSettings>,
    /// Direction-aware sun shading (also what makes shadows warm and cool).
    pub lighting: Option<LightingSettings>,
    /// Shader water.
    pub water: Option<WaterSettings>,
    /// Waving plants and leaves.
    pub waving: Option<WavingSettings>,
    /// Bloom.
    pub bloom: Option<BloomSettings>,
    /// Tone mapping and colour.
    pub tonemap: Option<TonemapSettings>,
}

impl CoreConfig {
    /// Every feature off.
    pub const OFF: CoreConfig = CoreConfig {
        sun_tilt: 0.0,
        shadows: None,
        lighting: None,
        water: None,
        waving: None,
        bloom: None,
        tonemap: None,
    };

    /// Whether the terrain draws with the pack's pipelines (any terrain-side feature is on).
    #[must_use]
    pub fn uses_terrain_variant(&self) -> bool {
        self.shadows.is_some()
            || self.lighting.is_some()
            || self.water.is_some()
            || self.waving.is_some()
    }

    /// Whether the final pass (bloom, tone map) runs.
    #[must_use]
    pub fn uses_final_pass(&self) -> bool {
        self.bloom.is_some() || self.tonemap.is_some()
    }

    /// What the mesher must write for this config (see [`PackMesh`]). Sun shading needs the
    /// flags too: it lights plants and leaves without a face shade, as vanilla draws them.
    #[must_use]
    pub fn mesh(&self) -> PackMesh {
        PackMesh {
            flat_shade: self.lighting.is_some(),
            flags: self.waving.is_some() || self.water.is_some() || self.lighting.is_some(),
        }
    }

    /// The config with every value clamped into its documented range and every non-finite value
    /// replaced by its default.
    #[must_use]
    pub fn sanitized(self) -> CoreConfig {
        fn fin(value: f32, fallback: f32, lo: f32, hi: f32) -> f32 {
            if value.is_finite() {
                value.clamp(lo, hi)
            } else {
                fallback
            }
        }
        let shadows = self.shadows.map(|s| {
            let d = ShadowSettings::default();
            ShadowSettings {
                distance: fin(s.distance, d.distance, 32.0, 256.0),
                resolution: match s.resolution {
                    0..=1535 => 1024,
                    1536..=3071 => 2048,
                    _ => 4096,
                },
                softness: fin(s.softness, d.softness, 0.0, 6.0),
                strength: fin(s.strength, d.strength, 0.0, 1.0),
                cutout_casters: s.cutout_casters,
                entity_casters: s.entity_casters,
            }
        });
        let lighting = self.lighting.map(|l| {
            let d = LightingSettings::default();
            LightingSettings {
                sun_share: fin(l.sun_share, d.sun_share, 0.0, 1.0),
                warmth: fin(l.warmth, d.warmth, 0.0, 2.0),
                night_blue: fin(l.night_blue, d.night_blue, 0.0, 2.0),
                shade_floor: fin(l.shade_floor, d.shade_floor, 0.2, 1.0),
            }
        });
        let water = self.water.map(|w| {
            let d = WaterSettings::default();
            WaterSettings {
                waves: fin(w.waves, d.waves, 0.0, 2.0),
                reflections: w.reflections,
                clarity: fin(w.clarity, d.clarity, 0.0, 1.0),
                tint: [
                    fin(w.tint[0], 1.0, 0.0, 2.0),
                    fin(w.tint[1], 1.0, 0.0, 2.0),
                    fin(w.tint[2], 1.0, 0.0, 2.0),
                ],
            }
        });
        let waving = self.waving.map(|w| {
            let d = WavingSettings::default();
            WavingSettings {
                plants: w.plants,
                leaves: w.leaves,
                strength: fin(w.strength, d.strength, 0.0, 2.0),
                speed: fin(w.speed, d.speed, 0.25, 3.0),
            }
        });
        let bloom = self.bloom.map(|b| {
            let d = BloomSettings::default();
            BloomSettings {
                strength: fin(b.strength, d.strength, 0.0, 1.0),
                threshold: fin(b.threshold, d.threshold, 0.0, 1.0),
                radius: fin(b.radius, d.radius, 0.0, 1.0),
            }
        });
        let tonemap = self.tonemap.map(|t| {
            let d = TonemapSettings::default();
            TonemapSettings {
                curve: t.curve,
                exposure: fin(t.exposure, d.exposure, -2.0, 2.0),
                contrast: fin(t.contrast, d.contrast, 0.5, 1.5),
                saturation: fin(t.saturation, d.saturation, 0.0, 2.0),
                vibrance: fin(t.vibrance, d.vibrance, -1.0, 1.0),
            }
        });
        CoreConfig {
            sun_tilt: fin(self.sun_tilt, 0.0, -45.0, 45.0),
            shadows,
            lighting,
            water,
            waving,
            bloom,
            tonemap,
        }
    }
}

/// What the terrain mesher writes for a pack. The default
/// (`flat_shade: false, flags: false`) is vanilla meshing, byte for byte; any other value
/// re-meshes the world once when it changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct PackMesh {
    /// Leave vanilla's fixed per-face shade (top 1.0, sides 0.8 and 0.6, bottom 0.5) out of the
    /// vertex colour: the pack's sun shading replaces it.
    pub flat_shade: bool,
    /// Write material flags (plants, leaves, water) into the vertex's two spare light bytes, for
    /// waving plants and shader water.
    pub flags: bool,
}

impl PackMesh {
    /// Vanilla meshing.
    pub const VANILLA: PackMesh = PackMesh {
        flat_shade: false,
        flags: false,
    };

    /// Whether this is vanilla meshing.
    #[must_use]
    pub fn is_vanilla(self) -> bool {
        self == PackMesh::VANILLA
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizing_clamps_and_replaces_non_finite_values() {
        let config = CoreConfig {
            shadows: Some(ShadowSettings {
                distance: f32::NAN,
                resolution: 9000,
                softness: -1.0,
                strength: 2.0,
                cutout_casters: true,
                entity_casters: false,
            }),
            bloom: Some(BloomSettings {
                strength: f32::INFINITY,
                threshold: 0.5,
                radius: 0.5,
            }),
            ..CoreConfig::OFF
        }
        .sanitized();
        let shadows = config.shadows.unwrap();
        assert_eq!(shadows.distance, ShadowSettings::default().distance);
        assert_eq!(shadows.resolution, 4096);
        assert_eq!(shadows.softness, 0.0);
        assert_eq!(shadows.strength, 1.0);
        assert_eq!(
            config.bloom.unwrap().strength,
            BloomSettings::default().strength
        );
    }

    #[test]
    fn the_mesh_mode_follows_the_features() {
        assert!(CoreConfig::OFF.mesh().is_vanilla());
        let lit = CoreConfig {
            lighting: Some(LightingSettings::default()),
            ..CoreConfig::OFF
        };
        assert_eq!(
            lit.mesh(),
            PackMesh {
                flat_shade: true,
                flags: true
            }
        );
        let post_only = CoreConfig {
            bloom: Some(BloomSettings::default()),
            tonemap: Some(TonemapSettings::default()),
            ..CoreConfig::OFF
        };
        assert!(post_only.mesh().is_vanilla());
        assert!(!post_only.uses_terrain_variant());
        assert!(post_only.uses_final_pass());
    }
}
