//! Biome table.
//!
//! Java input: `net/minecraft/world/biome/BiomeGenBase.java`. Only the client-visible
//! fields survive - temperature and rainfall drive grass/foliage tinting and snow-vs-rain
//! weather, and `water_color` tints water directly. Everything to do with terrain
//! generation is out of scope per §2.

use crate::generated::biomes::BIOMES;

/// Client-visible properties of a biome.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Biome {
    /// Biome id as it appears in the per-column biome array of a chunk packet.
    pub id: u8,
    /// Display name, e.g. `Extreme Hills`.
    pub name: &'static str,
    /// Originating Java class.
    pub java_class: &'static str,
    /// Whether this table row is registered. [`biome`] substitutes Plains for false rows,
    /// matching `Chunk.getBiome`; the generated table keeps them for exhaustive validation.
    pub registered: bool,
    /// Packed RGB biome colour (used by maps and the F3 overlay).
    pub color: u32,
    /// `BiomeGenBase.field_150609_ah`, copied from the registry metadata. Vanilla grass tinting
    /// uses temperature/rainfall colourizers and biome-class overrides instead of this field.
    pub grass_color_base: u32,
    /// Temperature. Nominally `0.0..=2.0`, but cold taiga is `-0.5`, so do not assume a
    /// lower bound of zero. Below `0.15` precipitation falls as snow.
    pub temperature: f32,
    /// Rainfall, `0.0..=1.0`.
    pub rainfall: f32,
    /// Packed RGB water tint.
    pub water_color: u32,
    /// Terrain base height copied from the registry metadata; no current tint, F3, or terrain
    /// generation consumer reads it.
    pub min_height: f32,
    /// Terrain height variation copied from the registry metadata; no current client consumer
    /// reads it.
    pub max_height: f32,
    /// Whether this biome snows instead of raining.
    pub snow: bool,
    /// Whether this biome has precipitation at all.
    pub rain: bool,
    /// `rainfall > 0.85`, which vanilla uses for a few visual effects.
    pub high_humidity: bool,
}

impl Biome {
    /// Vanilla's `BiomeGenBase.getFloatTemperature`: above y=64 the temperature falls off
    /// with height, which is what makes mountain tops snowy in an otherwise mild biome.
    ///
    /// Vanilla adds a per-column perlin term, `noise(x/8, z/8) * 4`, before the height
    /// falloff. `BiomeGenBase.temperatureNoise` has the fixed seed `1234L`, independent of the
    /// world seed; the render layer evaluates it from `(x, z)` and passes the result through
    /// `noise` (or callers may pass `0.0` to request the noise-free curve).
    // java: BiomeGenBase.java getFloatTemperature
    #[must_use]
    pub fn temperature_at(&self, y: i32, noise: f32) -> f32 {
        if y > 64 {
            self.temperature - (noise + y as f32 - 64.0) * 0.05 / 30.0
        } else {
            self.temperature
        }
    }

    /// Whether precipitation at this height falls as snow. `noise` is the same per-column
    /// term [`Biome::temperature_at`] takes.
    #[must_use]
    pub fn snows_at(&self, y: i32, noise: f32) -> bool {
        // Java branches to rain only when `temperature >= 0.15F`; spelling the
        // predicate this way also preserves its NaN behavior (NaN falls through
        // to snow instead of satisfying neither side of an ordered comparison).
        (self.rain || self.snow)
            && matches!(
                self.temperature_at(y, noise).partial_cmp(&0.15),
                Some(std::cmp::Ordering::Less) | None
            )
    }
}

/// Properties for a biome id. Frozen S01 API.
///
/// Total: every `u8` resolves. Unregistered ids resolve to Plains, matching the null substitution
/// at the tail of vanilla's `Chunk.getBiome`; the biome array in a chunk packet is raw server data.
///
/// # Example
/// ```
/// use mc_data::biome;
/// assert_eq!(biome(1).name, "Plains");
/// assert!(biome(2).temperature > 1.0);      // Desert
/// assert_eq!(biome(200).name, "Plains");
/// ```
#[inline]
#[must_use]
pub fn biome(id: u8) -> &'static Biome {
    let biome = &BIOMES[id as usize];
    if biome.registered {
        biome
    } else {
        &BIOMES[1]
    }
}

/// Every biome vanilla registers, ascending by id.
pub fn biomes() -> impl Iterator<Item = &'static Biome> {
    BIOMES.iter().filter(|b| b.registered)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_biome_id_resolves_without_panic() {
        for id in 0..=u8::MAX {
            let b = biome(id);
            let expected = if BIOMES[id as usize].registered {
                id
            } else {
                1
            };
            assert_eq!(b.id, expected);
            assert!(
                b.registered,
                "unregistered ids substitute registered Plains"
            );
            assert!(b.temperature.is_finite());
            assert!(b.rainfall.is_finite());
            let _ = b.snows_at(200, 0.0);
        }
    }

    #[test]
    fn cold_taiga_temperature_is_negative() {
        // A real vanilla value that trips any `temperature >= 0.0` assumption.
        let cold = biomes()
            .find(|b| b.name == "Cold Taiga")
            .expect("Cold Taiga");
        assert!(cold.temperature < 0.0, "{}", cold.temperature);
    }

    #[test]
    fn known_biomes() {
        assert_eq!(biome(0).name, "Ocean");
        assert_eq!(biome(1).name, "Plains");
        assert_eq!(biome(2).name, "Desert");
        assert!(!biome(2).rain, "desert has no rain");
        assert!(biome(12).snow, "ice plains snows");
        assert_eq!(biome(8).name, "Hell");
    }

    #[test]
    fn unregistered_wire_ids_fall_back_to_plains() {
        for id in [40, 42, 127, 128, 168, 254, 255] {
            assert!(!BIOMES[id as usize].registered, "negative control: id {id}");
            assert_eq!(biome(id), biome(1), "raw biome id {id}");
        }
        assert_ne!(
            BIOMES[42].temperature, BIOMES[1].temperature,
            "the generated placeholder must differ so this test bites on the old lookup"
        );
    }

    #[test]
    fn height_cools_the_air() {
        let plains = biome(1);
        assert!(plains.temperature_at(128, 0.0) < plains.temperature_at(64, 0.0));
        assert_eq!(plains.temperature_at(64, 0.0), plains.temperature);
        assert!(!plains.snows_at(64, 0.0));
    }

    #[test]
    fn snow_only_biomes_use_the_same_strict_temperature_gate_as_weather() {
        let snow_only = Biome {
            rain: false,
            snow: true,
            temperature: 0.0,
            ..*biome(12)
        };
        assert!(snow_only.snows_at(64, 0.0));

        let boundary = Biome {
            temperature: 0.15,
            ..snow_only
        };
        assert!(
            !boundary.snows_at(64, 0.0),
            "exactly 0.15 is rain, not snow"
        );
        let unknown_temperature = Biome {
            temperature: f32::NAN,
            ..snow_only
        };
        assert!(
            unknown_temperature.snows_at(64, 0.0),
            "Java's >= 0.15 rain branch falls through to snow for NaN"
        );
    }

    #[test]
    fn registered_count_matches_vanilla() {
        // 1.8.9 registers 40 base biomes plus their mutated (+128) variants.
        assert_eq!(biomes().count(), 61);
    }
}
