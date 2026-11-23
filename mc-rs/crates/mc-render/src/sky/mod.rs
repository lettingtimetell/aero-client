//! Sky dome, weather, and atmosphere.
//!
//! # Layout
//!
//! | module | covers | java |
//! |---|---|---|
//! | [`environment`] | time of day, weather, dimension, sky/fog colour maths | `World`, `WorldProvider*` |
//! | [`lightmap`] | the 256-entry block/sky lightmap, gamma, night vision | `EntityRenderer.updateLightmap` |
//! | [`fog`] | fog colour/mode state plus [`fog::FogPhase`], a depth-driven screen-space fog pass | `EntityRenderer.updateFogColor`/`setupFog` |
//! | [`stars`] | the fixed-seed star field | `RenderGlobal.renderStars` |
//! | [`dome`] | sky/void plane, sun, moon, stars, the End's skybox — [`dome::SkyPhase`] | `RenderGlobal.renderSky`/`renderSkyEnd` |
//! | [`clouds`] | fast/fancy cloud grids — [`clouds::CloudsPhase`] | `RenderGlobal.renderClouds`/`renderCloudsFancy` |
//! | [`weather`] | rain/snow streaks and lightning bolts — [`weather::WeatherPhase`] | `EntityRenderer.renderRainSnow` |
//! | [`textures`] | loads `textures/environment/**` from a resource manager | — |
//!
//! # Installing the phases
//!
//! ```no_run
//! use mc_render::{Gpu, PhaseList, PhaseOrder};
//! use mc_render::sky::{clouds::CloudsPhase, dome::SkyPhase, fog::FogPhase, weather::WeatherPhase};
//!
//! # fn wire(gpu: &Gpu, phases: &mut PhaseList) {
//! phases.insert(PhaseOrder::SKY, Box::new(SkyPhase::new(gpu)));
//! phases.insert(PhaseOrder::SKY, Box::new(CloudsPhase::new(gpu)));
//! phases.insert(PhaseOrder::WEATHER, Box::new(WeatherPhase::new(gpu)));
//! phases.insert(FogPhase::ORDER, Box::new(FogPhase::new(gpu)));
//! # }
//! ```
//!
//! **[`dome::SkyPhase`] and [`clouds::CloudsPhase`] must both be registered at
//! [`crate::PhaseOrder::SKY`]**, i.e. before anything that draws depth. `SkyPhase` paints the fog
//! colour over the whole frame first (vanilla's `clearColor(fogColor…)`), and `CloudsPhase` borrows
//! the shared depth buffer for vanilla's two-pass cloud render and resets it to `1.0` afterwards.
//! Both are safe exactly where vanilla puts them and nowhere later; `CloudsPhase`'s
//! `RenderPhase::record` documents the reasoning and names the test that pins it.
//!
//! Call `load_textures`/`load_texture` on each phase once a [`mc_assets::ResourceManager`] is
//! up, and set `environment`/`partial_ticks` (and, for weather, call
//! [`weather::build_columns`] + [`weather::WeatherPhase::upload`] once per tick) from whatever
//! drives the game loop — see each phase's docs for exactly which fields are externally driven,
//! the same pattern [`crate::terrain::phase::TerrainRenderer::sky_brightness`] already uses.

pub mod clouds;
// Resource-pack custom sky layers.
pub mod custom;
pub mod dome;
pub mod environment;
pub mod fog;
pub mod lightmap;
pub mod stars;
pub mod textures;
pub mod volumetric;
pub mod weather;

pub use clouds::{CloudMode, CloudsPhase};
pub use dome::SkyPhase;
pub use environment::{Dimension, Environment, Weather};
pub use fog::{
    CameraMedium, FogDistance, FogInputs, FogLightSmoothing, FogMode, FogPhase, FogState,
};
pub use lightmap::{BossColorModifier, Lightmap, LightmapParams, TorchFlicker};
pub use stars::{generate_stars, StarQuad};
pub use weather::{
    add_rain_particles, build_columns, lightning_bolt_segments, BoltSegment, ColumnEnvironment,
    LightningStrike, PrecipColumn, RainAmbientPass, WeatherPhase, BOLT_COLOR,
};

/// [`ColumnEnvironment`] backed by a real [`mc_world::World`], for driving
/// [`weather::build_columns`] against live world state.
///
/// java: `World.getBiomeGenForCoords`/`getPrecipitationHeight`/`getCombinedLight`, including
/// the biome temperature's perlin-noise term — [`crate::terrain::tint::temperature_noise_at`]
/// is `BiomeGenBase.temperatureNoise`, which is seeded from the literal `1234L` and therefore
/// needs no world state at all. (This doc used to say `mc-math`/`mc-world` had no noise
/// generator and that the term was worldgen state; both were wrong.)
///
/// [`WorldColumns::precipitation_height`] used to scan 256 cells down from the top
/// of the world per column, one `HashMap` probe each, with no cache — at fancy graphics that is
/// a 21×21 grid of columns × up to 256 hashed lookups, on the render thread, *every frame it is
/// raining*. It now reads `mc_world::World::precipitation_height`, which is
/// `Chunk.precipitationHeightMap`: an array index, with the descent only re-run for a
/// column whose cache a block change invalidated.
pub struct WorldColumns<'w>(pub &'w mc_world::World);

impl ColumnEnvironment for WorldColumns<'_> {
    fn temperature_at(&self, x: i32, y: i32, z: i32) -> f32 {
        let biome_id = self.biome_id(x, z);
        mc_data::biome(biome_id).temperature_at(y, crate::terrain::tint::temperature_noise_at(x, z))
    }

    fn snows_at(&self, x: i32, y: i32, z: i32) -> bool {
        mc_data::biome(self.biome_id(x, z))
            .snows_at(y, crate::terrain::tint::temperature_noise_at(x, z))
    }

    fn precipitation_height(&self, x: i32, z: i32) -> i32 {
        // java: `Chunk.getPrecipitationHeight`'s `-1` means "nothing in this column"; the
        // streak then starts from the bottom of the world, which is what the old scan's `0`
        // fallback also did.
        self.0.precipitation_height(x, z).max(0)
    }

    fn light_at(&self, x: i32, y: i32, z: i32) -> u8 {
        self.0.light_at(mc_math::BlockPos::new(x, y, z))
    }

    fn has_precipitation(&self, x: i32, z: i32) -> bool {
        // java: `BiomeGenBase.canRain()` is `!enableSnow && enableRain`, and the gate is
        // `canRain() || getEnableSnow()` — which is exactly `rain || snow`.
        let b = mc_data::biome(self.biome_id(x, z));
        b.rain || b.snow
    }
}

impl WorldColumns<'_> {
    fn biome_id(&self, x: i32, z: i32) -> u8 {
        // `World::biome_at` keeps the resident-byte/sentinel/provider split in one place. The
        // y=0 probe is intentional: biome providers are column sources, while the y-dependent
        // temperature adjustment happens in `temperature_at` below.
        self.0.biome_at(mc_math::BlockPos::new(x, 0, z))
    }
}

#[cfg(test)]
mod world_columns_tests {
    use super::*;

    #[test]
    fn an_unloaded_world_uses_seed0_provider_and_no_precipitation_height() {
        let world = mc_world::World::new();
        let columns = WorldColumns(&world);
        // Java 1.8.9 `World.getBiomeGenForCoords` delegates an unloaded column to
        // `WorldChunkManager.getBiomeGenerator`, whose seed-0 DEFAULT GenLayer stack produces
        // Forest (id 4) at (0, 0); the old Plains assertion described the pre-provider stub.
        let forest = mc_data::biome(4);
        assert_eq!(columns.biome_id(0, 0), forest.id);
        assert!((columns.temperature_at(0, 64, 0) - forest.temperature).abs() < 1e-6);
        // No blocks loaded -> the scan finds nothing solid, so precipitation "falls" from 0.
        assert_eq!(columns.precipitation_height(0, 0), 0);
    }

    /// A one-section column of stone with a chosen biome id, built through a real chunk packet
    /// because that is the only way in: `World` hands out no `&mut Chunk`.
    fn stone_column_with_biome(biome: u8) -> mc_world::World {
        let mut w = mc_world::World::new();
        let len = mc_world::expected_chunk_data_len(0b1, true, true);
        let mut payload = vec![0u8; len];
        // `BlockState::new(BlockId(1), 0)` is raw `1 << 4 == 16`, little-endian.
        for cell in 0..4096 {
            payload[cell * 2] = 16;
        }
        for b in payload[len - 256..].iter_mut() {
            *b = biome;
        }
        w.apply_chunk_data(mc_math::ChunkPos::new(0, 0), true, 0b1, &payload)
            .expect("well-formed payload");
        w
    }

    /// `EntityRenderer.renderRainSnow:1639` skips a column entirely unless
    /// `canRain() || getEnableSnow()`, so a desert stays dry no matter the weather.
    #[test]
    fn a_desert_column_has_no_precipitation_but_plains_does() {
        let desert = stone_column_with_biome(2);
        assert!(!WorldColumns(&desert).has_precipitation(3, 3));
        let plains = stone_column_with_biome(1);
        assert!(WorldColumns(&plains).has_precipitation(3, 3));
        // Ice plains (12) snow rather than rain, and still count.
        let ice = stone_column_with_biome(12);
        assert!(WorldColumns(&ice).has_precipitation(3, 3));
        // And the gate really removes the columns. The camera sits in the middle of the one
        // loaded column so the whole radius-5 disc lands on real biome data — an unloaded
        // column falls back to plains, which does rain.
        let eye = mc_math::DVec3::new(8.0, 64.0, 8.0);
        let dry = build_columns(&WorldColumns(&desert), eye, 0, false, 1.0);
        assert!(dry.is_empty(), "{} desert columns still rained", dry.len());
        let wet = build_columns(&WorldColumns(&plains), eye, 0, false, 1.0);
        assert!(!wet.is_empty());
    }

    #[test]
    fn world_columns_rain_only_cold_biome_routes_through_biome_snows_at() {
        // Extreme Hills is rain-enabled but not snow-flagged. Above y=64, vanilla's height
        // adjustment cools it below EntityRenderer's strict 0.15 precipitation split, so this
        // is the biting case for accidentally replacing `(rain || snow)` with `snow`.
        let extreme_hills = stone_column_with_biome(3);
        let columns = WorldColumns(&extreme_hills);
        let y = 128;
        assert!(mc_data::biome(3).rain);
        assert!(!mc_data::biome(3).snow);
        assert!(columns.temperature_at(3, y, 3) < 0.15);
        assert!(columns.snows_at(3, y, 3));

        let eye = mc_math::DVec3::new(8.0, 133.0, 8.0);
        let precipitation = build_columns(&columns, eye, 0, false, 1.0);
        assert!(!precipitation.is_empty());
        assert!(
            precipitation.iter().all(|column| column.is_snow),
            "rain-enabled cold Extreme Hills columns must use the live biome snow predicate"
        );
    }

    /// This used to be a 256-deep scan with a `HashMap` probe per step. It is now
    /// `Chunk.precipitationHeightMap`, and it agrees with the scan it replaced.
    #[test]
    fn precipitation_height_reads_the_chunks_cached_map() {
        let w = stone_column_with_biome(1);
        // Section 0 is solid stone, so rain lands on top of y = 15.
        assert_eq!(WorldColumns(&w).precipitation_height(3, 3), 16);
    }

    #[test]
    fn light_at_delegates_to_the_worlds_combined_light() {
        let world = mc_world::World::new();
        let columns = WorldColumns(&world);
        // An unloaded overworld-flagged column reports full skylight (`World::sky_light_at`'s
        // documented default), so `light_at` should agree with a direct query.
        let direct = world.light_at(mc_math::BlockPos::new(3, 70, 3));
        assert_eq!(columns.light_at(3, 70, 3), direct);
    }
}
