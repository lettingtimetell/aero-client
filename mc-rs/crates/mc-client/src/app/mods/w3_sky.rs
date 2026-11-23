//! The client half of `custom_sky`, the resource pack's
//! custom sky layers.
//!
//! The parser is `mc_assets::custom_sky`; the textures, pipelines and draws are
//! `mc_render::sky::custom`, drawn from inside the sky phase (`SkyPhase::custom_sky`). This file
//! only decides when to load and attach:
//!
//! * **Off** (the default): nothing is loaded and the sky phase holds `None`. The frame hook is
//!   one `u64` compare and one `bool` test.
//! * **On:** the active packs' sky is loaded once per resource reload (or anaglyph change):
//!   parsed, decoded in parallel and uploaded. With the reference pack that is 21 layers over 15
//!   textures. The loaded sky is kept here, so a rebuilt render stack (`reinstall`) gets it back
//!   without decoding again. A pack without a custom sky leaves the vanilla sky.
//! * **Turned off:** the renderer is detached and the textures freed.
//!
//! The time of day comes from the sky phase's own `Environment`, which `sync_environment` fills
//! from `mod_display_world_time`, so the Time Changer drives the custom sky
//! too.

use std::sync::Arc;

use mc_render::sky::custom::{CustomSkyAssets, CustomSkyRenderer};
use mc_render::{Camera, Gpu};

use super::super::Client;

/// The mod's id in the catalog and the settings file.
const MOD_ID: &str = "custom_sky";

/// What the client last applied.
#[derive(Debug, Default)]
pub(crate) struct W3State {
    /// The settings generation last read. 0 = never (the store's generation starts at 1).
    seen: u64,
    /// The mod is on.
    enabled: bool,
    /// The `smooth` option.
    smooth: bool,
    /// The loaded sky and the resource generation and anaglyph setting it was loaded for.
    /// `Some((.., None))`: loaded, and the packs have no drawable sky.
    loaded: Option<(u64, bool, Option<Arc<CustomSkyAssets>>)>,
}

impl Client {
    /// How much of the sky custom sky layers cover in `dimension`, 0..1, as the sky pass last
    /// drew them (`CustomSkyRenderer::cover`): 0 while the mod is off, the packs have no layers
    /// there, or every layer is faded out by the time of day, the weather or its biome/height
    /// condition (the vanilla sky shows then). The shader packs ask, so their fog leaves such a
    /// sky in view below the horizon. A few loads and compares and one `RefCell` borrow, no lock.
    pub(crate) fn custom_sky_cover(&self, dimension: mc_render::sky::Dimension) -> f32 {
        let has_layers = self.w3.enabled
            && matches!(
                &self.w3.loaded,
                Some((_, _, Some(assets))) if assets.layer_count(dimension) > 0
            );
        if !has_layers {
            return 0.0;
        }
        self.render.as_ref().map_or(0.0, |render| {
            render.sky.with(|phase| {
                phase
                    .custom_sky
                    .as_ref()
                    .map_or(0.0, CustomSkyRenderer::cover)
            })
        })
    }

    /// Keep the sky phase's custom sky in step with the switch and the packs.
    /// Off: one `u64` compare and one `bool` test per frame.
    pub(crate) fn drive_custom_sky(&mut self, gpu: &Gpu, camera: &Camera) {
        let generation = self.mods.settings.generation();
        if self.w3.seen != generation {
            self.w3.seen = generation;
            self.w3.enabled = self.mods.settings.enabled(MOD_ID);
            self.w3.smooth = self.mods.settings.bool(MOD_ID, "smooth");
            if !self.w3.enabled && self.w3.loaded.take().is_some() {
                // Off: detach, and the textures go with the last `Arc`.
                if let Some(render) = &self.render {
                    render.sky.with(|phase| phase.custom_sky = None);
                }
                tracing::info!("custom sky off");
            }
        }
        if !self.w3.enabled {
            return;
        }
        let Some(render) = &self.render else {
            return;
        };
        let key = (self.assets.resources.generation(), self.settings.anaglyph);
        let stale = match &self.w3.loaded {
            None => true,
            Some((resources, anaglyph, assets)) => {
                (*resources, *anaglyph) != key
                    || assets.as_ref().is_some_and(|assets| !assets.fits(gpu))
            }
        };
        if stale {
            // Once per resource reload: a reload can change the packs' skies.
            let assets = CustomSkyAssets::load(gpu, &self.assets.resources, key.1);
            self.w3.loaded = Some((key.0, key.1, assets));
        }
        let Some((_, _, Some(assets))) = &self.w3.loaded else {
            // No drawable sky in these packs: the vanilla sky stays.
            render.sky.with(|phase| phase.custom_sky = None);
            return;
        };
        let smooth = self.w3.smooth;
        // The camera's place, only for packs with `biomes=`/`heights=` layers: the render-view
        // entity's block position.
        let position = if assets.needs_position() {
            self.session.as_ref().map(|session| {
                let p = session.render_view_position(1.0).unwrap_or(camera.pos);
                let block = mc_math::BlockPos::new(
                    p.x.floor() as i32,
                    p.y.floor() as i32,
                    p.z.floor() as i32,
                );
                (session.state().world.biome_at(block), block.y)
            })
        } else {
            None
        };
        render.sky.with(|phase| {
            let attached = phase
                .custom_sky
                .as_ref()
                .is_some_and(|renderer| Arc::ptr_eq(renderer.assets(), assets));
            if !attached {
                // A new sky, or a rebuilt sky phase (`reinstall`) that starts without one.
                phase.custom_sky = Some(CustomSkyRenderer::new(gpu, Arc::clone(assets)));
            }
            if let Some(renderer) = phase.custom_sky.as_mut() {
                renderer.smooth = smooth;
                renderer.camera_biome = position.map(|(biome, _)| biome);
                renderer.camera_y = position.map(|(_, y)| y);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};

    /// A headless client over the real assets, its game dir a temp dir (so the mod settings
    /// file can never land in the user's `mc-rs`). `None` without the vanilla jar.
    fn headless_client() -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("SKIP: no vanilla jar in this checkout (set MC_JARS_DIR)");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    fn headless_gpu() -> Option<Gpu> {
        match Gpu::headless(
            64,
            64,
            mc_render::GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(gpu) => Some(gpu),
            Err(mc_render::RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter on this machine");
                None
            }
            Err(error) => panic!("headless GPU: {error}"),
        }
    }

    /// A folder pack with one always-on overworld layer: a 3x2 PNG.
    fn sky_pack(dir: &std::path::Path) {
        let sky = dir.join("assets/minecraft/mcpatcher/sky/world0");
        std::fs::create_dir_all(&sky).unwrap();
        std::fs::write(
            dir.join("pack.mcmeta"),
            r#"{"pack":{"pack_format":1,"description":"test sky"}}"#,
        )
        .unwrap();
        std::fs::write(sky.join("sky1.properties"), "blend=add\nrotate=false\n").unwrap();
        let mut png = std::io::Cursor::new(Vec::new());
        image::ImageEncoder::write_image(
            image::codecs::png::PngEncoder::new(&mut png),
            &[200u8; 6 * 4],
            3,
            2,
            image::ExtendedColorType::Rgba8,
        )
        .unwrap();
        std::fs::write(sky.join("sky1.png"), png.into_inner()).unwrap();
    }

    fn attached(client: &Client) -> Option<Arc<CustomSkyAssets>> {
        let render = client.render.as_ref().expect("installed");
        render.sky.with(|phase| {
            phase
                .custom_sky
                .as_ref()
                .map(|renderer| Arc::clone(renderer.assets()))
        })
    }

    #[test]
    fn the_switch_loads_attaches_survives_a_rebuild_and_frees_the_sky() {
        let Some((mut client, dir)) = headless_client() else {
            return;
        };
        let Some(gpu) = headless_gpu() else { return };
        let mut phases = mc_render::PhaseList::new();
        client.install(&gpu, &mut phases, 64, 64);
        let camera = Camera::default();

        // Off (the default): nothing is loaded or attached.
        client.drive_custom_sky(&gpu, &camera);
        assert!(attached(&client).is_none());
        assert!(client.w3.loaded.is_none());

        // On, but the vanilla packs have no custom sky: still the vanilla sky.
        client.mods.settings.set_enabled(MOD_ID, true);
        client.drive_custom_sky(&gpu, &camera);
        assert!(attached(&client).is_none());
        assert!(matches!(client.w3.loaded, Some((_, _, None))));

        // A pack with a sky: a resource reload brings it in.
        let pack_dir = dir.path().join("testsky");
        sky_pack(&pack_dir);
        let mut packs = mc_assets_packs(&client);
        packs.push(Box::new(
            mc_assets::FolderResourcePack::open(&pack_dir).expect("test pack"),
        ));
        client.assets.resources.reload(packs);
        client.drive_custom_sky(&gpu, &camera);
        let first = attached(&client).expect("the sky is attached");
        assert_eq!(first.stats().layers, 1);
        assert_eq!(first.stats().textures, 1);
        assert!(client.render.as_ref().unwrap().sky.with(|phase| phase
            .custom_sky
            .as_ref()
            .unwrap()
            .has_layers(mc_render::sky::Dimension::Overworld)));

        // A frame with nothing new keeps the same sky (no reload, no new renderer).
        client.drive_custom_sky(&gpu, &camera);
        assert!(Arc::ptr_eq(&first, &attached(&client).unwrap()));

        // A rebuilt render stack gets the same sky back without decoding again.
        client.reinstall(&gpu, &mut phases, 64, 64);
        assert!(
            attached(&client).is_none(),
            "a new sky phase starts without it"
        );
        client.drive_custom_sky(&gpu, &camera);
        assert!(Arc::ptr_eq(&first, &attached(&client).unwrap()));

        // Smooth follows its option.
        client.mods.settings.set_bool(MOD_ID, "smooth", true);
        client.drive_custom_sky(&gpu, &camera);
        assert!(client.render.as_ref().unwrap().sky.with(|phase| phase
            .custom_sky
            .as_ref()
            .unwrap()
            .smooth));

        // Another resource reload loads the sky again.
        let packs = {
            let mut packs = mc_assets_packs(&client);
            packs.push(Box::new(
                mc_assets::FolderResourcePack::open(&pack_dir).expect("test pack"),
            ));
            packs
        };
        client.assets.resources.reload(packs);
        client.drive_custom_sky(&gpu, &camera);
        let second = attached(&client).expect("still attached");
        assert!(
            !Arc::ptr_eq(&first, &second),
            "reloaded for the new generation"
        );

        // Off: detached and freed (this test and the state hold the only other references).
        drop((first, second));
        client.mods.settings.set_enabled(MOD_ID, false);
        client.drive_custom_sky(&gpu, &camera);
        assert!(attached(&client).is_none());
        assert!(client.w3.loaded.is_none());
    }

    /// The jar (and asset store) packs the client was started with, opened again.
    fn mc_assets_packs(client: &Client) -> Vec<Box<dyn mc_assets::ResourcePack>> {
        crate::app::assets::build_pack_stack(&client.config, &client.settings)
    }
}
