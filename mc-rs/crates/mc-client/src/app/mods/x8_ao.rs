//! The client half of `ambient_occlusion` and
//! `anisotropic_filtering`.
//!
//! * Ambient Occlusion is one render phase, [`AoPhase`] (`mc_render::post::ao`), registered at
//!   `PhaseOrder::AMBIENT_OCCLUSION` (340): after opaque terrain and entities, before the hitboxes
//!   (342) and nameplates (345, both moved after AO by X15), the selection outline, particles,
//!   translucent terrain, the fog and the hand.
//! * Anisotropic Filtering lives in the terrain renderer
//!   (`TerrainRenderer::set_anisotropy`, `mc_render::terrain::phase::aniso`).
//!
//! This file only moves settings into them: when the settings generation moves, the AO options
//! and switch and the AF level; every frame, and only while AO is on, whether a world is on screen.
//! With both mods off a frame costs one `u64` compare and one `bool` test here; the AO phase then
//! reports itself disabled (the `PhaseList` never records it, the frame is today's byte for byte)
//! and the terrain draws with its original pipelines.
//!
//! A resource-pack reload rebuilds every phase and the terrain renderer through
//! `Client::install`, which calls [`Client::install_x8`]: it resets the seen generation, so the next
//! frame pushes both mods' settings into the new objects.

use mc_mods::mods::anisotropic_filtering::LEVELS;
use mc_render::post::ao::{AoPhase, AoQuality, AoSettings};
use mc_render::{PhaseList, PhaseOrder};

use super::super::{Client, Shared, Ui};

/// What the client last pushed into the AO phase and the terrain renderer.
#[derive(Default)]
pub(crate) struct X8State {
    /// The settings generation last read. 0 = never (the store's generation starts at 1).
    seen: u64,
    /// Ambient Occlusion is switched on: the per-frame world check is worth doing.
    ao_on: bool,
    /// What the phase was last told about a world being on screen.
    in_world: bool,
    /// The AF level last pushed (0 = off).
    aniso_level: u32,
    /// The AO phase, shared with the `PhaseList`. `None` until `Client::install`.
    phase: Option<Shared<AoPhase>>,
}

/// The AO options as the phase wants them.
fn ao_settings(settings: &mc_mods::ModSettings) -> AoSettings {
    AoSettings {
        quality: AoQuality::from_index(settings.choice("ambient_occlusion", "quality")),
        radius: settings.f32("ambient_occlusion", "radius"),
        strength: settings.f32("ambient_occlusion", "strength") / 100.0,
        half_resolution: settings.bool("ambient_occlusion", "half_resolution"),
    }
}

/// The AF level the settings ask for: 0 when the mod is off.
fn aniso_level(settings: &mc_mods::ModSettings) -> u32 {
    if settings.enabled("anisotropic_filtering") {
        let index = settings.choice("anisotropic_filtering", "level");
        LEVELS[index.min(LEVELS.len() - 1)]
    } else {
        0
    }
}

impl Client {
    /// Register the AO phase. Runs from `Client::install`, so also after a resource-pack reload,
    /// which rebuilds every phase and the terrain renderer; both are synced on the next frame.
    pub(crate) fn install_x8(&mut self, phases: &mut PhaseList) {
        let phase = Shared::new(AoPhase::new());
        phases.insert(PhaseOrder::AMBIENT_OCCLUSION, Box::new(phase.clone()));
        self.x8 = X8State {
            phase: Some(phase),
            ..X8State::default()
        };
    }

    /// Push this frame's settings into the AO phase and the terrain renderer. Called once per
    /// frame from `Client::frame`.
    pub(crate) fn drive_x8(&mut self) {
        let generation = self.mods.settings.generation();
        if self.x8.seen != generation {
            self.x8.seen = generation;
            self.sync_x8_settings();
        }
        if !self.x8.ao_on {
            return;
        }
        let in_world = self.x8_world_on_screen();
        if in_world != self.x8.in_world {
            self.x8.in_world = in_world;
            if let Some(phase) = &self.x8.phase {
                phase.with(|phase| phase.set_in_world(in_world));
            }
        }
    }

    /// Read both mods' settings (only when the generation moved) and push them.
    fn sync_x8_settings(&mut self) {
        let settings = &self.mods.settings;
        let ao_on = settings.enabled("ambient_occlusion");
        let ao = ao_settings(settings);
        let level = aniso_level(settings);
        self.x8.ao_on = ao_on;
        if let Some(phase) = &self.x8.phase {
            phase.with(|phase| {
                phase.set_settings(ao);
                phase.set_enabled(ao_on);
            });
        }
        if let Some(render) = &self.render {
            render.terrain.borrow_mut().set_anisotropy(level);
            self.x8.aniso_level = level;
        }
    }

    /// Whether a world is on screen: a session whose player exists and is past the
    /// downloading-terrain screen (`Ui::Loading`). The main menu and loading screens get no AO.
    fn x8_world_on_screen(&self) -> bool {
        !matches!(self.ui, Ui::Loading)
            && self.session.as_ref().is_some_and(|session| {
                session.predictor().is_some() && !session.terrain_download_active()
            })
    }

    /// The AF level last pushed into the terrain renderer (0 = off). For tests.
    #[cfg(test)]
    pub(crate) fn x8_aniso_level(&self) -> u32 {
        self.x8.aniso_level
    }

    /// The AO phase's state, for tests.
    #[cfg(test)]
    pub(crate) fn x8_ao<R>(&self, f: impl FnOnce(&AoPhase) -> R) -> Option<R> {
        self.x8
            .phase
            .as_ref()
            .map(|phase| phase.with(|phase| f(phase)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::net::{ToGame, ToNet};
    use crate::session::{JoinSettings, Session};
    use mc_render::{Gpu, GpuOptions};
    use tokio::sync::mpsc::{Sender, UnboundedReceiver};

    /// A client on the real assets, or `None` (skip) without the vanilla jar. The `TempDir` is
    /// the game dir and must outlive the client.
    fn offline_client() -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("SKIP: no vanilla jar in this checkout");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    /// JoinGame (opens the downloading-terrain screen), then the first S08 (closes it). Keep both
    /// channel ends alive.
    fn join_world(client: &mut Client) -> (Sender<ToGame>, UnboundedReceiver<ToNet>) {
        let (handle, to_game, to_net) = crate::net::channel::NetHandle::new_pair();
        client.session = Some(Session::new(handle, JoinSettings::default()));
        to_game
            .try_send(ToGame::Packet(mc_protocol::play::S2CPlay::JoinGame(
                mc_protocol::play::s2c::JoinGame {
                    entity_id: 27,
                    hardcore_mode: false,
                    game_mode: 0,
                    dimension: 0,
                    difficulty: 2,
                    max_players: 20,
                    level_type: "default".into(),
                    reduced_debug_info: false,
                },
            )))
            .expect("join channel");
        client.poll();
        to_game
            .try_send(ToGame::Packet(
                mc_protocol::play::S2CPlay::PlayerPositionLook(
                    mc_protocol::play::s2c::PlayerPositionLook {
                        x: 8.5,
                        y: 70.0,
                        z: 8.5,
                        yaw: 0.0,
                        pitch: 0.0,
                        flags: 0,
                    },
                ),
            ))
            .expect("position channel");
        client.poll();
        (to_game, to_net)
    }

    fn terrain_level(client: &Client) -> u32 {
        client
            .render
            .as_ref()
            .expect("installed")
            .terrain
            .borrow()
            .anisotropy()
    }

    /// The real driver on a really installed client: both mods off by default (nothing runs,
    /// terrain at level 0), the AO options and switch reach the phase, AO runs only with a world
    /// on screen, the AF level reaches the terrain renderer and follows the choice, and a
    /// resource-pack style reinstall gets both pushed again on the next frame.
    #[test]
    fn a_real_client_feeds_both_mods_and_resyncs_after_a_reinstall() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let Ok(gpu) = Gpu::headless(
            32,
            32,
            GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) else {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            return;
        };
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, 32, 32);
        assert!(phases.names().any(|name| name == AoPhase::NAME));

        client.drive_x8();
        assert_eq!(client.x8_ao(|p| p.is_enabled()), Some(false));
        assert_eq!(client.x8_ao(|p| p.is_active()), Some(false));
        assert_eq!(terrain_level(&client), 0);

        // AO on in the main menu: the settings arrive, nothing runs.
        client.mods.settings.set_enabled("ambient_occlusion", true);
        client
            .mods
            .settings
            .set_f32("ambient_occlusion", "radius", 2.0);
        client.drive_x8();
        assert_eq!(client.x8_ao(|p| p.is_enabled()), Some(true));
        assert_eq!(client.x8_ao(|p| p.settings().radius), Some(2.0));
        assert_eq!(
            client.x8_ao(|p| p.is_active()),
            Some(false),
            "no world, no AO"
        );

        // In a world it runs, and never switches the frame to scene mode.
        let _channels = join_world(&mut client);
        client.drive_x8();
        assert!(client.x8_world_on_screen());
        assert_eq!(client.x8_ao(|p| p.is_active()), Some(true));
        assert!(!phases.scene_mode());

        // AF: on = 16x, the choice picks the level, off = 0.
        client
            .mods
            .settings
            .set_enabled("anisotropic_filtering", true);
        client.drive_x8();
        assert_eq!(terrain_level(&client), 16);
        assert_eq!(client.x8_aniso_level(), 16);
        client
            .mods
            .settings
            .set_choice("anisotropic_filtering", "level", 1);
        client.drive_x8();
        assert_eq!(terrain_level(&client), 4);

        // A reinstall rebuilds the phase and the terrain renderer; the next frame re-pushes both.
        client.reinstall(&gpu, &mut phases, 32, 32);
        assert_eq!(terrain_level(&client), 0, "a fresh renderer starts off");
        client.drive_x8();
        assert_eq!(terrain_level(&client), 4);
        assert_eq!(client.x8_ao(|p| p.settings().radius), Some(2.0));
        assert_eq!(client.x8_ao(|p| p.is_active()), Some(true));

        // Both off: the phase is disabled again and the terrain draws today's pipelines.
        client.mods.settings.set_enabled("ambient_occlusion", false);
        client
            .mods
            .settings
            .set_enabled("anisotropic_filtering", false);
        client.drive_x8();
        assert_eq!(client.x8_ao(|p| p.is_active()), Some(false));
        assert_eq!(client.x8_ao(|p| p.allocated()), Some(false));
        assert_eq!(terrain_level(&client), 0);
    }

    #[test]
    fn the_defaults_map_to_the_phase_defaults() {
        let settings = mc_mods::ModSettings::new();
        assert_eq!(ao_settings(&settings), AoSettings::default());
        assert!(
            !settings.enabled("ambient_occlusion"),
            "AO is off by default"
        );
        assert_eq!(aniso_level(&settings), 0, "AF is off by default");
    }

    #[test]
    fn the_options_reach_the_phase_settings() {
        let mut settings = mc_mods::ModSettings::new();
        settings.set_choice("ambient_occlusion", "quality", 3);
        settings.set_f32("ambient_occlusion", "radius", 2.5);
        settings.set_f32("ambient_occlusion", "strength", 150.0);
        settings.set_bool("ambient_occlusion", "half_resolution", false);
        assert_eq!(
            ao_settings(&settings),
            AoSettings {
                quality: AoQuality::Ultra,
                radius: 2.5,
                strength: 1.5,
                half_resolution: false,
            }
        );
        settings.set_enabled("anisotropic_filtering", true);
        assert_eq!(aniso_level(&settings), 16, "16x by default");
        for (index, level) in LEVELS.iter().enumerate() {
            settings.set_choice("anisotropic_filtering", "level", index);
            assert_eq!(aniso_level(&settings), *level);
        }
    }
}
