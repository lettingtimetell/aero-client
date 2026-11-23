//! The client half of `motion_blur` and `color_grading`.
//!
//! Both mods are one render phase, [`MotionGradePhase`] (`mc_render::post::motion_grade`),
//! registered at `PhaseOrder::POST` straight after vanilla's post chain. It therefore covers the
//! world, the hand, the screen overlays and any F4 shader group, and not the HUD, chat, screens
//! or the mod menu, which is where vanilla runs its own shader group (after `renderWorld`,
//! before `renderGameOverlay`). This file only moves settings into the phase:
//!
//! * when the settings generation moves: the blur strength and the effective CDL grade;
//! * every frame, and only while one of the two would draw something: whether a world is on
//!   screen.
//!
//! With both mods off, or the grade at its identity and the blur off, a frame costs one `u64`
//! compare and one `bool` test here, and the phase reports itself disabled, so `PhaseList`
//! keeps its direct frame path byte for byte and no texture of the phase exists.

use mc_render::post::motion_grade::{CdlParams, MotionGradePhase};
use mc_render::{Gpu, PhaseList, PhaseOrder};

use super::super::{Client, Shared, Ui};

/// What the client last pushed into the phase.
#[derive(Default)]
pub(crate) struct M7State {
    /// The settings generation last read. 0 = never (the store's generation starts at 1).
    seen: u64,
    /// The blur is on or the grade is not the identity: the world check is worth doing.
    active: bool,
    /// What the phase was last told about a world being on screen.
    in_world: bool,
    /// The phase, shared with the `PhaseList`. `None` until `Client::install`.
    phase: Option<Shared<MotionGradePhase>>,
}

impl Client {
    /// Register the motion-blur and colour-grading phase. Runs from `Client::install`, so also
    /// after a resource-pack reload, which rebuilds every phase; the new phase is synced on the
    /// next frame.
    pub(crate) fn install_m7_postfx(&mut self, gpu: &Gpu, phases: &mut PhaseList) {
        let phase = Shared::new(MotionGradePhase::new(gpu));
        // `PhaseList::insert` is stable: this lands after every POST phase the render stack
        // registered (overlays, then vanilla's post chain) and before the UI.
        phases.insert(PhaseOrder::POST, Box::new(phase.clone()));
        self.m7 = M7State {
            phase: Some(phase),
            ..M7State::default()
        };
    }

    /// Push this frame's settings into the phase. Called once per frame from `Client::frame`.
    pub(crate) fn drive_m7_postfx(&mut self) {
        let generation = self.mods.settings.generation();
        if self.m7.seen != generation {
            self.m7.seen = generation;
            self.sync_m7_settings();
        }
        if !self.m7.active {
            return;
        }
        let in_world = self.m7_world_on_screen();
        if in_world != self.m7.in_world {
            self.m7.in_world = in_world;
            if let Some(phase) = &self.m7.phase {
                phase.with(|phase| phase.set_in_world(in_world));
            }
        }
    }

    /// Read both mods' settings (only when the generation moved) and push them.
    fn sync_m7_settings(&mut self) {
        let settings = &self.mods.settings;
        // Only the Frame blend mode is this phase's; the per-pixel modes
        // (Camera, Camera + objects) run in the `post/velocity-blur` (`x9_aa.rs`).
        let frame_blend =
            settings.choice("motion_blur", "mode") == mc_mods::mods::motion_blur::MODE_FRAME_BLEND;
        let blur = if settings.enabled("motion_blur") && frame_blend {
            settings.f32("motion_blur", "strength") / 100.0
        } else {
            0.0
        };
        let grade = if settings.enabled("color_grading") {
            let get = |id: &str| settings.f32("color_grading", id);
            CdlParams::from_controls(
                get("master_slope"),
                get("master_offset"),
                get("master_power"),
                get("saturation"),
                [get("slope_r"), get("slope_g"), get("slope_b")],
                [get("offset_r"), get("offset_g"), get("offset_b")],
                [get("power_r"), get("power_g"), get("power_b")],
            )
        } else {
            CdlParams::IDENTITY
        };
        self.m7.active = blur > 0.0 || !grade.is_identity();
        if let Some(phase) = &self.m7.phase {
            phase.with(|phase| {
                phase.set_blur_strength(blur);
                phase.set_grade(grade);
            });
        }
    }

    /// Whether a world is on screen: a session whose player exists and is past the
    /// downloading-terrain screen (`Ui::Loading`). The main menu, its panorama and every loading
    /// screen are never blurred or graded.
    fn m7_world_on_screen(&self) -> bool {
        !matches!(self.ui, Ui::Loading)
            && self.session.as_ref().is_some_and(|session| {
                session.predictor().is_some() && !session.terrain_download_active()
            })
    }

    // A video export renders at fixed
    // timesteps, so the frame-blend blur weighs its history by the export's frame time instead
    // of the wall clock (`None` puts the wall clock back).
    pub(crate) fn m7_set_fixed_dt(&mut self, dt: Option<f32>) {
        if let Some(phase) = self.m7.phase.as_ref() {
            phase.with(|phase| phase.set_fixed_dt(dt));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::net::{ToGame, ToNet};
    use crate::session::{JoinSettings, Session};
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

    /// Join a world the way a server does it: JoinGame opens the downloading-terrain screen and
    /// the first S08 ([`position_look`]) closes it. Keep both channel ends alive.
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
        (to_game, to_net)
    }

    fn position_look(to_game: &Sender<ToGame>) {
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
    }

    fn phase(client: &Client) -> Shared<MotionGradePhase> {
        client.m7.phase.clone().expect("installed")
    }

    /// The settings reach the phase, the phase runs only with a world on screen, and both mods
    /// off (or at their neutral values) leave `PhaseList` on its direct path.
    #[test]
    fn settings_reach_the_phase_and_it_runs_only_in_a_world() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let gpu = match Gpu::headless(8, 8, mc_render::GpuOptions::default()) {
            Ok(gpu) => gpu,
            Err(mc_render::RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter");
                return;
            }
            Err(error) => panic!("headless context: {error}"),
        };
        let mut phases = PhaseList::new();
        client.install_m7_postfx(&gpu, &mut phases);
        assert!(phases.names().any(|name| name == MotionGradePhase::NAME));

        // Defaults: both mods off. Nothing runs.
        client.drive_m7_postfx();
        assert!(!phases.scene_mode());
        assert_eq!(phase(&client).with(|p| p.blur_strength()), 0.0);

        // On in the main menu: the strength arrives, but no world means no pass.
        client.mods.settings.set_enabled("motion_blur", true);
        client.drive_m7_postfx();
        assert_eq!(phase(&client).with(|p| p.blur_strength()), 0.5);
        assert!(!phases.scene_mode(), "the main menu is never blurred");

        // JoinGame: the downloading-terrain screen is still up.
        let (to_game, _to_net) = join_world(&mut client);
        client.drive_m7_postfx();
        assert!(matches!(client.ui, Ui::Loading));
        assert!(!phases.scene_mode(), "the terrain screen is never blurred");

        // The first S08 closes it: now the phase runs.
        position_look(&to_game);
        client.poll();
        client.drive_m7_postfx();
        assert!(client.m7_world_on_screen());
        assert!(phases.scene_mode());

        // The grade's controls map onto the effective CDL numbers.
        client.mods.settings.set_enabled("motion_blur", false);
        client.mods.settings.set_enabled("color_grading", true);
        client.drive_m7_postfx();
        assert!(
            !phases.scene_mode(),
            "blur off plus a grade at its identity is no pass at all"
        );
        let settings = &mut client.mods.settings;
        settings.set_f32("color_grading", "master_slope", 1.5);
        settings.set_f32("color_grading", "slope_r", 0.5);
        settings.set_f32("color_grading", "master_offset", 0.1);
        settings.set_f32("color_grading", "offset_b", -0.05);
        settings.set_f32("color_grading", "master_power", 2.0);
        settings.set_f32("color_grading", "power_g", 0.75);
        settings.set_f32("color_grading", "saturation", 1.25);
        client.drive_m7_postfx();
        let grade = phase(&client).with(|p| p.grade());
        let close = |a: [f32; 3], b: [f32; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6);
        assert!(close(grade.slope, [0.75, 1.5, 1.5]), "{grade:?}");
        assert!(close(grade.offset, [0.1, 0.1, 0.05]), "{grade:?}");
        assert!(close(grade.power, [2.0, 1.5, 2.0]), "{grade:?}");
        assert!((grade.saturation - 1.25).abs() < 1e-6, "{grade:?}");
        assert!(phases.scene_mode());

        // Off again: the phase is disabled and holds no history texture.
        client.mods.settings.set_enabled("color_grading", false);
        client.drive_m7_postfx();
        assert!(!phases.scene_mode());
        assert!(!phase(&client).with(|p| p.history_allocated()));
        assert!(!phase(&client).with(|p| p.is_active()));
    }
}
