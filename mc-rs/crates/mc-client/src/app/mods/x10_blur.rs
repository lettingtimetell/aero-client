//! The client half of `menu_blur` and
//! `super_secret_settings`.
//!
//! # Menu Blur
//!
//! The render half is [`MenuBlurPhase`] (`mc_render::post::blur`) at `PhaseOrder::MENU_BLUR`,
//! plus [`UiScreenPhase`] at `PhaseOrder::UI_SCREEN`, both registered by [`Client::install_x10`].
//! Every frame `build_ui` asks [`Client::menu_blur_split`], right after the HUD went into the UI
//! draw list, where to split that list: at the HUD's end (blur the HUD too, the default), at 0
//! (blur only the world), or not at all. A split makes the UI phase draw the HUD half, the blur
//! record over everything drawn so far, and the screen phase draw the open screen on top.
//!
//! Blurred: every screen open over a world (pause, options, inventories and containers, signs,
//! books, ...) except chat and the bed chat, the death screen, and the Aero mod menu, whose glass
//! already frosts the game (no double blur). "In inventories" off leaves containers and the
//! creative inventory sharp. The blur fades in over "Fade-in" ms when a blurred screen opens and
//! stays at full strength while you move between blurred screens; it goes at once when the last
//! one closes, so the game is sharp the moment you can act again.
//!
//! Off, or with no blurred screen open, a frame costs one `u64` compare and a `bool` here, no
//! split is set (the UI phase draws its list exactly as before), both phases report themselves
//! disabled, and the blur owns no texture (`tests/x10_menu_blur.rs`, `mc-ui/tests/
//! x10_ui_split.rs`).
//!
//! # Super Secret Settings
//!
//! Vanilla's options button (removed from the options screen) stepped through
//! `EntityRenderer.shaderResourceLocations`. The mod's "Shader" choice picks one directly; this
//! file drives the client's existing vanilla machinery with it (`shader_cycle`,
//! `pending_shader_action`, `drive_shader`), so loading, resource reloads, entity shaders and the
//! post phase work exactly as for the button:
//!
//! * the choice is applied at once, and again whenever something cleared it while a world is up
//!   and the camera is the local player: joining a world, respawning, changing dimension, the
//!   camera coming back from a spectated mob (vanilla's `setRenderViewEntity` drops the group in
//!   all of those);
//! * spectating a creeper, spider or Enderman keeps vanilla's entity shader;
//! * a group that fails to load is not retried every frame, only after the choice changes or the
//!   resources reload;
//! * F4 (vanilla `switchUseShader`) does not hide the mod's shader, so F4 can be bound to a key
//!   action such as Toggle Perspective without flashing the shader (item 7 of this wave);
//! * turning the mod off (or choosing Off) clears the group.
//!
//! Off, a frame costs one `u64` compare and a `bool`.

use std::time::{Duration, Instant};

use mc_render::post::blur::MenuBlurPhase;
use mc_render::post::{SuperSecretShaderAction, SUPER_SECRET_SHADER_GROUPS};
use mc_render::{PhaseList, PhaseOrder};
use mc_ui::widget::{UiPhase, UiScreenPhase};

use super::super::{Client, Shared, Ui};

/// What the client last read from the two mods and pushed into the renderer.
#[derive(Default)]
pub(crate) struct X10State {
    /// The settings generation last read. 0 = never (the store's generation starts at 1).
    seen: u64,
    /// `menu_blur` is on.
    blur_on: bool,
    /// Fade-in length in milliseconds.
    fade_ms: f32,
    /// Split after the HUD (blur it too) rather than before it.
    blur_hud: bool,
    /// Blur behind containers and the creative inventory too.
    inventories: bool,
    /// When the current run of blurred screens began; `None` while none is open.
    active_since: Option<Instant>,
    /// The blur phase, shared with the `PhaseList`. `None` until `Client::install`.
    phase: Option<Shared<MenuBlurPhase>>,
    /// The Super Secret group the mod wants: an index into [`SUPER_SECRET_SHADER_GROUPS`], or
    /// `None` (mod off, or "Off" chosen).
    sss_want: Option<usize>,
    /// A load this file queued has not been seen to land yet.
    sss_queued: bool,
    /// The group whose load failed, not retried until the choice or the resources change.
    sss_failed: Option<usize>,
}

/// The menu blur's fade-in: smoothstep over `fade_ms`, 1 at once for 0.
fn menu_blur_fade(elapsed: Duration, fade_ms: f32) -> f32 {
    if fade_ms <= 0.0 || !fade_ms.is_finite() {
        return 1.0;
    }
    let t = (elapsed.as_secs_f32() * 1000.0 / fade_ms).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The mod's "Shader" choice as an index into [`SUPER_SECRET_SHADER_GROUPS`]: choice 0 is Off.
fn super_secret_group(choice: usize) -> Option<usize> {
    choice
        .checked_sub(1)
        .filter(|&index| index < SUPER_SECRET_SHADER_GROUPS.len())
}

impl Client {
    /// Register the menu blur and the UI's screen half. Runs from `Client::install`, so also
    /// after a resource-pack reload, which rebuilds every phase (`ui` is the new UI phase).
    pub(crate) fn install_x10(&mut self, phases: &mut PhaseList, ui: &Shared<UiPhase>) {
        let phase = Shared::new(MenuBlurPhase::new());
        phases.insert(PhaseOrder::MENU_BLUR, Box::new(phase.clone()));
        phases.insert(PhaseOrder::UI_SCREEN, Box::new(UiScreenPhase(ui.0.clone())));
        self.x10.phase = Some(phase);
        // Push everything into the new phase on the next frame, and give a group that failed
        // under the old resources another chance under the new ones.
        self.x10.seen = 0;
        self.x10.active_since = None;
        self.x10.sss_failed = None;
        self.x10.sss_queued = false;
    }

    /// Read both mods' settings when the generation moved, and act on what changed.
    fn sync_x10(&mut self) {
        let generation = self.mods.settings.generation();
        if self.x10.seen == generation {
            return;
        }
        self.x10.seen = generation;
        let settings = &self.mods.settings;
        let blur_on = settings.enabled("menu_blur");
        let radius = settings.f32("menu_blur", "radius");
        self.x10.fade_ms = settings.f32("menu_blur", "fade_in");
        self.x10.blur_hud = settings.bool("menu_blur", "blur_hud");
        self.x10.inventories = settings.bool("menu_blur", "inventories");
        let want = if settings.enabled("super_secret_settings") {
            super_secret_group(settings.choice("super_secret_settings", "shader"))
        } else {
            None
        };

        self.x10.blur_on = blur_on;
        if let Some(phase) = &self.x10.phase {
            phase.with(|phase| {
                phase.set_radius(radius);
                if !blur_on {
                    // Off: no texture, no pipeline. The next blurred screen rebuilds them.
                    phase.release();
                }
            });
        }
        if !blur_on {
            self.x10.active_since = None;
        }

        if want != self.x10.sss_want {
            let had = self.x10.sss_want.is_some();
            self.x10.sss_want = want;
            self.x10.sss_failed = None;
            self.x10.sss_queued = false;
            if want.is_none() && had {
                self.clear_super_secret_selection();
            }
        }
    }

    // ------------------------------------------------------------------------- menu blur

    /// Menu Blur. Called by `build_ui` right after the HUD went into the frame's UI list
    /// (`hud_end` = the list's vertex count then). Drives the blur phase and returns where the
    /// list splits: the HUD's end, 0 (world only), or `None` (draw it whole, as without the mod).
    pub(crate) fn menu_blur_split(&mut self, hud_end: usize) -> Option<usize> {
        self.sync_x10();
        if !self.x10.blur_on {
            return None;
        }
        // Both halves' phases exist only after `install`; never split a list nobody finishes.
        let blurred = self.x10.phase.is_some() && self.menu_blur_wanted();
        let Some(phase) = self.x10.phase.as_ref().filter(|_| blurred) else {
            if self.x10.active_since.take().is_some() {
                if let Some(phase) = &self.x10.phase {
                    phase.with(|phase| phase.set_active(false));
                }
            }
            return None;
        };
        let now = Instant::now();
        let since = *self.x10.active_since.get_or_insert(now);
        let fade = menu_blur_fade(now.saturating_duration_since(since), self.x10.fade_ms);
        phase.with(|phase| {
            phase.set_active(true);
            phase.set_fade(fade);
        });
        Some(if self.x10.blur_hud { hud_end } else { 0 })
    }

    /// Whether the open screen is one the blur goes behind, with a world on screen behind it.
    fn menu_blur_wanted(&self) -> bool {
        match &self.ui {
            // No screen, chat, the bed chat (both keep the game in view on purpose), the death
            // screen, and the mod menu, whose glass frosts the frame itself.
            Ui::InGame | Ui::Chat(_) | Ui::Sleep(_) | Ui::GameOver(_) | Ui::ModMenu(_) => {
                return false;
            }
            // Screens that never have a world behind them.
            Ui::Loading
            | Ui::MainMenu(_)
            | Ui::Login(_)
            | Ui::Connecting(_)
            | Ui::Disconnected(_) => return false,
            Ui::Container(_) | Ui::Creative(_) if !self.x10.inventories => return false,
            _ => {}
        }
        self.session.as_ref().is_some_and(|session| {
            session.predictor().is_some() && !session.terrain_download_active()
        })
    }

    // ------------------------------------------------------------- Super Secret Settings

    /// Super Secret Settings, once per frame from `Client::frame`, right before `drive_shader`
    /// applies the frame's pending shader action. Re-queues the chosen group whenever something
    /// cleared it. Off: one generation compare and a `bool`.
    pub(crate) fn x10_super_secret_frame(&mut self) {
        self.sync_x10();
        let Some(index) = self.x10.sss_want else {
            return;
        };
        // Vanilla only keeps a Super Secret group while a world is loaded and the camera is the
        // player; spectating a creeper, spider or Enderman shows that mob's own shader instead.
        let local_world = self.session.as_ref().is_some_and(|session| {
            matches!(
                session.render_view_target(),
                crate::session::RenderViewTarget::Local
            )
        });
        if !local_world {
            self.x10.sss_queued = false;
            return;
        }
        // Something else (a camera change, a respawn) queued an action this frame: let it land.
        if self.pending_shader_action.is_some() {
            return;
        }
        if self.shader_cycle.index() != index {
            // The selection was lost: a world unload's `stopUseShader`, or our load failed.
            if self.x10.sss_failed == Some(index) {
                return;
            }
            if self.x10.sss_queued {
                tracing::warn!(
                    shader = SUPER_SECRET_SHADER_GROUPS[index],
                    "Super Secret Settings: the shader failed to load; not retrying"
                );
                self.x10.sss_failed = Some(index);
                self.x10.sss_queued = false;
                return;
            }
            let action = self.shader_cycle.select(Some(index));
            self.queue_super_secret(action);
            return;
        }
        if self.active_shader.is_none() {
            // Selected but not loaded: joining a world, a respawn or the camera's return from a
            // mob dropped the group (`setRenderViewEntity`). `loadShader` re-arms `useShader`.
            self.shader_cycle.set_use_shader(true);
            self.queue_super_secret(SuperSecretShaderAction::Load(
                SUPER_SECRET_SHADER_GROUPS[index],
            ));
            return;
        }
        self.x10.sss_queued = false;
        if !self.shader_visible {
            // F4 flipped `useShader`: the mod's shader stays on, so F4 is free for a binding.
            self.shader_visible = true;
            self.shader_cycle.set_use_shader(true);
        }
    }

    /// Queue a load the way the options button did (`Action::PressSuperSecretSettings`).
    fn queue_super_secret(&mut self, action: SuperSecretShaderAction) {
        self.shader_visible = matches!(action, SuperSecretShaderAction::Load(_));
        // `loadShader` deletes the previous group before building the next one.
        self.active_shader = None;
        self.pending_shader_action = Some(action);
        self.x10.sss_queued = true;
    }

    /// The mod was turned off (or "Off" chosen): drop the selected group, as vanilla's
    /// `stopUseShader` does. A mob's shader shown while spectating is left alone.
    fn clear_super_secret_selection(&mut self) {
        if self.shader_cycle.selected().is_none() {
            return;
        }
        self.shader_cycle.stop();
        let spectating = self.session.as_ref().is_some_and(|session| {
            !matches!(
                session.render_view_target(),
                crate::session::RenderViewTarget::Local
            )
        });
        if !spectating {
            self.shader_visible = false;
            self.active_shader = None;
            self.pending_shader_action = Some(SuperSecretShaderAction::Clear);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::net::{ToGame, ToNet};
    use crate::session::{JoinSettings, Session};
    use mc_render::Gpu;
    use tokio::sync::mpsc::{Sender, UnboundedReceiver};

    #[test]
    fn the_fade_is_a_smoothstep_and_zero_means_instant() {
        assert_eq!(menu_blur_fade(Duration::ZERO, 0.0), 1.0);
        assert_eq!(menu_blur_fade(Duration::ZERO, 150.0), 0.0);
        assert!((menu_blur_fade(Duration::from_millis(75), 150.0) - 0.5).abs() < 1e-6);
        assert_eq!(menu_blur_fade(Duration::from_millis(150), 150.0), 1.0);
        assert_eq!(menu_blur_fade(Duration::from_secs(9), 150.0), 1.0);
        assert_eq!(menu_blur_fade(Duration::from_millis(10), f32::NAN), 1.0);
    }

    #[test]
    fn the_shader_choices_are_off_then_vanillas_groups_in_order() {
        use mc_mods::mods::super_secret_settings::{
            ASCII_CHOICE, FIRST_AERO_CHOICE, SHADER_CHOICES,
        };
        // Aero's own effects follow vanilla's groups and map to
        // no vanilla group, so choosing one clears a loaded group as "Off" does.
        assert_eq!(FIRST_AERO_CHOICE, SUPER_SECRET_SHADER_GROUPS.len() + 1);
        assert!(SHADER_CHOICES.len() > FIRST_AERO_CHOICE);
        assert_eq!(super_secret_group(ASCII_CHOICE), None);
        assert_eq!(SHADER_CHOICES[0], "Off");
        assert_eq!(super_secret_group(0), None);
        assert_eq!(super_secret_group(99), None);
        for (index, stem) in SUPER_SECRET_SHADER_GROUPS.iter().enumerate() {
            assert_eq!(super_secret_group(index + 1), Some(index));
            // Each label is its group's file stem, made readable.
            let label: String = SHADER_CHOICES[index + 1]
                .to_ascii_lowercase()
                .replace(' ', "_")
                .replace("blobs_2", "blobs2");
            assert_eq!(label, *stem, "choice {} names {stem}", index + 1);
        }
    }

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

    /// JoinGame (the terrain screen opens), then the first S08 (it closes). Keep both ends.
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

    fn headless() -> Option<Gpu> {
        match Gpu::headless(8, 8, mc_render::GpuOptions::default()) {
            Ok(gpu) => Some(gpu),
            Err(mc_render::RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter");
                None
            }
            Err(error) => panic!("headless context: {error}"),
        }
    }

    fn blur(client: &Client) -> Shared<MenuBlurPhase> {
        client.x10.phase.clone().expect("installed")
    }

    /// The split follows the open screen and the settings; off, or with no blurred screen,
    /// nothing is split, nothing opts into the scene and the blur holds no GPU resource.
    #[test]
    fn the_blur_runs_behind_menus_over_a_world_and_nowhere_else() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        let Some(gpu) = headless() else {
            return;
        };
        let mut phases = PhaseList::new();
        let ui = Shared::new(UiPhase::new(&gpu, 8, 8, 1, false));
        client.install_x10(&mut phases, &ui);
        let names: Vec<_> = phases.names().collect();
        assert_eq!(names, [MenuBlurPhase::NAME, UiScreenPhase::NAME]);

        // Defaults: the mod is off. Even a pause screen over a world is not blurred.
        let (_to_game, _to_net) = join_world(&mut client);
        assert!(matches!(client.ui, Ui::InGame));
        client.ui = Ui::Pause(Box::default());
        assert_eq!(client.menu_blur_split(40), None);
        assert!(!phases.scene_mode());

        client.mods.settings.set_enabled("menu_blur", true);
        // No screen: nothing.
        client.ui = Ui::InGame;
        assert_eq!(client.menu_blur_split(40), None);
        assert!(!phases.scene_mode());
        // Pause: split after the HUD, the blur opts in and fades in from 0.
        client.ui = Ui::Pause(Box::default());
        assert_eq!(client.menu_blur_split(40), Some(40));
        assert!(phases.scene_mode());
        assert!(blur(&client).with(|b| b.is_active() && b.fade() < 0.5));
        // "Blur HUD" off: split before the HUD, world only.
        client
            .mods
            .settings
            .set_bool("menu_blur", "blur_hud", false);
        assert_eq!(client.menu_blur_split(40), Some(0));
        client.mods.settings.set_bool("menu_blur", "blur_hud", true);
        // Fade-in 0: full strength at once.
        client.mods.settings.set_f32("menu_blur", "fade_in", 0.0);
        assert_eq!(client.menu_blur_split(40), Some(40));
        assert_eq!(blur(&client).with(|b| b.fade()), 1.0);
        // The radius reaches the phase.
        client.mods.settings.set_f32("menu_blur", "radius", 30.0);
        client.menu_blur_split(40);
        assert_eq!(blur(&client).with(|b| b.radius()), 30.0);

        // Chat is never blurred (nor the death screen).
        let chat = mc_ui::screens::chat::ChatScreen::new("", 0, &client.font);
        client.ui = Ui::Chat(Box::new(chat));
        assert_eq!(client.menu_blur_split(40), None);
        assert!(!phases.scene_mode());
        assert!(!blur(&client).with(|b| b.targets_allocated()));
        // Nor the mod menu: its glass frosts the frame itself, and must not refract a blur.
        let menu = mc_ui::screens::modmenu::ModMenuScreen::new(
            client.mods.settings.catalog(),
            Instant::now(),
        );
        client.ui = Ui::ModMenu(Box::new(menu));
        assert_eq!(client.menu_blur_split(40), None);
        assert!(!phases.scene_mode());

        // Options over the world: blurred.
        client.ui = Ui::Options(Box::new(mc_ui::screens::options::OptionsScreen::new(true)));
        assert_eq!(client.menu_blur_split(40), Some(40));

        // "In inventories" off leaves the creative inventory sharp.
        client
            .mods
            .settings
            .set_bool("menu_blur", "inventories", false);
        client.ui = Ui::Creative(Box::new(mc_ui::screens::container::CreativeScreen::new()));
        assert_eq!(client.menu_blur_split(40), None);
        client
            .mods
            .settings
            .set_bool("menu_blur", "inventories", true);
        assert_eq!(client.menu_blur_split(40), Some(40));

        // Off again: nothing is split and the phase drops its pipelines too.
        client.mods.settings.set_enabled("menu_blur", false);
        assert_eq!(client.menu_blur_split(40), None);
        assert!(!phases.scene_mode());
        assert!(!blur(&client).with(|b| b.pipelines_built() || b.targets_allocated()));
    }

    #[test]
    fn without_a_world_or_before_install_nothing_is_split() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        client.mods.settings.set_enabled("menu_blur", true);
        client.ui = Ui::Pause(Box::default());
        // Not installed: no screen phase would draw the second half.
        assert_eq!(client.menu_blur_split(40), None);
        let Some(gpu) = headless() else {
            return;
        };
        let mut phases = PhaseList::new();
        let ui = Shared::new(UiPhase::new(&gpu, 8, 8, 1, false));
        client.install_x10(&mut phases, &ui);
        // Installed, but no world behind the options (the main menu's).
        client.ui = Ui::Options(Box::new(mc_ui::screens::options::OptionsScreen::new(false)));
        assert_eq!(client.menu_blur_split(40), None);
        assert!(!phases.scene_mode());
    }

    /// The mod applies its choice on join, re-applies it after a clear, never retries a broken
    /// group every frame, keeps it through F4, and clears it when turned off.
    #[test]
    fn super_secret_settings_applies_and_reapplies_the_chosen_group() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        use mc_mods::mods::super_secret_settings::SHADER_CHOICES;
        let creeper = SHADER_CHOICES.iter().position(|c| *c == "Creeper").unwrap();
        client
            .mods
            .settings
            .set_enabled("super_secret_settings", true);
        client
            .mods
            .settings
            .set_choice("super_secret_settings", "shader", creeper);
        // No world: nothing is queued.
        client.x10_super_secret_frame();
        assert_eq!(client.pending_shader_action, None);

        let (_to_game, _to_net) = join_world(&mut client);
        // The join queued vanilla's Clear; `drive_shader` applies it before our hook runs.
        client.pending_shader_action = None;
        client.x10_super_secret_frame();
        assert_eq!(
            client.pending_shader_action,
            Some(SuperSecretShaderAction::Load("creeper"))
        );
        assert_eq!(client.shader_cycle.selected(), Some("creeper"));
        assert!(client.shader_visible);

        // `drive_shader` loaded it.
        client.pending_shader_action = None;
        client.active_shader = Some("minecraft:shaders/post/creeper.json".to_owned());
        client.x10_super_secret_frame();
        assert_eq!(
            client.pending_shader_action, None,
            "loaded: nothing more to do"
        );

        // F4 flips useShader: the mod keeps its shader on.
        client.shader_visible = client.shader_cycle.switch_use_shader();
        assert!(!client.shader_visible);
        client.x10_super_secret_frame();
        assert!(client.shader_visible && client.shader_cycle.use_shader());

        // A respawn drops the group (`setRenderViewEntity`): it comes back.
        client.active_shader = None;
        client.shader_visible = false;
        client.x10_super_secret_frame();
        assert_eq!(
            client.pending_shader_action,
            Some(SuperSecretShaderAction::Load("creeper"))
        );

        // The load fails (`loadShader`'s error path): reported once, never retried per frame.
        client.pending_shader_action = None;
        client.shader_cycle.load_failed();
        client.shader_visible = false;
        client.x10_super_secret_frame();
        assert_eq!(client.pending_shader_action, None);
        for _ in 0..3 {
            client.x10_super_secret_frame();
            assert_eq!(client.pending_shader_action, None, "no retry loop");
        }
        // A different choice is tried.
        client
            .mods
            .settings
            .set_choice("super_secret_settings", "shader", creeper + 1);
        client.x10_super_secret_frame();
        assert_eq!(
            client.pending_shader_action,
            Some(SuperSecretShaderAction::Load("spider"))
        );
        client.pending_shader_action = None;
        client.active_shader = Some("minecraft:shaders/post/spider.json".to_owned());

        // Off: the group is cleared.
        client
            .mods
            .settings
            .set_enabled("super_secret_settings", false);
        client.x10_super_secret_frame();
        assert_eq!(
            client.pending_shader_action,
            Some(SuperSecretShaderAction::Clear)
        );
        assert_eq!(client.shader_cycle.selected(), None);
        assert!(!client.shader_visible);
        // And stays off.
        client.pending_shader_action = None;
        client.active_shader = None;
        client.x10_super_secret_frame();
        assert_eq!(client.pending_shader_action, None);
    }

    /// Spectating a creeper shows vanilla's creeper shader, not the mod's choice; back in the
    /// player's own view the choice returns.
    #[test]
    fn a_spectated_mobs_shader_wins_and_the_choice_returns_afterwards() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        use mc_mods::mods::super_secret_settings::SHADER_CHOICES;
        let notch = SHADER_CHOICES.iter().position(|c| *c == "Notch").unwrap();
        client
            .mods
            .settings
            .set_enabled("super_secret_settings", true);
        client
            .mods
            .settings
            .set_choice("super_secret_settings", "shader", notch);
        let (to_game, _to_net) = join_world(&mut client);
        client.pending_shader_action = None;
        client.x10_super_secret_frame();
        assert_eq!(
            client.pending_shader_action,
            Some(SuperSecretShaderAction::Load("notch"))
        );
        client.pending_shader_action = None;
        client.active_shader = Some("minecraft:shaders/post/notch.json".to_owned());

        // A creeper (type 50) appears in a loaded column and the server puts the camera on it
        // (S43). An entity outside loaded chunks is not resident, and S43 ignores it.
        for packet in [
            mc_protocol::play::S2CPlay::ChunkData(mc_protocol::chunk::ChunkData {
                chunk_x: 0,
                chunk_z: 0,
                ground_up_continuous: true,
                primary_bit_mask: 0xFFFF,
                data: vec![0; mc_world::expected_chunk_data_len(0xFFFF, true, true)],
            }),
            mc_protocol::play::S2CPlay::SpawnMob(mc_protocol::play::s2c::SpawnMob {
                entity_id: 42,
                entity_type: 50,
                x: 0,
                y: 2048,
                z: 0,
                yaw: mc_protocol::Angle(0),
                pitch: mc_protocol::Angle(0),
                head_pitch: mc_protocol::Angle(0),
                velocity_x: 0,
                velocity_y: 0,
                velocity_z: 0,
                metadata: Vec::new(),
            }),
            mc_protocol::play::S2CPlay::Camera(mc_protocol::play::s2c::Camera { entity_id: 42 }),
        ] {
            to_game.try_send(ToGame::Packet(packet)).expect("channel");
        }
        client.poll();
        assert_eq!(
            client.pending_shader_action,
            Some(SuperSecretShaderAction::Load("creeper")),
            "vanilla's loadEntityShader"
        );
        client.x10_super_secret_frame();
        assert_eq!(
            client.pending_shader_action,
            Some(SuperSecretShaderAction::Load("creeper")),
            "the mod leaves the mob's shader alone"
        );
        client.pending_shader_action = None;
        client.active_shader = Some("minecraft:shaders/post/creeper.json".to_owned());
        client.x10_super_secret_frame();
        assert_eq!(client.pending_shader_action, None);

        // Back to the player: vanilla clears the creeper shader, the mod's choice returns.
        to_game
            .try_send(ToGame::Packet(mc_protocol::play::S2CPlay::Camera(
                mc_protocol::play::s2c::Camera { entity_id: 27 },
            )))
            .expect("channel");
        client.poll();
        assert_eq!(
            client.pending_shader_action,
            Some(SuperSecretShaderAction::Clear)
        );
        client.pending_shader_action = None;
        client.active_shader = None;
        client.x10_super_secret_frame();
        assert_eq!(
            client.pending_shader_action,
            Some(SuperSecretShaderAction::Load("notch"))
        );
    }
}
