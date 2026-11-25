//! The version the client plays: 1.8.9 (the default) or 1.7.10.
//!
//! The player picks it with the title screen's version slider
//! (`mc_ui::screens::aero_title::version_slider`) or the "Game version" row in the Aero Title
//! Screen section of the mod menu, and it is saved with the mod settings
//! ([`ModSettings::game_version`]). `--gameVersion` plays another one for a single run without
//! saving it. Picking reloads nothing: the title screen's version line follows at once, an open
//! server list is pinged again for the new version, and the next connection and replay
//! recording use it.
//!
//! A connection reads the version once, when it opens (`Client::open_connection`), and keeps it:
//! everything inside a session asks [`Session::version`], never the setting, and the client's
//! 1.7.10 hooks ask [`Client::plays_1_7_10`].
//!
//! # Hooks
//! * `driver.rs`: `open_connection` (the connection's version), `open_multiplayer_screen` and
//!   `drive_pings` (the server list's), every title screen the client opens ([`Client::main_menu`],
//!   [`with_version_line`]), `join_settings` (1.7.10's client settings), `debug_info` and
//!   `describe` (F3 and crash reports), `frame` ([`Client::game_version_menu_frame`]).
//! * `aero_title.rs`: the slider's clicks ([`Client::version_slider_mouse`]) and its selection.
//! * `mod.rs`: the menu row ([`Client::fill_game_version_row`]).

use mc_mods::{ModEvent, ModSettings};
use mc_render::settings::GameSettings;
use mc_ui::screens::menu::MainMenuScreen;
use mc_version::GameVersion;

use super::super::{Client, Ui};
use crate::app::config::ClientConfig;
use crate::session::Session;

/// The mod whose menu section holds the Game version row: the title screen's own.
pub(crate) const MENU_MOD_ID: &str = "aero_title";
/// The Game version row's key in that section.
const MENU_ROW: u64 = 1;

/// The version the next connection plays: `--gameVersion` for this run, else the saved pick.
pub(crate) fn selected(config: &ClientConfig, settings: &ModSettings) -> GameVersion {
    config
        .game_version
        .unwrap_or_else(|| settings.game_version())
}

/// `ui`, with a title screen's version line naming the picked version (for the title screen
/// built before the mod settings are loaded).
pub(crate) fn with_version_line(mut ui: Ui, config: &ClientConfig, settings: &ModSettings) -> Ui {
    if let Ui::MainMenu(menu) = &mut ui {
        set_version_line(menu, selected(config, settings));
    }
    ui
}

/// Vanilla's version line for `version`: "Minecraft 1.8.9" (java18 `GuiMainMenu.drawScreen`) or
/// "Minecraft 1.7.10" (java17's).
fn set_version_line(menu: &mut MainMenuScreen, version: GameVersion) {
    let line = version.title_line();
    if menu.version_line != line {
        menu.version_line.clear();
        menu.version_line.push_str(line);
    }
}

/// The difficulty a 1.7.10 connection's client settings carry: the `options.txt` value, kept in
/// `0..=3` as `options.txt` writes it. java17 sends `GameSettings.difficulty` from
/// `handleJoinGame` without overwriting it from the server first (java18 does overwrite it).
pub(crate) fn settings_difficulty(settings: &GameSettings) -> u8 {
    settings.difficulty.rem_euclid(4) as u8
}

/// Whether a 1.7.10 connection's client settings ask to show capes: the cape skin-part toggle,
/// on unless turned off (java17's `showCape` option defaults to on too).
pub(crate) fn settings_show_cape(settings: &GameSettings) -> bool {
    settings.model_parts.get("cape").copied().unwrap_or(true)
}

/// The version after `version` in [`GameVersion::ALL`], or before it when not `forward`,
/// wrapping around.
fn cycle(version: GameVersion, forward: bool) -> GameVersion {
    let all = GameVersion::ALL;
    let index = all.iter().position(|v| *v == version).unwrap_or(0);
    let next = if forward {
        index + 1
    } else {
        index + all.len() - 1
    };
    all[next % all.len()]
}

impl Client {
    /// The version the next connection plays ([`selected`]).
    pub(crate) fn selected_game_version(&self) -> GameVersion {
        selected(&self.config, &self.mods.settings)
    }

    /// The version being played: the session's, or with no session the one picked for the next.
    pub(crate) fn current_game_version(&self) -> GameVersion {
        self.session
            .as_ref()
            .map_or_else(|| self.selected_game_version(), Session::version)
    }

    /// Whether the session plays 1.7.10 ([`Session::version`], fixed for the connection);
    /// `false` with no session. The one way the client's 1.7.10 hooks ask: an `Option` test and
    /// an enum compare.
    #[must_use]
    pub(crate) fn plays_1_7_10(&self) -> bool {
        self.session
            .as_ref()
            .is_some_and(|session| session.version().is_1_7_10())
    }

    /// The player picked `version` (the title screen's slider or the mod menu's row): it is
    /// saved for the next connection and ends a `--gameVersion` override, and the title
    /// screen's version line and an open server list follow at once. Nothing reloads, and a
    /// live session keeps the version it opened with. Silent: the slider plays its own button
    /// sound, the mod menu none.
    pub(crate) fn select_game_version(&mut self, version: GameVersion) {
        let shown = self.selected_game_version();
        self.config.game_version = None;
        self.mods.settings.set_game_version(version);
        self.refresh_version_lines();
        if version != shown {
            self.restamp_server_list(version);
        }
    }

    /// An open server list, up or waiting behind the mod menu, shown to `version` as opening it
    /// shows it (`open_multiplayer_screen`): the pings in flight are dropped under a fresh
    /// generation, so no answer to the other version's ping lands, and every row forgets what
    /// that version's ping learned and is pinged again with `version`'s protocol when it is next
    /// drawn.
    fn restamp_server_list(&mut self, version: GameVersion) {
        let behind_menu = matches!(self.mods.menu_return.as_deref(), Some(Ui::Multiplayer(_)));
        if !behind_menu && !matches!(self.ui, Ui::Multiplayer(_)) {
            return;
        }
        self.cancel_server_pings();
        let generation = self.allocate_server_ping_generation();
        let screen = if let Ui::Multiplayer(screen) = &mut self.ui {
            screen
        } else if let Some(Ui::Multiplayer(screen)) = self.mods.menu_return.as_deref_mut() {
            screen
        } else {
            return;
        };
        // Behind the menu the list is out of `cancel_server_pings`' reach: its requests go here.
        screen.cancel_ping_requests();
        screen.set_ping_generation(generation);
        screen.set_client_version(version);
    }

    /// A new title screen showing `splash`, its version line naming the picked version.
    pub(crate) fn main_menu(&self, splash: String) -> Box<MainMenuScreen> {
        let mut menu = MainMenuScreen::new(splash);
        set_version_line(&mut menu, self.selected_game_version());
        Box::new(menu)
    }

    /// The live title screen's version line, and the one waiting behind the mod menu.
    fn refresh_version_lines(&mut self) {
        let version = self.selected_game_version();
        if let Ui::MainMenu(menu) = &mut self.ui {
            set_version_line(menu, version);
        }
        if let Some(Ui::MainMenu(menu)) = self.mods.menu_return.as_deref_mut() {
            set_version_line(menu, version);
        }
    }

    /// The version slider's share of a mouse edge on the Aero title screen (from
    /// `aero_title_mouse`, after the update card and the account corner, before the buttons):
    /// `Some(true)` when the slider took it. A pick is applied at once, with vanilla's button
    /// sound, as a title-screen button's click.
    pub(crate) fn version_slider_mouse(&mut self, button: i32, pressed: bool) -> Option<bool> {
        let button = u8::try_from(button).ok()?;
        let (screen, (x, y)) = self.aero.live_screen()?;
        let slider = &mut screen.version_slider;
        if pressed {
            return slider.mouse_pressed(x, y, button).then_some(true);
        }
        if !slider.has_press() {
            return None;
        }
        if let Some(version) = slider.mouse_released(x, y, button) {
            self.select_game_version(version);
            self.play_button_press();
        }
        Some(true)
    }

    /// The window's focus changed (from `on_focus_changed`). Losing it drops a press held on the
    /// version slider: its release goes to another window, so the press would otherwise drag the
    /// knob after the pointer and pick on some later, unrelated release.
    pub(crate) fn version_slider_focus_changed(&mut self, focused: bool) {
        if focused {
            return;
        }
        if let Some((screen, _)) = self.aero.live_screen() {
            screen.version_slider.cancel_press();
        }
    }

    /// The Game version row in the Aero Title Screen section of the mod menu: the version the
    /// next connection plays, a choice to click through. While a session is open it is greyed
    /// and fixed: the session keeps the version it opened with.
    pub(crate) fn fill_game_version_row(&mut self, index: usize) {
        let version = self.selected_game_version();
        let in_session = self.session.is_some();
        let Some(rows) = self.mods.rows.get_mut(index) else {
            return;
        };
        rows.begin();
        let row = rows.push(MENU_ROW);
        row.label.push_str(if in_session {
            "Game version (pick on the title screen)"
        } else {
            "Game version"
        });
        row.badge.push_str(version.name());
        row.cycle = !in_session;
        row.muted = in_session;
    }

    /// This frame's clicks on the Game version row: each picks the next version (the one
    /// before on a right click). Nothing while a session is open. A length test while no menu
    /// event is queued.
    pub(crate) fn game_version_menu_frame(&mut self) {
        if self.mods.events.is_empty() || self.session.is_some() {
            return;
        }
        let mut picked = None;
        for event in &self.mods.events {
            if let ModEvent::RowCycle {
                mod_id: MENU_MOD_ID,
                row: MENU_ROW,
                forward,
            } = *event
            {
                let current = picked.unwrap_or_else(|| self.selected_game_version());
                picked = Some(cycle(current, forward));
            }
        }
        if let Some(version) = picked {
            self.select_game_version(version);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_command_line_override_wins_for_the_run_and_never_reaches_the_settings() {
        let mut settings = ModSettings::new();
        let mut config = ClientConfig::default();
        assert_eq!(
            selected(&config, &settings),
            GameVersion::V1_8_9,
            "the default"
        );
        settings.set_game_version(GameVersion::V1_7_10);
        assert_eq!(
            selected(&config, &settings),
            GameVersion::V1_7_10,
            "the saved pick"
        );
        config.game_version = Some(GameVersion::V1_8_9);
        assert_eq!(
            selected(&config, &settings),
            GameVersion::V1_8_9,
            "this run's"
        );
        assert_eq!(settings.game_version(), GameVersion::V1_7_10, "not saved");
    }

    #[test]
    fn the_title_screen_line_names_the_picked_version() {
        let mut settings = ModSettings::new();
        let config = ClientConfig::default();
        let title = || Ui::MainMenu(Box::new(MainMenuScreen::new("splash")));
        let line = |ui: &Ui| match ui {
            Ui::MainMenu(menu) => menu.version_line.clone(),
            _ => panic!("the title screen"),
        };
        let ui = with_version_line(title(), &config, &settings);
        assert_eq!(line(&ui), "Minecraft 1.8.9", "1.8.9's line is vanilla's");
        settings.set_game_version(GameVersion::V1_7_10);
        let ui = with_version_line(title(), &config, &settings);
        assert_eq!(line(&ui), "Minecraft 1.7.10");
        // Any other screen is left alone.
        assert!(matches!(
            with_version_line(Ui::InGame, &config, &settings),
            Ui::InGame
        ));
    }

    #[test]
    fn the_menu_row_cycles_through_every_version_both_ways() {
        for version in GameVersion::ALL {
            assert_ne!(cycle(version, true), version);
            assert_eq!(cycle(cycle(version, true), false), version);
        }
        assert_eq!(
            cycle(GameVersion::V1_8_9, true),
            GameVersion::V1_7_10,
            "wraps"
        );
    }

    #[test]
    fn a_1_7_10_connection_sends_the_clients_difficulty_and_cape_choice() {
        let mut settings = GameSettings::default();
        assert_eq!(
            settings_difficulty(&settings),
            2,
            "normal, vanilla's default"
        );
        assert!(settings_show_cape(&settings), "capes on, vanilla's default");
        settings.difficulty = 0;
        settings.model_parts.insert("cape".to_owned(), false);
        assert_eq!(settings_difficulty(&settings), 0);
        assert!(!settings_show_cape(&settings));
        settings.difficulty = 7;
        assert_eq!(
            settings_difficulty(&settings),
            3,
            "as options.txt writes it"
        );
    }
}

/// The version through the real client: `Client::new` on the real assets, the Aero title screen
/// on a headless GPU, the real input path, a connection to a listener on this machine. Needs the
/// vanilla jar (skips without it; set `MC_JARS_DIR`) and, for the title screen, a GPU adapter.
#[cfg(test)]
mod client_tests {
    use std::io::Read as _;
    use std::time::Duration;

    use mc_mods::ModEvent;
    use mc_render::{Gpu, GpuOptions, InputState, PhaseList, ShellApp as _};
    use mc_ui::screens::aero_title::AeroButton;
    use mc_version::GameVersion;
    use winit::event::MouseButton;

    use super::super::super::{Client, Ui};
    use super::MENU_MOD_ID;
    use crate::app::config::{ClientConfig, LoginMode};

    const SIZE: (u32, u32) = (854, 480);

    fn offline_client(game_version: Option<GameVersion>) -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            game_version,
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout (set MC_JARS_DIR)");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    fn gpu() -> Option<Gpu> {
        gpu_sized(SIZE)
    }

    fn gpu_sized((width, height): (u32, u32)) -> Option<Gpu> {
        let options = GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        };
        match Gpu::headless(width, height, options) {
            Ok(gpu) => Some(gpu),
            Err(mc_render::RenderError::NoAdapter) => {
                eprintln!("skip: no wgpu adapter available on this machine");
                None
            }
            Err(error) => panic!("headless GPU: {error}"),
        }
    }

    /// The Aero title screen up and settled (its pop-in over). Keep the phases while it is used.
    fn title(client: &mut Client, gpu: &Gpu) -> PhaseList {
        let mut phases = PhaseList::new();
        client.install(gpu, &mut phases, SIZE.0, SIZE.1);
        client.build_ui(gpu, 0.0);
        client
            .aero
            .set_live_clocks(Duration::from_secs(5), Duration::from_secs(1));
        client.build_ui(gpu, 0.0);
        phases
    }

    /// Where `version` sits on the live title screen's slider.
    fn slot_centre(client: &mut Client, version: GameVersion) -> (f32, f32) {
        let (screen, _) = client.aero.live_screen().expect("the title screen is up");
        screen.version_slider.version_rect(version).center()
    }

    /// A left click at `(x, y)` (physical pixels), pressed and released in two input passes.
    fn click(client: &mut Client, at: (f32, f32)) {
        click_sized(client, at, SIZE);
    }

    fn click_sized(client: &mut Client, (x, y): (f32, f32), (width, height): (u32, u32)) {
        let mut input = InputState::new();
        input.on_cursor_moved(f64::from(x), f64::from(y));
        input.on_mouse_button(MouseButton::Left, true);
        client.handle_screen_input(&mut input, width, height);
        input.on_mouse_button(MouseButton::Left, false);
        client.handle_screen_input(&mut input, width, height);
    }

    fn version_line(client: &Client) -> &str {
        match &client.ui {
            Ui::MainMenu(menu) => &menu.version_line,
            _ => panic!("the title screen"),
        }
    }

    #[test]
    fn a_click_on_the_slider_picks_saves_and_shows_the_version_reloading_nothing() {
        let Some(gpu) = gpu() else { return };
        let Some((mut client, _dir)) = offline_client(None) else {
            return;
        };
        let _phases = title(&mut client, &gpu);
        assert_eq!(version_line(&client), "Minecraft 1.8.9");
        let opened = client.aero.live_screen().expect("up").0.opened_at();
        let builds = client.aero_title_scene_builds();

        // The selected version: nothing happens, silently.
        let generation = client.mods.settings.generation();
        let newer = slot_centre(&mut client, GameVersion::V1_8_9);
        click(&mut client, newer);
        assert!(client.audio.last_play().is_none(), "no pick, no sound");
        assert_eq!(
            client.mods.settings.generation(),
            generation,
            "nothing to save"
        );

        let older = slot_centre(&mut client, GameVersion::V1_7_10);
        click(&mut client, older);
        assert_eq!(client.mods.settings.game_version(), GameVersion::V1_7_10);
        assert_ne!(client.mods.settings.generation(), generation, "to be saved");
        assert_eq!(version_line(&client), "Minecraft 1.7.10", "java17's line");
        let sound = client.audio.last_play().expect("vanilla's button sound");
        assert_eq!(sound.request.event.path, crate::app::audio::BUTTON_PRESS);
        // Nothing reloaded or restarted: the same title screen, the same scene.
        client.build_ui(&gpu, 0.0);
        assert!(matches!(client.ui, Ui::MainMenu(_)));
        let (screen, _) = client.aero.live_screen().expect("still up");
        assert_eq!(screen.opened_at(), opened, "no pop-in again");
        assert_eq!(screen.version_slider.selected(), GameVersion::V1_7_10);
        assert_eq!(client.aero_title_scene_builds(), builds);
    }

    /// A press held on the slider when the window loses focus never gets its release (it goes to
    /// another window). It must not drag the knob afterwards, nor pick on the next click.
    #[test]
    fn a_slider_press_cut_off_by_a_focus_loss_picks_nothing_later() {
        let Some(gpu) = gpu() else { return };
        let Some((mut client, _dir)) = offline_client(None) else {
            return;
        };
        let _phases = title(&mut client, &gpu);
        let knob = slot_centre(&mut client, GameVersion::V1_8_9);
        let mut input = InputState::new();
        input.on_cursor_moved(f64::from(knob.0), f64::from(knob.1));
        input.on_mouse_button(MouseButton::Left, true);
        client.handle_screen_input(&mut input, SIZE.0, SIZE.1);
        let (screen, _) = client.aero.live_screen().expect("up");
        assert!(screen.version_slider.has_press(), "pressed on the knob");

        // Focus goes with the button down: the shell drops the held button, no release comes.
        input.on_focus(false);
        client.on_focus_changed(false);
        let (screen, _) = client.aero.live_screen().expect("up");
        assert!(!screen.version_slider.has_press(), "dropped with the focus");
        input.on_focus(true);
        client.on_focus_changed(true);

        // Back, a click at Multiplayer's left end, far left of the old press, opens Multiplayer.
        let (screen, _) = client.aero.live_screen().expect("up");
        let multiplayer = screen.button_rect(AeroButton::Multiplayer);
        click(&mut client, (multiplayer.x + 8.0, multiplayer.center().1));
        assert!(
            matches!(client.ui, Ui::Multiplayer(_)),
            "Multiplayer opened"
        );
        assert_eq!(client.mods.settings.game_version(), GameVersion::V1_8_9);
    }

    #[test]
    fn the_command_line_version_shows_until_the_player_picks_one() {
        let Some(gpu) = gpu() else { return };
        let Some((mut client, _dir)) = offline_client(Some(GameVersion::V1_7_10)) else {
            return;
        };
        assert_eq!(version_line(&client), "Minecraft 1.7.10");
        let _phases = title(&mut client, &gpu);
        let (screen, _) = client.aero.live_screen().expect("up");
        assert_eq!(screen.version_slider.selected(), GameVersion::V1_7_10);
        let generation = client.mods.settings.generation();
        let newer = slot_centre(&mut client, GameVersion::V1_8_9);
        click(&mut client, newer);
        assert_eq!(client.config.game_version, None, "the override ends");
        assert_eq!(client.selected_game_version(), GameVersion::V1_8_9);
        assert_eq!(
            client.mods.settings.generation(),
            generation,
            "1.8.9 was the saved pick already: nothing to save"
        );
        assert_eq!(version_line(&client), "Minecraft 1.8.9");
    }

    /// The handshake's protocol number, as a server on this machine reads it from `client`'s
    /// next connection.
    fn next_connections_protocol(client: &mut Client) -> u8 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("loopback listener");
        let port = listener.local_addr().expect("listener address").port();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().expect("the client connects");
            socket
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("read timeout");
            // The frame's length, `C00Handshake`'s id and the protocol: one byte each here.
            let mut start = [0_u8; 3];
            socket.read_exact(&mut start).expect("the handshake");
            assert_eq!(start[1], 0x00, "C00Handshake");
            start[2]
        });
        client.connect("127.0.0.1", port);
        server.join().expect("the listener thread")
    }

    #[test]
    fn the_next_connection_plays_the_picked_version_and_keeps_it() {
        let Some((mut client, _dir)) = offline_client(None) else {
            return;
        };
        // Negative control: 1.8.9, the default.
        assert_eq!(next_connections_protocol(&mut client), 47);
        let session = client.session.as_ref().expect("a session");
        assert_eq!(session.version(), GameVersion::V1_8_9);

        let Some((mut client, _dir)) = offline_client(None) else {
            return;
        };
        client.select_game_version(GameVersion::V1_7_10);
        assert_eq!(next_connections_protocol(&mut client), 5);
        assert_eq!(
            client.session.as_ref().expect("a session").version(),
            GameVersion::V1_7_10
        );
        // The setting changes under the live session: what reads the version during the session
        // still names the session's, here a crash report's System Details.
        client.mods.settings.set_game_version(GameVersion::V1_8_9);
        assert_eq!(client.selected_game_version(), GameVersion::V1_8_9);
        assert_eq!(client.current_game_version(), GameVersion::V1_7_10);
        let mut report = crate::app::crash::CrashReport::new("test", "cause");
        client.describe(&mut report);
        let text = report.complete_report();
        assert!(text.contains("Minecraft Version: 1.7.10\n"), "{text}");
        assert!(text.contains("Protocol Version: 5\n"), "{text}");
    }

    #[test]
    fn the_mod_menu_row_cycles_the_version_and_is_greyed_in_a_session() {
        let Some((mut client, _dir)) = offline_client(None) else {
            return;
        };
        let index = client
            .mods
            .settings
            .index_of(MENU_MOD_ID)
            .expect("the title screen's mod");
        client.fill_menu_rows_at(index);
        let row = client.mods.rows[index].rows()[0].clone();
        assert_eq!(
            (row.label.as_str(), row.badge.as_str()),
            ("Game version", "1.8.9")
        );
        assert!(row.cycle && !row.muted);
        client.mods.events.push(ModEvent::RowCycle {
            mod_id: MENU_MOD_ID,
            row: row.key,
            forward: true,
        });
        client.game_version_menu_frame();
        assert_eq!(client.mods.settings.game_version(), GameVersion::V1_7_10);
        assert!(client.audio.last_play().is_none(), "the mod menu is silent");
        client.fill_menu_rows_at(index);
        assert_eq!(client.mods.rows[index].rows()[0].badge, "1.7.10");

        // In a session the row is greyed and fixed, and a click changes nothing.
        let (handle, _to_game, _from_game) = crate::net::NetHandle::new_pair();
        client.session = Some(crate::session::Session::new(handle, Default::default()));
        client.fill_menu_rows_at(index);
        let row = &client.mods.rows[index].rows()[0];
        assert!(row.muted && !row.cycle, "greyed");
        client.mods.events.push(ModEvent::RowCycle {
            mod_id: MENU_MOD_ID,
            row: row.key,
            forward: true,
        });
        client.game_version_menu_frame();
        assert_eq!(client.mods.settings.game_version(), GameVersion::V1_7_10);
    }

    /// A pick while the server list is open shows the list to the new version as opening it
    /// does (`open_multiplayer_screen`): the pings in flight are dropped under a fresh generation,
    /// and every row forgets the other version's answer, takes the new version's defaults and is
    /// pinged again (for the version `drive_pings` asks with). First with the list up, then from
    /// the mod menu's row with the list waiting behind the menu, as `open_mod_menu_over` leaves a
    /// screen. A pick of the version the list already shows changes nothing. Nothing is pinged
    /// here: no frame runs `drive_pings`.
    #[test]
    fn a_pick_pings_an_open_server_list_again_for_the_new_version() {
        use std::time::Instant;

        use mc_ui::screen_stack::ScreenCtx;
        use mc_ui::screens::multiplayer::server_data::PingState;
        use mc_ui::screens::multiplayer::{MultiplayerScreen, ServerEntry, StatusResponse};

        use super::MENU_ROW;

        /// The server list, up or behind the mod menu.
        fn list(client: &mut Client) -> &mut MultiplayerScreen {
            match &mut client.ui {
                Ui::Multiplayer(screen) => screen,
                _ => match client.mods.menu_return.as_deref_mut() {
                    Some(Ui::Multiplayer(screen)) => screen,
                    _ => panic!("no server list"),
                },
            }
        }
        /// The rows the next draw pings.
        fn next_pings(client: &mut Client) -> Vec<usize> {
            let screen = list(client);
            screen.queue_visible_pings();
            let queue = screen.take_ping_queue();
            queue.into_iter().map(|(index, _)| index).collect()
        }
        /// The first row answered by a `version` server; the second row's ping stays in flight.
        fn answer_first_row(client: &mut Client, version: GameVersion) {
            let json = format!(
                r#"{{"version":{{"name":"{}","protocol":{}}},"description":"A server"}}"#,
                version.name(),
                version.protocol()
            );
            let status = StatusResponse::parse(&json).expect("a status");
            let Ui::Multiplayer(screen) = &mut client.ui else {
                panic!("the server list is up");
            };
            screen.apply_ping(0, &status, 30, &client.assets.language);
            assert_eq!(screen.servers.entries()[0].protocol, version.protocol());
            assert_eq!(screen.servers.entries()[1].state, PingState::AwaitingStatus);
        }
        fn assert_restamped(client: &mut Client, version: GameVersion, generation: u64) {
            let screen = list(client);
            assert_ne!(screen.ping_generation(), generation, "a fresh generation");
            for entry in screen.servers.entries() {
                assert_eq!(entry.client_version, version, "{}", entry.name);
                assert_eq!(entry.state, PingState::Unqueried, "{}", entry.name);
                assert_eq!(entry.protocol, version.protocol(), "{}", entry.name);
                assert_eq!(entry.game_version, version.name(), "{}", entry.name);
                assert!(entry.motd.is_empty(), "{}", entry.name);
            }
        }

        let Some((mut client, _dir)) = offline_client(None) else {
            return;
        };
        client.servers.add(ServerEntry::new("First", "127.0.0.1:1"));
        client
            .servers
            .add(ServerEntry::new("Second", "127.0.0.1:2"));
        client.open_multiplayer_screen();
        let ctx = ScreenCtx::new(client.resolution(SIZE.0, SIZE.1), 0.0);
        let Ui::Multiplayer(screen) = &mut client.ui else {
            panic!("the server list is up");
        };
        screen.init(&ctx, &client.assets.language);
        assert_eq!(
            next_pings(&mut client),
            [0, 1],
            "the first draw pings both rows"
        );
        answer_first_row(&mut client, GameVersion::V1_8_9);
        assert!(
            next_pings(&mut client).is_empty(),
            "one row answered, one in flight"
        );

        // The list up.
        let generation = list(&mut client).ping_generation();
        client.select_game_version(GameVersion::V1_7_10);
        assert_restamped(&mut client, GameVersion::V1_7_10, generation);
        assert_eq!(client.selected_game_version(), GameVersion::V1_7_10);
        assert_eq!(
            next_pings(&mut client),
            [0, 1],
            "both rows are pinged again"
        );

        // The version the list shows already: nothing changes, the pings stay in flight.
        let generation = list(&mut client).ping_generation();
        client.select_game_version(GameVersion::V1_7_10);
        assert_eq!(list(&mut client).ping_generation(), generation);
        let states: Vec<_> = list(&mut client)
            .servers
            .entries()
            .iter()
            .map(|entry| entry.state)
            .collect();
        assert_eq!(states, [PingState::AwaitingStatus; 2]);

        // From the mod menu's row, the list behind the menu.
        answer_first_row(&mut client, GameVersion::V1_7_10);
        client.open_mod_menu_over(Instant::now());
        assert!(matches!(client.ui, Ui::ModMenu(_)));
        let generation = list(&mut client).ping_generation();
        client.mods.events.push(ModEvent::RowCycle {
            mod_id: MENU_MOD_ID,
            row: MENU_ROW,
            forward: true,
        });
        client.game_version_menu_frame();
        assert_eq!(client.mods.settings.game_version(), GameVersion::V1_8_9);
        assert_restamped(&mut client, GameVersion::V1_8_9, generation);
        client.close_mod_menu(Instant::now());
        assert!(matches!(client.ui, Ui::Multiplayer(_)));
        assert_eq!(
            next_pings(&mut client),
            [0, 1],
            "both rows are pinged again"
        );
    }

    /// The version slider over the real title screen (`Client::frame`: the greenhouse, its VHS
    /// finish and the crisp marks) at 1920x1080: 1.8.9 picked, the pointer over 1.7.10, and
    /// 1.7.10 picked by a click. Writes PNGs only when `MC_VERSION_SLIDER_REVIEW_DIR` is set:
    ///
    /// `MC_VERSION_SLIDER_REVIEW_DIR=<dir> cargo test -p mc-client --lib slider_review -- --ignored`
    #[test]
    #[ignore = "writes the review PNGs; run explicitly with MC_VERSION_SLIDER_REVIEW_DIR"]
    fn version_slider_review_renders() {
        const REVIEW: (u32, u32) = (1920, 1080);
        let Some(out) = std::env::var_os("MC_VERSION_SLIDER_REVIEW_DIR") else {
            eprintln!("skip: set MC_VERSION_SLIDER_REVIEW_DIR to write the review images");
            return;
        };
        let out = std::path::PathBuf::from(out);
        std::fs::create_dir_all(&out).expect("review dir");
        let Some(mut gpu) = gpu_sized(REVIEW) else {
            return;
        };
        let Some((mut client, _dir)) = offline_client(None) else {
            return;
        };
        let mut phases = PhaseList::new();
        client.install(&gpu, &mut phases, REVIEW.0, REVIEW.1);
        let mut camera = crate::app::driver::menu_camera();
        camera.set_viewport(REVIEW.0, REVIEW.1);
        // The first frame builds the title screen and its scene.
        client.frame(&gpu, &mut camera, 0.0);
        // 12 s into the scene, the pop-in long over.
        let (shown, popin) = (Duration::from_secs(12), Duration::from_secs(11));
        let mut shoot = |client: &mut Client, cursor: (f32, f32), name: &str| {
            let mut input = InputState::new();
            input.on_cursor_moved(f64::from(cursor.0), f64::from(cursor.1));
            client.handle_screen_input(&mut input, REVIEW.0, REVIEW.1);
            // The highlights and the knob ease on real time.
            for _ in 0..3 {
                client.aero.set_live_clocks(shown, popin);
                client.frame(&gpu, &mut camera, 0.0);
                std::thread::sleep(Duration::from_millis(150));
            }
            client.frame(&gpu, &mut camera, 0.0);
            let target = gpu.acquire().expect("offscreen frame");
            phases.render(&gpu, target.view(), &camera, 0.0);
            let pixels = gpu.read_pixels().expect("readback");
            let path = out.join(name);
            image::save_buffer(
                &path,
                &pixels,
                REVIEW.0,
                REVIEW.1,
                image::ExtendedColorType::Rgba8,
            )
            .expect("write the PNG");
            eprintln!("wrote {}", path.display());
        };
        let away = (REVIEW.0 as f32 * 0.8, REVIEW.1 as f32 * 0.5);
        shoot(&mut client, away, "greenhouse-1.8.9.png");
        let older = slot_centre(&mut client, GameVersion::V1_7_10);
        shoot(&mut client, older, "greenhouse-1.8.9-hover.png");
        click_sized(&mut client, older, REVIEW);
        assert_eq!(client.selected_game_version(), GameVersion::V1_7_10);
        shoot(&mut client, away, "greenhouse-1.7.10.png");
    }
}
