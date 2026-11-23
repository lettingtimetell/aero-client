//! The client half of `action_bar`, `titles`, `tab_list` and
//! `scrollable_tooltips`, and the pause menu's Multiplayer button.
//!
//! Every mod hook reads one generation-gated snapshot ([`X3State`]), so with all four mods off
//! each hook is a `u64` compare plus a few `bool` tests and every vanilla path runs unchanged:
//!
//! * ([`Client::x3_frame`]): sync, push the tab-list style (a small `Copy`),
//!   hand the tooltip options to an open container screen while the mod is on, and watch the
//!   in-game server list (below).
//! * ([`Client::x3_before_vanilla_hud`]), just before the vanilla HUD
//!   draw: detach the overlay message and the titles while their mods are on, and fill their
//!   `HudData` with vanilla's colour/alpha at this frame's partial ticks.
//! * In `mouse_wheel`: `TooltipScroll::wheel` (a `bool` while off).
//!
//! # Multiplayer from the pause menu (item 23)
//!
//! `PauseScreen`'s Multiplayer button sends `Action::OpenMultiplayer`; the list is the ordinary
//! `Ui::Multiplayer`, and the session keeps being polled and ticked behind it (the world keeps
//! running, keep-alives are answered: nothing is paused in multiplayer). Nothing but the pause
//! menu shows the list while a session is alive, so the rules key off the live session:
//!
//! * **Cancel / Escape** (`Action::Back`): In `go_back` returns to the
//!   pause menu instead of the title screen ([`Client::x3_back_to_pause`]).
//! * **Join** (`Action::Connect`, from the list or its Direct Connect screen):
//!   Runs [`Client::x3_leave_before_join`] first: the exact
//!   teardown of the pause menu's Disconnect button (`Client::disconnect`: the channel is closed
//!   with no packet, the world unloaded, server resource packs dropped), then the list is rebuilt
//!   so the join continues exactly as from the title screen's list (failure returns to the list).
//! * **Anything else** that replaces the list while it is open (the server kicks us, opens a
//!   window, sends a resource-pack prompt): the list's `ServerList` lives inside the screen, so
//!   `<<FW-X3:open-multiplayer>>` arms a per-frame watch ([`Client::x3_note_multiplayer_open`],
//!   [`Client::x3_watch_server_list`]) that notices the screen is gone and re-reads
//!   `servers.dat` (every list edit is written through at once). The same guard runs before the
//!   list is shown again and before the shutdown save, so a dropped
//!   list can never be saved as an empty `servers.dat`.

use mc_mods::ModSettings;
use mc_ui::screens::tab_list::{TabListStyle, TabPing};

use super::super::{Client, Ui};

/// The state on the client (`Client::x3`).
#[derive(Debug, Default)]
pub(crate) struct X3State {
    /// The settings generation the snapshot was taken at; 0 = never (the store starts at 1).
    seen: u64,
    action_bar: bool,
    titles: bool,
    /// The tab-list style: [`TabListStyle::VANILLA`] while `tab_list` is off.
    tab_style: TabListStyle,
    tooltips: bool,
    tooltip_step: i32,
    /// The options were last pushed into an open container screen as enabled, so turning the
    /// mod off pushes "off" once.
    tooltips_pushed: bool,
    /// A server list was opened over a live connection (from the pause menu) and has not been
    /// closed yet: the per-frame watch is armed.
    pub(crate) mp_over_game: bool,
    /// Confirm Disconnect's join question was answered "Join Server":
    /// the one `Action::Connect` it re-sends passes without asking again (`super::w18_fb`).
    pub(crate) fb_join_confirmed: bool,
}

impl X3State {
    /// Refresh the snapshot when the settings generation moved. Returns whether it did.
    fn sync(&mut self, settings: &ModSettings) -> bool {
        if self.seen == settings.generation() {
            return false;
        }
        self.seen = settings.generation();
        self.action_bar = settings.enabled("action_bar");
        self.titles = settings.enabled("titles");
        self.tab_style = if settings.enabled("tab_list") {
            TabListStyle {
                header: settings.bool("tab_list", "header"),
                footer: settings.bool("tab_list", "footer"),
                ping: match settings.choice("tab_list", "ping") {
                    0 => TabPing::Bars,
                    1 => TabPing::Numbers,
                    _ => TabPing::Hidden,
                },
                background: (settings.f32("tab_list", "background_opacity") / 100.0)
                    .clamp(0.0, 1.0),
                background_color: settings.color("tab_list", "background_color"),
                row_color: settings.color("tab_list", "row_color"),
            }
        } else {
            TabListStyle::VANILLA
        };
        self.tooltips = settings.enabled("scrollable_tooltips");
        self.tooltip_step = settings.f32("scrollable_tooltips", "step").round() as i32;
        true
    }
}

impl Client {
    /// Once per frame: one `u64` compare, a 12-byte copy and two `bool` tests
    /// while every X3 mod is off and no in-game server list is open.
    pub(crate) fn x3_frame(&mut self) {
        self.x3.sync(&self.mods.settings);
        // A plain copy every frame: the client replaces `tab_list` wholesale on world changes.
        self.tab_list.style = self.x3.tab_style;
        if self.x3.tooltips || self.x3.tooltips_pushed {
            let (on, step) = (self.x3.tooltips, self.x3.tooltip_step);
            match &mut self.ui {
                Ui::Container(screen) => screen.tooltip_scroll.set_options(on, step),
                Ui::Creative(screen) => screen.tooltip_scroll.set_options(on, step),
                _ => {}
            }
            self.x3.tooltips_pushed = on;
        }
        if self.x3.mp_over_game {
            self.x3_watch_server_list();
        }
    }

    /// Just before `Hud::draw_with_spectator_and_crosshair`: detach the
    /// overlay message and the titles while their mods are on, and hand their elements the text
    /// with the colour/alpha vanilla's draw would use at `partial_ticks`. Strings are copied only
    /// when they changed.
    pub(crate) fn x3_before_vanilla_hud(&mut self, partial_ticks: f32) {
        self.x3.sync(&self.mods.settings);
        self.hud.action_bar_detached = self.x3.action_bar;
        self.hud.titles_detached = self.x3.titles;
        if self.x3.action_bar {
            let bar = &mut self.mods.hud_data.action_bar;
            bar.color = self.hud.record_color(partial_ticks);
            if bar.text != self.hud.record_playing {
                bar.text.clear();
                bar.text.push_str(&self.hud.record_playing);
            }
        }
        if self.x3.titles {
            let titles = &mut self.mods.hud_data.titles;
            titles.alpha = self.hud.title_alpha(partial_ticks);
            if titles.title != self.hud.title {
                titles.title.clear();
                titles.title.push_str(&self.hud.title);
            }
            if titles.subtitle != self.hud.subtitle {
                titles.subtitle.clear();
                titles.subtitle.push_str(&self.hud.subtitle);
            }
        }
    }

    /// As `Action::OpenMultiplayer` is handled: a list opened over
    /// a live session (from the pause menu) is watched until it closes (see the module docs).
    pub(crate) fn x3_note_multiplayer_open(&mut self) {
        self.x3.mp_over_game = self.session.is_some();
    }

    /// In `go_back`, after the list handed its servers back: `true`
    /// (and the pause menu is on screen) while a connection is alive, i.e. the list was opened
    /// from the pause menu (nothing else shows it in a world); `false` leaves vanilla's way back
    /// to the title screen.
    pub(crate) fn x3_back_to_pause(&mut self) -> bool {
        self.x3.mp_over_game = false;
        if self.session.is_none() || !matches!(self.ui, Ui::Multiplayer(_)) {
            return false;
        }
        self.ui = Ui::Pause(Box::default());
        true
    }

    /// At the top of `Action::Connect`: joining a server while a
    /// connection is alive (from the in-game list or its Direct Connect screen) first leaves the
    /// current one exactly like the pause menu's Disconnect button (the channel closed, no
    /// packet; the world, its sounds and the server resource pack dropped), then rebuilds the
    /// list so the join that follows runs as from the title screen's list.
    pub(crate) fn x3_leave_before_join(&mut self) {
        self.x3.mp_over_game = false;
        if self.session.is_none() {
            return;
        }
        // The list owns the servers while it is open; `disconnect` replaces the UI.
        self.stash_servers();
        self.disconnect();
        self.open_multiplayer_screen();
    }

    /// The in-game list (or one of its sub-screens) was replaced without going through Cancel
    /// or Join (a kick, a server-opened window, a resource-pack prompt), or the connection ended
    /// underneath it: stop watching and, when the replaced screen took the server list with it,
    /// reload `servers.dat` at once.
    fn x3_watch_server_list(&mut self) {
        // Confirm Disconnect's join question holds the list too.
        let in_list = matches!(
            self.ui,
            Ui::Multiplayer(_) | Ui::DirectConnect(_) | Ui::AddServer(_)
        ) || matches!(
            self.ui,
            Ui::ConfirmDeleteServer(..) | Ui::ConfirmLeaveServer(..)
        );
        if in_list && self.session.is_some() {
            return;
        }
        self.x3.mp_over_game = false;
        if !in_list {
            self.cancel_server_pings();
            self.x3_reload_servers_if_lost();
        }
    }

    /// `self.servers` is empty while no list screen holds it, so the
    /// screen that held it was dropped (or the player really has no servers): re-read
    /// `servers.dat`, which every list edit writes through at once. Runs only when the server
    /// list is about to be shown or saved (`open_multiplayer_screen`, `save_server_list`) and
    /// when the in-game list was replaced, never per frame. Without it, quitting after such a
    /// drop (or, before this change, quitting while the delete confirmation was open) wrote an
    /// empty `servers.dat` over the player's list.
    pub(crate) fn x3_reload_servers_if_lost(&mut self) {
        if !self.servers.is_empty() || matches!(self.ui, Ui::Multiplayer(_)) {
            return;
        }
        let Ok(bytes) = std::fs::read(self.config.servers_path()) else {
            return;
        };
        match mc_ui::screens::multiplayer::ServerList::read(&bytes) {
            Ok(list) => {
                if !list.is_empty() {
                    tracing::info!(
                        servers = list.len(),
                        "server list restored from servers.dat"
                    );
                }
                self.servers = list;
            }
            Err(error) => tracing::warn!(%error, "servers.dat unreadable; list left empty"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::app::state::Transition;
    use mc_ui::screens::multiplayer::ServerEntry;
    use mc_ui::screens::Action;

    /// A real offline client over a temporary game dir (skips without the vanilla jar).
    fn offline_client() -> Option<(Client, tempfile::TempDir)> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
            eprintln!("skip: no vanilla jar in this checkout");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        Some((client, dir))
    }

    /// Put `client` in a world over a socketless session; the channel ends come back so the
    /// test can feed packets in and read what the client sends.
    fn join_fake_world(
        client: &mut Client,
    ) -> (
        tokio::sync::mpsc::Sender<crate::net::ToGame>,
        tokio::sync::mpsc::UnboundedReceiver<crate::net::ToNet>,
    ) {
        let (handle, to_game, to_net) = crate::net::channel::NetHandle::new_pair();
        client
            .machine
            .apply(Transition::Connect {
                host: "127.0.0.1".to_owned(),
                port: 25_565,
            })
            .expect("menu -> connecting");
        client
            .machine
            .apply(Transition::Joined)
            .expect("connecting -> in world");
        assert!(client.machine.state().is_in_world());
        client.session = Some(crate::Session::new(
            handle,
            crate::session::JoinSettings::default(),
        ));
        client.ui = Ui::InGame;
        (to_game, to_net)
    }

    fn two_servers(client: &mut Client) {
        client
            .servers
            .add(ServerEntry::new("Example", "mc.example.net"));
        client
            .servers
            .add(ServerEntry::new("Local", "127.0.0.1:25599"));
        client
            .write_server_list(&client.servers)
            .expect("servers.dat in the temp dir");
    }

    #[test]
    fn the_snapshot_is_vanilla_while_off_and_follows_the_options() {
        let mut settings = ModSettings::new();
        let mut state = X3State::default();
        assert!(state.sync(&settings));
        assert!(!state.sync(&settings), "same generation: no work");
        assert!(!state.action_bar && !state.titles && !state.tooltips);
        assert_eq!(state.tab_style, TabListStyle::VANILLA);

        settings.set_enabled("tab_list", true);
        state.sync(&settings);
        assert_eq!(
            state.tab_style,
            TabListStyle {
                ping: TabPing::Numbers,
                ..TabListStyle::VANILLA
            },
            "enabled at its defaults: ping numbers, everything else vanilla"
        );
        settings.set_bool("tab_list", "header", false);
        settings.set_bool("tab_list", "footer", false);
        settings.set_choice("tab_list", "ping", 2);
        settings.set_f32("tab_list", "background_opacity", 40.0);
        state.sync(&settings);
        assert_eq!(
            state.tab_style,
            TabListStyle {
                header: false,
                footer: false,
                ping: TabPing::Hidden,
                background: 0.4,
                ..TabListStyle::VANILLA
            }
        );
        settings.set_enabled("tab_list", false);
        state.sync(&settings);
        assert_eq!(state.tab_style, TabListStyle::VANILLA);

        settings.set_enabled("scrollable_tooltips", true);
        settings.set_enabled("action_bar", true);
        settings.set_enabled("titles", true);
        state.sync(&settings);
        assert!(state.tooltips && state.action_bar && state.titles);
        assert_eq!(state.tooltip_step, 10);
    }

    /// Off: nothing is detached and the HUD data is never written. On: detached, and the data
    /// carries vanilla's text, colour and alpha.
    #[test]
    fn the_vanilla_hud_hook_detaches_and_fills_only_while_on() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        client.hud.set_record_playing("\u{a7}eKills: 3", false);
        client.hud.display_title("Victory!", "GG");
        client.x3_before_vanilla_hud(0.0);
        assert!(!client.hud.action_bar_detached && !client.hud.titles_detached);
        assert!(client.mods.hud_data.action_bar.text.is_empty());
        assert!(client.mods.hud_data.titles.alpha.is_none());

        client.mods.settings.set_enabled("action_bar", true);
        client.mods.settings.set_enabled("titles", true);
        client.x3_before_vanilla_hud(0.5);
        assert!(client.hud.action_bar_detached && client.hud.titles_detached);
        let bar = &client.mods.hud_data.action_bar;
        assert_eq!(bar.text, "\u{a7}eKills: 3");
        assert_eq!(bar.color, client.hud.record_color(0.5));
        let titles = &client.mods.hud_data.titles;
        assert_eq!(
            (titles.title.as_str(), titles.subtitle.as_str()),
            ("Victory!", "GG")
        );
        assert_eq!(titles.alpha, client.hud.title_alpha(0.5));
        // The text buffer is reused, not reallocated, while the message stays the same.
        let ptr = client.mods.hud_data.action_bar.text.as_ptr();
        client.x3_before_vanilla_hud(0.7);
        assert_eq!(client.mods.hud_data.action_bar.text.as_ptr(), ptr);

        client.mods.settings.set_enabled("action_bar", false);
        client.mods.settings.set_enabled("titles", false);
        client.x3_before_vanilla_hud(0.0);
        assert!(
            !client.hud.action_bar_detached && !client.hud.titles_detached,
            "off again: vanilla draws them"
        );
    }

    #[test]
    fn the_frame_hook_pushes_the_tab_style_and_the_tooltip_options() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        client.x3_frame();
        assert_eq!(client.tab_list.style, TabListStyle::VANILLA);
        client.mods.settings.set_enabled("tab_list", true);
        client.x3_frame();
        assert_eq!(client.tab_list.style.ping, TabPing::Numbers);
        // A replaced overlay (world change) gets the style back on the next frame.
        client.tab_list = mc_ui::screens::tab_list::TabListOverlay::new();
        client.x3_frame();
        assert_eq!(client.tab_list.style.ping, TabPing::Numbers);

        client.ui = Ui::Container(Box::new(
            mc_ui::screens::container::ContainerScreen::player(0),
        ));
        client.x3_frame();
        let Ui::Container(screen) = &client.ui else {
            unreachable!()
        };
        assert!(
            !screen.tooltip_scroll.enabled(),
            "mod off: vanilla tooltips"
        );
        client
            .mods
            .settings
            .set_enabled("scrollable_tooltips", true);
        client.x3_frame();
        let Ui::Container(screen) = &client.ui else {
            unreachable!()
        };
        assert!(screen.tooltip_scroll.enabled());
    }

    /// Pause -> Multiplayer keeps the session; Cancel and Escape come back to the pause menu
    /// with the server list intact; packets keep being applied while the list is open.
    #[test]
    fn the_in_game_server_list_keeps_the_connection_and_returns_to_pause() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        two_servers(&mut client);
        let (to_game, mut to_net) = join_fake_world(&mut client);
        client.show_screen(Some(Ui::Pause(Box::default())));

        // The pause menu's button.
        client.handle_action(Action::OpenMultiplayer);
        assert!(matches!(client.ui, Ui::Multiplayer(_)));
        assert!(client.x3.mp_over_game);
        assert!(client.session.is_some(), "the connection is kept");
        assert!(client.machine.state().is_in_world());

        // The game keeps running behind the list: a chat packet is applied.
        to_game
            .try_send(crate::net::ToGame::Packet(
                mc_protocol::play::S2CPlay::Chat(mc_protocol::play::s2c::Chat {
                    component: r#"{"text":"still here"}"#.into(),
                    position: 0,
                }),
            ))
            .expect("queue a packet");
        client.poll();
        client.x3_frame();
        assert!(matches!(client.ui, Ui::Multiplayer(_)));
        assert!(client.x3.mp_over_game, "nothing replaced the list");
        assert!(
            client
                .chat
                .lines()
                .iter()
                .any(|line| line.text.contains("still here")),
            "the session is polled while the list is open"
        );

        // Cancel: back to the pause menu, the servers are handed back.
        client.handle_action(Action::Back);
        assert!(matches!(client.ui, Ui::Pause(_)));
        assert!(!client.x3.mp_over_game);
        assert_eq!(client.servers.len(), 2);
        assert!(client.session.is_some());

        // Escape takes the same way back.
        client.handle_action(Action::OpenMultiplayer);
        assert!(matches!(client.ui, Ui::Multiplayer(_)));
        client.handle_action(Action::Back);
        assert!(matches!(client.ui, Ui::Pause(_)));
        assert_eq!(client.servers.len(), 2);

        // Direct Connect's Cancel returns to the in-game list, whose Cancel still goes to pause.
        client.handle_action(Action::OpenMultiplayer);
        client.handle_action(Action::OpenDirectConnect);
        client.handle_action(Action::Back);
        assert!(matches!(client.ui, Ui::Multiplayer(_)));
        assert!(client.x3.mp_over_game);
        client.handle_action(Action::Back);
        assert!(matches!(client.ui, Ui::Pause(_)));

        // Nothing at all was sent to the server by browsing.
        let sent: Vec<_> = std::iter::from_fn(|| to_net.try_recv().ok()).collect();
        assert!(sent.is_empty(), "browsing the list sent {sent:?}");
    }

    /// Joining from the in-game list closes the current connection first (the Disconnect
    /// button's teardown: `ToNet::Disconnect`, no packet), then connects as from the title
    /// screen's list, whose parent the connecting screen keeps.
    #[test]
    fn joining_from_the_in_game_list_disconnects_cleanly_first() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        two_servers(&mut client);
        let (_to_game, mut to_net) = join_fake_world(&mut client);
        client.show_screen(Some(Ui::Pause(Box::default())));
        client.handle_action(Action::OpenMultiplayer);
        assert!(client.x3.mp_over_game);

        client.handle_action(Action::Connect(Box::new(ServerEntry::new(
            "Local",
            "127.0.0.1:25599",
        ))));
        let mut closed = false;
        while let Ok(message) = to_net.try_recv() {
            match message {
                crate::net::ToNet::Disconnect => closed = true,
                crate::net::ToNet::Send(packet) => panic!("leaving sent {packet:?}"),
            }
        }
        assert!(closed, "the old channel was closed");
        assert!(client.session.is_none(), "the old world is gone");
        assert!(!client.x3.mp_over_game);
        let Ui::Connecting(screen) = &client.ui else {
            panic!("the join continues on the connecting screen");
        };
        assert_eq!(
            screen.parent(),
            mc_ui::screens::connect::ConnectionParent::Multiplayer,
            "a failed join returns to the server list"
        );
        assert_eq!(client.servers.len(), 2, "the list survived the teardown");
        assert!(!client.machine.state().is_in_world());
        client.active_connect_resolution = None;
    }

    /// Pre-existing, fixed by the same guard: quitting while the title screen's delete
    /// confirmation is open (the list lives inside its boxed parent) wrote an empty servers.dat.
    #[test]
    fn quitting_from_the_delete_confirmation_keeps_servers_dat() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        two_servers(&mut client);
        client.handle_action(Action::OpenMultiplayer);
        client.handle_action(Action::ConfirmDeleteServer("Example".to_owned()));
        assert!(matches!(client.ui, Ui::ConfirmDeleteServer(..)));
        assert!(
            client.servers.is_empty(),
            "the dialog's parent holds the list"
        );
        use crate::app::shutdown::ShutdownTarget as _;
        client.save_server_list().expect("save");
        let saved = mc_ui::screens::multiplayer::ServerList::read(
            &std::fs::read(client.config.servers_path()).expect("servers.dat"),
        )
        .expect("parse");
        assert_eq!(saved.len(), 2);
    }

    /// The server replaces the in-game list (a kick here): the pause parent is forgotten and the
    /// server list, which the dropped screen held, is reloaded from servers.dat.
    #[test]
    fn a_kick_while_browsing_keeps_the_server_list() {
        let Some((mut client, _dir)) = offline_client() else {
            return;
        };
        two_servers(&mut client);
        let (_to_game, _to_net) = join_fake_world(&mut client);
        client.show_screen(Some(Ui::Pause(Box::default())));
        client.handle_action(Action::OpenMultiplayer);
        assert!(
            client.servers.is_empty(),
            "the list screen holds the servers"
        );

        // What a play disconnect does: end the connection, show the disconnected screen (which
        // replaces the list), drop the world.
        let reason = crate::DisconnectReason::ConnectionLost {
            detail: "kicked".to_owned(),
        };
        client
            .machine
            .apply(Transition::Ended(reason.clone()))
            .expect("in world -> disconnected");
        client.show_disconnected("disconnect.lost", &reason);
        client.session = None;
        client.x3_frame();
        assert!(!client.x3.mp_over_game);
        assert_eq!(client.servers.len(), 2, "reloaded from servers.dat");

        // A second kick that drops the list before the watch runs: the shutdown save re-reads
        // servers.dat instead of writing the empty in-memory list over it.
        client
            .machine
            .apply(Transition::BackToMenu)
            .expect("disconnected -> menu");
        let (_to_game, _to_net) = join_fake_world(&mut client);
        client.show_screen(Some(Ui::Pause(Box::default())));
        client.handle_action(Action::OpenMultiplayer);
        client.ui = Ui::InGame; // dropped without stashing, no frame in between
        use crate::app::shutdown::ShutdownTarget as _;
        client.save_server_list().expect("save");
        let saved = mc_ui::screens::multiplayer::ServerList::read(
            &std::fs::read(client.config.servers_path()).expect("servers.dat"),
        )
        .expect("parse");
        assert_eq!(saved.len(), 2, "quitting never wipes servers.dat");
        client.session = None;
        client.x3.mp_over_game = false;

        // The title screen's list is unaffected: opening it there sets no flag.
        client.ui = Ui::MainMenu(Box::new(mc_ui::screens::menu::MainMenuScreen::new(
            "splash".to_owned(),
        )));
        client.handle_action(Action::OpenMultiplayer);
        assert!(!client.x3.mp_over_game);
        client.handle_action(Action::Back);
        assert!(matches!(client.ui, Ui::MainMenu(_)));
    }
}
