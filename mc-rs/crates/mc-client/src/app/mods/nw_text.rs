//! The chat half: "scrolling up in chat is remembered and
//! prevents you from seeing new sent messages".
//!
//! java: `GuiChat.onGuiClosed` calls `GuiNewChat.resetScroll`, and `Minecraft.displayGuiScreen`
//! runs `onGuiClosed` on every way out of the chat screen: Escape and Enter
//! (`GuiChat.keyTyped` -> `displayGuiScreen(null)`, whether a message was sent or the line was
//! empty), a kick's `GuiDisconnected`, a dimension change's `GuiDownloadTerrain`, a
//! server-opened container. So in vanilla a closed chat is never scrolled, and the newest line
//! is always the bottom row of the HUD chat.
//!
//! mc-rs reset the scroll only in [`Client::show_screen`]. Escape and an empty Enter come back
//! from `ChatScreen::key_typed` as `Action::Back`, which `go_back` handled by assigning
//! `Ui::InGame` directly, so a chat scrolled up stayed scrolled after closing and every new
//! message arrived below the visible rows. `go_back` now takes the `show_screen(None)` seam,
//! and [`Client::nw_text_closed_chat_scroll`] enforces vanilla's
//! invariant once per frame for the routes that still replace the chat screen by direct
//! assignment.

use super::super::Client;

impl Client {
    /// A closed chat is at the bottom, as vanilla's
    /// `GuiChat.onGuiClosed` guarantees on every route out of the chat screen. Called once per
    /// frame with whether a chat-owning screen (`Ui::Chat`/`Ui::Sleep`) is open; costs one
    /// integer test.
    pub(crate) fn nw_text_closed_chat_scroll(&mut self, chat_open: bool) {
        if !chat_open && self.chat.scroll_pos() != 0 {
            self.chat.reset_scroll();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::Ui;
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::app::input::KeyPress;
    use crate::net::{ToGame, ToNet};
    use crate::session::{JoinSettings, Session};
    use mc_protocol::play::C2SPlay;
    use mc_ui::font::component::ChatComponent;
    use mc_ui::screens::chat::ChatLayout;
    use mc_ui::widget::keys::Modifiers;
    use tokio::sync::mpsc::UnboundedReceiver;
    use winit::keyboard::KeyCode;

    type Fixture = (Client, tempfile::TempDir, UnboundedReceiver<ToNet>);

    /// A client on the real assets with a live, socket-less session (closing a screen returns to
    /// the game, not the title screen). `None` = skip without the vanilla jar (`MC_JARS_DIR`).
    fn client_in_game() -> Option<Fixture> {
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
        let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        let (handle, to_game, mut to_net) = crate::net::channel::NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        to_game
            .try_send(ToGame::Packet(mc_protocol::play::S2CPlay::JoinGame(
                mc_protocol::play::s2c::JoinGame {
                    entity_id: 1,
                    hardcore_mode: false,
                    game_mode: 0,
                    dimension: 0,
                    difficulty: 2,
                    max_players: 20,
                    level_type: "default".into(),
                    reduced_debug_info: false,
                },
            )))
            .expect("channel");
        session.poll(16);
        client.session = Some(session);
        client.ui = Ui::InGame;
        drop(to_game);
        let _ = sent(&mut to_net);
        Some((client, dir, to_net))
    }

    fn sent(to_net: &mut UnboundedReceiver<ToNet>) -> Vec<C2SPlay> {
        let mut out = Vec::new();
        while let Ok(message) = to_net.try_recv() {
            if let ToNet::Send(packet) = message {
                out.push(packet);
            }
        }
        out
    }

    /// A received chat message, through the same `ChatLog::add_message` the S02 handler uses.
    fn receive(client: &mut Client, text: &str) {
        let layout = ChatLayout::from_settings(&client.settings);
        let chat_open = matches!(client.ui, Ui::Chat(_) | Ui::Sleep(_));
        client.chat.add_message(
            ChatComponent::text(text),
            0,
            client.tick as i32,
            &layout,
            &client.font,
            &client.assets.language,
            chat_open,
        );
    }

    /// The real keyboard path: the shell's buffered press, routed to the open screen.
    fn press(client: &mut Client, code: KeyCode, lwjgl: i32) {
        client.key_pressed(KeyPress {
            sequence: 0,
            code,
            lwjgl,
            typed: None,
            mods: Modifiers::NONE,
            repeat: false,
        });
    }

    /// Fill the log, open the chat and scroll it up with Page Up; returns the scroll position.
    fn open_scrolled(client: &mut Client, lines: usize) -> usize {
        for i in 0..lines {
            receive(client, &format!("old message {i}"));
        }
        client.open_chat("");
        assert!(matches!(client.ui, Ui::Chat(_)), "the chat must open");
        press(client, KeyCode::PageUp, 201);
        let pos = client.chat.scroll_pos();
        assert!(pos > 0, "Page Up must scroll the open chat");
        pos
    }

    /// The bottom HUD row (what a closed chat shows first) is `text`.
    fn bottom_row_is(client: &Client, text: &str) -> bool {
        let layout = ChatLayout::from_settings(&client.settings);
        client
            .chat
            .visible_lines(&layout, false)
            .first()
            .is_some_and(|line| line.text.contains(text))
    }

    #[test]
    fn escape_closes_a_scrolled_chat_at_the_bottom() {
        let Some((mut client, _dir, _to_net)) = client_in_game() else {
            return;
        };
        open_scrolled(&mut client, 60);
        press(&mut client, KeyCode::Escape, 1);
        assert!(matches!(client.ui, Ui::InGame), "Escape closes the chat");
        // Checked before any frame runs: the close itself must reset, as `onGuiClosed` does.
        assert_eq!(
            client.chat.scroll_pos(),
            0,
            "GuiChat.onGuiClosed -> resetScroll"
        );
        assert!(!client.chat.is_scrolled());
        receive(&mut client, "brand new message");
        assert!(
            bottom_row_is(&client, "brand new message"),
            "a message arriving after the close must be the bottom row of the closed chat"
        );
        // Reopening starts at the bottom too.
        client.open_chat("");
        assert_eq!(client.chat.scroll_pos(), 0);
    }

    #[test]
    fn enter_on_an_empty_line_closes_at_the_bottom() {
        let Some((mut client, _dir, mut to_net)) = client_in_game() else {
            return;
        };
        open_scrolled(&mut client, 60);
        press(&mut client, KeyCode::Enter, 28);
        assert!(
            matches!(client.ui, Ui::InGame),
            "an empty Enter closes the chat"
        );
        assert_eq!(client.chat.scroll_pos(), 0);
        assert!(
            sent(&mut to_net).is_empty(),
            "an empty line sends nothing (vanilla)"
        );
    }

    #[test]
    fn sending_a_message_closes_at_the_bottom_and_shows_the_reply() {
        let Some((mut client, _dir, mut to_net)) = client_in_game() else {
            return;
        };
        open_scrolled(&mut client, 60);
        let font = &client.font;
        if let Ui::Chat(screen) = &mut client.ui {
            screen.input.write_text("hello", font);
        }
        press(&mut client, KeyCode::Enter, 28);
        assert!(matches!(client.ui, Ui::InGame));
        assert_eq!(client.chat.scroll_pos(), 0);
        let packets = sent(&mut to_net);
        assert!(
            packets
                .iter()
                .any(|p| matches!(p, C2SPlay::ChatMessage(c) if c.message == "hello")),
            "the message is sent: {packets:?}"
        );
        receive(&mut client, "<Dev> hello");
        assert!(bottom_row_is(&client, "<Dev> hello"));
    }

    #[test]
    fn a_long_history_scrolled_far_up_still_closes_at_the_bottom() {
        let Some((mut client, _dir, _to_net)) = client_in_game() else {
            return;
        };
        // The Chat mod's longer history (1000 lines here).
        client.chat.set_max_lines(1000);
        open_scrolled(&mut client, 900);
        let layout = ChatLayout::from_settings(&client.settings);
        if let Ui::Chat(screen) = &mut client.ui {
            // 100 notches of 7 lines: far up, but short of the clamp at the oldest line.
            for _ in 0..100 {
                screen.mouse_wheel(1.0, false, &mut client.chat, &layout);
            }
        }
        assert!(client.chat.scroll_pos() > 700, "scrolled far up");
        // A message arriving while the chat is open and scrolled keeps the reader in place
        // (vanilla `setChatLine`: `scroll(1)`).
        let before = client.chat.scroll_pos();
        receive(&mut client, "while reading");
        assert_eq!(client.chat.scroll_pos(), before + 1);
        press(&mut client, KeyCode::Escape, 1);
        assert_eq!(client.chat.scroll_pos(), 0);
        assert!(bottom_row_is(&client, "while reading"));
    }

    /// The routes that still replace the chat screen by direct assignment (a kick's
    /// `Ui::Disconnected`, a dimension change's `Ui::Loading`...) are covered by the per-frame
    /// invariant. The first half is the negative control: such a route alone leaves the scroll.
    #[test]
    fn a_direct_replacement_is_brought_to_the_bottom_by_the_frame_invariant() {
        let Some((mut client, _dir, _to_net)) = client_in_game() else {
            return;
        };
        let scrolled = open_scrolled(&mut client, 60);
        // What a dimension-changing S07 does (`driver.rs`, `ClientEvent::Respawned`).
        client.ui = Ui::Loading;
        assert_eq!(
            client.chat.scroll_pos(),
            scrolled,
            "negative control: the direct assignment alone keeps the stale scroll"
        );
        client.nw_text_closed_chat_scroll(false);
        assert_eq!(client.chat.scroll_pos(), 0);
        assert!(!client.chat.is_scrolled());

        // While a chat-owning screen is open the invariant must not touch the reader's scroll.
        client.ui = Ui::InGame;
        let scrolled = open_scrolled(&mut client, 0);
        client.nw_text_closed_chat_scroll(true);
        assert_eq!(client.chat.scroll_pos(), scrolled);
    }
}
