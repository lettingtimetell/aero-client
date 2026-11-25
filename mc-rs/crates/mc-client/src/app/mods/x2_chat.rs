//! Chat Macros (item 1), 256 Character Chat (item 31) and the
//! client-side chat links of the F2 line (open the file, `[COPY]` it). The screenshot pipeline
//! itself (asynchronous readback, the encoder thread, the clipboard image) is `x2_screenshot.rs`.
//!
//! | hook | where | cost with the mod off |
//! |---|---|---|
//! | [`Client::x2_shell_frame`] | `<<FW-X2:shell-frame>>`, `ShellApp::frame` | `u64` + `bool` |
//! | [`Client::x2_chat_input_limit`] | `open_chat`, sleep screen (on open) | a `u64` compare |
//! | [`Client::x2_send_typed_chat`] | `Action::SendChat` / `SendSleepChat` | a `u64` compare |
//! | [`Client::x2_chat_click`] | `<<FW-X2:chat-click>>`, `mouse_clicked` (chat) | an `Option` test |
//!
//! * **Macros send exactly what typing does.** A slot's text goes through the chat input's
//!   character filter (`ChatAllowedCharacters`: no `§`, no control characters) and Java's trim
//!   when the settings change, then a press records it in the sent history (Up recalls it, as
//!   after Enter) and sends one C01 through [`Client::x2_send_typed_chat`], the same function
//!   the chat screen's Enter uses, so the length limit is the chat screen's too.
//! * **In game only, one send per press.** Keys are sampled once per frame as press edges, like
//!   the Spotify keys: a key that is already down when the game regains input (it was pressed in
//!   a screen, or held while one closed) sends nothing until it is pressed again, and the OS key
//!   repeat never reaches a per-frame "is it down" sample as a new edge.
//! * **256-character chat** caps the chat input (and the multiplayer sleep screen's) at 256 and
//!   cuts typed chat there. Always on, by design without an option. A `run_command` click from a
//!   server's component keeps vanilla's 100 (it is the server's text).
//!
//! The chat history. Both options are in the Chat mod and
//! apply **whether or not the mod is on**:
//!
//! | option | default | hook |
//! |---|---|---|
//! | "Keep chat between servers" | on | [`Client::x2_title_screen_chat`] (`show_main_menu`, `show_screen`) |
//! | "History length" | 1000 lines | [`Client::x2_sync`] (each frame: a `u64` compare) |
//!
//! * Vanilla clears chat (the lines and the sent-message history) only when the title screen
//!   is displayed (`Minecraft.displayGuiScreen`'s `GuiMainMenu` branch). With Keep on it never
//!   does; F3+D still clears it on purpose. Off is vanilla.
//! * Joining a server adds nothing to chat: no line naming the server, with Keep on or off.
//! * The history length is [`mc_ui::screens::chat::ChatLog::set_max_lines`]: both of the log's
//!   lists are capped there (vanilla 100). Only the visible rows are drawn, so a long history
//!   costs nothing per frame.

use mc_mods::mods::chat::{HISTORY_LENGTH, HISTORY_MAX, HISTORY_MIN, KEEP_CHAT};
use mc_mods::mods::chat_macros::{SLOTS, SLOT_IDS};
use mc_mods::ModSettings;
use mc_render::input::Input;
use mc_render::InputState;
use mc_ui::font::component::{ClickAction, ResolvedStyle};
use mc_ui::screens::chat::{java_trim, LONG_INPUT_LENGTH};
use mc_ui::widget::keys::filter_allowed_characters;

use super::super::{Client, Ui};
use super::x2_screenshot::ShotPipeline;
use crate::outbound::LONG_CHAT_LENGTH;

/// The Chat mod's id.
pub(crate) const CHAT: &str = "chat";
/// The Chat Macros mod's id.
pub(crate) const MACROS: &str = "chat_macros";

/// The state on the client (`Client::x2`).
#[derive(Debug, Default)]
pub(crate) struct X2State {
    /// The settings generation the cached flags below were read at (0 = never).
    seen: u64,
    /// "Keep chat between servers".
    keep_chat: bool,
    /// "History length", in display lines.
    history_lines: usize,
    macros: Macros,
    /// F2: readbacks in flight, the encoder thread, finished results.
    pub(crate) shots: ShotPipeline,
}

impl X2State {
    /// Re-read the cached options when the settings changed. Returns whether they did.
    fn sync(&mut self, settings: &ModSettings) -> bool {
        if self.seen == settings.generation() {
            return false;
        }
        self.seen = settings.generation();
        self.keep_chat = settings.bool(CHAT, KEEP_CHAT);
        self.history_lines = history_lines(settings.f32(CHAT, HISTORY_LENGTH));
        self.macros.sync(settings);
        true
    }
}

/// The History length slider's value as a line count (the descriptor already snaps it to
/// `100..=5000` on load and on every edit; this only guards the conversion).
fn history_lines(value: f32) -> usize {
    value.clamp(HISTORY_MIN, HISTORY_MAX) as usize
}

/// The Chat Macros slots, as the frame hook needs them.
#[derive(Debug)]
struct Macros {
    enabled: bool,
    keys: [Input; SLOTS],
    /// Each slot's text after the chat filter and Java's trim: exactly what Enter would send.
    /// Empty = the slot is idle.
    texts: [String; SLOTS],
    /// Whether each slot's key was down at the previous sample.
    down: [bool; SLOTS],
    /// False until one sample has been taken since the mod turned on, so a key held at that
    /// moment is not a press.
    armed: bool,
}

impl Default for Macros {
    fn default() -> Self {
        Self {
            enabled: false,
            keys: [Input::Unbound; SLOTS],
            texts: Default::default(),
            down: [false; SLOTS],
            armed: false,
        }
    }
}

/// What a macro text sends: the chat input's character filter, then `GuiChat`'s trim.
fn macro_message(text: &str) -> String {
    java_trim(&filter_allowed_characters(text)).to_owned()
}

impl Macros {
    fn sync(&mut self, settings: &ModSettings) {
        self.enabled = settings.enabled(MACROS);
        for (slot, (key, text)) in SLOT_IDS.iter().enumerate() {
            self.keys[slot] = Input::from_lwjgl(settings.key(MACROS, key));
            let message = macro_message(settings.text(MACROS, text));
            if self.texts[slot] != message {
                self.texts[slot] = message;
            }
        }
    }

    /// One sample of the slot keys. Calls `fire(slot)` for each press edge while `in_game`.
    fn sample(
        &mut self,
        is_down: impl Fn(Input) -> bool,
        in_game: bool,
        mut fire: impl FnMut(usize),
    ) {
        if !self.enabled {
            self.armed = false;
            return;
        }
        let armed = std::mem::replace(&mut self.armed, true);
        let slots = self.keys.iter().zip(&mut self.down).zip(&self.texts);
        for (slot, ((&key, was_down), text)) in slots.enumerate() {
            let down = key != Input::Unbound && is_down(key);
            let pressed = armed && down && !*was_down;
            *was_down = down;
            if pressed && in_game && !text.is_empty() {
                fire(slot);
            }
        }
    }
}

impl Client {
    /// Re-read the chat options when the settings changed, and apply the history length to the
    /// log. A `u64` compare when nothing changed.
    pub(crate) fn x2_sync(&mut self) {
        if self.x2.sync(&self.mods.settings) {
            self.chat.set_max_lines(self.x2.history_lines);
        }
    }

    /// java: `displayGuiScreen(new GuiMainMenu())` runs
    /// `GuiNewChat.clearChatMessages` (`Minecraft.java:1002-1006`): the lines and the sent
    /// history. With "Keep chat between servers" on (Aero's default) chat survives instead, for
    /// the whole client run. Off is vanilla.
    pub(crate) fn x2_title_screen_chat(&mut self) {
        self.x2_sync();
        if !self.x2.keep_chat {
            self.chat.clear();
        }
    }

    /// The Chat Macros keys, sampled once per frame from the live input.
    pub(crate) fn x2_shell_frame(&mut self, input: &InputState) {
        self.x2_sync();
        if !self.x2.macros.enabled && !self.x2.macros.armed {
            return;
        }
        let in_game =
            matches!(self.ui, Ui::InGame) && !self.debug_f3_held && self.session.is_some();
        let mut fired = [false; SLOTS];
        self.x2
            .macros
            .sample(|key| input.is_down(key), in_game, |slot| fired[slot] = true);
        for (slot, fired) in fired.into_iter().enumerate() {
            if fired {
                self.x2_send_macro(slot);
            }
        }
    }

    /// Send macro `slot`'s text as if it had been typed into chat and sent with Enter.
    fn x2_send_macro(&mut self, slot: usize) {
        let message = self.x2.macros.texts[slot].clone();
        if message.is_empty() {
            return;
        }
        // java: `GuiChat.keyTyped` -> `GuiScreen.sendChatMessage(s, true)` records it first.
        self.chat.add_sent(&message);
        self.x2_send_typed_chat(&message);
    }

    /// One C01 for chat the player typed (the chat screen's Enter, the multiplayer sleep
    /// screen's, a macro): cut at 256 UTF-16 units (always).
    pub(crate) fn x2_send_typed_chat(&mut self, message: &str) {
        if let Some(session) = &mut self.session {
            session.send_chat_with_limit(message, LONG_CHAT_LENGTH);
        }
    }

    /// The chat input's cap for a chat or multiplayer sleep screen opening now: 256 (always; vanilla's is 100).
    pub(crate) fn x2_chat_input_limit(&mut self) -> i32 {
        LONG_INPUT_LENGTH
    }

    /// The client's own chat links, which no server component can carry
    /// (the parser drops `open_file`, and `copy_image` has no JSON name at all). A left click on
    /// the F2 line's file name opens the screenshot; on `[COPY]` it copies the image on the
    /// encoder thread. Returns `true` when the click was one of these (it is then consumed, as
    /// vanilla's `handleComponentClick` consumes a click event). Nothing is sent to the server.
    /// Shift-clicks keep vanilla's meaning (insertion), so they are left to the chat screen.
    pub(crate) fn x2_chat_click(&mut self, style: Option<&ResolvedStyle>, shift: bool) -> bool {
        if shift {
            return false;
        }
        let Some(click) = style.and_then(|style| style.click_event.as_ref()) else {
            return false;
        };
        match click.action {
            ClickAction::OpenFile => {
                let path = std::path::PathBuf::from(&click.value);
                self.x2_open_screenshot(&path);
                true
            }
            ClickAction::CopyImage => {
                let path = std::path::PathBuf::from(&click.value);
                self.x2_copy_screenshot(path);
                true
            }
            _ => false,
        }
    }

    /// Whether [`Client::x2_chat_click`] would take a click on `style`, without acting (the
    /// 1.7.10 screen phase keeps these clicks in the frame, `app/send_sites_v5.rs`).
    pub(crate) fn x2_takes_chat_click(style: Option<&ResolvedStyle>, shift: bool) -> bool {
        !shift
            && style
                .and_then(|style| style.click_event.as_ref())
                .is_some_and(|click| {
                    matches!(click.action, ClickAction::OpenFile | ClickAction::CopyImage)
                })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::net::{ToGame, ToNet};
    use crate::session::{JoinSettings, Session};
    use mc_protocol::play::C2SPlay;
    use tokio::sync::mpsc::UnboundedReceiver;

    fn macros_with(keys: [i32; SLOTS], texts: [&str; SLOTS]) -> Macros {
        let mut settings = ModSettings::new();
        settings.set_enabled(MACROS, true);
        for (slot, (key, text)) in SLOT_IDS.iter().enumerate() {
            settings.set_value(MACROS, key, mc_mods::OptionValue::Key(keys[slot]));
            settings.set_value(
                MACROS,
                text,
                mc_mods::OptionValue::Text(texts[slot].to_owned().into()),
            );
        }
        let mut macros = Macros::default();
        macros.sync(&settings);
        macros
    }

    #[test]
    fn macro_text_is_filtered_and_trimmed_like_the_chat_input() {
        assert_eq!(macro_message("  /play solo_insane  "), "/play solo_insane");
        // `§` and control characters cannot be typed into chat; vanilla servers kick for them.
        assert_eq!(macro_message("\u{a7}cgg\u{7}"), "cgg");
        assert_eq!(macro_message("   "), "");
        assert_eq!(
            macro_message("gg\u{3000}"),
            "gg\u{3000}",
            "Java's trim keeps U+3000"
        );
    }

    /// One send per press: a held key (or the OS repeat, which never releases it between
    /// samples) sends once; a key already down when the mod is sampled first sends nothing;
    /// nothing fires while a screen is open, and a press made in a screen does not fire when the
    /// screen closes.
    #[test]
    fn macros_fire_once_per_press_and_only_in_game() {
        // G = LWJGL 34, H = 35.
        let mut m = macros_with([34, 35, 0, 0, 0, 0], ["gg", "", "x", "", "", ""]);
        let g = Input::from_lwjgl(34);
        let h = Input::from_lwjgl(35);
        let mut fired = Vec::new();
        {
            let mut sample = |m: &mut Macros, held: &[Input], in_game: bool| {
                let held = held.to_vec();
                m.sample(|k| held.contains(&k), in_game, |slot| fired.push(slot));
            };
            sample(&mut m, &[g], true); // already down on the first sample: not a press
            sample(&mut m, &[g], true);
            sample(&mut m, &[], true);
            sample(&mut m, &[g], true); // press
            sample(&mut m, &[g], true); // held / repeat
            sample(&mut m, &[g], true);
            sample(&mut m, &[], false);
            sample(&mut m, &[g], false); // pressed in a screen
            sample(&mut m, &[g], true); // the screen closed while held: no new press
            sample(&mut m, &[h], true); // slot 2 has no text: idle
        }
        assert_eq!(fired, vec![0]);
    }

    #[test]
    fn a_disabled_mod_never_fires_and_rearms_when_turned_back_on() {
        let mut m = macros_with([34, 0, 0, 0, 0, 0], ["gg", "", "", "", "", ""]);
        let g = Input::from_lwjgl(34);
        let mut fired = 0;
        m.enabled = false;
        m.sample(|k| k == g, true, |_| fired += 1);
        m.sample(|_| false, true, |_| fired += 1);
        m.sample(|k| k == g, true, |_| fired += 1);
        assert_eq!(fired, 0);
        m.enabled = true;
        m.sample(|k| k == g, true, |_| fired += 1); // held when turned on: not a press
        m.sample(|_| false, true, |_| fired += 1);
        m.sample(|k| k == g, true, |_| fired += 1);
        assert_eq!(fired, 1);
    }

    // ---- the client path: packets ---------------------------------------------------------

    type Fixture = (Client, tempfile::TempDir, UnboundedReceiver<ToNet>);

    /// A client on the real assets with a live session (`None` = skip without the vanilla jar).
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

    fn chat_packets(packets: &[C2SPlay]) -> Vec<String> {
        packets
            .iter()
            .filter_map(|packet| match packet {
                C2SPlay::ChatMessage(chat) => Some(chat.message.clone()),
                _ => None,
            })
            .collect()
    }

    /// A macro press sends exactly one C01 with the filtered, trimmed text, records it in the
    /// sent history like Enter, and is cut at the chat limit, 256 (always). Nothing else is sent.
    #[test]
    fn a_macro_press_sends_one_c01_through_the_chat_path() {
        let Some((mut client, _dir, mut to_net)) = client_in_game() else {
            return;
        };
        let long = format!("/msg Dev {}", "a".repeat(300));
        client.mods.settings.set_enabled(MACROS, true);
        client
            .mods
            .settings
            .set_value(MACROS, "key1", mc_mods::OptionValue::Key(34));
        client.mods.settings.set_value(
            MACROS,
            "text1",
            mc_mods::OptionValue::Text(" /play solo_insane ".to_owned().into()),
        );
        client
            .mods
            .settings
            .set_value(MACROS, "key2", mc_mods::OptionValue::Key(35));
        client.mods.settings.set_value(
            MACROS,
            "text2",
            mc_mods::OptionValue::Text(long.clone().into()),
        );
        client.x2.sync(&client.mods.settings);
        let fire = |client: &mut Client, slot: usize| {
            client.x2_send_macro(slot);
        };
        fire(&mut client, 0);
        let packets = sent(&mut to_net);
        assert_eq!(packets.len(), 1, "one packet: {packets:?}");
        assert_eq!(chat_packets(&packets), ["/play solo_insane"]);
        assert_eq!(
            client.chat.sent_messages().last().map(String::as_str),
            Some("/play solo_insane")
        );

        // The text option holds 256 characters; the cut is the chat limit.
        fire(&mut client, 1);
        let cut = chat_packets(&sent(&mut to_net));
        assert_eq!(cut.len(), 1);
        assert_eq!(cut[0].encode_utf16().count(), 256);
        assert!(long.starts_with(&cut[0]));
    }

    /// The shell-frame hook end to end on a real `InputState`: one C01 per press in game; the
    /// OS key repeat and holding send nothing more; a press while chat is open sends nothing,
    /// nor does the same hold after the screen closes; F3+key sends nothing.
    #[test]
    fn the_shell_frame_hook_sends_once_per_press_in_game_only() {
        let Some((mut client, _dir, mut to_net)) = client_in_game() else {
            return;
        };
        client.mods.settings.set_enabled(MACROS, true);
        client
            .mods
            .settings
            .set_value(MACROS, "key1", mc_mods::OptionValue::Key(34));
        client.mods.settings.set_value(
            MACROS,
            "text1",
            mc_mods::OptionValue::Text("gg".to_owned().into()),
        );
        let g = winit::keyboard::KeyCode::KeyG;
        assert_eq!(Input::from_lwjgl(34), Input::Key(g), "LWJGL 34 is G");
        let mut input = InputState::new();
        client.x2_shell_frame(&input); // arms with nothing held
        input.on_key(g, true, false);
        client.x2_shell_frame(&input); // the press
        input.on_key(g, true, true); // OS repeat
        client.x2_shell_frame(&input);
        client.x2_shell_frame(&input); // still held
        assert_eq!(chat_packets(&sent(&mut to_net)), ["gg"]);

        input.on_key(g, false, false);
        client.x2_shell_frame(&input);
        client.open_chat("");
        input.on_key(g, true, false);
        client.x2_shell_frame(&input); // pressed while chat is open
        client.ui = Ui::InGame; // chat closes while G is still held
        client.x2_shell_frame(&input);
        assert!(chat_packets(&sent(&mut to_net)).is_empty());

        input.on_key(g, false, false);
        client.debug_f3_held = true;
        client.x2_shell_frame(&input);
        input.on_key(g, true, false);
        client.x2_shell_frame(&input); // F3+G
        assert!(chat_packets(&sent(&mut to_net)).is_empty());

        client.debug_f3_held = false;
        input.on_key(g, false, false);
        client.x2_shell_frame(&input);
        input.on_key(g, true, false);
        client.x2_shell_frame(&input);
        assert_eq!(chat_packets(&sent(&mut to_net)), ["gg"]);
    }

    /// 256-character chat is always on, end to end on the client.
    /// The chat screen's input takes 256 and Enter's `Action::SendChat` sends all of it (even a
    /// longer Action is cut at 256), with the Chat mod off and on, and with a settings file from
    /// before this change that had the old option switched off.
    #[test]
    fn chat_is_always_256_characters_even_with_an_old_long_chat_off_file() {
        let Some((mut client, _dir, mut to_net)) = client_in_game() else {
            return;
        };
        let (old, report) = ModSettings::from_json_str(
            r#"{"mods":{"chat":{"enabled":false,"options":{"long_chat":false}}}}"#,
        );
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        let text = "z".repeat(300);
        for (settings, chat_on) in [(old.clone(), false), (old, true)] {
            client.mods.settings = settings;
            client.mods.settings.set_enabled(CHAT, chat_on);
            client.open_chat("");
            let Ui::Chat(screen) = &mut client.ui else {
                panic!("chat did not open");
            };
            screen.input.write_text(&text, &client.font);
            let typed = screen.input.text();
            assert_eq!(typed.len(), 256, "the input's cap (Chat on: {chat_on})");
            client.handle_action(mc_ui::screens::Action::SendChat(typed));
            assert_eq!(chat_packets(&sent(&mut to_net))[0].len(), 256);
            client.handle_action(mc_ui::screens::Action::SendChat(text.clone()));
            assert_eq!(
                chat_packets(&sent(&mut to_net))[0].len(),
                256,
                "a longer Action is cut at 256"
            );
        }
    }

    // ---- Chat history --------------------------------

    fn say(client: &mut Client, text: &str) {
        client.x2_chat_line(mc_ui::font::component::ChatComponent::text(text));
    }

    fn texts(client: &Client) -> Vec<String> {
        client
            .chat
            .lines()
            .iter()
            .map(|line| mc_ui::font::strip_formatting(&line.text).trim().to_owned())
            .collect()
    }

    /// Keep chat between servers (on by default, also with the Chat mod off): the title screen
    /// keeps the lines and the sent history, through both of its routes. Off, it is vanilla's
    /// `GuiMainMenu` clear. F3+D still clears either way.
    #[test]
    fn chat_survives_the_title_screen_by_default_and_off_is_vanilla() {
        let Some((mut client, _dir, _to_net)) = client_in_game() else {
            return;
        };
        assert!(
            !client.mods.settings.enabled(CHAT),
            "the Chat mod itself is off"
        );
        assert!(
            client.mods.settings.bool(CHAT, KEEP_CHAT),
            "Keep is on by default"
        );
        say(&mut client, "from the old server");
        client.chat.add_sent("/play solo_insane");
        client.show_main_menu();
        assert!(matches!(client.ui, Ui::MainMenu(_)));
        assert_eq!(texts(&client), ["from the old server"]);
        assert_eq!(client.chat.sent_messages(), ["/play solo_insane"]);
        client.ui = Ui::InGame;
        client.show_screen(Some(Ui::MainMenu(Box::new(
            mc_ui::screens::menu::MainMenuScreen::new(String::new()),
        ))));
        assert_eq!(texts(&client), ["from the old server"], "the other route");

        // Off: vanilla clears lines and history on the title screen.
        client.mods.settings.set_bool(CHAT, KEEP_CHAT, false);
        client.show_main_menu();
        assert!(client.chat.lines().is_empty());
        assert!(client.chat.sent_messages().is_empty());
    }

    /// History length reaches the log on the next sync (every frame): 1000 lines by default,
    /// 5000 at most, 100 is vanilla's cap.
    #[test]
    fn the_history_length_reaches_the_chat_log() {
        let Some((mut client, _dir, _to_net)) = client_in_game() else {
            return;
        };
        client.x2_sync();
        assert_eq!(client.chat.max_lines(), 1000);
        for i in 0..1200 {
            say(&mut client, &format!("line {i}"));
        }
        assert_eq!(client.chat.received_count(), 1000);
        client
            .mods
            .settings
            .set_value(CHAT, HISTORY_LENGTH, mc_mods::OptionValue::Float(5000.0));
        client.x2_shell_frame(&InputState::new());
        assert_eq!(client.chat.max_lines(), 5000);
        client
            .mods
            .settings
            .set_value(CHAT, HISTORY_LENGTH, mc_mods::OptionValue::Float(100.0));
        client.x2_sync();
        assert_eq!(client.chat.max_lines(), mc_ui::screens::chat::MAX_LINES);
        assert_eq!(client.chat.received_count(), 100);
        assert_eq!(texts(&client)[0], "line 1199", "the newest stay");
    }

    fn join_game() -> crate::sync::ClientEvent {
        crate::sync::ClientEvent::JoinedGame(Box::new(mc_protocol::play::s2c::JoinGame {
            entity_id: 1,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 2,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        }))
    }

    /// A JoinGame adds nothing to chat (no line naming the server), with Keep chat on (the
    /// default) or off: the lines kept from before stay exactly as they were. The end-to-end
    /// check over a real loopback connection is `tests/leak.rs` (chat stays empty across
    /// reconnects).
    #[test]
    fn a_join_adds_no_line_to_chat() {
        let Some((mut client, _dir, _to_net)) = client_in_game() else {
            return;
        };
        say(&mut client, "lobby chat");
        for keep in [true, false] {
            client.mods.settings.set_bool(CHAT, KEEP_CHAT, keep);
            client.react(&join_game());
            client.react(&join_game());
            assert_eq!(texts(&client), ["lobby chat"], "Keep chat {keep}");
            assert_eq!(client.chat.received_count(), 1, "Keep chat {keep}");
        }
    }

    /// A server's `run_command` click keeps vanilla's cut, although typed chat takes 256.
    #[test]
    fn a_server_run_command_keeps_the_vanilla_cut() {
        let Some((mut client, _dir, mut to_net)) = client_in_game() else {
            return;
        };
        client.handle_action(mc_ui::screens::Action::RunCommand("q".repeat(200)));
        assert_eq!(chat_packets(&sent(&mut to_net))[0].len(), 100);
    }

    // ---- F2 end to end --------------------------------------------------------------------

    /// The real hook path on a headless frame: F2 arms `screenshot_requested`, `after_render`
    /// only starts the readback (nothing is written yet), later frames (`x2_frame`) hand it to
    /// the encoder thread, and the chat then shows vanilla's line. Two F2s: two files with
    /// vanilla's names (`yyyy-MM-dd_HH.mm.ss.png`, then `_2` within the same second).
    #[test]
    fn f2_saves_off_the_frame_with_vanilla_names_and_a_chat_line() {
        let Some((mut client, _dir, mut to_net)) = client_in_game() else {
            return;
        };
        let mut gpu = match mc_render::Gpu::headless(
            64,
            48,
            mc_render::GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("SKIP: no headless GPU ({error})");
                return;
            }
        };
        let directory = client.config.game_dir.join("screenshots");
        for _ in 0..2 {
            let frame = gpu.acquire().expect("headless frame");
            client.screenshot_requested = true;
            client.x2_screenshot_after_render(&gpu, &frame);
            assert!(!client.screenshot_requested);
        }
        assert!(
            !directory.exists() || std::fs::read_dir(&directory).unwrap().next().is_none(),
            "nothing is written on the F2 frame itself"
        );
        let started = std::time::Instant::now();
        while !client.x2.shots.idle() {
            assert!(
                started.elapsed().as_secs() < 30,
                "the screenshots never landed"
            );
            std::thread::sleep(std::time::Duration::from_millis(2));
            client.x2_frame(&gpu);
        }
        let mut names: Vec<String> = std::fs::read_dir(&directory)
            .expect("screenshots/")
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(names.len(), 2, "{names:?}");
        let is_vanilla = |name: &str| {
            let stem = name.strip_suffix(".png").expect("png");
            let stem = stem.strip_suffix("_2").unwrap_or(stem);
            let b = stem.as_bytes();
            stem.len() == 19
                && [4, 7].iter().all(|&i| b[i] == b'-')
                && b[10] == b'_'
                && [13, 16].iter().all(|&i| b[i] == b'.')
                && stem.bytes().filter(u8::is_ascii_digit).count() == 14
        };
        assert!(names.iter().all(|name| is_vanilla(name)), "{names:?}");
        if names[0][..19] == names[1][..19] {
            assert_eq!(names[1], format!("{}_2.png", &names[0][..19]));
        }
        let image = image::open(directory.join(&names[0])).expect("decode");
        assert_eq!((image.width(), image.height()), (64, 48));
        assert_eq!(image.color(), image::ColorType::Rgb8);
        // Whatever the chat width wraps, both messages are in the log with their [COPY].
        let log: String = client
            .chat
            .lines()
            .iter()
            .map(|line| line.text.as_str())
            .collect();
        assert_eq!(log.matches("Saved screenshot as").count(), 2, "{log:?}");
        assert_eq!(log.matches("[COPY]").count(), 2, "{log:?}");
        assert!(sent(&mut to_net).is_empty(), "F2 sends nothing");
    }

    // ---- the F2 line's links --------------------------------------------------------------

    /// A small screenshot in the client's own `screenshots` folder, as the encoder writes it.
    fn screenshot_in(client: &Client) -> std::path::PathBuf {
        let pixels: Vec<u8> = (0..4 * 3)
            .flat_map(|i| [i * 20, 255 - i * 20, 90, 255])
            .collect();
        let path = super::super::super::save_screenshot_png(
            &client.config.game_dir.join("screenshots"),
            "2026-11-09_02.00.00",
            4,
            3,
            &pixels,
        )
        .expect("save");
        std::path::absolute(&path).expect("absolute")
    }

    /// Only a PNG directly inside this game directory's `screenshots` folder is ever opened or
    /// copied by a chat link.
    #[test]
    fn chat_links_only_reach_pngs_in_the_screenshots_folder() {
        let Some((client, dir, _to_net)) = client_in_game() else {
            return;
        };
        let shot = screenshot_in(&client);
        assert!(client.x2_screenshot_file(&shot).is_some());
        let outside = dir.path().join("2026-11-09_02.00.00.png");
        std::fs::copy(&shot, &outside).expect("copy");
        assert!(
            client.x2_screenshot_file(&outside).is_none(),
            "not in screenshots/"
        );
        let text = shot.with_extension("txt");
        std::fs::write(&text, "x").expect("write");
        assert!(client.x2_screenshot_file(&text).is_none(), "not a PNG");
        let missing = shot.with_file_name("missing.png");
        assert!(client.x2_screenshot_file(&missing).is_none(), "gone");
        let sneaky = client
            .config
            .game_dir
            .join("screenshots")
            .join("..")
            .join("2026-11-09_02.00.00.png");
        assert!(
            client.x2_screenshot_file(&sneaky).is_none(),
            "`..` resolves outside"
        );
    }

    /// `[COPY]` from the chat screen, end to end through `mouse_clicked`: the F2 line is in the
    /// log, chat is open, a left click lands on `[COPY]` (found by hit-testing the real chat
    /// layout). Nothing is sent to the server; the click returns with the copy queued on the
    /// encoder thread; a later frame posts the encoder's answer to the chat. The file here is
    /// not a decodable PNG, so the job stops before the OS clipboard (the clipboard write itself
    /// is `x2_screenshot::tests::copy_puts_a_dib_and_a_png_on_the_windows_clipboard`) and this
    /// test cannot race the clipboard tests.
    #[test]
    fn copy_from_the_chat_screen_sends_nothing_and_runs_off_the_frame() {
        use super::super::x2_screenshot::{screenshot_message, COPY_LABEL};
        use mc_ui::screen_stack::ScreenCtx;
        use mc_ui::screens::chat::ChatLayout;
        use mc_ui::widget::ScaledResolution;

        let Some((mut client, _dir, mut to_net)) = client_in_game() else {
            return;
        };
        let shot = screenshot_in(&client);
        std::fs::write(&shot, b"not a png").expect("overwrite");
        client.x2_chat_line(screenshot_message(&Ok(shot.clone())));
        client.open_chat("");
        let ctx = ScreenCtx::new(ScaledResolution::new(854, 480, 2, false), 0.0);
        let layout = ChatLayout::from_settings(&client.settings);
        let y = ctx.height - 28 - 4; // the newest line, chat open
        let x = (0..ctx.width)
            .find(|&x| {
                client
                    .chat
                    .style_at(x, y, &ctx, &layout, true, &client.font)
                    .and_then(|style| style.click_event.as_ref())
                    .is_some_and(|click| click.action == ClickAction::CopyImage)
            })
            .unwrap_or_else(|| panic!("no {COPY_LABEL} span on the newest chat line"));

        client.cursor = (x, y);
        let click = std::time::Instant::now();
        client.mouse_clicked(0, ctx);
        let click_ms = click.elapsed().as_secs_f64() * 1000.0;
        assert!(sent(&mut to_net).is_empty(), "[COPY] sends nothing");
        assert_eq!(
            client.x2.shots.in_flight(),
            1,
            "the copy runs on the encoder thread"
        );
        assert!(matches!(client.ui, Ui::Chat(_)), "chat stays open");

        let started = std::time::Instant::now();
        while client.x2.shots.in_flight() > 0 {
            assert!(started.elapsed().as_secs() < 20, "the copy never finished");
            std::thread::sleep(std::time::Duration::from_millis(5));
            client.x2_poll_results();
        }
        // The answer may wrap onto two display lines (newest first).
        let newest: String = client
            .chat
            .lines()
            .range(..2)
            .rev()
            .map(|line| line.text.as_str())
            .collect();
        assert!(
            newest.contains("Couldn't copy the screenshot"),
            "the encoder's answer reaches the chat: {newest:?}"
        );
        assert!(sent(&mut to_net).is_empty());
        eprintln!("[COPY] click handled in {click_ms:.2} ms (provisional); chat: {newest:?}");
    }

    /// A 1.7.10 session's screen pass asks [`Client::x2_takes_chat_click`] whether a chat click
    /// is one of the client's own links before it keeps the click in the frame
    /// (`app/send_sites_v5.rs`), so the copy must answer what [`Client::x2_chat_click`] then
    /// does: for no style, a style without a click and every click action, each with and
    /// without shift. The links name a file that does not exist, so nothing opens or copies.
    #[test]
    fn x2_takes_chat_click_answers_what_x2_chat_click_does() {
        use mc_ui::font::component::ClickEvent;

        let Some((mut client, dir, mut to_net)) = client_in_game() else {
            return;
        };
        let missing = dir.path().join("screenshots").join("missing.png");
        let mut styles = vec![None, Some(ResolvedStyle::default())];
        for action in [
            ClickAction::OpenUrl,
            ClickAction::OpenFile,
            ClickAction::RunCommand,
            ClickAction::TwitchUserInfo,
            ClickAction::SuggestCommand,
            ClickAction::ChangePage,
            ClickAction::CopyImage,
        ] {
            styles.push(Some(ResolvedStyle {
                click_event: Some(ClickEvent {
                    action,
                    value: missing.display().to_string(),
                }),
                ..ResolvedStyle::default()
            }));
        }
        let mut taken = 0;
        for style in &styles {
            for shift in [false, true] {
                let takes = Client::x2_takes_chat_click(style.as_ref(), shift);
                let took = client.x2_chat_click(style.as_ref(), shift);
                assert_eq!(takes, took, "{style:?}, shift {shift}");
                taken += usize::from(took);
            }
        }
        // Not vacuous: the two client-only actions are taken, without shift only.
        assert_eq!(taken, 2);
        assert!(sent(&mut to_net).is_empty(), "a chat link sends nothing");
    }
}
