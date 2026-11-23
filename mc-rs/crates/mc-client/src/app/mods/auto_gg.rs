//! Opt-in, delayed match-end chat. No worker, blocking wait, or network path of its own.
//!
//! Exact result titles work on any server. One minigame network's centered result summaries
//! (recognized by its host name) also work in chat; ordinary player chat, translations, action
//! bars and subtitles cannot trigger a send.
//! This deliberately recognizes a small English result vocabulary, not arbitrary server text.

use std::time::{Duration, Instant};

use mc_mods::ModSettings;
use mc_protocol::play::c2s::ChatVisibility;
use mc_ui::font::component::{ChatComponent, ComponentKind};
use mc_ui::screens::chat::java_trim;
use mc_ui::widget::keys::filter_allowed_characters;

use super::super::{chat_visibility_of, Client};
use crate::session::Phase;
use crate::sync::ChatPosition;

const ID: &str = "auto_gg";
/// One end-of-match burst often contains both a title and several result rows.
const COOLDOWN: Duration = Duration::from_secs(15);

#[derive(Debug, Default)]
pub(crate) struct AutoGgState {
    seen: u64,
    enabled: bool,
    message: String,
    delay: Duration,
    /// The selected host sends centered result summaries ([`is_summary_host`]).
    summary_host: bool,
    pending: Option<Instant>,
    last_trigger: Option<Instant>,
}

impl AutoGgState {
    fn sync(&mut self, settings: &ModSettings) {
        if self.seen == settings.generation() {
            return;
        }
        self.seen = settings.generation();
        let enabled = settings.enabled(ID);
        let message =
            java_trim(&filter_allowed_characters(settings.text(ID, "message"))).to_owned();
        let delay = Duration::from_millis(settings.f32(ID, "delay_ms").clamp(0.0, 3000.0) as u64);
        // Editing the message/delay must not send an earlier trigger with new settings.
        // Unrelated HUD changes must not discard a valid pending reply.
        if self.enabled != enabled || self.message != message || self.delay != delay {
            self.pending = None;
        }
        self.enabled = enabled;
        self.message = message;
        self.delay = delay;
    }

    fn trigger(&mut self, now: Instant) {
        if self.pending.is_some()
            || self
                .last_trigger
                .is_some_and(|last| now.saturating_duration_since(last) < COOLDOWN)
        {
            return;
        }
        self.pending = Some(now + self.delay);
        self.last_trigger = Some(now);
    }

    fn reset_connection(&mut self) {
        self.pending = None;
        self.last_trigger = None;
        self.summary_host = false;
    }
}

fn is_summary_host(host: &str) -> bool {
    let host = host.trim_end_matches('.');
    host.eq_ignore_ascii_case("hypixel.net")
        || host
            .get(host.len().saturating_sub(".hypixel.net".len())..)
            .is_some_and(|suffix| suffix.eq_ignore_ascii_case(".hypixel.net"))
}

/// Remove Minecraft formatting without normalizing the leading spaces used to distinguish
/// centered summaries from sender-prefixed chat.
fn without_formatting(text: &str) -> String {
    let mut chars = text.chars();
    let mut plain = String::with_capacity(text.len());
    while let Some(ch) = chars.next() {
        if ch == '\u{00a7}' {
            let _ = chars.next();
        } else {
            plain.push(ch);
        }
    }
    plain
}

fn result_title(text: &str) -> bool {
    let text = text.trim().trim_end_matches(['!', '.']).trim_end();
    [
        "VICTORY",
        "DEFEAT",
        "GAME OVER",
        "YOU WIN",
        "YOU WON",
        "YOU LOSE",
        "YOU LOST",
    ]
    .iter()
    .any(|result| text.eq_ignore_ascii_case(result))
}

/// Translated chat.type.text (including nested nodes) is player speech, even if its body is
/// styled to look exactly like a centered server result. Other nonliteral nodes are unnecessary
/// for the known summaries and are excluded too.
fn literal_component(component: &ChatComponent) -> bool {
    matches!(component.kind, ComponentKind::Text(_))
        && component.extra.iter().all(literal_component)
}

fn summary_result(text: &str, position: ChatPosition) -> bool {
    if !matches!(position, ChatPosition::Chat | ChatPosition::System) || text.contains(['\n', '\r'])
    {
        return false;
    }
    // Position 0 also carries the network's announcements. Require their centered layout; ordinary
    // player lines start with a name/rank/team/channel prefix, never the result label itself.
    if position == ChatPosition::Chat && !text.starts_with("  ") {
        return false;
    }
    let text = text.trim();
    if [
        "Winner: ",
        "Winners: ",
        "Winner - ",
        "Winners - ",
        "Winning Team: ",
        "Winning Team - ",
        "1st Killer - ",
        "1st Place - ",
        "Top Seeker: ",
    ]
    .iter()
    .any(|prefix| {
        text.strip_prefix(prefix)
            .is_some_and(|value| !value.trim().is_empty())
    }) {
        return true;
    }
    matches!(
        text,
        "You survived the Murderer!"
            | "You are the last survivor!"
            | "You are the last surviving hero!"
    )
}

impl Client {
    /// Called before opening any new transport, including a replay transport. Remember the
    /// selected host, not an SRV target/IP, so a normal connection to that network is recognized.
    pub(crate) fn auto_gg_connect(&mut self, host: &str) {
        self.auto_gg.reset_connection();
        let selected = self
            .current_server_address
            .as_deref()
            .and_then(|address| crate::net::ping::parse_server_address(address).ok());
        self.auto_gg.summary_host = is_summary_host(
            selected
                .as_ref()
                .map_or(host, |address| address.host.as_str()),
        );
    }

    /// World/respawn boundaries cancel unsent replies, retaining the connection's cooldown so
    /// a late result row cannot repeat a reply just sent before the boundary.
    pub(crate) fn auto_gg_cancel(&mut self) {
        self.auto_gg.pending = None;
    }

    pub(crate) fn auto_gg_disconnect(&mut self) {
        self.auto_gg.reset_connection();
    }

    fn auto_gg_can_send(&self) -> bool {
        self.auto_gg.enabled
            && !self.auto_gg.message.is_empty()
            && chat_visibility_of(self.settings.chat_visibility) == ChatVisibility::Full
            && self.config.replay.is_none()
            && self.session.as_ref().is_some_and(|session| {
                *session.phase() == Phase::Playing && session.replay_source().is_none()
            })
    }

    pub(crate) fn auto_gg_chat(&mut self, component: &ChatComponent, position: ChatPosition) {
        self.auto_gg.sync(&self.mods.settings);
        if !self.auto_gg_can_send()
            || !self.auto_gg.summary_host
            || !matches!(position, ChatPosition::Chat | ChatPosition::System)
            || !literal_component(component)
        {
            return;
        }
        let text = without_formatting(&component.unformatted_text(&self.assets.language));
        if summary_result(&text, position) {
            self.auto_gg.trigger(Instant::now());
        }
    }

    /// Only S45 SetTitle calls this, never subtitles/action bars/player chat.
    pub(crate) fn auto_gg_title(&mut self, json: &str) {
        self.auto_gg.sync(&self.mods.settings);
        if !self.auto_gg_can_send() {
            return;
        }
        let Ok(component) = ChatComponent::from_json(json) else {
            return;
        };
        if literal_component(&component)
            && result_title(&without_formatting(
                &component.unformatted_text(&self.assets.language),
            ))
        {
            self.auto_gg.trigger(Instant::now());
        }
    }

    /// Off/idle cost is one settings generation comparison and an Option test. The clock is
    /// read only while a reply is pending; no timer, worker or sleep affects frame/input timing.
    pub(crate) fn auto_gg_frame(&mut self) {
        self.auto_gg.sync(&self.mods.settings);
        if self.auto_gg.pending.is_some() {
            self.auto_gg_frame_at(Instant::now());
        }
    }

    fn auto_gg_frame_at(&mut self, now: Instant) {
        self.auto_gg.sync(&self.mods.settings);
        if !self.auto_gg_can_send() {
            self.auto_gg.pending = None;
            return;
        }
        if self.auto_gg.pending.is_some_and(|deadline| now >= deadline) {
            self.auto_gg.pending = None;
            let message = self.auto_gg.message.clone();
            // This is also Enter's path, including the current 100/256 character wire cap.
            // Server text never supplies the outbound message or a command.
            self.x2_send_typed_chat(&message);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::driver::LoginMode;
    use crate::app::ClientConfig;
    use crate::net::{NetHandle, ToGame, ToNet};
    use crate::session::{JoinSettings, Session};
    use crate::sync::ClientEvent;
    use mc_mods::OptionValue;
    use mc_protocol::play::{s2c, C2SPlay, S2CPlay};
    use tokio::sync::mpsc::UnboundedReceiver;

    fn join() -> s2c::JoinGame {
        s2c::JoinGame {
            entity_id: 1,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 2,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        }
    }

    type Fixture = (Client, tempfile::TempDir, UnboundedReceiver<ToNet>);

    fn client_in_game() -> Option<Fixture> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".into()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config
            .version_jar
            .as_ref()
            .is_some_and(|path| path.is_file())
        {
            eprintln!("SKIP: no vanilla jar in this checkout");
            return None;
        }
        let mut client = Client::new(config, &mut |_, _| {}).expect("real assets must load");
        let (handle, to_game, mut to_net) = NetHandle::new_pair();
        let mut session = Session::new(handle, JoinSettings::default());
        to_game
            .try_send(ToGame::Packet(S2CPlay::JoinGame(join())))
            .unwrap();
        session.poll(16);
        client.session = Some(session);
        client.ui = super::super::super::Ui::InGame;
        client.auto_gg_connect("mc.hypixel.net");
        let _ = sent(&mut to_net);
        Some((client, dir, to_net))
    }

    fn sent(to_net: &mut UnboundedReceiver<ToNet>) -> Vec<C2SPlay> {
        let mut packets = Vec::new();
        while let Ok(message) = to_net.try_recv() {
            if let ToNet::Send(packet) = message {
                packets.push(packet);
            }
        }
        packets
    }

    fn expect_chat(to_net: &mut UnboundedReceiver<ToNet>, message: &str) {
        let packets = sent(to_net);
        assert_eq!(packets.len(), 1, "exactly one packet: {packets:?}");
        let C2SPlay::ChatMessage(chat) = &packets[0] else {
            panic!("expected C01, got {packets:?}");
        };
        assert_eq!(chat.message, message);
    }

    fn title(text: &str) -> ClientEvent {
        ClientEvent::Title(Box::new(s2c::Title::SetTitle {
            text: serde_json::to_string(text).unwrap(),
        }))
    }

    fn chat(text: &str, position: ChatPosition) -> ClientEvent {
        ClientEvent::Chat {
            component: Box::new(ChatComponent::text(text)),
            position,
        }
    }

    fn pending(client: &Client) -> Instant {
        client.auto_gg.pending.expect("match end scheduled a reply")
    }

    #[test]
    fn host_matching_has_dns_label_boundaries() {
        for host in ["hypixel.net", "mc.hypixel.net", "MC.HYPIXEL.NET."] {
            assert!(is_summary_host(host), "{host}");
        }
        for host in [
            "hypixel.net.attacker.test",
            "fakehypixel.net",
            "localhost",
            "",
        ] {
            assert!(!is_summary_host(host), "{host}");
        }
    }

    #[test]
    fn result_detection_is_anchored_and_excludes_player_chat_and_actionbars() {
        assert!(summary_result("       Winner: Alex", ChatPosition::Chat));
        assert!(summary_result(
            "1st Killer - Alex - 4",
            ChatPosition::System
        ));
        for text in [
            "<Alex>        Winner: Alex",
            "[MVP+] Alex: Winner: Alex",
            "        [MVP+] Alex: Winner: Alex",
            "Alex: Winner - Alex",
            "Winner: Alex",
            "You won 5 coins!",
            "   Winner: ",
            "   Winner: Alex\nAlex: hello",
        ] {
            assert!(!summary_result(text, ChatPosition::Chat), "{text}");
        }
        assert!(!summary_result("   Winner: Alex", ChatPosition::ActionBar));
        assert!(!summary_result("   Winner: Alex", ChatPosition::Other(7)));
        for text in ["VICTORY!", "Game Over", "DEFEAT", "YOU WIN!"] {
            assert!(result_title(text));
        }
        for text in ["YOU DIED!", "VICTORY in 5 seconds", "<Alex> VICTORY!", ""] {
            assert!(!result_title(text));
        }
        let translated = ChatComponent::from_json(
            r#"{"text":"","extra":[{"translate":"chat.type.text","with":["Alex","       Winner: Alex"]}]}"#,
        ).unwrap();
        assert!(!literal_component(&translated));
        assert_eq!(without_formatting("§r   §a§lVICTORY!§r"), "   VICTORY!");
    }

    #[test]
    fn real_events_send_one_delayed_packet_for_duplicate_result_burst() {
        let Some((mut client, _dir, mut to_net)) = client_in_game() else {
            return;
        };
        client.react(&title("VICTORY!"));
        assert!(client.auto_gg.pending.is_none(), "disabled by default");
        client.mods.settings.set_enabled(ID, true);
        client.react(&chat("§r      §aWinner: Alex", ChatPosition::Chat));
        let deadline = pending(&client);
        client.react(&title("§6§lVICTORY!"));
        assert_eq!(
            pending(&client),
            deadline,
            "duplicates do not extend the delay"
        );
        client.auto_gg_frame_at(deadline - Duration::from_millis(1));
        assert!(sent(&mut to_net).is_empty());
        client.auto_gg_frame_at(deadline);
        expect_chat(&mut to_net, "gg");
        client.react(&chat("      1st Killer - Alex - 4", ChatPosition::Chat));
        client.react(&title("VICTORY!"));
        client.auto_gg_frame_at(deadline + Duration::from_secs(3));
        assert!(sent(&mut to_net).is_empty());
        // The clock is injectable: a later round on the same world may send again.
        client.auto_gg.trigger(deadline + COOLDOWN);
        client.auto_gg_frame_at(deadline + COOLDOWN + Duration::from_millis(500));
        expect_chat(&mut to_net, "gg");
    }

    #[test]
    fn real_events_ignore_player_messages_subtitles_actionbars_and_other_hosts() {
        let Some((mut client, _dir, mut to_net)) = client_in_game() else {
            return;
        };
        client.mods.settings.set_enabled(ID, true);
        for event in [
            chat("<Alex> VICTORY!", ChatPosition::Chat),
            chat("[MVP+] Alex: Winner: Alex", ChatPosition::Chat),
            chat("   Winner: Alex", ChatPosition::ActionBar),
            chat("VICTORY!", ChatPosition::System),
            title("YOU DIED!"),
            ClientEvent::Title(Box::new(s2c::Title::SetSubtitle {
                text: "\"VICTORY!\"".into(),
            })),
            ClientEvent::Chat {
                component: Box::new(
                    ChatComponent::from_json(
                        r#"{"translate":"chat.type.text","with":["Alex","       Winner: Alex"]}"#,
                    )
                    .unwrap(),
                ),
                position: ChatPosition::Chat,
            },
        ] {
            client.react(&event);
            assert!(client.auto_gg.pending.is_none(), "{event:?}");
        }
        client.auto_gg_connect("example.org");
        client.react(&chat("   Winner: Alex", ChatPosition::Chat));
        assert!(client.auto_gg.pending.is_none());
        client.react(&title("GAME OVER!"));
        client.auto_gg_frame_at(pending(&client));
        expect_chat(&mut to_net, "gg");
    }

    #[test]
    fn editable_message_uses_normal_chat_filter_trim_and_wire_cap() {
        let Some((mut client, _dir, mut to_net)) = client_in_game() else {
            return;
        };
        client.mods.settings.set_enabled(ID, true);
        // Typed chat is always cut at 256.
        for (configured, expected) in [
            (
                "  /ac g\ng\u{00a7}\u{007f}  ".to_owned(),
                "/ac gg".to_owned(),
            ),
            ("a".repeat(300), "a".repeat(256)),
        ] {
            client.auto_gg_connect("mc.hypixel.net");
            client
                .mods
                .settings
                .set_value(ID, "message", OptionValue::Text(configured.into()));
            client.react(&title("VICTORY!"));
            client.auto_gg_frame_at(pending(&client));
            expect_chat(&mut to_net, &expected);
        }
        client.auto_gg_connect("mc.hypixel.net");
        client
            .mods
            .settings
            .set_value(ID, "message", OptionValue::Text(" \n\t ".into()));
        client.react(&title("VICTORY!"));
        client.auto_gg_frame_at(Instant::now() + Duration::from_secs(4));
        assert!(sent(&mut to_net).is_empty());
        assert!(client.auto_gg.pending.is_none());
    }

    #[test]
    fn settings_edits_disable_and_restricted_chat_cancel_pending_reply() {
        let Some((mut client, _dir, mut to_net)) = client_in_game() else {
            return;
        };
        for edit in 0..5 {
            client.auto_gg_connect("mc.hypixel.net");
            client.settings.chat_visibility = 0;
            client.mods.settings.set_enabled(ID, true);
            client.react(&title("VICTORY!"));
            let deadline = pending(&client);
            match edit {
                0 => {
                    client.mods.settings.set_enabled(ID, false);
                }
                1 => {
                    client.mods.settings.set_value(
                        ID,
                        "message",
                        OptionValue::Text("well played".into()),
                    );
                }
                2 => {
                    client
                        .mods
                        .settings
                        .set_value(ID, "delay_ms", OptionValue::Float(3000.0));
                }
                3 => client.settings.chat_visibility = 1,
                _ => client.settings.chat_visibility = 2,
            }
            client.auto_gg_frame_at(deadline + Duration::from_secs(5));
            assert!(client.auto_gg.pending.is_none());
            assert!(sent(&mut to_net).is_empty(), "edit {edit}");
        }
        // Neither hidden nor commands-only chat should schedule automatic replies.
        for visibility in [1, 2] {
            client.settings.chat_visibility = visibility;
            client.react(&title("VICTORY!"));
            assert!(client.auto_gg.pending.is_none());
        }
    }

    #[test]
    fn lifecycle_events_and_replay_cancel_pending_reply() {
        let Some((mut client, _dir, mut to_net)) = client_in_game() else {
            return;
        };
        client.mods.settings.set_enabled(ID, true);
        for event in [
            ClientEvent::Respawned {
                dimension: 0,
                changed_world: false,
            },
            ClientEvent::Respawned {
                dimension: -1,
                changed_world: true,
            },
            ClientEvent::JoinedGame(Box::new(join())),
            ClientEvent::Disconnected(crate::net::DisconnectReason::ConnectionLost {
                detail: "test".into(),
            }),
        ] {
            client.auto_gg_connect("mc.hypixel.net");
            client.react(&title("VICTORY!"));
            let deadline = pending(&client);
            client.react(&event);
            client.auto_gg_frame_at(deadline);
            assert!(client.auto_gg.pending.is_none(), "{event:?}");
            assert!(sent(&mut to_net).is_empty(), "{event:?}");
        }
        client.auto_gg_connect("mc.hypixel.net");
        client.react(&title("VICTORY!"));
        let deadline = pending(&client);
        client.config.replay = Some("unused-test.mccap".into());
        client.auto_gg_frame_at(deadline);
        assert!(client.auto_gg.pending.is_none());
        client.react(&title("VICTORY!"));
        assert!(client.auto_gg.pending.is_none());
        assert!(sent(&mut to_net).is_empty());
        client.config.replay = None;
        client.auto_gg_connect("mc.hypixel.net");
        client.react(&title("VICTORY!"));
        let deadline = pending(&client);
        client.auto_gg_connect("another.example");
        client.auto_gg_frame_at(deadline);
        assert!(sent(&mut to_net).is_empty());
        assert!(!client.auto_gg.summary_host);
    }

    #[test]
    fn delay_is_configurable_and_unrelated_settings_preserve_pending_reply() {
        let mut settings = ModSettings::default();
        settings.set_enabled(ID, true);
        let mut state = AutoGgState::default();
        let now = Instant::now();
        for delay in [0, 500, 3000] {
            state.reset_connection();
            settings.set_value(ID, "delay_ms", OptionValue::Float(delay as f32));
            state.sync(&settings);
            state.trigger(now);
            assert_eq!(state.pending, Some(now + Duration::from_millis(delay)));
        }
        settings.set_enabled("fps", true);
        state.sync(&settings);
        assert_eq!(state.pending, Some(now + Duration::from_millis(3000)));
    }
}
