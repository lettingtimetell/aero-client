//! The 1.7.10 half of [`Session`]'s own sends: where a 1.7.10 connection sends a packet the 1.8.9
//! code would build differently, and when the screens' packets go out.
//!
//! Each `*_v5` method here does nothing (and returns `false` or hands its input back) on a 1.8.9
//! connection, so the 1.8.9 call site next to it keeps running exactly as before. The packets
//! themselves are built by [`crate::outbound::v5`].
//!
//! # When the screens' packets go out
//!
//! Vanilla handles a screen's mouse and key events inside the tick, in
//! `currentScreen.handleInput()`: after `updateController` (the tick's C09 and packet handling)
//! and the death and sleep screens, before the tick's game input and the player's movement
//! (java17 `client/Minecraft.java:1595-1628`). This client handles them per frame, after the
//! frame's ticks, so a click's C0E would leave after the movement packet of the tick it belongs
//! to. On a 1.7.10 connection the shell therefore holds every packet its per-frame screen pass
//! sends ([`Session::hold_screen_packets_v5`]) and the next tick releases them at that point
//! ([`Session::release_screen_packets_v5`]), in order, after the tick's packet replies. A frame
//! without a tick holds them until the next tick, as vanilla's queued events wait for it.

use mc_math::BlockPos;
use mc_protocol::play::C2SPlay;
use mc_protocol::v5::{c2s, C2SPlay5};
use mc_protocol::ItemStack;
use mc_ui::screens::sign::CommandBlockTarget;

use super::{JoinSettings, Session};
use crate::outbound::v5 as build;

/// What a [`Session`] keeps for its 1.7.10 sends. Never touched on a 1.8.9 connection.
#[derive(Debug, Clone, Default)]
pub(crate) struct SendsV5 {
    /// Whether the shell's per-frame screen pass is running.
    holding: bool,
    /// The packets that pass sent, in order, waiting for the next tick's screen phase.
    held: Vec<HeldSend>,
    /// The C15 last sent on this connection, in its 1.7.10 form.
    last_client_settings: Option<c2s::ClientSettings>,
    /// The first S01 was applied in this tick's screen phase ([`Session::send_join_packets_v5`]).
    fresh_terrain_screen: bool,
}

/// One held send: the two kinds share the network queue, so their order is kept.
#[derive(Debug, Clone)]
enum HeldSend {
    /// A packet the network task converts from its 1.8.9 form.
    V47(C2SPlay),
    /// A packet already in its 1.7.10 form.
    V5(C2SPlay5),
}

impl Session {
    /// Turn the screen-packet hold on or off around the shell's per-frame screen pass (module
    /// docs). Only a live 1.7.10 connection holds; a replay sends nothing anyway.
    pub(crate) fn hold_screen_packets_v5(&mut self, hold: bool) {
        self.sends_v5.holding = hold && self.version.is_1_7_10() && !self.handle.is_driven();
    }

    /// Send what the screen passes held, in order: the tick's screen phase (module docs). The
    /// same gate as [`Session::send`], applied now: nothing goes out once disconnected.
    pub(crate) fn release_screen_packets_v5(&mut self) {
        if self.sends_v5.held.is_empty() {
            return;
        }
        let mut held = std::mem::take(&mut self.sends_v5.held);
        for send in held.drain(..) {
            if !self.phase.is_live() {
                break;
            }
            let _ = match send {
                HeldSend::V47(packet) => self.handle.send(packet),
                HeldSend::V5(packet) => self.handle.send_v5(packet),
            };
        }
        // Keep the allocation for the next click.
        held.clear();
        self.sends_v5.held = held;
    }

    /// [`Session::send`]'s share of the hold: `None` when `packet` was held.
    pub(super) fn hold_screen_packet(&mut self, packet: C2SPlay) -> Option<C2SPlay> {
        if !self.sends_v5.holding {
            return Some(packet);
        }
        self.sends_v5.held.push(HeldSend::V47(packet));
        None
    }

    /// [`Session::send_v5`]'s share of the hold: `None` when `packet` was held.
    pub(super) fn hold_screen_packet_v5(&mut self, packet: C2SPlay5) -> Option<C2SPlay5> {
        if !self.sends_v5.holding {
            return Some(packet);
        }
        self.sends_v5.held.push(HeldSend::V5(packet));
        None
    }

    /// `handleJoinGame`'s two sends on a 1.7.10 connection: `sendSettingsToServer`'s C15 with
    /// the client's own difficulty and show-cape, then `MC|Brand` as raw UTF-8 (java17
    /// `client/network/NetHandlerPlayClient.java:240-241`). The C15 is remembered, so that an
    /// options save which changes none of its fields sends no repeat.
    ///
    /// `first_world`: this S01 created the world. java17 handles that one inside
    /// `GuiConnecting.updateScreen`, the screen phase itself, so the `GuiDownloadTerrain` it opens
    /// is first updated on the next tick ([`Session::settle_fresh_terrain_screen_v5`]); a later
    /// S01 is handled in `updateController`, before that tick's screen update.
    pub(super) fn send_join_packets_v5(&mut self, first_world: bool) {
        let settings = build::client_settings(&self.settings);
        self.sends_v5.last_client_settings = Some(settings.clone());
        self.send_v5(C2SPlay5::ClientSettings(settings));
        self.send_v5(build::brand(&self.settings.brand));
        self.sends_v5.fresh_terrain_screen = first_world;
    }

    /// [`Session::send_client_settings_if_changed`] on a 1.7.10 connection: the same rule (send
    /// only when a field differs from the last C15 this connection carried) over 1.7.10's six
    /// fields, the difficulty and show-cape included (java17 `GameSettings.sendSettingsToServer`,
    /// `client/settings/GameSettings.java:1163-1169`).
    /// The values are kept for the next S01's resend, which vanilla builds from the live options.
    pub(super) fn send_client_settings_if_changed_v5(&mut self, settings: &JoinSettings) -> bool {
        self.settings.locale.clone_from(&settings.locale);
        self.settings.view_distance = settings.view_distance;
        self.settings.chat_visibility = settings.chat_visibility;
        self.settings.chat_colors = settings.chat_colors;
        self.settings.displayed_skin_parts = settings.displayed_skin_parts;
        self.settings.difficulty = settings.difficulty;
        self.settings.show_cape = settings.show_cape;
        let packet = build::client_settings(settings);
        if self.sends_v5.last_client_settings.as_ref() == Some(&packet) {
            return false;
        }
        self.sends_v5.last_client_settings = Some(packet.clone());
        self.send_v5(C2SPlay5::ClientSettings(packet));
        true
    }

    /// Undo this tick's terrain-screen count when the first S01 opened the screen during this
    /// tick ([`Session::send_join_packets_v5`]): its first `updateScreen`, and so the 20-update
    /// keep-alive cadence, starts on the next tick (java17
    /// `client/gui/GuiDownloadTerrain.java:25-36`). Called right after the count.
    pub(super) fn settle_fresh_terrain_screen_v5(&mut self) {
        if !std::mem::take(&mut self.sends_v5.fresh_terrain_screen) {
            return;
        }
        if let Some(ticks) = &mut self.terrain_download_ticks {
            *ticks = ticks.wrapping_sub(1);
        }
    }

    /// The drop key on a 1.7.10 connection: `dropOneItem`'s C07 ([`build::drop_item`]). `false`
    /// on a 1.8.9 one.
    pub(crate) fn send_drop_v5(&mut self, whole_stack: bool) -> bool {
        if !self.version.is_1_7_10() {
            return false;
        }
        self.send_v5(build::drop_item(whole_stack));
        true
    }

    /// The sign editor's C12 on a 1.7.10 connection: the lines' plain text
    /// ([`build::update_sign`]). `false` on a 1.8.9 one.
    pub(crate) fn send_update_sign_v5(&mut self, position: BlockPos, lines: &[String; 4]) -> bool {
        if !self.version.is_1_7_10() {
            return false;
        }
        self.send_v5(build::update_sign(position, lines));
        true
    }

    /// The chat screen's C14 on a 1.7.10 connection: the text alone ([`build::tab_complete`]).
    /// `false` on a 1.8.9 one.
    pub(crate) fn send_tab_complete_v5(&mut self, text: &str) -> bool {
        if !self.version.is_1_7_10() {
            return false;
        }
        self.send_v5(build::tab_complete(text));
        true
    }

    /// A container screen's packets on a 1.7.10 connection, in their order: the plugin
    /// messages in their 1.7.10 form ([`build::container_plugin_message`]), the rest as they
    /// are. Hands `packets` back on a 1.8.9 connection.
    pub(crate) fn send_container_packets_v5(
        &mut self,
        packets: Vec<C2SPlay>,
    ) -> Option<Vec<C2SPlay>> {
        if !self.version.is_1_7_10() {
            return Some(packets);
        }
        for packet in packets {
            match packet {
                C2SPlay::PluginMessage(message) => {
                    match build::container_plugin_message(&message) {
                        Some(packet) => self.send_v5(packet),
                        None => tracing::error!(
                            channel = %message.channel,
                            "a container payload with no 1.7.10 form; not sent"
                        ),
                    }
                }
                packet => self.send(packet),
            }
        }
        None
    }

    /// The command block editor's `MC|AdvCdm` on a 1.7.10 connection, without the track-output
    /// flag ([`build::command_block`]); nothing when vanilla sends nothing. `false` on a 1.8.9
    /// connection.
    pub(crate) fn send_command_block_v5(
        &mut self,
        target: CommandBlockTarget,
        command: &str,
    ) -> bool {
        if !self.version.is_1_7_10() {
            return false;
        }
        match build::command_block(target, command) {
            Some(packet) => self.send_v5(packet),
            // java17: the C17 constructor (or the string write) throws inside the screen's try.
            None => tracing::warn!("Couldn't send command block info: payload too large"),
        }
        true
    }

    /// The book screen's `MC|BEdit`/`MC|BSign` on a 1.7.10 connection ([`build::book`]; the
    /// caller passes the stack as 1.7.10 sends it); nothing when vanilla sends nothing. `false`
    /// on a 1.8.9 connection.
    pub(crate) fn send_book_v5(&mut self, channel: &str, stack: &ItemStack) -> bool {
        if !self.version.is_1_7_10() {
            return false;
        }
        match build::book(channel, stack) {
            Some(packet) => self.send_v5(packet),
            // java17 `GuiScreenBook.sendBookToServer`'s catch.
            None => tracing::warn!("Couldn't send book info: payload too large"),
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use mc_math::BlockPos;
    use mc_physics::state::MoveInput;
    use mc_protocol::play::{c2s as v47, s2c, C2SPlay, S2CPlay};
    use mc_protocol::v5::{c2s, C2SPlay5};
    use mc_protocol::ItemStack;
    use mc_ui::screens::sign::CommandBlockTarget;
    use mc_version::GameVersion;
    use tokio::sync::mpsc::{Sender, UnboundedReceiver};

    use super::super::{JoinSettings, Session};
    use crate::net::{NetHandle, ToGame, ToNet};
    use crate::outbound::v5 as build;

    /// The settings a 1.7.10 join would read: Easy and capes hidden, so neither default
    /// matches by accident.
    fn join_settings() -> JoinSettings {
        JoinSettings {
            brand: "Aero".into(),
            difficulty: 1,
            show_cape: false,
            ..JoinSettings::default()
        }
    }

    fn rig(version: GameVersion) -> (Session, Sender<ToGame>, UnboundedReceiver<ToNet>) {
        let (handle, to_game, to_net) = NetHandle::new_pair();
        let session = Session::with_version(handle, join_settings(), version);
        (session, to_game, to_net)
    }

    fn join(session: &mut Session, to_game: &Sender<ToGame>) {
        let join = S2CPlay::JoinGame(s2c::JoinGame {
            entity_id: 3,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 3,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        });
        to_game.try_send(ToGame::Packet(join)).unwrap();
        session.poll(16);
    }

    /// Everything queued for the network task so far, both kinds, in order.
    fn take_sent(to_net: &mut UnboundedReceiver<ToNet>) -> Vec<ToNet> {
        let mut sent = Vec::new();
        while let Ok(command) = to_net.try_recv() {
            sent.push(command);
        }
        sent
    }

    fn sent_v5(sent: &[ToNet]) -> Vec<&C2SPlay5> {
        sent.iter()
            .map(|command| match command {
                ToNet::SendV5(packet) => packet,
                other => panic!("not a 1.7.10 packet: {other:?}"),
            })
            .collect()
    }

    #[test]
    fn a_1_7_10_join_sends_the_clients_settings_then_the_raw_brand() {
        let (mut session, to_game, mut to_net) = rig(GameVersion::V1_7_10);
        join(&mut session, &to_game);
        let sent = take_sent(&mut to_net);
        let sent = sent_v5(&sent);
        assert_eq!(
            sent,
            [
                &C2SPlay5::ClientSettings(build::client_settings(&join_settings())),
                &build::brand("Aero"),
            ]
        );
        // The client's own difficulty (Easy), not the server's (Hard), and its cape choice.
        let C2SPlay5::ClientSettings(settings) = sent[0] else {
            unreachable!()
        };
        assert_eq!((settings.difficulty, settings.show_cape), (1, false));

        // Negative control: a 1.8.9 session's join is unchanged, its two 1.8.9 packets.
        let (mut session, to_game, mut to_net) = rig(GameVersion::V1_8_9);
        join(&mut session, &to_game);
        let sent = take_sent(&mut to_net);
        assert!(
            matches!(
                sent.as_slice(),
                [
                    ToNet::Send(C2SPlay::ClientSettings(_)),
                    ToNet::Send(C2SPlay::PluginMessage(_))
                ]
            ),
            "{sent:?}"
        );
    }

    #[test]
    fn an_options_save_sends_only_when_a_1_7_10_field_changed() {
        let (mut session, to_game, mut to_net) = rig(GameVersion::V1_7_10);
        join(&mut session, &to_game);
        take_sent(&mut to_net);
        let mut settings = join_settings();
        assert!(
            !session.send_client_settings_if_changed(&settings),
            "as joined"
        );
        settings.displayed_skin_parts = 0x01;
        assert!(
            !session.send_client_settings_if_changed(&settings),
            "1.7.10 has no skin parts"
        );
        settings.difficulty = 3;
        assert!(session.send_client_settings_if_changed(&settings));
        settings.show_cape = true;
        assert!(session.send_client_settings_if_changed(&settings));
        assert!(!session.send_client_settings_if_changed(&settings));
        let sent = take_sent(&mut to_net);
        let sent = sent_v5(&sent);
        assert_eq!(sent.len(), 2, "{sent:?}");
        assert_eq!(
            sent[1],
            &C2SPlay5::ClientSettings(build::client_settings(&settings))
        );

        // Negative control: 1.8.9's rule ignores the difficulty and sends for a skin part.
        let (mut session, to_game, mut to_net) = rig(GameVersion::V1_8_9);
        join(&mut session, &to_game);
        take_sent(&mut to_net);
        let mut settings = join_settings();
        settings.difficulty = 3;
        assert!(!session.send_client_settings_if_changed(&settings));
        settings.displayed_skin_parts = 0x01;
        assert!(session.send_client_settings_if_changed(&settings));
    }

    #[test]
    fn screen_packets_wait_for_the_tick_and_keep_their_order() {
        let (mut session, to_game, mut to_net) = rig(GameVersion::V1_7_10);
        join(&mut session, &to_game);
        take_sent(&mut to_net);
        let close = C2SPlay::CloseWindow(v47::CloseWindow { window_id: 0 });
        let status = C2SPlay::ClientStatus(v47::ClientStatus {
            action: v47::ClientStatusAction::RequestStats,
        });

        session.hold_screen_packets_v5(true);
        session.send(close.clone());
        session.send_v5(build::anvil_name("x"));
        session.send(status.clone());
        session.hold_screen_packets_v5(false);
        assert!(take_sent(&mut to_net).is_empty(), "held until the tick");

        // Outside the pass nothing is held.
        let keep_alive = C2SPlay::KeepAlive(v47::KeepAlive { key: 0 });
        session.send(keep_alive.clone());
        session.release_screen_packets_v5();
        let sent_now = take_sent(&mut to_net);
        assert_eq!(sent_now.len(), 4, "{sent_now:?}");
        assert!(matches!(&sent_now[0], ToNet::Send(p) if *p == keep_alive));
        assert!(matches!(&sent_now[1], ToNet::Send(p) if *p == close));
        assert!(matches!(&sent_now[2], ToNet::SendV5(p) if *p == build::anvil_name("x")));
        assert!(matches!(&sent_now[3], ToNet::Send(p) if *p == status));

        session.release_screen_packets_v5();
        assert!(take_sent(&mut to_net).is_empty(), "released once");

        // Nothing goes out once the connection is gone.
        session.hold_screen_packets_v5(true);
        session.send(close.clone());
        session.hold_screen_packets_v5(false);
        to_game
            .try_send(ToGame::Disconnected(
                crate::net::DisconnectReason::ClosedByClient,
            ))
            .unwrap();
        session.poll(16);
        session.release_screen_packets_v5();
        assert!(take_sent(&mut to_net).is_empty());
    }

    #[test]
    fn a_1_8_9_session_never_holds_a_screen_packet() {
        let (mut session, to_game, mut to_net) = rig(GameVersion::V1_8_9);
        join(&mut session, &to_game);
        take_sent(&mut to_net);
        session.hold_screen_packets_v5(true);
        session.send(C2SPlay::CloseWindow(v47::CloseWindow { window_id: 0 }));
        assert_eq!(take_sent(&mut to_net).len(), 1, "sent at once, as before");
        session.hold_screen_packets_v5(false);
        session.release_screen_packets_v5();
        assert!(take_sent(&mut to_net).is_empty());
    }

    /// The tick (counting the join's own as the first) whose terrain screen sends `C00(0)`.
    fn first_terrain_keep_alive(version: GameVersion) -> Option<u32> {
        let (mut session, to_game, mut to_net) = rig(version);
        join(&mut session, &to_game);
        let mut keep_alives = Vec::new();
        for tick in 1..=45 {
            session.tick(&MoveInput::default());
            let terrain = take_sent(&mut to_net).into_iter().any(|command| {
                matches!(
                    command,
                    ToNet::Send(C2SPlay::KeepAlive(v47::KeepAlive { key: 0 }))
                )
            });
            if terrain {
                keep_alives.push(tick);
            }
        }
        let first = keep_alives.first().copied();
        // Then every 20 updates.
        assert_eq!(keep_alives.get(1).copied(), first.map(|first| first + 20));
        first
    }

    #[test]
    fn the_first_terrain_screen_counts_from_the_tick_after_the_join() {
        // 1.8.9 applies S01 before the tick, whose screen update is the screen's first.
        assert_eq!(first_terrain_keep_alive(GameVersion::V1_8_9), Some(20));
        // java17 applies it inside that tick's screen update, so the screen's first update is
        // the next tick's.
        assert_eq!(first_terrain_keep_alive(GameVersion::V1_7_10), Some(21));
    }

    #[test]
    fn the_send_sites_build_1_7_10_packets_and_leave_1_8_9_alone() {
        let (mut v5, _to_game_v5, mut to_net_v5) = rig(GameVersion::V1_7_10);
        let (mut v47, _to_game_v47, mut to_net_v47) = rig(GameVersion::V1_8_9);
        let lines = ["a", "b", "c", "d"]
            .map(|line| mc_ui::font::component::ChatComponent::text(line).to_json());
        let target = CommandBlockTarget::Block {
            position: BlockPos::new(1, 2, 3),
        };
        let book = ItemStack {
            item_id: 386,
            count: 1,
            damage: 0,
            nbt: None,
        };
        let anvil = {
            let mut payload = mc_protocol::PacketBuffer::new();
            payload.write_string("x");
            C2SPlay::PluginMessage(v47::PluginMessage {
                channel: "MC|ItemName".into(),
                data: payload.as_slice().to_vec(),
            })
        };
        let enchant = C2SPlay::EnchantItem(v47::EnchantItem {
            window_id: 1,
            button: 2,
        });
        let container = vec![enchant.clone(), anvil];

        for session in [&mut v5, &mut v47] {
            let v1_7_10 = session.version().is_1_7_10();
            assert_eq!(session.send_drop_v5(true), v1_7_10);
            assert_eq!(
                session.send_update_sign_v5(BlockPos::new(1, 2, 3), &lines),
                v1_7_10
            );
            assert_eq!(session.send_tab_complete_v5("/tp"), v1_7_10);
            assert_eq!(session.send_command_block_v5(target, "say hi"), v1_7_10);
            // Too long to send at all: handled (nothing sent) on 1.7.10.
            assert_eq!(
                session.send_command_block_v5(target, &"x".repeat(20_000)),
                v1_7_10
            );
            assert_eq!(session.send_book_v5("MC|BEdit", &book), v1_7_10);
            let handed_back = session.send_container_packets_v5(container.clone());
            assert_eq!(handed_back.is_none(), v1_7_10);
        }
        assert!(
            take_sent(&mut to_net_v47).is_empty(),
            "1.8.9: nothing sent here"
        );
        let sent = take_sent(&mut to_net_v5);
        // The over-long command sends nothing, as vanilla's caught exception.
        assert_eq!(sent.len(), 7, "{sent:?}");
        let expect_v5 = |command: &ToNet, packet: &C2SPlay5| {
            assert!(
                matches!(command, ToNet::SendV5(p) if p == packet),
                "{command:?}"
            );
        };
        expect_v5(&sent[0], &build::drop_item(true));
        expect_v5(
            &sent[1],
            &build::update_sign(BlockPos::new(1, 2, 3), &lines),
        );
        expect_v5(&sent[2], &build::tab_complete("/tp"));
        expect_v5(&sent[3], &build::command_block(target, "say hi").unwrap());
        expect_v5(&sent[4], &build::book("MC|BEdit", &book).unwrap());
        // The container's packets in their order: C11 converted on the task, then the name.
        assert!(matches!(&sent[5], ToNet::Send(p) if *p == enchant));
        expect_v5(&sent[6], &build::anvil_name("x"));
        // And the sign's plain text, the 1.7.10 lines.
        let ToNet::SendV5(C2SPlay5::UpdateSign(c2s::UpdateSign { lines, .. })) = &sent[1] else {
            unreachable!()
        };
        assert_eq!(lines, &["a", "b", "c", "d"].map(String::from));
    }
}
