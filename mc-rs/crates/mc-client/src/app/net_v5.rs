//! The 1.7.10 inbound drain inside the client's tick.
//!
//! The game thread's side, not part of `crate::net` (the connection tasks, among them the
//! 1.7.10 one in `net/task_v5.rs`). A child of `app::driver` (declared there), so it drives the
//! same private seams as the frame poll (`Client::poll_in_place`): apply one message, let the
//! replay recorder take it, fold the session phase into the state machine, react to that
//! message's events, then the next.
//!
//! A 1.8.9 session never reaches past the first test of either hook: its packets are applied by
//! the frame poll, unchanged. A 1.7.10 session's are applied here, at most
//! [`PACKETS_PER_DRAIN`] per tick (`crate::session`'s 1.7.10 module explains where and why).

use mc_render::Camera;

use super::Client;
use crate::session::v5::PACKETS_PER_DRAIN;
use crate::session::Session;

impl Client {
    /// The tick's in-world drain, right after `PlayerControllerMP.updateController`'s held-item
    /// sync (java17 `PlayerControllerMP.java:254-261`): replies such as C0F and the S08 C06 leave
    /// after the tick's C09 and before its input and movement packets.
    pub(super) fn drain_v5_in_world(&mut self, camera: &mut Camera) {
        if self
            .session
            .as_ref()
            .is_some_and(Session::drains_v5_in_world)
        {
            self.drain_v5(PACKETS_PER_DRAIN, Some(camera));
        }
    }

    /// The tick's drain before a world exists, at the screen-update step (java17
    /// `GuiConnecting.updateScreen`, `Minecraft.java:1618-1640`). The `S01` it applies makes the
    /// world, and the rest of this tick then runs with it, as in 1.7.10.
    pub(super) fn drain_v5_before_world(&mut self, camera: &mut Camera) {
        if self
            .session
            .as_ref()
            .is_some_and(Session::drains_v5_before_world)
        {
            self.drain_v5(PACKETS_PER_DRAIN, Some(camera));
        }
    }

    /// Everything a watched replay's 1.7.10 session has queued, right after playback pushed it:
    /// the recording already placed each packet at the time it was applied.
    pub(super) fn drain_v5_all(&mut self) {
        if self.plays_1_7_10() && self.session.as_ref().is_some_and(Session::is_driven) {
            self.drain_v5(usize::MAX, None);
        }
    }

    /// Apply up to `budget` queued 1.7.10 messages with the frame poll's apply-then-react loop,
    /// then the batch hooks the frame poll runs over its events.
    fn drain_v5(&mut self, budget: usize, camera: Option<&mut Camera>) {
        let mut events = Vec::new();
        self.pull_v5(&mut events);
        for _ in 0..budget {
            let first = events.len();
            let applied = match &mut self.session {
                Some(session) => session.apply_next_v5(&mut events),
                None => false,
            };
            if !applied {
                break;
            }
            self.rec_after_packet();
            if let Some(session) = &mut self.session {
                let phase = session.phase().clone();
                let effects = self.machine.on_phase(&phase);
                if !effects.is_empty() {
                    self.run_effects(&effects);
                }
            }
            for event in &events[first..] {
                self.react(event);
            }
        }
        if events.is_empty() {
            return;
        }
        self.markers_events(&events);
        self.react_to_break_progress(&events);
        self.anchor_scripted_camera(&events);
        // An S08 applied in this tick turns the player the tick's input then reads.
        if let Some(camera) = camera {
            self.sync_camera_to_position_correction(&events, camera);
        }
        // Harnesses read a frame's events here; the frame poll replaces them next frame.
        self.last_events.extend(events);
    }

    /// Take what a live 1.7.10 connection delivered since the frame poll, the way that poll does
    /// (without its once-a-frame clock step), so that the second tick of a two-tick frame drains
    /// packets that arrived during the first one, as java17's queue, which Netty fills as packets
    /// arrive, does. A close seen here ends the session at this tick, as `updateController`
    /// notices a closed channel (`PlayerControllerMP.java:258-269`).
    fn pull_v5(&mut self, events: &mut Vec<crate::sync::ClientEvent>) {
        if !self.session.as_ref().is_some_and(Session::pulls_v5_in_tick) {
            return;
        }
        loop {
            let first = events.len();
            let more = match &mut self.session {
                Some(session) => session.poll_next(events),
                None => break,
            };
            self.rec_after_packet();
            if let Some(session) = &mut self.session {
                let phase = session.phase().clone();
                let effects = self.machine.on_phase(&phase);
                if !effects.is_empty() {
                    self.run_effects(&effects);
                }
            }
            for event in &events[first..] {
                self.react(event);
            }
            if !more {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use mc_protocol::connection::LoginOutcome;
    use mc_protocol::play::{s2c as s2c47, S2CPlay};
    use mc_protocol::v5::{s2c, S2CPlay5};
    use mc_version::GameVersion;
    use tokio::sync::mpsc::{Sender, UnboundedReceiver};

    use super::*;
    use crate::app::config::{ClientConfig, LoginMode};
    use crate::net::{DisconnectReason, NetHandle, ToGame, ToNet};
    use crate::session::{JoinSettings, Phase};
    use crate::sync::ClientEvent;

    /// A client on the real assets, with no session. `None` (skip) without the vanilla jar, as
    /// the driver's own tests do.
    fn client() -> Option<(Client, tempfile::TempDir)> {
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

    /// [`client`] with an offline session of `version` on channels the test holds, already past
    /// login.
    fn client_with_session(
        version: GameVersion,
    ) -> Option<(
        Client,
        tempfile::TempDir,
        Sender<ToGame>,
        UnboundedReceiver<ToNet>,
    )> {
        let (mut client, dir) = client()?;
        let (handle, to_game, to_net) = NetHandle::new_pair();
        client.session = Some(Session::with_version(
            handle,
            JoinSettings::default(),
            version,
        ));
        to_game
            .try_send(ToGame::Connected(LoginOutcome {
                uuid: uuid::Uuid::from_u128(7),
                username: "Dev".to_owned(),
                compression_threshold: None,
                encrypted: false,
            }))
            .expect("room");
        Some((client, dir, to_game, to_net))
    }

    fn join_game_v5() -> ToGame {
        ToGame::PacketV5(Box::new(S2CPlay5::JoinGame(s2c::JoinGame {
            entity_id: 27,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 2,
            max_players: 20,
            level_type: "flat".to_owned(),
        })))
    }

    fn time_v5(world_time: i64) -> ToGame {
        ToGame::PacketV5(Box::new(S2CPlay5::TimeUpdate(s2c::TimeUpdate {
            total_world_time: world_time,
            world_time,
        })))
    }

    fn world_time(client: &Client) -> i64 {
        client
            .session
            .as_ref()
            .expect("session")
            .state()
            .env
            .world_time
    }

    fn phase(client: &Client) -> Phase {
        client.session.as_ref().expect("session").phase().clone()
    }

    /// 1.7.10 applies queued packets only in the tick, at most 1001 per drain, in arrival order:
    /// the frame poll applies none, the connecting-screen drain takes the S01 and the first 1000
    /// time updates, the in-world drain the rest. Once the connection has ended nothing still
    /// queued is applied (java17 `PlayerControllerMP.updateController` checks the channel first).
    #[test]
    fn packets_are_applied_in_the_tick_at_most_1001_per_drain_in_order() {
        let Some((mut client, _dir, to_game, _to_net)) = client_with_session(GameVersion::V1_7_10)
        else {
            return;
        };
        let mut camera = Camera::default();
        to_game.try_send(join_game_v5()).expect("room");
        for time in 0..1500 {
            to_game.try_send(time_v5(time)).expect("room");
        }

        // A frame: the login is handled, every packet is only queued.
        client.poll_in_place();
        assert_eq!(phase(&client), Phase::LoggingIn);
        assert_eq!(world_time(&client), 0);

        // Before the world, only the screen-update drain applies: S01 + 1000 updates = 1001.
        client.drain_v5_in_world(&mut camera);
        assert_eq!(world_time(&client), 0);
        client.drain_v5_before_world(&mut camera);
        assert!(client
            .last_events
            .iter()
            .any(|event| matches!(event, ClientEvent::JoinedGame(_))));
        assert_eq!(phase(&client), Phase::Playing);
        assert_eq!(world_time(&client), 999);

        // With a world, the in-world drain takes the rest, in order.
        client.drain_v5_before_world(&mut camera);
        assert_eq!(world_time(&client), 999);
        client.drain_v5_in_world(&mut camera);
        assert_eq!(world_time(&client), 1499);

        // A close drops what is still queued.
        for time in 1500..1505 {
            to_game.try_send(time_v5(time)).expect("room");
        }
        to_game
            .try_send(ToGame::Disconnected(DisconnectReason::ClosedByClient))
            .expect("room");
        client.poll_in_place();
        if let Some(session) = client.session.as_ref() {
            assert!(!session.phase().is_live());
        }
        client.drain_v5_in_world(&mut camera);
        if client.session.is_some() {
            assert_eq!(world_time(&client), 1499);
        }
    }

    /// Negative control: a 1.8.9 session's frame poll applies everything at once, uncapped, and
    /// the 1.7.10 drains do nothing.
    #[test]
    fn a_1_8_9_session_is_applied_by_the_frame_poll_only() {
        let Some((mut client, _dir, to_game, _to_net)) = client_with_session(GameVersion::V1_8_9)
        else {
            return;
        };
        let mut camera = Camera::default();
        for time in 0..1500 {
            to_game
                .try_send(ToGame::Packet(S2CPlay::TimeUpdate(s2c47::TimeUpdate {
                    total_world_time: time,
                    world_time: time,
                })))
                .expect("room");
        }
        client.poll_in_place();
        assert_eq!(world_time(&client), 1499);
        to_game
            .try_send(ToGame::Packet(S2CPlay::TimeUpdate(s2c47::TimeUpdate {
                total_world_time: 7,
                world_time: 7,
            })))
            .expect("room");
        client.drain_v5_before_world(&mut camera);
        client.drain_v5_in_world(&mut camera);
        assert_eq!(
            world_time(&client),
            1499,
            "only the frame poll applies 1.8.9 packets"
        );
        client.poll_in_place();
        assert_eq!(world_time(&client), 7);
    }

    /// java17's `handleJoinGame` keeps the client's difficulty option (which its C15 reports);
    /// java18's overwrites it with the S01's.
    #[test]
    fn a_1_7_10_join_keeps_the_difficulty_option() {
        let Some((mut client, _dir, to_game, _to_net)) = client_with_session(GameVersion::V1_7_10)
        else {
            return;
        };
        client.settings.difficulty = 3;
        to_game.try_send(join_game_v5()).expect("room");
        client.poll_in_place();
        client.drain_v5_before_world(&mut Camera::default());
        assert_eq!(phase(&client), Phase::Playing);
        assert_eq!(client.settings.difficulty, 3);

        let Some((mut client, _dir, to_game, _to_net)) = client_with_session(GameVersion::V1_8_9)
        else {
            return;
        };
        client.settings.difficulty = 3;
        to_game
            .try_send(ToGame::Packet(S2CPlay::JoinGame(s2c47::JoinGame {
                entity_id: 27,
                hardcore_mode: false,
                game_mode: 0,
                dimension: 0,
                difficulty: 2,
                max_players: 20,
                level_type: "flat".to_owned(),
                reduced_debug_info: false,
            })))
            .expect("room");
        client.poll_in_place();
        assert_eq!(client.settings.difficulty, 2, "1.8.9 control");
    }

    /// A packet that arrives after the frame poll is applied by the next tick of the same frame,
    /// as java17's queue, which Netty fills on arrival, is drained by every tick; a close seen
    /// there ends the session at that tick. Not on a 1.8.9 session.
    #[test]
    fn each_tick_takes_what_arrived_since_the_frame_poll() {
        let Some((mut client, _dir, to_game, _to_net)) = client_with_session(GameVersion::V1_7_10)
        else {
            return;
        };
        let mut camera = Camera::default();
        to_game.try_send(join_game_v5()).expect("room");
        client.poll_in_place();
        client.drain_v5_before_world(&mut camera);
        assert_eq!(phase(&client), Phase::Playing);

        // A frame poll, then a packet, then a tick: the tick applies it.
        client.poll_in_place();
        to_game.try_send(time_v5(41)).expect("room");
        client.drain_v5_in_world(&mut camera);
        assert_eq!(world_time(&client), 41);

        // A close that arrives between the frame poll and the tick ends the session there.
        client.poll_in_place();
        to_game.try_send(time_v5(42)).expect("room");
        to_game
            .try_send(ToGame::Disconnected(DisconnectReason::ClosedByClient))
            .expect("room");
        client.drain_v5_in_world(&mut camera);
        if let Some(session) = client.session.as_ref() {
            assert!(!session.phase().is_live());
            assert_eq!(
                session.state().env.world_time,
                41,
                "nothing applied after a close"
            );
        }
    }

    /// One protocol-5 clientbound packet as recorded: id VarInt, then the body.
    fn wire(packet: &S2CPlay5) -> Vec<u8> {
        let mut buf = mc_protocol::PacketBuffer::new();
        buf.write_varint(packet.id());
        packet.write(&mut buf);
        buf.as_slice().to_vec()
    }

    /// What a 1.7.10 server sends to put a player in the world: the join, one column of stone
    /// at y 64, the spawn S08.
    fn joining_packets_v5() -> Vec<S2CPlay5> {
        let mut raw = vec![0u8; 4096];
        raw[..256].fill(1);
        raw.extend(std::iter::repeat_n(0u8, 2048 * 2));
        raw.extend(std::iter::repeat_n(0xFFu8, 2048));
        raw.extend(std::iter::repeat_n(1u8, 256));
        let bulk = mc_protocol::v5::chunk::MapChunkBulk::from_raw(true, &[(0, 0, 1 << 4, 0, &raw)])
            .expect("bulk");
        let ToGame::PacketV5(join) = join_game_v5() else {
            unreachable!("a protocol-5 packet")
        };
        vec![
            *join,
            S2CPlay5::MapChunkBulk(bulk),
            S2CPlay5::PlayerPosLook(s2c::PlayerPosLook {
                x: 8.5,
                y: 65.620_000_004_768_37,
                z: 8.5,
                yaw: 0.0,
                pitch: 0.0,
                on_ground: false,
            }),
        ]
    }

    fn stone_at_y64(client: &Client) -> bool {
        client.session.as_ref().is_some_and(|session| {
            session
                .state()
                .world
                .block_at(mc_math::BlockPos::new(3, 64, 5))
                == mc_data::v5::state_from_v5(1 << 4)
        })
    }

    /// `--replay` of a capture recorded on a 1.7.10 connection (protocol 5 in its header) plays
    /// through a 1.7.10 session: it joins and applies its chunk in the ticks, as live. The same
    /// capture read by a 1.8.9 session would never join.
    #[test]
    fn a_protocol_5_capture_replays_into_the_world() {
        use mc_parity::capture::{CaptureMeta, Direction, Recorder};
        use mc_render::{GameSettings, InputState, ShellApp, TickCtx};

        let Some((mut client, dir)) = client() else {
            return;
        };
        let mut recorder = Recorder::new(CaptureMeta {
            protocol: 5,
            address: "test".into(),
            started_at_unix_millis: 0,
        });
        let mut login = mc_protocol::PacketBuffer::new();
        login.write_varint(0x02);
        mc_protocol::Packet::write(
            &mc_protocol::login::LoginSuccess {
                uuid: uuid::Uuid::from_u128(9),
                username: "Replayed".into(),
            },
            &mut login,
        );
        recorder.record_at(
            Direction::Clientbound,
            mc_protocol::State::Login,
            0,
            login.as_slice(),
        );
        for (micros, packet) in (10..).zip(joining_packets_v5()) {
            recorder.record_at(
                Direction::Clientbound,
                mc_protocol::State::Play,
                micros,
                &wire(&packet),
            );
        }
        let path = dir.path().join("v5.mccap");
        std::fs::write(&path, recorder.finish().encode().expect("encodes")).expect("writes");
        client.config.replay = Some(path);
        client.open_replay("", 0);
        assert_eq!(
            client.session.as_ref().map(Session::version),
            Some(GameVersion::V1_7_10)
        );

        let mut camera = Camera::default();
        let mut input = InputState::new();
        let mut settings = GameSettings::default();
        for tick in 0..40 {
            ShellApp::before_ticks(&mut client, &mut camera);
            ShellApp::tick(
                &mut client,
                &mut TickCtx {
                    input: &mut input,
                    settings: &mut settings,
                    camera: &mut camera,
                    tick,
                },
            );
            if stone_at_y64(&client) && phase(&client) == Phase::Playing {
                break;
            }
        }
        assert_eq!(phase(&client), Phase::Playing);
        assert!(stone_at_y64(&client), "the capture's chunk was applied");
    }
}
