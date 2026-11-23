//! A full join → play → disconnect sequence against an in-process server over loopback TCP.
//!
//! # What this is, and exactly what it does and does not prove
//!
//! S14's "done when" is *the client joins a real server, sees terrain and players, and can chat
//! and walk*. **There is no network access and no Minecraft server in this environment**, so
//! that cannot be done literally — the same wall S03, S06 and S07 hit and recorded. This file
//! is the strongest available substitute, and it follows S07's precedent
//! (`mc-protocol/tests/login_loopback.rs`) deliberately:
//!
//! * it is a **real `127.0.0.1` TCP socket**, not an in-memory pipe;
//! * the far side's **framing is independently implemented** here — the VarInt length prefix,
//!   the packet-id VarInt and the read buffering share no code with `mc-protocol`'s codec;
//! * the client under test is the real thing end to end:
//!   [`NetRuntime`](mc_client::net::NetRuntime) spawns the real connection task, and the frame
//!   loop is a genuinely **synchronous** loop on the test thread calling
//!   [`Session::poll`](mc_client::Session)/`tick` — it never awaits, which is the §11 property
//!   this section exists to establish.
//!
//! **Proved by this file:** the state machine reaches `Play`; the client answers `JoinGame`
//! with the settings and brand packets vanilla sends; a chunk payload off the wire produces the
//! right block at the right coordinate; player and mob spawns land in the ECS; an
//! `S08PacketPlayerPosLook` is answered with the `C06` the server requires; movement packets
//! flow at the right shape; chat goes both ways; a relative-move packet moves a tracked entity;
//! `KeepAlive` is echoed without the game loop being involved; and a Play-state `Disconnect`
//! surfaces its reason JSON to the UI.
//!
//! **Not proved by this file:** that a real vanilla 1.8.9 server accepts these bytes. The far
//! side is a test double. Its framing is independent, but both ends agree on the packet
//! *bodies* by construction, because they are encoded with `mc-protocol`'s own `Packet::write` —
//! hand-encoding 74 packet bodies a second time would test S06, which S06 already tested. Nor
//! does it exercise encryption or compression: those layers are S07's, and S07's own loopback
//! test covers them with an independent AES-CFB8 and zlib implementation. Loopback TCP is local
//! I/O, not network access.

use std::time::{Duration, Instant};

use mc_client::net::{channel::DisconnectOrigin, ConnectOptions, NetRuntime};
use mc_client::session::JoinSettings;
use mc_client::sync::{ClientEvent, TrackedKind};
use mc_client::{DisconnectReason, Phase, Session};
use mc_math::{BlockPos, ChunkPos, DVec3, Facing};
use mc_physics::state::MoveInput;
use mc_protocol::handshake::Handshake;
use mc_protocol::login::{LoginStart, LoginSuccess};
use mc_protocol::packet::Packet;
use mc_protocol::play::{c2s, s2c};
use mc_protocol::{Angle, PacketBuffer};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use uuid::Uuid;

// ---------------------------------------------------------------------------------------------
// An independently-framed peer. Nothing below this line uses `mc_protocol::codec`,
// `mc_protocol::Connection` or `PacketBuffer`'s VarInt helpers for *framing*.
// ---------------------------------------------------------------------------------------------

struct RefPeer {
    stream: TcpStream,
    buf: Vec<u8>,
}

impl RefPeer {
    fn new(stream: TcpStream) -> RefPeer {
        RefPeer {
            stream,
            buf: Vec::new(),
        }
    }

    async fn fill(&mut self, at_least: usize) {
        while self.buf.len() < at_least {
            let mut chunk = [0u8; 4096];
            let read = self.stream.read(&mut chunk).await.expect("socket read");
            assert!(read > 0, "peer closed the connection mid-frame");
            self.buf.extend_from_slice(&chunk[..read]);
        }
    }

    /// Read a VarInt from the front of the buffer, filling from the socket as needed.
    async fn read_varint(&mut self) -> i32 {
        let mut value: i32 = 0;
        let mut shift = 0;
        let mut index = 0usize;
        loop {
            self.fill(index + 1).await;
            let byte = self.buf[index];
            index += 1;
            value |= i32::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
            assert!(shift < 35, "VarInt too big");
        }
        self.buf.drain(..index);
        value
    }

    /// One whole frame: length prefix, then packet id, then body.
    async fn read_packet(&mut self) -> (i32, Vec<u8>) {
        let length = self.read_varint().await as usize;
        self.fill(length).await;
        let frame: Vec<u8> = self.buf.drain(..length).collect();

        let mut index = 0usize;
        let mut id: i32 = 0;
        let mut shift = 0;
        loop {
            let byte = frame[index];
            index += 1;
            id |= i32::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        (id, frame[index..].to_vec())
    }

    /// Read frames until `predicate` accepts one, returning its body.
    async fn read_until(&mut self, wanted_id: i32, predicate: impl Fn(&[u8]) -> bool) -> Vec<u8> {
        for _ in 0..2000 {
            let (id, body) = self.read_packet().await;
            if id == wanted_id && predicate(&body) {
                return body;
            }
        }
        panic!("never saw packet id {wanted_id:#04x}");
    }

    fn write_varint(out: &mut Vec<u8>, mut value: i32) {
        loop {
            let mut byte = (value & 0x7F) as u8;
            value = ((value as u32) >> 7) as i32;
            if value != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if value == 0 {
                return;
            }
        }
    }

    async fn send_raw(&mut self, id: i32, body: &[u8]) {
        let mut payload = Vec::new();
        Self::write_varint(&mut payload, id);
        payload.extend_from_slice(body);

        let mut frame = Vec::new();
        Self::write_varint(&mut frame, payload.len() as i32);
        frame.extend_from_slice(&payload);
        self.stream.write_all(&frame).await.expect("socket write");
        self.stream.flush().await.expect("flush");
    }

    /// Encode a typed packet's *body* with `mc-protocol` and frame it ourselves.
    async fn send<P: Packet>(&mut self, packet: &P) {
        let mut buffer = PacketBuffer::new();
        packet.write(&mut buffer);
        self.send_raw(P::ID, buffer.as_slice()).await;
    }
}

fn decode<P: Packet>(bytes: &[u8]) -> P {
    let mut buffer = PacketBuffer::from_slice(bytes);
    P::read(&mut buffer).expect("packet body")
}

// ---------------------------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------------------------

const CHUNK: ChunkPos = ChunkPos { x: 2, z: -3 };
/// Local coordinates inside section 0 of [`CHUNK`].
const LOCAL: (u8, u8, u8) = (1, 2, 3);
/// `Block.BLOCK_STATE_IDS` packing: stone is block id 1, meta 0 -> `1 << 4`.
const STONE: u16 = 1 << 4;
const KICK_REASON: &str = r#"{"translate":"multiplayer.disconnect.server_shutdown"}"#;
const SERVER_CHAT: &str = r#"{"text":"<Notch> hello there","color":"white"}"#;
const CLIENT_CHAT: &str = "hello from the rust client";

/// A single-section chunk payload with one stone block in it, built from the documented wire
/// layout (blocks for every selected section, then block light, then sky light, then biomes) —
/// the same layout S08 traced out of `Chunk.fillChunk`.
fn chunk_payload() -> Vec<u8> {
    let length = mc_world::expected_chunk_data_len(0b1, true, true);
    let mut data = vec![0u8; length];
    let (x, y, z) = LOCAL;
    // java: ExtendedBlockStorage indexes as `y << 8 | z << 4 | x`, little-endian u16.
    let index = (usize::from(y) << 8) | (usize::from(z) << 4) | usize::from(x);
    data[index * 2] = (STONE & 0xFF) as u8;
    data[index * 2 + 1] = (STONE >> 8) as u8;
    data
}

/// The server side of the conversation.
async fn scripted_server(listener: TcpListener) -> ScriptResult {
    let (socket, _) = listener.accept().await.expect("accept");
    socket.set_nodelay(true).expect("nodelay");
    let mut peer = RefPeer::new(socket);

    // --- handshake / login (offline mode: no EncryptionRequest) ---
    let (id, body) = peer.read_packet().await;
    assert_eq!(id, Handshake::ID);
    let handshake: Handshake = decode(&body);
    assert_eq!(handshake.protocol_version, 47, "protocol 47 only");

    let (id, body) = peer.read_packet().await;
    assert_eq!(id, LoginStart::ID);
    let login: LoginStart = decode(&body);
    let username = login.username.clone();

    peer.send(&LoginSuccess {
        uuid: Uuid::from_u128(0x0000_0000_0000_4000_8000_0000_0000_0001),
        username: username.clone(),
    })
    .await;

    // --- join ---
    peer.send(&s2c::JoinGame {
        entity_id: 42,
        hardcore_mode: false,
        game_mode: 0,
        dimension: 0,
        difficulty: 2,
        max_players: 20,
        level_type: "default".into(),
        reduced_debug_info: false,
    })
    .await;

    // java: handleJoinGame answers with C15PacketClientSettings then the MC|Brand payload.
    let (id, body) = peer.read_packet().await;
    assert_eq!(
        id,
        c2s::ClientSettings::ID,
        "the first thing a client sends after JoinGame"
    );
    let settings: c2s::ClientSettings = decode(&body);
    let (id, body) = peer.read_packet().await;
    assert_eq!(id, c2s::PluginMessage::ID, "then the plugin message");
    let brand: c2s::PluginMessage = decode(&body);

    // --- terrain and entities ---
    peer.send(&mc_protocol::chunk::ChunkData {
        chunk_x: CHUNK.x,
        chunk_z: CHUNK.z,
        ground_up_continuous: true,
        primary_bit_mask: 0b1,
        data: chunk_payload(),
    })
    .await;
    peer.send(&s2c::SpawnPlayer {
        entity_id: 100,
        player_uuid: Uuid::from_u128(0xABCD),
        x: 32 * 40,
        y: 32 * 70,
        z: 32 * -40,
        yaw: Angle::from_degrees(90.0),
        pitch: Angle(0),
        current_item: 0,
        metadata: Vec::new(),
    })
    .await;
    peer.send(&s2c::SpawnMob {
        entity_id: 101,
        entity_type: 54, // Zombie
        x: 0,
        y: 32 * 64,
        z: 0,
        yaw: Angle(0),
        pitch: Angle(0),
        head_pitch: Angle(0),
        velocity_x: 0,
        velocity_y: 0,
        velocity_z: 0,
        metadata: Vec::new(),
    })
    .await;
    peer.send(&s2c::PlayerListItem::AddPlayer(vec![s2c::AddPlayerEntry {
        uuid: Uuid::from_u128(0xABCD),
        username: "Notch".into(),
        properties: vec![],
        game_mode: s2c::GameMode::Survival,
        ping: 137,
        display_name: None,
    }]))
    .await;

    // --- place the player, which vanilla answers with a C06 confirmation ---
    peer.send(&s2c::PlayerPositionLook {
        x: 40.5,
        y: 71.0,
        z: -39.5,
        yaw: 180.0,
        pitch: 12.5,
        flags: 0,
    })
    .await;
    let body = peer
        .read_until(c2s::PlayerPositionAndLook::ID, |body| {
            let packet: c2s::PlayerPositionAndLook = decode(body);
            (packet.x - 40.5).abs() < 1e-9
        })
        .await;
    let confirmation: c2s::PlayerPositionAndLook = decode(&body);

    // --- chat, both ways ---
    peer.send(&s2c::Chat {
        component: SERVER_CHAT.into(),
        position: 0,
    })
    .await;
    let body = peer.read_until(c2s::ChatMessage::ID, |_| true).await;
    let client_chat: c2s::ChatMessage = decode(&body);

    // --- the player mines a block and places one, which is three more packet kinds ---
    //
    // Read strictly in wire order: `read_until` discards everything it scans past, so reading
    // the later packet first would swallow the earlier ones. java: the order is
    // `clickBlock` -> `onPlayerDamageBlock`'s STOP -> `onPlayerRightClick`'s
    // `syncCurrentPlayItem` -> the placement.
    let start: c2s::PlayerDigging = decode(
        &peer
            .read_until(c2s::PlayerDigging::ID, |body| {
                let packet: c2s::PlayerDigging = decode(body);
                packet.status == c2s::PlayerDiggingStatus::StartDestroyBlock
            })
            .await,
    );
    let stop: c2s::PlayerDigging = decode(
        &peer
            .read_until(c2s::PlayerDigging::ID, |body| {
                let packet: c2s::PlayerDigging = decode(body);
                packet.status == c2s::PlayerDiggingStatus::StopDestroyBlock
            })
            .await,
    );
    let held: c2s::HeldItemChange =
        decode(&peer.read_until(c2s::HeldItemChange::ID, |_| true).await);
    let placement: c2s::PlayerBlockPlacement = decode(
        &peer
            .read_until(c2s::PlayerBlockPlacement::ID, |_| true)
            .await,
    );

    // --- an entity moves ---
    peer.send(&s2c::EntityRelMove {
        entity_id: 101,
        delta_x: 32, // +1 block
        delta_y: 0,
        delta_z: -16, // -0.5 block
        on_ground: true,
    })
    .await;

    // --- keep-alive must be answered by the network task, not the frame ---
    peer.send(&s2c::KeepAlive { id: 0x0BAD_F00D }).await;
    let body = peer.read_until(c2s::KeepAlive::ID, |_| true).await;
    let echo: c2s::KeepAlive = decode(&body);

    // --- and finally a Play-state kick ---
    peer.send(&s2c::Disconnect {
        reason: KICK_REASON.into(),
    })
    .await;

    ScriptResult {
        username,
        settings,
        brand,
        confirmation,
        client_chat: client_chat.message,
        keep_alive_echo: echo.key,
        held,
        start,
        stop,
        placement,
    }
}

struct ScriptResult {
    username: String,
    settings: c2s::ClientSettings,
    brand: c2s::PluginMessage,
    confirmation: c2s::PlayerPositionAndLook,
    client_chat: String,
    keep_alive_echo: i32,
    held: c2s::HeldItemChange,
    start: c2s::PlayerDigging,
    stop: c2s::PlayerDigging,
    placement: c2s::PlayerBlockPlacement,
}

/// The block the test player mines, and the one it then places against.
const DIG_TARGET: BlockPos = BlockPos {
    x: 33,
    y: 2,
    z: -45,
};

/// Drive a real, **synchronous** frame loop until the session ends or the deadline passes.
fn drive(session: &mut Session, timeout: Duration) -> Vec<ClientEvent> {
    let deadline = Instant::now() + timeout;
    let mut collected = Vec::new();
    let mut chat_sent = false;
    while Instant::now() < deadline {
        let events = session.poll(512);
        let ended = events
            .iter()
            .any(|event| matches!(event, ClientEvent::Disconnected(_)));
        let spawned = events
            .iter()
            .any(|event| matches!(event, ClientEvent::TerrainDownloaded));
        collected.extend(events);
        if spawned && !chat_sent {
            // "can chat and walk": the chat goes out from the frame loop, exactly as a player
            // pressing enter would send it.
            session.send_chat(CLIENT_CHAT);
            // ... and then the player mines the stone block the chunk packet delivered and
            // places one back, through the same `PlayerControllerMP` seam a real input handler
            // would use.
            session.player_action(|out, packets| {
                out.start_digging(DIG_TARGET, Facing::Up, packets);
            });
            session.player_action(|out, packets| {
                out.finish_digging(DIG_TARGET, Facing::Up, packets);
            });
            session.player_action(|out, packets| {
                out.place_block(
                    DIG_TARGET,
                    Facing::Up,
                    DVec3::new(0.5, 1.0, 0.25),
                    None,
                    3,
                    packets,
                );
            });
            chat_sent = true;
        }
        if matches!(session.phase(), Phase::Playing) {
            session.tick(&MoveInput::default());
        }
        if ended {
            return collected;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    panic!("the session never finished");
}

#[test]
fn full_join_play_and_kick_sequence_over_real_loopback_tcp() {
    let runtime = NetRuntime::new().expect("runtime");
    let listener = runtime
        .handle()
        .block_on(TcpListener::bind("127.0.0.1:0"))
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let server = runtime.handle().spawn(scripted_server(listener));

    let handle = runtime.connect(ConnectOptions::offline("127.0.0.1", port, "Dev"));
    let mut session = Session::new(handle, JoinSettings::default());

    let events = drive(&mut session, Duration::from_secs(30));
    let script = runtime
        .handle()
        .block_on(server)
        .expect("the scripted server must run to completion");

    // ---- login and join ----
    assert_eq!(script.username, "Dev");
    assert_eq!(session.state().local.username, "Dev");
    assert_eq!(session.state().local.entity_id, 42);
    assert!(events
        .iter()
        .any(|event| matches!(event, ClientEvent::JoinedGame(_))));

    // java: handleJoinGame's two immediate replies.
    assert_eq!(script.settings.locale, "en_US");
    // JoinSettings::default follows Java 1.8.9's GameSettings default.
    assert_eq!(script.settings.view_distance, 12);
    assert_eq!(script.brand.channel, "MC|Brand");
    assert_eq!(
        &script.brand.data[1..],
        b"vanilla",
        "the payload is a length-prefixed protocol string"
    );

    // ---- terrain ----
    let (x, y, z) = LOCAL;
    let block = BlockPos::new(
        (CHUNK.x << 4) + i32::from(x),
        i32::from(y),
        (CHUNK.z << 4) + i32::from(z),
    );
    assert!(session.world().is_chunk_loaded(CHUNK));
    assert_eq!(
        session.world().block_at(block).id,
        STONE,
        "the chunk payload must decode to stone at {block:?}"
    );
    assert_eq!(
        session
            .world()
            .block_at(BlockPos::new(block.x + 1, block.y, block.z)),
        mc_data::BlockState::AIR
    );

    // ---- players and mobs ----
    let player = session.state().entities.get(100).expect("remote player");
    assert_eq!(player.kind, TrackedKind::Player);
    assert_eq!(player.uuid, Some(Uuid::from_u128(0xABCD)));
    let marker = session
        .state()
        .world
        .ecs
        .get::<&mc_client::sync::RemotePlayer>(player.entity)
        .expect("RemotePlayer component");
    assert_eq!(marker.network_id, 100);

    let zombie = session.state().entities.get(101).expect("zombie");
    assert_eq!(
        zombie.kind,
        TrackedKind::Type(mc_data::entity_type(54).unwrap())
    );
    // The relative move: +32/32 = +1.0 X and -16/32 = -0.5 Z, accumulated in fixed point.
    assert_eq!(zombie.server_pos, [32, 32 * 64, -16]);
    assert_eq!(zombie.target_pos, DVec3::new(1.0, 64.0, -0.5));

    // ---- tab list and latency ----
    assert_eq!(session.state().players.len(), 1);
    assert_eq!(
        session
            .state()
            .players
            .get(&Uuid::from_u128(0xABCD))
            .unwrap()
            .ping_ms,
        137
    );

    // ---- the position correction and its mandatory confirmation ----
    assert!(events
        .iter()
        .any(|event| matches!(event, ClientEvent::TerrainDownloaded)));
    assert_eq!(script.confirmation.x, 40.5);
    assert_eq!(script.confirmation.y, 71.0);
    assert_eq!(script.confirmation.z, -39.5);
    assert_eq!(script.confirmation.yaw, 180.0);
    assert_eq!(script.confirmation.pitch, 12.5);
    // java: handlePlayerPosLook (:717) sends `onGround = false` unconditionally.
    assert!(!script.confirmation.on_ground);
    let predictor = session.predictor().expect("predictor");
    assert_eq!(predictor.state.pos, DVec3::new(40.5, 71.0, -39.5));

    // ---- chat, both directions ----
    assert_eq!(script.client_chat, CLIENT_CHAT);
    let lang = mc_assets::Language::new("en_US");
    let heard: Vec<String> = events
        .iter()
        .filter_map(|event| match event {
            ClientEvent::Chat { component, .. } => Some(component.unformatted_text(&lang)),
            _ => None,
        })
        .collect();
    assert_eq!(heard, vec!["<Notch> hello there".to_owned()]);

    // ---- digging and placing reached the wire in vanilla's order and shape ----
    // java: `PlayerControllerMP.onPlayerRightClick` calls `syncCurrentPlayItem` first, and
    // `currentPlayerItem` starts at -1, so slot 3 is a real change.
    assert_eq!(script.held.slot_id, 3);
    assert_eq!(script.start.position, DIG_TARGET);
    assert_eq!(script.start.face, Facing::Up);
    assert_eq!(script.stop.position, DIG_TARGET);
    assert_eq!(script.placement.position, DIG_TARGET);
    assert_eq!(script.placement.placed_block_direction, 1, "EnumFacing.UP");
    // java: `(int)(facingX * 16.0F)` per axis.
    assert_eq!(
        (
            script.placement.cursor_x,
            script.placement.cursor_y,
            script.placement.cursor_z
        ),
        (8, 16, 4)
    );

    // ---- keep-alive was answered by the network task ----
    assert_eq!(script.keep_alive_echo, 0x0BAD_F00D);

    // ---- and the kick reached the UI with its reason intact ----
    match session.phase() {
        Phase::Disconnected(DisconnectReason::WithOrigin { reason, origin }) => {
            assert_eq!(*origin, DisconnectOrigin::Play);
            match reason.as_ref() {
                DisconnectReason::Kicked { reason_json } => assert_eq!(reason_json, KICK_REASON),
                other => panic!("expected a kick reason, got {other:?}"),
            }
        }
        other => panic!("expected a kick, got {other:?}"),
    }
    let reason = events
        .iter()
        .find_map(|event| match event {
            ClientEvent::Disconnected(reason) => Some(reason),
            _ => None,
        })
        .expect("a Disconnected event");
    let component =
        mc_ui::font::component::ChatComponent::from_json(&reason.to_component_json()).unwrap();
    assert_eq!(
        component.unformatted_text(&lang),
        "multiplayer.disconnect.server_shutdown",
        "the reason renders as a component, which is what the disconnect screen needs"
    );
}

/// The server-list ping, over the same real socket, through the real
/// [`mc_client::net::ping`] — MOTD, version, player counts and a real 64×64 PNG favicon.
///
/// java: `OldServerPinger.tryCompatibilityPing`'s modern path, whose results populate
/// `ServerData`.
#[test]
fn server_list_ping_over_real_loopback_tcp() {
    // Build a genuine PNG here rather than committing a fixture, so the decode path is what is
    // under test rather than the provenance of a blob.
    let favicon = {
        let image =
            image::RgbaImage::from_fn(64, 64, |x, y| image::Rgba([x as u8, y as u8, 0x20, 0xFF]));
        let mut buffer = std::io::Cursor::new(Vec::new());
        image
            .write_to(&mut buffer, image::ImageFormat::Png)
            .expect("encode");
        format!(
            "data:image/png;base64,{}",
            base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                buffer.into_inner()
            )
        )
    };
    let json = format!(
        r#"{{"version":{{"name":"1.8.9","protocol":47}},
             "players":{{"max":100,"online":3,"sample":[{{"name":"Notch","id":"00000000-0000-0000-0000-000000000001"}}]}},
             "description":{{"text":"An ","extra":[{{"text":"MC","color":"gold"}}," Server"]}},
             "favicon":"{favicon}"}}"#
    );

    let runtime = NetRuntime::new().expect("runtime");
    let listener = runtime
        .handle()
        .block_on(TcpListener::bind("127.0.0.1:0"))
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();

    let response = json.clone();
    let server = runtime.handle().spawn(async move {
        use mc_protocol::status::{Ping, Pong, Request, Response};
        let (socket, _) = listener.accept().await.expect("accept");
        let mut peer = RefPeer::new(socket);

        let (id, body) = peer.read_packet().await;
        assert_eq!(id, Handshake::ID);
        let handshake: Handshake = decode(&body);
        assert_eq!(
            handshake.next_state,
            mc_protocol::handshake::NextState::Status
        );
        assert_eq!(handshake.protocol_version, 47);

        let (id, body) = peer.read_packet().await;
        assert_eq!(id, Request::ID);
        assert!(body.is_empty(), "the status Request has no body");
        peer.send(&Response { json: response }).await;

        let (id, body) = peer.read_packet().await;
        assert_eq!(id, Ping::ID);
        let ping: Ping = decode(&body);
        // The Pong must echo the payload verbatim; S07 treats a mismatch as an error.
        peer.send(&Pong {
            payload: ping.payload,
        })
        .await;
    });

    let ping = runtime
        .handle()
        .block_on(mc_client::net::ping("127.0.0.1", port))
        .expect("ping");
    runtime.handle().block_on(server).expect("server task");

    let lang = mc_assets::Language::new("en_US");
    assert_eq!(ping.version_name, "1.8.9");
    assert_eq!(ping.protocol, 47);
    assert_eq!(ping.online_players, Some(3));
    assert_eq!(ping.max_players, Some(100));
    assert_eq!(ping.sample.len(), 1);
    assert_eq!(ping.sample[0].name, "Notch");
    assert_eq!(ping.motd.unformatted_text(&lang), "An MC Server");
    // The gold `extra` must survive as a style, which is why the tree is kept rather than the
    // flattened string `ServerData.serverMOTD` holds.
    assert!(ping.motd.formatted_text(&lang).contains("\u{a7}6MC"));
    assert_eq!(ping.population_info(), "\u{a7}73\u{a7}8/\u{a7}7100");

    let icon = ping.favicon.expect("a 64x64 favicon");
    assert_eq!((icon.width, icon.height), (64, 64));
    assert_eq!(icon.rgba.len(), 64 * 64 * 4);
    // Row 0, column 5 was written as (5, 0, 0x20, 0xFF).
    assert_eq!(&icon.rgba[5 * 4..5 * 4 + 4], &[5, 0, 0x20, 0xFF]);
    assert!(ping.latency < Duration::from_secs(5));
    assert_eq!(ping.raw_json, json);
}

/// A server that closes the socket before the login write completes must produce
/// the login handler's vanilla generic I/O component, not a hang.
#[test]
fn a_server_that_hangs_up_during_login_reports_generic_io_component() {
    let runtime = NetRuntime::new().expect("runtime");
    let listener = runtime
        .handle()
        .block_on(TcpListener::bind("127.0.0.1:0"))
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();
    runtime.handle().spawn(async move {
        let (socket, _) = listener.accept().await.expect("accept");
        drop(socket);
    });

    let handle = runtime.connect(ConnectOptions::offline("127.0.0.1", port, "Dev"));
    let mut session = Session::new(handle, JoinSettings::default());
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        session.poll(64);
        if let Phase::Disconnected(reason) = session.phase() {
            match reason {
                DisconnectReason::WithOrigin { reason, origin } => {
                    assert_eq!(*origin, DisconnectOrigin::Login);
                    assert_eq!(
                        reason.to_component_json(),
                        serde_json::json!({
                            "translate": "disconnect.genericReason",
                            "with": ["Internal Exception: java.io.IOException"],
                        })
                        .to_string()
                    );
                }
                other => panic!("unexpected login disconnect reason: {other:?}"),
            }
            return;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    panic!("the session never reported a disconnect");
}

/// Nothing is listening at all: `Connection::connect` fails, and the failure must arrive as a
/// disconnect rather than as a panic inside the task.
#[test]
fn a_refused_connection_becomes_a_disconnect_event() {
    let runtime = NetRuntime::new().expect("runtime");
    // Bind then immediately drop, so the port is almost certainly closed.
    let port = {
        let listener = runtime
            .handle()
            .block_on(TcpListener::bind("127.0.0.1:0"))
            .expect("bind");
        listener.local_addr().expect("addr").port()
    };

    let handle = runtime.connect(ConnectOptions::offline("127.0.0.1", port, "Dev"));
    let mut session = Session::new(handle, JoinSettings::default());
    let deadline = Instant::now() + Duration::from_secs(15);
    while Instant::now() < deadline {
        let events = session.poll(64);
        if events
            .iter()
            .any(|event| matches!(event, ClientEvent::Disconnected(_)))
        {
            assert!(matches!(session.phase(), Phase::Disconnected(_)));
            return;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    panic!("a refused connection never reported a disconnect");
}
