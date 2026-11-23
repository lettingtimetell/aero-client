//! Shared rig for S21's headless integration tests.
//!
//! # What this is, stated before anything uses it
//!
//! S21's "done when" is *a fresh user can launch, authenticate, join a **public** 1.8.9 server,
//! and play*. **That is not achievable in this environment**: there is no network access, no
//! Minecraft server, and no display. The strongest available substitute — and what this file
//! builds — is:
//!
//! * a **loopback double**, not a server: a scripted peer on `127.0.0.1` whose *framing* is
//!   written here from the protocol-47 spec and shares no code with `mc_protocol`'s codec, but
//!   whose packet *bodies* are encoded with `mc-protocol`'s own `Packet::write`. It therefore
//!   proves the client's state machine, dispatch and application paths; it does **not** prove
//!   that a real vanilla server accepts these bytes. This is S07's and S14's precedent and their
//!   disclosure applies unchanged.
//! * **mocked authentication**: `ClientConfig`'s offline mode. `mc_auth`'s Microsoft device-code
//!   flow has never contacted Microsoft or the Mojang session server from this workspace, and
//!   nothing here changes that.
//! * a **real GPU**: `mc_render::Gpu::headless` on this machine resolves to a real adapter, so
//!   `PhaseList::render` and `read_pixels` are genuine draws and genuine readback — but offscreen
//!   only. No window has ever been opened by this project.
//!
//! Everything else is the real thing: the real 1.8.9 assets off disk, the real baked models and
//! stitched atlas, S11's mesher, S13's bit-exact physics, S15/S16/S17's screens, and S14's
//! connection task on a real TCP socket.

#![allow(dead_code)]

use std::time::{Duration, Instant};

use mc_client::app::config::LoginMode;
use mc_client::app::{Client, ClientConfig};
use mc_math::ChunkPos;
use mc_protocol::handshake::Handshake;
use mc_protocol::login::{LoginStart, LoginSuccess};
use mc_protocol::packet::Packet;
use mc_protocol::play::{c2s, s2c};
use mc_protocol::PacketBuffer;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, RenderError};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use uuid::Uuid;

/// The chunk the scripted server sends.
pub const CHUNK: ChunkPos = ChunkPos { x: 0, z: 0 };
/// The local Y of the stone floor inside section 0.
pub const FLOOR_Y: usize = 4;
/// `Block.BLOCK_STATE_IDS`: stone is block id 1, meta 0.
pub const STONE: u16 = 1 << 4;
/// Where the server puts the player: standing on top of the floor.
pub const SPAWN: (f64, f64, f64) = (8.5, (FLOOR_Y + 1) as f64, 8.5);
/// Looking down 45°, so the floor fills the lower half of the frame.
pub const SPAWN_PITCH: f32 = 45.0;
/// What the server says in chat.
pub const SERVER_CHAT: &str = r#"{"text":"<Notch> welcome","color":"white"}"#;
/// What the client says back.
pub const CLIENT_CHAT: &str = "hello from the rust client";
/// The block the player mines: the floor block directly under spawn.
pub const DIG_TARGET: mc_math::BlockPos = mc_math::BlockPos {
    x: 8,
    y: FLOOR_Y as i32,
    z: 8,
};

// -------------------------------------------------------------------------------------------
// An independently-framed peer. Nothing below this comment uses `mc_protocol::codec` or
// `mc_protocol::Connection` for framing.
// -------------------------------------------------------------------------------------------

/// The far side of the socket.
pub struct RefPeer {
    stream: TcpStream,
    buf: Vec<u8>,
    /// Every frame this peer has read, in order, so a test can assert on traffic it did not
    /// explicitly wait for — the movement cadence, for instance, which is interleaved with
    /// everything else.
    pub seen: Vec<(i32, Vec<u8>)>,
}

impl RefPeer {
    /// Wrap an accepted socket.
    pub fn new(stream: TcpStream) -> RefPeer {
        stream.set_nodelay(true).expect("nodelay");
        RefPeer {
            stream,
            buf: Vec::new(),
            seen: Vec::new(),
        }
    }

    async fn fill(&mut self, at_least: usize) {
        if !self.try_fill(at_least).await {
            let seen_ids: Vec<String> = self
                .seen
                .iter()
                .map(|(id, _)| format!("{id:#04x}"))
                .collect();
            panic!(
                "peer closed the connection mid-frame; need {at_least} bytes, have {}; seen ids {seen_ids:?}",
                self.buf.len(),
            );
        }
    }

    /// Fill, reporting a clean close rather than panicking on it.
    async fn try_fill(&mut self, at_least: usize) -> bool {
        while self.buf.len() < at_least {
            let mut chunk = [0u8; 8192];
            match self.stream.read(&mut chunk).await {
                Ok(0) | Err(_) => return false,
                Ok(read) => self.buf.extend_from_slice(&chunk[..read]),
            }
        }
        true
    }

    /// Read packets until the client hangs up, which is how a cycle ends.
    ///
    /// java: `NetworkManager.channelInactive`. A server does not get told a client is leaving;
    /// the socket simply closes.
    pub async fn drain_until_closed(&mut self) {
        loop {
            if !self.try_fill(1).await {
                return;
            }
            // A whole frame may not be here yet; read one and stop if the socket dies mid-frame.
            let length = self.read_varint().await as usize;
            if !self.try_fill(length).await {
                return;
            }
            self.buf.drain(..length);
        }
    }

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

    /// Try to read a VarInt while allowing a status-ping task to be canceled by the client.
    ///
    /// `GuiMultiplayer.connectToServer` clears pending pingers before opening the login socket.
    /// The independent loopback peer therefore has to treat EOF on a status socket as a normal
    /// close, while [`read_varint`](Self::read_varint) remains strict for the login/play script.
    async fn try_read_varint(&mut self) -> Option<i32> {
        let mut value: i32 = 0;
        let mut shift = 0;
        let mut index = 0usize;
        loop {
            if !self.try_fill(index + 1).await {
                return None;
            }
            let byte = self.buf[index];
            index += 1;
            value |= i32::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift >= 35 {
                return None;
            }
        }
        self.buf.drain(..index);
        Some(value)
    }

    /// One whole frame: length prefix, then packet id, then body.
    pub async fn read_packet(&mut self) -> (i32, Vec<u8>) {
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
        let body = frame[index..].to_vec();
        self.seen.push((id, body.clone()));
        (id, body)
    }

    /// Read one frame, returning `None` when a canceled status connection closes cleanly or
    /// halfway through its frame. This is intentionally separate from [`read_packet`], whose
    /// strict panic remains useful for the actual login/play protocol assertions below.
    async fn try_read_packet(&mut self) -> Option<(i32, Vec<u8>)> {
        let length = usize::try_from(self.try_read_varint().await?).ok()?;
        if length == 0 || !self.try_fill(length).await {
            return None;
        }
        let frame: Vec<u8> = self.buf.drain(..length).collect();

        let mut index = 0usize;
        let mut id: i32 = 0;
        let mut shift = 0;
        loop {
            let &byte = frame.get(index)?;
            index += 1;
            id |= i32::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift >= 35 {
                return None;
            }
        }
        let body = frame[index..].to_vec();
        self.seen.push((id, body.clone()));
        Some((id, body))
    }

    /// Read frames until the client hangs up, **recording every one** in
    /// [`RefPeer::seen`].
    ///
    /// [`RefPeer::drain_until_closed`] deliberately does not: it drains the byte buffer without
    /// ever splitting a frame into id and body, so `seen` stays empty. That is fine for the leak
    /// test, which only needs the socket serviced, and useless for a test that wants to assert
    /// on what the client *said* — which is what `input_dispatch.rs` needs, because "the screen
    /// changed" would pass for a client that changed the screen and told the server nothing.
    pub async fn record_until_closed(&mut self) {
        loop {
            if !self.try_fill(1).await {
                return;
            }
            let length = self.read_varint().await as usize;
            if !self.try_fill(length).await {
                return;
            }
            let frame: Vec<u8> = self.buf.drain(..length).collect();
            let mut index = 0usize;
            let mut id: i32 = 0;
            let mut shift = 0;
            loop {
                let Some(&byte) = frame.get(index) else {
                    return;
                };
                index += 1;
                id |= i32::from(byte & 0x7F) << shift;
                if byte & 0x80 == 0 {
                    break;
                }
                shift += 7;
            }
            self.seen.push((id, frame[index..].to_vec()));
        }
    }

    /// Every recorded packet with the given id.
    pub fn all<P: Packet>(&self) -> Vec<P> {
        self.seen
            .iter()
            .filter(|(id, _)| *id == P::ID)
            .map(|(_, body)| decode::<P>(body))
            .collect()
    }

    /// Read frames until one with `wanted_id` satisfies `predicate`, returning its body.
    pub async fn read_until(
        &mut self,
        wanted_id: i32,
        predicate: impl Fn(&[u8]) -> bool,
    ) -> Vec<u8> {
        for _ in 0..20_000 {
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

    /// Best-effort status response. A canceled pinger may close the socket between the request
    /// and this response; that is a normal Java lifecycle edge, so the listener must keep
    /// accepting the subsequent login connection instead of panicking here.
    async fn try_send_raw(&mut self, id: i32, body: &[u8]) -> bool {
        let mut payload = Vec::new();
        Self::write_varint(&mut payload, id);
        payload.extend_from_slice(body);

        let mut frame = Vec::new();
        Self::write_varint(&mut frame, payload.len() as i32);
        frame.extend_from_slice(&payload);
        self.stream.write_all(&frame).await.is_ok() && self.stream.flush().await.is_ok()
    }

    /// Encode a typed packet's *body* with `mc-protocol` and frame it here.
    pub async fn send<P: Packet>(&mut self, packet: &P) {
        let mut buffer = PacketBuffer::new();
        packet.write(&mut buffer);
        self.send_raw(P::ID, buffer.as_slice()).await;
    }

    /// Encode and send a packet, returning false if a canceled status peer has already gone away.
    async fn try_send<P: Packet>(&mut self, packet: &P) -> bool {
        let mut buffer = PacketBuffer::new();
        packet.write(&mut buffer);
        self.try_send_raw(P::ID, buffer.as_slice()).await
    }
}

/// Decode a packet body with `mc-protocol`.
pub fn decode<P: Packet>(bytes: &[u8]) -> P {
    let mut buffer = PacketBuffer::from_slice(bytes);
    P::read(&mut buffer).expect("packet body")
}

/// A chunk column with a full stone floor at [`FLOOR_Y`] in its lowest section.
///
/// `section_mask` selects which of the sixteen sections are present on the wire; only the lowest
/// one is given blocks, so a larger mask costs storage (which is what the leak check wants to
/// see move) without costing mesher time (which would only make that test slow).
pub fn floor_chunk(section_mask: u16) -> Vec<u8> {
    let length = mc_world::expected_chunk_data_len(section_mask, true, true);
    let mut data = vec![0u8; length];
    assert!(section_mask & 0b1 != 0, "section 0 carries the floor");
    for z in 0..16usize {
        for x in 0..16usize {
            let index = (FLOOR_Y << 8) | (z << 4) | x;
            data[index * 2] = (STONE & 0xFF) as u8;
            data[index * 2 + 1] = (STONE >> 8) as u8;
        }
    }
    // Full sky light in every section, so the mesher's brightness is not zero.
    let sections = usize::from(section_mask.count_ones() as u16);
    let sky_start = sections * (8192 + 2048);
    for byte in &mut data[sky_start..sky_start + sections * 2048] {
        *byte = 0xFF;
    }
    data
}

/// The status JSON the scripted peer answers a server-list ping with.
pub const STATUS_JSON: &str = r#"{"version":{"name":"1.8.9","protocol":47},
     "players":{"max":20,"online":1,"sample":[{"name":"Notch","id":"00000000-0000-0000-0000-000000000001"}]},
     "description":{"text":"Loopback double"}}"#;

/// Handshake → login → join → terrain → spawn, then hand the peer back for the test to script.
///
/// java: the opening of any real session — `NetHandlerLoginServer` then
/// `PlayerList.initializeConnectionToPlayer`.
///
/// **Answers server-list pings first.** A real server multiplexes status and login on one port,
/// and `MultiplayerScreen` pings every row as soon as it opens, so a peer that assumed the first
/// connection was a login would decode a status `Request` as a `LoginStart`. Getting this wrong
/// is what the first run of the smoke test did, and it is a genuine property of the client under
/// test rather than a quirk of the double.
pub async fn accept_and_join(listener: tokio::net::TcpListener, entity_id: i32) -> RefPeer {
    serve_join(&listener, entity_id, &[CHUNK], 0b1).await
}

/// As [`accept_and_join`], but on a borrowed listener and with a chosen set of chunks, so a test
/// can reuse one port across many connect/disconnect cycles and choose how much world arrives.
pub async fn serve_join(
    listener: &tokio::net::TcpListener,
    entity_id: i32,
    chunks: &[ChunkPos],
    section_mask: u16,
) -> RefPeer {
    use mc_protocol::handshake::NextState;
    use mc_protocol::status::{Ping, Pong, Request, Response};

    let mut peer = loop {
        let (socket, _) = listener.accept().await.expect("accept");
        let mut peer = RefPeer::new(socket);

        // A multiplayer-screen pinger can be canceled between TCP connect and its first request;
        // that socket is not the selected login attempt and must not tear down this listener.
        let Some((id, body)) = peer.try_read_packet().await else {
            continue;
        };
        assert_eq!(id, Handshake::ID);
        let handshake: Handshake = decode(&body);
        assert_eq!(handshake.protocol_version, 47, "protocol 47 only");

        if handshake.next_state == NextState::Login {
            break peer;
        }

        // java: `NetHandlerStatusServer` — Request/Response then Ping/Pong, echoing the payload.
        let Some((id, body)) = peer.try_read_packet().await else {
            continue;
        };
        assert_eq!(id, Request::ID);
        assert!(body.is_empty(), "the status Request has no body");
        if !peer
            .try_send(&Response {
                json: STATUS_JSON.to_owned(),
            })
            .await
        {
            continue;
        }
        let Some((id, body)) = tokio::time::timeout(Duration::from_secs(1), peer.try_read_packet())
            .await
            .ok()
            .flatten()
        else {
            continue;
        };
        assert_eq!(id, Ping::ID);
        let ping: Ping = decode(&body);
        if !peer
            .try_send(&Pong {
                payload: ping.payload,
            })
            .await
        {
            continue;
        }
    };

    let (id, body) = peer.read_packet().await;
    assert_eq!(id, LoginStart::ID);
    let login: LoginStart = decode(&body);

    peer.send(&LoginSuccess {
        uuid: Uuid::from_u128(0x0000_0000_0000_4000_8000_0000_0000_0001),
        username: login.username.clone(),
    })
    .await;

    peer.send(&s2c::JoinGame {
        entity_id,
        hardcore_mode: false,
        game_mode: 0,
        dimension: 0,
        difficulty: 2,
        max_players: 20,
        level_type: "default".into(),
        reduced_debug_info: false,
    })
    .await;

    // java: `handleJoinGame` answers with C15PacketClientSettings then the MC|Brand payload.
    let (id, _) = peer.read_packet().await;
    assert_eq!(id, c2s::ClientSettings::ID);
    let (id, _) = peer.read_packet().await;
    assert_eq!(id, c2s::PluginMessage::ID);

    for chunk in chunks {
        peer.send(&mc_protocol::chunk::ChunkData {
            chunk_x: chunk.x,
            chunk_z: chunk.z,
            ground_up_continuous: true,
            primary_bit_mask: section_mask,
            data: floor_chunk(section_mask),
        })
        .await;
    }

    peer.send(&s2c::PlayerPositionLook {
        x: SPAWN.0,
        y: SPAWN.1,
        z: SPAWN.2,
        yaw: 0.0,
        pitch: SPAWN_PITCH,
        flags: 0,
    })
    .await;

    // The client must confirm before the server accepts anything it says.
    peer.read_until(c2s::PlayerPositionAndLook::ID, |body| {
        let packet: c2s::PlayerPositionAndLook = decode(body);
        (packet.x - SPAWN.0).abs() < 1e-9
    })
    .await;

    peer
}

// -------------------------------------------------------------------------------------------
// The client side
// -------------------------------------------------------------------------------------------

/// A headless GPU, or `None` when this machine has no adapter.
pub fn gpu(width: u32, height: u32) -> Option<Gpu> {
    match Gpu::headless(
        width,
        height,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => Some(gpu),
        Err(RenderError::NoAdapter) => {
            eprintln!("skip: no wgpu adapter available on this machine");
            None
        }
        Err(error) => panic!("headless context creation failed: {error}"),
    }
}

/// A config pointed at the repository's real 1.8.9 assets, with a scratch game directory and
/// offline-mode credentials.
pub fn config(game_dir: &std::path::Path, username: &str) -> Option<ClientConfig> {
    let config = ClientConfig {
        game_dir: game_dir.to_path_buf(),
        login: LoginMode::Offline(username.to_owned()),
        ..ClientConfig::default()
    }
    .with_discovered_assets();
    config.version_jar.as_ref()?.is_file().then_some(config)
}

/// Start the client in `game_dir` on vanilla's title screen: write a mod settings file with the
/// Aero title screen (`aero_title`, on by default) switched off, the way
/// `modwave_m3_controls.rs` switches a mod on. For tests that click, draw or byte-compare the
/// vanilla title screen. Call it before `Client::new`, on a scratch game dir only.
pub fn vanilla_title_screen(game_dir: &std::path::Path) {
    std::fs::write(
        game_dir.join(mc_mods::SETTINGS_FILE_NAME),
        r#"{"format":1,"mods":{"aero_title":{"enabled":false}}}"#,
    )
    .expect("write the scratch mod settings");
}

/// Poll, tick and render one frame, exactly as the shell would.
///
/// java: one iteration of `Minecraft.runGameLoop` — `processReceivedPackets`, `runTick`, then
/// the render half.
pub fn pump(
    client: &mut Client,
    gpu: &mut Gpu,
    phases: &mut PhaseList,
    camera: &mut Camera,
    input: &mc_physics::state::MoveInput,
) {
    client.poll();
    client.tick(input);
    client.frame(gpu, camera, 0.0);
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), camera, 0.0);
}

/// Poll and render one frame **without** advancing a game tick.
///
/// The main menu's panorama (S21 -> S16 §13's wiring: `Client::build_ui` now passes it a real,
/// nonzero step count) advances once per [`Client::tick`], the same way vanilla's rotates
/// continuously while sitting at the menu. A pixel-for-pixel comparison across something that is
/// not itself a tick — a resource-pack reload, chief among the callers of this — must hold that
/// animation still or the frames legitimately differ for a reason that has nothing to do with
/// what is being tested. `client.poll()` is still called: it drains async job results (the login
/// flow, a ping) and is inert with none in flight, so nothing about "reload while idle" changes
/// by keeping it.
pub fn pump_no_tick(
    client: &mut Client,
    gpu: &mut Gpu,
    phases: &mut PhaseList,
    camera: &mut Camera,
) {
    client.poll();
    client.frame(gpu, camera, 0.0);
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), camera, 0.0);
}

/// Pump until `done` returns true or the deadline passes.
#[allow(clippy::too_many_arguments)]
pub fn pump_until(
    client: &mut Client,
    gpu: &mut Gpu,
    phases: &mut PhaseList,
    camera: &mut Camera,
    input: &mc_physics::state::MoveInput,
    timeout: Duration,
    what: &str,
    mut done: impl FnMut(&Client) -> bool,
) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        pump(client, gpu, phases, camera, input);
        if done(client) {
            return;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let session_state = client.session().map(|session| {
        (
            session.world().is_chunk_loaded(CHUNK),
            session.predictor().map(|predictor| predictor.state.pos),
            session.state().local.done_loading_terrain,
        )
    });
    panic!(
        "timed out waiting for {what}; machine={:?}, ui={}, session_phase={:?}, session_state={session_state:?}, last_events={:?}",
        client.machine.state(),
        client.ui.name(),
        client.session().map(|session| session.phase()),
        client.last_events,
    );
}

/// `PhaseList`'s default clear colour, rounded into `Rgba8Unorm` bytes: `0.043, 0.070, 0.145`.
pub const CLEAR_RGB: [u8; 3] = [11, 18, 37];

/// How many pixels differ from the clear colour, as a crude "something was drawn" measure.
pub fn drawn_pixels(pixels: &[u8], clear: [u8; 3]) -> usize {
    pixels
        .chunks_exact(4)
        .filter(|p| p[0] != clear[0] || p[1] != clear[1] || p[2] != clear[2])
        .count()
}
