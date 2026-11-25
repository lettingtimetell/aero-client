//! The 1.7.10 (protocol 5) connection: open, log in, then pump the full-duplex `Play` stream.
//!
//! Declared inside [`crate::net::task`], whose `run` selects it once from
//! `ConnectOptions.version`. It reuses that module's helpers (the read-activity deadline, the
//! disconnect reasons, the ordered spill FIFO) and leaves its 1.8.9 pump untouched: a 1.8.9
//! connection never runs a line of this file.
//!
//! java17: `network/NetworkManager.java` (the pipeline and `channelRead0`), the priority handlers
//! of `client/network/NetHandlerPlayClient.java`.
//!
//! # What differs from the 1.8.9 connection
//!
//! | | 1.7.10 | here |
//! |---|---|---|
//! | Socket | `IP_TOS` 24 and `TCP_NODELAY` off, each set in a `try` that ignores a refusal (`NetworkManager.provideLanClient`, `:239-262`) | the same, through tokio's `TcpSocket` |
//! | Read timeout | `ReadTimeoutHandler(20)` (`:262`) | [`READ_TIMEOUT_V5`] |
//! | Compression | none: no codec exists, login `0x03` and play `0x46` are unknown ids | never enabled |
//! | Handled on the network thread | only `hasPriority()` packets: `S00 KeepAlive` and `S40 Disconnect` (`channelRead0`, `:103-116`) | the same; nothing else bypasses the queue |
//! | Everything else | queued, drained by the tick at most 1001 at a time (`processReceivedPackets`, `:188-215`) | [`ToGame::PacketV5`] (the session queues it for its tick) |
//! | Chunk packets | inflated in `readPacketData` on the network thread | inflated by the decoder, then converted to the 1.8.9 form ([`crate::net::chunk_v5`]) and unpacked in the background like a 1.8.9 column |
//!
//! Outbound, a [`ToNet::SendV5`] is written exactly as built; a [`ToNet::Send`] from code shared
//! with 1.8.9 goes through [`C2SPlay5::from_v47`], and one with no exact protocol-5 form is
//! logged and dropped, never approximated.
//!
//! A server close, a read timeout or a decode error ends the connection with the same reasons as
//! 1.8.9 (`NetworkManager.exceptionCaught` and `channelInactive` are the same in both). Packets
//! still queued for the tick when it ends are never applied: `PlayerControllerMP.updateController`
//! checks `isChannelOpen()` before draining (`:258-269`), so here no spill is flushed either.

use std::collections::VecDeque;
use std::time::Duration;

use mc_protocol::connection::{handshake_for, Connection, ConnectionReader, ConnectionWriter};
use mc_protocol::connection_v5::{login_v5_offline, login_v5_with_auth_policy};
use mc_protocol::handshake::NextState;
use mc_protocol::v5::{c2s, s2c, C2SPlay5, S2CPlay5, V5Ctx};
use mc_protocol::{PacketBuffer, ProtoError};
use mc_ui::font::component::{ChatComponent, ComponentKind};
use mc_version::GameVersion;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::mpsc::{Sender, UnboundedReceiver};

use super::{
    await_with_read_timeout, component_reason, deadline_after_activity, disconnect_reason_for,
    enqueue_scheduled, generic_exception_reason, login_failure_reason, malformed_component_reason,
    open_failure_reason, timeout_reason, ActivityInstant, ConnectOptions, Credentials, OpenError,
    ReadActivity, ReadTimedOut, READ_TIMEOUT,
};
use crate::error::ClientError;
use crate::net::capture::CaptureSink;
use crate::net::channel::{DisconnectOrigin, DisconnectReason, ToGame, ToNet};

/// How long the server may go silent. java17: `new ReadTimeoutHandler(20)`
/// (`network/NetworkManager.java:262`); 1.8.9 waits 30 s.
pub const READ_TIMEOUT_V5: Duration = Duration::from_secs(20);

/// How long the TCP connect may take: Netty's default connect timeout, which 1.7.10's bootstrap
/// does not override (`NetworkManager.provideLanClient`), the same 30 s as 1.8.9's.
const CONNECT_TIMEOUT_V5: Duration = READ_TIMEOUT;

/// java17: `IP_TOS` 24 (`network/NetworkManager.java:246`).
const IP_TOS_V5: u32 = 24;

/// [`super::run`] for a 1.7.10 connection.
pub(super) async fn run(
    opts: ConnectOptions,
    to_game: Sender<ToGame>,
    from_game: UnboundedReceiver<ToNet>,
    raw_tap: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    let address = format!("{}:{}", opts.host, opts.port);
    let protocol = GameVersion::V1_7_10.protocol();
    let mut capture = match &opts.capture {
        Some(path) => CaptureSink::to_file_for_protocol(path.clone(), &address, protocol),
        None => CaptureSink::from_env_for_protocol(&address, protocol),
    };
    // The replay recorder's tap, as on a 1.8.9 connection.
    capture.set_raw_tap(raw_tap);
    tracing::info!(host = %opts.host, port = opts.port, version = "1.7.10", "connecting");
    let (reason, origin) = match connect_and_login(
        &opts,
        &to_game,
        &mut capture,
        CONNECT_TIMEOUT_V5,
        READ_TIMEOUT_V5,
    )
    .await
    {
        Ok((conn, inbound_activity)) => {
            capture.set_state(mc_protocol::State::Play);
            let (reader, writer) = conn.split();
            let reason = play_loop(
                reader,
                writer,
                &to_game,
                from_game,
                inbound_activity,
                READ_TIMEOUT_V5,
                &mut capture,
            )
            .await;
            (reason, DisconnectOrigin::Play)
        }
        Err(reason) => (reason, DisconnectOrigin::Login),
    };
    capture.finish();
    let _ = to_game
        .send(ToGame::Disconnected(reason.with_origin(origin)))
        .await;
}

/// TCP connect with 1.7.10's socket options → `ToGame::SocketConnected` → handshake (protocol
/// 5, next = Login) → the protocol-5 login → `ToGame::Connected`. As 1.8.9's
/// `connect_and_login_with_timeouts`, with separate connect and read budgets.
async fn connect_and_login(
    opts: &ConnectOptions,
    to_game: &Sender<ToGame>,
    capture: &mut CaptureSink,
    connect_budget: Duration,
    read_budget: Duration,
) -> Result<
    (
        Connection<ReadActivity<tokio::net::TcpStream>>,
        UnboundedReceiver<ActivityInstant>,
    ),
    DisconnectReason,
> {
    let connected = match tokio::time::timeout(connect_budget, open_connection(opts)).await {
        Ok(Ok(connected)) => connected,
        Ok(Err(error)) => return Err(open_failure_reason(&error, opts)),
        Err(_) => return Err(timeout_reason()),
    };
    to_game
        .send(ToGame::SocketConnected)
        .await
        .map_err(|_| disconnect_reason_for(&ClientError::NotConnected))?;

    let (activity_tx, mut activity_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut conn = Connection::new(ReadActivity::new(connected, activity_tx));
    let login_sequence = async {
        handshake_for(
            &mut conn,
            &opts.host,
            opts.port,
            NextState::Login,
            GameVersion::V1_7_10,
        )
        .await?;
        capture.set_state(mc_protocol::State::Login);

        let mut observe = |direction: mc_protocol::connection::Direction, payload: &[u8]| {
            capture.record_login(direction, payload);
        };
        let outcome = match &opts.credentials {
            Credentials::Online(session) => {
                let session: &mc_auth::Session = session;
                let join = |server_hash: String| async move {
                    opts.session_client()?
                        .join_server(session, &server_hash)
                        .await
                        .map_err(ProtoError::from)
                };
                login_v5_with_auth_policy(
                    &mut conn,
                    &session.username,
                    join,
                    Some(&mut observe),
                    opts.auth_policy,
                )
                .await?
            }
            Credentials::Offline { username } => {
                login_v5_offline(&mut conn, username, Some(&mut observe)).await?
            }
        };
        Ok::<_, ClientError>(outcome)
    };
    let outcome = match await_with_read_timeout(login_sequence, &mut activity_rx, read_budget).await
    {
        Ok(result) => result.map_err(|err| login_failure_reason(&err, opts))?,
        Err(ReadTimedOut) => return Err(timeout_reason()),
    };
    to_game
        .send(ToGame::Connected(outcome))
        .await
        .map_err(|_| disconnect_reason_for(&ClientError::NotConnected))?;
    Ok((conn, activity_rx))
}

/// Resolve once and connect to the first IPv4 address, as 1.8.9's `open_connection`, with
/// 1.7.10's socket options. java17: `NetworkManager.provideLanClient`
/// (`network/NetworkManager.java:239-262`) sets `IP_TOS` 24, then `TCP_NODELAY` false, each in
/// a `try` that ignores a refusal; then connects.
async fn open_connection(opts: &ConnectOptions) -> Result<tokio::net::TcpStream, OpenError> {
    let addresses = tokio::net::lookup_host((opts.host.as_str(), opts.port))
        .await
        .map_err(|_| OpenError::UnknownHost)?;
    let resolved = crate::net::first_ipv4_address(addresses).ok_or(OpenError::UnknownHost)?;
    let connect_error = |error: std::io::Error| OpenError::Connect {
        error: ProtoError::Io(error),
        resolved,
    };
    let socket = tokio::net::TcpSocket::new_v4().map_err(connect_error)?;
    configure_socket(&socket);
    socket.connect(resolved).await.map_err(connect_error)
}

/// 1.7.10's socket options, in its order, each refusal ignored: `IP_TOS` 24, then `TCP_NODELAY`
/// false (Nagle's algorithm on).
fn configure_socket(socket: &tokio::net::TcpSocket) {
    if let Err(error) = socket.set_tos_v4(IP_TOS_V5) {
        tracing::debug!(%error, "IP_TOS refused");
    }
    if let Err(error) = socket.set_nodelay(false) {
        tracing::debug!(%error, "TCP_NODELAY refused");
    }
}

/// The protocol-5 `Play` pump: [`super::play_loop`]'s structure (the read-activity deadline, the
/// biased select preferring outbound packets, the ordered spill FIFO) with 1.7.10's routing.
///
/// # Cancel-safety
///
/// As [`super::play_loop`]: `read_payload` is a `select!` arm, `send_payload` only runs inside an
/// arm's body, so a frame is never abandoned half written.
pub(super) async fn play_loop<R, W>(
    mut reader: ConnectionReader<R>,
    mut writer: ConnectionWriter<W>,
    to_game: &Sender<ToGame>,
    mut from_game: UnboundedReceiver<ToNet>,
    mut inbound_activity: UnboundedReceiver<ActivityInstant>,
    timeout_budget: Duration,
    capture: &mut CaptureSink,
) -> DisconnectReason
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut pending_to_game = VecDeque::new();
    let mut read_deadline = deadline_after_activity(ActivityInstant::now(), timeout_budget);
    // The dimension's sky flag chunk columns are laid out with (an overworld until a JoinGame
    // says otherwise), and the local entity id a 1.8.9-built C0A carries; both follow
    // JoinGame/Respawn in stream order.
    let mut chunk_sky = true;
    let mut ctx = V5Ctx { entity_id: 0 };

    loop {
        tokio::select! {
            biased;

            Some(activity_at) = inbound_activity.recv() => {
                let mut latest_activity = activity_at;
                while let Ok(next_activity) = inbound_activity.try_recv() {
                    latest_activity = latest_activity.max(next_activity);
                }
                read_deadline = deadline_after_activity(latest_activity, timeout_budget);
            }

            () = tokio::time::sleep_until(read_deadline) => return timeout_reason(),

            command = from_game.recv() => match command {
                Some(ToNet::SendV5(packet)) => {
                    if let Err(err) = send(&mut writer, &packet, capture).await {
                        return disconnect_reason_for(&err);
                    }
                }
                Some(ToNet::Send(packet)) => match C2SPlay5::from_v47(&packet, &ctx) {
                    Ok(packet) => {
                        if let Err(err) = send(&mut writer, &packet, capture).await {
                            return disconnect_reason_for(&err);
                        }
                    }
                    Err(refused) => {
                        tracing::error!(%refused, "no exact 1.7.10 form for this packet; not sent");
                    }
                },
                Some(ToNet::Disconnect) | None => return DisconnectReason::ClosedByClient,
            },

            permit = to_game.reserve(), if !pending_to_game.is_empty() => match permit {
                Ok(permit) => permit.send(
                    pending_to_game
                        .pop_front()
                        .expect("reserve arm is guarded by a non-empty FIFO"),
                ),
                Err(_) => return DisconnectReason::ClosedByClient,
            },

            read = reader.read_payload() => {
                let mut payload = match read {
                    Ok(payload) => payload,
                    Err(err) => return disconnect_reason_for(&ClientError::from(err)),
                };
                capture.record(mc_parity::Direction::Clientbound, payload.as_slice());
                let raw = capture.tapped(payload.as_slice());
                let packet = match decode(&mut payload) {
                    Ok(packet) => packet,
                    Err(err) => return disconnect_reason_for(&err),
                };
                if let Err(reason) = validate_components(&packet) {
                    return reason;
                }

                match packet {
                    // --- the two priority handlers 1.7.10 runs on the network thread ---
                    S2CPlay5::KeepAlive(keep_alive) => {
                        // java17: NetHandlerPlayClient.handleKeepAlive (:1368-1371).
                        let echo = C2SPlay5::KeepAlive(c2s::KeepAlive { key: keep_alive.id });
                        if let Err(err) = send(&mut writer, &echo, capture).await {
                            return disconnect_reason_for(&err);
                        }
                    }
                    S2CPlay5::Disconnect(s2c::Disconnect { reason }) => {
                        return disconnect_reason(&reason);
                    }

                    // --- everything else waits for the tick ---
                    other => {
                        match &other {
                            S2CPlay5::JoinGame(join) => {
                                chunk_sky = crate::sync::state::dimension_has_sky_light(
                                    i32::from(join.dimension),
                                );
                                ctx.entity_id = join.entity_id;
                            }
                            S2CPlay5::Respawn(respawn) => {
                                chunk_sky =
                                    crate::sync::state::dimension_has_sky_light(respawn.dimension);
                            }
                            _ => {}
                        }
                        let message =
                            crate::net::chunk_v5::message_v5(other, chunk_sky).with_raw_v5(raw);
                        if enqueue_scheduled(to_game, &mut pending_to_game, message).is_err() {
                            return DisconnectReason::ClosedByClient;
                        }
                    }
                }
            }
        }
    }
}

/// Serialise one protocol-5 packet (id VarInt, then body) and write it, recording it as 1.8.9's
/// `send` does.
async fn send<W: AsyncWrite + Unpin>(
    writer: &mut ConnectionWriter<W>,
    packet: &C2SPlay5,
    capture: &mut CaptureSink,
) -> Result<(), ClientError> {
    let mut payload = PacketBuffer::new();
    payload.write_varint(packet.id());
    packet.write(&mut payload);
    payload.finish_write()?;
    capture.record(mc_parity::Direction::Serverbound, payload.as_slice());
    writer.send_payload(payload.as_slice()).await?;
    Ok(())
}

/// Split the id off a payload and decode it as protocol 5: an unknown id (including 1.8.9's
/// `0x41`-`0x49`) and leftover bytes are fatal, as in both versions
/// (java17 `util/MessageDeserializer.java:42-50`).
fn decode(payload: &mut PacketBuffer) -> Result<S2CPlay5, ClientError> {
    let id = payload.read_varint()?;
    let packet = S2CPlay5::read(id, payload)?;
    if !payload.is_empty() {
        return Err(ProtoError::TrailingBytes {
            packet_id: id,
            remaining: payload.remaining(),
        }
        .into());
    }
    Ok(packet)
}

/// The protocol-5 fields read with `IChatComponent.Serializer` while decoding: only `S02`'s
/// message and `S40`'s reason (java17 `S02PacketChat.java:33`, `S40PacketDisconnect`). Every
/// other text field of protocol 5 is a plain string. A field the deserializer rejects is a
/// decoder exception, which closes the connection; the reason it closes with is returned.
///
/// Malformed JSON fails as in 1.8.9. Beyond that, java17's deserializer
/// (`util/IChatComponent.java:44-144`) builds only text and translation components: an object
/// with neither `text` nor `translate`, such as 1.8's `score` and `selector` components, throws
/// `JsonParseException` ("Don't know how to turn ... into a Component"), also inside
/// translation arguments, `extra` siblings and hover texts, which it deserializes the same way.
fn validate_components(packet: &S2CPlay5) -> Result<(), DisconnectReason> {
    let json = match packet {
        S2CPlay5::Chat(chat) => &chat.component,
        S2CPlay5::Disconnect(disconnect) => &disconnect.reason,
        _ => return Ok(()),
    };
    match ChatComponent::from_json_nullable(json) {
        Err(error) => Err(malformed_component_reason(&error)),
        Ok(Some(component)) if !is_component_v5(&component) => Err(generic_exception_reason(
            "com.google.gson.JsonParseException",
        )),
        Ok(_) => Ok(()),
    }
}

/// Whether java17's `IChatComponent.Serializer` could have built `component`: text and
/// translations only, all the way down.
fn is_component_v5(component: &ChatComponent) -> bool {
    let own = match &component.kind {
        ComponentKind::Text(_) => true,
        ComponentKind::Translate { with, .. } => with.iter().all(is_component_v5),
        ComponentKind::Score { .. } | ComponentKind::Selector(_) => false,
    };
    own && component.extra.iter().all(is_component_v5)
        && component
            .style
            .hover_event
            .as_ref()
            .is_none_or(|hover| is_component_v5(&hover.value))
}

/// The reason an `S40` ends the connection with.
///
/// java17: `handleDisconnect` closes the channel with the packet's component
/// (`NetHandlerPlayClient.java:627-630`). A Gson-null component leaves no exit message, and
/// `updateController` then shows `"Disconnected from server"` (`PlayerControllerMP.java:262-268`);
/// 1.8.9 says `"Disconnected"` there.
fn disconnect_reason(reason_json: &str) -> DisconnectReason {
    match ChatComponent::from_json_nullable(reason_json) {
        Ok(Some(_)) => DisconnectReason::Kicked {
            reason_json: reason_json.to_owned(),
        },
        Ok(None) => component_reason(serde_json::json!({ "text": "Disconnected from server" })),
        Err(error) => malformed_component_reason(&error),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use mc_protocol::handshake::Handshake;
    use mc_protocol::login::{LoginStart, LoginSuccess};
    use mc_protocol::packet::Packet;
    use mc_protocol::v5::chunk::MapChunkBulk;
    use mc_protocol::v5::s2c;

    use super::*;
    use crate::net::NetRuntime;
    use crate::session::{JoinSettings, Phase, Session};
    use crate::sync::ClientEvent;

    /// One protocol-5 clientbound packet as a frame payload: id VarInt, then the body.
    fn payload(packet: &S2CPlay5) -> Vec<u8> {
        let mut buf = PacketBuffer::new();
        buf.write_varint(packet.id());
        packet.write(&mut buf);
        buf.as_slice().to_vec()
    }

    fn login_payload<P: Packet>(packet: &P) -> Vec<u8> {
        let mut buf = PacketBuffer::new();
        buf.write_varint(P::ID);
        packet.write(&mut buf);
        buf.as_slice().to_vec()
    }

    /// The next serverbound payload with packet id `id`, skipping (and keeping) the others.
    async fn read_until(
        conn: &mut Connection<tokio::net::TcpStream>,
        id: u8,
        seen: &mut Vec<Vec<u8>>,
    ) -> Vec<u8> {
        loop {
            let frame = conn.read_payload().await.expect("client frame");
            let bytes = frame.as_slice().to_vec();
            seen.push(bytes.clone());
            if bytes.first() == Some(&id) {
                return bytes;
            }
        }
    }

    /// A stone floor in section 4 of column (0, 0) as a protocol-5 bulk, with sky light.
    fn stone_column_bulk() -> MapChunkBulk {
        let mut raw = vec![0u8; 4096];
        raw[..256].fill(1); // y = 64: stone
        raw.extend(std::iter::repeat_n(0u8, 2048)); // metadata
        raw.extend(std::iter::repeat_n(0u8, 2048)); // block light
        raw.extend(std::iter::repeat_n(0xFFu8, 2048)); // sky light
        raw.extend(std::iter::repeat_n(1u8, 256)); // biomes
        MapChunkBulk::from_raw(true, &[(0, 0, 1 << 4, 0, &raw)]).expect("bulk")
    }

    /// A 1.7.10 server over real TCP, through the real connection task and session: handshake
    /// 5, the protocol-5 login, join, a bulk chunk, a keep-alive answered on the network side
    /// with the same int, an S08 answered from the tick with 1.7.10's exact C06, then a kick.
    #[test]
    fn a_1_7_10_connection_logs_in_plays_and_ends() {
        let server_rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .expect("server runtime");
        let listener = server_rt
            .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
            .expect("loopback listener");
        let port = listener.local_addr().expect("address").port();

        let server = server_rt.spawn(async move {
            let (socket, _) = listener.accept().await.expect("accept");
            let mut conn = Connection::new(socket);
            let (id, mut body) = conn.read_packet().await.expect("handshake");
            assert_eq!(id, Handshake::ID);
            let handshake = Handshake::read(&mut body).expect("decode handshake");
            assert_eq!(handshake.protocol_version, 5);
            assert_eq!(handshake.next_state, NextState::Login);
            let (id, mut body) = conn.read_packet().await.expect("login start");
            assert_eq!(id, LoginStart::ID);
            assert_eq!(
                LoginStart::read(&mut body).expect("decode").username,
                "Tester"
            );
            let success = LoginSuccess {
                uuid: uuid::Uuid::from_u128(0x5eed),
                username: "Tester".to_owned(),
            };
            conn.send_payload(&login_payload(&success))
                .await
                .expect("send");

            let join = S2CPlay5::JoinGame(s2c::JoinGame {
                entity_id: 77,
                hardcore_mode: false,
                game_mode: 0,
                dimension: 0,
                difficulty: 1,
                max_players: 20,
                level_type: "flat".to_owned(),
            });
            conn.send_payload(&payload(&join)).await.expect("send");
            let bulk = S2CPlay5::MapChunkBulk(stone_column_bulk());
            conn.send_payload(&payload(&bulk)).await.expect("send");
            let keep_alive = S2CPlay5::KeepAlive(s2c::KeepAlive { id: 0x1234_5678 });
            conn.send_payload(&payload(&keep_alive))
                .await
                .expect("send");

            let mut seen = Vec::new();
            let echo = read_until(&mut conn, 0x00, &mut seen).await;

            let look = S2CPlay5::PlayerPosLook(s2c::PlayerPosLook {
                x: 8.5,
                y: 65.620_000_004_768_37,
                z: 8.5,
                yaw: 400.0,
                pitch: -100.0,
                on_ground: true,
            });
            conn.send_payload(&payload(&look)).await.expect("send");
            let reply = read_until(&mut conn, 0x06, &mut seen).await;

            let kick = S2CPlay5::Disconnect(s2c::Disconnect {
                reason: r#"{"text":"Bye"}"#.to_owned(),
            });
            conn.send_payload(&payload(&kick)).await.expect("send");
            (echo, reply, seen)
        });

        let net = NetRuntime::new().expect("client runtime");
        let opts =
            ConnectOptions::offline("127.0.0.1", port, "Tester").with_version(GameVersion::V1_7_10);
        let mut session = Session::with_version(
            net.connect(opts),
            JoinSettings::default(),
            GameVersion::V1_7_10,
        );
        let mut events = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline && session.phase().is_live() {
            // The frame poll: lifecycle messages are handled, play messages only queued.
            session.begin_poll();
            while session.poll_next(&mut events) {}
            // The tick's drain, as the client runs it.
            if session.drains_v5_before_world() || session.drains_v5_in_world() {
                for _ in 0..crate::session::v5::PACKETS_PER_DRAIN {
                    if !session.apply_next_v5(&mut events) {
                        break;
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(2));
        }

        let (echo, reply, seen) = server_rt.block_on(server).expect("server script");
        // The keep-alive echo: C00 with the int id.
        assert_eq!(echo, [0x00, 0x12, 0x34, 0x56, 0x78]);
        // java17 handlePlayerPosLook's C06: posX, box bottom, eye stance, posZ, the packet's yaw,
        // pitch and on-ground flag; the box bottom is the eye less (double)1.62F, exactly 64.
        let mut expected = vec![0x06];
        expected.extend_from_slice(&8.5f64.to_be_bytes());
        expected.extend_from_slice(&64.0f64.to_be_bytes());
        expected.extend_from_slice(&65.620_000_004_768_37f64.to_be_bytes());
        expected.extend_from_slice(&8.5f64.to_be_bytes());
        expected.extend_from_slice(&400.0f32.to_be_bytes());
        expected.extend_from_slice(&(-100.0f32).to_be_bytes());
        expected.push(1);
        assert_eq!(reply, expected);
        // Nothing in 1.8.9 form reached the server (C15/C17 here are 1.7.10's, or refused).
        for frame in &seen {
            let mut buf = PacketBuffer::from_slice(frame);
            let id = buf.read_varint().expect("id");
            assert!(
                C2SPlay5::read(id, &mut buf).is_ok(),
                "not protocol 5: {frame:02x?}"
            );
            assert!(buf.is_empty(), "trailing bytes: {frame:02x?}");
        }

        // The join, the chunk and the S08 were applied, in the tick.
        assert!(events
            .iter()
            .any(|event| matches!(event, ClientEvent::JoinedGame(_))));
        assert!(events
            .iter()
            .any(|event| matches!(event, ClientEvent::TerrainDownloaded)));
        assert_eq!(
            session
                .state()
                .world
                .block_at(mc_math::BlockPos::new(3, 64, 5)),
            mc_data::v5::state_from_v5(1 << 4)
        );
        assert_eq!(session.state().local.entity_id, 77);
        // The kick ended it with the server's reason.
        let Phase::Disconnected(reason) = session.phase() else {
            panic!("still connected: {:?}", session.phase());
        };
        assert_eq!(reason.to_component_json(), r#"{"text":"Bye"}"#);
    }

    /// The play pump over an in-memory duplex, with the far end returned.
    fn rig() -> (
        crate::net::NetHandle,
        Connection<tokio::io::DuplexStream>,
        tokio::task::JoinHandle<DisconnectReason>,
    ) {
        let (a, b) = tokio::io::duplex(1 << 20);
        let (handle, to_game, _immediate, from_game) = crate::net::NetHandle::new_live_pair();
        let (activity_tx, activity_rx) = tokio::sync::mpsc::unbounded_channel();
        let (reader, writer) = Connection::new(ReadActivity::new(a, activity_tx)).split();
        let task = tokio::spawn(async move {
            play_loop(
                reader,
                writer,
                &to_game,
                from_game,
                activity_rx,
                READ_TIMEOUT_V5,
                &mut CaptureSink::disabled(),
            )
            .await
        });
        (handle, Connection::new(b), task)
    }

    /// Outbound: a protocol-5 packet is written as built; a 1.8.9 packet with an exact
    /// protocol-5 form is converted (C01 chat, C0A with the S01's entity id); one without
    /// (a 1.8.9 C15) is refused, nothing written, and the order of the rest is kept.
    #[tokio::test]
    async fn outbound_packets_are_converted_or_refused_never_approximated() {
        use mc_protocol::play::{c2s as c2s47, C2SPlay};

        let (mut handle, mut server, task) = rig();
        // The S01 tells the pump the entity id a 1.8.9-built C0A carries.
        let join = S2CPlay5::JoinGame(s2c::JoinGame {
            entity_id: 42,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 1,
            max_players: 20,
            level_type: "default".to_owned(),
        });
        server.send_payload(&payload(&join)).await.expect("send");
        loop {
            if let Some(ToGame::PacketV5(_)) = handle.try_recv() {
                break;
            }
            tokio::task::yield_now().await;
        }

        handle
            .send(C2SPlay::ClientSettings(c2s47::ClientSettings {
                locale: "en_US".into(),
                view_distance: 8,
                chat_visibility: c2s47::ChatVisibility::Full,
                chat_colors: true,
                displayed_skin_parts: 0x7F,
            }))
            .expect("queued");
        handle
            .send(C2SPlay::ChatMessage(c2s47::ChatMessage {
                message: "hi".into(),
            }))
            .expect("queued");
        handle
            .send(C2SPlay::Animation(c2s47::Animation))
            .expect("queued");
        handle
            .send_v5(C2SPlay5::HeldItemChange(c2s::HeldItemChange { slot_id: 3 }))
            .expect("queued");

        let mut chat = PacketBuffer::new();
        chat.write_varint(0x01);
        chat.write_string("hi");
        assert_eq!(
            server.read_payload().await.expect("C01").as_slice(),
            chat.as_slice()
        );
        assert_eq!(
            server.read_payload().await.expect("C0A").as_slice(),
            [0x0A, 0, 0, 0, 42, 1],
            "int entity id, then 1 for a swing"
        );
        assert_eq!(
            server.read_payload().await.expect("C09").as_slice(),
            [0x09, 0, 3]
        );
        handle.disconnect().expect("disconnect");
        assert_eq!(task.await.expect("pump"), DisconnectReason::ClosedByClient);
    }

    /// 1.7.10's socket options, read back from a socket first set the other way (Nagle off,
    /// type of service 0), so the test fails if either call is lost: Nagle's algorithm on and
    /// IP_TOS 24. Then a connected 1.7.10 stream keeps Nagle on; a 1.8.9 one has it off.
    #[tokio::test]
    async fn the_1_7_10_socket_options_are_set() {
        let socket = tokio::net::TcpSocket::new_v4().expect("socket");
        socket.set_nodelay(true).expect("nodelay on");
        socket.set_tos_v4(0).expect("type of service 0");
        configure_socket(&socket);
        assert!(!socket.nodelay().expect("nodelay"));
        assert_eq!(socket.tos_v4().expect("type of service"), 24);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener");
        let port = listener.local_addr().expect("address").port();
        let accept = tokio::spawn(async move { listener.accept().await });
        let opts =
            ConnectOptions::offline("127.0.0.1", port, "Tester").with_version(GameVersion::V1_7_10);
        let stream = open_connection(&opts).await.expect("connects");
        assert!(!stream.nodelay().expect("nodelay"));
        drop(accept.await);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener");
        let port = listener.local_addr().expect("address").port();
        let accept = tokio::spawn(async move { listener.accept().await });
        let v47 = Connection::connect(("127.0.0.1", port))
            .await
            .expect("connects");
        assert!(v47.get_ref().nodelay().expect("nodelay"), "1.8.9 control");
        drop(accept.await);
    }

    /// 1.7.10's chat grammar: text and translations only. A `score` or `selector` component, at
    /// the top or in a translation argument, a sibling or a hover text, fails the decode (java17
    /// `IChatComponent.Serializer` throws `JsonParseException`); 1.8.9's decoder accepts them.
    #[test]
    fn chat_and_kick_reasons_follow_the_1_7_10_grammar() {
        let chat = |component: &str| {
            S2CPlay5::Chat(s2c::Chat {
                component: component.to_owned(),
            })
        };
        for accepted in [
            r#"{"text":"hi"}"#,
            r#""plain""#,
            r#"{"translate":"chat.type.text","with":["a",{"text":"b"}]}"#,
            r#"[{"text":"a"},{"text":"b","extra":[{"text":"c"}]}]"#,
            "null",
        ] {
            assert!(validate_components(&chat(accepted)).is_ok(), "{accepted}");
        }
        for rejected in [
            r#"{"score":{"name":"a","objective":"b"}}"#,
            r#"{"selector":"@p"}"#,
            r#"{"translate":"x","with":[{"selector":"@a"}]}"#,
            r#"{"text":"a","extra":[{"selector":"@a"}]}"#,
            r#"{"text":"a","hoverEvent":{"action":"show_text","value":{"selector":"@p"}}}"#,
        ] {
            let Err(reason) = validate_components(&chat(rejected)) else {
                panic!("{rejected} accepted");
            };
            assert!(
                reason.to_component_json().contains("JsonParseException"),
                "{rejected}"
            );
            assert!(
                super::super::validate_component(rejected).is_ok(),
                "1.8.9 control: {rejected}"
            );
        }
        let kick = S2CPlay5::Disconnect(s2c::Disconnect {
            reason: r#"{"selector":"@p"}"#.to_owned(),
        });
        assert!(validate_components(&kick).is_err());
        assert!(validate_components(&chat("{not json")).is_err());
    }

    /// The environment variable naming a local vanilla 1.7.10 server (`host:port`) for
    /// [`a_live_1_7_10_server_accepts_join_walk_chat_and_leave`]. Offline mode, loopback only.
    const LIVE_SERVER_ENV: &str = "MC_V5_LIVE_SERVER";

    /// One client tick as the client runs it: the frame poll, the controller sync, the 1.7.10
    /// drain, then the player tick.
    fn live_tick(
        session: &mut Session,
        input: &mc_physics::MoveInput,
        events: &mut Vec<ClientEvent>,
    ) {
        session.begin_poll();
        while session.poll_next(events) {}
        session.update_controller();
        if session.drains_v5_before_world() || session.drains_v5_in_world() {
            for _ in 0..crate::session::v5::PACKETS_PER_DRAIN {
                if !session.apply_next_v5(events) {
                    break;
                }
            }
        }
        if matches!(session.phase(), Phase::Playing) {
            let _ = session.tick(input);
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    fn corrections(events: &[ClientEvent]) -> usize {
        events
            .iter()
            .filter(|event| matches!(event, ClientEvent::PositionCorrection(_)))
            .count()
    }

    /// Against a real vanilla 1.7.10 server: log in, receive the spawn chunks and the S08, walk
    /// two seconds without being set back, chat and see the server's echo, then leave cleanly.
    /// Ignored by default; run with `MC_V5_LIVE_SERVER=127.0.0.1:<port>` and `--ignored`.
    #[test]
    #[ignore = "needs a local vanilla 1.7.10 server named by MC_V5_LIVE_SERVER"]
    fn a_live_1_7_10_server_accepts_join_walk_chat_and_leave() {
        let Ok(address) = std::env::var(LIVE_SERVER_ENV) else {
            eprintln!("skip: {LIVE_SERVER_ENV} is not set");
            return;
        };
        let (host, port) = address.rsplit_once(':').expect("host:port");
        let port: u16 = port.parse().expect("port");
        let net = NetRuntime::new().expect("client runtime");
        let opts =
            ConnectOptions::offline(host, port, "NetLive1").with_version(GameVersion::V1_7_10);
        let mut session = Session::with_version(
            net.connect(opts),
            JoinSettings::default(),
            GameVersion::V1_7_10,
        );
        let mut events = Vec::new();
        let still = mc_physics::MoveInput::default();

        // Join: the world, the spawn S08 and the ground under the player.
        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline
            && !events
                .iter()
                .any(|event| matches!(event, ClientEvent::TerrainDownloaded))
        {
            live_tick(&mut session, &still, &mut events);
        }
        assert!(
            matches!(session.phase(), Phase::Playing),
            "{:?}",
            session.phase()
        );
        for _ in 0..40 {
            live_tick(&mut session, &still, &mut events);
        }
        let start = session.predictor().expect("player").state.pos;
        let below = mc_math::BlockPos::new(
            start.x.floor() as i32,
            (start.y - 0.5).floor() as i32,
            start.z.floor() as i32,
        );
        assert_ne!(
            session.state().world.block_at(below),
            mc_data::BlockState::AIR,
            "the spawn chunks arrived and the player stands on them"
        );
        let corrections_at_rest = corrections(&events);

        // Walk: the server accepts the 1.7.10 movement packets and sets nothing back.
        let walk = mc_physics::MoveInput {
            forward: true,
            yaw: 0.0,
            ..mc_physics::MoveInput::default()
        };
        for _ in 0..40 {
            live_tick(&mut session, &walk, &mut events);
        }
        for _ in 0..30 {
            live_tick(&mut session, &still, &mut events);
        }
        let end = session.predictor().expect("player").state.pos;
        let walked = ((end.x - start.x).powi(2) + (end.z - start.z).powi(2)).sqrt();
        assert!(walked > 2.0, "walked only {walked} blocks");
        assert_eq!(
            corrections(&events),
            corrections_at_rest,
            "the server set the player back"
        );

        // Chat: the server broadcasts the line back.
        session.send_chat("net live test");
        let deadline = Instant::now() + Duration::from_secs(5);
        let heard = |events: &[ClientEvent]| {
            events.iter().any(|event| {
                matches!(event, ClientEvent::Chat { component, .. }
                    if format!("{component:?}").contains("net live test"))
            })
        };
        while Instant::now() < deadline && !heard(&events) {
            live_tick(&mut session, &still, &mut events);
        }
        assert!(heard(&events), "no chat echo");

        // Leave.
        session.disconnect();
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline && session.phase().is_live() {
            live_tick(&mut session, &still, &mut events);
        }
        let Phase::Disconnected(reason) = session.phase() else {
            panic!("still connected");
        };
        assert_eq!(reason.title_key(), "disconnect.lost");
        assert!(matches!(
            reason,
            DisconnectReason::WithOrigin { reason, .. }
                if **reason == DisconnectReason::ClosedByClient
        ));
    }
}
