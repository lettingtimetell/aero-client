//! End-to-end login against an in-process server over loopback TCP.
//!
//! # Why this exists, and exactly what it does and does not prove
//!
//! S07's task list asks for "an integration test against a real 1.8.9 server (offline-mode
//! first)". There is no network access and no running Minecraft server in this environment, so
//! that cannot be done literally — the same limitation S06 recorded for "a captured vanilla
//! login+play session". Rather than skip it, this file stands up a server on
//! `127.0.0.1:0` that does the **real cryptographic work**:
//!
//! * it generates a genuine 1024-bit RSA keypair and publishes it as X.509 SPKI DER, exactly as
//!   `CryptManager.generateKeyPair` + `PublicKey.getEncoded()` would;
//! * it really decrypts the client's `C01PacketEncryptionResponse` with the private half and
//!   checks that the verify token round-tripped;
//! * it switches its own side to AES-CFB8 under the decrypted secret, using a from-scratch
//!   implementation of the mode (`tests/common/mod.rs`) that shares no code with the client's;
//! * it enables compression at a threshold and sends packets on both sides of it;
//! * its framing, VarInts and zlib calls are all independently written.
//!
//! **Proved by this file:** the key exchange interoperates with a real RSA implementation; the
//! join hash the client computes matches the secret the server actually received (hazard H3);
//! the CFB8 stream stays continuous across packet, frame and TCP-segment boundaries (hazard
//! H4); compression sits inside encryption; and the state machine reaches
//! [`State::Play`](mc_protocol::State::Play) with the right `LoginSuccess` data.
//!
//! **Not proved by this file:** that a real vanilla 1.8.9 server accepts these bytes — the far
//! side here is a test double, and while its transport layers are independent, both ends agree
//! on the S06 packet *bodies* by construction. Nor does it exercise
//! [`mc_auth::join_server`]: the Mojang session-server call is stubbed out, because reaching it
//! needs network access and a real account (see S03's own outstanding live-verification task).
//! Loopback TCP is local I/O, not network access.

mod common;

use common::RefPeer;
use mc_protocol::connection::{
    handshake, login, login_offline, ping_server, Direction, LoginOutcome,
};
use mc_protocol::handshake::{Handshake, NextState};
use mc_protocol::login::{
    Disconnect, EncryptionRequest, EncryptionResponse, LoginStart, LoginSuccess, SetCompression,
};
use mc_protocol::packet::Packet;
use mc_protocol::play::{c2s, s2c, C2SPlay, S2CPlay};
use mc_protocol::{Connection, PacketBuffer, ProtoError, State, PROTOCOL_VERSION};
use rand::rngs::OsRng;
use rsa::pkcs8::EncodePublicKey;
use rsa::{Pkcs1v15Encrypt, RsaPrivateKey};
use tokio::net::TcpListener;
use uuid::Uuid;

const THRESHOLD: usize = 64;
const SERVER_ID: &str = "a1b2c3d4e5f60718";
const VERIFY_TOKEN: [u8; 4] = [0x0D, 0xF0, 0xAD, 0x8B];

fn body_of<P: Packet>(packet: &P) -> Vec<u8> {
    let mut buf = PacketBuffer::new();
    packet.write(&mut buf);
    buf.as_slice().to_vec()
}

fn decode<P: Packet>(bytes: &[u8]) -> P {
    let mut buf = PacketBuffer::from_slice(bytes);
    let packet = P::read(&mut buf).expect("packet body");
    assert!(buf.is_empty(), "trailing bytes after {} bytes", bytes.len());
    packet
}

/// The clientbound burst the server sends after `LoginSuccess`, deliberately alternating
/// side of the compression threshold. Packets 2 and 4 compress; 1, 3 and 5 do not.
///
/// The alternation matters: a CFB8 stream that is re-initialised per packet (hazard H4) still
/// decodes packet 1 correctly and fails from packet 2 on, and a compression layer applied on
/// the wrong side of encryption fails on the first *compressed* packet rather than the first
/// packet. Interleaving the two cases makes both failures unambiguous.
fn play_burst() -> Vec<S2CPlay> {
    let long_chat = format!(r#"{{"text":"{}"}}"#, "the quick brown fox ".repeat(25));
    let longer_chat = format!(r#"{{"text":"{}"}}"#, "0123456789abcdef".repeat(40));
    vec![
        S2CPlay::KeepAlive(s2c::KeepAlive { id: 1 }),
        S2CPlay::Chat(s2c::Chat {
            component: long_chat,
            position: 0,
        }),
        S2CPlay::KeepAlive(s2c::KeepAlive { id: 2 }),
        S2CPlay::Chat(s2c::Chat {
            component: longer_chat,
            position: 1,
        }),
        S2CPlay::KeepAlive(s2c::KeepAlive { id: 3 }),
    ]
}

/// The full online-mode-shaped sequence: handshake → `LoginStart` → `EncryptionRequest` →
/// `EncryptionResponse` → AES-CFB8 both ways → `SetCompression` → `LoginSuccess` → `Play`.
#[tokio::test]
async fn client_reaches_play_against_a_loopback_server_doing_real_crypto() {
    // Keygen is CPU-bound; do it before any async work so it cannot starve the runtime.
    let private = RsaPrivateKey::new(&mut OsRng, 1024).expect("1024-bit RSA keygen");
    let public_key_der = private
        .to_public_key()
        .to_public_key_der()
        .expect("SPKI DER")
        .as_bytes()
        .to_vec();

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");

    let expected_uuid = Uuid::from_u128(0x0693_9f22_e93c_4e4e_9d67_2c1d_4f0a_66b1);
    let server_der = public_key_der.clone();

    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.expect("accept");
        socket.set_nodelay(true).expect("nodelay");
        let mut peer = RefPeer::new(socket);

        // --- handshake ---
        let (id, body) = peer.read_packet().await;
        assert_eq!(id, Handshake::ID, "first packet must be the handshake");
        let hs: Handshake = decode(&body);
        assert_eq!(hs.protocol_version, PROTOCOL_VERSION);
        assert_eq!(hs.next_state, NextState::Login);
        assert_eq!(hs.server_port, addr.port());

        // --- login start ---
        let (id, body) = peer.read_packet().await;
        assert_eq!(id, LoginStart::ID);
        let start: LoginStart = decode(&body);

        // --- encryption request ---
        peer.send_packet(
            EncryptionRequest::ID,
            &body_of(&EncryptionRequest {
                server_id: SERVER_ID.to_owned(),
                public_key_der: server_der,
                verify_token: VERIFY_TOKEN.to_vec(),
            }),
        )
        .await;

        // --- encryption response: really decrypt it ---
        let (id, body) = peer.read_packet().await;
        assert_eq!(id, EncryptionResponse::ID);
        let response: EncryptionResponse = decode(&body);
        let secret = private
            .decrypt(Pkcs1v15Encrypt, &response.shared_secret_encrypted)
            .expect("the client's shared secret must decrypt with our private key");
        let token = private
            .decrypt(Pkcs1v15Encrypt, &response.verify_token_encrypted)
            .expect("the client's verify token must decrypt with our private key");
        assert_eq!(token, VERIFY_TOKEN, "verify token did not round-trip");
        assert_eq!(secret.len(), 16, "the AES key must be 128 bits");
        let key: [u8; 16] = secret.try_into().expect("16-byte key");

        // Everything from here is encrypted, in both directions.
        peer.enable_encryption(&key);

        // --- set compression: encrypted, but itself still uncompressed ---
        peer.send_packet(
            SetCompression::ID,
            &body_of(&SetCompression {
                threshold: THRESHOLD as i32,
            }),
        )
        .await;
        peer.set_compression(THRESHOLD);

        // --- login success, then a burst of play packets, all back to back ---
        peer.send_packet(
            LoginSuccess::ID,
            &body_of(&LoginSuccess {
                uuid: expected_uuid,
                username: "Notch".to_owned(),
            }),
        )
        .await;
        for packet in play_burst() {
            let mut buf = PacketBuffer::new();
            packet.write(&mut buf);
            peer.send_packet(packet.id(), buf.as_slice()).await;
        }

        // --- and read what the client sends back, across the threshold both ways ---
        let mut from_client = Vec::new();
        for _ in 0..2 {
            let (id, body) = peer.read_packet().await;
            from_client
                .push(C2SPlay::read(id, &mut PacketBuffer::from_slice(&body)).expect("c2s packet"));
        }

        (start.username, key, from_client)
    });

    // ---------------- the client under test ----------------
    let mut conn = Connection::connect(addr).await.expect("connect");
    handshake(&mut conn, "127.0.0.1", addr.port(), NextState::Login)
        .await
        .expect("handshake");
    assert_eq!(conn.state(), State::Login);

    let mut join_hash = None;
    let outcome: LoginOutcome = login(
        &mut conn,
        "Notch",
        |hash| {
            join_hash = Some(hash);
            async { Ok(()) }
        },
        None,
    )
    .await
    .expect("login");

    assert_eq!(conn.state(), State::Play, "the client must end up in Play");
    assert!(outcome.encrypted, "encryption should be on");
    assert_eq!(outcome.compression_threshold, Some(THRESHOLD));
    assert_eq!(outcome.uuid, expected_uuid);
    assert_eq!(outcome.username, "Notch");

    // The five-packet burst, read through the now-live encryption and compression layers.
    let mut received = Vec::new();
    for _ in 0..5 {
        let (id, mut body) = conn.read_packet().await.expect("play packet");
        received.push(S2CPlay::read(id, &mut body).expect("s2c packet"));
        assert!(body.is_empty(), "packet body was not fully consumed");
    }
    assert_eq!(received, play_burst());

    // Send in both size classes too, so the outbound compression path is exercised live.
    conn.send(&c2s::KeepAlive { key: 3 }).await.expect("send");
    let long_message = "x".repeat(90);
    conn.send(&c2s::ChatMessage {
        message: long_message.clone(),
    })
    .await
    .expect("send");

    let (username, key, from_client) = server.await.expect("server task");
    assert_eq!(username, "Notch");
    assert_eq!(
        from_client,
        vec![
            C2SPlay::KeepAlive(c2s::KeepAlive { key: 3 }),
            C2SPlay::ChatMessage(c2s::ChatMessage {
                message: long_message
            }),
        ]
    );

    // Hazard H3, end to end: the hash the client registered must be the one the *server* would
    // compute from the secret it actually decrypted and the key it actually published.
    assert_eq!(
        join_hash.expect("join callback must have run"),
        mc_auth::server_hash(SERVER_ID, &key, &public_key_der),
        "the join hash does not match the secret the server received"
    );
}

/// The `.mccap` capture "begins at `Play`" because
/// `connection::login` had no observer, so `EncryptionRequest`, the RSA response,
/// `SetCompression` and `LoginSuccess` were invisible to any capture. This proves `login`'s new
/// `observe` hook closes exactly that gap, over the same real-RSA, real-AES-CFB8, real-TCP
/// online-mode-shaped sequence the test above proves the login logic itself against — so this is
/// evidence about the hook, not a restatement of that test with extra bookkeeping.
///
/// # What is checked
///
/// * every frame the sequence puts on the wire is reported, in wire order, with the right
///   [`Direction`] — including `EncryptionRequest`/the encryption response, which cross the wire
///   unencrypted, and `SetCompression`/`LoginSuccess`, which the reference server (`RefPeer`)
///   really enciphers with its own independent AES-CFB8 before this test's client ever sees the
///   bytes;
/// * the hook's bytes decode as real, plaintext packet frames (VarInt id + body) even for the
///   two frames that crossed the wire as ciphertext — proving the hook fires with the *logical*
///   frame (post-decrypt, matching `Connection::read_payload`), not raw socket bytes;
/// * nothing about the sequence itself changed: the `LoginOutcome` this run reaches is identical
///   in shape to the one the hookless test above reaches.
#[tokio::test]
async fn the_login_observer_hook_covers_every_frame_h3_and_h4_are_about() {
    let private = RsaPrivateKey::new(&mut OsRng, 1024).expect("1024-bit RSA keygen");
    let public_key_der = private
        .to_public_key()
        .to_public_key_der()
        .expect("SPKI DER")
        .as_bytes()
        .to_vec();

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let expected_uuid = Uuid::from_u128(0x0bad_cafe_dead_beef_0bad_cafe_dead_beef);
    let server_der = public_key_der.clone();

    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.expect("accept");
        socket.set_nodelay(true).expect("nodelay");
        let mut peer = RefPeer::new(socket);

        let (id, _) = peer.read_packet().await;
        assert_eq!(id, Handshake::ID);

        let (id, body) = peer.read_packet().await;
        assert_eq!(id, LoginStart::ID);
        let start: LoginStart = decode(&body);

        peer.send_packet(
            EncryptionRequest::ID,
            &body_of(&EncryptionRequest {
                server_id: SERVER_ID.to_owned(),
                public_key_der: server_der,
                verify_token: VERIFY_TOKEN.to_vec(),
            }),
        )
        .await;

        let (id, body) = peer.read_packet().await;
        assert_eq!(id, EncryptionResponse::ID);
        let response: EncryptionResponse = decode(&body);
        let secret = private
            .decrypt(Pkcs1v15Encrypt, &response.shared_secret_encrypted)
            .expect("the client's shared secret must decrypt with our private key");
        let key: [u8; 16] = secret.try_into().expect("16-byte key");

        // From here on the reference peer's own independent CFB8 really enciphers every byte it
        // sends — the point being that the hook must still report plaintext for these frames.
        peer.enable_encryption(&key);

        peer.send_packet(
            SetCompression::ID,
            &body_of(&SetCompression {
                threshold: THRESHOLD as i32,
            }),
        )
        .await;
        peer.set_compression(THRESHOLD);

        peer.send_packet(
            LoginSuccess::ID,
            &body_of(&LoginSuccess {
                uuid: expected_uuid,
                username: "Notch".to_owned(),
            }),
        )
        .await;

        start.username
    });

    let mut conn = Connection::connect(addr).await.expect("connect");
    handshake(&mut conn, "127.0.0.1", addr.port(), NextState::Login)
        .await
        .expect("handshake");

    let mut frames: Vec<(Direction, Vec<u8>)> = Vec::new();
    let mut observe = |direction: Direction, bytes: &[u8]| frames.push((direction, bytes.to_vec()));

    let outcome = login(
        &mut conn,
        "Notch",
        |_hash| async { Ok(()) },
        Some(&mut observe),
    )
    .await
    .expect("login");

    assert_eq!(outcome.uuid, expected_uuid);
    assert_eq!(outcome.username, "Notch");
    assert!(outcome.encrypted, "encryption should be on");
    assert_eq!(outcome.compression_threshold, Some(THRESHOLD));

    // Exactly the five login-state frames, in wire order, each tagged with the direction it
    // actually travelled.
    let ids: Vec<(Direction, i32)> = frames
        .iter()
        .map(|(dir, bytes)| {
            let mut buf = PacketBuffer::from_slice(bytes);
            (*dir, buf.read_varint().expect("leading varint id"))
        })
        .collect();
    assert_eq!(
        ids,
        vec![
            (Direction::Serverbound, LoginStart::ID),
            (Direction::Clientbound, EncryptionRequest::ID),
            (Direction::Serverbound, EncryptionResponse::ID),
            (Direction::Clientbound, SetCompression::ID),
            (Direction::Clientbound, LoginSuccess::ID),
        ],
        "the hook must see every login-state frame, in order, correctly directed"
    );

    // The two frames the reference peer actually enciphered on the wire must still have reached
    // the hook as plaintext: decoding them with the real packet types must succeed and match
    // what the server sent.
    let mut set_compression_body = PacketBuffer::from_slice(&frames[3].1);
    assert_eq!(
        set_compression_body.read_varint().unwrap(),
        SetCompression::ID
    );
    assert_eq!(
        SetCompression::read(&mut set_compression_body)
            .unwrap()
            .threshold,
        THRESHOLD as i32
    );

    let mut login_success_body = PacketBuffer::from_slice(&frames[4].1);
    assert_eq!(login_success_body.read_varint().unwrap(), LoginSuccess::ID);
    let success = LoginSuccess::read(&mut login_success_body).unwrap();
    assert_eq!(success.uuid, expected_uuid);
    assert_eq!(success.username, "Notch");

    let username = server.await.expect("server task");
    assert_eq!(username, "Notch");
}

/// The literal reading of "offline-mode first": a server with `online-mode=false` never sends
/// `EncryptionRequest` at all, and 1.8.9 servers only enable compression if configured to.
#[tokio::test]
async fn offline_mode_server_skips_encryption_entirely() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let expected_uuid = Uuid::from_u128(0x1111_2222_3333_4444_5555_6666_7777_8888);

    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.expect("accept");
        let mut peer = RefPeer::new(socket);
        let (id, _) = peer.read_packet().await;
        assert_eq!(id, Handshake::ID);
        let (id, body) = peer.read_packet().await;
        assert_eq!(id, LoginStart::ID);
        let start: LoginStart = decode(&body);
        peer.send_packet(
            LoginSuccess::ID,
            &body_of(&LoginSuccess {
                uuid: expected_uuid,
                username: start.username.clone(),
            }),
        )
        .await;
        peer.send_packet(
            S2CPlay::KeepAlive(s2c::KeepAlive { id: 7 }).id(),
            &body_of(&s2c::KeepAlive { id: 7 }),
        )
        .await;
        start.username
    });

    let mut conn = Connection::connect(addr).await.expect("connect");
    handshake(&mut conn, "127.0.0.1", addr.port(), NextState::Login)
        .await
        .expect("handshake");
    let outcome = login_offline(&mut conn, "Dev", None).await.expect("login");

    assert_eq!(conn.state(), State::Play);
    assert!(!outcome.encrypted);
    assert_eq!(outcome.compression_threshold, None);
    assert_eq!(outcome.uuid, expected_uuid);

    let (id, mut body) = conn.read_packet().await.expect("play packet");
    assert_eq!(
        S2CPlay::read(id, &mut body).unwrap(),
        S2CPlay::KeepAlive(s2c::KeepAlive { id: 7 })
    );
    assert_eq!(server.await.expect("server task"), "Dev");
}

/// A kick during login must surface the server's reason JSON verbatim, so S16 can render it.
#[tokio::test]
async fn a_login_kick_surfaces_the_reason_over_real_tcp() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let reason = r#"{"translate":"multiplayer.disconnect.outdated_server","with":["1.8.9"]}"#;

    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.expect("accept");
        let mut peer = RefPeer::new(socket);
        let _ = peer.read_packet().await; // handshake
        let _ = peer.read_packet().await; // login start
        peer.send_packet(
            Disconnect::ID,
            &body_of(&Disconnect {
                reason: reason.to_owned(),
            }),
        )
        .await;
    });

    let mut conn = Connection::connect(addr).await.expect("connect");
    handshake(&mut conn, "127.0.0.1", addr.port(), NextState::Login)
        .await
        .expect("handshake");
    match login_offline(&mut conn, "Notch", None).await {
        Err(ProtoError::Disconnected { reason: got }) => assert_eq!(got, reason),
        other => panic!("expected a Disconnected error, got {other:?}"),
    }
    // The connection must not have advanced past Login.
    assert_eq!(conn.state(), State::Login);
    server.await.expect("server task");
}

/// Server-list ping over real TCP, the whole way through [`ping_server`].
#[tokio::test]
async fn status_ping_over_loopback_tcp() {
    use mc_protocol::status::{Ping, Pong, Request, Response};

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let json = r#"{"version":{"name":"1.8.9","protocol":47},"players":{"max":20,"online":0},"description":{"text":"A Minecraft Server"}}"#;

    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.expect("accept");
        let mut peer = RefPeer::new(socket);
        let (id, body) = peer.read_packet().await;
        assert_eq!(id, Handshake::ID);
        let hs: Handshake = decode(&body);
        assert_eq!(hs.next_state, NextState::Status);
        assert_eq!(hs.protocol_version, PROTOCOL_VERSION);

        let (id, body) = peer.read_packet().await;
        assert_eq!(id, Request::ID);
        assert!(body.is_empty(), "status Request has no body");
        peer.send_packet(
            Response::ID,
            &body_of(&Response {
                json: json.to_owned(),
            }),
        )
        .await;

        let (id, body) = peer.read_packet().await;
        assert_eq!(id, Ping::ID);
        let ping: Ping = decode(&body);
        peer.send_packet(
            Pong::ID,
            &body_of(&Pong {
                payload: ping.payload,
            }),
        )
        .await;
    });

    let status = ping_server("127.0.0.1", addr.port()).await.expect("ping");
    assert_eq!(status.json, json);
    server.await.expect("server task");
}
