//! The protocol-5 login over loopback TCP against an independent server: real RSA, real AES/CFB8,
//! the `i16`-prefixed encryption blobs, and the play state reached afterwards.
//!
//! The far side is `tests/common`'s reference peer (its own CFB8 and framing, sharing no code with
//! `mc_protocol::{crypt, connection}`); only the packet bodies come from this crate, and those are
//! pinned against the 1.7.10 classes by `v5_wire_oracle.rs`.

mod common;

use std::sync::{Arc, Mutex};

use common::RefPeer;
use mc_protocol::connection::{handshake_for, Direction, LoginAuthPolicy};
use mc_protocol::connection_v5::{login_v5_offline, login_v5_with_auth_policy};
use mc_protocol::handshake::{Handshake, NextState};
use mc_protocol::login::{LoginStart, LoginSuccess};
use mc_protocol::v5::login::{EncryptionRequest, EncryptionResponse};
use mc_protocol::v5::{s2c, S2CPlay5};
use mc_protocol::{Connection, Packet, PacketBuffer, ProtoError, State};
use mc_version::GameVersion;
use rand::rngs::OsRng;
use rsa::pkcs8::EncodePublicKey;
use rsa::{Pkcs1v15Encrypt, RsaPrivateKey};
use tokio::net::TcpListener;
use uuid::Uuid;

const VERIFY_TOKEN: [u8; 4] = [0x5E, 0xC0, 0x11, 0x7E];

fn body_of<P: Packet>(packet: &P) -> Vec<u8> {
    let mut buf = PacketBuffer::new();
    packet.write(&mut buf);
    buf.as_slice().to_vec()
}

fn decode<P: Packet>(bytes: &[u8]) -> P {
    let mut buf = PacketBuffer::from_slice(bytes);
    let packet = P::read(&mut buf).expect("packet body");
    assert!(buf.is_empty(), "trailing bytes");
    packet
}

/// Accept one client, check its protocol-5 handshake and `LoginStart`, then run `script`.
async fn serve<F, Fut>(script: F) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>)
where
    F: FnOnce(RefPeer) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send,
{
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.expect("accept");
        let mut peer = RefPeer::new(socket);
        let (id, body) = peer.read_packet().await;
        assert_eq!(id, Handshake::ID);
        let handshake: Handshake = decode(&body);
        assert_eq!(handshake.protocol_version, 5);
        assert_eq!(handshake.next_state, NextState::Login);
        let (id, body) = peer.read_packet().await;
        assert_eq!(id, LoginStart::ID);
        let start: LoginStart = decode(&body);
        assert_eq!(start.username, "Player");
        script(peer).await;
    });
    (addr, server)
}

async fn client(addr: std::net::SocketAddr) -> Connection<tokio::net::TcpStream> {
    let mut conn = Connection::connect(addr).await.expect("connect");
    handshake_for(
        &mut conn,
        "127.0.0.1",
        addr.port(),
        NextState::Login,
        GameVersion::V1_7_10,
    )
    .await
    .expect("handshake");
    conn
}

#[tokio::test]
async fn encrypted_login_reaches_play_with_i16_blobs() {
    let private = RsaPrivateKey::new(&mut OsRng, 1024).expect("1024-bit RSA keygen");
    let der = private
        .to_public_key()
        .to_public_key_der()
        .expect("SPKI DER")
        .as_bytes()
        .to_vec();
    let server_der = der.clone();
    let (addr, server) = serve(move |mut peer| async move {
        let request = EncryptionRequest {
            server_id: String::new(),
            public_key_der: server_der,
            verify_token: VERIFY_TOKEN.to_vec(),
        };
        peer.send_packet(EncryptionRequest::ID, &body_of(&request))
            .await;

        let (id, body) = peer.read_packet().await;
        assert_eq!(id, EncryptionResponse::ID);
        // Two i16 lengths of 128 (a 1024-bit modulus): `00 80`.
        assert_eq!(&body[..2], &[0x00, 0x80]);
        assert_eq!(&body[130..132], &[0x00, 0x80]);
        let response: EncryptionResponse = decode(&body);
        let secret = private
            .decrypt(Pkcs1v15Encrypt, &response.shared_secret_encrypted)
            .expect("secret");
        let token = private
            .decrypt(Pkcs1v15Encrypt, &response.verify_token_encrypted)
            .expect("token");
        assert_eq!(token, VERIFY_TOKEN);
        peer.enable_encryption(&secret.try_into().expect("16-byte key"));

        let success = LoginSuccess {
            uuid: Uuid::from_u128(0x1234),
            username: "Player".into(),
        };
        peer.send_packet(LoginSuccess::ID, &body_of(&success)).await;
        // First play packet, through the cipher: protocol 5's int keep-alive.
        peer.send_packet(0x00, &[0xDE, 0xAD, 0xBE, 0xEF]).await;
        // The client's echo comes back through the cipher too.
        let (id, body) = peer.read_packet().await;
        assert_eq!((id, body.as_slice()), (0x00, &[0xDE, 0xAD, 0xBE, 0xEF][..]));
    })
    .await;

    let hashes = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&hashes);
    let mut conn = client(addr).await;
    let mut frames = Vec::new();
    let mut observe = |direction: Direction, bytes: &[u8]| frames.push((direction, bytes.to_vec()));
    let outcome = login_v5_with_auth_policy(
        &mut conn,
        "Player",
        move |hash| {
            seen.lock().unwrap().push(hash);
            async { Ok(()) }
        },
        Some(&mut observe),
        LoginAuthPolicy::Strict,
    )
    .await
    .expect("login");
    assert!(outcome.encrypted);
    assert_eq!(outcome.compression_threshold, None);
    assert_eq!(outcome.uuid, Uuid::from_u128(0x1234));
    assert_eq!(conn.state(), State::Play);
    assert_eq!(hashes.lock().unwrap().len(), 1, "one session-server join");
    // LoginStart, EncryptionRequest, EncryptionResponse, LoginSuccess.
    assert_eq!(frames.len(), 4);

    let (id, mut body) = conn.read_packet().await.expect("play packet");
    let packet = S2CPlay5::read(id, &mut body).expect("S00");
    assert_eq!(
        packet,
        S2CPlay5::KeepAlive(s2c::KeepAlive { id: -559_038_737 })
    );
    conn.send(&mc_protocol::v5::c2s::KeepAlive { key: -559_038_737 })
        .await
        .expect("echo");
    server.await.expect("server script");
}

#[tokio::test]
async fn a_second_encryption_request_ends_the_login() {
    let private = RsaPrivateKey::new(&mut OsRng, 1024).expect("1024-bit RSA keygen");
    let der = private
        .to_public_key()
        .to_public_key_der()
        .expect("SPKI DER")
        .as_bytes()
        .to_vec();
    let (addr, server) = serve(move |mut peer| async move {
        let request = body_of(&EncryptionRequest {
            server_id: String::new(),
            public_key_der: der,
            verify_token: VERIFY_TOKEN.to_vec(),
        });
        peer.send_packet(EncryptionRequest::ID, &request).await;
        let (_, body) = peer.read_packet().await;
        let response: EncryptionResponse = decode(&body);
        let secret = private
            .decrypt(Pkcs1v15Encrypt, &response.shared_secret_encrypted)
            .expect("secret");
        peer.enable_encryption(&secret.try_into().expect("16-byte key"));
        peer.send_packet(EncryptionRequest::ID, &request).await;
    })
    .await;
    let mut conn = client(addr).await;
    let result = login_v5_offline(&mut conn, "Player", None).await;
    assert!(
        matches!(result, Err(ProtoError::AlreadyEncrypted)),
        "{result:?}"
    );
    server.await.expect("server script");
}

#[tokio::test]
async fn set_compression_is_unknown_in_protocol_5() {
    let (addr, server) = serve(move |mut peer| async move {
        // 1.8.9's login SetCompression(256).
        peer.send_packet(0x03, &[0x80, 0x02]).await;
    })
    .await;
    let mut conn = client(addr).await;
    let result = login_v5_offline(&mut conn, "Player", None).await;
    assert!(
        matches!(result, Err(ProtoError::UnexpectedPacket { got: 3, .. })),
        "{result:?}"
    );
    server.await.expect("server script");
}

#[tokio::test]
async fn a_negative_blob_length_is_rejected_before_any_join() {
    let (addr, server) = serve(move |mut peer| async move {
        // Empty server id, then an i16 length of -1.
        peer.send_packet(EncryptionRequest::ID, &[0x00, 0xFF, 0xFF])
            .await;
    })
    .await;
    let joined = Arc::new(Mutex::new(false));
    let flag = Arc::clone(&joined);
    let mut conn = client(addr).await;
    let result = login_v5_with_auth_policy(
        &mut conn,
        "Player",
        move |_| {
            *flag.lock().unwrap() = true;
            async { Ok(()) }
        },
        None,
        LoginAuthPolicy::Strict,
    )
    .await;
    assert!(
        matches!(result, Err(ProtoError::InvalidByteArrayLen(-1))),
        "{result:?}"
    );
    assert!(!*joined.lock().unwrap());
    server.await.expect("server script");
}
