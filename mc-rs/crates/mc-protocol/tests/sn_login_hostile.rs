//! What a hostile server can do with the login handshake's key exchange.
//!
//! The far side is the same independent peer `login_loopback.rs` uses (`tests/common`), on
//! loopback TCP, doing real RSA. Three properties of `NetHandlerLoginClient`'s exchange are pinned:
//! a server gets exactly one key exchange and one session-server join per connection; once the
//! client has answered an `EncryptionRequest`, nothing the server sends is read as plaintext
//! again; and a public key too large to be a server's is refused at once, before any join.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use common::RefPeer;
use mc_protocol::connection::{handshake, login};
use mc_protocol::handshake::NextState;
use mc_protocol::login::{EncryptionRequest, EncryptionResponse, LoginStart, LoginSuccess};
use mc_protocol::packet::Packet;
use mc_protocol::{Connection, PacketBuffer, ProtoError};
use rand::rngs::OsRng;
use rsa::pkcs8::EncodePublicKey;
use rsa::{BigUint, Pkcs1v15Encrypt, RsaPrivateKey, RsaPublicKey};
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
    P::read(&mut buf).expect("packet body")
}

fn request(public_key_der: Vec<u8>) -> Vec<u8> {
    body_of(&EncryptionRequest {
        server_id: String::new(),
        public_key_der,
        verify_token: VERIFY_TOKEN.to_vec(),
    })
}

/// Accept one client, read its handshake and `LoginStart`, and hand the peer to `script`.
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
        let _handshake = peer.read_packet().await;
        let (id, _) = peer.read_packet().await;
        assert_eq!(id, LoginStart::ID);
        script(peer).await;
    });
    (addr, server)
}

async fn client(addr: std::net::SocketAddr) -> Connection<tokio::net::TcpStream> {
    let mut conn = Connection::connect(addr).await.expect("connect");
    handshake(&mut conn, "127.0.0.1", addr.port(), NextState::Login)
        .await
        .expect("handshake");
    conn
}

/// A second `EncryptionRequest`, sent inside the encrypted stream, ends the login: no second
/// key exchange, no second session-server join with a hash the server picked.
#[tokio::test]
async fn a_second_encryption_request_ends_the_login_without_a_second_join() {
    let private = RsaPrivateKey::new(&mut OsRng, 1024).expect("1024-bit RSA keygen");
    let der = private
        .to_public_key()
        .to_public_key_der()
        .expect("SPKI DER")
        .as_bytes()
        .to_vec();
    let (addr, server) = serve(move |mut peer| async move {
        peer.send_packet(EncryptionRequest::ID, &request(der.clone()))
            .await;
        let (id, body) = peer.read_packet().await;
        assert_eq!(id, EncryptionResponse::ID);
        let response: EncryptionResponse = decode(&body);
        let secret = private
            .decrypt(Pkcs1v15Encrypt, &response.shared_secret_encrypted)
            .expect("secret");
        peer.enable_encryption(&secret.try_into().expect("16-byte key"));
        peer.send_packet(EncryptionRequest::ID, &request(der)).await;
    })
    .await;

    let joins = AtomicUsize::new(0);
    let mut conn = client(addr).await;
    let result = login(
        &mut conn,
        "Player",
        |_hash| {
            joins.fetch_add(1, Ordering::SeqCst);
            async { Ok(()) }
        },
        None,
    )
    .await;
    assert!(
        matches!(result, Err(ProtoError::AlreadyEncrypted)),
        "{result:?}"
    );
    assert_eq!(joins.load(Ordering::SeqCst), 1);
    server.await.expect("server script");
}

/// After the client has answered an `EncryptionRequest` it reads only through the cipher: a
/// server that carries on in plaintext gets no login, whatever it sends.
#[tokio::test]
async fn plaintext_after_the_key_exchange_never_logs_in() {
    let private = RsaPrivateKey::new(&mut OsRng, 1024).expect("1024-bit RSA keygen");
    let der = private
        .to_public_key()
        .to_public_key_der()
        .expect("SPKI DER")
        .as_bytes()
        .to_vec();
    let (addr, server) = serve(move |mut peer| async move {
        peer.send_packet(EncryptionRequest::ID, &request(der)).await;
        let (id, _) = peer.read_packet().await;
        assert_eq!(id, EncryptionResponse::ID);
        // Encryption deliberately left off on this side.
        for _ in 0..4 {
            peer.send_packet(
                LoginSuccess::ID,
                &body_of(&LoginSuccess {
                    uuid: Uuid::from_u128(7),
                    username: "Player".to_owned(),
                }),
            )
            .await;
        }
        // Dropping the peer closes the socket, so a half-read frame ends in EOF, not a wait.
    })
    .await;

    let mut conn = client(addr).await;
    let result = tokio::time::timeout(
        Duration::from_secs(20),
        login(&mut conn, "Player", |_hash| async { Ok(()) }, None),
    )
    .await
    .expect("the login ends");
    assert!(result.is_err(), "{result:?}");
    assert!(conn.is_encrypted());
    server.await.expect("server script");
}

/// The SPKI DER of an RSA public key with an odd `bits`-bit modulus. Not a real key pair (no
/// key generation), which is all a parser sees.
fn public_key_of_bits(bits: usize) -> Vec<u8> {
    let modulus = (BigUint::from(1u8) << (bits - 1)) + BigUint::from(1u8);
    RsaPublicKey::new_with_max_size(modulus, BigUint::from(65_537u32), bits)
        .expect("a public key of that size")
        .to_public_key_der()
        .expect("SPKI DER")
        .as_bytes()
        .to_vec()
}

/// Keys over 4096 bits are refused when the request is read, before the session-server join.
/// Servers use 1024-bit keys; vanilla's Java runtime would take up to 16384 bits, which no
/// server sends.
#[tokio::test]
async fn an_oversized_server_key_is_refused_before_the_join() {
    let (addr, server) = serve(|mut peer| async move {
        peer.send_packet(EncryptionRequest::ID, &request(public_key_of_bits(16_384)))
            .await;
    })
    .await;

    let joins = AtomicUsize::new(0);
    let mut conn = client(addr).await;
    let result = login(
        &mut conn,
        "Player",
        |_hash| {
            joins.fetch_add(1, Ordering::SeqCst);
            async { Ok(()) }
        },
        None,
    )
    .await;
    assert!(
        matches!(result, Err(ProtoError::PublicKey(_))),
        "{result:?}"
    );
    assert_eq!(joins.load(Ordering::SeqCst), 0);
    server.await.expect("server script");

    // The bound is the parser's: 4096 bits is still a key.
    assert!(mc_protocol::crypt::parse_public_key_der(&public_key_of_bits(4096)).is_ok());
    assert!(mc_protocol::crypt::parse_public_key_der(&public_key_of_bits(4097)).is_err());
}
