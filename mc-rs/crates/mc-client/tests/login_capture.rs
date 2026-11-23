//! Closed at the `mc-client` wiring layer.
//!
//! `mc-protocol/tests/login_loopback.rs::the_login_observer_hook_covers_every_frame_h3_and_h4_are_about`
//! proves `connection::login`'s new observer hook itself, end to end, against a real RSA/AES-CFB8
//! reference server. This file's only job is the layer above that: does a `.mccap` file produced
//! by the real `net::task` connection path (`NetRuntime::connect`, exactly what
//! `mc_client::Session` drives) actually contain the login-sequence frames once they land in
//! `net::capture::CaptureSink::record_login`.
//!
//! The fake server here is deliberately simple — offline mode, no encryption, no compression —
//! because encryption/compression correctness is S07's concern, proved independently by the file
//! above. It is built directly from `mc_protocol::Connection`, the same public, symmetric type
//! the real client uses, hand-driven from the server's side; that is a smaller, more honest tool
//! for a wiring test than reimplementing an independent reference peer would be.

use std::time::{Duration, Instant};

use mc_client::net::{ConnectOptions, NetRuntime, ToGame};
use mc_parity::{Capture, Direction};
use mc_protocol::login::{LoginStart, LoginSuccess};
use mc_protocol::{Connection, Packet, PacketBuffer, State};
use tokio::net::TcpListener;
use uuid::Uuid;

/// How often the test thread polls the [`mc_client::net::NetHandle`]. Loopback TCP on
/// `127.0.0.1`, so this is expected to resolve in well under a second in practice.
const POLL: Duration = Duration::from_millis(5);

#[test]
fn the_mccap_capture_covers_the_login_sequence_not_only_play() {
    let runtime = NetRuntime::new().expect("runtime");
    let expected_uuid = Uuid::from_u128(0x1122_3344_5566_7788_99aa_bbcc_ddee_ff00);

    let listener = runtime
        .handle()
        .block_on(async { TcpListener::bind("127.0.0.1:0").await.expect("bind") });
    let addr = listener.local_addr().expect("local addr");

    // The fake server: real TCP, `mc_protocol::Connection` driven by hand from the other side,
    // offline mode. It only has to survive long enough to send `LoginSuccess`.
    runtime.handle().spawn(async move {
        let (socket, _) = listener.accept().await.expect("accept");
        let mut conn = Connection::new(socket);

        let (id, _) = conn.read_packet().await.expect("handshake");
        assert_eq!(id, mc_protocol::handshake::Handshake::ID);
        conn.set_state(State::Login).expect("-> login");

        let (id, mut body) = conn.read_packet().await.expect("login start");
        assert_eq!(id, LoginStart::ID);
        let start = LoginStart::read(&mut body).expect("login start body");

        conn.send(&LoginSuccess {
            uuid: expected_uuid,
            username: start.username,
        })
        .await
        .expect("login success");
        conn.set_state(State::Play).expect("-> play");

        // Hold the socket open (rather than dropping `conn` and racing the client's own
        // disconnect) until the client closes its end, which it does once the test asks it to.
        let _ = conn.read_payload().await;
    });

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("login.mccap");

    let mut handle = runtime.connect(
        ConnectOptions::offline("127.0.0.1", addr.port(), "Notch").capture_to(path.clone()),
    );

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut connected = false;
    while Instant::now() < deadline {
        match handle.try_recv() {
            Some(ToGame::Connected(outcome)) => {
                assert_eq!(outcome.uuid, expected_uuid);
                assert_eq!(outcome.username, "Notch");
                connected = true;
                break;
            }
            Some(ToGame::Disconnected(reason)) => {
                panic!("disconnected before Connected: {reason:?}")
            }
            _ => std::thread::sleep(POLL),
        }
    }
    assert!(connected, "never received ToGame::Connected");

    // Ask the connection task to stop and wait for it to actually finish, so the capture file is
    // flushed (`CaptureSink::finish` runs on `net::task::run`'s way out) before we read it back.
    // Best-effort: if the task has already exited for some other reason there is nobody left to
    // tell, and the `is_closed` poll below still catches that case correctly.
    let _ = handle.disconnect();
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline && !handle.is_closed() {
        handle.try_recv();
        std::thread::sleep(POLL);
    }
    assert!(handle.is_closed(), "the connection task never finished");

    let capture = Capture::read_file(&path).expect("the capture must parse");
    assert!(!capture.is_empty(), "the capture must not be empty");

    // The two frames this row is about: `LoginStart` out, `LoginSuccess` back, both tagged
    // `State::Login` — proving the capture no longer "begins at Play".
    let login_frames: Vec<_> = capture
        .frames()
        .iter()
        .filter(|f| f.state == State::Login)
        .collect();
    assert_eq!(
        login_frames.len(),
        2,
        "expected exactly LoginStart + LoginSuccess in State::Login, got {login_frames:?}"
    );
    assert_eq!(login_frames[0].direction, Direction::Serverbound);
    assert_eq!(login_frames[0].packet_id(), Some(LoginStart::ID));
    assert_eq!(login_frames[1].direction, Direction::Clientbound);
    assert_eq!(login_frames[1].packet_id(), Some(LoginSuccess::ID));

    let success = LoginSuccess::read(&mut PacketBuffer::from_slice(
        login_frames[1].body().expect("LoginSuccess has a body"),
    ))
    .expect("LoginSuccess body decodes");
    assert_eq!(success.uuid, expected_uuid);
    assert_eq!(success.username, "Notch");
}
