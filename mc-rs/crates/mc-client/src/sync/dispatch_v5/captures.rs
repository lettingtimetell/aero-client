//! Every clientbound Play packet of real sessions with a vanilla 1.7.10 server, applied in
//! order through [`apply_v5`]: none may fail.
//!
//! The captures are private and never part of this repository: point `MC_V5_CAPTURES_DIR` at a
//! directory of `*.mc5cap` files (the `mc5wire` format, described in `mc-protocol`'s
//! `tests/v5_captures.rs`) to run this test; without it the test skips.

use std::collections::BTreeMap;
use std::path::PathBuf;

use mc_protocol::PacketBuffer;

use super::*;

/// One recorded frame: its connection state (3 is Play), its direction and its bytes (VarInt id
/// and body).
struct Frame {
    state: u8,
    clientbound: bool,
    bytes: Vec<u8>,
}

/// The `mc5wire` container: magic, a little-endian header length and JSON header, then records
/// of direction, state, a timestamp, a length and the frame.
fn read_capture(bytes: &[u8]) -> Vec<Frame> {
    assert_eq!(&bytes[..8], b"MC5WIRE\0", "magic");
    let header_len = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes")) as usize;
    let header = std::str::from_utf8(&bytes[12..12 + header_len]).expect("UTF-8 header");
    assert!(
        header.contains("\"protocol\":5"),
        "not a protocol-5 capture"
    );
    let mut at = 12 + header_len;
    let mut frames = Vec::new();
    while at < bytes.len() {
        let len = u32::from_le_bytes(bytes[at + 10..at + 14].try_into().expect("4 bytes")) as usize;
        frames.push(Frame {
            clientbound: bytes[at] == 0,
            state: bytes[at + 1],
            bytes: bytes[at + 14..at + 14 + len].to_vec(),
        });
        at += 14 + len;
    }
    frames
}

#[test]
fn every_captured_clientbound_packet_applies_without_an_error() {
    let Some(dir) = std::env::var_os("MC_V5_CAPTURES_DIR").map(PathBuf::from) else {
        println!("skip: MC_V5_CAPTURES_DIR is not set");
        return;
    };
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("MC_V5_CAPTURES_DIR")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "mc5cap"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .mc5cap files in MC_V5_CAPTURES_DIR");

    let mut applied: BTreeMap<i32, usize> = BTreeMap::new();
    let mut failures = Vec::new();
    let mut sessions = 0;
    for path in &files {
        let name = path
            .file_name()
            .expect("name")
            .to_string_lossy()
            .into_owned();
        let frames = read_capture(&std::fs::read(path).expect("capture"));
        if !frames.iter().any(|frame| frame.state == 3) {
            continue; // a server-list ping
        }
        sessions += 1;
        let mut state = ClientState::new();
        let mut events = Vec::new();
        for (index, frame) in frames.iter().enumerate() {
            if frame.state != 3 || !frame.clientbound {
                continue;
            }
            let mut buf = PacketBuffer::from_slice(&frame.bytes);
            let id = buf.read_varint().expect("packet id");
            let packet = S2CPlay5::read(id, &mut buf)
                .unwrap_or_else(|error| panic!("{name} #{index} {id:#04x}: {error}"));
            // The network task answers these itself (java17 `NetworkManager.channelRead0`).
            if packet.has_priority() {
                continue;
            }
            events.clear();
            if let Err(error) = apply_v5(&mut state, packet, &mut events) {
                failures.push(format!("{name} #{index} {id:#04x}: {error}"));
            }
            *applied.entry(id).or_default() += 1;
        }
        // The error path is live: a display slot 1.7.10 does not have is refused.
        let refused = apply_v5(
            &mut state,
            S2CPlay5::DisplayScoreboard(s2c5::DisplayScoreboard {
                position: 3,
                score_name: String::new(),
            }),
            &mut events,
        );
        assert!(refused.is_err(), "{name}: the negative control must fail");
    }
    println!(
        "{sessions} play captures, {} packets applied, {} distinct ids",
        applied.values().sum::<usize>(),
        applied.len()
    );
    for (id, count) in &applied {
        println!("  {id:#04x}: {count}");
    }
    assert!(sessions > 0, "no play capture in MC_V5_CAPTURES_DIR");
    assert!(failures.is_empty(), "handler errors: {failures:#?}");
}
