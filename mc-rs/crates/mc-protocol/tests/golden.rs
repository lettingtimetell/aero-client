//! Golden-vector tests: hand-derived wire bytes, decoded through this crate's `Packet::read`,
//! independent of `Packet::write` — so a matching bug in *both* directions (the one thing a
//! self round-trip test can't catch) still fails here.
//!
//! # On "captured vanilla traffic" (S06's stated done-when criterion)
//!
//! This environment has no network access and no running 1.8.9 server to capture a real
//! login+play session against, so per the task's documented fallback ("if you can't capture
//! real traffic, build a golden fixture from known-good encoded bytes per packet type and
//! document the limitation") — that is what this file, and the per-packet unit tests beside
//! every packet type in `src/`, are. Every byte vector below is derived by hand from the
//! protocol-47 wire format (VarInt encoding rules, big-endian fixed-width fields, UTF-8
//! string length prefixes), not copied from this crate's own encoder, so it exercises the
//! decoder against an independently-reasoned expectation. **This is a documented gap**: it
//! proves internal consistency and protocol-rule correctness, not "a real vanilla server
//! accepted/produced these exact bytes." Closing that gap needs either a packet-capture
//! recording (S22's parity harness territory) or a live server to connect S07 against.
//!
//! The state-transition packets covered here (`Handshake`, `LoginStart`, `LoginSuccess`,
//! `SetCompression`, status `Response`) gate every single connection, so they matter more per
//! byte than most; the 100 Play packets each carry their own synthetic round-trip test instead
//! (see `src/play/`).

use mc_protocol::handshake::{Handshake, NextState};
use mc_protocol::login::{LoginStart, LoginSuccess, SetCompression};
use mc_protocol::play::{c2s, s2c};
use mc_protocol::status::Response;
use mc_protocol::{Packet, PacketBuffer};
use uuid::Uuid;

#[test]
fn handshake_matches_hand_derived_bytes() {
    // VarInt(47)=0x2F, VarInt(9)+"localhost" (9 ASCII bytes), u16(25565)=0x63DD BE, VarInt(2)=Login.
    let wire: &[u8] = &[
        0x2F, 0x09, b'l', b'o', b'c', b'a', b'l', b'h', b'o', b's', b't', 0x63, 0xDD, 0x02,
    ];
    let mut buf = PacketBuffer::from_slice(wire);
    let pkt = Handshake::read(&mut buf).unwrap();
    assert_eq!(pkt.protocol_version, 47);
    assert_eq!(pkt.server_address, "localhost");
    assert_eq!(pkt.server_port, 25565);
    assert_eq!(pkt.next_state, NextState::Login);
    assert!(buf.is_empty());
}

#[test]
fn login_start_matches_hand_derived_bytes() {
    // VarInt(5) + "Notch".
    let wire: &[u8] = &[0x05, b'N', b'o', b't', b'c', b'h'];
    let mut buf = PacketBuffer::from_slice(wire);
    let pkt = LoginStart::read(&mut buf).unwrap();
    assert_eq!(pkt.username, "Notch");
    assert!(buf.is_empty());
}

#[test]
fn login_success_matches_hand_derived_bytes_dashed_uuid() {
    let mut wire = vec![0x24u8]; // VarInt(36): the nil UUID's dashed-string length
    wire.extend_from_slice(b"00000000-0000-0000-0000-000000000000");
    wire.push(0x01); // VarInt(1)
    wire.push(b'A');
    let mut buf = PacketBuffer::from_slice(&wire);
    let pkt = LoginSuccess::read(&mut buf).unwrap();
    assert_eq!(pkt.uuid, Uuid::nil());
    assert_eq!(pkt.username, "A");
    assert!(buf.is_empty());
}

#[test]
fn set_compression_matches_hand_derived_bytes() {
    // VarInt(256): 256 & 0x7F == 0, so byte0 = 0x80, then (256 >> 7) == 2, byte1 = 0x02.
    let wire: &[u8] = &[0x80, 0x02];
    let mut buf = PacketBuffer::from_slice(wire);
    let pkt = SetCompression::read(&mut buf).unwrap();
    assert_eq!(pkt.threshold, 256);
    assert!(buf.is_empty());
}

#[test]
fn status_response_matches_hand_derived_bytes() {
    // VarInt(2) + "{}"
    let wire: &[u8] = &[0x02, b'{', b'}'];
    let mut buf = PacketBuffer::from_slice(wire);
    let pkt = Response::read(&mut buf).unwrap();
    assert_eq!(pkt.json, "{}");
    assert!(buf.is_empty());
}

#[test]
fn s2c_keep_alive_matches_hand_derived_bytes() {
    // VarInt(300): 300 & 0x7F = 44 (0x2C), so byte0 = 0x2C|0x80 = 0xAC; 300>>7 = 2, byte1 = 0x02.
    let wire: &[u8] = &[0xAC, 0x02];
    let mut buf = PacketBuffer::from_slice(wire);
    let pkt = s2c::KeepAlive::read(&mut buf).unwrap();
    assert_eq!(pkt.id, 300);
    assert!(buf.is_empty());
}

#[test]
fn c2s_keep_alive_matches_hand_derived_bytes() {
    // Same VarInt(300) encoding as the clientbound KeepAlive above — both directions share the
    // same primitive, just a different field name (`key` vs `id`).
    let wire: &[u8] = &[0xAC, 0x02];
    let mut buf = PacketBuffer::from_slice(wire);
    let pkt = c2s::KeepAlive::read(&mut buf).unwrap();
    assert_eq!(pkt.key, 300);
    assert!(buf.is_empty());
}

#[test]
fn varint_negative_one_is_five_0xff_bytes_then_0x0f() {
    // A well-known protocol-47 VarInt vector: -1 as an unsigned 32-bit pattern is all-ones,
    // which needs the full 5-byte VarInt encoding (0xFF 0xFF 0xFF 0xFF 0x0F).
    let wire: &[u8] = &[0xFF, 0xFF, 0xFF, 0xFF, 0x0F];
    let mut buf = PacketBuffer::from_slice(wire);
    assert_eq!(buf.read_varint().unwrap(), -1);
    assert!(buf.is_empty());
}
