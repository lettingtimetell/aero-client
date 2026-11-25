//! Every frame of real protocol-5 sessions with a vanilla 1.7.10 server decodes, consumes its
//! bytes exactly, and encodes back to the same bytes.
//!
//! The captures are private and never part of this repository: point `MC_V5_CAPTURES_DIR` at a
//! directory of `*.mc5cap` files to run this test; without it the test skips. The `mc5wire`
//! format: an 8-byte magic `MC5WIRE\0`, a little-endian `u32` header length and a JSON header,
//! then records of `u8` direction (0 clientbound, 1 serverbound), `u8` state (0 handshaking,
//! 1 status, 2 login, 3 play), `u64` microseconds, `u32` length and the frame (VarInt id, body).
//!
//! The serverbound side of these captures was written by a test tool, not by a 1.7.10 client, so
//! it checks only that the codec reads what a server reads; outbound bytes are pinned by the
//! Java-written fixtures (`v5_wire_oracle.rs`).

mod v5_support;

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use mc_protocol::handshake::Handshake;
use mc_protocol::login::{Disconnect, LoginStart, LoginSuccess};
use mc_protocol::status::{Ping, Pong, Request, Response};
use mc_protocol::v5::login::{EncryptionRequest, EncryptionResponse};
use mc_protocol::v5::{C2SPlay5, S2CPlay5};
use mc_protocol::{Packet, PacketBuffer, ProtoError, State};

struct Record {
    serverbound: bool,
    state: u8,
    frame: Vec<u8>,
}

fn read_capture(bytes: &[u8]) -> Vec<Record> {
    assert_eq!(&bytes[..8], b"MC5WIRE\0", "magic");
    let header_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&bytes[12..12 + header_len]).expect("UTF-8 header");
    assert!(
        header.contains("\"protocol\":5"),
        "not a protocol-5 capture: {header}"
    );
    let mut at = 12 + header_len;
    let mut out = Vec::new();
    while at < bytes.len() {
        let direction = bytes[at];
        let state = bytes[at + 1];
        let len = u32::from_le_bytes(bytes[at + 10..at + 14].try_into().unwrap()) as usize;
        let frame = bytes[at + 14..at + 14 + len].to_vec();
        at += 14 + len;
        out.push(Record {
            serverbound: direction == 1,
            state,
            frame,
        });
    }
    out
}

fn re_encode<P: Packet>(id: i32, packet: &P) -> Vec<u8> {
    let mut buf = PacketBuffer::new();
    buf.write_varint(id);
    packet.write(&mut buf);
    buf.finish_write().expect("encodable");
    buf.as_slice().to_vec()
}

fn decode_one<P: Packet>(id: i32, body: &mut PacketBuffer) -> Result<Vec<u8>, ProtoError> {
    let packet = P::read(body)?;
    Ok(re_encode(id, &packet))
}

/// Decode one frame by (state, direction) and return its re-encoding.
fn decode(record: &Record) -> Result<(i32, Vec<u8>, usize), ProtoError> {
    let mut buf = PacketBuffer::from_slice(&record.frame);
    let id = buf.read_varint()?;
    let unknown = |state| ProtoError::UnknownPacketId { state, id };
    let bytes = match (record.state, record.serverbound) {
        (0, true) if id == Handshake::ID => decode_one::<Handshake>(id, &mut buf)?,
        (1, true) if id == Request::ID => decode_one::<Request>(id, &mut buf)?,
        (1, true) if id == Ping::ID => decode_one::<Ping>(id, &mut buf)?,
        (1, false) if id == Response::ID => decode_one::<Response>(id, &mut buf)?,
        (1, false) if id == Pong::ID => decode_one::<Pong>(id, &mut buf)?,
        (2, true) if id == LoginStart::ID => decode_one::<LoginStart>(id, &mut buf)?,
        (2, true) if id == EncryptionResponse::ID => {
            decode_one::<EncryptionResponse>(id, &mut buf)?
        }
        (2, false) if id == Disconnect::ID => decode_one::<Disconnect>(id, &mut buf)?,
        (2, false) if id == EncryptionRequest::ID => decode_one::<EncryptionRequest>(id, &mut buf)?,
        (2, false) if id == LoginSuccess::ID => decode_one::<LoginSuccess>(id, &mut buf)?,
        (0, _) => return Err(unknown(State::Handshaking)),
        (1, _) => return Err(unknown(State::Status)),
        (2, _) => return Err(unknown(State::Login)),
        (3, true) => {
            let packet = C2SPlay5::read(id, &mut buf)?;
            let mut out = PacketBuffer::new();
            out.write_varint(id);
            packet.write(&mut out);
            out.finish_write()?;
            out.as_slice().to_vec()
        }
        (3, false) => {
            let packet = S2CPlay5::read(id, &mut buf)?;
            let mut out = PacketBuffer::new();
            out.write_varint(id);
            packet.write(&mut out);
            out.finish_write()?;
            out.as_slice().to_vec()
        }
        (state, _) => panic!("unknown state {state}"),
    };
    Ok((id, bytes, buf.remaining()))
}

#[test]
fn every_captured_frame_decodes_and_re_encodes() {
    let Some(dir) = std::env::var_os("MC_V5_CAPTURES_DIR").map(PathBuf::from) else {
        println!("skip: MC_V5_CAPTURES_DIR is not set");
        return;
    };
    let mut files: Vec<PathBuf> = fs::read_dir(&dir)
        .expect("MC_V5_CAPTURES_DIR")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "mc5cap"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .mc5cap files in MC_V5_CAPTURES_DIR");

    let mut seen: BTreeMap<(u8, bool, i32), usize> = BTreeMap::new();
    let (mut frames, mut relaxed) = (0usize, 0usize);
    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        for (index, record) in read_capture(&fs::read(path).expect("capture"))
            .iter()
            .enumerate()
        {
            let label = format!(
                "{name} #{index} ({}, state {})",
                if record.serverbound { "C2S" } else { "S2C" },
                record.state
            );
            let (id, bytes, trailing) =
                decode(record).unwrap_or_else(|e| panic!("{label}: decode failed: {e}"));
            assert_eq!(trailing, 0, "{label} id {id:#04x}: trailing bytes");
            relaxed +=
                v5_support::assert_wire_eq(&format!("{label} id {id:#04x}"), &bytes, &record.frame);
            *seen
                .entry((record.state, record.serverbound, id))
                .or_default() += 1;
            frames += 1;
        }
    }
    let play_s2c: Vec<i32> = seen
        .keys()
        .filter(|(state, serverbound, _)| *state == 3 && !serverbound)
        .map(|(_, _, id)| *id)
        .collect();
    println!(
        "{} captures, {frames} frames, {} distinct (state, direction, id), {} clientbound play ids, \
         {relaxed} gzip blobs equal after inflating",
        files.len(),
        seen.len(),
        play_s2c.len()
    );
    for ((state, serverbound, id), count) in &seen {
        println!(
            "  state {state} {} {id:#04x}: {count}",
            if *serverbound { "C2S" } else { "S2C" }
        );
    }
}
