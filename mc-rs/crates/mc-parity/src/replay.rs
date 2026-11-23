//! Deterministic replay of a [`Capture`].
//!
//! # What "deterministic" means here
//!
//! Replay walks the capture's frames in **capture order** — not sorted by timestamp, because
//! two directions are buffered independently and a sort would invent an ordering the session
//! never had. Nothing in this module reads a clock, a random source, the filesystem, or any
//! global state, so [`Replayer::run`] on the same bytes produces the same [`ReplayReport`]
//! every time. `tests/capture_replay.rs` asserts that rather than assuming it.
//!
//! # What replay checks
//!
//! For each frame:
//!
//! 1. **Decode** — the frame's leading varint is the packet id, and `(state, direction, id)`
//!    selects a decoder from `mc-protocol`. A failure is recorded, not raised: a capture that
//!    contains one unknown packet is still worth replaying the other 40,000 frames of.
//! 2. **Full consumption** — a decoder that stops short of the frame's end has silently
//!    dropped trailing fields. That is recorded separately from a hard decode error, because
//!    it is the failure mode that a `Result`-only check misses entirely.
//! 3. **Re-encode** — the decoded value is written back out and compared with the captured
//!    bytes. This is the strongest single property a capture can support: it means the type
//!    is a *lossless* model of the wire, so the client can echo a server's own bytes back.
//!
//! Point 3 is exactly what S06's "golden test against captured vanilla traffic" wanted and
//! could not have. With a real capture in hand this module answers it in one call.

use std::collections::BTreeMap;

use mc_protocol::play::{C2SPlay, S2CPlay};
use mc_protocol::{handshake, login, status, Packet, PacketBuffer, ProtoError, State};

use crate::capture::{read_varint, Capture, Direction, Frame};

/// One decoded packet, across every state and both directions.
///
/// Play packets keep `mc-protocol`'s own dispatch enums rather than being flattened into a
/// hundred variants here; the non-Play states have few enough packets to name individually.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum Decoded {
    /// The only handshake-state packet, and always serverbound.
    Handshake(handshake::Handshake),
    StatusRequest,
    StatusPing(status::Ping),
    StatusResponse(status::Response),
    StatusPong(status::Pong),
    LoginStart(login::LoginStart),
    EncryptionResponse(login::EncryptionResponse),
    LoginDisconnect(login::Disconnect),
    EncryptionRequest(login::EncryptionRequest),
    LoginSuccess(login::LoginSuccess),
    SetCompression(login::SetCompression),
    /// Any of the 74 clientbound Play packets.
    PlayClientbound(Box<S2CPlay>),
    /// Any of the 26 serverbound Play packets.
    PlayServerbound(Box<C2SPlay>),
}

impl Decoded {
    /// This packet's wire id within its (state, direction) namespace.
    #[must_use]
    pub fn id(&self) -> i32 {
        match self {
            Decoded::Handshake(_) => handshake::Handshake::ID,
            Decoded::StatusRequest => status::Request::ID,
            Decoded::StatusPing(_) => status::Ping::ID,
            Decoded::StatusResponse(_) => status::Response::ID,
            Decoded::StatusPong(_) => status::Pong::ID,
            Decoded::LoginStart(_) => login::LoginStart::ID,
            Decoded::EncryptionResponse(_) => login::EncryptionResponse::ID,
            Decoded::LoginDisconnect(_) => login::Disconnect::ID,
            Decoded::EncryptionRequest(_) => login::EncryptionRequest::ID,
            Decoded::LoginSuccess(_) => login::LoginSuccess::ID,
            Decoded::SetCompression(_) => login::SetCompression::ID,
            Decoded::PlayClientbound(p) => p.id(),
            Decoded::PlayServerbound(p) => p.id(),
        }
    }

    /// A short, stable name for reporting — the variant name, not the packet's Java class.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Decoded::Handshake(_) => "Handshake",
            Decoded::StatusRequest => "StatusRequest",
            Decoded::StatusPing(_) => "StatusPing",
            Decoded::StatusResponse(_) => "StatusResponse",
            Decoded::StatusPong(_) => "StatusPong",
            Decoded::LoginStart(_) => "LoginStart",
            Decoded::EncryptionResponse(_) => "EncryptionResponse",
            Decoded::LoginDisconnect(_) => "LoginDisconnect",
            Decoded::EncryptionRequest(_) => "EncryptionRequest",
            Decoded::LoginSuccess(_) => "LoginSuccess",
            Decoded::SetCompression(_) => "SetCompression",
            Decoded::PlayClientbound(_) => "PlayClientbound",
            Decoded::PlayServerbound(_) => "PlayServerbound",
        }
    }

    /// Re-encode to a complete frame payload: the packet-id varint followed by the body.
    ///
    /// The result is directly comparable with [`Frame::payload`].
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut body = PacketBuffer::new();
        match self {
            Decoded::Handshake(p) => p.write(&mut body),
            Decoded::StatusRequest => status::Request.write(&mut body),
            Decoded::StatusPing(p) => p.write(&mut body),
            Decoded::StatusResponse(p) => p.write(&mut body),
            Decoded::StatusPong(p) => p.write(&mut body),
            Decoded::LoginStart(p) => p.write(&mut body),
            Decoded::EncryptionResponse(p) => p.write(&mut body),
            Decoded::LoginDisconnect(p) => p.write(&mut body),
            Decoded::EncryptionRequest(p) => p.write(&mut body),
            Decoded::LoginSuccess(p) => p.write(&mut body),
            Decoded::SetCompression(p) => p.write(&mut body),
            Decoded::PlayClientbound(p) => p.write(&mut body),
            Decoded::PlayServerbound(p) => p.write(&mut body),
        }
        let mut out = Vec::with_capacity(body.as_slice().len() + 5);
        crate::capture::write_varint(&mut out, self.id());
        out.extend_from_slice(body.as_slice());
        out
    }
}

/// How much of a frame a decoder consumed, and what came out.
#[derive(Debug, Clone)]
pub struct DecodeOutcome {
    /// The decoded packet.
    pub packet: Decoded,
    /// Bytes left unread after the decoder finished. Non-zero means the type is missing
    /// trailing fields, which no `Result` check would have surfaced.
    pub unread: usize,
}

/// Decode one captured frame.
///
/// # Errors
/// The underlying [`ProtoError`], or [`ProtoError::Eof`] when the frame is too short even to
/// hold a packet-id varint.
pub fn decode_frame(frame: &Frame) -> Result<DecodeOutcome, ProtoError> {
    let Some((id, used)) = read_varint(&frame.payload) else {
        return Err(ProtoError::Eof("the packet id varint"));
    };
    let mut buf = PacketBuffer::from_slice(&frame.payload[used..]);
    let packet = decode_body(frame.state, frame.direction, id, &mut buf)?;
    Ok(DecodeOutcome {
        packet,
        unread: buf.remaining(),
    })
}

/// Decode a packet body, given its state, direction and id.
///
/// Exposed separately from [`decode_frame`] so a live recorder — or a fuzz driver that
/// synthesises `(state, direction, id)` triples directly — can use the same dispatch.
///
/// # Errors
/// [`ProtoError::UnknownPacketId`] for an id with no packet in that (state, direction), or
/// whatever the selected packet's `read` returns.
pub fn decode_body(
    state: State,
    direction: Direction,
    id: i32,
    buf: &mut PacketBuffer,
) -> Result<Decoded, ProtoError> {
    let unknown = || ProtoError::UnknownPacketId { state, id };
    match (state, direction) {
        (State::Handshaking, Direction::Serverbound) if id == handshake::Handshake::ID => {
            Ok(Decoded::Handshake(handshake::Handshake::read(buf)?))
        }
        (State::Status, Direction::Serverbound) => match id {
            0x00 => {
                status::Request::read(buf)?;
                Ok(Decoded::StatusRequest)
            }
            0x01 => Ok(Decoded::StatusPing(status::Ping::read(buf)?)),
            _ => Err(unknown()),
        },
        (State::Status, Direction::Clientbound) => match id {
            0x00 => Ok(Decoded::StatusResponse(status::Response::read(buf)?)),
            0x01 => Ok(Decoded::StatusPong(status::Pong::read(buf)?)),
            _ => Err(unknown()),
        },
        (State::Login, Direction::Serverbound) => match id {
            0x00 => Ok(Decoded::LoginStart(login::LoginStart::read(buf)?)),
            0x01 => Ok(Decoded::EncryptionResponse(
                login::EncryptionResponse::read(buf)?,
            )),
            _ => Err(unknown()),
        },
        (State::Login, Direction::Clientbound) => match id {
            0x00 => Ok(Decoded::LoginDisconnect(login::Disconnect::read(buf)?)),
            0x01 => Ok(Decoded::EncryptionRequest(login::EncryptionRequest::read(
                buf,
            )?)),
            0x02 => Ok(Decoded::LoginSuccess(login::LoginSuccess::read(buf)?)),
            0x03 => Ok(Decoded::SetCompression(login::SetCompression::read(buf)?)),
            _ => Err(unknown()),
        },
        (State::Play, Direction::Clientbound) => {
            Ok(Decoded::PlayClientbound(Box::new(S2CPlay::read(id, buf)?)))
        }
        (State::Play, Direction::Serverbound) => {
            Ok(Decoded::PlayServerbound(Box::new(C2SPlay::read(id, buf)?)))
        }
        _ => Err(unknown()),
    }
}

/// A frame that did not decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeFailure {
    /// Index into [`Capture::frames`].
    pub frame: usize,
    /// Which way it was going.
    pub direction: Direction,
    /// The state it was captured in.
    pub state: State,
    /// The packet id, if the leading varint was even readable.
    pub id: Option<i32>,
    /// The error, rendered. `ProtoError` is not `Clone`, and a report is a value.
    pub error: String,
}

/// A frame that decoded but did not survive a round trip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReencodeMismatch {
    /// Index into [`Capture::frames`].
    pub frame: usize,
    /// The decoded packet's name.
    pub kind: &'static str,
    /// The packet id.
    pub id: i32,
    /// Bytes the decoder never read. Non-zero here is the usual cause of a mismatch.
    pub unread: usize,
    /// Length of the captured frame payload.
    pub captured_len: usize,
    /// Length of the re-encoded payload.
    pub reencoded_len: usize,
    /// Index of the first differing byte, or `None` when one is a prefix of the other.
    pub first_diff: Option<usize>,
}

/// What a replay found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReplayReport {
    /// Frames seen.
    pub frames: usize,
    /// Frames that decoded without error.
    pub decoded: usize,
    /// Frames that decoded *and* re-encoded to byte-identical bytes.
    pub byte_identical: usize,
    /// How many frames of each `(state, direction, id)` triple were seen, decoded or not.
    /// A `BTreeMap` so the report's `Debug` output is stable, which matters when a CI failure
    /// is read as text.
    pub seen: BTreeMap<(u8, u8, i32), usize>,
    /// Every decode failure, in frame order.
    pub decode_failures: Vec<DecodeFailure>,
    /// Every round-trip failure, in frame order.
    pub reencode_mismatches: Vec<ReencodeMismatch>,
}

impl ReplayReport {
    /// Whether every frame decoded and re-encoded byte-identically.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.decode_failures.is_empty()
            && self.reencode_mismatches.is_empty()
            && self.decoded == self.frames
            && self.byte_identical == self.frames
    }

    /// How many distinct `(state, direction, id)` triples the capture exercised. This is the
    /// coverage number a capture is worth judging by.
    #[must_use]
    pub fn distinct_packets(&self) -> usize {
        self.seen.len()
    }

    /// A one-line summary suitable for a test's stdout.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} frames, {} decoded, {} byte-identical re-encodes, {} distinct packet ids, \
             {} decode failures, {} round-trip mismatches",
            self.frames,
            self.decoded,
            self.byte_identical,
            self.distinct_packets(),
            self.decode_failures.len(),
            self.reencode_mismatches.len()
        )
    }
}

/// Replays a capture through `mc-protocol`'s decoders.
#[derive(Debug)]
pub struct Replayer<'a> {
    capture: &'a Capture,
    direction: Option<Direction>,
}

impl<'a> Replayer<'a> {
    /// Replay every frame of `capture`.
    #[must_use]
    pub fn new(capture: &'a Capture) -> Replayer<'a> {
        Replayer {
            capture,
            direction: None,
        }
    }

    /// Restrict replay to one direction — useful when only the server's half is interesting.
    #[must_use]
    pub fn only(mut self, direction: Direction) -> Replayer<'a> {
        self.direction = Some(direction);
        self
    }

    /// Run, collecting a report and doing nothing else with the decoded packets.
    #[must_use]
    pub fn run(&self) -> ReplayReport {
        self.run_with(|_, _| {})
    }

    /// Run, handing each successfully decoded packet to `sink` in capture order.
    ///
    /// This is the hook a subsystem-level parity test uses: feed the decoded packets into
    /// `mc_world`, `mc_physics` or a screen, and compare the resulting state with a reference.
    pub fn run_with(&self, mut sink: impl FnMut(&Frame, &Decoded)) -> ReplayReport {
        let mut report = ReplayReport::default();
        for (index, frame) in self.capture.frames().iter().enumerate() {
            if self.direction.is_some_and(|d| d != frame.direction) {
                continue;
            }
            report.frames += 1;
            let id = frame.packet_id();
            *report
                .seen
                .entry((
                    crate::capture::state_tag(frame.state),
                    frame.direction.tag(),
                    id.unwrap_or(i32::MIN),
                ))
                .or_default() += 1;

            match decode_frame(frame) {
                Ok(outcome) => {
                    report.decoded += 1;
                    let reencoded = outcome.packet.encode();
                    if outcome.unread == 0 && reencoded == frame.payload {
                        report.byte_identical += 1;
                    } else {
                        let first_diff = reencoded
                            .iter()
                            .zip(&frame.payload)
                            .position(|(a, b)| a != b);
                        report.reencode_mismatches.push(ReencodeMismatch {
                            frame: index,
                            kind: outcome.packet.kind(),
                            id: outcome.packet.id(),
                            unread: outcome.unread,
                            captured_len: frame.payload.len(),
                            reencoded_len: reencoded.len(),
                            first_diff,
                        });
                    }
                    sink(frame, &outcome.packet);
                }
                Err(e) => report.decode_failures.push(DecodeFailure {
                    frame: index,
                    direction: frame.direction,
                    state: frame.state,
                    id,
                    error: e.to_string(),
                }),
            }
        }
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{CaptureMeta, Recorder};

    #[test]
    fn a_handshake_frame_decodes_and_re_encodes_identically() {
        let pkt = handshake::Handshake {
            protocol_version: 47,
            server_address: "localhost".into(),
            server_port: 25565,
            next_state: handshake::NextState::Login,
        };
        let decoded = Decoded::Handshake(pkt.clone());
        let payload = decoded.encode();

        let mut rec = Recorder::new(CaptureMeta {
            protocol: 47,
            address: "localhost:25565".into(),
            started_at_unix_millis: 0,
        });
        rec.record_at(Direction::Serverbound, State::Handshaking, 0, &payload);
        let capture = rec.finish();

        let report = Replayer::new(&capture).run();
        assert!(report.is_clean(), "{report:?}");
        assert_eq!(report.frames, 1);
        assert_eq!(report.byte_identical, 1);
    }

    #[test]
    fn a_decoder_that_stops_short_is_reported_even_though_it_returned_ok() {
        // `status::Request` reads nothing. Append a stray byte and the decode still succeeds,
        // but `unread` catches it — the failure mode a `Result`-only check cannot see.
        let mut payload = Decoded::StatusRequest.encode();
        payload.push(0xFF);

        let mut rec = Recorder::new(CaptureMeta {
            protocol: 47,
            address: "x".into(),
            started_at_unix_millis: 0,
        });
        rec.record_at(Direction::Serverbound, State::Status, 0, &payload);
        let capture = rec.finish();

        let report = Replayer::new(&capture).run();
        assert_eq!(report.decoded, 1, "the decode itself succeeds");
        assert!(!report.is_clean(), "but the replay must not be clean");
        assert_eq!(report.reencode_mismatches.len(), 1);
        assert_eq!(report.reencode_mismatches[0].unread, 1);
    }

    #[test]
    fn an_unknown_id_is_recorded_not_raised() {
        let mut rec = Recorder::new(CaptureMeta {
            protocol: 47,
            address: "x".into(),
            started_at_unix_millis: 0,
        });
        // Play clientbound stops at 0x49.
        rec.record_at(Direction::Clientbound, State::Play, 0, &[0x7F]);
        rec.record_at(Direction::Clientbound, State::Play, 1, &[0x00, 0x2A]);
        let capture = rec.finish();

        let report = Replayer::new(&capture).run();
        assert_eq!(report.frames, 2);
        assert_eq!(report.decoded, 1, "the good frame still replays");
        assert_eq!(report.decode_failures.len(), 1);
        assert_eq!(report.decode_failures[0].id, Some(0x7F));
    }

    #[test]
    fn the_same_id_in_two_states_decodes_to_two_different_packets() {
        // 0x00 serverbound is `Handshake` in Handshaking, `LoginStart` in Login, and
        // `KeepAlive` in Play. Getting the state wrong is a whole class of replay bug.
        let mut rec = Recorder::new(CaptureMeta {
            protocol: 47,
            address: "x".into(),
            started_at_unix_millis: 0,
        });
        rec.record_at(
            Direction::Serverbound,
            State::Login,
            0,
            &Decoded::LoginStart(login::LoginStart {
                username: "Dev".into(),
            })
            .encode(),
        );
        rec.record_at(
            Direction::Serverbound,
            State::Play,
            1,
            &Decoded::PlayServerbound(Box::new(C2SPlay::KeepAlive(
                mc_protocol::play::c2s::KeepAlive { key: 7 },
            )))
            .encode(),
        );
        let capture = rec.finish();

        let mut kinds = Vec::new();
        let report = Replayer::new(&capture).run_with(|_, d| kinds.push(d.kind()));
        assert!(report.is_clean(), "{report:?}");
        assert_eq!(kinds, vec!["LoginStart", "PlayServerbound"]);
    }
}
