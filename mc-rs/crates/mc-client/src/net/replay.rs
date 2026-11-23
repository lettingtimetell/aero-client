//! A capture source that drives a session from a `.mccap` file instead of a socket.
//!
//! *"Every in-world number in §1 came from a live server whose world we
//! do not control and cannot reproduce. `mc-parity` already has three captures and a `Replayer`,
//! but it replays only at the protocol-decode level — it never reaches the renderer. Wire a
//! capture source into `mc-client`'s `net` channel so a session can be driven from a file at a
//! fixed cadence."* This module is that source.
//!
//! # The contract it satisfies
//!
//! [`crate::net::task::run`] is a tokio task that pushes [`ToGame`] messages into a bounded
//! channel; [`NetHandle::try_recv`](crate::net::NetHandle::try_recv) pops them on the game
//! thread. A [`ReplaySource`] is the *same shape from the game loop's side* — `try_recv` still
//! hands back `Connected`, then a stream of `Packet`s — but there is no task, no socket and no
//! runtime. The messages come out of a file.
//!
//! # Why the cadence is counted in polls, not in seconds
//!
//! A benchmark control has to be reproducible on a machine under different load, so **nothing
//! here reads a clock.** The source carries a *virtual* clock advanced by a fixed
//! [`ReplaySource::step_micros`] on every [`ReplaySource::advance`] — and `advance` is called
//! exactly once per [`Session::poll`](crate::session::Session::poll), which the game loop calls
//! exactly once per frame. So packet *i* of the capture is applied on the same frame index in
//! every run of the same file, whether the machine renders at 200 fps or 2000.
//!
//! A wall-clock cadence would have been the obvious alternative and is exactly wrong for this:
//! at 2500 fps a 1 ms tolerance is 2–3 frames of jitter, so the join burst would land on
//! different frames every run and the section/quad/draw counts would not repeat.
//!
//! Timestamps are **normalised to the first replayed packet**, so a capture whose login took
//! half a second does not begin with 500 frames of nothing.
//!
//! # What is dropped, and why each one
//!
//! | dropped | reason |
//! |---|---|
//! | every serverbound frame | the client is the one producing those; replaying them would double them |
//! | `Handshaking` / `Status` / `Login` frames | there is no login to perform. `LoginSuccess`, if present, is used to build the [`LoginOutcome`] and is not otherwise replayed |
//! | Play `KeepAlive` | [`crate::net::task::play_loop`] answers these on the network thread and never forwards them; a replay has nobody to answer |
//! | Play `SetCompression` | transport state for a socket that does not exist |
//! | Play `Disconnect` | **the one deliberate infidelity.** A capture that ends in a kick would end the bench mid-run. The count is reported by [`ReplaySummary::disconnects_dropped`] so it can never be silent |
//!
//! # What happens at the end of the capture
//!
//! Nothing. The source goes quiet and the session stays in
//! [`Phase::Playing`](crate::session::Phase::Playing) with its world resident, which is what a
//! benchmark wants: load the world from the file, then keep rendering it for as many frames as
//! `--frames` asks for. [`ReplaySource::is_exhausted`] reports the transition. It deliberately
//! does **not** synthesise a `Disconnected`, because that would unload the world and leave the
//! rest of the run measuring an empty scene.

use std::path::{Path, PathBuf};

use mc_parity::capture::{Capture, Direction};
use mc_parity::replay::{decode_frame, Decoded};
use mc_protocol::connection::LoginOutcome;
use mc_protocol::play::S2CPlay;
use mc_protocol::{ProtoError, State};

use crate::error::ClientError;
use crate::net::channel::ToGame;

/// The default virtual-clock step: one millisecond of captured time per game-loop poll.
///
/// Chosen so the capture's own pacing survives — a join burst that arrived over 200 ms of
/// captured time is delivered over 200 frames rather than all at once — while a 44-second
/// capture still finishes inside a 44 000-frame run. `--replayStep` overrides it; a very large
/// step degenerates to "deliver everything on the first frame", which is a legitimate (and
/// equally deterministic) way to ask for the whole world up front.
pub const DEFAULT_STEP_MICROS: u64 = 1_000;

/// What a capture turned out to contain, once decoded.
///
/// Reported rather than assumed: the server these captures came
/// from has a **minigame lobby** at spawn, so the number of chunk sections a capture can deliver
/// is the thing that decides whether a replay bench is a heavy load or a floor measurement.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReplaySummary {
    /// Frames in the file, both directions, every state.
    pub frames: usize,
    /// Clientbound `Play` frames that decoded into a packet this source will replay.
    pub replayed: usize,
    /// Frames that failed to decode. Non-zero means the capture and these decoders disagree.
    pub decode_failures: usize,
    /// `ChunkData` packets kept.
    pub chunk_data: usize,
    /// `MapChunkBulk` packets kept.
    pub chunk_bulk: usize,
    /// Chunk sections delivered: the popcount of every column bitmask in the capture. This is
    /// the upper bound on what can reach the mesher, and therefore the honest measure of how
    /// much world a replay bench is loading.
    pub sections: usize,
    /// Play `Disconnect` packets dropped. See the module header — this is the one infidelity.
    pub disconnects_dropped: usize,
    /// Span of the replayed packets in captured microseconds, after normalisation.
    pub span_micros: u64,
}

impl ReplaySummary {
    /// A one-line summary for a log or a test's stdout.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "{} frames, {} replayable, {} chunk packets ({} ChunkData + {} MapChunkBulk) \
             delivering {} sections over {:.1} s of captured time, {} decode failures, \
             {} Disconnects dropped",
            self.frames,
            self.replayed,
            self.chunk_data + self.chunk_bulk,
            self.chunk_data,
            self.chunk_bulk,
            self.sections,
            self.span_micros as f64 / 1e6,
            self.decode_failures,
            self.disconnects_dropped,
        )
    }
}

/// One decoded packet and the captured time it should be released at.
#[derive(Debug)]
struct Timed {
    micros: u64,
    packet: S2CPlay,
    /// The frame's exact bytes, kept only for a source built to feed the
    /// replay recorder ([`ReplaySource::from_file_keeping_bytes`]).
    raw: Option<Box<[u8]>>,
}

/// The queue the source hands packets out of: a `VecDeque` because delivery is strictly
/// front-to-back and the packets are moved out rather than cloned.
type Queue = std::collections::VecDeque<Timed>;

/// A `.mccap` file, decoded once and handed out at a fixed virtual cadence.
///
/// See the module header for the contract, the cadence, and what is dropped.
#[derive(Debug)]
pub struct ReplaySource {
    path: PathBuf,
    packets: Queue,
    /// The virtual clock, in captured microseconds. Advanced only by [`ReplaySource::advance`].
    now_micros: u64,
    step_micros: u64,
    /// The `Connected` message, taken on the first `next()` call.
    login: Option<LoginOutcome>,
    summary: ReplaySummary,
}

impl ReplaySource {
    /// Read and decode a capture.
    ///
    /// # Errors
    ///
    /// [`ClientError::Io`] if the file cannot be read or is not a valid capture, and if it
    /// carries no replayable clientbound `Play` packet at all — a file that would drive nothing
    /// is a configuration mistake worth failing loudly on rather than a silent empty session.
    pub fn from_file(
        path: impl AsRef<Path>,
        step_micros: u64,
    ) -> Result<ReplaySource, ClientError> {
        ReplaySource::from_file_keeping_bytes(path, step_micros, false)
    }

    /// [`ReplaySource::from_file`], optionally keeping every replayed frame's
    /// exact bytes next to its decoded packet, so the replay recorder's tap
    /// ([`crate::net::NetHandle::set_raw_tap`]) can record a capture-driven session exactly as a
    /// live one. Keeping them costs the capture's payload size in memory, so only a session
    /// that may be recorded asks for it.
    ///
    /// # Errors
    ///
    /// As [`ReplaySource::from_file`].
    pub fn from_file_keeping_bytes(
        path: impl AsRef<Path>,
        step_micros: u64,
        keep_bytes: bool,
    ) -> Result<ReplaySource, ClientError> {
        let path = path.as_ref();
        let capture = Capture::read_file(path)
            .map_err(|error| std::io::Error::other(format!("{}: {error}", path.display())))?;
        ReplaySource::from_capture_keeping_bytes(
            &capture,
            path.to_path_buf(),
            step_micros,
            keep_bytes,
        )
    }

    /// Decode an already-parsed capture. Exposed for tests, which build one in memory.
    ///
    /// # Errors
    ///
    /// As [`ReplaySource::from_file`].
    pub fn from_capture(
        capture: &Capture,
        path: PathBuf,
        step_micros: u64,
    ) -> Result<ReplaySource, ClientError> {
        ReplaySource::from_capture_keeping_bytes(capture, path, step_micros, false)
    }

    /// [`ReplaySource::from_capture`], optionally keeping the frames' bytes
    /// (see [`ReplaySource::from_file_keeping_bytes`]).
    ///
    /// # Errors
    ///
    /// As [`ReplaySource::from_file`].
    pub fn from_capture_keeping_bytes(
        capture: &Capture,
        path: PathBuf,
        step_micros: u64,
        keep_bytes: bool,
    ) -> Result<ReplaySource, ClientError> {
        let mut summary = ReplaySummary {
            frames: capture.len(),
            ..ReplaySummary::default()
        };
        let mut packets: Queue = Queue::new();
        let mut login: Option<LoginOutcome> = None;

        for frame in capture.frames() {
            if frame.direction != Direction::Clientbound {
                continue;
            }
            let outcome = match decode_frame(frame) {
                Ok(outcome) => outcome,
                Err(_) => {
                    summary.decode_failures += 1;
                    continue;
                }
            };
            // Captures are stored at the same packet boundary as vanilla's
            // `MessageDeserializer`: a successful packet read is valid only when it consumed
            // the complete frame. Do this before inspecting `outcome.packet`, so a malformed
            // LoginSuccess cannot change the synthetic login and a malformed Play packet cannot
            // enter the replay queue.
            if outcome.unread != 0 {
                return Err(ProtoError::TrailingBytes {
                    packet_id: outcome.packet.id(),
                    remaining: outcome.unread,
                }
                .into());
            }
            match (frame.state, outcome.packet) {
                (State::Login, Decoded::LoginSuccess(success)) => {
                    login = Some(LoginOutcome {
                        uuid: success.uuid,
                        username: success.username,
                        compression_threshold: None,
                        encrypted: false,
                    });
                }
                (State::Play, Decoded::PlayClientbound(packet)) => {
                    let packet = *packet;
                    match &packet {
                        // Answered on the network thread and never forwarded; see `net::task`.
                        S2CPlay::KeepAlive(_) | S2CPlay::SetCompression(_) => continue,
                        S2CPlay::Disconnect(_) => {
                            summary.disconnects_dropped += 1;
                            continue;
                        }
                        S2CPlay::ChunkData(chunk) => {
                            summary.chunk_data += 1;
                            summary.sections += chunk.primary_bit_mask.count_ones() as usize;
                        }
                        S2CPlay::MapChunkBulk(bulk) => {
                            summary.chunk_bulk += 1;
                            summary.sections += bulk
                                .columns
                                .iter()
                                .map(|c| c.primary_bit_mask.count_ones() as usize)
                                .sum::<usize>();
                        }
                        _ => {}
                    }
                    packets.push_back(Timed {
                        micros: frame.micros,
                        packet,
                        raw: keep_bytes.then(|| Box::from(&frame.payload[..])),
                    });
                }
                _ => {}
            }
        }

        if packets.is_empty() {
            return Err(ClientError::Io(std::io::Error::other(format!(
                "{}: no replayable clientbound Play packet in this capture",
                path.display()
            ))));
        }

        // Normalise to the first replayed packet, so the run does not open with a stretch of
        // empty frames covering the login. Capture order is preserved: `micros` is only ever
        // compared against the virtual clock, never sorted on — two directions are buffered
        // independently and a sort would invent an ordering the session never had (the same
        // reasoning `mc_parity::replay` records for its own frame walk).
        let base = packets.iter().map(|p| p.micros).min().unwrap_or(0);
        let mut span = 0;
        for timed in &mut packets {
            timed.micros = timed.micros.saturating_sub(base);
            span = span.max(timed.micros);
        }
        summary.replayed = packets.len();
        summary.span_micros = span;

        let username = login
            .as_ref()
            .map_or_else(|| "Replay".to_owned(), |o| o.username.clone());
        let login = login.unwrap_or(LoginOutcome {
            uuid: uuid::Uuid::from_u128(0),
            username,
            compression_threshold: None,
            encrypted: false,
        });

        tracing::info!(path = %path.display(), summary = %summary.line(), "capture loaded for replay");
        Ok(ReplaySource {
            path,
            packets,
            now_micros: 0,
            step_micros: step_micros.max(1),
            login: Some(login),
            summary,
        })
    }

    /// The file this source was built from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// What the capture contained.
    #[must_use]
    pub fn summary(&self) -> &ReplaySummary {
        &self.summary
    }

    /// How many packets are still to be delivered.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.packets.len()
    }

    /// Whether every packet has been handed out.
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.packets.is_empty()
    }

    /// The virtual clock, in captured microseconds since the first replayed packet.
    #[must_use]
    pub fn virtual_micros(&self) -> u64 {
        self.now_micros
    }

    /// Advance the virtual clock by one step. Called once per
    /// [`Session::poll`](crate::session::Session::poll), i.e. once per frame.
    ///
    /// This is the *only* thing that makes time pass here. A run that never polls never
    /// advances, which is precisely why the replay does not vary with frame rate.
    pub fn advance(&mut self) {
        self.now_micros = self.now_micros.saturating_add(self.step_micros);
    }

    /// The next message, or `None` when nothing is due yet (or the capture is spent).
    pub fn next_message(&mut self) -> Option<ToGame> {
        self.next_message_tapped(false)
    }

    /// [`ReplaySource::next_message`], with the packet's exact bytes attached
    /// ([`ToGame::RawPacket`]) when `tap` is set and this source kept them.
    pub fn next_message_tapped(&mut self, tap: bool) -> Option<ToGame> {
        if let Some(login) = self.login.take() {
            return Some(ToGame::Connected(login));
        }
        // Strictly less, so each poll delivers the half-open window `[now - step, now)` and a
        // source that has never been advanced delivers nothing at all. With `<=` the packets
        // captured at time zero would escape before the clock had started, which is a
        // one-frame difference on a live run and an off-by-one in every test of the cadence.
        if self.packets.front()?.micros >= self.now_micros {
            return None;
        }
        // Moved, not cloned: a `MapChunkBulk` is hundreds of kilobytes and cloning one per
        // delivery would put an allocation the live path does not have into the measurement.
        let timed = self.packets.pop_front()?;
        Some(ToGame::Packet(timed.packet).with_raw(timed.raw.filter(|_| tap)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_parity::capture::{CaptureMeta, Recorder};
    use mc_protocol::play::s2c;
    use mc_protocol::{Packet, PacketBuffer, ProtoError};

    /// Encode one clientbound Play packet the way a capture stores it: id varint then body.
    fn payload(packet: &S2CPlay) -> Vec<u8> {
        let mut buf = PacketBuffer::new();
        buf.write_varint(packet.id());
        packet.write(&mut buf);
        buf.as_slice().to_vec()
    }

    fn chat(text: &str) -> S2CPlay {
        S2CPlay::Chat(s2c::Chat {
            component: format!(r#"{{"text":"{text}"}}"#),
            position: 0,
        })
    }

    /// A capture whose Play frames sit at the given captured microsecond offsets.
    fn capture_at(offsets: &[u64]) -> Capture {
        let mut rec = Recorder::new(CaptureMeta {
            protocol: 47,
            address: "test".into(),
            started_at_unix_millis: 0,
        });
        for (i, micros) in offsets.iter().enumerate() {
            rec.record_at(
                Direction::Clientbound,
                State::Play,
                *micros,
                &payload(&chat(&format!("m{i}"))),
            );
        }
        rec.finish()
    }

    fn drain(source: &mut ReplaySource) -> Vec<ToGame> {
        let mut out = Vec::new();
        while let Some(message) = source.next_message() {
            out.push(message);
        }
        out
    }

    #[test]
    fn the_first_message_is_always_connected() {
        let capture = capture_at(&[0]);
        let mut source = ReplaySource::from_capture(&capture, "x".into(), 1000).expect("a source");
        assert!(matches!(source.next_message(), Some(ToGame::Connected(_))));
    }

    /// The property the whole module exists for: which frame a packet lands on is a function of
    /// the poll count and nothing else. Two sources stepped the same number of times must have
    /// produced the identical release schedule.
    #[test]
    fn the_release_schedule_is_a_function_of_the_poll_count_alone() {
        let capture = capture_at(&[0, 500, 1_500, 1_600, 9_000]);
        let schedule = |step: u64, polls: usize| -> Vec<usize> {
            let mut source =
                ReplaySource::from_capture(&capture, "x".into(), step).expect("a source");
            let mut per_poll = Vec::new();
            for _ in 0..polls {
                source.advance();
                per_poll.push(drain(&mut source).len());
            }
            per_poll
        };

        let a = schedule(1_000, 12);
        let b = schedule(1_000, 12);
        assert_eq!(
            a, b,
            "the same file and the same poll count must replay identically"
        );
        // Poll 1 carries `Connected` plus the two packets in `[0, 1000)`; poll 2 carries the two
        // in `[1000, 2000)`; then nothing until the window reaches past 9000 µs on poll 10.
        assert_eq!(a, vec![3, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], "{a:?}");

        // The control: change the cadence and the schedule *must* change, or the assertion above
        // is measuring nothing.
        let faster = schedule(2_000, 12);
        assert_ne!(
            a, faster,
            "a different step must produce a different schedule"
        );
        assert_eq!(faster.iter().sum::<usize>(), a.iter().sum::<usize>());
    }

    /// Nothing is released until the clock is advanced — the property that makes the source
    /// independent of wall time.
    #[test]
    fn a_source_that_is_never_advanced_delivers_only_the_login() {
        let capture = capture_at(&[0, 1, 2]);
        let mut source = ReplaySource::from_capture(&capture, "x".into(), 1000).expect("a source");
        let first = drain(&mut source);
        assert_eq!(first.len(), 1, "only Connected");
        assert!(matches!(first[0], ToGame::Connected(_)));
        assert_eq!(source.remaining(), 3);
        assert!(!source.is_exhausted());
        source.advance();
        assert_eq!(drain(&mut source).len(), 3);
        assert!(source.is_exhausted());
    }

    /// The capture's own start offset must not become dead frames at the head of a run.
    #[test]
    fn timestamps_are_normalised_to_the_first_replayed_packet() {
        let capture = capture_at(&[5_000_000, 5_000_100]);
        let mut source = ReplaySource::from_capture(&capture, "x".into(), 1000).expect("a source");
        assert_eq!(source.summary().span_micros, 100);
        source.advance();
        assert_eq!(drain(&mut source).len(), 3, "Connected plus both packets");
    }

    /// The three packets the live task handles itself must not be replayed, and a Play
    /// `Disconnect` must be counted rather than silently swallowed.
    #[test]
    fn transport_packets_are_dropped_and_the_dropped_disconnect_is_counted() {
        let mut rec = Recorder::new(CaptureMeta {
            protocol: 47,
            address: "x".into(),
            started_at_unix_millis: 0,
        });
        for (micros, packet) in [
            (0u64, S2CPlay::KeepAlive(s2c::KeepAlive { id: 7 })),
            (
                1,
                S2CPlay::SetCompression(s2c::SetCompression { threshold: 256 }),
            ),
            (2, chat("kept")),
            (
                3,
                S2CPlay::Disconnect(s2c::Disconnect {
                    reason: r#"{"text":"bye"}"#.to_owned(),
                }),
            ),
        ] {
            rec.record_at(
                Direction::Clientbound,
                State::Play,
                micros,
                &payload(&packet),
            );
        }
        // A serverbound frame the client itself sent: replaying it would double it.
        rec.record_at(Direction::Serverbound, State::Play, 4, &{
            let mut buf = PacketBuffer::new();
            buf.write_varint(0x00);
            buf.write_varint(7);
            buf.as_slice().to_vec()
        });
        let capture = rec.finish();

        let mut source = ReplaySource::from_capture(&capture, "x".into(), 1000).expect("a source");
        assert_eq!(source.summary().replayed, 1);
        assert_eq!(source.summary().disconnects_dropped, 1);
        source.advance();
        let messages = drain(&mut source);
        assert_eq!(messages.len(), 2);
        assert!(matches!(messages[1], ToGame::Packet(S2CPlay::Chat(_))));
    }

    /// A file with nothing to replay is a configuration mistake, not an empty session.
    #[test]
    fn a_capture_with_no_play_traffic_is_an_error() {
        let mut rec = Recorder::new(CaptureMeta {
            protocol: 47,
            address: "x".into(),
            started_at_unix_millis: 0,
        });
        rec.record_at(Direction::Serverbound, State::Play, 0, &[0x00, 0x01]);
        let capture = rec.finish();
        assert!(ReplaySource::from_capture(&capture, "x".into(), 1000).is_err());
    }

    /// A capture frame is already the body passed to vanilla's `MessageDeserializer`; a valid
    /// packet prefix plus extra data must be rejected before a replay can create or queue it.
    #[test]
    fn a_clientbound_play_frame_with_trailing_bytes_is_rejected_before_replay() {
        let mut rec = Recorder::new(CaptureMeta {
            protocol: 47,
            address: "x".into(),
            started_at_unix_millis: 0,
        });
        let mut malformed = payload(&chat("prefix is valid"));
        malformed.push(0xAA);
        rec.record_at(Direction::Clientbound, State::Play, 0, &malformed);

        let err = ReplaySource::from_capture(&rec.finish(), "x".into(), 1000)
            .expect_err("a packet body with a trailing byte is malformed");
        assert!(matches!(
            err,
            ClientError::Protocol(ProtoError::TrailingBytes {
                packet_id: s2c::Chat::ID,
                remaining: 1,
            })
        ));
    }

    #[test]
    fn a_fully_consumed_clientbound_play_frame_still_builds_a_replay() {
        let capture = capture_at(&[0]);
        let source = ReplaySource::from_capture(&capture, "x".into(), 1000)
            .expect("the valid control must remain replayable");
        assert_eq!(source.summary().replayed, 1);
    }

    /// Chunk content is what decides whether a replay bench is heavy or trivial, so it is
    /// counted at load rather than guessed at.
    #[test]
    fn the_summary_counts_the_sections_a_capture_delivers() {
        let mut rec = Recorder::new(CaptureMeta {
            protocol: 47,
            address: "x".into(),
            started_at_unix_millis: 0,
        });
        let mask = 0b1011u16; // three sections
        let packet = S2CPlay::ChunkData(mc_protocol::chunk::ChunkData {
            chunk_x: 1,
            chunk_z: 2,
            ground_up_continuous: true,
            primary_bit_mask: mask,
            data: vec![0u8; mc_world::expected_chunk_data_len(mask, true, true)],
        });
        rec.record_at(Direction::Clientbound, State::Play, 0, &payload(&packet));
        let capture = rec.finish();

        let source = ReplaySource::from_capture(&capture, "x".into(), 1000).expect("a source");
        assert_eq!(source.summary().chunk_data, 1);
        assert_eq!(source.summary().sections, 3);
    }
}
