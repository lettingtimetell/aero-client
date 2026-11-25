//! Optional recording of a live session into `mc-parity`'s `.mccap` container.
//!
//! *"Wire `mc_parity::StreamRecorder` into `mc-client`'s `net` task
//! behind a config flag or env var — feed it the post-decrypt, post-decompress bytes and call
//! `set_state` where the client already transitions."* This module is that wiring.
//!
//! # Where the bytes come from, and why it is this exact point
//!
//! The hook sits in `net::task::play_loop`, on the two places a whole packet payload
//! exists as a flat byte slice: immediately after `ConnectionReader::read_payload` returns, and
//! immediately before `ConnectionWriter::send_payload` is called. Both are **post-decryption and
//! post-decompression** — `mc_protocol::connection` decrypts on the way into its framing buffer
//! and decompresses on the way out of it — which is precisely the layer the capture format
//! stores, so a capture is replayable without the session key and independently of whether
//! `SetCompression` had arrived yet.
//!
//! # Two honest notes about the seam
//!
//! * [`StreamRecorder`] wants a **raw length-prefixed stream** and does its own varint frame
//!   splitting, deliberately written apart from `mc_protocol::codec` so a framing bug in the
//!   crate under test is catchable. By the time `mc-client` sees a payload the length prefix is
//!   already gone, so [`CaptureSink::record`] puts one back with `mc_parity`'s own
//!   [`frame_bytes`] and lets the recorder split it off again. That round trip is *not* an
//!   independent check of `mc_protocol`'s framing — it cannot be, from here. `mc_parity`'s
//!   plain `Recorder` would be the tighter fit for this call site; `StreamRecorder` is used
//!   because it is what carries `set_state`, and because it leaves the hook free to move down
//!   to the socket later without this file changing.
//! * The capture **begins at `Play`**. The handshake, status and login frames are read and
//!   written inside `mc_protocol::connection`'s own `handshake`/`login` helpers, which own that
//!   I/O and expose no observer, so they are not visible from this side. A capture therefore
//!   proves things about Play-state traffic and says nothing about the login sequence.
//!
//! # Failure policy
//!
//! Recording is diagnostics. Nothing here can end a session: a bad path, a full disk or a
//! recorder that refuses a frame disables the sink with a warning and the game plays on.

use std::path::{Path, PathBuf};

use mc_parity::capture::{frame_bytes, CaptureMeta, Direction, StreamRecorder};
use mc_protocol::State;

/// The environment variable that turns recording on: set it to the path of the `.mccap` file to
/// write.
///
/// An env var rather than a command-line option on purpose — vanilla's `Main.java` option set is
/// reproduced faithfully in [`crate::app::config`], and a debugging switch that no launcher will
/// ever pass does not belong in it.
pub const CAPTURE_ENV: &str = "MC_CAPTURE";

/// A session recorder that may or may not be recording.
///
/// The disabled form costs one `Option` check per packet and no allocation, so the hook can sit
/// unconditionally on the hot path.
#[derive(Debug)]
pub struct CaptureSink {
    inner: Option<Inner>,
    /// The replay recorder's tap flag: while it
    /// is set, [`CaptureSink::tapped`] copies each clientbound payload for the game thread.
    tap: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

#[derive(Debug)]
struct Inner {
    recorder: StreamRecorder,
    path: PathBuf,
}

impl CaptureSink {
    /// A sink that records nothing.
    #[must_use]
    pub fn disabled() -> CaptureSink {
        CaptureSink {
            inner: None,
            tap: None,
        }
    }

    /// Share the replay recorder's tap flag with this sink.
    pub fn set_raw_tap(&mut self, tap: std::sync::Arc<std::sync::atomic::AtomicBool>) {
        self.tap = Some(tap);
    }

    /// A copy of one clientbound payload (packet id varint then body,
    /// decrypted and decompressed) while the replay recorder's tap is on; `None`, after one
    /// relaxed load, while it is off.
    #[inline]
    #[must_use]
    pub fn tapped(&self, payload: &[u8]) -> Option<Box<[u8]>> {
        let on = self
            .tap
            .as_ref()
            .is_some_and(|tap| tap.load(std::sync::atomic::Ordering::Relaxed));
        on.then(|| Box::from(payload))
    }

    /// Read [`CAPTURE_ENV`] and start recording if it names a path.
    ///
    /// `address` is stored in the capture header as the session's `host:port`. It is purely
    /// informational — nothing in replay depends on it.
    #[must_use]
    pub fn from_env(address: &str) -> CaptureSink {
        match std::env::var_os(CAPTURE_ENV) {
            Some(path) if !path.is_empty() => CaptureSink::to_file(PathBuf::from(path), address),
            _ => CaptureSink::disabled(),
        }
    }

    /// Start recording to an explicit path.
    #[must_use]
    pub fn to_file(path: impl Into<PathBuf>, address: &str) -> CaptureSink {
        let path = path.into();
        tracing::info!(path = %path.display(), address, "recording this session");
        CaptureSink {
            inner: Some(Inner {
                recorder: StreamRecorder::new(CaptureMeta::new(address)),
                path,
            }),
            tap: None,
        }
    }

    /// [`CaptureSink::from_env`] for a connection speaking `protocol` (5 for 1.7.10), which the
    /// capture header records so a replay decodes it with that protocol's packets.
    #[must_use]
    pub fn from_env_for_protocol(address: &str, protocol: i32) -> CaptureSink {
        match std::env::var_os(CAPTURE_ENV) {
            Some(path) if !path.is_empty() => {
                CaptureSink::to_file_for_protocol(PathBuf::from(path), address, protocol)
            }
            _ => CaptureSink::disabled(),
        }
    }

    /// [`CaptureSink::to_file`] for a connection speaking `protocol`; see
    /// [`CaptureSink::from_env_for_protocol`].
    #[must_use]
    pub fn to_file_for_protocol(
        path: impl Into<PathBuf>,
        address: &str,
        protocol: i32,
    ) -> CaptureSink {
        let path = path.into();
        tracing::info!(path = %path.display(), address, protocol, "recording this session");
        CaptureSink {
            inner: Some(Inner {
                recorder: StreamRecorder::new(CaptureMeta {
                    protocol,
                    ..CaptureMeta::new(address)
                }),
                path,
            }),
            tap: None,
        }
    }

    /// Whether anything is being recorded.
    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.inner.is_some()
    }

    /// How many frames have been recorded so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.as_ref().map_or(0, |i| i.recorder.len())
    }

    /// Whether nothing has been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Tag every subsequent frame with a new connection state.
    ///
    /// Called from the one place the client itself transitions — after the login sequence
    /// returns, which is where `mc_protocol::connection::login` has just moved the connection to
    /// [`State::Play`]. A packet id is meaningless without the state it was read in.
    pub fn set_state(&mut self, state: State) {
        if let Some(inner) = &mut self.inner {
            inner.recorder.set_state(state);
        }
    }

    /// Record one whole packet payload: the varint packet id followed by the body.
    ///
    /// A recorder error disables the sink rather than propagating: see the module's failure
    /// policy.
    pub fn record(&mut self, direction: Direction, payload: &[u8]) {
        let Some(inner) = &mut self.inner else {
            return;
        };
        // The length prefix was stripped by the frame codec before this payload existed; the
        // recorder splits on one. See the module header.
        if let Err(error) = inner.recorder.push(direction, &frame_bytes(payload)) {
            tracing::warn!(%error, "capture disabled: the recorder refused a frame");
            self.inner = None;
        }
    }

    /// Record one frame reported by `mc_protocol::connection::login`'s observer hook.
    ///
    /// The `.mccap` capture used to begin at `Play` because
    /// `handshake`/`login` owned the pre-`Play` I/O and exposed no observer. `login` now takes
    /// an optional `&mut dyn FnMut(mc_protocol::connection::Direction, &[u8])`, called once per
    /// `Handshaking`/`Login`-state frame at exactly the level [`CaptureSink::record`] already
    /// expects — a VarInt packet id followed by the body, decrypted and decompressed for an
    /// inbound frame and not yet compressed or encrypted for an outbound one (see `login`'s own
    /// doc comment for why that is the right level).
    ///
    /// The only thing this method does beyond [`CaptureSink::record`] is translate
    /// `mc_protocol::connection::Direction` into this crate's [`Direction`]. The two enums have
    /// the exact same two variants but cannot be the same type: `mc-protocol` cannot depend on
    /// `mc-parity`, which depends on `mc-protocol` for `Connection` and the packet types — doing
    /// so would be a cycle. See `mc_protocol::connection::Direction`'s own doc for the other
    /// side of this note.
    pub fn record_login(&mut self, direction: mc_protocol::connection::Direction, payload: &[u8]) {
        let direction = match direction {
            mc_protocol::connection::Direction::Clientbound => Direction::Clientbound,
            mc_protocol::connection::Direction::Serverbound => Direction::Serverbound,
        };
        self.record(direction, payload);
    }

    /// Stop recording and write the capture out. A disabled sink does nothing.
    ///
    /// Returns the path written, so a caller (a test, in practice) can assert on it.
    pub fn finish(&mut self) -> Option<PathBuf> {
        let inner = self.inner.take()?;
        let frames = inner.recorder.len();
        let capture = inner.recorder.finish();
        match write_capture(&capture, &inner.path) {
            Ok(bytes) => {
                tracing::info!(
                    path = %inner.path.display(),
                    frames,
                    bytes,
                    "capture written"
                );
                Some(inner.path)
            }
            Err(error) => {
                tracing::warn!(path = %inner.path.display(), %error, "capture could not be written");
                None
            }
        }
    }
}

/// Write a capture, creating the parent directory if it does not exist, and report its size.
fn write_capture(capture: &mc_parity::Capture, path: &Path) -> Result<usize, std::io::Error> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let bytes = capture
        .encode()
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    std::fs::write(path, &bytes)?;
    Ok(bytes.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_parity::{Capture, Replayer};

    #[test]
    fn a_disabled_sink_records_nothing_and_writes_nothing() {
        let mut sink = CaptureSink::disabled();
        assert!(!sink.is_recording());
        sink.set_state(State::Play);
        sink.record(Direction::Clientbound, &[0x00, 0x01]);
        assert!(sink.is_empty());
        assert_eq!(sink.finish(), None);
    }

    /// The whole round trip this module exists for: payloads in, a parseable `.mccap` on disk,
    /// and `mc_parity`'s own replayer able to decode it.
    #[test]
    fn recorded_payloads_round_trip_through_a_file_and_replay_clean() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nested").join("session.mccap");

        let mut sink = CaptureSink::to_file(&path, "127.0.0.1:25565");
        sink.set_state(State::Play);
        // A real KeepAlive both ways, encoded by `mc-protocol` — the point being that these are
        // payload bytes with no length prefix, exactly as `play_loop` sees them.
        let mut clientbound = mc_protocol::PacketBuffer::new();
        clientbound.write_varint(0x00);
        clientbound.write_varint(0x4321);
        sink.record(Direction::Clientbound, clientbound.as_slice());

        let mut serverbound = mc_protocol::PacketBuffer::new();
        serverbound.write_varint(0x00);
        serverbound.write_varint(0x4321);
        sink.record(Direction::Serverbound, serverbound.as_slice());

        assert_eq!(sink.len(), 2);
        assert_eq!(sink.finish().as_ref(), Some(&path));
        assert!(sink.finish().is_none(), "finishing twice must be harmless");

        let capture = Capture::read_file(&path).expect("the capture must parse");
        assert_eq!(capture.len(), 2);
        assert_eq!(capture.meta.address, "127.0.0.1:25565");
        assert_eq!(capture.meta.protocol, 47);
        assert_eq!(capture.frames()[0].direction, Direction::Clientbound);
        assert_eq!(capture.frames()[1].direction, Direction::Serverbound);
        assert!(capture.frames().iter().all(|f| f.state == State::Play));

        let report = Replayer::new(&capture).run();
        assert!(report.is_clean(), "{report:?}");
    }

    /// The state tag has to follow `set_state`, or every id in the file is read in the wrong
    /// namespace.
    #[test]
    fn frames_carry_the_state_they_were_recorded_in() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("states.mccap");
        let mut sink = CaptureSink::to_file(&path, "x");
        sink.record(Direction::Serverbound, &[0x00]);
        sink.set_state(State::Play);
        sink.record(Direction::Clientbound, &[0x00, 0x2A]);
        sink.finish();

        let capture = Capture::read_file(&path).expect("parse");
        let states: Vec<State> = capture.frames().iter().map(|f| f.state).collect();
        assert_eq!(states, vec![State::Handshaking, State::Play]);
    }

    /// `record_login` must translate `mc_protocol::connection::Direction` faithfully and
    /// otherwise behave exactly like `record` — proven by tagging `State::Login` in between two
    /// calls, mirroring `net::task::connect_and_login`'s real usage: `set_state(Login)` right
    /// after the handshake, then every login-sequence frame through this method.
    #[test]
    fn record_login_bridges_mc_protocols_direction_and_tags_the_login_state() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("login.mccap");
        let mut sink = CaptureSink::to_file(&path, "x");

        sink.set_state(State::Login);
        sink.record_login(
            mc_protocol::connection::Direction::Serverbound,
            &[0x00, b'N'],
        );
        sink.record_login(mc_protocol::connection::Direction::Clientbound, &[0x02]);
        sink.set_state(State::Play);
        sink.record(Direction::Clientbound, &[0x00, 0x2A]);
        sink.finish();

        let capture = Capture::read_file(&path).expect("parse");
        assert_eq!(capture.len(), 3);
        assert_eq!(capture.frames()[0].direction, Direction::Serverbound);
        assert_eq!(capture.frames()[0].state, State::Login);
        assert_eq!(capture.frames()[1].direction, Direction::Clientbound);
        assert_eq!(capture.frames()[1].state, State::Login);
        assert_eq!(capture.frames()[2].state, State::Play);
    }

    /// A capture is diagnostics, not a feature: an unwritable path must not be able to take a
    /// session down with it.
    #[test]
    fn an_unwritable_path_is_a_warning_not_a_failure() {
        let dir = tempfile::tempdir().expect("tempdir");
        // A directory where the file should be: `fs::write` cannot replace it.
        let path = dir.path().join("occupied.mccap");
        std::fs::create_dir(&path).expect("mkdir");

        let mut sink = CaptureSink::to_file(&path, "x");
        sink.set_state(State::Play);
        sink.record(Direction::Clientbound, &[0x00, 0x2A]);
        assert_eq!(
            sink.finish(),
            None,
            "the write fails and is reported as None"
        );
        assert!(!sink.is_recording());
    }
}
