//! The packet-capture container: format, recorder, and strict parser.
//!
//! # What a capture holds
//!
//! One capture is a single client↔server session. Each record is one **packet frame** as it
//! looked *after* decompression and *after* decryption — that is, the varint packet id followed
//! by the packet body, exactly the bytes [`mc_protocol`]'s decoders expect. Storing the plain
//! frame rather than the raw socket bytes is a deliberate choice:
//!
//! * a capture is replayable without the session key, so it can be committed and shared;
//! * replay does not depend on whether `SetCompression` had arrived yet, which is state that
//!   lives in the transport, not in the packet;
//! * the transport layers (`mc_protocol::crypt`, `mc_protocol::compress`) have their own
//!   dedicated tests in S07 and are not what a *parity* capture is trying to pin down.
//!
//! The connection state is recorded per frame, because a packet id means nothing without it.
//!
//! # Container layout (`MCCAP001`, all integers big-endian)
//!
//! ```text
//! off  size  field
//!   0     8  magic "MCCAP001"
//!   8     1  format version (1)
//!   9     1  flags — bit 0: the payload section is zlib-compressed
//!  10     4  protocol version (i32; 47 for 1.8.9)
//!  14     8  capture start, unix milliseconds (i64)
//!  22     2  server address length (u16)
//!  24     n  server address, UTF-8
//!   .     4  frame count (u32)
//!   .     4  uncompressed payload length (u32)
//!   .     4  stored payload length (u32)
//!   .     m  payload
//! ```
//!
//! Payload = `frame count` records of:
//!
//! ```text
//!   1  direction (0 = clientbound, 1 = serverbound)
//!   1  connection state (0 handshaking, 1 status, 2 login, 3 play)
//!   8  microseconds since capture start (u64)
//!   4  payload length (u32)
//!   n  packet frame: varint packet id, then the packet body
//! ```
//!
//! # Untrusted input
//!
//! [`Capture::parse`] never panics and never allocates on a declared length it has not first
//! checked against [`CaptureLimits`]. The compressed path decompresses through a bounded
//! [`Read::take`], so a zlib bomb is refused rather than exhausting memory. `tests/fuzz.rs`
//! throws random and mutated bytes at the parser to keep that honest.
//!
//! # Evidence
//!
//! Container tests use synthetic frames for parser limits. Retained real-server fixtures and
//! opt-in live client tests exercise independently produced bytes. Replay reports establish
//! only the packet shapes actually recorded.

use std::io::{Read, Write};
use std::time::Instant;

use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use mc_protocol::State;

use crate::ParityError;

/// The container's magic bytes.
pub const CAPTURE_MAGIC: &[u8; 8] = b"MCCAP001";

/// The container format version this build reads and writes.
pub const CAPTURE_VERSION: u8 = 1;

/// Bit 0 of the header's flags byte: the payload section is zlib-compressed.
const FLAG_COMPRESSED: u8 = 0x01;

/// Byte offset of the frame-count field, i.e. the fixed part of the header before the address.
const HEADER_FIXED_LEN: usize = 24;

/// Which way a frame was travelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Server → client.
    Clientbound,
    /// Client → server.
    Serverbound,
}

impl Direction {
    /// The wire tag stored in a capture.
    #[must_use]
    pub const fn tag(self) -> u8 {
        match self {
            Direction::Clientbound => 0,
            Direction::Serverbound => 1,
        }
    }

    /// Parse a wire tag.
    ///
    /// # Errors
    /// [`ParityError::InvalidTag`] if the byte is not 0 or 1.
    pub fn from_tag(tag: u8) -> Result<Direction, ParityError> {
        match tag {
            0 => Ok(Direction::Clientbound),
            1 => Ok(Direction::Serverbound),
            other => Err(ParityError::InvalidTag {
                what: "direction",
                value: other,
            }),
        }
    }
}

/// The wire tag for a connection state.
///
/// Deliberately written out here rather than derived from [`State`]'s declaration order: the
/// capture format must not change meaning if someone reorders that enum.
#[must_use]
pub const fn state_tag(state: State) -> u8 {
    match state {
        State::Handshaking => 0,
        State::Status => 1,
        State::Login => 2,
        State::Play => 3,
    }
}

/// Parse a connection-state wire tag.
///
/// # Errors
/// [`ParityError::InvalidTag`] if the byte is not 0–3.
pub fn state_from_tag(tag: u8) -> Result<State, ParityError> {
    match tag {
        0 => Ok(State::Handshaking),
        1 => Ok(State::Status),
        2 => Ok(State::Login),
        3 => Ok(State::Play),
        other => Err(ParityError::InvalidTag {
            what: "connection state",
            value: other,
        }),
    }
}

/// Ceilings applied while parsing an untrusted capture, so a hostile file cannot make the
/// parser allocate.
#[derive(Debug, Clone, Copy)]
pub struct CaptureLimits {
    /// Maximum number of frames a capture may declare.
    pub max_frames: u32,
    /// Maximum length of a single packet frame, in bytes. The default matches
    /// [`mc_protocol::codec::DEFAULT_MAX_FRAME_LEN`], which is what the live decoder enforces.
    pub max_frame_len: u32,
    /// Maximum total uncompressed payload length, in bytes.
    pub max_payload_len: u32,
    /// Maximum server-address length, in bytes.
    pub max_address_len: u16,
}

impl Default for CaptureLimits {
    /// Generous enough for a multi-hour session (a million frames, 256 MiB), tight enough that
    /// a corrupt length field is rejected instead of allocated.
    fn default() -> Self {
        CaptureLimits {
            max_frames: 1_000_000,
            max_frame_len: mc_protocol::codec::DEFAULT_MAX_FRAME_LEN as u32,
            max_payload_len: 256 * 1024 * 1024,
            max_address_len: 512,
        }
    }
}

/// What a capture is a capture *of*.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureMeta {
    /// Protocol version in use — 47 for 1.8.9.
    pub protocol: i32,
    /// The server address the session was against, `host:port` or free text for a synthetic
    /// capture. Purely informational; nothing in replay depends on it.
    pub address: String,
    /// Wall-clock start of the capture, unix milliseconds. Informational: frame timestamps are
    /// relative and are what replay uses.
    pub started_at_unix_millis: i64,
}

impl CaptureMeta {
    /// A capture of protocol 47 against `address`, starting "now" as far as the wall clock is
    /// concerned.
    #[must_use]
    pub fn new(address: impl Into<String>) -> CaptureMeta {
        let millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis().min(i64::MAX as u128) as i64);
        CaptureMeta {
            protocol: mc_protocol::PROTOCOL_VERSION,
            address: address.into(),
            started_at_unix_millis: millis,
        }
    }
}

/// One captured packet frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// Which way it was going.
    pub direction: Direction,
    /// The connection state it was read in. A packet id is meaningless without this.
    pub state: State,
    /// Microseconds since the start of the capture.
    pub micros: u64,
    /// The packet frame: a varint packet id followed by the packet body, post-decompression
    /// and post-decryption.
    pub payload: Vec<u8>,
}

impl Frame {
    /// The packet id, read as a leading varint.
    ///
    /// Returns `None` for an empty or malformed leading varint — a capture is untrusted input,
    /// so this is an `Option` rather than a panic.
    #[must_use]
    pub fn packet_id(&self) -> Option<i32> {
        read_varint(&self.payload).map(|(id, _)| id)
    }

    /// The packet body with the leading packet-id varint stripped.
    ///
    /// Returns `None` when the leading varint is missing or malformed.
    #[must_use]
    pub fn body(&self) -> Option<&[u8]> {
        read_varint(&self.payload).map(|(_, used)| &self.payload[used..])
    }
}

/// A parsed or freshly recorded session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capture {
    /// What this is a capture of.
    pub meta: CaptureMeta,
    frames: Vec<Frame>,
}

impl Capture {
    /// An empty capture.
    #[must_use]
    pub fn new(meta: CaptureMeta) -> Capture {
        Capture {
            meta,
            frames: Vec::new(),
        }
    }

    /// Every frame, in capture order.
    #[must_use]
    pub fn frames(&self) -> &[Frame] {
        &self.frames
    }

    /// How many frames this capture holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether this capture holds no frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Append a frame. Timestamps are not required to be monotonic — a capture taken from two
    /// independently buffered directions legitimately interleaves — so nothing is asserted here.
    pub fn push(&mut self, frame: Frame) {
        self.frames.push(frame);
    }

    /// Frames matching a direction, in order.
    pub fn frames_in(&self, direction: Direction) -> impl Iterator<Item = &Frame> {
        self.frames.iter().filter(move |f| f.direction == direction)
    }

    /// Serialise to the container format, zlib-compressing the payload section.
    ///
    /// # Errors
    /// [`ParityError::Io`] if the zlib encoder fails, which in practice it does not for an
    /// in-memory writer.
    pub fn encode(&self) -> Result<Vec<u8>, ParityError> {
        self.encode_with(true)
    }

    /// Serialise, choosing whether to compress. The uncompressed form exists so a diff of two
    /// captures is meaningful and so the fuzz corpus can mutate payload bytes directly.
    ///
    /// # Errors
    /// [`ParityError::Io`] if the zlib encoder fails.
    pub fn encode_with(&self, compress: bool) -> Result<Vec<u8>, ParityError> {
        let mut payload = Vec::new();
        for f in &self.frames {
            payload.push(f.direction.tag());
            payload.push(state_tag(f.state));
            payload.extend_from_slice(&f.micros.to_be_bytes());
            payload.extend_from_slice(&(f.payload.len() as u32).to_be_bytes());
            payload.extend_from_slice(&f.payload);
        }

        let stored = if compress {
            let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
            enc.write_all(&payload)?;
            enc.finish()?
        } else {
            payload.clone()
        };

        let address = self.meta.address.as_bytes();
        let mut out = Vec::with_capacity(HEADER_FIXED_LEN + address.len() + 12 + stored.len());
        out.extend_from_slice(CAPTURE_MAGIC);
        out.push(CAPTURE_VERSION);
        out.push(if compress { FLAG_COMPRESSED } else { 0 });
        out.extend_from_slice(&self.meta.protocol.to_be_bytes());
        out.extend_from_slice(&self.meta.started_at_unix_millis.to_be_bytes());
        // A >u16 address is a caller bug, not untrusted input; truncating on a char boundary
        // keeps `encode` infallible for that case rather than adding an error variant nothing
        // would ever match on.
        let address = &address[..address.len().min(u16::MAX as usize)];
        out.extend_from_slice(&(address.len() as u16).to_be_bytes());
        out.extend_from_slice(address);
        out.extend_from_slice(&(self.frames.len() as u32).to_be_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        out.extend_from_slice(&(stored.len() as u32).to_be_bytes());
        out.extend_from_slice(&stored);
        Ok(out)
    }

    /// Parse a capture with the default limits.
    ///
    /// # Errors
    /// Any [`ParityError`] parse variant. Never panics, whatever the bytes are.
    pub fn parse(bytes: &[u8]) -> Result<Capture, ParityError> {
        Capture::parse_with(bytes, CaptureLimits::default())
    }

    /// Parse a capture with explicit limits.
    ///
    /// # Errors
    /// [`ParityError::BadMagic`], [`ParityError::UnsupportedVersion`],
    /// [`ParityError::Truncated`], [`ParityError::LimitExceeded`], [`ParityError::InvalidTag`],
    /// [`ParityError::InvalidUtf8`], [`ParityError::CountMismatch`],
    /// [`ParityError::TrailingBytes`], or [`ParityError::Io`] from the zlib layer.
    pub fn parse_with(bytes: &[u8], limits: CaptureLimits) -> Result<Capture, ParityError> {
        let mut r = Cursor::new(bytes);

        let magic = r.take_bytes(8, "capture magic")?;
        if magic != CAPTURE_MAGIC {
            return Err(ParityError::BadMagic {
                expected: "MCCAP001",
                found: String::from_utf8_lossy(magic).into_owned(),
            });
        }
        let version = r.u8("format version")?;
        if version != CAPTURE_VERSION {
            return Err(ParityError::UnsupportedVersion {
                container: "capture",
                found: version,
                supported: CAPTURE_VERSION,
            });
        }
        let flags = r.u8("flags")?;
        let protocol = r.i32("protocol version")?;
        let started_at_unix_millis = r.i64("capture start")?;

        let address_len = r.u16("address length")?;
        if address_len > limits.max_address_len {
            return Err(ParityError::LimitExceeded {
                what: "server address length",
                value: u64::from(address_len),
                max: u64::from(limits.max_address_len),
            });
        }
        let address_bytes = r.take_bytes(address_len as usize, "server address")?;
        let address = std::str::from_utf8(address_bytes)
            .map_err(|_| ParityError::InvalidUtf8("the server address"))?
            .to_owned();

        let frame_count = r.u32("frame count")?;
        if frame_count > limits.max_frames {
            return Err(ParityError::LimitExceeded {
                what: "frame count",
                value: u64::from(frame_count),
                max: u64::from(limits.max_frames),
            });
        }
        let payload_len = r.u32("payload length")?;
        if payload_len > limits.max_payload_len {
            return Err(ParityError::LimitExceeded {
                what: "uncompressed payload length",
                value: u64::from(payload_len),
                max: u64::from(limits.max_payload_len),
            });
        }
        let stored_len = r.u32("stored payload length")?;
        let stored = r.take_bytes(stored_len as usize, "payload")?;
        if r.remaining() != 0 {
            return Err(ParityError::TrailingBytes {
                what: "the capture payload",
                count: r.remaining(),
            });
        }

        let payload = if flags & FLAG_COMPRESSED != 0 {
            // Bounded: a zlib bomb inflates to at most the declared length plus one byte, and
            // the mismatch is then reported rather than allocated.
            let mut out = Vec::with_capacity(payload_len as usize);
            ZlibDecoder::new(stored)
                .take(u64::from(payload_len) + 1)
                .read_to_end(&mut out)?;
            if out.len() != payload_len as usize {
                return Err(ParityError::CountMismatch {
                    what: "decompressed payload bytes",
                    declared: payload_len as usize,
                    actual: out.len(),
                });
            }
            out
        } else {
            if stored.len() != payload_len as usize {
                return Err(ParityError::CountMismatch {
                    what: "stored payload bytes",
                    declared: payload_len as usize,
                    actual: stored.len(),
                });
            }
            stored.to_vec()
        };

        let mut p = Cursor::new(&payload);
        let mut frames = Vec::with_capacity((frame_count as usize).min(4096));
        while p.remaining() != 0 {
            let direction = Direction::from_tag(p.u8("frame direction")?)?;
            let state = state_from_tag(p.u8("frame state")?)?;
            let micros = p.u64("frame timestamp")?;
            let len = p.u32("frame payload length")?;
            if len > limits.max_frame_len {
                return Err(ParityError::LimitExceeded {
                    what: "frame payload length",
                    value: u64::from(len),
                    max: u64::from(limits.max_frame_len),
                });
            }
            let payload = p.take_bytes(len as usize, "frame payload")?.to_vec();
            frames.push(Frame {
                direction,
                state,
                micros,
                payload,
            });
            if frames.len() > frame_count as usize {
                return Err(ParityError::CountMismatch {
                    what: "frames",
                    declared: frame_count as usize,
                    actual: frames.len(),
                });
            }
        }
        if frames.len() != frame_count as usize {
            return Err(ParityError::CountMismatch {
                what: "frames",
                declared: frame_count as usize,
                actual: frames.len(),
            });
        }

        Ok(Capture {
            meta: CaptureMeta {
                protocol,
                address,
                started_at_unix_millis,
            },
            frames,
        })
    }

    /// Read a capture from disk.
    ///
    /// # Errors
    /// [`ParityError::Io`] if the file cannot be read, then anything [`Capture::parse`] returns.
    pub fn read_file(path: impl AsRef<std::path::Path>) -> Result<Capture, ParityError> {
        Capture::parse(&std::fs::read(path)?)
    }

    /// Write this capture to disk, compressed.
    ///
    /// # Errors
    /// [`ParityError::Io`].
    pub fn write_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), ParityError> {
        std::fs::write(path, self.encode()?)?;
        Ok(())
    }
}

/// Builds a [`Capture`] frame by frame.
///
/// Two clocks are available. [`Recorder::record`] stamps against a monotonic [`Instant`] taken
/// when the recorder was constructed, which is what a live recording wants;
/// [`Recorder::record_at`] takes the timestamp as a parameter, which is what a *test* wants,
/// because a harness whose own output depends on wall-clock time cannot assert byte equality.
#[derive(Debug)]
pub struct Recorder {
    capture: Capture,
    started: Instant,
}

impl Recorder {
    /// Start recording.
    #[must_use]
    pub fn new(meta: CaptureMeta) -> Recorder {
        Recorder {
            capture: Capture::new(meta),
            started: Instant::now(),
        }
    }

    /// Record a frame, timestamped from the monotonic clock.
    pub fn record(&mut self, direction: Direction, state: State, payload: &[u8]) {
        let micros = self.started.elapsed().as_micros().min(u64::MAX as u128) as u64;
        self.record_at(direction, state, micros, payload);
    }

    /// Record a frame with an explicit timestamp, in microseconds since capture start.
    pub fn record_at(&mut self, direction: Direction, state: State, micros: u64, payload: &[u8]) {
        self.capture.push(Frame {
            direction,
            state,
            micros,
            payload: payload.to_vec(),
        });
    }

    /// How many frames have been recorded so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.capture.len()
    }

    /// Whether nothing has been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.capture.is_empty()
    }

    /// Finish, handing back the capture.
    #[must_use]
    pub fn finish(self) -> Capture {
        self.capture
    }
}

/// A [`Recorder`] that sits on a **live byte stream** and does its own frame splitting.
///
/// Feed it whatever the socket handed you, in whatever sizes it arrived — one frame, half a
/// frame, or nine frames and a fragment. It buffers per direction, splits on the varint length
/// prefix, and records each complete frame.
///
/// The varint reader here is written from the protocol description rather than reused from
/// `mc_protocol::codec`, on purpose: a harness that shares its framing with the crate under
/// test cannot catch a framing bug in it. S14 made the same call for its loopback test.
#[derive(Debug)]
pub struct StreamRecorder {
    recorder: Recorder,
    state: State,
    max_frame_len: u32,
    clientbound: Vec<u8>,
    serverbound: Vec<u8>,
}

impl StreamRecorder {
    /// Start recording a live stream. The initial state is [`State::Handshaking`], which is
    /// where every connection starts.
    #[must_use]
    pub fn new(meta: CaptureMeta) -> StreamRecorder {
        StreamRecorder {
            recorder: Recorder::new(meta),
            state: State::Handshaking,
            max_frame_len: CaptureLimits::default().max_frame_len,
            clientbound: Vec::new(),
            serverbound: Vec::new(),
        }
    }

    /// The state frames are currently being tagged with.
    #[must_use]
    pub fn state(&self) -> State {
        self.state
    }

    /// Move to a new connection state. The caller drives this from the same place the real
    /// client does — after writing the handshake, and after `LoginSuccess`.
    pub fn set_state(&mut self, state: State) {
        self.state = state;
    }

    /// Bytes buffered for a direction that do not yet form a complete frame.
    #[must_use]
    pub fn pending(&self, direction: Direction) -> usize {
        match direction {
            Direction::Clientbound => self.clientbound.len(),
            Direction::Serverbound => self.serverbound.len(),
        }
    }

    /// Feed raw stream bytes, timestamped from the monotonic clock. Returns how many complete
    /// frames were recorded.
    ///
    /// # Errors
    /// [`ParityError::LimitExceeded`] if a length prefix declares a frame larger than the
    /// configured ceiling — that is a desync or a hostile peer, and buffering on its say-so is
    /// exactly what the ceiling exists to prevent.
    pub fn push(&mut self, direction: Direction, bytes: &[u8]) -> Result<usize, ParityError> {
        let micros = self
            .recorder
            .started
            .elapsed()
            .as_micros()
            .min(u64::MAX as u128) as u64;
        self.push_at(direction, micros, bytes)
    }

    /// Feed raw stream bytes with an explicit timestamp. Every frame completed by this call is
    /// stamped with it.
    ///
    /// # Errors
    /// [`ParityError::LimitExceeded`], as [`StreamRecorder::push`].
    pub fn push_at(
        &mut self,
        direction: Direction,
        micros: u64,
        bytes: &[u8],
    ) -> Result<usize, ParityError> {
        let state = self.state;
        let max = self.max_frame_len;
        let buffer = match direction {
            Direction::Clientbound => &mut self.clientbound,
            Direction::Serverbound => &mut self.serverbound,
        };
        buffer.extend_from_slice(bytes);

        let mut consumed = 0usize;
        let mut recorded = 0usize;
        loop {
            let rest = &buffer[consumed..];
            let Some((len, used)) = read_varint(rest) else {
                // Either fewer than 5 bytes are present (wait for more) or the prefix is longer
                // than a varint may be, which `read_varint` reports the same way. Distinguish:
                // 5 readable bytes with no terminator is a malformed prefix.
                if rest.len() >= 5 {
                    return Err(ParityError::LimitExceeded {
                        what: "frame length prefix varint bytes",
                        value: rest.len() as u64,
                        max: 5,
                    });
                }
                break;
            };
            if len < 0 || len as u32 > max {
                return Err(ParityError::LimitExceeded {
                    what: "declared frame length",
                    value: u64::from(len as u32),
                    max: u64::from(max),
                });
            }
            let len = len as usize;
            if rest.len() < used + len {
                break;
            }
            self.recorder.capture.push(Frame {
                direction,
                state,
                micros,
                payload: rest[used..used + len].to_vec(),
            });
            recorded += 1;
            consumed += used + len;
        }
        buffer.drain(..consumed);
        Ok(recorded)
    }

    /// How many frames have been recorded so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.recorder.len()
    }

    /// Whether nothing has been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.recorder.is_empty()
    }

    /// Finish, handing back the capture. Any partial frame still buffered is discarded — it was
    /// never a packet.
    #[must_use]
    pub fn finish(self) -> Capture {
        self.recorder.finish()
    }
}

/// Encode `payload` as a length-prefixed stream frame, the shape [`StreamRecorder`] splits.
///
/// Present so tests (and a future live recorder's self-check) can build stream bytes without
/// reaching into `mc_protocol`'s codec.
#[must_use]
pub fn frame_bytes(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 5);
    write_varint(&mut out, payload.len() as i32);
    out.extend_from_slice(payload);
    out
}

/// Read a varint from the front of `bytes`, returning the value and how many bytes it used.
///
/// `None` means "not a complete, legal varint here" — either fewer bytes than the encoding
/// needs, or more than the 5 a 32-bit value can ever occupy. Written independently of
/// `mc_protocol`'s reader; see [`StreamRecorder`] for why.
#[must_use]
pub fn read_varint(bytes: &[u8]) -> Option<(i32, usize)> {
    let mut value: u32 = 0;
    for (i, byte) in bytes.iter().take(5).enumerate() {
        value |= u32::from(byte & 0x7F) << (7 * i);
        if byte & 0x80 == 0 {
            return Some((value as i32, i + 1));
        }
    }
    None
}

/// Append `value` as a varint. The inverse of [`read_varint`].
pub fn write_varint(out: &mut Vec<u8>, value: i32) {
    let mut v = value as u32;
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// A checked, panic-free reader over a byte slice. Every `take` reports how much was needed and
/// how much was left, which is what makes a truncated capture diagnosable instead of just
/// "invalid".
struct Cursor<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Cursor<'a> {
        Cursor { data, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.offset
    }

    fn take_bytes(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], ParityError> {
        if self.remaining() < n {
            return Err(ParityError::Truncated {
                what,
                need: n,
                have: self.remaining(),
            });
        }
        let out = &self.data[self.offset..self.offset + n];
        self.offset += n;
        Ok(out)
    }

    fn u8(&mut self, what: &'static str) -> Result<u8, ParityError> {
        Ok(self.take_bytes(1, what)?[0])
    }

    fn array<const N: usize>(&mut self, what: &'static str) -> Result<[u8; N], ParityError> {
        let bytes = self.take_bytes(N, what)?;
        let mut out = [0u8; N];
        out.copy_from_slice(bytes);
        Ok(out)
    }

    fn u16(&mut self, what: &'static str) -> Result<u16, ParityError> {
        Ok(u16::from_be_bytes(self.array::<2>(what)?))
    }

    fn u32(&mut self, what: &'static str) -> Result<u32, ParityError> {
        Ok(u32::from_be_bytes(self.array::<4>(what)?))
    }

    fn i32(&mut self, what: &'static str) -> Result<i32, ParityError> {
        Ok(i32::from_be_bytes(self.array::<4>(what)?))
    }

    fn u64(&mut self, what: &'static str) -> Result<u64, ParityError> {
        Ok(u64::from_be_bytes(self.array::<8>(what)?))
    }

    fn i64(&mut self, what: &'static str) -> Result<i64, ParityError> {
        Ok(i64::from_be_bytes(self.array::<8>(what)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> CaptureMeta {
        CaptureMeta {
            protocol: 47,
            address: "synthetic.invalid:25565".into(),
            started_at_unix_millis: 1_700_000_000_000,
        }
    }

    #[test]
    fn varint_round_trips_across_every_length_boundary() {
        for v in [
            0,
            1,
            127,
            128,
            255,
            16_383,
            16_384,
            2_097_151,
            2_097_152,
            268_435_455,
            268_435_456,
            i32::MAX,
            -1,
            i32::MIN,
        ] {
            let mut out = Vec::new();
            write_varint(&mut out, v);
            assert!(out.len() <= 5, "{v} encoded to {} bytes", out.len());
            assert_eq!(read_varint(&out), Some((v, out.len())), "{v}");
        }
    }

    #[test]
    fn an_incomplete_varint_is_none_not_a_panic() {
        assert_eq!(read_varint(&[]), None);
        assert_eq!(read_varint(&[0x80]), None);
        assert_eq!(read_varint(&[0x80, 0x80, 0x80, 0x80, 0x80]), None);
    }

    #[test]
    fn a_capture_round_trips_through_the_container() {
        let mut rec = Recorder::new(meta());
        rec.record_at(Direction::Serverbound, State::Handshaking, 0, &[0x00, 0x2f]);
        rec.record_at(
            Direction::Serverbound,
            State::Login,
            10,
            &[0x00, 0x03, b'a', b'b', b'c'],
        );
        rec.record_at(Direction::Clientbound, State::Login, 20, &[0x02]);
        rec.record_at(Direction::Clientbound, State::Play, 30, &[0x00, 0x01]);
        let capture = rec.finish();

        for compressed in [true, false] {
            let bytes = capture.encode_with(compressed).expect("encode");
            let back = Capture::parse(&bytes).expect("parse");
            assert_eq!(back, capture, "compressed={compressed}");
        }
    }

    #[test]
    fn encoding_is_deterministic() {
        let mut rec = Recorder::new(meta());
        for i in 0..64u64 {
            rec.record_at(
                Direction::Clientbound,
                State::Play,
                i * 50,
                &[0x00, i as u8],
            );
        }
        let capture = rec.finish();
        assert_eq!(
            capture.encode().expect("a"),
            capture.encode().expect("b"),
            "the same capture must serialise to the same bytes twice"
        );
    }

    #[test]
    fn frame_id_and_body_split_on_the_varint() {
        let f = Frame {
            direction: Direction::Clientbound,
            state: State::Play,
            micros: 0,
            payload: vec![0xA1, 0x01, 0xDE, 0xAD],
        };
        // 0xA1 0x01 = varint 0xA1 -> 0x21 | (1 << 7) = 161
        assert_eq!(f.packet_id(), Some(161));
        assert_eq!(f.body(), Some(&[0xDE, 0xAD][..]));

        let empty = Frame {
            payload: Vec::new(),
            ..f.clone()
        };
        assert_eq!(empty.packet_id(), None);
        assert_eq!(empty.body(), None);
    }

    #[test]
    fn the_stream_recorder_splits_frames_regardless_of_chunk_boundaries() {
        let payloads: Vec<Vec<u8>> = vec![
            vec![0x00, 0x2f, 0x09],
            vec![0x01],
            (0..300u16).map(|i| i as u8).collect(),
            vec![0x21, 0xff, 0xee],
        ];
        let mut stream = Vec::new();
        for p in &payloads {
            stream.extend_from_slice(&frame_bytes(p));
        }

        // Every chunk size from 1 byte at a time to the whole stream at once.
        for chunk in [1usize, 2, 3, 7, 64, 255, 256, stream.len()] {
            let mut rec = StreamRecorder::new(meta());
            rec.set_state(State::Play);
            let mut recorded = 0;
            for (i, part) in stream.chunks(chunk).enumerate() {
                recorded += rec
                    .push_at(Direction::Clientbound, i as u64, part)
                    .expect("push");
            }
            assert_eq!(recorded, payloads.len(), "chunk size {chunk}");
            assert_eq!(rec.pending(Direction::Clientbound), 0, "chunk size {chunk}");
            let capture = rec.finish();
            let got: Vec<Vec<u8>> = capture.frames().iter().map(|f| f.payload.clone()).collect();
            assert_eq!(got, payloads, "chunk size {chunk}");
        }
    }

    #[test]
    fn the_stream_recorder_keeps_the_two_directions_apart() {
        let mut rec = StreamRecorder::new(meta());
        rec.set_state(State::Play);
        // Interleave half-frames from both directions.
        let a = frame_bytes(&[0x0A, 0x01]);
        let b = frame_bytes(&[0x0B, 0x02, 0x03]);
        rec.push_at(Direction::Clientbound, 0, &a[..2]).expect("a1");
        rec.push_at(Direction::Serverbound, 1, &b[..1]).expect("b1");
        assert_eq!(rec.len(), 0);
        rec.push_at(Direction::Clientbound, 2, &a[2..]).expect("a2");
        rec.push_at(Direction::Serverbound, 3, &b[1..]).expect("b2");

        let capture = rec.finish();
        assert_eq!(capture.len(), 2);
        assert_eq!(capture.frames()[0].direction, Direction::Clientbound);
        assert_eq!(capture.frames()[0].payload, vec![0x0A, 0x01]);
        assert_eq!(capture.frames()[1].direction, Direction::Serverbound);
        assert_eq!(capture.frames()[1].payload, vec![0x0B, 0x02, 0x03]);
    }

    #[test]
    fn the_stream_recorder_tags_frames_with_the_state_they_arrived_in() {
        let mut rec = StreamRecorder::new(meta());
        rec.push_at(Direction::Serverbound, 0, &frame_bytes(&[0x00]))
            .expect("handshake");
        rec.set_state(State::Login);
        rec.push_at(Direction::Serverbound, 1, &frame_bytes(&[0x00]))
            .expect("login start");
        rec.set_state(State::Play);
        rec.push_at(Direction::Clientbound, 2, &frame_bytes(&[0x01]))
            .expect("play");

        let capture = rec.finish();
        let states: Vec<State> = capture.frames().iter().map(|f| f.state).collect();
        assert_eq!(
            states,
            vec![State::Handshaking, State::Login, State::Play],
            "the same packet id means different things in different states"
        );
    }

    #[test]
    fn an_oversized_length_prefix_is_refused_rather_than_buffered() {
        let mut rec = StreamRecorder::new(meta());
        let mut bytes = Vec::new();
        write_varint(&mut bytes, 64 * 1024 * 1024);
        let err = rec
            .push_at(Direction::Clientbound, 0, &bytes)
            .expect_err("a 64 MiB frame must be refused");
        assert!(
            matches!(err, ParityError::LimitExceeded { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn a_five_byte_prefix_with_no_terminator_is_an_error_not_an_infinite_buffer() {
        let mut rec = StreamRecorder::new(meta());
        let err = rec
            .push_at(
                Direction::Clientbound,
                0,
                &[0x80, 0x80, 0x80, 0x80, 0x80, 0x00],
            )
            .expect_err("malformed varint prefix");
        assert!(
            matches!(err, ParityError::LimitExceeded { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn parse_rejects_a_bad_magic() {
        let err = Capture::parse(b"NOTACAP!more bytes here").expect_err("bad magic");
        assert!(matches!(err, ParityError::BadMagic { .. }), "got {err:?}");
    }

    #[test]
    fn parse_rejects_a_future_version() {
        let capture = Capture::new(meta());
        let mut bytes = capture.encode().expect("encode");
        bytes[8] = CAPTURE_VERSION + 1;
        let err = Capture::parse(&bytes).expect_err("future version");
        assert!(
            matches!(err, ParityError::UnsupportedVersion { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn parse_rejects_every_truncation_of_a_valid_capture() {
        let mut rec = Recorder::new(meta());
        rec.record_at(Direction::Clientbound, State::Play, 5, &[0x00, 0x01, 0x02]);
        let bytes = rec.finish().encode().expect("encode");
        for cut in 0..bytes.len() {
            // Any prefix of a valid capture must be an error, never a panic and never a
            // silently-shorter success.
            let err = Capture::parse(&bytes[..cut]);
            assert!(err.is_err(), "a {cut}-byte prefix parsed successfully");
        }
    }

    #[test]
    fn parse_rejects_a_frame_count_that_disagrees_with_the_payload() {
        let mut rec = Recorder::new(meta());
        rec.record_at(Direction::Clientbound, State::Play, 0, &[0x00]);
        rec.record_at(Direction::Clientbound, State::Play, 1, &[0x01]);
        let capture = rec.finish();
        let mut bytes = capture.encode_with(false).expect("encode");
        // The frame count sits right after the address.
        let count_at = HEADER_FIXED_LEN + capture.meta.address.len();
        bytes[count_at..count_at + 4].copy_from_slice(&7u32.to_be_bytes());
        let err = Capture::parse(&bytes).expect_err("count mismatch");
        assert!(
            matches!(err, ParityError::CountMismatch { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn parse_rejects_an_undefined_state_tag() {
        let mut rec = Recorder::new(meta());
        rec.record_at(Direction::Clientbound, State::Play, 0, &[0x00]);
        let capture = rec.finish();
        let mut bytes = capture.encode_with(false).expect("encode");
        // First payload byte is the direction, second the state.
        let payload_at = HEADER_FIXED_LEN + capture.meta.address.len() + 12;
        bytes[payload_at + 1] = 9;
        let err = Capture::parse(&bytes).expect_err("bad state tag");
        assert!(matches!(err, ParityError::InvalidTag { .. }), "got {err:?}");
    }

    #[test]
    fn parse_refuses_an_absurd_declared_payload_length() {
        let capture = Capture::new(meta());
        let mut bytes = capture.encode_with(false).expect("encode");
        let payload_len_at = HEADER_FIXED_LEN + capture.meta.address.len() + 4;
        bytes[payload_len_at..payload_len_at + 4].copy_from_slice(&u32::MAX.to_be_bytes());
        let err = Capture::parse(&bytes).expect_err("absurd payload length");
        assert!(
            matches!(err, ParityError::LimitExceeded { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn direction_and_state_tags_round_trip() {
        for d in [Direction::Clientbound, Direction::Serverbound] {
            assert_eq!(Direction::from_tag(d.tag()).expect("tag"), d);
        }
        for s in [State::Handshaking, State::Status, State::Login, State::Play] {
            assert_eq!(state_from_tag(state_tag(s)).expect("tag"), s);
        }
        assert!(Direction::from_tag(2).is_err());
        assert!(state_from_tag(4).is_err());
    }
}
