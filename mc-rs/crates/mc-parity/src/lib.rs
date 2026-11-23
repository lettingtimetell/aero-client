#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Mechanical proof of vanilla parity: packet capture and replay, simulation traces,
//! render snapshots, and a shared fuzz harness.
//!
//! # What this crate is for
//!
//! Every other section proves its own correctness in its own way. This crate exists so the
//! *technique* is reusable and so the claims are auditable from one place:
//!
//! | Module | What it gives you |
//! |---|---|
//! | [`capture`] | A framed, timestamped, direction-tagged, zlib-compressed capture container, a recorder that can sit on a live TCP byte stream, and a strict parser for untrusted capture files. |
//! | [`replay`] | Deterministic replay: every captured frame is decoded through `mc-protocol` and **re-encoded**, and the re-encoded bytes are compared with the captured ones. |
//! | [`synthetic`] | A scripted, realistically shaped protocol-47 session, standing in for the real capture this environment cannot produce. |
//! | [`trace`] | A generic per-step numeric trace format (`MCTRACE1`) and a comparator that is **bit-exact by default**. |
//! | [`physics_trace`] | The S13 Java-dumped physics fixtures, read into [`trace::Trace`] and compared with a live `mc-physics` replay through the generic comparator. |
//! | [`snapshot`] | Offscreen frame capture plus a real perceptual metric (CIEDE2000 ΔE₀₀ + SSIM), not a byte compare. |
//! | [`fuzz`] | Panic-freedom and round-trip oracles over NBT and protocol, with a deterministic driver so the same run reproduces without `proptest`. |
//! | [`hazards`] | The §12 hazard table as machine-checked data: each hazard names the file that proves it, and a test asserts those files exist and contain what is claimed. |
//! | [`matrix`] | The multi-server compatibility matrix's *shape*, as data. Deliberately **not executed** — see that module's honesty note. |
//!
//! # The honesty rule this crate lives by
//!
//! A parity harness that overstates its coverage is worse than no harness: it converts an
//! unknown into a false assurance. So every module here documents precisely what its tests
//! prove and what they do not. Two things in particular are **not** proven anywhere in this
//! crate, and no amount of code here can change that:
//!
//! * **No real capture exists.** This environment has no network and no Minecraft server of any
//!   kind. [`capture`] and [`replay`] are exercised against captures this crate builds itself
//!   from `mc-protocol`'s own encoders. That proves the container, the framing, the strict
//!   parser and the decode/re-encode loop are correct; it does not prove a real server's bytes
//!   decode. The moment someone records a real session, the same code answers that question
//!   with no changes.
//! * **No vanilla screenshot exists.** There is a real GPU here but no display and no running
//!   vanilla client. [`snapshot`]'s metric is proven on synthetic pairs with known differences;
//!   the reference images this workspace commits are its own output, so they prove
//!   *no-regression*, not *parity*. S11 recorded the same gap.
//!
//! The one place in this workspace where "parity" is meant literally is `mc-physics`: S13's
//! fixtures come out of the real recompiled 1.8.9 bytecode under a JDK, and [`physics_trace`]
//! re-proves them through this crate's generic comparator at tolerance `0.0`.

pub mod capture;
pub mod fuzz;
pub mod hazards;
pub mod matrix;
pub mod physics_trace;
pub mod replay;
pub mod snapshot;
pub mod synthetic;
pub mod trace;

pub use capture::{
    Capture, CaptureLimits, CaptureMeta, Direction, Frame, Recorder, StreamRecorder,
};
pub use hazards::{Hazard, HazardProof, HAZARDS};
pub use replay::{Decoded, ReplayReport, Replayer};
pub use snapshot::{perceptual_diff, PerceptualDiff, Snapshot};
pub use trace::{Tolerance, Trace, TraceDiff, Value};

use thiserror::Error;

/// Errors returned by `mc-parity`.
///
/// Capture files are **untrusted input** — they come off a wire, or off disk from whoever
/// recorded them. §11's "no `unwrap()` on anything derived from network input" therefore
/// applies to every parser in this crate, and every malformed-input path lands on one of the
/// variants below rather than on a panic. [`fuzz::capture_from_bytes`] is the oracle that keeps
/// that true, driven by `proptest` and by [`fuzz::run`]'s deterministic loop.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ParityError {
    /// A file did not start with the expected magic bytes.
    #[error("mc-parity: bad magic: expected {expected:?}, found {found:?}")]
    BadMagic {
        /// The magic this parser requires.
        expected: &'static str,
        /// The first bytes actually present (lossily rendered), or a short marker if the input
        /// was shorter than the magic itself.
        found: String,
    },

    /// The container's format version is newer (or older) than this build understands.
    #[error(
        "mc-parity: unsupported {container} format version {found} (this build reads {supported})"
    )]
    UnsupportedVersion {
        /// Which container: `"capture"` or `"trace"`.
        container: &'static str,
        /// The version byte read from the file.
        found: u8,
        /// The version this build implements.
        supported: u8,
    },

    /// The input ended before a fixed-size or length-prefixed field was satisfied.
    #[error("mc-parity: truncated while reading {what}: need {need} more bytes, {have} remain")]
    Truncated {
        /// The field being read when the input ran out.
        what: &'static str,
        /// How many bytes the field needed.
        need: usize,
        /// How many were left.
        have: usize,
    },

    /// A declared length exceeded a [`CaptureLimits`] ceiling. Hitting this is the parser
    /// refusing to allocate on a hostile file's say-so, not a bug.
    #[error("mc-parity: {what} is {value}, which exceeds the limit of {max}")]
    LimitExceeded {
        /// What was oversized.
        what: &'static str,
        /// The declared value.
        value: u64,
        /// The configured ceiling.
        max: u64,
    },

    /// A tag byte (direction, connection state, or trace value kind) was not a defined value.
    #[error("mc-parity: {what} tag {value} is not a defined value")]
    InvalidTag {
        /// Which tag.
        what: &'static str,
        /// The undefined byte.
        value: u8,
    },

    /// Bytes remained after the last structurally complete record.
    #[error("mc-parity: {count} trailing bytes after the end of {what}")]
    TrailingBytes {
        /// What had just finished parsing.
        what: &'static str,
        /// How many bytes were left over.
        count: usize,
    },

    /// A string field was not valid UTF-8.
    #[error("mc-parity: {0} is not valid UTF-8")]
    InvalidUtf8(&'static str),

    /// The declared frame/step count did not match the number actually decoded.
    #[error("mc-parity: {what}: header declares {declared} but the payload holds {actual}")]
    CountMismatch {
        /// What was being counted.
        what: &'static str,
        /// The header's claim.
        declared: usize,
        /// What the payload actually contained.
        actual: usize,
    },

    /// Two traces (or two snapshots) cannot be compared because their shapes differ.
    #[error("mc-parity: cannot compare: {0}")]
    ShapeMismatch(String),

    /// Compression or file I/O failed.
    #[error("mc-parity: i/o failed: {0}")]
    Io(#[from] std::io::Error),

    /// A PNG could not be decoded or encoded.
    #[error("mc-parity: image codec failed: {0}")]
    Image(String),

    /// Building or reading back from the GPU failed. Carries the `mc-render` error's message
    /// rather than the error itself so this crate's error type stays cheap to construct and
    /// `Send + Sync` without dragging wgpu's types into it.
    #[error("mc-parity: render: {0}")]
    Render(String),
}

impl From<mc_render::RenderError> for ParityError {
    fn from(e: mc_render::RenderError) -> Self {
        ParityError::Render(e.to_string())
    }
}
