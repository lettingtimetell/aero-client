//! Video and audio encoding for replay exports.
//!
//! The client hands it finished frames (NV12, converted on the GPU) and mixed audio at exact
//! timestamps; this crate turns them into an MP4 through Windows Media Foundation (hardware
//! H.264/H.265 when the GPU has an encoder, software otherwise, AAC audio), at any frame rate, and
//! can aim at a maximum file size. Unsafe COM calls live here, not in `mc-client` (which forbids
//! `unsafe`).
//!
//! # Pieces
//!
//! * [`EncoderConfig`] describes one output file; [`EncoderThread`] encodes it on a worker thread
//!   (frames in, bounded queue, buffers recycled, cancel deletes the partial file).
//! * [`Mp4Writer`] is the same encoder used directly on the calling thread (tests, tools).
//! * [`nv12`] has the NV12 layout and the CPU reference conversion the GPU shader must match.
//! * [`size`] turns a byte limit into a bitrate, and decides whether a finished file needs a second,
//!   smaller pass.
//! * [`mix`] is the offline audio mixer (resampling voices with pitch, gain and pan into a 48 kHz
//!   stereo track).
//! * [`probe`] lists the encoders this PC has (for the export dialog).
//!
//! "Replay contract" and the brief.
//!
//! # Frame rates
//!
//! Any rate works, 1000 and 2000 fps included: frame times are exact rationals in 100 ns units
//! (the MP4 track timescale), and the encoder is given the real rate. Where a codec level cannot
//! describe the rate at that size, the encoder is asked for the highest level it has; see
//! `mf/writer.rs` for what was measured.

#![warn(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

mod config;
mod error;
pub mod faststart;
pub mod mix;
pub mod nv12;
pub mod size;
mod thread;
pub mod tracks;

#[cfg(windows)]
mod mf;

pub use config::{
    AacBitrate, AudioConfig, EncodeSummary, EncoderConfig, EncoderInfo, EncoderPreference,
    FrameRate, GpuAdapter, RateControl, VideoCodec,
};
pub use error::VideoError;
pub use thread::{AudioSource, EncoderProgress, EncoderThread, SilentAudio};

#[cfg(windows)]
pub use mf::probe;
#[cfg(windows)]
pub use mf::verify;
#[cfg(windows)]
pub use mf::writer::Mp4Writer;

/// Encoders found on this PC (see [`probe::encoders`]).
#[cfg(not(windows))]
pub mod probe {
    use crate::VideoCodec;

    /// One encoder.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct EncoderEntry {
        /// What it encodes.
        pub codec: VideoCodec,
        /// Runs on the GPU.
        pub hardware: bool,
        /// Its name.
        pub name: String,
    }

    /// No encoders off Windows.
    pub fn encoders() -> Vec<EncoderEntry> {
        Vec::new()
    }

    /// Never available off Windows.
    pub fn media_foundation_available() -> bool {
        false
    }

    /// Never available off Windows.
    pub fn codec_support(_codec: VideoCodec) -> Option<bool> {
        None
    }
}

/// The encoder used on the calling thread. Off Windows it cannot be created.
#[cfg(not(windows))]
pub struct Mp4Writer {
    _private: (),
}

#[cfg(not(windows))]
impl Mp4Writer {
    /// Always fails: Media Foundation is Windows-only.
    pub fn create(_config: EncoderConfig) -> Result<Self, VideoError> {
        Err(VideoError::Unavailable(
            "video export needs Windows Media Foundation".to_owned(),
        ))
    }

    /// Unreachable off Windows.
    pub fn info(&self) -> &EncoderInfo {
        unreachable!("an Mp4Writer cannot be created off Windows")
    }

    /// Unreachable off Windows.
    pub fn write_frame(&mut self, _nv12: &[u8]) -> Result<(), VideoError> {
        unreachable!("an Mp4Writer cannot be created off Windows")
    }

    /// Unreachable off Windows.
    pub fn write_audio(&mut self, _pcm: &[i16]) -> Result<(), VideoError> {
        unreachable!("an Mp4Writer cannot be created off Windows")
    }

    /// Unreachable off Windows.
    pub fn finish(self) -> Result<EncodeSummary, VideoError> {
        unreachable!("an Mp4Writer cannot be created off Windows")
    }

    /// Unreachable off Windows.
    pub fn frames_written(&self) -> u64 {
        0
    }

    /// Unreachable off Windows.
    pub fn audio_frames_written(&self) -> u64 {
        0
    }

    /// Unreachable off Windows.
    pub fn video_frames_in_encoder(&self) -> Option<u64> {
        None
    }

    /// Unreachable off Windows.
    pub fn extra_audio_tracks(&self) -> usize {
        0
    }

    /// Unreachable off Windows.
    pub fn write_audio_track(&mut self, _track: usize, _pcm: &[i16]) -> Result<(), VideoError> {
        unreachable!("an Mp4Writer cannot be created off Windows")
    }
}
