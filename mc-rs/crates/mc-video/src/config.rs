//! What one export file looks like: size, frame rate, codec, rate control, audio.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::VideoError;

/// 100 ns units per second: Media Foundation's clock, and the MP4 track timescale it writes.
pub const HNS_PER_SECOND: u64 = 10_000_000;

/// The video codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VideoCodec {
    /// H.264 / AVC, High profile. Plays everywhere.
    H264,
    /// H.265 / HEVC, Main profile. Smaller files; needs a hardware encoder (or the HEVC Video
    /// Extensions) on the exporting PC.
    H265,
}

impl VideoCodec {
    /// A short name for logs and the dialog.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::H264 => "H.264",
            Self::H265 => "H.265",
        }
    }
}

/// An exact frame rate, `num / den` frames per second.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameRate {
    /// Frames per `den` seconds.
    pub num: u32,
    /// Seconds per `num` frames.
    pub den: u32,
}

impl FrameRate {
    /// `num / den` frames per second.
    #[must_use]
    pub const fn new(num: u32, den: u32) -> Self {
        Self { num, den }
    }

    /// A whole number of frames per second.
    #[must_use]
    pub const fn whole(fps: u32) -> Self {
        Self { num: fps, den: 1 }
    }

    /// Frames per second as a float.
    #[must_use]
    pub fn as_f64(self) -> f64 {
        f64::from(self.num) / f64::from(self.den.max(1))
    }

    /// Whether both parts are non-zero.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.num > 0 && self.den > 0
    }

    /// When frame `index` starts, in 100 ns units: `index * den / num` seconds, rounded to the
    /// nearest unit. Exact rational arithmetic, so frame `n` of a 144 fps video never drifts.
    #[must_use]
    pub fn frame_start_hns(self, index: u64) -> i64 {
        let num = u128::from(self.num.max(1));
        let scaled = u128::from(index) * u128::from(HNS_PER_SECOND) * u128::from(self.den);
        ((scaled + num / 2) / num) as i64
    }

    /// How long frame `index` lasts, in 100 ns units (the gap to the next frame's start). For
    /// rates that divide 10^7 (25, 50, 100, 200, 250, 500, 1000, 2000...) every frame is the same
    /// length; for the others (30, 60, 144...) lengths differ by at most one unit and never
    /// drift.
    #[must_use]
    pub fn frame_duration_hns(self, index: u64) -> i64 {
        self.frame_start_hns(index + 1) - self.frame_start_hns(index)
    }

    /// When frame `index` starts, in microseconds since the first frame, rounded down.
    #[must_use]
    pub fn frame_start_micros(self, index: u64) -> u64 {
        let num = u128::from(self.num.max(1));
        (u128::from(index) * 1_000_000 * u128::from(self.den) / num) as u64
    }

    /// The number of whole frames that fit in `micros` microseconds (a range of exactly that
    /// length exports this many frames).
    #[must_use]
    pub fn frames_in_micros(self, micros: u64) -> u64 {
        let den = u128::from(self.den.max(1));
        (u128::from(micros) * u128::from(self.num) / (1_000_000 * den)) as u64
    }
}

/// How the encoder spends bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateControl {
    /// Constant quality (1 = worst, 100 = best): the size follows the content. The bitrate is
    /// still capped at `max_bitrate` bits per second when that is set.
    Quality {
        /// 1..=100.
        quality: u8,
        /// Optional cap in bits per second.
        max_bitrate: Option<u32>,
    },
    /// An average bitrate in bits per second, with peaks allowed up to 1.5x (peak-constrained
    /// VBR).
    Bitrate {
        /// Bits per second.
        average: u32,
    },
    /// A constant bitrate in bits per second: the most predictable size, used for size-limited
    /// exports.
    ConstantBitrate {
        /// Bits per second.
        bitrate: u32,
    },
}

/// Which encoders may be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncoderPreference {
    /// The GPU's encoder when there is one, else software.
    #[default]
    Auto,
    /// Only a hardware (GPU) encoder; fail if there is none.
    HardwareOnly,
    /// Only Windows' software encoder.
    SoftwareOnly,
}

/// The graphics adapter the game renders with, as the renderer reports
/// it. A hardware encoder then runs on that adapter instead of wherever Media Foundation's list
/// would put it (see `mf/device.rs`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct GpuAdapter {
    /// PCI vendor ID (0x10DE NVIDIA, 0x1002 AMD, 0x8086 Intel).
    pub vendor: u32,
    /// PCI device ID.
    pub device: u32,
    /// The adapter's name, to tell apart two adapters with the same IDs.
    pub name: String,
}

/// The AAC bitrates Windows' AAC encoder accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AacBitrate {
    /// 96 kbit/s.
    Kbps96,
    /// 128 kbit/s.
    #[default]
    Kbps128,
    /// 160 kbit/s.
    Kbps160,
    /// 192 kbit/s.
    Kbps192,
}

impl AacBitrate {
    /// Bits per second.
    #[must_use]
    pub fn bits_per_second(self) -> u32 {
        self.bytes_per_second() * 8
    }

    /// Bytes per second (what `MF_MT_AUDIO_AVG_BYTES_PER_SECOND` takes).
    #[must_use]
    pub fn bytes_per_second(self) -> u32 {
        match self {
            Self::Kbps96 => 12_000,
            Self::Kbps128 => 16_000,
            Self::Kbps160 => 20_000,
            Self::Kbps192 => 24_000,
        }
    }
}

/// The audio track: 16-bit PCM in, AAC-LC out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioConfig {
    /// 44100 or 48000.
    pub sample_rate: u32,
    /// 1 or 2.
    pub channels: u16,
    /// AAC bitrate.
    pub bitrate: AacBitrate,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            channels: 2,
            bitrate: AacBitrate::Kbps128,
        }
    }
}

/// One output file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderConfig {
    /// Where the finished MP4 goes.
    pub path: PathBuf,
    /// Width in pixels (even).
    pub width: u32,
    /// Height in pixels (even).
    pub height: u32,
    /// Frames per second, exact.
    pub frame_rate: FrameRate,
    /// H.264 or H.265.
    pub codec: VideoCodec,
    /// Bits.
    pub rate: RateControl,
    /// Hardware or software.
    pub encoder: EncoderPreference,
    /// Frames between keyframes; `None` = about one second's worth (capped at 600).
    pub keyframe_interval: Option<u32>,
    /// The audio track, if any.
    pub audio: Option<AudioConfig>,
    /// Put the index (`moov`) before the data, so the file starts playing before it has fully
    /// downloaded (chat apps, browsers). Costs one extra pass over the file when finishing.
    pub faststart: bool,
    /// The file's creation and modification time (the movie's, its tracks' and their media's;
    /// Windows Explorer's "Media created"). `None` keeps Media Foundation's: the moment the file
    /// was finished, so two encodes of the same frames differ in those bytes. A replay export
    /// gives the recording's own time, so exporting a range twice makes the same file. Written by
    /// the fast-start rewrite, so it needs `faststart`.
    pub created: Option<SystemTime>,
    /// More audio tracks after the first, one per name, in the
    /// `audio` format; their samples come from [`crate::AudioSource::fill_track`]. Empty (the
    /// default): one audio track. With extra tracks the first is named "Mix" and is the default
    /// track, and a copy of the mix follows the last one: Windows' players play a file's last
    /// audio track, browsers and editors its first ([`crate::tracks`]).
    pub audio_tracks: Vec<String>,
    /// The adapter the frames are rendered on. With a hardware
    /// encoder allowed, the encoder runs there. `None` (the default): Media Foundation's own
    /// choice, as before.
    pub adapter: Option<GpuAdapter>,
}

impl EncoderConfig {
    /// A config with sensible defaults: high quality, automatic encoder choice, no audio,
    /// fast-start on.
    #[must_use]
    pub fn new(
        path: impl Into<PathBuf>,
        width: u32,
        height: u32,
        frame_rate: FrameRate,
        codec: VideoCodec,
    ) -> Self {
        Self {
            path: path.into(),
            width,
            height,
            frame_rate,
            codec,
            rate: RateControl::Quality {
                quality: 80,
                max_bitrate: None,
            },
            encoder: EncoderPreference::Auto,
            keyframe_interval: None,
            audio: None,
            faststart: true,
            created: None,
            audio_tracks: Vec::new(),
            adapter: None,
        }
    }

    /// Check what Media Foundation would otherwise reject with an opaque error.
    ///
    /// # Errors
    /// [`VideoError::InvalidConfig`] naming the problem.
    pub fn validate(&self) -> Result<(), VideoError> {
        let invalid = |what: String| Err(VideoError::InvalidConfig(what));
        if self.width < 16 || self.height < 16 {
            return invalid(format!("{}x{} is too small", self.width, self.height));
        }
        if !self.width.is_multiple_of(2) || !self.height.is_multiple_of(2) {
            return invalid(format!(
                "{}x{}: width and height must be even (NV12)",
                self.width, self.height
            ));
        }
        if self.width > 8192 || self.height > 8192 {
            return invalid(format!("{}x{} is too large", self.width, self.height));
        }
        if !self.frame_rate.is_valid() {
            return invalid("the frame rate must be above zero".to_owned());
        }
        if self.frame_rate.as_f64() > 10_000.0 {
            return invalid(format!(
                "{:.0} fps is above the 10000 fps limit",
                self.frame_rate.as_f64()
            ));
        }
        match self.rate {
            RateControl::Quality { quality, .. } if !(1..=100).contains(&quality) => {
                return invalid(format!("quality {quality} is outside 1..=100"));
            }
            RateControl::Bitrate { average: bits }
            | RateControl::ConstantBitrate { bitrate: bits }
                if bits < 10_000 =>
            {
                return invalid(format!("{bits} bit/s is too low"));
            }
            _ => {}
        }
        if let Some(audio) = self.audio {
            if audio.sample_rate != 44_100 && audio.sample_rate != 48_000 {
                return invalid(format!(
                    "{} Hz audio: AAC takes 44100 or 48000",
                    audio.sample_rate
                ));
            }
            if audio.channels != 1 && audio.channels != 2 {
                return invalid(format!("{} audio channels: 1 or 2", audio.channels));
            }
        }
        if self.path.as_os_str().is_empty() {
            return invalid("no output path".to_owned());
        }
        Ok(())
    }

    /// Bytes in one NV12 frame of this size.
    #[must_use]
    pub fn nv12_frame_bytes(&self) -> usize {
        crate::nv12::frame_bytes(self.width, self.height)
    }

    /// The keyframe interval actually used.
    #[must_use]
    pub fn effective_keyframe_interval(&self) -> u32 {
        self.keyframe_interval
            .unwrap_or_else(|| self.frame_rate.as_f64().round().clamp(1.0, 600.0) as u32)
            .max(1)
    }

    /// Where the file is written while encoding; renamed to [`Self::path`] only once it is
    /// complete, so a cancelled or crashed export never leaves a broken `.mp4` behind.
    #[must_use]
    pub fn partial_path(&self) -> PathBuf {
        partial_path_for(&self.path)
    }
}

/// `<path>.partial`: the in-progress name for `path`.
#[must_use]
pub fn partial_path_for(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".partial");
    PathBuf::from(name)
}

/// The encoder Media Foundation actually picked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderInfo {
    /// The codec.
    pub codec: VideoCodec,
    /// Whether it runs on the GPU.
    pub hardware: bool,
    /// Its friendly name, for example "NVIDIA H.264 Encoder MFT".
    pub name: String,
    /// Whether the file has an audio track.
    pub audio: bool,
    /// The rate the encoder was configured with. Equal to the real rate unless the encoder
    /// refused it (an H.264 level's macroblock rate at high frame rates, or Windows' software
    /// encoders above 120 fps); the file's timestamps always carry the real rate.
    pub nominal_rate: FrameRate,
}

/// What a finished encode produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeSummary {
    /// The finished file.
    pub path: PathBuf,
    /// Its size in bytes.
    pub bytes: u64,
    /// Video frames written.
    pub frames: u64,
    /// Video duration in 100 ns units.
    pub duration_hns: i64,
    /// Audio sample frames written (per channel).
    pub audio_frames: u64,
    /// The encoder used.
    pub info: EncoderInfo,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_times_are_exact_for_rates_that_divide_ten_million() {
        for fps in [25, 50, 100, 200, 250, 400, 500, 1000, 2000] {
            let rate = FrameRate::whole(fps);
            let step = 10_000_000 / i64::from(fps);
            for index in [0_u64, 1, 2, 999, 123_456] {
                assert_eq!(
                    rate.frame_start_hns(index),
                    index as i64 * step,
                    "{fps} fps"
                );
                assert_eq!(rate.frame_duration_hns(index), step, "{fps} fps");
            }
        }
    }

    #[test]
    fn frame_times_never_drift_for_rates_that_do_not_divide() {
        let rate = FrameRate::whole(144);
        // One hour of 144 fps: the start of frame n is n/144 s to the nearest 100 ns.
        let n = 144 * 3600;
        assert_eq!(rate.frame_start_hns(n), 36_000_000_000);
        let durations: std::collections::BTreeSet<i64> =
            (0..1000).map(|i| rate.frame_duration_hns(i)).collect();
        assert!(
            durations.iter().all(|d| (69_444..=69_445).contains(d)),
            "{durations:?}"
        );
        let ntsc = FrameRate::new(60_000, 1001);
        assert_eq!(ntsc.frame_start_hns(60_000), 10_010_000_000);
    }

    #[test]
    fn micros_and_frame_counts_agree() {
        let rate = FrameRate::whole(2000);
        assert_eq!(rate.frames_in_micros(1_000_000), 2000);
        assert_eq!(rate.frame_start_micros(2000), 1_000_000);
        assert_eq!(rate.frame_start_micros(1), 500);
        let rate = FrameRate::whole(144);
        assert_eq!(rate.frames_in_micros(30_000_000), 4320);
        assert_eq!(rate.frame_start_micros(1), 6944);
    }

    #[test]
    fn validation_catches_what_media_foundation_would_reject() {
        let base = EncoderConfig::new("x.mp4", 1920, 1080, FrameRate::whole(60), VideoCodec::H264);
        assert!(base.validate().is_ok());
        let mut odd = base.clone();
        odd.width = 1921;
        assert!(odd.validate().is_err());
        let mut zero_rate = base.clone();
        zero_rate.frame_rate = FrameRate::new(0, 1);
        assert!(zero_rate.validate().is_err());
        let mut bad_audio = base.clone();
        bad_audio.audio = Some(AudioConfig {
            sample_rate: 22_050,
            ..AudioConfig::default()
        });
        assert!(bad_audio.validate().is_err());
        let mut fast = base;
        fast.frame_rate = FrameRate::whole(2000);
        assert!(fast.validate().is_ok());
    }

    #[test]
    fn partial_path_appends_a_suffix() {
        assert_eq!(
            partial_path_for(Path::new("C:/Videos/clip.mp4")),
            PathBuf::from("C:/Videos/clip.mp4.partial")
        );
    }
}
