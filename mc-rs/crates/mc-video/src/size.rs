//! Hitting a maximum file size (a 10 MB or 50 MB upload limit...).
//!
//! The plan: give the video a constant bitrate computed from the byte limit and the duration,
//! after taking off the audio, the MP4 index and a safety margin; encode; and if the finished
//! file is still over (an encoder that overshoots), encode again at a bitrate scaled down by how
//! much it missed. [`plan`] is the first pass, [`retry`] the second and later ones.

use crate::AacBitrate;

/// Fraction of the limit left unused on the first pass, for encoder overshoot. Measured on real
/// SkyWars frames (720p60, an orbiting camera, 4 s): NVENC's CBR came out 5-11% over its target
/// (H.264 +7%/+10%/+11% at 5/10/20 Mbit/s with a one-second buffer). A second pass re-renders the
/// whole range, so the first one aims low enough to fit almost always.
pub const FIRST_PASS_MARGIN: f64 = 0.10;

/// Extra fraction kept free on a retry, on top of the measured overshoot.
pub const RETRY_MARGIN: f64 = 0.03;

/// The lowest video bitrate an export will use, in bits per second. Below this H.264 falls apart
/// at any useful size, so a limit that would need less cannot be met honestly.
pub const MIN_VIDEO_BITRATE: u32 = 150_000;

/// How many passes an export makes at most before giving up on a size limit.
pub const MAX_PASSES: u32 = 3;

/// MP4 index (`moov`) and container overhead for a file, in bytes: a fixed part plus each video
/// frame's entries (`stsz`, `ctts`, chunk offsets) and each AAC frame's (`stsz`). Deliberately on
/// the high side; at 2000 fps it is a real part of a small file's budget.
#[must_use]
pub fn container_overhead_bytes(video_frames: u64, duration_s: f64, audio: bool) -> u64 {
    let aac_frames = if audio {
        (duration_s * 48_000.0 / 1024.0).ceil() as u64
    } else {
        0
    };
    16 * 1024 + video_frames * 20 + aac_frames * 8
}

/// What to encode with to stay under a byte limit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SizePlan {
    /// Constant video bitrate, bits per second.
    pub video_bitrate: u32,
    /// The AAC bitrate (lowered to 96 kbit/s when the video would otherwise starve), or `None`
    /// without audio.
    pub audio: Option<AacBitrate>,
    /// The size this plan expects, in bytes.
    pub expected_bytes: u64,
    /// `false` when the limit is so small for this duration that the video bitrate had to be
    /// raised to [`MIN_VIDEO_BITRATE`]: the file will be over the limit. The dialog should say so
    /// before exporting.
    pub fits: bool,
}

/// Bytes the audio track takes.
fn audio_bytes(audio: Option<AacBitrate>, duration_s: f64) -> u64 {
    audio.map_or(0, |a| {
        (f64::from(a.bytes_per_second()) * duration_s).ceil() as u64
    })
}

/// The first-pass plan for `max_bytes` over `duration_s` seconds with `video_frames` frames.
#[must_use]
pub fn plan(
    max_bytes: u64,
    duration_s: f64,
    video_frames: u64,
    audio: Option<AacBitrate>,
) -> SizePlan {
    let duration_s = duration_s.max(0.001);
    let overhead = container_overhead_bytes(video_frames, duration_s, audio.is_some());
    let budget = (max_bytes as f64 * (1.0 - FIRST_PASS_MARGIN)) as u64;
    let video_bits = |audio: Option<AacBitrate>| -> f64 {
        let used = overhead + audio_bytes(audio, duration_s);
        budget.saturating_sub(used) as f64 * 8.0 / duration_s
    };
    let mut audio = audio;
    let mut bits = video_bits(audio);
    // Starving video for 128 kbit/s audio is a bad trade: drop the audio to 96 kbit/s first.
    if audio.is_some() && bits < 1_000_000.0 {
        audio = Some(AacBitrate::Kbps96);
        bits = video_bits(audio);
    }
    let fits = bits >= f64::from(MIN_VIDEO_BITRATE);
    let video_bitrate = bits.clamp(f64::from(MIN_VIDEO_BITRATE), f64::from(u32::MAX)) as u32;
    let expected_bytes = (f64::from(video_bitrate) * duration_s / 8.0) as u64
        + audio_bytes(audio, duration_s)
        + overhead;
    SizePlan {
        video_bitrate,
        audio,
        expected_bytes,
        fits,
    }
}

/// After a pass at `video_bitrate` produced `actual_bytes` against a limit of `max_bytes`, the
/// video bitrate for the next pass, or `None` when the file fits (or no lower bitrate is left).
#[must_use]
pub fn retry(
    max_bytes: u64,
    actual_bytes: u64,
    video_bitrate: u32,
    duration_s: f64,
    video_frames: u64,
    audio: Option<AacBitrate>,
) -> Option<u32> {
    if actual_bytes <= max_bytes || video_bitrate <= MIN_VIDEO_BITRATE {
        return None;
    }
    let duration_s = duration_s.max(0.001);
    let fixed = container_overhead_bytes(video_frames, duration_s, audio.is_some())
        + audio_bytes(audio, duration_s);
    let actual_video = actual_bytes.saturating_sub(fixed).max(1) as f64;
    let target_video = (max_bytes as f64 * (1.0 - RETRY_MARGIN)) - fixed as f64;
    if target_video <= 0.0 {
        return Some(MIN_VIDEO_BITRATE);
    }
    let scaled = f64::from(video_bitrate) * (target_video / actual_video);
    // Always make real progress, even when the encoder's overshoot was tiny.
    let next = scaled.min(f64::from(video_bitrate) * 0.97);
    Some(next.max(f64::from(MIN_VIDEO_BITRATE)) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ten_megabytes_for_thirty_seconds() {
        let p = plan(10_000_000, 30.0, 1800, Some(AacBitrate::Kbps128));
        assert!(p.fits);
        assert_eq!(p.audio, Some(AacBitrate::Kbps128));
        // 9 MB budget, minus 480 kB of audio and ~60 kB of index: about 2.25 Mbit/s of video.
        assert!(
            (2_150_000..2_350_000).contains(&p.video_bitrate),
            "{}",
            p.video_bitrate
        );
        assert!(p.expected_bytes <= 9_100_000, "{}", p.expected_bytes);
    }

    #[test]
    fn high_frame_rates_pay_for_their_index() {
        let low = plan(10_000_000, 10.0, 600, None);
        let high = plan(10_000_000, 10.0, 20_000, None);
        assert!(high.video_bitrate < low.video_bitrate);
        assert!(high.expected_bytes <= 10_000_000);
    }

    #[test]
    fn starved_video_drops_audio_to_96k_then_reports_it_cannot_fit() {
        let p = plan(2_000_000, 60.0, 3600, Some(AacBitrate::Kbps192));
        assert_eq!(p.audio, Some(AacBitrate::Kbps96));
        let hopeless = plan(500_000, 300.0, 18_000, Some(AacBitrate::Kbps128));
        assert!(!hopeless.fits);
        assert_eq!(hopeless.video_bitrate, MIN_VIDEO_BITRATE);
    }

    #[test]
    fn retry_scales_by_the_overshoot_and_stops_when_it_fits() {
        assert_eq!(
            retry(10_000_000, 9_900_000, 2_400_000, 30.0, 1800, None),
            None
        );
        let next = retry(10_000_000, 11_000_000, 2_400_000, 30.0, 1800, None).unwrap();
        assert!(next < 2_400_000 * 9 / 10, "{next}");
        // A tiny overshoot still lowers the bitrate by at least 3%.
        let small = retry(10_000_000, 10_000_100, 2_400_000, 30.0, 1800, None).unwrap();
        assert!(small <= 2_400_000 * 97 / 100, "{small}");
    }
}
