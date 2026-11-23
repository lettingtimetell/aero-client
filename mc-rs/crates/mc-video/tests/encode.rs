//! Real encodes checked with ffprobe/ffmpeg: frame rates up to 2000 fps, H.264 and H.265, AAC,
//! frame order, colour, max-size targets, cancel.
//!
//! Needs Windows Media Foundation and `ffprobe`/`ffmpeg` on PATH; each test says so and passes
//! without checking when either is missing. Files go to `MC_VIDEO_SAMPLES_DIR` when set (kept for
//! inspection), else to a temp folder.

#![cfg(windows)]

use std::path::{Path, PathBuf};
use std::process::Command;

use mc_video::{
    probe, size, AacBitrate, AudioConfig, AudioSource, EncoderConfig, EncoderPreference,
    EncoderThread, FrameRate, RateControl, VideoCodec,
};

fn out_dir() -> PathBuf {
    let dir = std::env::var_os("MC_VIDEO_SAMPLES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("mc-video-tests"));
    std::fs::create_dir_all(&dir).expect("output dir");
    dir
}

fn tools_ready(codec: VideoCodec) -> bool {
    let ffprobe = Command::new("ffprobe")
        .arg("-version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !ffprobe {
        eprintln!("ffprobe not on PATH: skipping");
        return false;
    }
    if probe::codec_support(codec).is_none() {
        eprintln!("no {} encoder on this PC: skipping", codec.label());
        return false;
    }
    true
}

fn ffprobe(path: &Path) -> serde_json::Value {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=index,codec_type,codec_name,profile,level,width,height,r_frame_rate,\
             avg_frame_rate,nb_frames,nb_read_frames,duration,color_range,color_space,\
             color_transfer,color_primaries,sample_rate,channels:format=duration,size",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .expect("run ffprobe");
    assert!(
        out.status.success(),
        "ffprobe failed on {}: {}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("ffprobe JSON")
}

fn stream<'a>(probe: &'a serde_json::Value, kind: &str) -> &'a serde_json::Value {
    probe["streams"]
        .as_array()
        .expect("streams")
        .iter()
        .find(|s| s["codec_type"] == kind)
        .unwrap_or_else(|| panic!("no {kind} stream in {probe}"))
}

fn text<'a>(value: &'a serde_json::Value, key: &str) -> &'a str {
    value[key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} missing in {value}"))
}

fn number(value: &serde_json::Value, key: &str) -> f64 {
    match &value[key] {
        serde_json::Value::String(s) => s.parse().unwrap_or_else(|_| panic!("{key}={s}")),
        serde_json::Value::Number(n) => n.as_f64().expect("number"),
        other => panic!("{key} missing: {other} in {value}"),
    }
}

/// Bits drawn as 16x16 squares along the top row, so a decoded frame says which frame it is.
const INDEX_BITS: usize = 16;
const SQUARE: usize = 16;

/// Cheap synthetic NV12 frames (fast even unoptimised): a scrolling gradient, a moving coloured
/// box, the frame index in binary along the top, and optionally full-frame noise (the worst case
/// for rate control).
struct Synth {
    w: usize,
    h: usize,
    gradient: Vec<u8>,
    noise: Option<Vec<u8>>,
}

impl Synth {
    fn new(w: u32, h: u32, noisy: bool) -> Self {
        let (w, h) = (w as usize, h as usize);
        let gradient: Vec<u8> = (0..w * 2).map(|x| 16 + ((x % w) * 219 / w) as u8).collect();
        let noise = noisy.then(|| {
            let mut state = 0x2545_F491_4F6C_DD1D_u64;
            (0..w * h * 2)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    16 + (state % 220) as u8
                })
                .collect()
        });
        Self {
            w,
            h,
            gradient,
            noise,
        }
    }

    fn frame(&self, index: u64, fps: f64, out: &mut [u8]) {
        let (w, h) = (self.w, self.h);
        let t = index as f64 / fps;
        let (luma, chroma) = out.split_at_mut(w * h);
        match &self.noise {
            Some(noise) => {
                let offset = (index as usize * 4099 * 7) % (noise.len() - w * h);
                luma.copy_from_slice(&noise[offset..offset + w * h]);
            }
            None => {
                let shift = (t * 240.0) as usize % w;
                for row in luma.chunks_exact_mut(w) {
                    row.copy_from_slice(&self.gradient[shift..shift + w]);
                }
            }
        }
        chroma.fill(128);
        // The box: 1/6 of the height, crossing the frame every 2 s.
        let side = (h / 6).max(8) & !1;
        let x0 = ((t * 0.5).fract() * (w - side) as f64) as usize & !1;
        let y0 = (h / 2 + ((t * 3.0).sin() * (h / 4) as f64) as usize).min(h - side) & !1;
        for y in y0..y0 + side {
            luma[y * w + x0..y * w + x0 + side].fill(200);
        }
        for y in y0 / 2..(y0 + side) / 2 {
            for x in (x0 / 2)..(x0 + side) / 2 {
                chroma[y * w + x * 2] = 90;
                chroma[y * w + x * 2 + 1] = 210;
            }
        }
        // The index bits, last so nothing covers them.
        for bit in 0..INDEX_BITS {
            let on = (index >> bit) & 1 == 1;
            let value = if on { 235 } else { 16 };
            for y in 0..SQUARE {
                let start = y * w + bit * SQUARE;
                luma[start..start + SQUARE].fill(value);
            }
            for y in 0..SQUARE / 2 {
                let start = y * w + bit * SQUARE;
                chroma[start..start + SQUARE].fill(128);
            }
        }
    }
}

/// A 440 Hz tone on the left, 660 Hz on the right.
struct Tone {
    frame: u64,
    rate: u32,
}

impl AudioSource for Tone {
    fn fill(&mut self, out: &mut [i16]) {
        for pair in out.chunks_exact_mut(2) {
            let t = self.frame as f64 / f64::from(self.rate);
            pair[0] = ((t * 440.0 * std::f64::consts::TAU).sin() * 8000.0) as i16;
            pair[1] = ((t * 660.0 * std::f64::consts::TAU).sin() * 8000.0) as i16;
            self.frame += 1;
        }
    }
}

fn encode(
    config: EncoderConfig,
    frames: u64,
    synth: &Synth,
    audio: Option<Box<dyn AudioSource>>,
) -> mc_video::EncodeSummary {
    let fps = config.frame_rate.as_f64();
    let mut config = config;
    if std::env::var_os("MC_VIDEO_NO_FASTSTART").is_some() {
        config.faststart = false;
    }
    let mut encoder = EncoderThread::spawn(config, audio, 8).expect("start the encoder");
    eprintln!("encoder: {:?}", encoder.info());
    for index in 0..frames {
        let mut buffer = encoder.buffer();
        synth.frame(index, fps, &mut buffer);
        encoder.submit(buffer).expect("submit");
    }
    encoder.finish().expect("finish")
}

/// Decode the top strip of every frame and read back the frame indices.
fn decoded_indices(path: &Path, width: u32) -> Vec<u64> {
    let strip_w = INDEX_BITS * SQUARE;
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args([
            "-vf",
            &format!("crop={strip_w}:{SQUARE}:0:0"),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ])
        .output()
        .expect("run ffmpeg");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = width;
    let frame_len = strip_w * SQUARE;
    out.stdout
        .chunks_exact(frame_len)
        .map(|strip| {
            (0..INDEX_BITS).fold(0_u64, |acc, bit| {
                let centre = (SQUARE / 2) * strip_w + bit * SQUARE + SQUARE / 2;
                acc | (u64::from(strip[centre] > 128) << bit)
            })
        })
        .collect()
}

/// When the last of `frames` frames at `fps` starts, in 100 ns units.
fn config_last_time(fps: u32, frames: u64) -> i64 {
    FrameRate::whole(fps).frame_start_hns(frames.saturating_sub(1))
}

fn check_rate_and_count(codec: VideoCodec, fps: u32, width: u32, height: u32, seconds: f64) {
    let frames = (f64::from(fps) * seconds).round() as u64;
    let name = format!(
        "rate_{}_{fps}fps_{width}x{height}.mp4",
        codec.label().replace('.', "")
    );
    let path = out_dir().join(name);
    let mut config = EncoderConfig::new(&path, width, height, FrameRate::whole(fps), codec);
    config.rate = RateControl::Quality {
        quality: 70,
        max_bitrate: None,
    };
    let summary = encode(config, frames, &Synth::new(width, height, false), None);
    assert_eq!(summary.frames, frames);
    let probe = ffprobe(&path);
    let video = stream(&probe, "video");
    let expected_codec = match codec {
        VideoCodec::H264 => "h264",
        VideoCodec::H265 => "hevc",
    };
    eprintln!("{fps} fps {}: {video}", codec.label());
    assert_eq!(text(video, "codec_name"), expected_codec);
    assert_eq!(text(video, "r_frame_rate"), format!("{fps}/1"));
    assert_eq!(text(video, "avg_frame_rate"), format!("{fps}/1"));
    assert_eq!(
        number(video, "nb_read_frames") as u64,
        frames,
        "decoded frame count"
    );
    assert_eq!(number(video, "width") as u32, width);
    assert_eq!(number(video, "height") as u32, height);
    let duration = number(&probe["format"], "duration");
    assert!(
        (duration - seconds).abs() <= 1.0 / f64::from(fps) + 0.002,
        "duration {duration} for {seconds} s"
    );
    if std::env::var_os("MC_VIDEO_NO_FASTSTART").is_none() {
        assert!(
            mc_video::faststart::index_is_first(&path).expect("read boxes"),
            "fast start: moov before mdat"
        );
    }
    // Windows' own decoders (what the Media Player app uses) decode every frame too.
    let windows = mc_video::verify::decode_video(&path).expect("Windows decodes the file");
    assert_eq!(windows.frames, frames, "frames Windows decoded");
    let last = config_last_time(fps, frames);
    assert!(
        (windows.last_time_hns - last).abs() <= 1,
        "Windows' last frame time {} vs {last}",
        windows.last_time_hns
    );
    // Every frame decodes, in order, none dropped or duplicated.
    let indices = decoded_indices(&path, width);
    let expected: Vec<u64> = (0..frames).map(|i| i & 0xFFFF).collect();
    assert_eq!(indices.len(), expected.len(), "decoded frames");
    let first_bad = indices.iter().zip(&expected).position(|(a, b)| a != b);
    assert_eq!(first_bad, None, "frame order broken at {first_bad:?}");
}

#[test]
fn h264_at_30_60_240_1000_and_2000_fps() {
    if !tools_ready(VideoCodec::H264) {
        return;
    }
    for fps in [30, 60, 240, 1000, 2000] {
        let seconds = if fps >= 1000 { 1.5 } else { 2.0 };
        check_rate_and_count(VideoCodec::H264, fps, 1280, 720, seconds);
    }
}

#[test]
fn h265_at_30_60_240_1000_and_2000_fps() {
    if !tools_ready(VideoCodec::H265) {
        return;
    }
    for fps in [30, 60, 240, 1000, 2000] {
        let seconds = if fps >= 1000 { 1.5 } else { 2.0 };
        check_rate_and_count(VideoCodec::H265, fps, 1280, 720, seconds);
    }
}

#[test]
fn big_frames_at_high_rates() {
    // Beyond H.264 level 5.2 (1080p at 2000 fps, 4K at 1000 fps): the encoder runs at a lower
    // nominal rate and the file still carries the real one.
    if tools_ready(VideoCodec::H264) {
        check_rate_and_count(VideoCodec::H264, 2000, 1920, 1080, 0.1);
        check_rate_and_count(VideoCodec::H264, 1000, 3840, 2160, 0.05);
    }
    if tools_ready(VideoCodec::H265) {
        check_rate_and_count(VideoCodec::H265, 2000, 3840, 2160, 0.05);
    }
}

#[test]
fn an_awkward_rate_keeps_exact_time() {
    if !tools_ready(VideoCodec::H264) {
        return;
    }
    check_rate_and_count(VideoCodec::H264, 144, 640, 360, 3.0);
}

#[test]
fn audio_track_is_aac_and_ends_with_the_video() {
    if !tools_ready(VideoCodec::H264) {
        return;
    }
    let path = out_dir().join("audio_h264_60fps_tone.mp4");
    let mut config = EncoderConfig::new(&path, 640, 360, FrameRate::whole(60), VideoCodec::H264);
    config.audio = Some(AudioConfig {
        sample_rate: 48_000,
        channels: 2,
        bitrate: AacBitrate::Kbps128,
    });
    let summary = encode(
        config,
        180,
        &Synth::new(640, 360, false),
        Some(Box::new(Tone {
            frame: 0,
            rate: 48_000,
        })),
    );
    assert_eq!(summary.audio_frames, 144_000, "3 s of 48 kHz audio");
    let probe = ffprobe(&path);
    let audio = stream(&probe, "audio");
    eprintln!("{audio}");
    assert_eq!(text(audio, "codec_name"), "aac");
    assert_eq!(number(audio, "sample_rate") as u32, 48_000);
    assert_eq!(number(audio, "channels") as u32, 2);
    let audio_duration = number(audio, "duration");
    assert!(
        (audio_duration - 3.0).abs() < 0.05,
        "audio lasts {audio_duration}"
    );
    let total = number(&probe["format"], "duration");
    assert!((total - 3.0).abs() < 0.05, "file lasts {total}");
}

#[test]
fn colour_is_bt709_limited_and_decodes_back_to_the_source() {
    if !tools_ready(VideoCodec::H264) {
        return;
    }
    let (w, h) = (320_u32, 240_u32);
    // A still image of colour bars, encoded at high quality.
    let bars: [[u8; 3]; 8] = [
        [235, 235, 235],
        [235, 235, 16],
        [16, 235, 235],
        [16, 235, 16],
        [235, 16, 235],
        [235, 16, 16],
        [16, 16, 235],
        [40, 90, 160],
    ];
    let mut rgba = vec![0_u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let bar = bars[(x * 8 / w) as usize];
            let o = ((y * w + x) * 4) as usize;
            rgba[o..o + 3].copy_from_slice(&bar);
            rgba[o + 3] = 255;
        }
    }
    let mut nv12 = vec![0_u8; mc_video::nv12::frame_bytes(w, h)];
    mc_video::nv12::from_rgba(
        &rgba,
        w,
        h,
        (w * 4) as usize,
        mc_video::nv12::PixelOrder::Rgba,
        &mut nv12,
    );
    let path = out_dir().join("colour_bars_h264.mp4");
    let mut config = EncoderConfig::new(&path, w, h, FrameRate::whole(30), VideoCodec::H264);
    config.rate = RateControl::Quality {
        quality: 95,
        max_bitrate: None,
    };
    let mut encoder = EncoderThread::spawn(config, None, 4).expect("encoder");
    for _ in 0..30 {
        encoder.submit(nv12.clone()).expect("submit");
    }
    encoder.finish().expect("finish");
    let probe = ffprobe(&path);
    let video = stream(&probe, "video");
    eprintln!("{video}");
    assert_eq!(text(video, "color_range"), "tv");
    assert_eq!(text(video, "color_space"), "bt709");
    // Decode one frame with ffmpeg, which reads the stream's own colour tags.
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&path)
        .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgba", "-"])
        .output()
        .expect("ffmpeg");
    assert!(out.status.success());
    let decoded = out.stdout;
    assert_eq!(decoded.len(), rgba.len());
    let psnr = mc_video::nv12::psnr(&rgba, &decoded);
    eprintln!("colour bars PSNR {psnr:.1} dB");
    assert!(
        psnr > 30.0,
        "decoded colours differ from the source: {psnr:.1} dB"
    );
}

fn size_limited(limit: u64, seconds: f64, noisy: bool, codec: VideoCodec, tag: &str) {
    let (w, h, fps) = (1920_u32, 1080_u32, 60_u32);
    let frames = (f64::from(fps) * seconds) as u64;
    let audio = Some(AacBitrate::Kbps128);
    let plan = size::plan(limit, seconds, frames, audio);
    assert!(plan.fits);
    let mut bitrate = plan.video_bitrate;
    let synth = Synth::new(w, h, noisy);
    for pass in 1..=size::MAX_PASSES {
        let path = out_dir().join(format!(
            "size_{tag}_{}MB_{}_pass{pass}.mp4",
            limit / 1_000_000,
            codec.label().replace('.', "")
        ));
        let mut config = EncoderConfig::new(&path, w, h, FrameRate::whole(fps), codec);
        config.rate = RateControl::ConstantBitrate { bitrate };
        config.audio = plan.audio.map(|bitrate| AudioConfig {
            bitrate,
            ..AudioConfig::default()
        });
        let summary = encode(
            config,
            frames,
            &synth,
            Some(Box::new(Tone {
                frame: 0,
                rate: 48_000,
            })),
        );
        eprintln!(
            "{tag} {} MB pass {pass}: {} bytes at {bitrate} bit/s ({:.1}% of the limit, {})",
            limit / 1_000_000,
            summary.bytes,
            summary.bytes as f64 * 100.0 / limit as f64,
            summary.info.name
        );
        match size::retry(limit, summary.bytes, bitrate, seconds, frames, plan.audio) {
            None => {
                assert!(summary.bytes <= limit, "over the limit with no retry left");
                return;
            }
            Some(next) => bitrate = next,
        }
    }
    panic!(
        "{tag}: still over {limit} bytes after {} passes",
        size::MAX_PASSES
    );
}

#[test]
fn max_size_8_10_and_25_megabytes_are_met() {
    if !tools_ready(VideoCodec::H264) {
        return;
    }
    for (limit, seconds) in [(8_000_000, 20.0), (10_000_000, 30.0), (25_000_000, 30.0)] {
        size_limited(limit, seconds, false, VideoCodec::H264, "motion");
    }
    // Pure noise: the encoder cannot reach any quality, so this is all about rate control.
    size_limited(10_000_000, 10.0, true, VideoCodec::H264, "noise");
}

#[test]
fn max_size_is_met_with_h265_too() {
    if !tools_ready(VideoCodec::H265) {
        return;
    }
    size_limited(10_000_000, 15.0, true, VideoCodec::H265, "noise");
}

#[test]
fn cancelling_leaves_no_file_behind() {
    if !tools_ready(VideoCodec::H264) {
        return;
    }
    let path = out_dir().join("cancelled.mp4");
    let _ = std::fs::remove_file(&path);
    let config = EncoderConfig::new(&path, 640, 360, FrameRate::whole(60), VideoCodec::H264);
    let partial = config.partial_path();
    let synth = Synth::new(640, 360, false);
    let mut encoder = EncoderThread::spawn(config, None, 4).expect("encoder");
    for index in 0..30 {
        let mut buffer = encoder.buffer();
        synth.frame(index, 60.0, &mut buffer);
        encoder.submit(buffer).expect("submit");
    }
    assert!(partial.exists(), "the partial file exists while encoding");
    encoder.cancel();
    assert!(!partial.exists(), "partial file deleted");
    assert!(!path.exists(), "no finished file");
}

#[test]
fn software_encoder_works_too() {
    if !tools_ready(VideoCodec::H264) {
        return;
    }
    let path = out_dir().join("software_h264_60fps.mp4");
    let mut config = EncoderConfig::new(&path, 640, 360, FrameRate::whole(60), VideoCodec::H264);
    config.encoder = EncoderPreference::SoftwareOnly;
    let summary = encode(config, 60, &Synth::new(640, 360, false), None);
    assert!(!summary.info.hardware, "{:?}", summary.info);
    let probe = ffprobe(&path);
    assert_eq!(number(stream(&probe, "video"), "nb_read_frames") as u64, 60);
}
