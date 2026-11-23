//! A panic on the encoder thread is a failed export, not the game's crash.
//!
//! The client's panic hook records the first panic on any thread as the run's crash report unless
//! the thread is inside `mc_core::panic_scope` at that moment. The encoder thread's owner already
//! turns a panic into an error (`VideoError::WorkerGone`) and the export screen shows it, so the
//! thread must run contained, and its owner must still get the error.
//!
//! Needs Windows Media Foundation with an H.264 encoder; without one the test says so and passes
//! without checking (as `tests/encode.rs` does). No ffprobe needed.

#![cfg(windows)]

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use mc_video::{
    probe, AacBitrate, AudioConfig, AudioSource, EncoderConfig, EncoderThread, FrameRate,
    VideoCodec, VideoError,
};

/// The soundtrack source of a broken export: it panics the first time the encoder asks for audio,
/// after noting whether the encoder thread was contained at that moment (what the hook checks:
/// the hook runs before any unwinding).
struct PanickingAudio {
    fills: Arc<AtomicU32>,
    contained: Arc<AtomicBool>,
}

impl AudioSource for PanickingAudio {
    fn fill(&mut self, _out: &mut [i16]) {
        self.fills.fetch_add(1, Ordering::SeqCst);
        self.contained
            .store(mc_core::panic_scope::is_contained(), Ordering::SeqCst);
        panic!("test audio source failed");
    }
}

#[test]
fn an_encoder_thread_panic_is_contained_and_ends_the_export_with_an_error() {
    if probe::codec_support(VideoCodec::H264).is_none() {
        eprintln!("no H.264 encoder on this PC: skipping");
        return;
    }
    let dir = tempfile_dir();
    let path = dir.join("worker_panic.mp4");
    let mut config = EncoderConfig::new(&path, 320, 180, FrameRate::whole(30), VideoCodec::H264);
    config.audio = Some(AudioConfig {
        sample_rate: 48_000,
        channels: 2,
        bitrate: AacBitrate::Kbps160,
    });
    let partial = config.partial_path();
    let fills = Arc::new(AtomicU32::new(0));
    let contained = Arc::new(AtomicBool::new(false));
    let audio = PanickingAudio {
        fills: Arc::clone(&fills),
        contained: Arc::clone(&contained),
    };
    let mut encoder = EncoderThread::spawn(config, Some(Box::new(audio)), 4).expect("encoder");
    // Frame 0 needs no audio yet; finishing writes the audio up to the video's end, which asks
    // the source and so panics on the encoder thread.
    let buffer = encoder.buffer();
    encoder.submit(buffer).expect("first frame");
    let error = encoder.finish().expect_err("the export fails");
    assert!(
        matches!(&error, VideoError::WorkerGone(why) if why == "panicked"),
        "the owner gets the panic as before: {error:?}"
    );
    assert_eq!(fills.load(Ordering::SeqCst), 1, "the source was asked once");
    assert!(
        contained.load(Ordering::SeqCst),
        "the encoder thread panicked outside panic_scope: the client would record it as the \
         run's crash"
    );
    assert!(!path.exists(), "no finished file");
    assert!(!partial.exists(), "the partial file is gone");
}

fn tempfile_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("mc-video-worker-panic-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}
