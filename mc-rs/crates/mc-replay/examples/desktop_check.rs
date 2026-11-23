//! Open the real desktop audio capture (Windows process loopback) for a moment, check that a
//! continuous 24 kHz mono stream arrives, and push it through the encoder and a real replay file
//! that is read back and **deleted** right after. Nothing is played.
//!
//! ```text
//! cargo run -p mc-replay --example desktop_check -- <scratch dir> [all|discord] [seconds]
//! ```
//!
//! The scratch dir is where the temporary replay goes (it is removed again); the seconds default
//! to 2 and are capped at 5.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use mc_replay::desktop::{self, DesktopCapture, DesktopEncoder, DesktopSource};
use mc_replay::{Reader, ReplayMeta, StreamId, Writer, WriterOptions};

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(dir) = args.next().map(PathBuf::from) else {
        eprintln!("usage: desktop_check <scratch dir> [all|discord] [seconds]");
        std::process::exit(2);
    };
    let mode = args.next();
    if mode.as_deref() == Some("stress") {
        // Many short captures in one process, then the whole heap is validated: the activation
        // once corrupted it.
        let rounds = args
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10u32)
            .min(50);
        let started = Instant::now();
        let mut delivered = 0;
        for _ in 0..rounds {
            let capture = DesktopCapture::start(DesktopSource::AllExceptGame);
            std::thread::sleep(Duration::from_millis(150));
            delivered += capture.delivered();
            if let Some(error) = capture.error() {
                println!("error: {error}");
            }
            capture.stop();
        }
        println!(
            "{rounds} captures opened and closed in {:.2} s, {delivered} samples; heap valid: {}",
            started.elapsed().as_secs_f64(),
            desktop::heap_is_valid()
        );
        return;
    }
    let source = match mode.as_deref() {
        Some("discord") => DesktopSource::Discord,
        _ => DesktopSource::AllExceptGame,
    };
    let seconds = args
        .next()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(2.0)
        .clamp(0.2, 5.0);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join("desktop_check.aeroreplay");

    let started = Instant::now();
    let capture = DesktopCapture::start(source);
    let mut meta = ReplayMeta {
        title: "desktop audio check".into(),
        ..ReplayMeta::default()
    };
    let mut writer = Writer::create(&path, &meta, WriterOptions::default()).expect("writer");
    let mut encoder = DesktopEncoder::default();
    let mut pending = Vec::new();
    let mut received = 0u64;
    let mut first_at = None;
    let mut loudest = 0f64;
    let mut statuses = Vec::new();
    while started.elapsed() < Duration::from_secs_f64(seconds) {
        std::thread::sleep(Duration::from_millis(10));
        let status = capture.status();
        if statuses.last() != Some(&status) {
            println!(
                "{:>6.3} s  status {status:?}",
                started.elapsed().as_secs_f64()
            );
            statuses.push(status);
        }
        let before = pending.len();
        capture.drain(&mut pending);
        let got = pending.len() - before;
        if got > 0 && first_at.is_none() {
            first_at = Some(started.elapsed());
        }
        received += got as u64;
        while pending.len() >= desktop::FRAME_SAMPLES {
            let frame: Vec<i16> = pending.drain(..desktop::FRAME_SAMPLES).collect();
            let rms = (frame.iter().map(|&s| f64::from(s).powi(2)).sum::<f64>()
                / frame.len() as f64)
                .sqrt();
            loudest = loudest.max(rms);
            let now = started.elapsed().as_micros() as u64;
            encoder.push(&frame, |_, bytes| {
                let mut record = 0u32.to_le_bytes().to_vec();
                record.extend_from_slice(bytes);
                writer.push(now, StreamId::Desktop, &record);
            });
        }
    }
    let elapsed = started.elapsed();
    let stop_at = Instant::now();
    capture.stop();
    let stop_took = stop_at.elapsed();
    meta.duration = elapsed.as_micros() as u64;
    meta.has_desktop = encoder.kept > 0;
    meta.desktop_format = meta.has_desktop.then(desktop::format);
    writer
        .finish(meta)
        .wait(Duration::from_secs(5))
        .expect("file written");

    // Read it back: every record decodes.
    let reader = Reader::open(&path).expect("reads back");
    let mut records = 0;
    let mut samples = Vec::new();
    for record in reader.records() {
        let record = record.expect("record");
        if record.stream == StreamId::Desktop {
            desktop::decode_record(&record.payload, &mut samples).expect("decodes");
            records += 1;
        }
    }
    let size = std::fs::metadata(&path).map_or(0, |m| m.len());
    std::fs::remove_file(&path).expect("temporary replay deleted");
    let rate = received as f64 / elapsed.as_secs_f64();
    println!("source: {}", source.label());
    println!(
        "first samples after {:?}; {received} samples in {:.2} s = {rate:.0} samples/s",
        first_at,
        elapsed.as_secs_f64()
    );
    println!(
        "frames encoded {}, kept {} (loudest frame RMS {loudest:.1}; the gate is 16)",
        encoder.frames(),
        encoder.kept
    );
    println!(
        "file {size} bytes, {records} desktop records read back ({} samples), deleted: {}",
        samples.len(),
        !path.exists()
    );
    println!("stop took {stop_took:?}");
}
