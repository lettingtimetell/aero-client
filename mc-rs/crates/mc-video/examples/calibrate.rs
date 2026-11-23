//! Quality-preset calibration on real game frames.
//!
//! Reads raw NV12 frames (the export harness's `raw=` dump), encodes them at a range of quality
//! values and in CBR at a few bitrates, decodes each file with ffmpeg and prints bitrate and luma
//! PSNR against the source, so the export's "Small file / Balanced / High / Near lossless" presets
//! can be chosen by measurement.
//!
//! `cargo run -p mc-video --example calibrate -- <frames.nv12> <width> <height> <fps> [out dir]`

#[cfg(windows)]
fn main() {
    use std::io::Read;
    use std::process::Command;

    use mc_video::{EncoderConfig, EncoderThread, FrameRate, RateControl, VideoCodec};

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: calibrate <frames.nv12> <width> <height> <fps> [out dir]");
        std::process::exit(2);
    }
    let (w, h): (u32, u32) = (args[2].parse().unwrap(), args[3].parse().unwrap());
    let fps: u32 = args[4].parse().unwrap();
    let out = args
        .get(5)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("mc-video-calibrate"));
    std::fs::create_dir_all(&out).unwrap();
    let frame = mc_video::nv12::frame_bytes(w, h);
    let mut raw = Vec::new();
    std::fs::File::open(&args[1])
        .unwrap()
        .read_to_end(&mut raw)
        .unwrap();
    let frames: Vec<&[u8]> = raw.chunks_exact(frame).collect();
    let seconds = frames.len() as f64 / f64::from(fps);
    println!(
        "{} frames of {w}x{h} at {fps} fps ({seconds:.2} s)",
        frames.len()
    );
    let luma = (w * h) as usize;

    let run = |name: String, codec: VideoCodec, rate: RateControl| {
        let path = out.join(format!("{name}.mp4"));
        let mut config = EncoderConfig::new(&path, w, h, FrameRate::whole(fps), codec);
        config.rate = rate;
        let mut encoder = EncoderThread::spawn(config, None, 8).expect("encoder");
        for f in &frames {
            let mut buffer = encoder.buffer();
            buffer.copy_from_slice(f);
            encoder.submit(buffer).expect("submit");
        }
        let summary = encoder.finish().expect("finish");
        let decoded = Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(&path)
            .args(["-f", "rawvideo", "-pix_fmt", "nv12", "-"])
            .output()
            .expect("ffmpeg");
        let mut sum = 0.0;
        let mut n = 0;
        for (a, b) in frames.iter().zip(decoded.stdout.chunks_exact(frame)) {
            sum += mc_video::nv12::psnr(&a[..luma], &b[..luma]).min(99.0);
            n += 1;
        }
        let mbps = summary.bytes as f64 * 8.0 / seconds / 1e6;
        println!(
            "{name:<24} {:>8.2} Mbit/s  {:>6.2} dB  ({} of {} frames decoded, {})",
            mbps,
            sum / f64::from(n.max(1)),
            n,
            frames.len(),
            summary.info.name
        );
    };
    // `size=8,10,25`: the export's size targeting (plan, encode, retry) on these frames instead.
    if let Some(limits) = args.get(6).and_then(|a| a.strip_prefix("size=")) {
        use mc_video::{size, AacBitrate};
        for codec in [VideoCodec::H264, VideoCodec::H265] {
            for mb in limits.split(',').filter_map(|m| m.parse::<f64>().ok()) {
                let limit = (mb * 1e6) as u64;
                let plan = size::plan(
                    limit,
                    seconds,
                    frames.len() as u64,
                    Some(AacBitrate::Kbps128),
                );
                let mut bitrate = plan.video_bitrate;
                for pass in 1..=size::MAX_PASSES {
                    let path = out.join(format!("size_{}_{mb}MB_pass{pass}.mp4", codec.label()));
                    let mut config = EncoderConfig::new(&path, w, h, FrameRate::whole(fps), codec);
                    config.rate = RateControl::ConstantBitrate { bitrate };
                    config.audio = plan.audio.map(|bitrate| mc_video::AudioConfig {
                        bitrate,
                        ..mc_video::AudioConfig::default()
                    });
                    let mut encoder = EncoderThread::spawn(config, None, 8).expect("encoder");
                    for f in &frames {
                        let mut buffer = encoder.buffer();
                        buffer.copy_from_slice(f);
                        encoder.submit(buffer).expect("submit");
                    }
                    let summary = encoder.finish().expect("finish");
                    println!(
                        "{} {mb} MB pass {pass}: {} bytes ({:.1}% of the limit) at {:.2} Mbit/s, {}",
                        codec.label(),
                        summary.bytes,
                        summary.bytes as f64 * 100.0 / limit as f64,
                        f64::from(bitrate) / 1e6,
                        summary.info.name
                    );
                    match size::retry(
                        limit,
                        summary.bytes,
                        bitrate,
                        seconds,
                        frames.len() as u64,
                        plan.audio,
                    ) {
                        Some(next) => bitrate = next,
                        None => break,
                    }
                }
            }
        }
        return;
    }
    for codec in [VideoCodec::H264, VideoCodec::H265] {
        let tag = codec.label().replace('.', "");
        for quality in [40_u8, 50, 55, 60, 68, 75, 80, 85, 92, 98] {
            run(
                format!("{tag}_q{quality}"),
                codec,
                RateControl::Quality {
                    quality,
                    max_bitrate: None,
                },
            );
        }
        for mbps in [2_u32, 5, 10, 20] {
            run(
                format!("{tag}_cbr{mbps}"),
                codec,
                RateControl::ConstantBitrate {
                    bitrate: mbps * 1_000_000,
                },
            );
        }
    }
}

#[cfg(not(windows))]
fn main() {}
