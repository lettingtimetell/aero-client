//! A per-frame timeline of the export's encoder thread, to find where
//! a hardware-encoder export stalls.
//!
//! Feeds NV12 frames and 48 kHz stereo audio to Media Foundation exactly as the export's encoder
//! thread does (`thread.rs`: audio in 4096-frame chunks up to each frame's start, then the frame)
//! and times every call. `mode=thread` instead drives [`mc_video::EncoderThread`] and times each
//! `submit`, which is what the render thread waits on.
//!
//! `cargo run -p mc-video --example stall_probe -- <out.mp4> [key=value ...]`
//!
//! Keys: `size` (1920x1080), `fps` (60), `seconds` (10), `encoder` (`auto` or `software`), `audio`
//! (1), `mode` (`direct` or `thread`), `raw` (NV12 frames from the export harness's `raw=` dump,
//! looped; synthetic frames without it), `csv` (the timeline), `spike` (ms, 10).

#[cfg(windows)]
fn main() {
    use std::io::{Read, Write};
    use std::time::Instant;

    use mc_video::{
        AacBitrate, AudioConfig, AudioSource, EncoderConfig, EncoderPreference, EncoderThread,
        FrameRate, Mp4Writer, VideoCodec,
    };

    /// A quiet tone with a little noise, as the export's mix would give.
    struct Tone {
        n: u64,
        seed: u32,
    }
    impl AudioSource for Tone {
        fn fill(&mut self, out: &mut [i16]) {
            for pair in out.chunks_exact_mut(2) {
                self.seed = self
                    .seed
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223);
                let noise = f64::from((self.seed >> 16) as u16) / 65536.0 - 0.5;
                let t = self.n as f64 / 48_000.0;
                let v = (t * 440.0 * std::f64::consts::TAU).sin() * 0.2 + noise * 0.05;
                let s = (v * 32767.0) as i16;
                pair[0] = s;
                pair[1] = s;
                self.n += 1;
            }
        }
    }

    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(out) = args.first() else {
        eprintln!("usage: stall_probe <out.mp4> [key=value ...]");
        std::process::exit(2);
    };
    let mut size = (1920_u32, 1080_u32);
    let mut fps = 60_u32;
    let mut seconds = 10.0_f64;
    let mut preference = EncoderPreference::Auto;
    let mut audio = true;
    let mut threaded = false;
    let mut raw = None;
    let mut csv = None;
    let mut spike_ms = 10.0_f64;
    // `feed=<frames/s>` (mode=thread): hand frames over at this rate at most, like a render loop
    // of that speed (0: as fast as possible).
    let mut feed = 0.0_f64;
    for pair in &args[1..] {
        let (key, value) = pair.split_once('=').unwrap_or((pair.as_str(), ""));
        match key {
            "size" => {
                let (w, h) = value.split_once('x').expect("size=WxH");
                size = (w.parse().expect("width"), h.parse().expect("height"));
            }
            "fps" => fps = value.parse().expect("fps"),
            "seconds" => seconds = value.parse().expect("seconds"),
            "encoder" => {
                preference = if value == "software" {
                    EncoderPreference::SoftwareOnly
                } else {
                    EncoderPreference::Auto
                };
            }
            "audio" => audio = value != "0",
            "mode" => threaded = value == "thread",
            "raw" => raw = Some(value.to_owned()),
            "csv" => csv = Some(value.to_owned()),
            "spike" => spike_ms = value.parse().expect("spike"),
            "feed" => feed = value.parse().expect("feed"),
            other => panic!("unknown key {other}"),
        }
    }
    let (w, h) = size;
    let frame_bytes = mc_video::nv12::frame_bytes(w, h);
    let frames: Vec<Vec<u8>> = match &raw {
        Some(path) => {
            let mut data = Vec::new();
            std::fs::File::open(path)
                .expect("raw frames")
                .read_to_end(&mut data)
                .expect("raw frames");
            data.chunks_exact(frame_bytes).map(<[u8]>::to_vec).collect()
        }
        None => (0..16_u32)
            .map(|k| {
                // A smooth gradient with moving, textured blocks: some detail and some motion.
                let mut f = vec![128_u8; frame_bytes];
                let luma = (w * h) as usize;
                for y in 0..h {
                    for x in 0..w {
                        let block = ((x + k * 24) / 64 + y / 64) % 3;
                        let tex = ((x * 7 + y * 13 + k * 5) % 32) as u8;
                        let base = ((x + y) * 160 / (w + h)) as u8 + 40;
                        f[(y * w + x) as usize] =
                            base.saturating_add(if block == 0 { tex } else { 0 });
                    }
                }
                for (i, v) in f[luma..].iter_mut().enumerate() {
                    *v = 128_u8.wrapping_add(((i as u32 / 97 + k) % 24) as u8);
                }
                f
            })
            .collect(),
    };
    assert!(!frames.is_empty(), "no frames");
    let total = (seconds * f64::from(fps)).round() as u64;
    let mut config = EncoderConfig::new(out, w, h, FrameRate::whole(fps), VideoCodec::H264);
    config.encoder = preference;
    config.audio = audio.then_some(AudioConfig {
        sample_rate: 48_000,
        channels: 2,
        bitrate: AacBitrate::Kbps128,
    });
    // (frame, start since the first frame µs, audio chunks, audio µs, video µs)
    let mut rows: Vec<(u64, u64, u32, u64, u64)> = Vec::with_capacity(total as usize);
    let began;
    if threaded {
        let source: Option<Box<dyn AudioSource>> =
            audio.then(|| Box::new(Tone { n: 0, seed: 1 }) as Box<dyn AudioSource>);
        let mut encoder = EncoderThread::spawn(config, source, 6).expect("encoder");
        println!(
            "encoder: {} (hardware {})",
            encoder.info().name,
            encoder.info().hardware
        );
        began = Instant::now();
        // A render loop whose frames take 1/feed s each: the next frame is ready that long
        // after the last one was handed over.
        let mut ready = Instant::now();
        for index in 0..total {
            if feed > 0.0 {
                while Instant::now() < ready {
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
            }
            let mut buffer = encoder.buffer();
            buffer.copy_from_slice(&frames[(index as usize) % frames.len()]);
            let at = began.elapsed();
            encoder.submit(buffer).expect("submit");
            let took = began.elapsed() - at;
            if feed > 0.0 {
                ready = Instant::now() + std::time::Duration::from_secs_f64(1.0 / feed);
            }
            rows.push((index, at.as_micros() as u64, 0, 0, took.as_micros() as u64));
        }
        let finishing = Instant::now();
        let summary = encoder.finish().expect("finish");
        println!(
            "finish {:.1} ms; {} frames, {} bytes",
            finishing.elapsed().as_secs_f64() * 1e3,
            summary.frames,
            summary.bytes
        );
    } else {
        let mut writer = Mp4Writer::create(config).expect("encoder");
        println!(
            "encoder: {} (hardware {})",
            writer.info().name,
            writer.info().hardware
        );
        let mut tone = Tone { n: 0, seed: 1 };
        let mut pcm = Vec::new();
        const CHUNK: u64 = 4096;
        began = Instant::now();
        for index in 0..total {
            let at = began.elapsed();
            let mut chunks = 0;
            if audio {
                // thread.rs: bring the audio up to this frame's start, in whole chunks.
                let start = FrameRate::whole(fps).frame_start_hns(writer.frames_written());
                let target = (start.max(0) as u64) * 48_000 / 10_000_000;
                while writer.audio_frames_written() + CHUNK <= target {
                    pcm.resize(CHUNK as usize * 2, 0);
                    tone.fill(&mut pcm);
                    writer.write_audio(&pcm).expect("audio");
                    chunks += 1;
                }
            }
            let mid = began.elapsed();
            writer
                .write_frame(&frames[(index as usize) % frames.len()])
                .expect("frame");
            let end = began.elapsed();
            rows.push((
                index,
                at.as_micros() as u64,
                chunks,
                (mid - at).as_micros() as u64,
                (end - mid).as_micros() as u64,
            ));
        }
        let finishing = Instant::now();
        let summary = writer.finish().expect("finish");
        println!(
            "finish {:.1} ms; {} frames, {} bytes",
            finishing.elapsed().as_secs_f64() * 1e3,
            summary.frames,
            summary.bytes
        );
    }
    let wall = began.elapsed().as_secs_f64();
    println!(
        "{total} frames of {w}x{h} at {fps} fps in {wall:.2} s ({:.0} frames/s)",
        total as f64 / wall
    );
    let mut stalls = 0;
    let mut previous: Option<(u64, u64)> = None;
    for &(index, at, chunks, audio_us, video_us) in &rows {
        if (audio_us + video_us) as f64 / 1e3 >= spike_ms {
            stalls += 1;
            let since = previous.map_or(String::new(), |(i, t)| {
                format!(
                    ", {} frames / {:.0} ms after the last",
                    index - i,
                    (at - t) as f64 / 1e3
                )
            });
            println!(
                "  frame {index:5} (index mod fps {:3}) at {:8.1} ms: audio {:6.1} ms ({chunks} \
                 chunks), video {:6.1} ms{since}",
                index % u64::from(fps),
                at as f64 / 1e3,
                audio_us as f64 / 1e3,
                video_us as f64 / 1e3,
            );
            previous = Some((index, at));
        }
    }
    let blocked: u64 = rows.iter().map(|r| r.3 + r.4).sum();
    println!(
        "{stalls} calls of {spike_ms} ms or more; {:.2} s inside Media Foundation in all",
        blocked as f64 / 1e6
    );
    if let Some(path) = csv {
        let mut file = std::io::BufWriter::new(std::fs::File::create(path).expect("csv"));
        writeln!(file, "frame,start_us,audio_chunks,audio_us,video_us").expect("csv");
        for (index, at, chunks, audio_us, video_us) in rows {
            writeln!(file, "{index},{at},{chunks},{audio_us},{video_us}").expect("csv");
        }
    }
}

#[cfg(not(windows))]
fn main() {}
