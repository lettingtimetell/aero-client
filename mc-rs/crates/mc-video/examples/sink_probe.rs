//! The export's sink writer set up as `mf/writer.rs` does, with the
//! sink writer's own statistics logged after every frame, and switches to try other setups.
//!
//! `cargo run -p mc-video --example sink_probe -- <out.mp4> [key=value ...]`
//!
//! Keys: `size` (1920x1080), `fps` (60), `seconds` (6), `encoder` (`auto`/`software`), `audio`
//! (1), `throttle` (1: the sink writer's own throttling, as the export; 0 turns it off), `backlog`
//! (frames; 0 = none: wait before each frame while more than this many video frames are inside
//! the sink writer, not yet handed to the file), `counter` (`encoded`: count the frames not yet
//! out of the encoder instead), `csv` (the per-frame timeline).

#[cfg(windows)]
#[allow(clippy::too_many_lines)]
fn main() {
    use std::io::Write;
    use std::time::{Duration, Instant};

    use windows::core::{GUID, HSTRING};
    use windows::Win32::Media::MediaFoundation::*;
    use windows::Win32::System::Com::{CoInitializeEx, COINIT_MULTITHREADED};

    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(out) = args.first() else {
        eprintln!("usage: sink_probe <out.mp4> [key=value ...]");
        std::process::exit(2);
    };
    let (mut w, mut h, mut fps, mut seconds) = (1920_u32, 1080_u32, 60_u32, 6.0_f64);
    let (mut hardware, mut audio, mut throttle, mut backlog) = (true, true, true, 0_u64);
    let mut by_encoded = false;
    let mut csv = None;
    for pair in &args[1..] {
        let (key, value) = pair.split_once('=').unwrap_or((pair.as_str(), ""));
        match key {
            "size" => {
                let (a, b) = value.split_once('x').expect("size=WxH");
                (w, h) = (a.parse().expect("w"), b.parse().expect("h"));
            }
            "fps" => fps = value.parse().expect("fps"),
            "seconds" => seconds = value.parse().expect("seconds"),
            "encoder" => hardware = value != "software",
            "audio" => audio = value != "0",
            "throttle" => throttle = value != "0",
            "backlog" => backlog = value.parse().expect("backlog"),
            "counter" => by_encoded = value == "encoded",
            "csv" => csv = Some(value.to_owned()),
            other => panic!("unknown key {other}"),
        }
    }
    let pack = |a: u32, b: u32| (u64::from(a) << 32) | u64::from(b);
    let frame_bytes = mc_video::nv12::frame_bytes(w, h);
    // Synthetic frames as in stall_probe: a gradient with moving textured blocks.
    let frames: Vec<Vec<u8>> = (0..16_u32)
        .map(|k| {
            let mut f = vec![128_u8; frame_bytes];
            let luma = (w * h) as usize;
            for y in 0..h {
                for x in 0..w {
                    let block = ((x + k * 24) / 64 + y / 64) % 3;
                    let tex = ((x * 7 + y * 13 + k * 5) % 32) as u8;
                    let base = ((x + y) * 160 / (w + h)) as u8 + 40;
                    f[(y * w + x) as usize] = base.saturating_add(if block == 0 { tex } else { 0 });
                }
            }
            for (i, v) in f[luma..].iter_mut().enumerate() {
                *v = 128_u8.wrapping_add(((i as u32 / 97 + k) % 24) as u8);
            }
            f
        })
        .collect();
    // Attribute setters that panic on failure (a probe, not the product).
    let u32_of = |t: &IMFAttributes, key: &GUID, value: u32| {
        // SAFETY: a live attribute store and a valid key.
        unsafe { t.SetUINT32(key, value) }.expect("attribute");
    };
    let u64_of = |t: &IMFAttributes, key: &GUID, value: u64| {
        // SAFETY: as above.
        unsafe { t.SetUINT64(key, value) }.expect("attribute");
    };
    let guid_of = |t: &IMFAttributes, key: &GUID, value: &GUID| {
        // SAFETY: as above.
        unsafe { t.SetGUID(key, value) }.expect("attribute");
    };

    // SAFETY (whole block): plain Media Foundation calls on live objects created here, on one
    // thread, in the order the sink writer documents.
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        MFStartup(MF_VERSION, MFSTARTUP_LITE).expect("MFStartup");
        let mut attrs = None;
        MFCreateAttributes(&mut attrs, 4).expect("attributes");
        let attrs = attrs.expect("attributes");
        let hardware = u32::from(hardware);
        let mp4 = MFTranscodeContainerType_MPEG4;
        u32_of(&attrs, &MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, hardware);
        guid_of(&attrs, &MF_TRANSCODE_CONTAINERTYPE, &mp4);
        if !throttle {
            u32_of(&attrs, &MF_SINK_WRITER_DISABLE_THROTTLING, 1);
        }
        let url = HSTRING::from(out.as_str());
        let writer = MFCreateSinkWriterFromURL(&url, None, &attrs).expect("writer");

        let colour = |t: &IMFAttributes| {
            let limited = MFNominalRange_16_235.0 as u32;
            u32_of(t, &MF_MT_VIDEO_NOMINAL_RANGE, limited);
            u32_of(t, &MF_MT_YUV_MATRIX, MFVideoTransferMatrix_BT709.0 as u32);
            u32_of(t, &MF_MT_VIDEO_PRIMARIES, MFVideoPrimaries_BT709.0 as u32);
            u32_of(t, &MF_MT_TRANSFER_FUNCTION, MFVideoTransFunc_709.0 as u32);
        };
        let progressive = MFVideoInterlace_Progressive.0 as u32;
        let pixels = f64::from(w) * f64::from(h);
        let bitrate = (pixels * 0.1 * f64::from(fps).powf(0.75) * 60f64.powf(0.25))
            .clamp(500_000.0, 400_000_000.0) as u32;
        let vout = MFCreateMediaType().expect("media type");
        guid_of(&vout, &MF_MT_MAJOR_TYPE, &MFMediaType_Video);
        guid_of(&vout, &MF_MT_SUBTYPE, &MFVideoFormat_H264);
        u32_of(&vout, &MF_MT_AVG_BITRATE, bitrate);
        u32_of(&vout, &MF_MT_INTERLACE_MODE, progressive);
        u64_of(&vout, &MF_MT_FRAME_SIZE, pack(w, h));
        u64_of(&vout, &MF_MT_FRAME_RATE, pack(fps, 1));
        u64_of(&vout, &MF_MT_PIXEL_ASPECT_RATIO, pack(1, 1));
        let high = eAVEncH264VProfile_High.0 as u32;
        u32_of(&vout, &MF_MT_MPEG2_PROFILE, high);
        colour(&vout);
        let video = writer.AddStream(&vout).expect("video stream");
        let vin = MFCreateMediaType().expect("media type");
        guid_of(&vin, &MF_MT_MAJOR_TYPE, &MFMediaType_Video);
        guid_of(&vin, &MF_MT_SUBTYPE, &MFVideoFormat_NV12);
        u32_of(&vin, &MF_MT_INTERLACE_MODE, progressive);
        u64_of(&vin, &MF_MT_FRAME_SIZE, pack(w, h));
        u64_of(&vin, &MF_MT_FRAME_RATE, pack(fps, 1));
        u64_of(&vin, &MF_MT_PIXEL_ASPECT_RATIO, pack(1, 1));
        u32_of(&vin, &MF_MT_DEFAULT_STRIDE, w);
        u32_of(&vin, &MF_MT_ALL_SAMPLES_INDEPENDENT, 1);
        colour(&vin);
        // The export's quality preset "High" with everything `mf/writer.rs` hands over.
        let mut params = None;
        MFCreateAttributes(&mut params, 8).expect("attributes");
        let params = params.expect("attributes");
        let quality_mode = eAVEncCommonRateControlMode_Quality.0 as u32;
        u32_of(&params, &CODECAPI_AVEncCommonRateControlMode, quality_mode);
        u32_of(&params, &CODECAPI_AVEncCommonQuality, 80);
        u32_of(&params, &CODECAPI_AVEncMPVGOPSize, fps.min(600));
        u32_of(&params, &CODECAPI_AVEncCommonQualityVsSpeed, 70);
        writer
            .SetInputMediaType(video, &vin, &params)
            .expect("video input");
        let audio_stream = if audio {
            let aout = MFCreateMediaType().expect("media type");
            guid_of(&aout, &MF_MT_MAJOR_TYPE, &MFMediaType_Audio);
            guid_of(&aout, &MF_MT_SUBTYPE, &MFAudioFormat_AAC);
            u32_of(&aout, &MF_MT_AUDIO_BITS_PER_SAMPLE, 16);
            u32_of(&aout, &MF_MT_AUDIO_SAMPLES_PER_SECOND, 48_000);
            u32_of(&aout, &MF_MT_AUDIO_NUM_CHANNELS, 2);
            u32_of(&aout, &MF_MT_AUDIO_AVG_BYTES_PER_SECOND, 16_000);
            u32_of(&aout, &MF_MT_AAC_PAYLOAD_TYPE, 0);
            u32_of(&aout, &MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION, 0x29);
            let stream = writer.AddStream(&aout).expect("audio stream");
            let ain = MFCreateMediaType().expect("media type");
            guid_of(&ain, &MF_MT_MAJOR_TYPE, &MFMediaType_Audio);
            guid_of(&ain, &MF_MT_SUBTYPE, &MFAudioFormat_PCM);
            u32_of(&ain, &MF_MT_AUDIO_BITS_PER_SAMPLE, 16);
            u32_of(&ain, &MF_MT_AUDIO_SAMPLES_PER_SECOND, 48_000);
            u32_of(&ain, &MF_MT_AUDIO_NUM_CHANNELS, 2);
            u32_of(&ain, &MF_MT_AUDIO_BLOCK_ALIGNMENT, 4);
            u32_of(&ain, &MF_MT_AUDIO_AVG_BYTES_PER_SECOND, 4 * 48_000);
            writer
                .SetInputMediaType(stream, &ain, None)
                .expect("audio input");
            Some(stream)
        } else {
            None
        };
        writer.BeginWriting().expect("begin");

        let write = |stream: u32, bytes: &[u8], time: i64, duration: i64| {
            let buffer = MFCreateMemoryBuffer(bytes.len() as u32).expect("buffer");
            let mut data = std::ptr::null_mut();
            buffer.Lock(&mut data, None, None).expect("lock");
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len());
            buffer.Unlock().expect("unlock");
            buffer.SetCurrentLength(bytes.len() as u32).expect("length");
            let sample = MFCreateSample().expect("sample");
            sample.AddBuffer(&buffer).expect("add buffer");
            sample.SetSampleTime(time).expect("time");
            sample.SetSampleDuration(duration).expect("duration");
            writer.WriteSample(stream, &sample).expect("write");
        };
        let stats = |stream: u32| {
            let mut s = MF_SINK_WRITER_STATISTICS {
                cb: std::mem::size_of::<MF_SINK_WRITER_STATISTICS>() as u32,
                ..Default::default()
            };
            let _ = writer.GetStatistics(stream, &mut s);
            s
        };
        let total = (seconds * f64::from(fps)).round() as u64;
        let frame_start = |i: u64| (i as u128 * 10_000_000 / u128::from(fps)) as i64;
        let mut audio_written = 0_u64;
        let pcm: Vec<u8> = (0..4096 * 4).map(|i| ((i * 37) % 251) as u8 / 8).collect();
        let mut rows = Vec::with_capacity(total as usize);
        let began = Instant::now();
        for index in 0..total {
            let at = began.elapsed();
            // Own back-pressure (backlog=K): wait while more than K video frames are inside the
            // sink writer (received but not yet handed to the file, or not yet encoded).
            let mut waited = Duration::ZERO;
            if backlog > 0 {
                let from = Instant::now();
                loop {
                    let s = stats(video);
                    let done = if by_encoded {
                        s.qwNumSamplesEncoded
                    } else {
                        s.qwNumSamplesProcessed
                    };
                    if s.qwNumSamplesReceived.saturating_sub(done) < backlog {
                        break;
                    }
                    std::thread::sleep(Duration::from_micros(500));
                }
                waited = from.elapsed();
            }
            let a0 = Instant::now();
            if let Some(stream) = audio_stream {
                // thread.rs: bring the audio up to this frame's start, in whole chunks.
                let target = (frame_start(index).max(0) as u64) * 48_000 / 10_000_000;
                while audio_written + 4096 <= target {
                    let t0 = (audio_written as u128 * 10_000_000 / 48_000) as i64;
                    let t1 = ((audio_written + 4096) as u128 * 10_000_000 / 48_000) as i64;
                    write(stream, &pcm, t0, t1 - t0);
                    audio_written += 4096;
                }
            }
            let v0 = Instant::now();
            let frame = &frames[(index as usize) % frames.len()];
            let (start, end) = (frame_start(index), frame_start(index + 1));
            write(video, frame, start, end - start);
            let v1 = Instant::now();
            let s = stats(video);
            let a = audio_stream.map(stats).unwrap_or_default();
            rows.push((
                index,
                at.as_micros(),
                waited.as_micros(),
                (v0 - a0).as_micros(),
                (v1 - v0).as_micros(),
                s.qwNumSamplesReceived,
                s.qwNumSamplesEncoded,
                s.qwNumSamplesProcessed,
                s.dwByteCountQueued,
                s.dwNumOutstandingSinkSampleRequests,
                a.qwNumSamplesReceived,
                a.qwNumSamplesProcessed,
            ));
        }
        let wall = began.elapsed().as_secs_f64();
        let finishing = Instant::now();
        writer.Finalize().expect("finalize");
        let finish = finishing.elapsed().as_secs_f64();
        println!(
            "{total} frames {w}x{h} at {fps} fps: written in {wall:.2} s, finish {finish:.2} s, \
             total {:.2} s ({:.0} frames/s overall)",
            wall + finish,
            total as f64 / (wall + finish)
        );
        let blocked: Vec<f64> = rows
            .iter()
            .map(|r| (r.2 + r.3 + r.4) as f64 / 1e3)
            .collect();
        let mut sorted = blocked.clone();
        sorted.sort_by(f64::total_cmp);
        let p = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];
        println!(
            "per-frame wait (own + audio + video): median {:.2} ms, p90 {:.2}, p99 {:.2}, max \
             {:.2}; frames waiting 10 ms or more: {}",
            p(0.5),
            p(0.9),
            p(0.99),
            sorted[sorted.len() - 1],
            blocked.iter().filter(|&&b| b >= 10.0).count()
        );
        if let Some(path) = csv {
            let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("csv"));
            writeln!(
                f,
                "frame,start_us,own_wait_us,audio_us,video_us,v_received,v_encoded,v_processed,\
                 v_bytes_queued,v_requests,a_received,a_processed"
            )
            .expect("csv");
            for r in rows {
                writeln!(
                    f,
                    "{},{},{},{},{},{},{},{},{},{},{},{}",
                    r.0, r.1, r.2, r.3, r.4, r.5, r.6, r.7, r.8, r.9, r.10, r.11
                )
                .expect("csv");
            }
        }
        drop(writer);
        let _ = MFShutdown();
    }
}

#[cfg(not(windows))]
fn main() {}
