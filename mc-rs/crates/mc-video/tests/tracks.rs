//! Separate audio tracks: a real encode with the mixed track
//! plus two more, checked with ffprobe/ffmpeg and Windows' own decoder.
//!
//! Needs Windows Media Foundation and `ffprobe`/`ffmpeg` on PATH; passes without checking when
//! either is missing. Files go to `MC_VIDEO_SAMPLES_DIR` when set, else to a temp folder.

#![cfg(windows)]

use std::path::{Path, PathBuf};
use std::process::Command;

use mc_video::{
    probe, AacBitrate, AudioConfig, AudioSource, EncoderConfig, EncoderThread, FrameRate,
    VideoCodec,
};

fn out_dir() -> PathBuf {
    let dir = std::env::var_os("MC_VIDEO_SAMPLES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("mc-video-tests"));
    std::fs::create_dir_all(&dir).expect("output dir");
    dir
}

fn tools_ready() -> bool {
    let tool = |name: &str| {
        Command::new(name)
            .arg("-version")
            .output()
            .is_ok_and(|o| o.status.success())
    };
    if !tool("ffprobe") || !tool("ffmpeg") {
        eprintln!("ffprobe/ffmpeg not on PATH: skipping");
        return false;
    }
    if probe::codec_support(VideoCodec::H264).is_none() {
        eprintln!("no H.264 encoder on this PC: skipping");
        return false;
    }
    true
}

/// A different tone on every track: the mix 440 Hz, extra track `n` 660 Hz and 990 Hz...
struct Tones {
    rate: u32,
    /// The first sample frame of the last `fill`, and its length.
    start: u64,
    len: u64,
}

fn tone(out: &mut [i16], start: u64, rate: u32, freq: f64) {
    for (i, pair) in out.chunks_exact_mut(2).enumerate() {
        let t = (start + i as u64) as f64 / f64::from(rate);
        let v = ((t * freq * std::f64::consts::TAU).sin() * 9000.0) as i16;
        pair[0] = v;
        pair[1] = v;
    }
}

const FREQS: [f64; 3] = [440.0, 660.0, 990.0];

impl AudioSource for Tones {
    fn fill(&mut self, out: &mut [i16]) {
        self.start += self.len;
        self.len = (out.len() / 2) as u64;
        tone(out, self.start, self.rate, FREQS[0]);
    }

    fn fill_track(&mut self, track: usize, out: &mut [i16]) {
        assert_eq!(
            (out.len() / 2) as u64,
            self.len,
            "the same stretch as the mix"
        );
        tone(out, self.start, self.rate, FREQS[track + 1]);
    }
}

/// The power of `freq` in mono 16-bit samples (Goertzel), relative to the total power.
fn share(samples: &[i16], rate: f64, freq: f64) -> f64 {
    let w = std::f64::consts::TAU * freq / rate;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0_f64, 0.0_f64);
    let mut total = 0.0;
    for &s in samples {
        let x = f64::from(s);
        total += x * x;
        let s0 = x + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    let power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
    power / (total * samples.len() as f64 / 2.0).max(1.0)
}

fn decode_track(path: &Path, track: usize) -> Vec<i16> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args([
            "-map",
            &format!("0:a:{track}"),
            "-ac",
            "1",
            "-ar",
            "48000",
            "-f",
            "s16le",
            "-",
        ])
        .output()
        .expect("run ffmpeg");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

#[test]
fn separate_tracks_follow_a_normal_mixed_track_that_players_pick() {
    if !tools_ready() {
        return;
    }
    let path = out_dir().join("re_tracks_h264.mp4");
    let _ = std::fs::remove_file(&path);
    let (w, h) = (320, 180);
    let mut config = EncoderConfig::new(&path, w, h, FrameRate::whole(30), VideoCodec::H264);
    config.audio = Some(AudioConfig {
        sample_rate: 48_000,
        channels: 2,
        bitrate: AacBitrate::Kbps128,
    });
    config.audio_tracks = vec!["Game sound".to_owned(), "Microphone".to_owned()];
    let source = Tones {
        rate: 48_000,
        start: 0,
        len: 0,
    };
    let mut encoder = EncoderThread::spawn(config, Some(Box::new(source)), 8).expect("encoder");
    for index in 0..60_u64 {
        let mut buffer = encoder.buffer();
        let (luma, chroma) = buffer.split_at_mut((w * h) as usize);
        luma.fill(16 + (index * 3) as u8);
        chroma.fill(128);
        encoder.submit(buffer).expect("submit");
    }
    let summary = encoder.finish().expect("finish");
    assert_eq!(summary.audio_frames, 96_000, "2 s of the mixed track");

    // ffprobe: one video and three audio streams, only the first audio one the default, named.
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,duration:stream_tags=handler_name:\
             stream_disposition=default",
            "-of",
            "json",
        ])
        .arg(&path)
        .output()
        .expect("ffprobe");
    let probe: serde_json::Value = serde_json::from_slice(&out.stdout).expect("json");
    eprintln!("{probe}");
    let streams = probe["streams"].as_array().expect("streams");
    let audio: Vec<&serde_json::Value> = streams
        .iter()
        .filter(|s| s["codec_type"] == "audio")
        .collect();
    assert_eq!(
        audio.len(),
        4,
        "the mix, two tracks, and the mix again for Windows"
    );
    let defaults: Vec<i64> = audio
        .iter()
        .map(|s| s["disposition"]["default"].as_i64().expect("default"))
        .collect();
    assert_eq!(
        defaults,
        vec![1, 0, 0, 0],
        "only the first mix is the default track"
    );
    let names: Vec<&str> = audio
        .iter()
        .map(|s| s["tags"]["handler_name"].as_str().expect("name"))
        .collect();
    assert_eq!(names, vec!["Mix", "Game sound", "Microphone", "Mix"]);
    let video = streams
        .iter()
        .find(|s| s["codec_type"] == "video")
        .expect("video");
    assert_eq!(video["disposition"]["default"].as_i64(), Some(1));
    for stream in &audio {
        let duration: f64 = stream["duration"]
            .as_str()
            .and_then(|d| d.parse().ok())
            .expect("duration");
        assert!(
            (duration - 2.0).abs() < 0.05,
            "every track lasts 2 s: {duration}"
        );
    }

    // Each track holds its own tone, and the last is the mix again; a player's default pick
    // (ffmpeg's `-map 0:a`, the first default track) is the mix.
    for (track, freq) in FREQS.iter().chain(&FREQS[..1]).enumerate() {
        let samples = decode_track(&path, track);
        assert!(
            samples.len() > 90_000,
            "track {track}: {} samples",
            samples.len()
        );
        let body = &samples[4_800..samples.len() - 4_800];
        let own = share(body, 48_000.0, *freq);
        for (other, other_freq) in FREQS.iter().enumerate() {
            if *other_freq != *freq {
                let leak = share(body, 48_000.0, *other_freq);
                assert!(
                    own > 50.0 * leak,
                    "track {track}: {own} vs {other}'s {leak}"
                );
            }
        }
    }
    let default_pick = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&path)
        .args(["-ac", "1", "-ar", "48000", "-f", "s16le", "-"])
        .output()
        .expect("ffmpeg");
    let picked: Vec<i16> = default_pick
        .stdout
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();
    let body = &picked[4_800..picked.len() - 4_800];
    assert!(
        share(body, 48_000.0, FREQS[0]) > 50.0 * share(body, 48_000.0, FREQS[1]),
        "a player without track choice plays the mix"
    );

    // Windows' own decoders (what the Media Player app uses) still open and decode the file.
    let windows = mc_video::verify::decode_video(&path).expect("Windows decodes the file");
    assert_eq!(windows.frames, 60);
    // Media Foundation's source reader (a low-level decoding API) selects the *last* audio
    // stream by default, whatever the headers say (measured: enabled flags, alternate groups,
    // track ids and volumes all ignored); recorded here so a change shows.
    let reader = windows_default_selection(&path);
    eprintln!("Media Foundation source reader default selection: {reader:?}");
    assert_eq!(
        reader,
        vec![true, false, false, false, true],
        "the source reader takes the last audio stream: the mix's copy"
    );
    // What a Windows player plays: the media playback stack the Media Player app is built on
    // also picks the last audio track, which is the mix's copy.
    let (tracks, selected) = media_player_selection(&path);
    eprintln!("Windows media playback: {tracks} audio tracks, track {selected} selected");
    assert_eq!(
        (tracks, selected),
        (4, 3),
        "Windows' player plays the mix (its copy)"
    );
}

/// The audio tracks Windows' media playback stack (`MediaPlaybackItem`, as in the Media Player
/// app) sees in `path`, and the one it selects.
fn media_player_selection(path: &Path) -> (u32, i32) {
    use windows::core::HSTRING;
    use windows::Foundation::Uri;
    use windows::Media::Core::MediaSource;
    use windows::Media::Playback::MediaPlaybackItem;
    use windows::Win32::System::Com::{CoInitializeEx, COINIT_MULTITHREADED};
    // SAFETY: initialises COM for this thread (WinRT needs an apartment); never uninitialised,
    // as the test thread ends right after.
    let _ = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
    let absolute = std::fs::canonicalize(path).expect("absolute path");
    let text = absolute.to_string_lossy().replace('\\', "/");
    let url = format!("file:///{}", text.trim_start_matches("//?/"));
    let uri = Uri::CreateUri(&HSTRING::from(url)).expect("a file URI");
    let source = MediaSource::CreateFromUri(&uri).expect("a media source");
    let item = MediaPlaybackItem::Create(&source).expect("a playback item");
    let open = source.OpenAsync().expect("open");
    // Started 0, Completed 1, Canceled 2, Error 3.
    let started = std::time::Instant::now();
    while open.Status().expect("status").0 == 0 && started.elapsed().as_secs() < 10 {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(open.Status().expect("status").0, 1, "the file opens");
    let tracks = item.AudioTracks().expect("audio tracks");
    (
        tracks.Size().unwrap_or(0),
        tracks.SelectedIndex().unwrap_or(-2),
    )
}

/// Which streams Windows' Media Foundation selects by default, in file order.
fn windows_default_selection(path: &Path) -> Vec<bool> {
    use windows::core::HSTRING;
    use windows::Win32::Media::MediaFoundation::{
        IMFAttributes, MFCreateSourceReaderFromURL, MFShutdown, MFStartup, MFSTARTUP_FULL,
        MF_VERSION,
    };
    use windows::Win32::System::Com::{CoInitializeEx, COINIT_MULTITHREADED};
    let mut selected = Vec::new();
    // SAFETY: COM and Media Foundation are started for this thread and shut down after the
    // reader (scoped inside) is released; the URL is a valid wide string.
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        MFStartup(MF_VERSION, MFSTARTUP_FULL).expect("Media Foundation");
        {
            let url = HSTRING::from(path.as_os_str());
            let reader =
                MFCreateSourceReaderFromURL(&url, None::<&IMFAttributes>).expect("a source reader");
            let mut index = 0;
            while let Ok(on) = reader.GetStreamSelection(index) {
                selected.push(on.as_bool());
                index += 1;
            }
        }
        let _ = MFShutdown();
    }
    selected
}
