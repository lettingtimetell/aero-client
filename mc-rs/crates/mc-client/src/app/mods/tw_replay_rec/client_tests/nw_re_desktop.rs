//! Desktop audio through the real recorder (a fake capture fed a
//! tone; nothing is opened or played). Uses the recorder's client test helpers.

use std::path::Path;
use std::time::{Duration, Instant};

use mc_replay::{Library, Reader, StreamId};

use super::super::ids;
use super::{chatty_capture, client, drive, open};

/// 100 ms of a 500 Hz tone at 24 kHz, continuing from sample `from`.
fn tone(from: usize) -> Vec<i16> {
    let rate = mc_replay::desktop::SAMPLE_RATE as usize;
    (from..from + rate / 10)
        .map(|i| {
            let t = i as f32 / rate as f32;
            (6000.0 * (2.0 * std::f32::consts::PI * 500.0 * t).sin()) as i16
        })
        .collect()
}

/// "Record desktop audio" on with "Record every game": the recording holds a desktop stream of
/// its own (not the microphone's), frames exactly 20 ms apart that cover the recording, the tone
/// intact, and the metadata says so.
#[test]
fn a_recording_carries_desktop_audio_as_its_own_stream() {
    let Some((mut client, dir)) = client() else {
        return;
    };
    client.mods.settings.set_enabled("replay", true);
    client
        .mods
        .settings
        .set_bool("replay", ids::DESKTOP_AUDIO, true);
    let capture = mc_replay::desktop::DesktopCapture::fake();
    let feeder = capture.feeder();
    client.rec.test_desktop = Some(capture);
    open(
        &mut client,
        &chatty_capture(8),
        Path::new("desktop.mccap"),
        100_000,
    );
    let mut fed = 0usize;
    drive(
        &mut client,
        100,
        Duration::from_millis(100),
        |client, frame| {
            let chunk = tone(fed);
            feeder.feed(&chunk);
            fed += chunk.len();
            if frame == 20 {
                let conn = client.rec.conn.as_deref().expect("following the session");
                assert!(
                    conn.desktop.is_some(),
                    "desktop audio is open while recording"
                );
                assert!(
                    conn.mic.is_none(),
                    "the microphone is a separate option, still off"
                );
            }
        },
    );
    assert!(client.rec.test_desktop.is_none(), "the capture was opened");
    client.session = None;
    client.rec_frame(Instant::now() + Duration::from_secs(120));
    assert!(client.rec.conn.is_none(), "closed with the session");
    client.rec_shutdown();

    let files = Library::new(dir.path()).list_dir(false);
    assert_eq!(files.len(), 1, "one recording");
    let reader = Reader::open(&files[0].path).expect("opens");
    let meta = reader.meta().clone();
    assert!(meta.has_desktop, "{meta:?}");
    assert_eq!(meta.desktop_format, Some(mc_replay::desktop::format()));
    assert!(!meta.has_mic, "no microphone stream");
    let records: Vec<_> = reader.records().map(|r| r.expect("record")).collect();
    assert!(
        records.iter().all(|r| r.stream != StreamId::Mic),
        "nothing in the microphone's stream"
    );
    let mut pcm = Vec::new();
    let starts: Vec<u64> = records
        .iter()
        .filter(|r| r.stream == StreamId::Desktop)
        .map(|r| {
            let lead = mc_replay::desktop::decode_record(&r.payload, &mut pcm).expect("decodes");
            r.time - u64::from(lead)
        })
        .collect();
    // About 9.9 s of tone delivered in 10 s of frames: ~490 frames of 20 ms.
    assert!(starts.len() >= 450, "{} desktop frames", starts.len());
    assert!(
        starts.windows(2).all(|w| w[1] - w[0] == 20_000),
        "consecutive frames exactly 20 ms apart: {:?}",
        starts
            .windows(2)
            .map(|w| w[1] as i64 - w[0] as i64)
            .filter(|d| *d != 20_000)
            .take(5)
            .collect::<Vec<_>>()
    );
    let peak = pcm.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
    assert!(peak > 5200 && peak < 6800, "the tone survives: {peak}");
    let duration = meta.duration;
    let last = *starts.last().expect("frames");
    assert!(
        last + 300_000 >= duration,
        "the stream runs to the recording's end: {last} vs {duration}"
    );
}

/// "Record desktop audio" on and the clip key bound, no recording running: desktop audio opens
/// for the clip buffer's ring, and a clip carries frames covering its whole window, merged into
/// the world in time order, with the tone intact and the metadata set (as the microphone does).
#[test]
fn a_clip_carries_desktop_audio_with_no_recording_running() {
    let Some((mut client, dir)) = client() else {
        return;
    };
    client.mods.settings.set_enabled("replay", true);
    client
        .mods
        .settings
        .set_bool("replay", ids::AUTO_START, false);
    client
        .mods
        .settings
        .set_bool("replay", ids::DESKTOP_AUDIO, true);
    client.mods.settings.set_key("replay", ids::KEY_CLIP, 88);
    let capture = mc_replay::desktop::DesktopCapture::fake();
    let feeder = capture.feeder();
    client.rec.test_desktop = Some(capture);
    open(
        &mut client,
        &chatty_capture(44),
        Path::new("desktop.mccap"),
        100_000,
    );
    let mut fed = 0usize;
    drive(
        &mut client,
        450,
        Duration::from_millis(100),
        |client, frame| {
            let chunk = tone(fed);
            feeder.feed(&chunk);
            fed += chunk.len();
            if frame == 5 {
                let conn = client.rec.conn.as_deref().expect("following the session");
                assert!(
                    conn.desktop.is_some(),
                    "desktop audio opens for the clip ring"
                );
                assert!(!client.rec.recording(), "no recording runs");
            }
        },
    );
    client.rec.clip_key();
    client.session = None;
    client.rec_frame(Instant::now() + Duration::from_secs(120));
    assert!(client.rec.conn.is_none(), "closed with the session");
    client.rec_shutdown();

    let library = Library::new(dir.path());
    assert!(library.list_dir(false).is_empty(), "no recording was made");
    let clips = library.list_dir(true);
    assert_eq!(clips.len(), 1);
    let reader = Reader::open(&clips[0].path).expect("the clip opens");
    let meta = reader.meta().clone();
    assert!(
        meta.has_desktop && meta.desktop_format.is_some(),
        "{meta:?}"
    );
    assert!(!meta.has_mic);
    let (start, end) = meta.window();
    let records: Vec<_> = reader.records().map(|r| r.expect("record")).collect();
    assert!(
        records.windows(2).all(|w| w[0].time <= w[1].time),
        "time order"
    );
    let mut pcm = Vec::new();
    let starts: Vec<u64> = records
        .iter()
        .filter(|r| r.stream == StreamId::Desktop)
        .map(|r| {
            let lead = mc_replay::desktop::decode_record(&r.payload, &mut pcm).expect("decodes");
            r.time - u64::from(lead)
        })
        .collect();
    assert!(
        starts.len() >= 1490,
        "about 50 frames a second over 30 s: {}",
        starts.len()
    );
    let first = *starts.first().expect("frames");
    let last = *starts.last().expect("frames");
    assert!(
        first <= start,
        "covers the window's start: {first} > {start}"
    );
    assert!(
        last + 40_000 >= end,
        "covers the window's end: {last} vs {end}"
    );
    assert!(starts.windows(2).all(|w| w[1] - w[0] == 20_000), "no gaps");
    let peak = pcm.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
    assert!(peak > 5200 && peak < 6800, "the tone survives: {peak}");
}

/// Off by default: with "Record desktop audio" off nothing is opened (the fake stays unused), the
/// recording has no desktop stream and its metadata says so. A negative control for the test
/// above (same capture, same driving).
#[test]
fn desktop_audio_is_off_by_default_and_opens_nothing() {
    let Some((mut client, dir)) = client() else {
        return;
    };
    client.mods.settings.set_enabled("replay", true);
    assert!(
        !client.mods.settings.bool("replay", ids::DESKTOP_AUDIO),
        "off by default"
    );
    let capture = mc_replay::desktop::DesktopCapture::fake();
    let feeder = capture.feeder();
    client.rec.test_desktop = Some(capture);
    open(
        &mut client,
        &chatty_capture(3),
        Path::new("desktop.mccap"),
        100_000,
    );
    let mut fed = 0usize;
    drive(&mut client, 40, Duration::from_millis(100), |client, _| {
        let chunk = tone(fed);
        feeder.feed(&chunk);
        fed += chunk.len();
        let conn = client.rec.conn.as_deref().expect("following the session");
        assert!(conn.desktop.is_none(), "never opened");
    });
    assert!(
        client.rec.test_desktop.is_some(),
        "the capture was never taken"
    );
    client.session = None;
    client.rec_frame(Instant::now() + Duration::from_secs(120));
    client.rec_shutdown();
    let files = Library::new(dir.path()).list_dir(false);
    assert_eq!(files.len(), 1);
    let reader = Reader::open(&files[0].path).expect("opens");
    assert!(!reader.meta().has_desktop);
    assert!(reader
        .records()
        .map(|r| r.expect("record"))
        .all(|r| r.stream != StreamId::Desktop));
}
