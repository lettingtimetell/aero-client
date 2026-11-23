//! Replays are shared between players, so a file can be damaged or crafted. Opening one must not
//! exhaust memory, and its metadata must not carry values that make the browser, viewer or
//! exporter allocate, loop or overflow.
//!
//! This binary counts the bytes live on the test's own thread (a counting global allocator; the
//! library is unchanged), so a memory bound is a measured fact.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::Path;
use std::time::Duration;

use mc_replay::{
    read_info, MicFormat, Reader, ReplayError, ReplayKind, ReplayMeta, StreamId, Writer,
    WriterOptions, MAX_META_TIME, MAX_STARTED_UNIX_MS, RECORDS_MEMORY_BASE,
};

struct Counting;

thread_local! {
    static LIVE: Cell<i64> = const { Cell::new(0) };
    static PEAK: Cell<i64> = const { Cell::new(0) };
}

fn grew(bytes: usize) {
    let _ = LIVE.try_with(|live| {
        live.set(live.get() + bytes as i64);
        let _ = PEAK.try_with(|peak| peak.set(peak.get().max(live.get())));
    });
}

fn shrank(bytes: usize) {
    let _ = LIVE.try_with(|live| live.set(live.get() - bytes as i64));
}

// SAFETY: forwards every call to the system allocator unchanged; only counts on the side.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        grew(layout.size());
        // SAFETY: the caller's contract is passed through.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        grew(layout.size());
        // SAFETY: as above.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        grew(new_size);
        shrank(layout.size());
        // SAFETY: as above.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        shrank(layout.size());
        // SAFETY: as above.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The most bytes live on this thread at once while `run` ran, above what was live before it.
fn peak_during<T>(run: impl FnOnce() -> T) -> (T, i64) {
    let before = LIVE.with(Cell::get);
    PEAK.with(|peak| peak.set(before));
    let out = run();
    (out, PEAK.with(Cell::get) - before)
}

/// A file with no footer and one block: `count` copies of `record`, compressed as the writer
/// compresses (one zstd frame). The reader walks the blocks, as it does for a file a crash cut.
fn crafted(count: u32, record: &[u8], meta_json: &str) -> Vec<u8> {
    crafted_with_streams(count, record, meta_json, 1)
}

/// [`crafted`] with the block's stream mask (1 packets, 8 markers).
fn crafted_with_streams(count: u32, record: &[u8], meta_json: &str, streams: u16) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"AEROREPL");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(meta_json.len() as u32).to_le_bytes());
    out.extend_from_slice(meta_json.as_bytes());
    out.extend_from_slice(&crc32fast::hash(meta_json.as_bytes()).to_le_bytes());

    let raw = record.repeat(count as usize);
    let stored = zstd::bulk::compress(&raw, 3).expect("compress");
    let mut header = [0u8; 44];
    header[0..4].copy_from_slice(b"ARBK");
    header[4..8].copy_from_slice(&(stored.len() as u32).to_le_bytes());
    header[8..12].copy_from_slice(&(raw.len() as u32).to_le_bytes());
    header[12..16].copy_from_slice(&count.to_le_bytes());
    // first and last time 0; codec 1 (zstd); flags 1 (keyframe); the stream mask.
    header[32] = 1;
    header[33] = 1;
    header[34..36].copy_from_slice(&streams.to_le_bytes());
    header[36..40].copy_from_slice(&crc32fast::hash(&stored).to_le_bytes());
    let crc = crc32fast::hash(&header[0..40]);
    header[40..44].copy_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(&stored);
    out
}

/// One packet record of no bytes: time delta 0, tag 0 (packet), length 0.
const EMPTY_RECORD: [u8; 3] = [0, 0, 0];

#[test]
fn a_block_of_millions_of_empty_records_is_refused_within_its_budget() {
    // 16M records: 48 MB once decompressed, a few kilobytes on disk.
    let count = 16_000_000u32;
    let bytes = crafted(count, &EMPTY_RECORD, &ReplayMeta::default().to_json());
    let file_len = bytes.len();
    assert!(
        file_len < 64 * 1024,
        "the crafted file is small: {file_len} bytes"
    );
    let reader = Reader::from_bytes(bytes).expect("the header is fine");
    assert_eq!(reader.blocks().len(), 1);
    assert_eq!(reader.blocks()[0].records, count);

    // The negative control: every record really is there to read, and holding them the way
    // the viewer did (one `Record` each) is over 600 MB from this small file.
    let mut cursor = reader.cursor();
    let mut seen = 0u64;
    while let Some(record) = cursor.next().expect("a valid block") {
        assert_eq!(record.stream, StreamId::Packet);
        seen += 1;
    }
    assert_eq!(seen, u64::from(count));
    let held = seen * std::mem::size_of::<mc_replay::Record>() as u64;
    assert!(held > 600 << 20, "{held} bytes");

    let (result, peak) = peak_during(|| reader.all_records());
    match result {
        Err(ReplayError::TooLarge(why)) => eprintln!("refused: {why}"),
        other => panic!("expected TooLarge, got {:?}", other.map(|r| r.len())),
    }
    eprintln!("peak while refusing: {} MiB", peak >> 20);
    // The budget counts 64 bytes a record (256 MiB here); the `Vec` holding them doubles, so at
    // the moment it grows the old and new buffers are both live. Bounded by the budget, not by
    // the 16M records the file holds (the old collector peaked near 1 GiB on this file, and
    // grows without bound with more blocks).
    assert!(
        peak < 2 * (RECORDS_MEMORY_BASE as i64) + (64 << 20),
        "peak {peak} bytes"
    );

    // The viewer's packed store (both ways it reads a file) applies the same budget.
    let (result, peak) = peak_during(|| mc_replay::RecordStore::read_all(&reader));
    assert!(
        matches!(result, Err(ReplayError::TooLarge(_))),
        "read_all: {:?}",
        result.map(|store| store.len())
    );
    eprintln!("RecordStore::read_all peak: {} MiB", peak >> 20);
    assert!(
        peak < 2 * (RECORDS_MEMORY_BASE as i64) + (64 << 20),
        "{peak}"
    );
    let (result, peak) = peak_during(|| mc_replay::RecordStore::read_all_in_file(reader, 1024));
    assert!(
        matches!(result, Err(ReplayError::TooLarge(_))),
        "read_all_in_file: {:?}",
        result.map(|store| store.len())
    );
    eprintln!("RecordStore::read_all_in_file peak: {} MiB", peak >> 20);
    assert!(
        peak < 2 * (RECORDS_MEMORY_BASE as i64) + (64 << 20),
        "{peak}"
    );
}

/// The start-up repair (files a crash left without a footer get one, on a helper thread) used
/// to collect every marker record: 4 million empty ones in this small file became 4 million
/// markers in memory and in the footer it wrote. Now at most the metadata's 4096.
#[test]
fn the_startup_repair_keeps_a_bounded_number_of_markers() {
    let game = tempfile::tempdir().expect("tempdir");
    let library = mc_replay::Library::new(game.path());
    std::fs::create_dir_all(library.replays_dir()).expect("replays folder");
    let path = library.replays_dir().join("shared.aeroreplay");
    // Each record: time delta 0, tag 3 (marker), length 0.
    let bytes = crafted_with_streams(4_000_000, &[0, 3, 0], &ReplayMeta::default().to_json(), 8);
    std::fs::write(&path, bytes).expect("write");
    // Older than the two minutes a recording in progress is left alone for.
    std::fs::File::options()
        .write(true)
        .open(&path)
        .expect("open")
        .set_modified(std::time::SystemTime::now() - Duration::from_secs(600))
        .expect("age the file");

    let reader = Reader::open(&path).expect("open");
    let (markers, peak) = peak_during(|| mc_replay::markers_in(&reader));
    assert_eq!(markers.len(), mc_replay::MAX_META_MARKERS);
    eprintln!("markers_in peak: {} MiB", peak >> 20);
    assert!(peak < 32 << 20, "{peak}");
    drop(reader);

    let repaired = library.repair_unfinished();
    assert_eq!(repaired, vec![path.clone()]);
    let info = read_info(&path).expect("listed");
    assert!(!info.recovered, "it has a footer now");
    assert_eq!(info.meta.markers.len(), mc_replay::MAX_META_MARKERS);
    assert!(
        info.size_bytes < 1 << 20,
        "a small footer: {} bytes",
        info.size_bytes
    );
}

#[test]
fn a_real_recording_reads_the_same_through_the_budget() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("game.aeroreplay");
    let options = WriterOptions {
        block_bytes: 16 * 1024,
        sync_interval: None,
        ..WriterOptions::default()
    };
    let mut writer = Writer::create(&path, &ReplayMeta::default(), options).expect("create");
    for i in 0..20_000u64 {
        let payload: Vec<u8> = (0..(i % 300) as u8).collect();
        let stream = if i % 7 == 0 {
            StreamId::Client(3)
        } else {
            StreamId::Packet
        };
        writer.push(i * 50_000, stream, &[&[0x21u8][..], &payload].concat());
    }
    let last = ReplayMeta {
        duration: 19_999 * 50_000,
        ..ReplayMeta::default()
    };
    writer
        .finish(last)
        .wait(Duration::from_secs(30))
        .expect("finish");

    let reader = Reader::open(&path).expect("open");
    let all = reader
        .all_records()
        .expect("a real recording is within budget");
    let plain: Vec<_> = reader.records().collect::<Result<_, _>>().expect("records");
    assert_eq!(all.len(), 20_000);
    assert_eq!(all, plain);
}

#[test]
fn a_header_longer_than_the_file_is_refused_before_it_is_allocated() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("short.aeroreplay");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"AEROREPL");
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&(16u32 << 20).to_le_bytes()); // 16 MiB of metadata, it says
    bytes.extend_from_slice(b"{}");
    std::fs::write(&path, &bytes).expect("write");
    let (result, peak) = peak_during(|| read_info(&path));
    assert!(matches!(result, Err(ReplayError::Corrupt(_))), "{result:?}");
    assert!(
        peak < 1 << 20,
        "the browser's listing allocated {peak} bytes"
    );
}

/// A file whose header carries `meta_json` and one packet record (delta 0, tag 0 for a packet,
/// length 1, packet id 0).
fn write_with_meta(path: &Path, meta_json: &str) {
    std::fs::write(path, crafted(1, &[0, 0, 1, 0], meta_json)).expect("write");
}

#[test]
fn impossible_metadata_values_are_brought_into_range() {
    let hostile = format!(
        r#"{{"started_unix_ms":18446744073709551615,"duration":18446744073709551615,
            "view_start":18446744073709551615,
            "kind":{{"type":"clip","start":18446744073709551615,"end":18446744073709551615}},
            "title":"{long}","server":"{long}","player_name":"{long}",
            "markers":[{markers}],
            "mic_format":{{"codec":"ima-adpcm","sample_rate":4000000000,"channels":1,
                           "frame_samples":480}},
            "desktop_format":{{"codec":"ima-adpcm","sample_rate":24000,"channels":0,
                               "frame_samples":480}},
            "has_mic":true,"has_desktop":true}}"#,
        long = "x".repeat(100_000),
        markers = vec![r#"{"time":18446744073709551615,"label":"m"}"#; 10_000].join(","),
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("hostile.aeroreplay");
    write_with_meta(&path, &hostile);

    for meta in [
        read_info(&path).expect("listed").meta,
        Reader::open(&path).expect("opened").meta().clone(),
    ] {
        assert_eq!(meta.started_unix_ms, MAX_STARTED_UNIX_MS);
        assert_eq!(meta.duration, MAX_META_TIME);
        assert_eq!(meta.view_start, MAX_META_TIME);
        assert_eq!(
            meta.kind,
            ReplayKind::Clip {
                start: MAX_META_TIME,
                end: MAX_META_TIME
            }
        );
        assert_eq!(meta.title.chars().count(), mc_replay::MAX_META_TEXT_CHARS);
        assert_eq!(meta.server.chars().count(), mc_replay::MAX_META_TEXT_CHARS);
        assert_eq!(meta.markers.len(), mc_replay::MAX_META_MARKERS);
        assert!(meta.markers.iter().all(|m| m.time == MAX_META_TIME));
        assert_eq!(meta.mic_format, None, "a 4 GHz microphone is dropped");
        assert_eq!(meta.desktop_format, None, "zero channels is dropped");
        // The start plus the whole window still fits every clock the client converts it to.
        let end_ms = meta.started_unix_ms + meta.duration / 1000;
        assert!(std::time::UNIX_EPOCH
            .checked_add(Duration::from_millis(end_ms))
            .is_some());
    }
}

#[test]
fn metadata_the_client_writes_is_unchanged() {
    let meta = ReplayMeta {
        title: "Sky Wars \u{b7} Nov 12, 9:41 PM".to_owned(),
        server: "mc.example.net".to_owned(),
        player_name: "Dev".to_owned(),
        player_uuid: "00000000-0000-0000-0000-000000000000".to_owned(),
        started_unix_ms: 1_794_541_686_841,
        duration: 3 * 3600 * 1_000_000,
        view_start: 1_000_000,
        world_resets: vec![0, 60_000_000],
        markers: vec![mc_replay::Marker {
            time: 5_000_000,
            label: "Bookmark".to_owned(),
        }],
        has_mic: true,
        mic_format: Some(mc_replay::mic::format()),
        has_desktop: true,
        desktop_format: Some(mc_replay::desktop::format()),
        tags: [("rb.version".to_owned(), "3".to_owned())].into(),
        ..ReplayMeta::default()
    };
    let json = meta.to_json();
    let read = ReplayMeta::from_json(&json).expect("parse");
    assert_eq!(read, meta);
    assert_eq!(read.to_json(), json);
    assert!(mc_replay::mic::format().is_plausible());
    assert!(!MicFormat {
        codec: "ima-adpcm".to_owned(),
        sample_rate: 0,
        channels: 1,
        frame_samples: 480,
    }
    .is_plausible());
}
