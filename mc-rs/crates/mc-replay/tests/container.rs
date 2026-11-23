//! The container end to end: writer → file → reader, crash recovery, the session buffer and its
//! clips, mid-world starts, file cuts and the library.

use std::path::Path;
use std::time::{Duration, Instant};

use mc_replay::{
    cut_file, read_info, Library, Reader, Record, ReplayKind, ReplayMeta, StreamId, Writer,
    WriterOptions,
};

/// Small blocks, so a few hundred records make many blocks.
fn small_blocks() -> WriterOptions {
    WriterOptions {
        level: 3,
        block_bytes: 4096,
        block_interval: Duration::from_secs(3600),
        sync_interval: None,
    }
}

/// A deterministic mix of streams, sizes and times (several records per frame time).
fn records(count: usize) -> Vec<Record> {
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut time = 0u64;
    (0..count)
        .map(|i| {
            if i % 3 == 0 {
                time += 1000 + next() % 9000;
            }
            let stream = match next() % 10 {
                0..=6 => StreamId::Packet,
                7 | 8 => StreamId::Client((next() % 6) as u8),
                _ => StreamId::Mic,
            };
            let len = match next() % 20 {
                0 => 5000 + (next() % 20_000) as usize,
                _ => (next() % 120) as usize,
            };
            let payload = (0..len)
                .map(|j| ((j as u64 * 31 + next() % 7) & 0xFF) as u8)
                .collect();
            Record {
                time,
                stream,
                payload,
            }
        })
        .collect()
}

fn meta(title: &str) -> ReplayMeta {
    ReplayMeta {
        title: title.to_owned(),
        server: "mc.example.net".to_owned(),
        player_name: "Dev".to_owned(),
        started_unix_ms: 1_700_000_000_000,
        ..ReplayMeta::default()
    }
}

fn write_all(path: &Path, records: &[Record], options: WriterOptions) -> ReplayMeta {
    let mut writer = Writer::create(path, &meta("round trip"), options).expect("create");
    for record in records {
        writer.push(record.time, record.stream, &record.payload);
    }
    let mut last = meta("round trip");
    last.duration = records.last().map_or(0, |r| r.time);
    let finished = writer.finish(last.clone()).wait(Duration::from_secs(30));
    let finished = finished.expect("finish");
    assert_eq!(finished.path, path);
    last
}

fn read_all(reader: &Reader) -> Vec<Record> {
    reader
        .records()
        .collect::<Result<Vec<_>, _>>()
        .expect("records")
}

/// Wait (up to 20 s) until `done` holds; the worker runs on its own thread.
fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Wait until the worker has stopped finishing blocks (everything sealed is written).
fn wait_for_quiet(writer: &Writer) {
    let mut last = u64::MAX;
    loop {
        std::thread::sleep(Duration::from_millis(60));
        let now = writer.stats().blocks;
        if now == last {
            return;
        }
        last = now;
    }
}

#[test]
fn every_record_comes_back_byte_for_byte_and_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.aeroreplay");
    let input = records(3000);
    let meta = write_all(&path, &input, small_blocks());

    let reader = Reader::open(&path).expect("open");
    assert!(!reader.recovered());
    assert_eq!(reader.meta(), &meta);
    assert!(
        reader.blocks().len() > 20,
        "{} blocks",
        reader.blocks().len()
    );
    assert_eq!(read_all(&reader), input);
    // The index agrees with the records.
    let total: u64 = reader.blocks().iter().map(|b| u64::from(b.records)).sum();
    assert_eq!(total, input.len() as u64);
    for pair in reader.blocks().windows(2) {
        assert!(pair[0].last_time <= pair[1].first_time);
        assert!(pair[0].offset < pair[1].offset);
    }

    let info = read_info(&path).expect("info");
    assert_eq!(info.meta, meta);
    assert!(!info.recovered);
    assert_eq!(info.size_bytes, std::fs::metadata(&path).unwrap().len());
}

#[test]
fn blocks_are_compressed_and_a_cursor_reads_without_copying() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("z.aeroreplay");
    // Repetitive packets, like entity movement.
    let input: Vec<Record> = (0..20_000u64)
        .map(|i| Record {
            time: i * 50_000,
            stream: StreamId::Packet,
            payload: vec![0x15, (i % 40) as u8, 0, 1, 0xFF, 0x02, 1],
        })
        .collect();
    write_all(&path, &input, WriterOptions::default());
    let size = std::fs::metadata(&path).unwrap().len();
    let raw: usize = input.iter().map(|r| r.payload.len() + 3).sum();
    assert!(size * 10 < raw as u64, "{size} bytes on disk for {raw} raw");

    let reader = Reader::open(&path).unwrap();
    let mut cursor = reader.cursor();
    let mut seen = 0;
    while let Some(record) = cursor.next().unwrap() {
        assert_eq!(record.payload, &input[seen].payload[..]);
        assert_eq!(record.time, input[seen].time);
        seen += 1;
    }
    assert_eq!(seen, input.len());
}

#[test]
fn seeking_starts_at_the_block_that_holds_the_time() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s.aeroreplay");
    let input = records(2000);
    write_all(&path, &input, small_blocks());
    let reader = Reader::open(&path).unwrap();
    for &target in &[0, 1, input[700].time, input[1999].time, u64::MAX] {
        let mut cursor = reader.cursor();
        cursor.seek(target);
        let first = cursor.next().unwrap().map(|r| r.time);
        let expected_first_at_or_after = input.iter().find(|r| r.time >= target).map(|r| r.time);
        match (first, expected_first_at_or_after) {
            (Some(first), Some(expected)) => {
                assert!(first <= expected, "seek({target}) began after the target");
                // Never more than one block early.
                let block = reader.block_for(target);
                assert_eq!(first, reader.blocks()[block].first_time);
            }
            (None, None) => {}
            other => panic!("seek({target}): {other:?}"),
        }
    }
}

/// A crash: the writer never finishes. Everything the worker wrote reads back; a cut-off or
/// damaged block ends the file there.
#[test]
fn a_crash_leaves_a_file_that_reads_up_to_the_last_written_block() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("crash.aeroreplay");
    let input = records(1500);
    let mut writer = Writer::create(&path, &meta("crash"), small_blocks()).unwrap();
    for record in &input[..1000] {
        writer.push(record.time, record.stream, &record.payload);
    }
    writer.flush();
    assert_eq!(writer.stats().records, 1000);
    wait_for_quiet(&writer);
    // "Crash": copy the file as it is on disk while the writer is still open.
    let snapshot = dir.path().join("snapshot.aeroreplay");
    std::fs::copy(&path, &snapshot).unwrap();
    // Records after the flush were never written; they must not appear.
    for record in &input[1000..] {
        writer.push(record.time, record.stream, &record.payload);
    }
    drop(writer);

    let reader = Reader::open(&snapshot).expect("a crashed file still opens");
    assert!(reader.recovered());
    assert_eq!(read_all(&reader), input[..1000].to_vec());
    assert_eq!(reader.meta().title, "crash");
    assert_eq!(reader.duration(), input[999].time);
    let info = read_info(&snapshot).unwrap();
    assert!(info.recovered);
    assert_eq!(info.meta.duration, input[999].time);

    // Cut the file inside its last block: that block is lost, the rest reads.
    let bytes = std::fs::read(&snapshot).unwrap();
    let last_block = reader.blocks().last().unwrap();
    let cut = last_block.offset as usize + 44 + (last_block.stored_len as usize / 2);
    let reader = Reader::from_bytes(bytes[..cut].to_vec()).unwrap();
    let kept: u64 = reader.blocks().iter().map(|b| u64::from(b.records)).sum();
    assert_eq!(read_all(&reader), input[..kept as usize].to_vec());
    assert_eq!(
        reader.blocks().len() + 1,
        Reader::open(&snapshot).unwrap().blocks().len()
    );

    // A flipped byte in a middle block's payload: the blocks before it are kept.
    let mut damaged = bytes.clone();
    let middle = Reader::open(&snapshot).unwrap().blocks()[5];
    damaged[middle.offset as usize + 44 + 3] ^= 0x40;
    let reader = Reader::from_bytes(damaged).unwrap();
    assert_eq!(reader.blocks().len(), 5);

    // Dropping the writer without `finish` still wrote a footer to the original.
    wait_until("the footer", || {
        read_info(&path).is_ok_and(|info| !info.recovered)
    });
    let finished = Reader::open(&path).unwrap();
    assert!(!finished.recovered(), "Drop writes the footer");
    assert_eq!(read_all(&finished), input);
}

#[test]
fn the_library_repairs_a_crashed_file_lists_renames_and_deletes() {
    let dir = tempfile::tempdir().unwrap();
    let library = Library::new(dir.path());
    let first = library.unique_path(false, "2026-11-12 21.41 mc.example.net");
    assert_eq!(
        first.file_name().unwrap().to_str().unwrap(),
        "2026-11-12 21.41 mc.example.net.aeroreplay"
    );
    let input = records(400);
    write_all(&first, &input, small_blocks());
    assert_ne!(
        library.unique_path(false, "2026-11-12 21.41 mc.example.net"),
        first
    );
    assert!(library
        .unique_path(true, "a:b/c?")
        .ends_with("clips/a-b-c-.aeroreplay"));

    // A clip, unfinished (as a crash leaves it), backdated past the live-file grace period.
    let clip = library.unique_path(true, "clip");
    let mut writer = Writer::create(&clip, &meta("a clip"), small_blocks()).unwrap();
    for record in &input[..200] {
        writer.push(record.time, record.stream, &record.payload);
    }
    writer.flush();
    wait_for_quiet(&writer);
    let crashed = dir
        .path()
        .join("replays")
        .join("clips")
        .join("crashed.aeroreplay");
    std::fs::copy(&clip, &crashed).unwrap();
    writer
        .finish(meta("a clip"))
        .wait(Duration::from_secs(10))
        .unwrap();
    std::fs::remove_file(&clip).unwrap();
    let old = std::time::SystemTime::now() - Duration::from_secs(600);
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&crashed)
        .unwrap();
    file.set_modified(old).unwrap();
    drop(file);

    let listed = library.list();
    assert_eq!(listed.len(), 2);
    let crashed_entry = listed.iter().find(|e| e.is_clip).unwrap();
    assert!(crashed_entry.recovered);
    assert_eq!(library.repair_unfinished(), vec![crashed.clone()]);
    let repaired = Reader::open(&crashed).unwrap();
    assert!(!repaired.recovered());
    assert!(!read_all(&repaired).is_empty());

    library.rename(&first, "  Best game  ").unwrap();
    assert_eq!(read_info(&first).unwrap().meta.title, "Best game");
    assert_eq!(
        read_all(&Reader::open(&first).unwrap()),
        input,
        "a rename keeps every record"
    );

    let usage = library.usage();
    assert_eq!((usage.full_count, usage.clip_count), (1, 1));
    // Outside the library, or not a replay: refused.
    assert!(library.delete(&dir.path().join("x.aeroreplay")).is_err());
    library.delete(&first).unwrap();
    assert!(!first.exists());
}

#[test]
fn the_storage_cap_deletes_the_oldest_recordings_and_never_clips() {
    let dir = tempfile::tempdir().unwrap();
    let library = Library::new(dir.path());
    let input = records(300);
    let mut paths = Vec::new();
    for i in 0..4u64 {
        let path = library.unique_path(false, &format!("game {i}"));
        let mut m = meta(&format!("game {i}"));
        m.started_unix_ms = 1_000 + i;
        let mut writer = Writer::create(&path, &m, small_blocks()).unwrap();
        for record in &input {
            writer.push(record.time, record.stream, &record.payload);
        }
        writer.finish(m).wait(Duration::from_secs(10)).unwrap();
        // Old enough not to count as a live recording.
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_modified(std::time::SystemTime::now() - Duration::from_secs(600))
            .unwrap();
        paths.push(path);
    }
    let clip = library.unique_path(true, "keep me");
    write_all(&clip, &input, small_blocks());
    let each = std::fs::metadata(&paths[0]).unwrap().len();
    let deleted = library.enforce_cap(each * 2 + 10, None);
    assert_eq!(
        deleted,
        vec![paths[0].clone(), paths[1].clone()],
        "oldest first"
    );
    assert!(paths[2].exists() && paths[3].exists() && clip.exists());
    assert!(library.enforce_cap(0, Some(&paths[3])).contains(&paths[2]));
    assert!(paths[3].exists(), "the recording in progress is kept");
    assert!(clip.exists(), "clips are never deleted");
}

/// The clip key's path: a memory writer holds the session from its last world reset on, and a
/// clip is that, retimed so the reset is 0, after a state prefix.
#[test]
fn the_session_buffer_keeps_the_last_world_and_saves_clips() {
    let dir = tempfile::tempdir().unwrap();
    let mut buffer = Writer::memory(small_blocks(), 64 << 20);
    let pkt = |time: u64, tag: u8| (time, StreamId::Packet, vec![tag; 40]);
    // A first world, then a world reset at 5 s, then 30 s of play.
    buffer.keyframe();
    for (time, stream, payload) in (0..50).map(|i| pkt(i * 100_000, 1)) {
        buffer.push(time, stream, &payload);
    }
    buffer.keyframe();
    let world_two: Vec<_> = (0..300).map(|i| pkt(5_000_000 + i * 100_000, 2)).collect();
    for (time, stream, payload) in &world_two {
        buffer.push(*time, *stream, payload);
    }
    buffer.push(35_000_000, StreamId::Client(3), b"camera");
    let path = dir.path().join("clips").join("c.aeroreplay");
    let prefix = vec![Record {
        time: 0,
        stream: StreamId::Packet,
        payload: b"\x01join".to_vec(),
    }];
    let mut clip_meta = meta("clip");
    clip_meta.world_resets = vec![0, 5_000_000];
    let finished = buffer
        .save_clip(
            &path,
            clip_meta,
            prefix.clone(),
            (20_000_000, 35_000_000),
            Vec::new(),
        )
        .wait(Duration::from_secs(10))
        .expect("clip");
    assert_eq!(
        finished.meta.kind,
        ReplayKind::Clip {
            start: 15_000_000,
            end: 30_000_000
        }
    );
    assert_eq!(finished.meta.window(), (15_000_000, 30_000_000));
    assert_eq!(finished.meta.duration, 30_000_000);
    assert_eq!(finished.meta.world_resets, vec![0]);

    let reader = Reader::open(&path).unwrap();
    assert!(reader.blocks()[0].prefix);
    assert!(reader.blocks()[1].keyframe);
    let got = read_all(&reader);
    assert_eq!(got[0], prefix[0]);
    assert_eq!(
        got.len(),
        1 + world_two.len() + 1,
        "the first world is gone"
    );
    for (record, (time, _, payload)) in got[1..].iter().zip(&world_two) {
        assert_eq!(record.time, time - 5_000_000);
        assert_eq!(&record.payload, payload);
    }
    assert_eq!(got.last().unwrap().stream, StreamId::Client(3));
    assert!(buffer.stats().ready);

    // A second clip with the same name is refused (create_new), not overwritten.
    assert!(buffer
        .save_clip(&path, meta("again"), Vec::new(), (0, 1), Vec::new())
        .wait(Duration::from_secs(10))
        .is_err());
}

/// Records kept outside the buffer (the client's microphone ring) are merged into a clip in
/// time order: between blocks, inside blocks (recompressed) and after the last one. A world
/// reset's block still starts with the reset, even with an extra record at the same time.
#[test]
fn a_clip_merges_records_kept_outside_the_buffer_in_time_order() {
    let dir = tempfile::tempdir().unwrap();
    let mut buffer = Writer::memory(small_blocks(), 64 << 20);
    buffer.keyframe();
    buffer.push(10_000_000, StreamId::Packet, b"\x07reset");
    let mut world = vec![(10_000_000u64, b"\x07reset".to_vec())];
    for i in 1..400u64 {
        let time = 10_000_000 + i * 50_000;
        let payload = vec![0x15, (i % 251) as u8, 7, 7, 7];
        buffer.push(time, StreamId::Packet, &payload);
        world.push((time, payload));
    }
    // Mic frames every 20 ms from the reset to 2 s after the last world record.
    let mic: Vec<Record> = (0..1500u64)
        .map(|i| Record {
            time: 10_000_000 + i * 20_000,
            stream: StreamId::Mic,
            payload: vec![0xAA, (i % 256) as u8],
        })
        .collect();
    // An extra record from before the world: dropped.
    let mut extra = vec![Record {
        time: 1,
        stream: StreamId::Mic,
        payload: vec![0xBB],
    }];
    extra.extend(mic.iter().cloned());
    let path = dir.path().join("mic-clip.aeroreplay");
    let finished = buffer
        .save_clip(
            &path,
            meta("mic"),
            Vec::new(),
            (20_000_000, 40_000_000),
            extra,
        )
        .wait(Duration::from_secs(10))
        .expect("clip");
    assert_eq!(finished.meta.window(), (10_000_000, 30_000_000));

    let reader = Reader::open(&path).unwrap();
    let got = read_all(&reader);
    assert!(got.windows(2).all(|w| w[0].time <= w[1].time), "time order");
    assert_eq!(got[0].payload, b"\x07reset", "the reset still comes first");
    assert!(reader.blocks()[0].keyframe);
    let packets: Vec<(u64, Vec<u8>)> = got
        .iter()
        .filter(|r| r.stream == StreamId::Packet)
        .map(|r| (r.time + 10_000_000, r.payload.clone()))
        .collect();
    assert_eq!(packets, world, "every world record, in order");
    let mics: Vec<(u64, Vec<u8>)> = got
        .iter()
        .filter(|r| r.stream == StreamId::Mic)
        .map(|r| (r.time + 10_000_000, r.payload.clone()))
        .collect();
    let expected: Vec<(u64, Vec<u8>)> = mic.iter().map(|r| (r.time, r.payload.clone())).collect();
    assert_eq!(
        mics, expected,
        "every mic record from the world on, at its time"
    );
    let total: u64 = reader.blocks().iter().map(|b| u64::from(b.records)).sum();
    assert_eq!(total as usize, world.len() + mic.len());
}

#[test]
fn the_session_buffer_is_bounded_and_waits_for_a_world() {
    let mut buffer = Writer::memory(small_blocks(), 64 * 1024);
    // No world reset yet: nothing is kept.
    for i in 0..100u64 {
        buffer.push(i, StreamId::Packet, &[7; 100]);
    }
    buffer.flush();
    wait_for_quiet(&buffer);
    assert!(buffer.stats().blocks > 0);
    assert!(!buffer.stats().ready);
    assert_eq!(buffer.stats().stored_bytes, 0);
    buffer.keyframe();
    buffer.push(1000, StreamId::Packet, &[1; 10]);
    buffer.flush();
    wait_until("ready", || buffer.stats().ready);
    assert!(buffer.stats().stored_bytes > 0);
    // Incompressible data past the 64 KiB cap: it drops everything until the next reset.
    let mut seed = 1u32;
    for i in 0..400u64 {
        let noise: Vec<u8> = (0..400)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 24) as u8
            })
            .collect();
        buffer.push(2000 + i, StreamId::Packet, &noise);
    }
    buffer.flush();
    wait_until("the overflow", || buffer.stats().overflowed);
    wait_for_quiet(&buffer);
    let stats = buffer.stats();
    assert!(stats.overflowed && !stats.ready, "{stats:?}");
    assert_eq!(stats.stored_bytes, 0, "{stats:?}");
    buffer.keyframe();
    buffer.push(10_000, StreamId::Packet, &[1; 10]);
    buffer.flush();
    wait_until("the next world", || buffer.stats().ready);
    let stats = buffer.stats();
    assert!(stats.ready && !stats.overflowed, "{stats:?}");
}

/// The start key in the middle of a world: the file begins with the prefix and the buffered
/// world (retimed), then continues live on the same clock.
#[test]
fn a_recording_started_mid_world_begins_with_the_buffered_world() {
    let dir = tempfile::tempdir().unwrap();
    let mut buffer = Writer::memory(small_blocks(), 64 << 20);
    buffer.keyframe();
    let base = 3_000_000u64;
    let buffered: Vec<_> = (0..200u64)
        .map(|i| (base + i * 10_000, vec![i as u8; 30]))
        .collect();
    for (time, payload) in &buffered {
        buffer.push(*time, StreamId::Packet, payload);
    }
    let snapshot = buffer.snapshot();
    let path = dir.path().join("mid.aeroreplay");
    let prefix = vec![Record {
        time: 0,
        stream: StreamId::Packet,
        payload: b"\x38tab".to_vec(),
    }];
    let mut m = meta("mid");
    m.view_start = buffered.last().unwrap().0 - base;
    let mut writer =
        Writer::create_with_prelude(&path, &m, small_blocks(), prefix.clone(), snapshot).unwrap();
    // Live records, on the file clock (session time minus the base).
    let live: Vec<_> = (0..50u64)
        .map(|i| (2_100_000 + i * 10_000, vec![0xEE; 12]))
        .collect();
    for (time, payload) in &live {
        writer.push(*time, StreamId::Packet, payload);
    }
    writer
        .finish(m.clone())
        .wait(Duration::from_secs(10))
        .unwrap();

    let reader = Reader::open(&path).unwrap();
    let got = read_all(&reader);
    assert_eq!(got[0], prefix[0]);
    assert_eq!(got.len(), 1 + buffered.len() + live.len());
    for (record, (time, payload)) in got[1..=buffered.len()].iter().zip(&buffered) {
        assert_eq!((record.time, &record.payload), (time - base, payload));
    }
    for (record, (time, payload)) in got[1 + buffered.len()..].iter().zip(&live) {
        assert_eq!((record.time, &record.payload), (*time, payload));
    }
    assert_eq!(reader.meta().window().0, m.view_start);
}

#[test]
fn a_file_cut_keeps_the_prefix_the_caller_asks_for_and_the_world_from_its_reset() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("full.aeroreplay");
    let mut writer = Writer::create(&path, &meta("full"), small_blocks()).unwrap();
    // World one (0-10 s), with "sticky" records tagged 0x38; world two from 10 s.
    writer.keyframe();
    writer.push(0, StreamId::Packet, b"\x01join");
    for i in 1..100u64 {
        let tag = if i % 10 == 0 { 0x38 } else { 0x15 };
        writer.push(i * 100_000, StreamId::Packet, &[tag, i as u8]);
    }
    writer.keyframe();
    writer.push(10_000_000, StreamId::Packet, b"\x07respawn");
    for i in 1..200u64 {
        writer.push(10_000_000 + i * 100_000, StreamId::Packet, &[0x15, i as u8]);
    }
    let mut m = meta("full");
    m.world_resets = vec![0, 10_000_000];
    writer.finish(m).wait(Duration::from_secs(10)).unwrap();

    let reader = Reader::open(&path).unwrap();
    let out = dir.path().join("cut.aeroreplay");
    let cut = cut_file(&reader, (15_000_000, 20_000_000), &out, |r| {
        matches!(r.payload.first(), Some(0x01 | 0x38))
    })
    .unwrap();
    assert_eq!(cut.base_time, 10_000_000);
    assert_eq!(
        cut.prefix_records,
        1 + 9,
        "the join and the nine 0x38 records"
    );
    let clip = Reader::open(&out).unwrap();
    assert_eq!(
        clip.meta().kind,
        ReplayKind::Clip {
            start: 5_000_000,
            end: 10_000_000
        }
    );
    assert_eq!(clip.meta().world_resets, vec![0]);
    let got = read_all(&clip);
    assert_eq!(got[0].payload, b"\x01join");
    assert!(got[..10].iter().all(|r| r.time == 0));
    assert_eq!(got[10].payload, b"\x07respawn");
    assert_eq!(got[10].time, 0);
    assert!(got.last().unwrap().time >= 10_000_000);
}
