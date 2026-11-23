//! Replay files are shared between players. Every part of one is checksummed, so random damage
//! mostly stops at a checksum; this fuzzer also re-seals what it damages (the metadata with its
//! checksum, a block's records stored uncompressed with fresh checksums and the footer dropped,
//! as a crash leaves a file), so the damage reaches the record and metadata decoders. Nothing may
//! panic, loop or allocate without bound.
//!
//! A few seconds; for a longer run by hand:
//!
//! ```text
//! REPLAY_FUZZ_ROUNDS=20 REPLAY_FUZZ_SEED=12345 cargo test -p mc-replay --test hostile_replay_files -- --nocapture
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::panic::AssertUnwindSafe;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use mc_replay::{
    markers_in, read_info, Marker, Reader, RecordStore, ReplayMeta, StreamId, Writer, WriterOptions,
};

struct Counting;

const REFUSED_ALLOCATION: usize = 1 << 30;

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

// SAFETY: forwards every call to the system allocator unchanged, except that a request over
// `REFUSED_ALLOCATION` gets a null pointer, which every caller must already handle.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > REFUSED_ALLOCATION {
            return std::ptr::null_mut();
        }
        grew(layout.size());
        // SAFETY: the caller's contract is passed through.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() > REFUSED_ALLOCATION {
            return std::ptr::null_mut();
        }
        grew(layout.size());
        // SAFETY: as above.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > REFUSED_ALLOCATION {
            return std::ptr::null_mut();
        }
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

const MIB: i64 = 1 << 20;

fn peak_during<T>(run: impl FnOnce() -> T) -> (T, i64) {
    let before = LIVE.with(Cell::get);
    PEAK.with(|peak| peak.set(before));
    let out = run();
    (out, PEAK.with(Cell::get) - before)
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
}

const INTERESTING: &[&[u8]] = &[
    &[0],
    &[0xFF],
    &[0x7F],
    &[0x80],
    &[0xFF, 0xFF, 0xFF, 0xFF, 0x0F],
    &[0x81, 0x80, 0x80, 0x80],
    &[0xFF; 10],
    b"{",
    b"}",
    b"[",
    b"\"",
    b":",
    b",",
    b"null",
    b"-1",
    b"1e309",
    b"18446744073709551615",
    b"\"markers\"",
    b"\"duration\"",
    b"\"world_resets\"",
    b"\"started_unix_ms\"",
    b"\"mic_format\"",
    b"\"tags\"",
];

fn mutate(rng: &mut Rng, data: &mut Vec<u8>) {
    for _ in 0..=rng.below(4) {
        let len = data.len();
        let token = INTERESTING[rng.below(INTERESTING.len())];
        match rng.below(7) {
            0 if len > 0 => {
                let at = rng.below(len);
                data[at] ^= 1 << rng.below(8);
            }
            1 if len > 0 => {
                let at = rng.below(len);
                data[at] = rng.next() as u8;
            }
            2 => {
                let at = rng.below(len + 1);
                data.splice(at..at, token.iter().copied());
            }
            3 if len > 0 => {
                let at = rng.below(len);
                let end = (at + 1 + rng.below(32)).min(len);
                data.drain(at..end);
            }
            4 if len > 0 => {
                let at = rng.below(len);
                let end = (at + 1 + rng.below(64)).min(len);
                let piece = data[at..end].to_vec();
                let to = rng.below(data.len() + 1);
                data.splice(to..to, piece);
            }
            5 if len > 1 => data.truncate(rng.below(len)),
            _ => {
                let at = rng.below(len + 1);
                let times = 1 + rng.below(100);
                let run: Vec<u8> = token
                    .iter()
                    .copied()
                    .cycle()
                    .take(token.len() * times)
                    .collect();
                data.splice(at..at, run);
            }
        }
    }
}

// ---- the container, as far as re-sealing needs it ---------------------------------------------

const FILE_HEADER_FIXED: usize = 16;
const BLOCK_HEADER_LEN: usize = 44;
const CODEC_STORED: u8 = 0;
const CODEC_ZSTD: u8 = 1;

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

struct Parsed {
    meta_json: Vec<u8>,
    /// `(the 44-byte header, the decompressed records)`.
    blocks: Vec<([u8; BLOCK_HEADER_LEN], Vec<u8>)>,
    footer: Vec<u8>,
}

fn parse(file: &[u8]) -> Parsed {
    let meta_len = u32_at(file, 12) as usize;
    let meta_json = file[FILE_HEADER_FIXED..FILE_HEADER_FIXED + meta_len].to_vec();
    let mut pos = FILE_HEADER_FIXED + meta_len + 4;
    let mut blocks = Vec::new();
    while file.get(pos..pos + 4) == Some(b"ARBK".as_slice()) {
        let header: [u8; BLOCK_HEADER_LEN] = file[pos..pos + BLOCK_HEADER_LEN]
            .try_into()
            .expect("a block header");
        let stored_len = u32_at(&header, 4) as usize;
        let raw_len = u32_at(&header, 8) as usize;
        let payload = &file[pos + BLOCK_HEADER_LEN..pos + BLOCK_HEADER_LEN + stored_len];
        let raw = match header[32] {
            CODEC_ZSTD => zstd::bulk::decompress(payload, raw_len).expect("a writer's block"),
            _ => payload.to_vec(),
        };
        blocks.push((header, raw));
        pos += BLOCK_HEADER_LEN + stored_len;
    }
    Parsed {
        meta_json,
        blocks,
        footer: file[pos..].to_vec(),
    }
}

/// A file from parts: the header around `meta_json` with its checksum, each block stored
/// uncompressed with fresh lengths and checksums, then `footer` (empty: none, as after a crash).
fn seal(meta_json: &[u8], blocks: &[([u8; BLOCK_HEADER_LEN], Vec<u8>)], footer: &[u8]) -> Vec<u8> {
    let mut out = b"AEROREPL".to_vec();
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(meta_json.len() as u32).to_le_bytes());
    out.extend_from_slice(meta_json);
    out.extend_from_slice(&crc32fast::hash(meta_json).to_le_bytes());
    for (header, raw) in blocks {
        let mut header = *header;
        header[4..8].copy_from_slice(&(raw.len() as u32).to_le_bytes());
        header[8..12].copy_from_slice(&(raw.len() as u32).to_le_bytes());
        header[32] = CODEC_STORED;
        header[36..40].copy_from_slice(&crc32fast::hash(raw).to_le_bytes());
        let crc = crc32fast::hash(&header[0..40]);
        header[40..44].copy_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(raw);
    }
    out.extend_from_slice(footer);
    out
}

fn seed_replays(dir: &std::path::Path) -> Vec<Vec<u8>> {
    let options = WriterOptions {
        block_bytes: 2048,
        sync_interval: None,
        ..WriterOptions::default()
    };
    let mut files = Vec::new();
    for (index, records) in [200u64, 400].into_iter().enumerate() {
        let path = dir.join(format!("seed{index}.aeroreplay"));
        let mut meta = ReplayMeta {
            title: format!("seed {index}"),
            server: "test.example".to_owned(),
            started_unix_ms: 1_700_000_000_000,
            markers: vec![Marker {
                time: 2_000_000,
                label: "mark".to_owned(),
            }],
            world_resets: vec![0, 1_000_000],
            ..ReplayMeta::default()
        };
        let mut writer = Writer::create(&path, &meta, options).expect("create");
        writer.keyframe();
        for record in 0..records {
            let stream = match record % 7 {
                0 => StreamId::Client((record % 8) as u8),
                1 => StreamId::Marker,
                2 => StreamId::Mic,
                _ => StreamId::Packet,
            };
            let payload: Vec<u8> = (0..(record % 90) as u8).collect();
            writer.push(record * 50_000, stream, &payload);
        }
        meta.duration = records * 50_000;
        writer
            .finish(meta)
            .wait(Duration::from_secs(30))
            .expect("finish");
        files.push(std::fs::read(&path).expect("read the seed"));
    }
    files
}

fn read_everything(bytes: Vec<u8>) {
    let Ok(reader) = Reader::from_bytes(bytes) else {
        return;
    };
    let _ = (reader.meta(), reader.blocks().len(), reader.recovered());
    let duration = reader.duration();
    for time in [0, duration / 2, duration, u64::MAX] {
        let _ = reader.block_for(time);
        let _ = reader.keyframe_before(time);
    }
    let mut cursor = reader.cursor();
    let mut records = 0usize;
    while let Ok(Some(record)) = cursor.next() {
        let _ = (record.time, record.stream, record.payload.len());
        records += 1;
        if records > 2_000_000 {
            break;
        }
    }
    cursor.seek(duration / 3);
    let _ = cursor.next();
    let _ = reader.all_records();
    let _ = RecordStore::read_all(&reader);
    let _ = markers_in(&reader);
    for index in 0..reader.blocks().len().min(64) {
        let _ = reader.block_raw(index);
    }
}

fn env_number(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[test]
fn damaged_replay_files_never_panic_hang_or_balloon() {
    let dir = tempfile::tempdir().expect("tempdir");
    let seeds = seed_replays(dir.path());
    let parsed: Vec<Parsed> = seeds.iter().map(|file| parse(file)).collect();
    let counts: Vec<usize> = parsed.iter().map(|p| p.blocks.len()).collect();
    assert!(
        counts.iter().all(|&count| count >= 2),
        "seeds with several blocks: {counts:?}"
    );
    // Control: re-sealing without damage gives a file the reader takes whole, so the damage the
    // rounds re-seal does reach the record decoders.
    for (file, parts) in seeds.iter().zip(&parsed) {
        let count = |bytes: Vec<u8>| {
            Reader::from_bytes(bytes)
                .expect("a readable file")
                .all_records()
                .expect("whole")
                .len()
        };
        let resealed = seal(&parts.meta_json, &parts.blocks, &[]);
        assert_eq!(count(resealed), count(file.clone()));
    }
    let probe = dir.path().join("probe.aeroreplay");
    let rounds = 1500 * env_number("REPLAY_FUZZ_ROUNDS", 1).max(1) as usize;
    let seed = env_number("REPLAY_FUZZ_SEED", 0x00AE_0DEF_11E5);
    let (done, finished) = mpsc::channel();
    std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(move || {
            let mut rng = Rng(seed);
            let started = Instant::now();
            for round in 0..rounds {
                let which = rng.below(seeds.len());
                let original = &parsed[which];
                let bytes = match rng.below(5) {
                    // The metadata, re-sealed (footer kept: its own copy of the metadata wins).
                    0 => {
                        let mut json = original.meta_json.clone();
                        mutate(&mut rng, &mut json);
                        let footer = if rng.below(2) == 0 {
                            &original.footer[..]
                        } else {
                            &[]
                        };
                        seal(&json, &original.blocks, footer)
                    }
                    // A block's records, re-sealed; no footer.
                    1 | 2 => {
                        let mut blocks = original.blocks.clone();
                        let at = rng.below(blocks.len());
                        mutate(&mut rng, &mut blocks[at].1);
                        if rng.below(4) == 0 {
                            // The record count, times and stream mask too.
                            let field = [12, 16, 24, 34][rng.below(4)];
                            let value = rng.next().to_le_bytes();
                            let width = if field == 34 {
                                2
                            } else if field == 12 {
                                4
                            } else {
                                8
                            };
                            blocks[at].0[field..field + width].copy_from_slice(&value[..width]);
                        }
                        seal(&original.meta_json, &blocks, &[])
                    }
                    // The footer, as stored (its checksum stops most damage).
                    3 => {
                        let mut footer = original.footer.clone();
                        mutate(&mut rng, &mut footer);
                        let mut file = seeds[which].clone();
                        file.truncate(file.len() - original.footer.len());
                        file.extend_from_slice(&footer);
                        file
                    }
                    // Anywhere, unsealed.
                    _ => {
                        let mut file = seeds[which].clone();
                        mutate(&mut rng, &mut file);
                        file
                    }
                };
                let began = Instant::now();
                let shown = bytes.len();
                let (outcome, peak) = peak_during(|| {
                    std::panic::catch_unwind(AssertUnwindSafe(|| {
                        if round % 25 == 0 {
                            std::fs::write(&probe, &bytes).expect("write the probe");
                            let _ = read_info(&probe);
                        }
                        read_everything(bytes);
                    }))
                });
                let took = began.elapsed();
                assert!(
                    outcome.is_ok(),
                    "round {round} (seed {seed:#x}) panicked ({shown} bytes)"
                );
                assert!(
                    took < Duration::from_secs(5),
                    "round {round} (seed {seed:#x}) took {took:?}"
                );
                assert!(
                    peak < 256 * MIB,
                    "round {round} (seed {seed:#x}) held {peak} bytes"
                );
            }
            eprintln!(
                "replay files: {rounds} rounds, seed {seed:#x}, {:?}",
                started.elapsed()
            );
            let _ = done.send(());
        })
        .expect("spawn");
    match finished.recv_timeout(Duration::from_secs(600)) {
        Ok(()) => {}
        Err(mpsc::RecvTimeoutError::Timeout) => panic!("an input never returned"),
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("the fuzz worker failed"),
    }
}
