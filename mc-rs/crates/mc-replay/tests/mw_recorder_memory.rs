//! The replay writer's zstd context while recording — how much memory it
//! holds, and what a smaller one costs in file size and time.
//!
//! The writer's worker keeps one `zstd::bulk::Compressor` for the life of a recording and feeds
//! it blocks of about 1 MiB or 2 s of records. This cuts a real recording's records into blocks
//! the same way, compresses every block with each candidate setting, checks each block
//! decompresses to the same bytes, and prints the context's size (`ZSTD_sizeof_CCtx`, which the
//! Rust allocator never sees: zstd allocates with C `malloc`), the total compressed size and the
//! time.
//!
//! Ignored by default (it needs a recording). Run:
//!
//! ```text
//! MW_REPLAY=<file.aeroreplay> cargo test -p mc-replay --test mw_recorder_memory -- --ignored --nocapture
//! ```

use std::time::{Duration, Instant};

use mc_replay::{Reader, Record, StreamId};

/// `WriterOptions::default().block_bytes`.
const BLOCK_BYTES: usize = 1 << 20;
/// `WriterOptions::default().block_interval`, in replay microseconds.
const BLOCK_INTERVAL: u64 = 2_000_000;
/// The writer's per-record allowance (`RECORD_OVERHEAD`).
const RECORD_OVERHEAD: usize = 22;

fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// Cut records into raw blocks as the writer does: a block is sealed before a record that would
/// overflow its buffer, once it reaches `BLOCK_BYTES`, or once it spans `BLOCK_INTERVAL`. Records
/// are laid out like the writer's (time delta, stream tag, length, payload).
fn cut_blocks(records: &[Record]) -> Vec<Vec<u8>> {
    let mut blocks = Vec::new();
    let mut raw: Vec<u8> = Vec::with_capacity(BLOCK_BYTES + 4096);
    let mut first = 0u64;
    let mut prev = 0u64;
    for record in records {
        let needed = record.payload.len() + RECORD_OVERHEAD;
        let span = record.time.saturating_sub(first);
        if !raw.is_empty() && (raw.len() + needed > BLOCK_BYTES + 4096 || span >= BLOCK_INTERVAL) {
            blocks.push(std::mem::replace(
                &mut raw,
                Vec::with_capacity(BLOCK_BYTES + 4096),
            ));
        }
        if raw.is_empty() {
            first = record.time;
            prev = record.time;
        }
        put_varint(&mut raw, record.time.saturating_sub(prev));
        match record.stream {
            StreamId::Packet => raw.extend_from_slice(&[0, 0]),
            StreamId::Client(kind) => raw.extend_from_slice(&[1, kind]),
            _ => raw.extend_from_slice(&[2, 0]),
        }
        put_varint(&mut raw, record.payload.len() as u64);
        raw.extend_from_slice(&record.payload);
        prev = record.time;
        if raw.len() >= BLOCK_BYTES {
            blocks.push(std::mem::replace(
                &mut raw,
                Vec::with_capacity(BLOCK_BYTES + 4096),
            ));
        }
    }
    if !raw.is_empty() {
        blocks.push(raw);
    }
    blocks
}

struct Outcome {
    label: &'static str,
    stored: usize,
    time: Duration,
    peak_cctx: usize,
}

fn run(label: &'static str, params: &[zstd::zstd_safe::CParameter], blocks: &[Vec<u8>]) -> Outcome {
    let mut compressor = zstd::bulk::Compressor::new(9).expect("a zstd context");
    compressor.include_checksum(true).expect("checksum flag");
    for param in params {
        compressor.set_parameter(*param).expect("a valid parameter");
    }
    let mut out = Vec::new();
    let mut stored = 0usize;
    let mut peak_cctx = compressor.context_mut().sizeof();
    let mut time = Duration::ZERO;
    for block in blocks {
        out.clear();
        out.reserve(zstd::zstd_safe::compress_bound(block.len()));
        let started = Instant::now();
        compressor
            .compress_to_buffer(block.as_slice(), &mut out)
            .expect("compress a block");
        time += started.elapsed();
        stored += out.len().min(block.len());
        peak_cctx = peak_cctx.max(compressor.context_mut().sizeof());
        let back = zstd::bulk::decompress(&out, block.len()).expect("decompress a block");
        assert_eq!(&back, block, "{label}: a block must round-trip");
    }
    Outcome {
        label,
        stored,
        time,
        peak_cctx,
    }
}

/// What a recording's record bytes are made of: per stream, and per packet id for packets, the
/// count and payload bytes, largest first.
#[test]
#[ignore = "needs a recording (MW_REPLAY)"]
fn record_bytes_by_kind() {
    let Ok(path) = std::env::var("MW_REPLAY") else {
        eprintln!("skip: set MW_REPLAY to a .aeroreplay file");
        return;
    };
    let reader = Reader::open(&path).expect("open the recording");
    let mut kinds: std::collections::BTreeMap<String, (u64, u64, u64)> = Default::default();
    let mut total = 0u64;
    let mut cursor = reader.cursor();
    while let Some(record) = cursor.next().expect("a record") {
        let key = match record.stream {
            StreamId::Packet => {
                let id = record.payload.first().copied().unwrap_or(0xFF);
                format!("packet 0x{id:02X}")
            }
            StreamId::Client(kind) => format!("client {kind}"),
            other => format!("{other:?}"),
        };
        let len = record.payload.len() as u64;
        let entry = kinds.entry(key).or_default();
        entry.0 += 1;
        entry.1 += len;
        entry.2 = entry.2.max(len);
        total += len;
    }
    let mut rows: Vec<_> = kinds.into_iter().collect();
    rows.sort_by_key(|(_, (_, bytes, _))| std::cmp::Reverse(*bytes));
    println!(
        "{:<14} {:>9} {:>11} {:>7} {:>9}",
        "kind", "count", "bytes", "share", "largest"
    );
    for (key, (count, bytes, largest)) in rows.iter().take(25) {
        println!(
            "{key:<14} {count:>9} {bytes:>11} {:>6.1}% {largest:>9}",
            100.0 * *bytes as f64 / total.max(1) as f64
        );
    }
    println!("total payload {:.1} MiB", total as f64 / 1048576.0);
}

#[test]
#[ignore = "needs a recording (MW_REPLAY)"]
fn writer_context_size_and_ratio() {
    let Ok(path) = std::env::var("MW_REPLAY") else {
        eprintln!("skip: set MW_REPLAY to a .aeroreplay file");
        return;
    };
    let reader = Reader::open(&path).expect("open the recording");
    let records: Vec<Record> = reader
        .records()
        .collect::<Result<_, _>>()
        .expect("read every record");
    let blocks = cut_blocks(&records);
    let raw: usize = blocks.iter().map(Vec::len).sum();
    let largest = blocks.iter().map(Vec::len).max().unwrap_or(0);
    println!(
        "{} records, {} blocks, {:.1} MiB raw, largest block {} KiB, mean {} KiB",
        records.len(),
        blocks.len(),
        raw as f64 / 1048576.0,
        largest / 1024,
        raw / blocks.len().max(1) / 1024
    );

    use zstd::zstd_safe::CParameter::{HashLog, WindowLog};
    let candidates: Vec<(&'static str, Vec<zstd::zstd_safe::CParameter>)> = vec![
        ("level 9 (current)", vec![]),
        ("level 9, hash 20", vec![HashLog(20)]),
        ("level 9, hash 19", vec![HashLog(19)]),
        ("level 9, hash 18", vec![HashLog(18)]),
        ("level 9, hash 17", vec![HashLog(17)]),
        (
            "level 9, hash 18, window 19",
            vec![HashLog(18), WindowLog(19)],
        ),
        (
            "level 9, hash 17, window 19",
            vec![HashLog(17), WindowLog(19)],
        ),
    ];
    let mut outcomes = Vec::new();
    // Two passes, alternating, so a warm-up or a background burst does not favour one setting.
    for pass in 0..2 {
        for (label, params) in &candidates {
            let outcome = run(label, params, &blocks);
            if pass == 1 {
                outcomes.push(outcome);
            }
        }
    }
    let base = &outcomes[0];
    println!(
        "{:<30} {:>11} {:>9} {:>10} {:>9} {:>11}",
        "setting", "stored KiB", "vs now", "time ms", "vs now", "context KiB"
    );
    for outcome in &outcomes {
        println!(
            "{:<30} {:>11} {:>+8.2}% {:>10.1} {:>+8.1}% {:>11}",
            outcome.label,
            outcome.stored / 1024,
            100.0 * (outcome.stored as f64 / base.stored as f64 - 1.0),
            outcome.time.as_secs_f64() * 1e3,
            100.0 * (outcome.time.as_secs_f64() / base.time.as_secs_f64() - 1.0),
            outcome.peak_cctx / 1024,
        );
    }
}
