//! How well replay files compress: zstd (what the writer uses, at several levels) against
//! deflate (zlib's algorithm, what `.mccap` captures use) on the clientbound Play packets of real
//! captures, with the writer's own block cutting (2 s or 1 MiB).
//!
//! ```text
//! cargo run -p mc-replay --example compression -- <capture.mccap>...
//! ```

use std::io::{Read as _, Write as _};
use std::time::{Duration, Instant};

use mc_replay::{Reader, ReplayMeta, StreamId, Writer, WriterOptions};

/// The clientbound Play frames of a `.mccap` capture (`mc-parity`'s container, read here
/// directly so this example does not pull the whole client in): `(micros, payload)`, minus the
/// three packets the network task keeps (KeepAlive, Disconnect, SetCompression).
fn clientbound_play(path: &str) -> Vec<(u64, Vec<u8>)> {
    let bytes = std::fs::read(path).expect("read the capture");
    assert_eq!(&bytes[0..8], b"MCCAP001", "a .mccap capture");
    let be32 = |at: usize| u32::from_be_bytes(bytes[at..at + 4].try_into().expect("4 bytes"));
    let compressed = bytes[9] & 1 != 0;
    let address_len = usize::from(u16::from_be_bytes([bytes[22], bytes[23]]));
    let at = 24 + address_len;
    let count = be32(at) as usize;
    let raw_len = be32(at + 4) as usize;
    let stored_len = be32(at + 8) as usize;
    let stored = &bytes[at + 12..at + 12 + stored_len];
    let payload = if compressed {
        let mut out = Vec::with_capacity(raw_len);
        flate2::read::ZlibDecoder::new(stored)
            .read_to_end(&mut out)
            .expect("inflate");
        out
    } else {
        stored.to_vec()
    };
    let mut frames = Vec::new();
    let mut pos = 0;
    for _ in 0..count {
        let direction = payload[pos];
        let state = payload[pos + 1];
        let micros = u64::from_be_bytes(payload[pos + 2..pos + 10].try_into().expect("8 bytes"));
        let len = u32::from_be_bytes(payload[pos + 10..pos + 14].try_into().expect("4")) as usize;
        let frame = payload[pos + 14..pos + 14 + len].to_vec();
        pos += 14 + len;
        // Clientbound (0), Play (3); every Play id is below 0x80, so its first byte.
        if direction == 0 && state == 3 && !matches!(frame.first(), Some(0x00 | 0x40 | 0x46)) {
            frames.push((micros, frame));
        }
    }
    frames
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: compression <capture.mccap>...");
        std::process::exit(2);
    }
    let dir = std::env::temp_dir().join(format!("mc-replay-compression-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    for path in &paths {
        // What a recording holds: the clientbound Play packets the game applies.
        let frames = clientbound_play(path);
        let base = frames.first().map_or(0, |f| f.0);
        let span = frames.last().map_or(0, |f| f.0) - base;
        let raw: usize = frames.iter().map(|f| f.1.len()).sum();
        let minutes = span as f64 / 60e6;
        let input_size = std::fs::metadata(path).map_or(0, |m| m.len());
        println!("\n## {path}");
        println!(
            "{} packets over {:.1} s; {:.2} MB of packet bytes ({:.2} MB/min raw); the .mccap \
             (zlib, both directions) is {:.2} MB",
            frames.len(),
            span as f64 / 1e6,
            raw as f64 / 1e6,
            raw as f64 / 1e6 / minutes,
            input_size as f64 / 1e6
        );
        println!("| codec | file MB | MB/min | ratio | compress time |");
        println!("|---|---:|---:|---:|---:|");
        let mut reference = None;
        for level in [1, 3, 6, 9, 12, 15, 19] {
            let file = dir.join(format!("level{level}.aeroreplay"));
            let _ = std::fs::remove_file(&file);
            let options = WriterOptions {
                level,
                sync_interval: None,
                ..WriterOptions::default()
            };
            let started = Instant::now();
            let mut writer = Writer::create(&file, &ReplayMeta::default(), options).expect("w");
            let clock = Instant::now();
            let mut push_time = Duration::ZERO;
            for (micros, payload) in &frames {
                let time = micros - base;
                let at = Instant::now();
                writer.push(time, StreamId::Packet, payload);
                // The game thread ticks the writer once per frame with the frame clock.
                writer.tick(clock + Duration::from_micros(time));
                push_time += at.elapsed();
            }
            if level == 9 {
                println!(
                    "(game thread: {:.0} ns per packet pushed, {:.2} ms for the whole scene)",
                    push_time.as_nanos() as f64 / frames.len() as f64,
                    push_time.as_secs_f64() * 1e3
                );
            }
            let finished = writer
                .finish(ReplayMeta::default())
                .wait(Duration::from_secs(600))
                .expect("finish");
            let took = started.elapsed();
            let size = finished.bytes as f64;
            println!(
                "| zstd {level} | {:.3} | {:.3} | {:.1}x | {:.0} ms |",
                size / 1e6,
                size / 1e6 / minutes,
                raw as f64 / size,
                took.as_secs_f64() * 1e3
            );
            if level == 9 {
                reference = Some(file);
            }
        }
        // Deflate on exactly the same blocks (decompressed from the level-9 file).
        let reference = reference.expect("level 9 written");
        let reader = Reader::open(&reference).expect("read back");
        let blocks: Vec<Vec<u8>> = (0..reader.blocks().len())
            .map(|i| reader.block_raw(i).expect("block"))
            .collect();
        let overhead = std::fs::metadata(&reference).expect("size").len()
            - reader
                .blocks()
                .iter()
                .map(|b| u64::from(b.stored_len))
                .sum::<u64>();
        for level in [6u32, 9] {
            let started = Instant::now();
            let mut total = overhead as usize;
            for block in &blocks {
                let mut encoder =
                    flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(level));
                encoder.write_all(block).expect("deflate");
                total += encoder.finish().expect("deflate").len();
            }
            let took = started.elapsed();
            println!(
                "| deflate {level} (same blocks) | {:.3} | {:.3} | {:.1}x | {:.0} ms |",
                total as f64 / 1e6,
                total as f64 / 1e6 / minutes,
                raw as f64 / total as f64,
                took.as_secs_f64() * 1e3
            );
        }
        // And deflate over the whole stream at once (no random access), as a bound.
        let mut whole = Vec::with_capacity(raw);
        for block in &blocks {
            whole.extend_from_slice(block);
        }
        let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(9));
        encoder.write_all(&whole).expect("deflate");
        let one = encoder.finish().expect("deflate").len();
        println!(
            "| deflate 9 (one stream, no seeking) | {:.3} | {:.3} | {:.1}x | |",
            one as f64 / 1e6,
            one as f64 / 1e6 / minutes,
            raw as f64 / one as f64
        );
        println!("({} blocks)", blocks.len());
        // Every record reads back.
        let back: Vec<_> = reader
            .records()
            .map(|r| r.expect("record").payload)
            .collect();
        assert_eq!(back.len(), frames.len());
        assert!(back.iter().zip(&frames).all(|(a, b)| a == &b.1));
    }
    let _ = std::fs::remove_dir_all(&dir);
}
