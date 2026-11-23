//! Print what an `.aeroreplay` file holds, and optionally check its packets byte for byte
//! against the `.mccap` capture that drove the session it recorded.
//!
//! ```text
//! cargo run -p mc-replay --example inspect -- <file.aeroreplay> [<capture.mccap>]
//! ```

use std::collections::BTreeMap;
use std::io::Read as _;

use mc_replay::{read_info, Reader, StreamId};

/// The clientbound Play frames of a `.mccap` capture the client applies (see
/// `examples/compression.rs`).
fn clientbound_play(path: &str) -> Vec<Vec<u8>> {
    let bytes = std::fs::read(path).expect("read the capture");
    assert_eq!(&bytes[0..8], b"MCCAP001", "a .mccap capture");
    let be32 = |at: usize| u32::from_be_bytes(bytes[at..at + 4].try_into().expect("4 bytes"));
    let compressed = bytes[9] & 1 != 0;
    let address_len = usize::from(u16::from_be_bytes([bytes[22], bytes[23]]));
    let at = 24 + address_len;
    let count = be32(at) as usize;
    let stored_len = be32(at + 8) as usize;
    let stored = &bytes[at + 12..at + 12 + stored_len];
    let payload = if compressed {
        let mut out = Vec::new();
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
        let (direction, state) = (payload[pos], payload[pos + 1]);
        let len = u32::from_be_bytes(payload[pos + 10..pos + 14].try_into().expect("4")) as usize;
        let frame = payload[pos + 14..pos + 14 + len].to_vec();
        pos += 14 + len;
        if direction == 0 && state == 3 && !matches!(frame.first(), Some(0x00 | 0x40 | 0x46)) {
            frames.push(frame);
        }
    }
    frames
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: inspect <file.aeroreplay> [<capture.mccap>]");
        std::process::exit(2);
    };
    let info = read_info(path).expect("read the header and footer");
    let reader = Reader::open(path).expect("open");
    println!("file: {path} ({} bytes)", info.size_bytes);
    println!("recovered (no footer): {}", reader.recovered());
    println!("meta: {}", reader.meta().to_json());
    println!("window: {:?}", reader.meta().window());
    let blocks = reader.blocks();
    println!(
        "blocks: {} ({} prefix, {} keyframe)",
        blocks.len(),
        blocks.iter().filter(|b| b.prefix).count(),
        blocks.iter().filter(|b| b.keyframe).count()
    );
    let mut per_stream: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut packets = Vec::new();
    let mut last_time = 0;
    let mut ordered = true;
    for record in reader.records() {
        let record = record.expect("record");
        ordered &= record.time >= last_time;
        last_time = record.time;
        let key = match record.stream {
            StreamId::Client(kind) => format!("Client({kind})"),
            other => format!("{other:?}"),
        };
        let entry = per_stream.entry(key).or_default();
        entry.0 += 1;
        entry.1 += record.payload.len();
        if record.stream == StreamId::Packet {
            packets.push(record.payload);
        }
    }
    for (stream, (count, bytes)) in &per_stream {
        println!("  {stream}: {count} records, {bytes} bytes");
    }
    println!(
        "times non-decreasing: {ordered}; last record at {:.3} s",
        last_time as f64 / 1e6
    );
    if let Some(capture) = args.get(1) {
        let expected = clientbound_play(capture);
        let same =
            packets.len() == expected.len() && packets.iter().zip(&expected).all(|(a, b)| a == b);
        let first_difference = packets.iter().zip(&expected).position(|(a, b)| a != b);
        println!(
            "against {capture}: {} packets recorded, {} applied from the capture; identical \
             byte for byte: {same} (first difference: {first_difference:?})",
            packets.len(),
            expected.len()
        );
        if !same {
            std::process::exit(1);
        }
    }
}
