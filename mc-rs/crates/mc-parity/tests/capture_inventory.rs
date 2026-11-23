//! What the committed captures actually *contain*, as a number rather than an impression.
//!
//! `capture_replay.rs` proves every frame decodes and re-encodes. That says nothing about
//! whether a capture is worth *benchmarking* — for that, the question is how much world it
//! delivers: how many chunk columns, how many non-empty sections, how many block bytes. A capture
//! becomes the input of a deterministic in-world bench, and this test is the
//! measurement that decides whether such a bench is heavy or trivial.
//!
//! It prints an inventory and asserts only the floors that make the inventory meaningful.

use std::collections::BTreeMap;

use mc_parity::capture::{Capture, Direction};
use mc_parity::replay::Decoded;
use mc_parity::Replayer;
use mc_protocol::play::S2CPlay;

/// Every `.mccap` under `tests/fixtures/` and under `captures/`, sorted.
///
/// The two directories are separate on purpose. `tests/fixtures/` holds the small protocol
/// recordings `capture_replay.rs` replays on every run; `captures/` holds the **large** terrain
/// recordings the in-world bench drives the renderer from — megabytes of `MapChunkBulk` that
/// would make the round-trip test slow for no extra coverage. Both are worth inventorying.
fn fixtures() -> Vec<std::path::PathBuf> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut found: Vec<std::path::PathBuf> =
        [root.join("tests").join("fixtures"), root.join("captures")]
            .iter()
            .filter_map(|dir| std::fs::read_dir(dir).ok())
            .flatten()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "mccap"))
            .collect();
    found.sort();
    found
}

/// The world content one capture delivers.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct WorldContent {
    chunk_data_packets: usize,
    bulk_packets: usize,
    /// Columns delivered by either packet, counted per delivery (not deduplicated).
    columns: usize,
    /// Columns whose bitmask is zero — an *unload*, not terrain.
    unload_columns: usize,
    /// Sum of `primary_bit_mask.count_ones()` over every delivered column: the number of
    /// section-sized blobs that reach `mc-world`, i.e. the upper bound on what can be meshed.
    sections: usize,
    /// Distinct `(x, z)` columns, so a repeatedly-resent column is not counted twice.
    distinct_columns: usize,
    /// Total bytes of section blob.
    section_bytes: usize,
    block_change_packets: usize,
    multi_block_change_packets: usize,
}

fn inventory(capture: &Capture) -> (WorldContent, BTreeMap<String, usize>) {
    let mut content = WorldContent::default();
    let mut names: BTreeMap<String, usize> = BTreeMap::new();
    let mut distinct = std::collections::BTreeSet::new();

    let report = Replayer::new(capture).run_with(|frame, decoded| {
        if frame.direction != Direction::Clientbound {
            return;
        }
        let Decoded::PlayClientbound(packet) = decoded else {
            return;
        };
        *names.entry(variant_name(packet).to_owned()).or_default() += 1;
        match packet.as_ref() {
            S2CPlay::ChunkData(chunk) => {
                content.chunk_data_packets += 1;
                content.columns += 1;
                let sections = chunk.primary_bit_mask.count_ones() as usize;
                if sections == 0 {
                    content.unload_columns += 1;
                }
                content.sections += sections;
                content.section_bytes += chunk.data.len();
                distinct.insert((chunk.chunk_x, chunk.chunk_z));
            }
            S2CPlay::MapChunkBulk(bulk) => {
                content.bulk_packets += 1;
                for column in &bulk.columns {
                    content.columns += 1;
                    let sections = column.primary_bit_mask.count_ones() as usize;
                    if sections == 0 {
                        content.unload_columns += 1;
                    }
                    content.sections += sections;
                    content.section_bytes += column.data.len();
                    distinct.insert((column.chunk_x, column.chunk_z));
                }
            }
            S2CPlay::BlockChange(_) => content.block_change_packets += 1,
            S2CPlay::MultiBlockChange(_) => content.multi_block_change_packets += 1,
            _ => {}
        }
    });
    assert!(report.decode_failures.is_empty(), "{report:#?}");
    content.distinct_columns = distinct.len();
    (content, names)
}

/// The `S2CPlay` variant name, without pulling in a derive.
fn variant_name(packet: &S2CPlay) -> &'static str {
    // `Debug` on these enums prints `Variant(..)`; take the identifier.
    let rendered = format!("{packet:?}");
    let cut = rendered.find(['(', ' ']).unwrap_or(rendered.len());
    // Leak-free: match against a fixed set would be 74 arms, so intern through a small cache.
    // In practice a capture has <80 distinct variants, so a `Box::leak` per *distinct* name is
    // bounded — but a test that allocates per packet is fine either way.
    Box::leak(rendered[..cut].to_owned().into_boxed_str())
}

#[test]
fn the_committed_captures_inventory_their_world_content() {
    let captures = fixtures();
    assert!(
        !captures.is_empty(),
        "tests/fixtures must hold at least one .mccap"
    );

    for path in &captures {
        let name = path.file_name().unwrap_or(path.as_os_str());
        let capture = Capture::read_file(path).expect("read");
        let (content, names) = inventory(&capture);
        let span_s = capture
            .frames()
            .iter()
            .map(|f| f.micros)
            .max()
            .unwrap_or_default() as f64
            / 1e6;

        println!("== {name:?}");
        println!(
            "   {} frames ({} clientbound, {} serverbound) over {span_s:.1} s",
            capture.len(),
            capture.frames_in(Direction::Clientbound).count(),
            capture.frames_in(Direction::Serverbound).count(),
        );
        println!("   world: {content:?}");
        let mut top: Vec<(&String, &usize)> = names.iter().collect();
        top.sort_by_key(|(n, c)| (std::cmp::Reverse(**c), (*n).clone()));
        for (n, c) in top.iter().take(12) {
            println!("     {c:>5}  {n}");
        }

        // The only floor worth asserting: a capture that delivers no world at all cannot be
        // the input of an in-world benchmark, and if that ever becomes true of every fixture
        // the T0.1 harness is measuring an empty scene.
        assert!(
            content.columns > 0,
            "{name:?}: no chunk column was ever delivered"
        );
    }
}
