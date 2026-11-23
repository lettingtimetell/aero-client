//! What leaving large packet payloads in the file costs the viewer in
//! time, against holding every record, on a real recording.
//!
//! For each arm, alternating (A B B A ...): open the file and build the store (`read_all` or
//! `read_all_in_file`), then walk every record in order reading its payload the way playback
//! does (one `BlockCache`), then walk it again from the middle (a seek's replay). Prints each
//! arm's times and the bytes its store holds. Timings need an optimised build:
//!
//! ```text
//! MW_REPLAY=<file.aeroreplay> cargo test --profile perf-ab -p mc-replay --test mw_store_timing \
//!   -- --ignored --nocapture
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use mc_replay::{BlockCache, Reader, RecordStore};

/// The viewer's threshold (`MW_IN_FILE_MIN_BYTES` in the client).
const MIN_LEN: usize = 1024;

struct Arm {
    open: Vec<Duration>,
    walk: Vec<Duration>,
    half: Vec<Duration>,
    held: usize,
}

fn median(samples: &[Duration]) -> f64 {
    let mut ms: Vec<f64> = samples.iter().map(|d| d.as_secs_f64() * 1e3).collect();
    ms.sort_by(f64::total_cmp);
    ms[ms.len() / 2]
}

/// Read every payload from `start` on, in order, as playback does; the bytes seen.
fn walk(store: &RecordStore, start: usize) -> usize {
    let mut cache = BlockCache::default();
    let mut seen = 0usize;
    for index in start..store.len() {
        let payload = store.payload_with(index, &mut cache).expect("a payload");
        seen += payload.len() + usize::from(payload.first().copied().unwrap_or(0));
    }
    seen
}

fn run(path: &str, in_file: bool, arm: &mut Arm) -> usize {
    let started = Instant::now();
    let reader = Reader::open(path).expect("open the recording");
    let store = if in_file {
        RecordStore::read_all_in_file(reader, MIN_LEN).expect("records")
    } else {
        let store = RecordStore::read_all(&reader).expect("records");
        drop(reader);
        store
    };
    arm.open.push(started.elapsed());
    arm.held = store.heap_bytes();
    let started = Instant::now();
    let seen = black_box(walk(&store, 0));
    arm.walk.push(started.elapsed());
    let started = Instant::now();
    black_box(walk(&store, store.len() / 2));
    arm.half.push(started.elapsed());
    seen
}

#[test]
#[ignore = "needs a recording (MW_REPLAY) and an optimised build"]
fn in_file_payloads_cost_in_time() {
    let Ok(path) = std::env::var("MW_REPLAY") else {
        eprintln!("skip: set MW_REPLAY to a .aeroreplay file");
        return;
    };
    let rounds: usize = std::env::var("MW_ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4);
    let mut held = Arm {
        open: Vec::new(),
        walk: Vec::new(),
        half: Vec::new(),
        held: 0,
    };
    let mut file = Arm {
        open: Vec::new(),
        walk: Vec::new(),
        half: Vec::new(),
        held: 0,
    };
    // A warm-up of each, not counted.
    let mut scratch = Arm {
        open: Vec::new(),
        walk: Vec::new(),
        half: Vec::new(),
        held: 0,
    };
    let held_seen = run(&path, false, &mut scratch);
    let file_seen = run(&path, true, &mut scratch);
    assert_eq!(held_seen, file_seen, "both arms read the same bytes");
    for round in 0..rounds {
        // A B B A: neither arm always goes first.
        let order = if round % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        for in_file in order {
            let arm = if in_file { &mut file } else { &mut held };
            run(&path, in_file, arm);
        }
    }
    println!(
        "{:<10} {:>12} {:>10} {:>10} {:>12}",
        "arm", "store MiB", "open ms", "walk ms", "half walk ms"
    );
    for (label, arm) in [("held", &held), ("in file", &file)] {
        println!(
            "{label:<10} {:>12.1} {:>10.1} {:>10.1} {:>12.1}",
            arm.held as f64 / 1048576.0,
            median(&arm.open),
            median(&arm.walk),
            median(&arm.half),
        );
    }
}
