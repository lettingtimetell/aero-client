//! Where does the ~0.2 MB per reconnect go?
//!
//! `leak::repeated_connect_disconnect_cycles_leak_nothing` measures the process **resident set**
//! across connect/disconnect cycles and has been flaky since Nov 7-11 at ~200 KB per cycle. A
//! resident set mixes three things: memory the program still holds (a real leak), memory the
//! allocator keeps for reuse, and GPU-driver memory. This probe runs the same cycles against the
//! same loopback double with a counting global allocator and reports the **live heap** (bytes
//! allocated minus bytes freed, every thread) after each cycle, next to the resident set. A real
//! leak grows the live heap; allocator retention and driver memory do not.
//!
//! Ignored by default (GPU + vanilla jar). Run:
//!
//! ```text
//! MC_JARS_DIR=<jars> cargo test -p mc-client --test nw_f7_reconnect_heap -- --ignored --nocapture
//! ```

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::Duration;

use common::{config, gpu, pump, pump_until, serve_join};
use mc_client::app::crash::process_memory_bytes;
use mc_client::app::state::GameState;
use mc_client::app::Client;
use mc_math::ChunkPos;
use mc_physics::state::MoveInput;
use mc_render::{Camera, PhaseList};
use tokio::net::TcpListener;

struct Counting;

static LIVE: AtomicI64 = AtomicI64::new(0);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
/// Live bytes by allocation size class (power of two buckets), to name what grows.
static BY_CLASS: [AtomicI64; 40] = [const { AtomicI64::new(0) }; 40];

fn class(size: usize) -> usize {
    ((usize::BITS - size.max(1).leading_zeros()) as usize).min(39)
}

/// Book `size` live bytes in (positive `sign`) or out of (negative) the totals.
fn track(size: usize, sign: i64) {
    LIVE.fetch_add(sign * size as i64, Ordering::Relaxed);
    BY_CLASS[class(size)].fetch_add(sign * size as i64, Ordering::Relaxed);
}

// SAFETY: forwards every call to `System` unchanged; the counters are plain atomics.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: same contract as the caller's.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            track(layout.size(), 1);
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: same contract as the caller's.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            track(layout.size(), 1);
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        track(layout.size(), -1);
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: same contract as the caller's.
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            track(layout.size(), -1);
            track(new_size, 1);
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        new
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const WIDTH: u32 = 256;
const HEIGHT: u32 = 192;

fn chunks() -> Vec<ChunkPos> {
    let mut out = Vec::new();
    for x in -1..=1 {
        for z in -1..=1 {
            out.push(ChunkPos { x, z });
        }
    }
    out
}

fn classes() -> [i64; 40] {
    std::array::from_fn(|index| BY_CLASS[index].load(Ordering::Relaxed))
}

#[test]
#[ignore = "GPU + vanilla jar; a measurement, not a gate"]
fn live_heap_per_reconnect_cycle() {
    let cycles: usize = std::env::var("NW_F7_CYCLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(15);
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    // `NW_F7_KEEP_CHAT=0`: switch "Keep chat between servers" off, so chat kept between
    // connections is not part of the measurement.
    if std::env::var("NW_F7_KEEP_CHAT").is_ok_and(|value| value == "0") {
        std::fs::write(
            dir.path().join(mc_mods::SETTINGS_FILE_NAME),
            r#"{"format":1,"mods":{"chat":{"enabled":true,"options":{"keep_chat":false}}}}"#,
        )
        .expect("write the scratch mod settings");
    }
    let Some(config) = config(dir.path(), "Dev") else {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    };
    let mut client = Client::new(config, &mut |_, _| {}).expect("start");
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, WIDTH, HEIGHT);
    let mut camera = Camera::default();
    let listener = client
        .runtime()
        .handle()
        .block_on(TcpListener::bind("127.0.0.1:0"))
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let server = client.runtime().handle().spawn(async move {
        let chunks = chunks();
        for cycle in 0..cycles {
            let mut peer = serve_join(&listener, 40 + cycle as i32, &chunks, 0xFFFF).await;
            peer.drain_until_closed().await;
        }
    });

    let mut previous = (LIVE.load(Ordering::Relaxed), classes());
    println!("cycle  live_heap_kb  delta_kb  small_live_kb  resident_mb  allocs");
    for cycle in 0..cycles {
        assert!(matches!(client.machine.state(), GameState::Menu));
        client.connect("127.0.0.1", port);
        pump_until(
            &mut client,
            &mut gpu,
            &mut phases,
            &mut camera,
            &MoveInput::default(),
            Duration::from_secs(30),
            "the world to load",
            |client| {
                client.machine.state().is_in_world()
                    && client
                        .session()
                        .is_some_and(|session| session.world().loaded_chunk_count() == 9)
            },
        );
        for _ in 0..12 {
            pump(
                &mut client,
                &mut gpu,
                &mut phases,
                &mut camera,
                &MoveInput::default(),
            );
        }
        client.disconnect();
        for _ in 0..3 {
            pump(
                &mut client,
                &mut gpu,
                &mut phases,
                &mut camera,
                &MoveInput::default(),
            );
        }
        // Let background threads (mesh workers, the connection task) finish their drops.
        std::thread::sleep(Duration::from_millis(200));
        for _ in 0..3 {
            pump(
                &mut client,
                &mut gpu,
                &mut phases,
                &mut camera,
                &MoveInput::default(),
            );
        }
        let live = LIVE.load(Ordering::Relaxed);
        let now = classes();
        // Live bytes in allocations of at most 64 KiB: the classes a slowly growing collection
        // lives in, without the multi-megabyte buffers whose free timing swings the total.
        let small: i64 = now[..=16].iter().sum();
        println!(
            "{cycle:5}  {:12.1}  {:8.1}  {:13.1}  {:11.1}  {}",
            live as f64 / 1024.0,
            (live - previous.0) as f64 / 1024.0,
            small as f64 / 1024.0,
            process_memory_bytes().unwrap_or(0) as f64 / 1_048_576.0,
            ALLOCS.load(Ordering::Relaxed),
        );
        if cycle >= 3 {
            let mut grown: Vec<(usize, i64)> = now
                .iter()
                .zip(previous.1.iter())
                .enumerate()
                .map(|(index, (a, b))| (index, a - b))
                .filter(|&(_, delta)| delta != 0)
                .collect();
            grown.sort_by_key(|&(_, delta)| -delta.abs());
            let top: Vec<String> = grown
                .iter()
                .take(4)
                .map(|(index, delta)| format!("<=2^{index}: {:+.1} KB", *delta as f64 / 1024.0))
                .collect();
            println!("        by size class: {}", top.join(", "));
        }
        previous = (live, now);
    }
    client
        .runtime()
        .handle()
        .block_on(server)
        .expect("the scripted peer must serve every cycle");
}
