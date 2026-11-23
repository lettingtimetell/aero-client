//! Memory-leak check across repeated connect/disconnect cycles.
//!
//! java: `Minecraft.loadWorld(null)` (`:2354`) is the only thing standing between a session and
//! a leak — it drops the world, the entity list, the chunk renders and the sound handles, and
//! then calls `System.gc()`. Rust has no GC to lean on, so the equivalent question is sharper:
//! **does everything a session allocated actually get dropped when the session does?**
//!
//! # What this measures, and what that is worth
//!
//! Two independent things, because either alone is weak:
//!
//! 1. **Structural**: every collection the client owns per session — the world's chunks, the ECS
//!    arena, the tab list, and the set of chunk sections holding GPU buffers — must be back at
//!    its baseline after each cycle, exactly, not approximately (the chat log, which Aero keeps
//!    between servers, must stay empty: the loopback server sends no chat and a join adds no
//!    line). This is the
//!    assertion that would fail *first* and *loudest* on a real leak, and it does not depend on
//!    the allocator's mood.
//! 2. **Live heap**: every byte allocated minus every byte freed, on every thread, counted by
//!    this test binary's global allocator, sampled after a warm-up and again after the measured
//!    cycles. This is the one that catches a leak the structural check has no name for — a `Vec`
//!    grown inside a subsystem, a session kept alive by a stray handle.
//!
//! This half used to gate on the process **resident set**, and
//! the test was flaky (~0.2 MB per cycle, sometimes far more). The probe
//! (`tests/nw_f7_reconnect_heap.rs`) ran the same cycles with a counting allocator: the live heap
//! is flat (the only steady growth then, ~16 KB per cycle, was a chat line each join added, capped
//! by "History length"; joins add no chat line any more), while the resident set of the very same
//! binary grew
//! anywhere from 0.1 to 5 MB per cycle between runs. A working set mixes in memory the system
//! allocator keeps for reuse and GPU-driver memory, so it was measuring the allocator's mood. The
//! resident set is still printed; the gate is the live heap, which is exact. The two tests share a
//! lock so the control's deliberate 20 MB never lands inside the other test's window. See
//! `tests/common/mod.rs` for what the server is (a loopback double, not a real server).
//!
//! Where the "~0.1-0.2 MB per reconnect" went, named block by block
//! with the leak harness's allocation-site tracer (`tests/ml_leak_harness.rs`, `ML_TRACE=1`):
//! bounded structures filling during the first ~10 cycles (the glyph-run cache as chat lines
//! move, UI vertex buffers growing to their working size, the title screen's per-visit uploads,
//! one copy each) and, at the time, one chat line per join (joins add none any more). None of it
//! is a session surviving its session. The gate now waits out
//! that warm-up and adds two sharp per-cycle limits (live blocks, and bytes in small blocks),
//! which a leaked world exceeds by two orders of magnitude while a late-freed multi-megabyte
//! buffer cannot move them; the 12 MB total stays as a backstop.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use common::{config, gpu, pump, pump_until, serve_join};
use mc_client::app::crash::process_memory_bytes;
use mc_client::app::state::GameState;
use mc_client::app::Client;
use mc_math::ChunkPos;
use mc_physics::state::MoveInput;
use mc_render::{Camera, PhaseList};
use tokio::net::TcpListener;

/// Counts live heap bytes and blocks (allocated minus freed, all threads) and forwards to
/// `System`. Also the live bytes in blocks under [`SMALL_BLOCK`].
struct LiveHeap;

static LIVE_BYTES: AtomicI64 = AtomicI64::new(0);
static LIVE_BLOCKS: AtomicI64 = AtomicI64::new(0);
static SMALL_LIVE_BYTES: AtomicI64 = AtomicI64::new(0);

/// Blocks below this size are where a session's objects live (entities, chunk sections, map
/// entries, strings). The multi-megabyte buffers above it (atlases, vertex and staging buffers)
/// are few and are freed a frame or two after a sample now and then, which is what made a
/// byte-total gate swing; they are still counted as blocks.
const SMALL_BLOCK: usize = 64 * 1024;

fn book(size: usize, sign: i64) {
    LIVE_BYTES.fetch_add(sign * size as i64, Ordering::Relaxed);
    LIVE_BLOCKS.fetch_add(sign, Ordering::Relaxed);
    if size < SMALL_BLOCK {
        SMALL_LIVE_BYTES.fetch_add(sign * size as i64, Ordering::Relaxed);
    }
}

// SAFETY: every call forwards to `System` unchanged; the counters are plain atomics.
unsafe impl GlobalAlloc for LiveHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: same contract as the caller's.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            book(layout.size(), 1);
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: same contract as the caller's.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            book(layout.size(), 1);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        book(layout.size(), -1);
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: same contract as the caller's.
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            book(layout.size(), -1);
            book(new_size, 1);
        }
        new
    }
}

#[global_allocator]
static GLOBAL: LiveHeap = LiveHeap;

fn live_heap_bytes() -> i64 {
    LIVE_BYTES.load(Ordering::Relaxed)
}

/// (live blocks, live bytes in blocks under [`SMALL_BLOCK`]).
fn live_blocks_and_small_bytes() -> (i64, i64) {
    (
        LIVE_BLOCKS.load(Ordering::Relaxed),
        SMALL_LIVE_BYTES.load(Ordering::Relaxed),
    )
}

/// Held by both tests: each one's measurement window must see only its own allocations.
static SERIAL: Mutex<()> = Mutex::new(());

const WIDTH: u32 = 256;
const HEIGHT: u32 = 192;

/// Cycles run before the first memory sample, so one-time costs — pipeline compilation, the
/// tokio worker stacks, the allocator's first big arenas — are already paid.
///
/// 3 -> 10. The leak harness (`tests/ml_leak_harness.rs`,
/// `ML_TRACE=1`) named every block the early cycles keep: the glyph-run cache filling as the
/// chat lines move, UI vertex buffers growing to their working capacity, the title screen's
/// per-visit uploads (one copy each, replaced every visit). All of them are bounded and all of
/// them settle by the tenth cycle; measured before that, they read as "~0.1-0.2 MB per
/// reconnect" without being a leak.
const WARMUP_CYCLES: usize = 10;
/// Cycles measured.
const MEASURED_CYCLES: usize = 12;

/// A 3×3 block of chunk columns, each sent with **all sixteen** sections present.
///
/// That is 9 × 16 × (8192 + 2048 + 2048) + 9 × 256 ≈ **1.7 MB of `ChunkSection` storage per
/// cycle**, which is the point: if a session's world survives its session, twelve cycles leak
/// over 20 MB and the resident-set check cannot miss it. Only the lowest section carries blocks,
/// so the mesher stays cheap.
const SECTION_MASK: u16 = 0xFFFF;

fn chunks() -> Vec<ChunkPos> {
    let mut out = Vec::new();
    for x in -1..=1 {
        for z in -1..=1 {
            out.push(ChunkPos { x, z });
        }
    }
    out
}

/// Bytes of live-heap growth permitted across [`MEASURED_CYCLES`].
///
/// Chosen against what a leak would cost rather than against what looks tidy: leaking one
/// session's world per cycle is ≥ 1.7 MB × 12 ≈ 20 MB, so a 12 MB ceiling catches it. The live
/// heap has no allocator or driver noise, only multi-megabyte buffers that are freed a frame or
/// two later than a sample (a few MB either way, measured).
const MAX_GROWTH_BYTES: u64 = 12 * 1024 * 1024;

/// Live blocks a measured cycle may add, on average. A session's world alone is hundreds of
/// blocks (144 chunk sections, its entities, maps, strings), so a leaked session fails this many
/// times over (the control below reads 144 a cycle for the sections alone). Measured after the
/// warm-up at 8-15 blocks a cycle (three runs, when each join still added a chat line, so this
/// limit has more room now).
const MAX_BLOCKS_PER_CYCLE: i64 = 40;

/// Bytes in blocks under [`SMALL_BLOCK`] a measured cycle may add, on average: measured after
/// the warm-up at 11-14 KB a cycle (three runs). A leaked session's chunk sections alone are
/// 1.7 MB a cycle.
const MAX_SMALL_BYTES_PER_CYCLE: i64 = 48 * 1024;

/// Pump a few frames with a short pause between, so work finishing on other threads (mesh
/// workers, the connection task's drop) is applied before the live heap is sampled.
fn settle(
    client: &mut Client,
    gpu: &mut mc_render::Gpu,
    phases: &mut PhaseList,
    camera: &mut Camera,
) {
    for _ in 0..3 {
        pump(client, gpu, phases, camera, &MoveInput::default());
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn repeated_connect_disconnect_cycles_leak_nothing() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    // The title screen every disconnect returns to is the Aero one (the `aero_title` mod is on
    // by default), so each cycle also leaves and re-enters the greenhouse.
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
    let total = WARMUP_CYCLES + MEASURED_CYCLES;

    // One peer per cycle on one port, each serving a join and then waiting for the client to
    // hang up — which is what a `Disconnect` button looks like from a server's side.
    let server = client.runtime().handle().spawn(async move {
        let chunks = chunks();
        for cycle in 0..total {
            let mut peer = serve_join(&listener, 40 + cycle as i32, &chunks, SECTION_MASK).await;
            peer.drain_until_closed().await;
        }
    });

    let mut baseline = None;
    let mut live_baseline = 0;
    let mut small_baseline = (0, 0);
    for cycle in 0..total {
        run_cycle(&mut client, &mut gpu, &mut phases, &mut camera, port, cycle);
        if cycle + 1 == WARMUP_CYCLES {
            settle(&mut client, &mut gpu, &mut phases, &mut camera);
            live_baseline = live_heap_bytes();
            small_baseline = live_blocks_and_small_bytes();
            baseline = process_memory_bytes();
        }
    }
    settle(&mut client, &mut gpu, &mut phases, &mut camera);
    let live_after = live_heap_bytes();
    let small_after = live_blocks_and_small_bytes();

    client
        .runtime()
        .handle()
        .block_on(server)
        .expect("the scripted peer must serve every cycle");

    assert_eq!(client.machine.joins() as usize, total);
    assert_eq!(client.machine.disconnects() as usize, total);
    // Structural, for the title screen: the greenhouse was drawn (so this test covers it), and
    // it was built once and reused on every later visit. Rebuilding it per visit is what grew
    // the GPU driver's memory by megabytes over the first visits (aero-title leak follow-up).
    assert_eq!(
        client.aero_title_scene_builds(),
        1,
        "the Aero title screen's greenhouse must be built once and reused across visits"
    );

    // Reported, not gated: the working set includes freed memory the allocator keeps and GPU
    // driver memory, which made the old resident-set gate flaky (see the module docs).
    if let (Some(baseline), Some(after)) = (baseline, process_memory_bytes()) {
        let growth = after.saturating_sub(baseline);
        println!(
            "resident set: {:.1} MB after {WARMUP_CYCLES} warm-up cycles, {:.1} MB after \
             {MEASURED_CYCLES} more — growth {:.1} MB ({:.0} KB/cycle), not gated",
            baseline as f64 / 1_048_576.0,
            after as f64 / 1_048_576.0,
            growth as f64 / 1_048_576.0,
            growth as f64 / 1024.0 / MEASURED_CYCLES as f64,
        );
    }
    let live_growth = live_after - live_baseline;
    println!(
        "live heap: {:.1} MB after {WARMUP_CYCLES} warm-up cycles, {:.1} MB after \
         {MEASURED_CYCLES} more — growth {:.1} KB ({:.1} KB/cycle)",
        live_baseline as f64 / 1_048_576.0,
        live_after as f64 / 1_048_576.0,
        live_growth as f64 / 1024.0,
        live_growth as f64 / 1024.0 / MEASURED_CYCLES as f64,
    );
    assert!(
        live_growth < MAX_GROWTH_BYTES as i64,
        "the live heap grew {live_growth} bytes over {MEASURED_CYCLES} cycles, which is more than \
         one session's world per cycle would have to cost; a session is being kept alive"
    );
    // The sharp gates. Per cycle, in blocks and in bytes of small
    // blocks, so a leak of far less than a whole world fails, and a large buffer that happens to
    // be freed a frame after the sample cannot.
    let blocks_per_cycle = (small_after.0 - small_baseline.0) / MEASURED_CYCLES as i64;
    let small_per_cycle = (small_after.1 - small_baseline.1) / MEASURED_CYCLES as i64;
    println!(
        "per measured cycle: {blocks_per_cycle:+} live blocks, {:+.1} KB in blocks under {} KB",
        small_per_cycle as f64 / 1024.0,
        SMALL_BLOCK / 1024
    );
    assert!(
        blocks_per_cycle <= MAX_BLOCKS_PER_CYCLE,
        "each reconnect kept {blocks_per_cycle} more live blocks (allowed \
         {MAX_BLOCKS_PER_CYCLE}); run `ml_leak_harness reconnect` with ML_TRACE=1 to name them"
    );
    assert!(
        small_per_cycle <= MAX_SMALL_BYTES_PER_CYCLE,
        "each reconnect kept {small_per_cycle} more bytes in small blocks (allowed \
         {MAX_SMALL_BYTES_PER_CYCLE}); run `ml_leak_harness reconnect` with ML_TRACE=1"
    );
}

/// One join → play → disconnect cycle, with the structural assertions on both ends.
fn run_cycle(
    client: &mut Client,
    gpu: &mut mc_render::Gpu,
    phases: &mut PhaseList,
    camera: &mut Camera,
    port: u16,
    cycle: usize,
) {
    // ---- before: nothing from the previous cycle survives ----
    assert!(
        matches!(client.machine.state(), GameState::Menu),
        "cycle {cycle} must start on the menu, got {:?}",
        client.machine.state()
    );
    assert!(
        client.session().is_none(),
        "cycle {cycle}: a session survived"
    );
    assert_eq!(
        client.uploaded_section_count(),
        0,
        "cycle {cycle}: chunk meshes survived"
    );
    // "Keep chat between servers" (on by default) keeps chat for the
    // whole client run, capped by "History length". The loopback server sends no chat and a join
    // adds no line of its own, so chat is still empty after every earlier cycle.
    assert_eq!(
        (client.chat.lines().len(), client.chat.received_count()),
        (0, 0),
        "cycle {cycle}: a join added a chat line"
    );

    client.connect("127.0.0.1", port);

    pump_until(
        client,
        gpu,
        phases,
        camera,
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

    // Let the mesher and the uploader actually run, so there is GPU memory to leak.
    for _ in 0..12 {
        pump(client, gpu, phases, camera, &MoveInput::default());
    }
    let session = client.session().expect("a live session");
    assert_eq!(session.world().loaded_chunk_count(), 9);
    assert!(
        client.uploaded_section_count() > 0,
        "cycle {cycle}: nothing reached the GPU, so nothing is being tested"
    );

    // ---- disconnect ----
    client.disconnect();
    assert!(matches!(client.machine.state(), GameState::Menu));
    assert!(
        client.session().is_none(),
        "cycle {cycle}: the session must go"
    );
    assert_eq!(
        client.uploaded_section_count(),
        0,
        "cycle {cycle}: every chunk mesh must be released with the world"
    );

    // A few frames on the menu, so anything deferred to the next frame actually happens.
    for _ in 0..3 {
        pump(client, gpu, phases, camera, &MoveInput::default());
    }
}

/// A control for the measurement above.
///
/// A "no growth" result is only worth something if the instrument could have seen the growth it
/// is looking for. This retains exactly what [`MEASURED_CYCLES`] leaked sessions' worth of chunk
/// storage would be — twelve copies of a nine-column, sixteen-section world — and asserts the
/// live heap moves by enough to trip [`MAX_GROWTH_BYTES`]. If this ever fails, the leak test
/// above is not proving anything and must be fixed rather than trusted.
#[test]
fn the_memory_measurement_can_see_a_leak_of_the_size_being_tested() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let baseline = live_heap_bytes();

    // One column: sixteen sections of blocks + block light + sky light, plus the biome array.
    const COLUMN_BYTES: usize = 16 * (8192 + 2048 + 2048) + 256;
    const COLUMNS: usize = 9;

    let mut leaked: Vec<Vec<u8>> = Vec::with_capacity(MEASURED_CYCLES * COLUMNS);
    for cycle in 0..MEASURED_CYCLES {
        for column in 0..COLUMNS {
            // Written, not merely reserved: an untouched allocation may never become resident.
            let mut block = vec![0u8; COLUMN_BYTES];
            for (index, byte) in block.iter_mut().enumerate().step_by(4096) {
                *byte = (cycle + column + index) as u8;
            }
            leaked.push(block);
        }
    }

    let growth = live_heap_bytes() - baseline;
    println!(
        "control: retaining {:.1} MB of chunk storage moved the live heap by {:.1} MB",
        (leaked.len() * COLUMN_BYTES) as f64 / 1_048_576.0,
        growth as f64 / 1_048_576.0,
    );
    assert!(
        growth > MAX_GROWTH_BYTES as i64,
        "the live-heap measurement could not see {} bytes of retained chunk storage \
         (growth {growth}), so the leak test above proves nothing",
        leaked.len() * COLUMN_BYTES
    );
    // Keep it alive to the very end, so the compiler cannot elide any of it.
    assert_eq!(leaked.len(), MEASURED_CYCLES * COLUMNS);
    drop(leaked);

    // The same for the per-cycle gates. A leaked session keeps its
    // world as many small blocks (one per chunk section: 16 per column), not nine big ones;
    // retaining exactly that per cycle must trip both the block gate and the small-byte gate.
    const SECTION_BYTES: usize = 8192 + 2048 + 2048;
    let before = live_blocks_and_small_bytes();
    let mut sections: Vec<Vec<u8>> = Vec::with_capacity(MEASURED_CYCLES * COLUMNS * 16);
    for cycle in 0..MEASURED_CYCLES {
        for section in 0..COLUMNS * 16 {
            let mut block = vec![0u8; SECTION_BYTES];
            block[section % SECTION_BYTES] = cycle as u8;
            sections.push(block);
        }
    }
    let after = live_blocks_and_small_bytes();
    let blocks_per_cycle = (after.0 - before.0) / MEASURED_CYCLES as i64;
    let small_per_cycle = (after.1 - before.1) / MEASURED_CYCLES as i64;
    println!(
        "control: a leaked world's sections per cycle read as {blocks_per_cycle} blocks and \
         {:.1} KB of small blocks per cycle",
        small_per_cycle as f64 / 1024.0
    );
    assert!(
        blocks_per_cycle > MAX_BLOCKS_PER_CYCLE,
        "the block gate cannot see a leaked world ({blocks_per_cycle} blocks per cycle)"
    );
    assert!(
        small_per_cycle > MAX_SMALL_BYTES_PER_CYCLE,
        "the small-byte gate cannot see a leaked world ({small_per_cycle} bytes per cycle)"
    );
    assert_eq!(sections.len(), MEASURED_CYCLES * COLUMNS * 16);
}
