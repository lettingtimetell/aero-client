//! Where the world's memory goes in a real game.
//!
//! Plays a recorded game (`.aeroreplay`) through the real client, headless (no window, no GPU),
//! and at evenly spaced replay times reports:
//!
//! * the **live heap** (bytes allocated minus bytes freed, on every thread, from a counting
//!   allocator) and the process's private bytes and working set;
//! * the world: chunk columns and sections, and a census of what the sections hold (all-air
//!   sections, sections of one block state, light arrays holding one value), with the bytes the
//!   column and section structs take;
//! * entities (tracked and in the arena), tab-list rows, scoreboard objectives and teams.
//!
//! Ignored by default (it needs the vanilla jar and a replay). Run:
//!
//! ```text
//! MC_JARS_DIR=<jars> MW_REPLAY=<file.aeroreplay> MW_GAME_DIR=<scratch dir> \
//!   cargo test -p mc-client --test mw_world_memory -- --ignored --nocapture
//! ```
//!
//! `MW_STEPS` (default 24) sets how many sample points the replay is cut into, and
//! `MW_CENSUS_EVERY` (default 1) how often the section census runs. Set `MC_RC_CHECKPOINTS=0` to
//! measure the world alone; with checkpoints on, the viewer's world copies are counted too.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashSet;
use std::sync::atomic::{AtomicI64, Ordering};

use mc_client::app::{Client, ClientConfig};
use mc_client::sync::TrackedEntity;
use mc_world::chunk::{Chunk, ChunkSection};

struct Counting;

static LIVE: AtomicI64 = AtomicI64::new(0);
static PEAK: AtomicI64 = AtomicI64::new(0);
/// Live bytes and live blocks by allocation size class (power-of-two buckets), to name what holds
/// the memory.
static BY_CLASS: [AtomicI64; 40] = [const { AtomicI64::new(0) }; 40];
static BLOCKS_BY_CLASS: [AtomicI64; 40] = [const { AtomicI64::new(0) }; 40];

fn class(size: usize) -> usize {
    ((usize::BITS - size.max(1).leading_zeros()) as usize).min(39)
}

/// Allocations (and growths) of at least [`BIG`] bytes, as `(new size, old size, step)`, in the
/// order they happened: a fixed array, so recording one never allocates.
const BIG: usize = 4 << 20;
static BIG_LOG: [[AtomicI64; 3]; 64] =
    [const { [AtomicI64::new(0), AtomicI64::new(0), AtomicI64::new(0)] }; 64];
static BIG_COUNT: AtomicI64 = AtomicI64::new(0);
/// The harness's current sample step, for [`BIG_LOG`].
static STEP: AtomicI64 = AtomicI64::new(-1);

fn note_big(size: usize, old: usize) {
    if size < BIG {
        return;
    }
    let index = BIG_COUNT.fetch_add(1, Ordering::Relaxed);
    if let Some(slot) = usize::try_from(index).ok().and_then(|i| BIG_LOG.get(i)) {
        slot[0].store(size as i64, Ordering::Relaxed);
        slot[1].store(old as i64, Ordering::Relaxed);
        slot[2].store(STEP.load(Ordering::Relaxed), Ordering::Relaxed);
    }
}

/// Book `size` live bytes in (positive `sign`) or out of (negative) the total.
fn track(size: usize, sign: i64) {
    let delta = sign * size as i64;
    let now = LIVE.fetch_add(delta, Ordering::Relaxed) + delta;
    BY_CLASS[class(size)].fetch_add(delta, Ordering::Relaxed);
    BLOCKS_BY_CLASS[class(size)].fetch_add(sign, Ordering::Relaxed);
    if sign > 0 {
        PEAK.fetch_max(now, Ordering::Relaxed);
    }
}

/// The live bytes and blocks in each size class now.
fn classes() -> Vec<(i64, i64)> {
    BY_CLASS
        .iter()
        .zip(&BLOCKS_BY_CLASS)
        .map(|(bytes, blocks)| {
            (
                bytes.load(Ordering::Relaxed),
                blocks.load(Ordering::Relaxed),
            )
        })
        .collect()
}

/// Print the size classes that grew most between `before` and now.
fn print_class_growth(label: &str, before: &[(i64, i64)]) {
    let now = classes();
    let mut rows: Vec<(usize, i64, i64)> = now
        .iter()
        .zip(before)
        .enumerate()
        .map(|(class, (now, before))| (class, now.0 - before.0, now.1 - before.1))
        .collect();
    rows.sort_by_key(|row| -row.1);
    println!("{label}: live bytes by allocation size, largest growth first");
    for (class, bytes, blocks) in rows.iter().take(8) {
        let low = if *class == 0 { 0 } else { 1u64 << (class - 1) };
        println!(
            "  {:>10} .. {:>10} B: {:>+8.2} MiB in {:>+8} blocks",
            low,
            (1u64 << class) - 1,
            *bytes as f64 / MIB,
            blocks
        );
    }
}

// SAFETY: forwards every call to `System` unchanged; the counters are plain atomics.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: same contract as the caller's.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            track(layout.size(), 1);
            note_big(layout.size(), 0);
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: same contract as the caller's.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            track(layout.size(), 1);
            note_big(layout.size(), 0);
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
            note_big(new_size, layout.size());
        }
        new
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const MIB: f64 = 1024.0 * 1024.0;

/// The process's private bytes and working set (Windows), or zeros elsewhere.
fn process_memory() -> (u64, u64) {
    #[cfg(windows)]
    {
        #[repr(C)]
        #[derive(Default)]
        struct CountersEx {
            cb: u32,
            page_fault_count: u32,
            peak_working_set: usize,
            working_set: usize,
            quota_peak_paged: usize,
            quota_paged: usize,
            quota_peak_non_paged: usize,
            quota_non_paged: usize,
            pagefile: usize,
            peak_pagefile: usize,
            private_usage: usize,
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentProcess() -> isize;
            fn K32GetProcessMemoryInfo(process: isize, counters: *mut CountersEx, cb: u32) -> i32;
        }
        let mut counters = CountersEx {
            cb: std::mem::size_of::<CountersEx>() as u32,
            ..CountersEx::default()
        };
        // SAFETY: a plain query into a correctly sized, writable struct for this process.
        let ok =
            unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
        if ok != 0 {
            return (counters.private_usage as u64, counters.working_set as u64);
        }
    }
    (0, 0)
}

/// What the loaded sections hold.
#[derive(Debug, Default, Clone, Copy)]
struct Census {
    columns: usize,
    /// Section slots holding a section.
    sections: usize,
    /// Distinct sections (slots can share one `Arc` only across worlds, so this equals
    /// `sections` unless something cloned a column).
    unique_sections: usize,
    /// Sections whose every block is air.
    air: usize,
    /// Sections holding one block state in all 4096 cells (air included).
    uniform_blocks: usize,
    /// Block-light arrays holding one value in all cells, and how many of those hold 0.
    uniform_block_light: usize,
    zero_block_light: usize,
    /// Sky-light arrays holding one value, and how many of those hold 15 or 0.
    uniform_sky_light: usize,
    full_sky_light: usize,
    dark_sky_light: usize,
}

impl Census {
    fn take(client: &Client) -> Option<Census> {
        let session = client.session()?;
        let world = &session.state().world;
        let mut census = Census::default();
        let mut seen: HashSet<usize> = HashSet::new();
        for (_, chunk) in world.loaded_chunks() {
            census.columns += 1;
            for section in chunk.sections().iter().flatten() {
                census.sections += 1;
                if !seen.insert(std::sync::Arc::as_ptr(section) as usize) {
                    continue;
                }
                census.unique_sections += 1;
                census.add(section);
            }
        }
        Some(census)
    }

    fn add(&mut self, section: &ChunkSection) {
        let first_block = section.block(0, 0, 0);
        let first_block_light = section.block_light(0, 0, 0);
        let first_sky_light = section.sky_light(0, 0, 0);
        let (mut blocks_uniform, mut block_light_uniform, mut sky_light_uniform) =
            (true, true, true);
        for y in 0..16u8 {
            for z in 0..16u8 {
                for x in 0..16u8 {
                    blocks_uniform &= section.block(x, y, z) == first_block;
                    block_light_uniform &= section.block_light(x, y, z) == first_block_light;
                    sky_light_uniform &= section.sky_light(x, y, z) == first_sky_light;
                }
            }
        }
        if blocks_uniform {
            self.uniform_blocks += 1;
            if first_block == mc_data::BlockState::AIR {
                self.air += 1;
            }
        }
        if block_light_uniform {
            self.uniform_block_light += 1;
            if first_block_light == 0 {
                self.zero_block_light += 1;
            }
        }
        if sky_light_uniform {
            self.uniform_sky_light += 1;
            if first_sky_light == 15 {
                self.full_sky_light += 1;
            } else if first_sky_light == 0 {
                self.dark_sky_light += 1;
            }
        }
    }

    /// The bytes the column and section structs take (sections in their `Arc`s).
    fn struct_bytes(&self) -> usize {
        self.columns * std::mem::size_of::<Chunk>()
            + self.unique_sections * (std::mem::size_of::<ChunkSection>() + 16)
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn live_mib() -> f64 {
    LIVE.load(Ordering::Relaxed) as f64 / MIB
}

#[test]
#[ignore = "needs the vanilla jar and a replay (MW_REPLAY)"]
fn world_memory_over_a_replay() {
    let Ok(replay) = std::env::var("MW_REPLAY") else {
        eprintln!("skip: set MW_REPLAY to a .aeroreplay file");
        return;
    };
    let game_dir = std::env::var("MW_GAME_DIR").expect("MW_GAME_DIR (a scratch folder)");
    let steps = env_usize("MW_STEPS", 24).max(1);
    let census_every = env_usize("MW_CENSUS_EVERY", 1).max(1);

    println!(
        "sizes: Chunk {} B, ChunkSection {} B, TrackedEntity {} B, TileEntity {} B, ToGame {} B",
        std::mem::size_of::<Chunk>(),
        std::mem::size_of::<ChunkSection>(),
        std::mem::size_of::<TrackedEntity>(),
        std::mem::size_of::<mc_world::TileEntity>(),
        std::mem::size_of::<mc_client::ToGame>(),
    );

    let config = ClientConfig {
        game_dir: std::path::PathBuf::from(&game_dir),
        login: mc_client::app::config::LoginMode::Offline("MemoryBench".to_owned()),
        ..ClientConfig::default()
    }
    .with_discovered_assets();
    assert!(
        config.version_jar.as_ref().is_some_and(|jar| jar.is_file()),
        "no vanilla jar: set MC_JARS_DIR"
    );
    let mut client = Client::new(config, &mut |_, _| {}).expect("start the client");
    let (private_boot, _) = process_memory();
    let live_boot = live_mib();
    println!(
        "boot: live {live_boot:.1} MiB, private {:.1} MiB",
        private_boot as f64 / MIB
    );

    // The records alone, outside the client: what the viewer's record store costs, so the rest
    // of the open (the viewer's indexes and state) can be told apart.
    {
        let live_before = LIVE.load(Ordering::Relaxed);
        let reader = mc_replay::Reader::open(&replay).expect("read the replay");
        let live_reader = LIVE.load(Ordering::Relaxed);
        let store = mc_replay::RecordStore::read_all(&reader).expect("the records");
        let live_store = LIVE.load(Ordering::Relaxed);
        println!(
            "records alone: {} records, {:.1} MiB payload, store {:.1} MiB reserved and {:.1} MiB \
             live, file {:.1} MiB",
            store.len(),
            store.payload_bytes() as f64 / MIB,
            store.heap_bytes() as f64 / MIB,
            (live_store - live_reader) as f64 / MIB,
            (live_reader - live_before) as f64 / MIB,
        );
        drop(store);
        drop(reader);
    }
    client
        .replay_open(std::path::Path::new(&replay))
        .expect("open the replay");
    let (start, end) = client.replay_view_range();
    client.replay_render_at(start);
    let classes_open = classes();
    let live_open = live_mib();
    println!(
        "open: live {live_open:.1} MiB (+{:.1} for the replay's records), view {:.1}..{:.1} s",
        live_open - live_boot,
        start as f64 / 1e6,
        end as f64 / 1e6
    );
    println!(
        "{:>7} {:>9} {:>9} {:>9} | {:>5} {:>6} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8} | {:>5} {:>5} {:>4} {:>4} {:>5}",
        "t (s)",
        "live MiB",
        "priv MiB",
        "ws MiB",
        "cols",
        "secs",
        "air",
        "1blk",
        "bl1",
        "bl0",
        "sky1",
        "sky15",
        "world MiB",
        "ents",
        "ecs",
        "tab",
        "obj",
        "teams"
    );
    let mut last = Census::default();
    let mut peak_census = Census::default();
    let mut sums = Census::default();
    let mut census_samples = 0usize;
    for step in 0..=steps {
        let time = start + (end - start) * step as u64 / steps as u64;
        STEP.store(step as i64, Ordering::Relaxed);
        client.replay_render_at(time);
        if step % census_every == 0 || step == steps {
            if let Some(census) = Census::take(&client) {
                last = census;
                census_samples += 1;
                sums.columns += census.columns;
                sums.unique_sections += census.unique_sections;
                sums.air += census.air;
                sums.uniform_blocks += census.uniform_blocks;
                sums.uniform_block_light += census.uniform_block_light;
                sums.zero_block_light += census.zero_block_light;
                sums.uniform_sky_light += census.uniform_sky_light;
                sums.full_sky_light += census.full_sky_light;
                sums.dark_sky_light += census.dark_sky_light;
                if census.unique_sections > peak_census.unique_sections {
                    peak_census = census;
                }
            }
        }
        let (private, working_set) = process_memory();
        let (entities, ecs, tab, objectives, teams) =
            client.session().map_or((0, 0, 0, 0, 0), |session| {
                let state = session.state();
                (
                    state.entities.len(),
                    state.world.ecs.len() as usize,
                    state.players.len(),
                    state.scoreboard.objective_count(),
                    state.scoreboard.teams().count(),
                )
            });
        println!(
            "{:>7.1} {:>9.1} {:>9.1} {:>9.1} | {:>5} {:>6} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8.1} | {:>5} {:>5} {:>4} {:>4} {:>5}",
            time as f64 / 1e6,
            live_mib(),
            private as f64 / MIB,
            working_set as f64 / MIB,
            last.columns,
            last.unique_sections,
            last.air,
            last.uniform_blocks,
            last.uniform_block_light,
            last.zero_block_light,
            last.uniform_sky_light,
            last.full_sky_light,
            last.struct_bytes() as f64 / MIB,
            entities,
            ecs,
            tab,
            objectives,
            teams,
        );
    }
    let pct = |part: usize, whole: usize| {
        if whole == 0 {
            0.0
        } else {
            100.0 * part as f64 / whole as f64
        }
    };
    println!(
        "census over {census_samples} samples ({} section samples): air {:.1} %, one block \
         state {:.1} %, uniform block light {:.1} % (zero {:.1} %), uniform sky light {:.1} % \
         (15: {:.1} %, 0: {:.1} %)",
        sums.unique_sections,
        pct(sums.air, sums.unique_sections),
        pct(sums.uniform_blocks, sums.unique_sections),
        pct(sums.uniform_block_light, sums.unique_sections),
        pct(sums.zero_block_light, sums.unique_sections),
        pct(sums.uniform_sky_light, sums.unique_sections),
        pct(sums.full_sky_light, sums.unique_sections),
        pct(sums.dark_sky_light, sums.unique_sections),
    );
    println!(
        "peak world: {} columns, {} sections, {:.1} MiB of column and section structs \
         ({:.0} B per column, {:.2} sections per column)",
        peak_census.columns,
        peak_census.unique_sections,
        peak_census.struct_bytes() as f64 / MIB,
        peak_census.struct_bytes() as f64 / peak_census.columns.max(1) as f64,
        peak_census.unique_sections as f64 / peak_census.columns.max(1) as f64,
    );
    if let Some(session) = client.session() {
        println!(
            "at the end: {} tile entities",
            session.state().world.tile_entity_count()
        );
    }
    print_class_growth("in game (end of replay vs just opened)", &classes_open);
    let big = usize::try_from(BIG_COUNT.load(Ordering::Relaxed)).unwrap_or(0);
    println!(
        "allocations of {} MiB or more ({big}), as new <- old bytes at a step (-1: before the walk):",
        BIG >> 20
    );
    for slot in BIG_LOG.iter().take(big) {
        println!(
            "  {:>12} <- {:>12} at step {}",
            slot[0].load(Ordering::Relaxed),
            slot[1].load(Ordering::Relaxed),
            slot[2].load(Ordering::Relaxed)
        );
    }
    let live_before_close = live_mib();
    client.replay_close();
    let live_closed = live_mib();
    println!(
        "close: live {live_closed:.1} MiB ({:+.1} MiB at close), peak live {:.1} MiB",
        live_closed - live_before_close,
        PEAK.load(Ordering::Relaxed) as f64 / MIB
    );
}
