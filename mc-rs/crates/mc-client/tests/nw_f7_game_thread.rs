//! A headless instrument for the game thread's non-render work.
//!
//! Drives a [`Session`] from a `.mccap` capture exactly the way the lockstep `skywars_bench`
//! does (one `--replayStep` of captured time per frame, one client tick every 50 ms of it), with
//! no window, no GPU and no renderer, and reports per frame:
//!
//! * **allocations** made by packet apply (`Session::poll_all`) and by the session tick
//!   (`Session::tick`), counted by a global allocator in this test binary (exact and
//!   deterministic: the counts are the evidence for an allocation-reuse change);
//! * **time** of the same two phases (attribution only: never a speed claim, which only
//!   `optimization/quick_ab.py` makes);
//! * a **digest** of the tracked entities after every tick (every `TrackedEntity`'s `Debug`
//!   text in world-list order, plus its ECS position/rotation/velocity/metadata), so a change
//!   that should be invisible can be shown to leave the simulation bit-identical.
//!
//! Ignored by default (it needs a capture). Run with:
//!
//! ```text
//! NW_F7_CAPTURE=<file.mccap> [NW_F7_FRAMES=60000] [NW_F7_STEP=1000] [NW_F7_DIGEST=1]
//! [NW_F7_OUT=<summary.txt>] [NW_F7_DIGESTS_OUT=<per-tick.txt>]
//!   cargo test -p mc-client --test nw_f7_game_thread -- --ignored --nocapture
//! ```
//!
//! The kill switches of the candidates are ordinary environment variables read once, so
//! running this twice with a switch off and on compares the two arms in one binary.

use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::Write as _;
use std::hash::{Hash as _, Hasher as _};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use mc_client::net::{NetHandle, ReplaySource};
use mc_client::session::JoinSettings;
use mc_client::sync::EntityTracker;
use mc_client::Session;
use mc_physics::state::MoveInput;

/// Counts every allocation (and reallocation) the process makes.
struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

// SAFETY: forwards every call to `System` unchanged; the counters are plain atomics.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        // SAFETY: same contract as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> u64 {
    ALLOCS.load(Ordering::Relaxed)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Drop every `index: {...}` map from a Debug dump (the same normalisation as the `rc_seek_determinism.rs`): an NBT compound keeps a name -> position `HashMap` with a fresh
/// `RandomState` beside its ordered entries, so it prints in a different order on every decode.
fn without_hash_indexes(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("index: {") {
        out.push_str(&rest[..at]);
        let mut depth = 0usize;
        let mut end = rest.len();
        for (offset, ch) in rest[at..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = at + offset + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        out.push_str("index: _");
        rest = &rest[end..];
    }
    out.push_str(rest);
    out
}

/// Make a `Debug` dump independent of hash-map iteration order: the entries of every `{...}`
/// block (struct bodies and maps alike) are canonicalised recursively and sorted. Struct fields
/// are unique names, so sorting them loses nothing; maps (`HashMap`/`FastMap` with a per-process
/// seed: attribute snapshots, potion amplifiers, NBT indexes) print in a fresh order in every
/// process, which is not simulation state. Lists (`[...]`) and tuples keep their order.
fn canonical(text: &str) -> String {
    fn block_end(bytes: &[u8], open: usize) -> usize {
        // Index of the `}` matching `bytes[open] == b'{'`, skipping string literals.
        let mut depth = 0usize;
        let mut at = open;
        while at < bytes.len() {
            match bytes[at] {
                b'"' => at = string_end(bytes, at),
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return at;
                    }
                }
                _ => {}
            }
            at += 1;
        }
        bytes.len() - 1
    }
    fn string_end(bytes: &[u8], open: usize) -> usize {
        // Index of the closing quote of the literal opened at `open` (Debug escapes `"` as `\"`).
        let mut at = open + 1;
        while at < bytes.len() {
            match bytes[at] {
                b'\\' => at += 2,
                b'"' => return at,
                _ => at += 1,
            }
        }
        bytes.len() - 1
    }
    fn split_top(inner: &str) -> Vec<String> {
        let bytes = inner.as_bytes();
        let mut parts = Vec::new();
        let (mut depth, mut start, mut at) = (0i32, 0usize, 0usize);
        while at < bytes.len() {
            match bytes[at] {
                b'"' => at = string_end(bytes, at),
                b'{' | b'[' | b'(' => depth += 1,
                b'}' | b']' | b')' => depth -= 1,
                b',' if depth == 0 => {
                    parts.push(inner[start..at].trim().to_owned());
                    start = at + 1;
                }
                _ => {}
            }
            at += 1;
        }
        let last = inner[start..].trim();
        if !last.is_empty() {
            parts.push(last.to_owned());
        }
        parts
    }
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    let mut copied = 0;
    while at < bytes.len() {
        match bytes[at] {
            b'"' => at = string_end(bytes, at) + 1,
            b'{' => {
                let end = block_end(bytes, at);
                out.push_str(&text[copied..at]);
                let mut parts: Vec<String> = split_top(&text[at + 1..end])
                    .iter()
                    .map(|part| canonical(part))
                    .collect();
                parts.sort();
                out.push('{');
                out.push_str(&parts.join(", "));
                out.push('}');
                at = end + 1;
                copied = at;
            }
            _ => at += 1,
        }
    }
    out.push_str(&text[copied..]);
    out
}

/// Every tracked entity in world-list order: its tracker record and its ECS components.
/// `NW_F7_DUMP=<file>` with `NW_F7_DUMP_TICK=<n>` also writes tick `n`'s full text.
fn digest(session: &Session, scratch: &mut String) -> u64 {
    let state = session.state();
    let world = &state.world;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for (id, tracked) in state.entities.iter() {
        scratch.clear();
        let _ = write!(scratch, "{id} {tracked:?}");
        let entity = tracked.entity;
        if let Ok(position) = world.ecs.get::<&mc_world::entity::Position>(entity) {
            let _ = write!(scratch, " P{:?}", position.0);
        }
        if let Ok(rotation) = world.ecs.get::<&mc_world::entity::Rotation>(entity) {
            let _ = write!(scratch, " R{:?}", *rotation);
        }
        if let Ok(velocity) = world.ecs.get::<&mc_world::entity::Velocity>(entity) {
            let _ = write!(scratch, " V{:?}", velocity.0);
        }
        if let Ok(metadata) = world.ecs.get::<&mc_world::entity::Metadata>(entity) {
            let _ = write!(scratch, " M{:?}", metadata.0);
        }
        let text = canonical(&without_hash_indexes(scratch));
        text.hash(&mut hasher);
        if let Some(dump) = DUMP.with(|dump| dump.borrow_mut().as_mut().map(std::mem::take)) {
            let mut dump = dump;
            dump.push_str(&text);
            dump.push('\n');
            DUMP.with(|cell| *cell.borrow_mut() = Some(dump));
        }
    }
    state.entities.len().hash(&mut hasher);
    hasher.finish()
}

thread_local! {
    /// The text of the tick being dumped (`NW_F7_DUMP_TICK`), collected by [`digest`].
    static DUMP: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

#[derive(Default)]
struct Phase {
    calls: u64,
    allocs: u64,
    nanos: u128,
    max_nanos: u128,
}

impl Phase {
    fn add(&mut self, allocs: u64, nanos: u128) {
        self.calls += 1;
        self.allocs += allocs;
        self.nanos += nanos;
        self.max_nanos = self.max_nanos.max(nanos);
    }

    fn line(&self, name: &str) -> String {
        let calls = self.calls.max(1);
        format!(
            "{name:6} calls {:7}  allocs {:9} ({:8.2}/call)  time {:9.3} ms \
             ({:7.3} us/call, max {:8.3} us)",
            self.calls,
            self.allocs,
            self.allocs as f64 / calls as f64,
            self.nanos as f64 / 1e6,
            self.nanos as f64 / 1e3 / calls as f64,
            self.max_nanos as f64 / 1e3,
        )
    }
}

#[test]
#[ignore = "needs NW_F7_CAPTURE=<file.mccap>"]
fn game_thread_work_over_a_capture() {
    let Ok(capture) = std::env::var("NW_F7_CAPTURE") else {
        eprintln!("skip: set NW_F7_CAPTURE");
        return;
    };
    let frames = env_u64("NW_F7_FRAMES", 60_000);
    let step = env_u64("NW_F7_STEP", 1_000);
    let with_digest = env_u64("NW_F7_DIGEST", 1) != 0;
    let skip_frames = env_u64("NW_F7_SKIP_FRAMES", 0);
    let dump_tick = std::env::var("NW_F7_DUMP_TICK")
        .ok()
        .and_then(|value| value.parse::<u64>().ok());
    let frames_per_tick = (50_000 / step.max(1)).max(1);
    // `MC_PROFILE=1`: the client's own profiler (every `profile_scope!`), dumped at the end.
    let profiling = mc_core::profile::enabled();
    if profiling {
        // The client's own subscriber: the formatter (set `RUST_LOG=warn` to quiet it) plus the
        // profiler layer, whose clock `init_tracing` starts.
        mc_core::init_tracing();
    }

    let source = ReplaySource::from_file(&capture, step).expect("capture");
    let mut session = Session::new(NetHandle::replay(source), JoinSettings::default());
    // The same reproducible tracker stream the lockstep bench installs.
    session.state_mut().entities = EntityTracker::with_random_seed(1);
    // `NW_F7_PARTICLES=1`: also run the client's per-tick entity particle emitters (potion
    // swirls, guardian and wolf effects) into a seeded particle system after every tick, as
    // `Client::tick` does, and fold the particle count into each tick's digest.
    let with_particles = env_u64("NW_F7_PARTICLES", 0) != 0;
    let mut particles = mc_render::particle::ParticleSystem::with_seed(1);

    let mut poll = Phase::default();
    let mut tick = Phase::default();
    let mut digests = std::collections::hash_map::DefaultHasher::new();
    let mut last_digest = 0;
    let mut digest_lines = String::new();
    let mut scratch = String::new();
    let mut events = 0usize;
    // The slowest polls: (nanoseconds, frame, events), to map back to the capture's packets.
    let mut slowest: Vec<(u128, u64, usize)> = Vec::new();
    let mut max_entities = 0usize;
    let input = MoveInput::default();
    for frame in 1..=frames {
        let measured = frame > skip_frames;
        let a0 = allocs();
        let t0 = Instant::now();
        let batch = session.poll_all();
        let nanos = t0.elapsed().as_nanos();
        let a1 = allocs();
        events += batch.len();
        if measured {
            poll.add(a1 - a0, nanos);
            slowest.push((nanos, frame, batch.len()));
            if slowest.len() > 64 {
                slowest.sort_unstable_by(|a, b| b.cmp(a));
                slowest.truncate(32);
            }
        }
        drop(batch);
        if frame % frames_per_tick == 0 {
            let a0 = allocs();
            let t0 = Instant::now();
            let packet = session.tick(&input);
            let nanos = t0.elapsed().as_nanos();
            let a1 = allocs();
            drop(packet);
            if measured {
                tick.add(a1 - a0, nanos);
            }
            max_entities = max_entities.max(session.state().entities.len());
            if with_particles {
                let view_pos = session
                    .predictor()
                    .map_or(mc_math::DVec3::ZERO, |predictor| predictor.state.pos);
                let state = session.state_mut();
                mc_client::sync::watched::tick_entity_particles(
                    &mut state.entities,
                    &state.world,
                    view_pos,
                    &mut particles,
                );
            }
            if with_digest {
                let dumping = dump_tick == Some(tick.calls);
                if dumping {
                    DUMP.with(|dump| *dump.borrow_mut() = Some(String::new()));
                }
                last_digest = digest(&session, &mut scratch);
                last_digest.hash(&mut digests);
                let particle_count = particles.len();
                particle_count.hash(&mut digests);
                let _ = writeln!(
                    digest_lines,
                    "{} {last_digest:016x} particles {particle_count}",
                    tick.calls
                );
                if dumping {
                    let text = DUMP
                        .with(|dump| dump.borrow_mut().take())
                        .unwrap_or_default();
                    if let Ok(path) = std::env::var("NW_F7_DUMP") {
                        std::fs::write(path, text).expect("write the dump");
                    }
                }
            }
        }
    }
    let summary = format!(
        "capture {capture}\nframes {frames} (step {step} us, first {skip_frames} unmeasured), \
         events {events}, max tracked {max_entities}\n{}\n{}\ndigest of every tick {:016x} \
         (last {last_digest:016x}){}\n",
        poll.line("poll"),
        tick.line("tick"),
        digests.finish(),
        if with_digest { "" } else { " (digest off)" },
    );
    print!("{summary}");
    let mut kinds = std::collections::BTreeMap::<String, usize>::new();
    for (_, tracked) in session.state().entities.iter() {
        *kinds.entry(format!("{:?}", tracked.kind)).or_default() += 1;
    }
    println!("kinds at the end: {kinds:?}");
    slowest.sort_unstable_by(|a, b| b.cmp(a));
    slowest.truncate(16);
    let slow: Vec<String> = slowest
        .iter()
        .map(|(nanos, frame, events)| format!("f{frame}:{:.0}us/{events}ev", *nanos as f64 / 1e3))
        .collect();
    println!("slowest polls: {}", slow.join(" "));
    if profiling {
        mc_core::profile::dump();
    }
    if let Ok(out) = std::env::var("NW_F7_OUT") {
        std::fs::write(out, &summary).expect("write the summary");
    }
    if let Ok(out) = std::env::var("NW_F7_DIGESTS_OUT") {
        std::fs::write(out, &digest_lines).expect("write the per-tick digests");
    }
}
