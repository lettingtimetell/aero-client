//! SkyWars lockstep benchmark wrapper: the runtime half of the `skywars` scene.
//!
//! Drives the ordinary client over a `--replay` capture, but replaces the wall-time tick clock with
//! a lockstep one: every frame advances the capture by `S` µs (`--replayStep S`) **and** the
//! simulation by exactly `S` µs, so one game tick runs every `K = 50 000 / S` frames. Frame `n` of
//! the scene is therefore the same scene in every build, whatever its frame rate (study
//! `optimization/scratch/skywars-scene-2026-11-05/code-notes/replay-fixture-harness.md` §2.3–2.5).
//! Async meshing stays on; `--diagnosticSimulation` is refused. The wrapper also plays the viewer's
//! scripted input (clicks, held keys, hotbar, look, approach) from the recipe, labels every frame
//! with its recipe segment in `frames.csv`, and writes a per-frame sidecar.
//!
//! No client source is modified: everything below goes through public `mc-client`/`mc-render`
//! APIs (see "Public API surface" at the end of this header).
//!
//! # Command line
//!
//! ```text
//! skywars_bench --script <recipe.json> [--lockstepMicros S] [--seed N] [--sidecar <csv>]
//!               [--screenshots] <ordinary client flags>
//! ```
//!
//! Wrapper flags are removed before the rest reaches `ClientConfig::parse_args`:
//!
//! | flag | default | meaning |
//! |---|---|---|
//! | `--script <path>` | required | the scene recipe (schema below) |
//! | `--lockstepMicros S` | `1000` | capture/simulation µs per frame. Must equal `--replayStep`, the recipe's `lockstep_micros`, and divide 50 000 |
//! | `--seed N` | recipe `seed` | client RNG seed: particle system and entity-tracker streams |
//! | `--sidecar <path>` | `<MC_FRAME_TRACE>.sidecar.csv` | per-frame sidecar CSV. One of the two must exist |
//! | `--screenshots` | off | take the recipe's `screenshots`. Off by default because a readback frame is a timing outlier; benchmark plans must not pass it |
//!
//! Required client flags: `--replay <capture>`, `--replayCamera fixed`, `--replayStep S`.
//! Always pass `--gameDir <scratch>` for manual runs (quick_ab adds its own). Do not pass
//! `--sidecar` from a quick_ab plan: every run would write the same path.
//!
//! Exit codes: `0` scene completed and every self-check passed; `2` bad arguments or recipe;
//! `3` a self-check failed (lockstep tick count, replay decode failures, JoinGame entity id, the
//! replay never opening, a key binding that did not take, a screenshot that could not be saved);
//! `4` the run stopped before the end of the last segment (`--frames` too small, window closed);
//! `1` the shell or client failed.
//!
//! # Allocator and animation clock
//!
//! * The bench runs on the shipped client's allocator (mimalloc, as `main.rs` installs it), so
//!   its numbers, and the PGO training that runs it, exercise what players run. Before, it ran
//!   on the system allocator. `MC_W18_OC_BENCH_MIMALLOC=0` selects the system allocator in the
//!   same binary, for an allocator A/B. The receipt and stderr name the allocator of the run.
//! * Every wall-clock animation (the enchantment glint, the HUD's clocks, the world border's and
//!   the end portal's scroll) follows the lockstep time: `step × S` µs, in whole ms, through
//!   `mc_render::entity::item::pin_animation_clock` before every draw (0 until the session
//!   opens). Two runs then draw identical frames, so their screenshots compare byte for byte.
//!   `MC_W18_OC_BENCH_CLOCK=0` keeps the live clocks. A player's clocks are never pinned.
//!
//! # Time model (what a "tick" in the recipe means)
//!
//! * Tick `T` is **server tick `T` counted from `JoinGame`**: capture time `[T·50 000, (T+1)·50 000)` µs.
//!   The generator stamps every packet of server tick `T` at exactly `T·50 000` µs.
//! * Lockstep frame `n` (1-based, counting only frames after the replay session opened) delivers
//!   the capture window `[(n−1)·S, n·S)` and then runs the client ticks that are due: after frame
//!   `n` the client has run exactly `⌊n·S / 50 000⌋` ticks (self-checked every frame).
//! * So capture tick `T` spans frames `T·K+1 ..= (T+1)·K`. Its packets are applied on the first of
//!   those frames, and the client tick runs on the last. **Scripted input "at tick `T`" is applied
//!   to that client tick**: after server tick `T`'s packets, before server tick `T+1`'s. A server
//!   reaction to a press at tick `T` (hurt status, S12, sounds) belongs at tick `T+1`.
//! * Every frame's `frames.csv` `phase` is the name of the segment containing its capture tick
//!   `⌊(n−1)·S / 50 000⌋`, so a segment `[a, b)` labels exactly `(b−a)·K` consecutive frames.
//!   Frames before the session opens keep the client's own labels (`loading`, `connecting`, ...).
//! * The wrapper stops by itself after frame `end_tick·K` of the last segment.
//!
//! # Recipe schema `mc-rs-skywars-scene/1` (authoritative)
//!
//! JSON, written by `skywars_fixture.rs`. The wrapper reads only the keys below and ignores every
//! other key, so the generator can record any extra metadata at the top level or inside `local`.
//! Every `tick`, `from` and `to` is a server tick as defined above; ranges are half-open
//! `[from, to)` and must have `from < to`. Entries may come in any order; entries for the same
//! tick add up (attack presses) or apply in file order (hotbar).
//!
//! ```text
//! {
//!   "schema": "mc-rs-skywars-scene/1",          // required, exact string
//!   "k": 50,                                    // required: frames per tick; k * lockstep_micros == 50000
//!   "lockstep_micros": 1000,                    // required: must equal --lockstepMicros and --replayStep
//!   "seed": 1,                                  // required: i64, the default for --seed
//!   "segments": [                               // required, non-empty, contiguous, first starts at 0
//!     {"name": "sw_join",  "start_tick": 0,    "end_tick": 300},
//!     {"name": "sw_fight", "start_tick": 300,  "end_tick": 1500},
//!     {"name": "sw_tail",  "start_tick": 1500, "end_tick": 1600}
//!   ],                                          // names: [A-Za-z0-9_.-]{1,64}, unique
//!   "screenshots": [[tick, "name"], ...],       // optional; see "Screenshots" below
//!   "local": {                                  // optional: no scripted input when absent
//!     "entity_id": 1,                           // required when `local` exists: JoinGame entity id (checked)
//!     "attack_presses": [[tick, count], ...],   // count >= 0 left-click taps in that tick's client tick
//!     "keys": [{"from": t0, "to": t1, "down": ["forward", "sprint"]}, ...],
//!     "hotbar": [[tick, slot], ...],            // slot 0..=8 (S09 numbering); tapped as key.hotbar.<slot+1>
//!     "look": [{"from": t0, "to": t1, "fixed": [yaw, pitch]},
//!              {"from": t0, "to": t1, "aim_entity": 17}],
//!     "look_track": [[tick, yaw, pitch], ...],  // keyframes, linearly interpolated, held after the last
//!     "approach": [{"from": t0, "to": t1, "entity": 17, "keep_within": 2.6, "sprint": true}]
//!   }
//! }
//! ```
//!
//! ## `local` semantics
//!
//! * **`attack_presses`** — each count is that many *taps* of the attack binding (press and
//!   release in the same tick, so a tap on a block never turns into held digging). The client's
//!   own `handle_game_input` turns each into `click_mouse`: arm swing, and an attack with local
//!   prediction (crit / magic-crit emitters) if the tick's crosshair is on an entity. Taps are
//!   only acted on while the UI accepts game input (`InGame`, i.e. after the first S08 closed the
//!   terrain screen); taps issued earlier are drained by the client and counted as dropped.
//!   Vanilla's 10 000-tick left-click lockout from a closing screen also swallows taps in the
//!   **first** in-game tick (it is cleared at the end of that tick), so start clicking later.
//!   Several taps in one tick behave as vanilla: after a tap that misses, the 10-tick miss lockout
//!   swallows the rest of that tick's taps.
//! * **`keys`** — held for every client tick in `[from, to)`. Names: `forward`, `back`, `left`,
//!   `right`, `jump`, `sneak`, `sprint`, `playerlist` (Tab overlay), `use` (use item; held use with
//!   a sword is blocking). Unknown names are an error. A one-tick `jump` range is one jump.
//! * **`hotbar`** — one tap of `key.hotbar.<slot+1>` in that tick.
//! * **`look`** ranges take precedence over `look_track`. With neither covering a tick the look is
//!   left alone (S08 packets and earlier script values stay in effect).
//!   * `fixed: [yaw, pitch]` — degrees, Minecraft convention (yaw 0 = +Z, 90 = −X; pitch + = down).
//!   * `aim_entity: id` — aim the eyes at the tracked remote entity's **body centre**: its
//!     interpolated position plus half its height (players 1.8; other types use their registry
//!     height). Per tick it uses current positions; per rendered frame it uses the frame's
//!     interpolated eye and target, so the crosshair stays on the drawn body. While the entity is
//!     not tracked (or not resident), the previous look is held.
//! * **`look_track`** — keyframes `[tick, yaw, pitch]`; the look at time `t` is the linear
//!   interpolation of the two surrounding keyframes (yaw along the shortest arc), the first
//!   keyframe's value from `tick−1` on, and the last keyframe's value forever after it. A track with
//!   one keyframe per tick is simply followed.
//! * **`approach`** — for each client tick in `[from, to)` where `entity` is tracked: if the
//!   horizontal (XZ) distance from the viewer's feet to the entity's current position exceeds
//!   `keep_within` blocks, hold `forward` (plus `sprint` when `"sprint": true`), else hold
//!   neither. Distance is measured at the start of the tick, before the viewer moves. Movement
//!   follows the current yaw, so pair it with `aim_entity` on the same entity. Held keys from
//!   `keys` are added on top (a `keys` forward cannot be released by `approach`).
//! * `aim_entity` and `approach` are **supported** through public APIs, so the generator may turn
//!   viewer knockback (S12 to the viewer) on: `approach` walks the viewer back into range.
//!
//! ## Per-frame look
//!
//! Scripted looks are applied in the client tick (shell camera and the predictor, so the tick's
//! crosshair pick and movement use them) **and** on every rendered frame (shell camera), so the
//! rendered camera moves smoothly between ticks. A frame `n` evaluates the look at fractional
//! time `t = n·S/50 000 − 1` and uses the rule of tick `⌈t⌉`: on the frame where tick `T` runs,
//! `t = T` exactly, and the frames in between move toward the next tick's look.
//!
//! ## Screenshots
//!
//! `screenshots: [[tick, "name"], ...]` (top level). With `--screenshots`, the frame on which
//! tick `tick`'s client tick runs (the last frame of capture tick `tick`) is read back after
//! rendering and before present (the same `AcquiredFrame::read_pixels` seam as
//! `optimization_lifecycle.rs`'s checkpoint images) and saved to
//! `<sidecar dir>/shots/<name>.png`, created exclusively. `name`: `[A-Za-z0-9_.-]{1,64}`, unique.
//! Without `--screenshots` they are ignored (and listed as skipped).
//!
//! # Sidecar CSV
//!
//! One row per lockstep frame, buffered in memory and written once at exit:
//! `frame,step,virtual_micros,tick,segment,particles,entities,packets_remaining,in_game,attacks,hit,yaw,pitch,pos_x,pos_y,pos_z,shot`.
//! `frame` is the `frames.csv` `frame_index` that frame gets (alignment only: it depends on the
//! number of loading frames and must not be compared across runs); `step` is `n`; `tick` is the
//! client's tick count after the frame; `segment` is the label; `in_game` 1 while the UI is
//! `InGame`; `attacks` the cumulative taps issued; `hit` the network id under the tick crosshair
//! or `-1`; `yaw/pitch/pos_*` the local player's after the frame (empty before `JoinGame`);
//! `shot` 1 on a screenshot frame. Every column except `frame` (and `shot` when one run took
//! screenshots) must match between two runs of the same binary, recipe and seed.
//!
//! # Public API surface used
//!
//! `Client::{new_deferred, particles, settings, config, ui, hit, session, session_mut,
//! tick_count, is_loading, fatal_error, shutdown, shutdown_report}`, `ShellApp for Client` (all
//! 18 methods, forwarded),
//! `Session::{state, state_mut, predictor, predictor_mut, replay_source, phase}`,
//! `ClientState::{entities, local, world}`, `EntityTracker::{with_random_seed, len, get,
//! is_resident}`, `TrackedEntity::{kind, render_position}`, `ReplaySource::{virtual_micros,
//! remaining, summary}`, `ParticleSystem::{with_seed, set_density, len}`, `KeyBindings::{get,
//! set_state, is_down, take_pressed}`, `DiagnosticClock::{Step, Frozen, Invalid}`,
//! `AcquiredFrame::read_pixels`.
//! Not public, so approximated: `TrackedEntity::dimensions()` (`pub(crate)`) would give the exact
//! live height of a resized entity (slime size, baby mobs); the wrapper uses `PLAYER_HEIGHT` for
//! players and the registry height otherwise, which is exact for players.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context as _, Result};
use mc_client::app::campath::CameraScript;
use mc_client::app::driver::Ui;
use mc_client::sync::{EntityTracker, TrackedKind};
use mc_client::{Client, ClientConfig, Phase};
use mc_math::DVec3;
use mc_physics::state::{PLAYER_EYE_HEIGHT, PLAYER_HEIGHT};
use mc_render::frame::telemetry::FrameWorkStats;
use mc_render::input::keys as bind;
use mc_render::particle::{ParticleDensity, ParticleSystem};
use mc_render::shell::DiagnosticClock;
use mc_render::{
    AcquiredFrame, Camera, FrameCtx, GameSettings, Gpu, Input, KeyBindings, PhaseList, Shell,
    ShellApp, ShellConfig, TickCtx,
};
use serde_json::{json, Map, Value};

/// Simulation microseconds per game tick (20 TPS).
const TICK_MICROS: u64 = 50_000;
const SCHEMA: &str = "mc-rs-skywars-scene/1";
/// The salts the diagnostic mode XORs into its seed (study §2.5), so a wrapper run and a
/// diagnostic run with the same seed draw the same streams.
const PARTICLE_SEED_SALT: i64 = 0x5041_5254_4943_4C45;
const ENTITY_SEED_SALT: i64 = 0x454E_5449_5449_4553;
/// Frames after loading finished by which the replay session must exist.
const SESSION_GRACE_FRAMES: u64 = 600;
const MAX_PRESSES_PER_TICK: u64 = 64;
/// Upper bound on the pre-reserved sidecar rows, so a bogus recipe cannot request gigabytes.
const MAX_RESERVED_ROWS: u64 = 4_000_000;
const SIDECAR_HEADER: &str = "frame,step,virtual_micros,tick,segment,particles,entities,\
packets_remaining,in_game,attacks,hit,yaw,pitch,pos_x,pos_y,pos_z,shot";

const EXIT_RUNTIME: u8 = 1;
const EXIT_USAGE: u8 = 2;
const EXIT_CHECK: u8 = 3;
const EXIT_INCOMPLETE: u8 = 4;

// ---------------------------------------------------------------------------------------------
// Allocator
// ---------------------------------------------------------------------------------------------

/// The shipped client's allocator, with a same-binary switch to the system one for an A/B.
///
/// `main.rs` installs mimalloc behind the default-on `mimalloc` feature; this does the same for
/// the bench, so the bench's numbers and the PGO training that runs it exercise the allocator
/// players run. `MC_W18_OC_BENCH_MIMALLOC=0` selects the system allocator instead. The choice is
/// made once, inside the process's first allocation, from the environment read through the OS
/// without allocating, and never changes afterwards, so every block is freed, resized or zeroed
/// by the allocator that made it. Afterwards each call costs one relaxed load and a branch.
#[cfg(all(feature = "mimalloc", not(feature = "dhat-heap")))]
mod bench_alloc {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicU8, Ordering};

    use mimalloc::MiMalloc;

    /// The switch: `0` selects the system allocator; anything else, or unset, mimalloc.
    const SWITCH: &str = "MC_W18_OC_BENCH_MIMALLOC";
    const UNDECIDED: u8 = 0;
    const MIMALLOC: u8 = 1;
    const SYSTEM: u8 = 2;

    static CHOICE: AtomicU8 = AtomicU8::new(UNDECIDED);

    pub struct BenchAllocator;

    /// Whether `SWITCH` is exactly `0`, read without touching the Rust heap (this runs inside the
    /// first allocation).
    #[cfg(windows)]
    fn system_requested() -> bool {
        #[link(name = "kernel32")]
        extern "system" {
            fn GetEnvironmentVariableW(name: *const u16, buffer: *mut u16, size: u32) -> u32;
        }
        let mut name = [0u16; SWITCH.len() + 1];
        for (wide, byte) in name.iter_mut().zip(SWITCH.bytes()) {
            *wide = u16::from(byte);
        }
        let mut value = [0u16; 4];
        // SAFETY: `name` is NUL-terminated (its last unit stays 0) and `value`'s length is
        // passed, so the call reads and writes only these two stack arrays.
        let len = unsafe {
            GetEnvironmentVariableW(name.as_ptr(), value.as_mut_ptr(), value.len() as u32)
        };
        len == 1 && value[0] == u16::from(b'0')
    }

    /// Elsewhere the bench always uses mimalloc (it is benchmarked on Windows).
    #[cfg(not(windows))]
    fn system_requested() -> bool {
        let _ = SWITCH;
        false
    }

    #[inline]
    fn use_system() -> bool {
        match CHOICE.load(Ordering::Relaxed) {
            MIMALLOC => false,
            SYSTEM => true,
            _ => decide(),
        }
    }

    #[cold]
    fn decide() -> bool {
        let choice = if system_requested() { SYSTEM } else { MIMALLOC };
        // The first allocation happens before any other thread exists; should two calls ever
        // race, both read the same environment, so both store the same value.
        match CHOICE.compare_exchange(UNDECIDED, choice, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => choice == SYSTEM,
            Err(existing) => existing == SYSTEM,
        }
    }

    /// The allocator this run uses, for the receipt.
    pub fn name() -> &'static str {
        if use_system() {
            "system"
        } else {
            "mimalloc"
        }
    }

    // SAFETY: every method forwards its arguments unchanged to one of two sound allocators, and
    // the choice between them is fixed before the first block is handed out.
    unsafe impl GlobalAlloc for BenchAllocator {
        #[inline]
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if use_system() {
                System.alloc(layout)
            } else {
                MiMalloc.alloc(layout)
            }
        }

        #[inline]
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            if use_system() {
                System.alloc_zeroed(layout)
            } else {
                MiMalloc.alloc_zeroed(layout)
            }
        }

        #[inline]
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            if use_system() {
                System.dealloc(ptr, layout);
            } else {
                MiMalloc.dealloc(ptr, layout);
            }
        }

        #[inline]
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            if use_system() {
                System.realloc(ptr, layout, new_size)
            } else {
                MiMalloc.realloc(ptr, layout, new_size)
            }
        }
    }
}

#[cfg(all(feature = "mimalloc", not(feature = "dhat-heap")))]
#[global_allocator]
static GLOBAL: bench_alloc::BenchAllocator = bench_alloc::BenchAllocator;

/// The allocator this run uses, for the receipt and stderr.
fn allocator_name() -> &'static str {
    #[cfg(all(feature = "mimalloc", not(feature = "dhat-heap")))]
    {
        bench_alloc::name()
    }
    #[cfg(not(all(feature = "mimalloc", not(feature = "dhat-heap"))))]
    {
        "system (built without the mimalloc feature)"
    }
}

/// Whether wall-clock animations follow the lockstep time
/// (`MC_W18_OC_BENCH_CLOCK=0` keeps the live clocks). Read once.
fn animation_clock_pinned() -> bool {
    std::env::var_os("MC_W18_OC_BENCH_CLOCK").is_none_or(|value| value != "0")
}

/// Whether the skin and cape download worker may run. Off unless
/// `MC_W18_OC_BENCH_SKIN_DOWNLOADS=1` (kept only to compare against the old behavior): a
/// benchmark must never reach the network, and a skin arriving at a run-dependent frame would
/// change what is drawn. Players then keep the default skin, and the `<assetsDir>/skins` cache
/// is neither read nor written.
fn skin_downloads_allowed() -> bool {
    std::env::var_os("MC_W18_OC_BENCH_SKIN_DOWNLOADS").is_some_and(|value| value == "1")
}

// ---------------------------------------------------------------------------------------------
// Recipe
// ---------------------------------------------------------------------------------------------

/// A key the script can hold for a range of ticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeldKey {
    Forward,
    Back,
    Left,
    Right,
    Jump,
    Sneak,
    Sprint,
    PlayerList,
    Use,
}

impl HeldKey {
    const ALL: [HeldKey; 9] = [
        HeldKey::Forward,
        HeldKey::Back,
        HeldKey::Left,
        HeldKey::Right,
        HeldKey::Jump,
        HeldKey::Sneak,
        HeldKey::Sprint,
        HeldKey::PlayerList,
        HeldKey::Use,
    ];
    const NAMES: [&'static str; 9] = [
        "forward",
        "back",
        "left",
        "right",
        "jump",
        "sneak",
        "sprint",
        "playerlist",
        "use",
    ];

    fn parse(name: &str) -> Option<Self> {
        Self::NAMES
            .iter()
            .position(|known| *known == name)
            .map(|index| Self::ALL[index])
    }

    fn binding(self) -> &'static str {
        match self {
            HeldKey::Forward => bind::FORWARD,
            HeldKey::Back => bind::BACK,
            HeldKey::Left => bind::LEFT,
            HeldKey::Right => bind::RIGHT,
            HeldKey::Jump => bind::JUMP,
            HeldKey::Sneak => bind::SNEAK,
            HeldKey::Sprint => bind::SPRINT,
            HeldKey::PlayerList => bind::PLAYER_LIST,
            HeldKey::Use => bind::USE_ITEM,
        }
    }

    fn bit(self) -> u16 {
        1 << (self as u16)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct HeldSet(u16);

impl HeldSet {
    fn insert(&mut self, key: HeldKey) {
        self.0 |= key.bit();
    }

    fn extend(&mut self, other: HeldSet) {
        self.0 |= other.0;
    }

    fn contains(self, key: HeldKey) -> bool {
        self.0 & key.bit() != 0
    }
}

#[derive(Debug, Clone, Copy)]
enum LookRule {
    Fixed { yaw: f32, pitch: f32 },
    Aim { entity: i32 },
}

#[derive(Debug, Clone, Copy)]
struct Approach {
    entity: i32,
    keep_within: f64,
    sprint: bool,
}

/// A value active for the half-open tick range `[from, to)`.
#[derive(Debug, Clone)]
struct Ranged<T> {
    from: u64,
    to: u64,
    value: T,
}

impl<T> Ranged<T> {
    fn covers(&self, tick: u64) -> bool {
        self.from <= tick && tick < self.to
    }
}

#[derive(Debug, Clone)]
struct Segment {
    /// Leaked once at startup: `ShellApp::diagnostic_phase` hands out `&'static str`.
    name: &'static str,
    start: u64,
    end: u64,
}

#[derive(Debug, Default)]
struct LocalScript {
    entity_id: i32,
    attacks: BTreeMap<u64, u64>,
    keys: Vec<Ranged<HeldSet>>,
    hotbar: BTreeMap<u64, Vec<u8>>,
    look: Vec<Ranged<LookRule>>,
    look_track: Vec<(u64, f32, f32)>,
    approach: Vec<Ranged<Approach>>,
}

#[derive(Debug)]
struct Recipe {
    k: u64,
    lockstep_micros: u64,
    seed: i64,
    segments: Vec<Segment>,
    screenshots: BTreeMap<u64, Vec<String>>,
    local: Option<LocalScript>,
}

impl Recipe {
    fn total_ticks(&self) -> u64 {
        self.segments.last().map_or(0, |segment| segment.end)
    }

    fn segment_index(&self, tick: u64) -> Option<usize> {
        self.segments
            .iter()
            .position(|segment| segment.start <= tick && tick < segment.end)
    }
}

type Fields = Map<String, Value>;

fn object<'a>(value: &'a Value, what: &str) -> Result<&'a Fields, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{what} must be an object"))
}

fn field<'a>(fields: &'a Fields, key: &str, what: &str) -> Result<&'a Value, String> {
    present(fields, key).ok_or_else(|| format!("{what}.{key} is required"))
}

/// A key that exists and is not `null`.
fn present<'a>(fields: &'a Fields, key: &str) -> Option<&'a Value> {
    fields.get(key).filter(|value| !value.is_null())
}

fn array<'a>(value: &'a Value, what: &str) -> Result<&'a Vec<Value>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{what} must be an array"))
}

fn tuple<'a>(value: &'a Value, len: usize, what: &str) -> Result<&'a [Value], String> {
    let items = array(value, what)?;
    if items.len() != len {
        return Err(format!(
            "{what} must have {len} elements, got {}",
            items.len()
        ));
    }
    Ok(items)
}

fn unsigned(value: &Value, what: &str) -> Result<u64, String> {
    value
        .as_u64()
        .ok_or_else(|| format!("{what} must be a non-negative integer, got {value}"))
}

fn signed(value: &Value, what: &str) -> Result<i64, String> {
    value
        .as_i64()
        .ok_or_else(|| format!("{what} must be an integer, got {value}"))
}

fn int_id(value: &Value, what: &str) -> Result<i32, String> {
    let raw = signed(value, what)?;
    i32::try_from(raw).map_err(|_| format!("{what} {raw} is not a 32-bit entity id"))
}

fn finite(value: &Value, what: &str) -> Result<f64, String> {
    value
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or_else(|| format!("{what} must be a finite number, got {value}"))
}

fn string<'a>(value: &'a Value, what: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .ok_or_else(|| format!("{what} must be a string, got {value}"))
}

fn check_name(name: &str, what: &str) -> Result<(), String> {
    let valid = (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'));
    if valid {
        Ok(())
    } else {
        Err(format!(
            "{what} name {name:?} must match [A-Za-z0-9_.-]{{1,64}}"
        ))
    }
}

fn tick_within(value: &Value, total: u64, what: &str) -> Result<u64, String> {
    let tick = unsigned(value, what)?;
    if tick >= total {
        return Err(format!(
            "{what}: tick {tick} is not before the scene's end tick {total}"
        ));
    }
    Ok(tick)
}

fn tick_range(fields: &Fields, total: u64, what: &str) -> Result<(u64, u64), String> {
    let from = unsigned(field(fields, "from", what)?, &format!("{what}.from"))?;
    let to = unsigned(field(fields, "to", what)?, &format!("{what}.to"))?;
    if from >= to || to > total {
        return Err(format!(
            "{what}: range [{from}, {to}) must satisfy from < to <= {total} (the scene's end tick)"
        ));
    }
    Ok((from, to))
}

fn pitch_in_range(pitch: f64, what: &str) -> Result<f32, String> {
    if (-90.0..=90.0).contains(&pitch) {
        Ok(pitch as f32)
    } else {
        Err(format!("{what}: pitch {pitch} is outside [-90, 90]"))
    }
}

fn reject_overlaps<T>(ranges: &mut [Ranged<T>], what: &str) -> Result<(), String> {
    ranges.sort_by_key(|range| range.from);
    for pair in ranges.windows(2) {
        if pair[1].from < pair[0].to {
            return Err(format!(
                "{what}: ranges [{}, {}) and [{}, {}) overlap",
                pair[0].from, pair[0].to, pair[1].from, pair[1].to
            ));
        }
    }
    Ok(())
}

fn parse_recipe(text: &str) -> Result<Recipe, String> {
    let root: Value =
        serde_json::from_str(text).map_err(|error| format!("not valid JSON: {error}"))?;
    let top = object(&root, "recipe")?;
    let schema = string(field(top, "schema", "recipe")?, "recipe.schema")?;
    if schema != SCHEMA {
        return Err(format!("schema is {schema:?}, expected {SCHEMA:?}"));
    }
    let k = unsigned(field(top, "k", "recipe")?, "recipe.k")?;
    let lockstep_micros = unsigned(
        field(top, "lockstep_micros", "recipe")?,
        "recipe.lockstep_micros",
    )?;
    if k == 0 || lockstep_micros == 0 || k.checked_mul(lockstep_micros) != Some(TICK_MICROS) {
        return Err(format!(
            "k ({k}) x lockstep_micros ({lockstep_micros}) must equal {TICK_MICROS}"
        ));
    }
    let seed = signed(field(top, "seed", "recipe")?, "recipe.seed")?;

    let raw_segments = array(field(top, "segments", "recipe")?, "recipe.segments")?;
    if raw_segments.is_empty() {
        return Err("recipe.segments must not be empty".to_owned());
    }
    let mut segments = Vec::with_capacity(raw_segments.len());
    let mut names = BTreeSet::new();
    let mut next_start = 0;
    for (index, raw) in raw_segments.iter().enumerate() {
        let what = format!("segments[{index}]");
        let fields = object(raw, &what)?;
        let name = string(field(fields, "name", &what)?, &format!("{what}.name"))?;
        check_name(name, &what)?;
        if !names.insert(name.to_owned()) {
            return Err(format!("{what}: duplicate segment name {name:?}"));
        }
        let start = unsigned(
            field(fields, "start_tick", &what)?,
            &format!("{what}.start_tick"),
        )?;
        let end = unsigned(
            field(fields, "end_tick", &what)?,
            &format!("{what}.end_tick"),
        )?;
        if start != next_start {
            return Err(format!(
                "{what} starts at tick {start}, expected {next_start}: segments must tile the \
                 capture contiguously from tick 0"
            ));
        }
        if end <= start {
            return Err(format!(
                "{what}: end_tick {end} must exceed start_tick {start}"
            ));
        }
        next_start = end;
        segments.push(Segment {
            name: Box::leak(name.to_owned().into_boxed_str()),
            start,
            end,
        });
    }
    let total = next_start;
    if total.checked_mul(TICK_MICROS).is_none() {
        return Err(format!("the scene's end tick {total} is too large"));
    }

    let mut screenshots: BTreeMap<u64, Vec<String>> = BTreeMap::new();
    if let Some(raw) = present(top, "screenshots") {
        let mut seen = BTreeSet::new();
        for (index, entry) in array(raw, "recipe.screenshots")?.iter().enumerate() {
            let what = format!("screenshots[{index}]");
            let pair = tuple(entry, 2, &what)?;
            let tick = tick_within(&pair[0], total, &what)?;
            let name = string(&pair[1], &format!("{what} name"))?;
            check_name(name, &what)?;
            if !seen.insert(name.to_owned()) {
                return Err(format!("{what}: duplicate screenshot name {name:?}"));
            }
            screenshots.entry(tick).or_default().push(name.to_owned());
        }
    }

    let local = match present(top, "local") {
        Some(raw) => Some(parse_local(raw, total)?),
        None => None,
    };
    Ok(Recipe {
        k,
        lockstep_micros,
        seed,
        segments,
        screenshots,
        local,
    })
}

fn parse_local(value: &Value, total: u64) -> Result<LocalScript, String> {
    let fields = object(value, "local")?;
    let local_id = int_id(field(fields, "entity_id", "local")?, "local.entity_id")?;
    let mut script = LocalScript {
        entity_id: local_id,
        ..LocalScript::default()
    };

    if let Some(raw) = present(fields, "attack_presses") {
        for (index, entry) in array(raw, "local.attack_presses")?.iter().enumerate() {
            let what = format!("local.attack_presses[{index}]");
            let pair = tuple(entry, 2, &what)?;
            let tick = tick_within(&pair[0], total, &what)?;
            let count = unsigned(&pair[1], &format!("{what} count"))?;
            let total_in_tick = script.attacks.entry(tick).or_insert(0);
            *total_in_tick += count;
            if *total_in_tick > MAX_PRESSES_PER_TICK {
                return Err(format!(
                    "{what}: more than {MAX_PRESSES_PER_TICK} presses in tick {tick}"
                ));
            }
        }
        script.attacks.retain(|_, count| *count > 0);
    }

    if let Some(raw) = present(fields, "keys") {
        for (index, entry) in array(raw, "local.keys")?.iter().enumerate() {
            let what = format!("local.keys[{index}]");
            let entry = object(entry, &what)?;
            let (from, to) = tick_range(entry, total, &what)?;
            let mut held = HeldSet::default();
            for name in array(field(entry, "down", &what)?, &format!("{what}.down"))? {
                let name = string(name, &format!("{what}.down"))?;
                let key = HeldKey::parse(name).ok_or_else(|| {
                    format!(
                        "{what}.down: unknown key {name:?}; expected one of {}",
                        HeldKey::NAMES.join(", ")
                    )
                })?;
                held.insert(key);
            }
            script.keys.push(Ranged {
                from,
                to,
                value: held,
            });
        }
    }

    if let Some(raw) = present(fields, "hotbar") {
        for (index, entry) in array(raw, "local.hotbar")?.iter().enumerate() {
            let what = format!("local.hotbar[{index}]");
            let pair = tuple(entry, 2, &what)?;
            let tick = tick_within(&pair[0], total, &what)?;
            let slot = unsigned(&pair[1], &format!("{what} slot"))?;
            let slot = u8::try_from(slot)
                .ok()
                .filter(|slot| *slot <= 8)
                .ok_or_else(|| format!("{what}: slot {slot} must be 0..=8 (S09 numbering)"))?;
            script.hotbar.entry(tick).or_default().push(slot);
        }
    }

    if let Some(raw) = present(fields, "look") {
        for (index, entry) in array(raw, "local.look")?.iter().enumerate() {
            let what = format!("local.look[{index}]");
            let entry = object(entry, &what)?;
            let (from, to) = tick_range(entry, total, &what)?;
            let rule = match (present(entry, "fixed"), present(entry, "aim_entity")) {
                (Some(fixed), None) => {
                    let pair = tuple(fixed, 2, &format!("{what}.fixed"))?;
                    let yaw = finite(&pair[0], &format!("{what}.fixed yaw"))? as f32;
                    let pitch =
                        pitch_in_range(finite(&pair[1], &format!("{what}.fixed pitch"))?, &what)?;
                    LookRule::Fixed { yaw, pitch }
                }
                (None, Some(id)) => {
                    let entity = int_id(id, &format!("{what}.aim_entity"))?;
                    if entity == local_id {
                        return Err(format!("{what}: cannot aim at the viewer's own entity id"));
                    }
                    LookRule::Aim { entity }
                }
                _ => {
                    return Err(format!(
                        "{what} needs exactly one of \"fixed\" or \"aim_entity\""
                    ))
                }
            };
            script.look.push(Ranged {
                from,
                to,
                value: rule,
            });
        }
        reject_overlaps(&mut script.look, "local.look")?;
    }

    if let Some(raw) = present(fields, "look_track") {
        let mut seen = BTreeSet::new();
        for (index, entry) in array(raw, "local.look_track")?.iter().enumerate() {
            let what = format!("local.look_track[{index}]");
            let triple = tuple(entry, 3, &what)?;
            let tick = tick_within(&triple[0], total, &what)?;
            let yaw = finite(&triple[1], &format!("{what} yaw"))? as f32;
            let pitch = pitch_in_range(finite(&triple[2], &format!("{what} pitch"))?, &what)?;
            if !seen.insert(tick) {
                return Err(format!("{what}: duplicate keyframe for tick {tick}"));
            }
            script.look_track.push((tick, yaw, pitch));
        }
        script.look_track.sort_by_key(|keyframe| keyframe.0);
    }

    if let Some(raw) = present(fields, "approach") {
        for (index, entry) in array(raw, "local.approach")?.iter().enumerate() {
            let what = format!("local.approach[{index}]");
            let entry = object(entry, &what)?;
            let (from, to) = tick_range(entry, total, &what)?;
            let entity = int_id(field(entry, "entity", &what)?, &format!("{what}.entity"))?;
            if entity == local_id {
                return Err(format!(
                    "{what}: cannot approach the viewer's own entity id"
                ));
            }
            let keep_within = finite(
                field(entry, "keep_within", &what)?,
                &format!("{what}.keep_within"),
            )?;
            if keep_within <= 0.0 {
                return Err(format!("{what}: keep_within must be positive"));
            }
            let sprint = match present(entry, "sprint") {
                None => false,
                Some(value) => value
                    .as_bool()
                    .ok_or_else(|| format!("{what}.sprint must be a boolean"))?,
            };
            script.approach.push(Ranged {
                from,
                to,
                value: Approach {
                    entity,
                    keep_within,
                    sprint,
                },
            });
        }
        reject_overlaps(&mut script.approach, "local.approach")?;
    }
    Ok(script)
}

// ---------------------------------------------------------------------------------------------
// Scripted look and movement
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum LookOutcome {
    /// No rule covers this time: leave the camera alone.
    Unscripted,
    /// Apply this yaw/pitch.
    Set(f32, f32),
    /// An `aim_entity` target is not tracked: hold the previous look.
    TargetMissing,
}

/// Map an angle difference to `[-180, 180)`.
fn wrap_degrees(degrees: f32) -> f32 {
    let wrapped = degrees.rem_euclid(360.0);
    if wrapped >= 180.0 {
        wrapped - 360.0
    } else {
        wrapped
    }
}

/// Minecraft yaw/pitch that look along `delta`: yaw 0 = +Z, 90 = -X; positive pitch looks down.
/// The inverse of `mc_client::app::interact::look_vector`.
fn look_angles(delta: DVec3) -> (f32, f32) {
    let horizontal = (delta.x * delta.x + delta.z * delta.z).sqrt();
    let yaw = (-delta.x).atan2(delta.z).to_degrees();
    let pitch = -delta.y.atan2(horizontal).to_degrees();
    (yaw as f32, pitch as f32)
}

/// Height of a tracked entity's bounding box. Exact for players; the registry (constructor)
/// height otherwise, because the live `TrackedEntity::dimensions()` is `pub(crate)`.
fn body_height(kind: TrackedKind) -> f32 {
    match kind {
        TrackedKind::Player => PLAYER_HEIGHT,
        other => other
            .entity_type()
            .map_or(PLAYER_HEIGHT, |entity_type| entity_type.info().height),
    }
}

/// Yaw/pitch from the viewer's eye to `entity`'s body centre, both interpolated by `partial`
/// (1.0 = the current tick positions, which is what the tick's crosshair pick uses).
fn aim_at(client: &Client, entity: i32, partial: f32) -> Option<(f32, f32)> {
    let session = client.session()?;
    let predictor = session.predictor()?;
    let state = session.state();
    if !state.entities.is_resident(entity) {
        return None;
    }
    let tracked = state.entities.get(entity)?;
    let centre = tracked.render_position(&state.world, partial)
        + DVec3::new(0.0, f64::from(body_height(tracked.kind)) * 0.5, 0.0);
    let feet = if partial >= 1.0 {
        predictor.state.pos
    } else {
        predictor.state.prev_pos
            + (predictor.state.pos - predictor.state.prev_pos) * f64::from(partial)
    };
    // Mirrors `Session::render_view_eye_position` for the local player.
    let eye_height = if state.local.sleeping {
        0.2
    } else if predictor.state.sneaking {
        PLAYER_EYE_HEIGHT - 0.08
    } else {
        PLAYER_EYE_HEIGHT
    };
    Some(look_angles(
        centre - (feet + DVec3::new(0.0, f64::from(eye_height), 0.0)),
    ))
}

/// Horizontal distance from the viewer's feet to `entity`'s current position.
fn horizontal_distance(client: &Client, entity: i32) -> Option<f64> {
    let session = client.session()?;
    let predictor = session.predictor()?;
    let state = session.state();
    if !state.entities.is_resident(entity) {
        return None;
    }
    let target = state
        .entities
        .get(entity)?
        .render_position(&state.world, 1.0);
    let delta = target - predictor.state.pos;
    Some((delta.x * delta.x + delta.z * delta.z).sqrt())
}

/// The `look_track` value at fractional tick `time`; `None` before the track applies.
fn track_at(track: &[(u64, f32, f32)], time: f64, rule_tick: u64) -> Option<(f32, f32)> {
    let first = *track.first()?;
    if rule_tick < first.0 {
        return None;
    }
    let after = track.partition_point(|keyframe| keyframe.0 as f64 <= time);
    if after == 0 {
        return Some((first.1, first.2));
    }
    let (t0, yaw0, pitch0) = track[after - 1];
    let Some(&(t1, yaw1, pitch1)) = track.get(after) else {
        return Some((yaw0, pitch0));
    };
    let fraction = ((time - t0 as f64) / (t1 - t0) as f64) as f32;
    Some((
        yaw0 + wrap_degrees(yaw1 - yaw0) * fraction,
        pitch0 + (pitch1 - pitch0) * fraction,
    ))
}

/// Resolve the scripted look for `rule_tick` (ranges) / `time` (track). The yaw is unwrapped
/// against `current_yaw` so the camera never spins through a 360-degree jump.
fn resolve_look(
    client: &Client,
    local: &LocalScript,
    rule_tick: u64,
    time: f64,
    partial: f32,
    current_yaw: f32,
) -> LookOutcome {
    let rule = local
        .look
        .iter()
        .find(|range| range.covers(rule_tick))
        .map(|range| range.value);
    let target = match rule {
        Some(LookRule::Fixed { yaw, pitch }) => Some((yaw, pitch)),
        Some(LookRule::Aim { entity }) => match aim_at(client, entity, partial) {
            Some(angles) => Some(angles),
            None => return LookOutcome::TargetMissing,
        },
        None => track_at(&local.look_track, time, rule_tick),
    };
    match target {
        Some((yaw, pitch)) => LookOutcome::Set(
            current_yaw + wrap_degrees(yaw - current_yaw),
            pitch.clamp(-90.0, 90.0),
        ),
        None => LookOutcome::Unscripted,
    }
}

fn binding_of(bindings: &KeyBindings, name: &str) -> Result<Input, String> {
    let input = bindings
        .get(name)
        .ok_or_else(|| format!("no key binding named {name}"))?
        .binding();
    if !input.is_bound() {
        return Err(format!("{name} is unbound, so the script cannot press it"));
    }
    Ok(input)
}

/// Press and release `name` within this tick: one queued press, never a held button (a held
/// attack would turn a miss onto a block into client-predicted digging).
fn tap(bindings: &mut KeyBindings, name: &str) -> Result<(), String> {
    let input = binding_of(bindings, name)?;
    bindings.set_state(input, true);
    if !bindings.is_down(name) {
        return Err(format!(
            "pressing {input:?} did not reach {name}: another action shares that binding"
        ));
    }
    bindings.set_state(input, false);
    Ok(())
}

/// Hold `name` for this tick. `set_state` also queues a press; a hold that continues from the
/// previous tick drops that press again, so a held key presses once, like a physical one.
fn hold(bindings: &mut KeyBindings, name: &str, continuing: bool) -> Result<(), String> {
    let input = binding_of(bindings, name)?;
    bindings.set_state(input, true);
    if !bindings.is_down(name) {
        return Err(format!(
            "holding {input:?} did not reach {name}: another action shares that binding"
        ));
    }
    if continuing {
        bindings.take_pressed(name);
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The wrapper
// ---------------------------------------------------------------------------------------------

/// One sidecar row: the state after one lockstep frame.
#[derive(Debug, Clone, Copy)]
struct Row {
    frame: i64,
    step: u64,
    virtual_micros: u64,
    tick: u64,
    segment: Option<usize>,
    particles: usize,
    entities: usize,
    packets_remaining: usize,
    in_game: bool,
    attacks: u64,
    hit: i32,
    pose: Option<(f32, f32, DVec3)>,
    shot: bool,
}

/// What a lockstep frame observed of the session, collected before any `&mut self` call.
struct Observation {
    virtual_micros: Option<u64>,
    remaining: usize,
    playing: bool,
    local_id: i32,
    entities: usize,
    pose: Option<(f32, f32, DVec3)>,
}

struct Bench {
    client: Client,
    recipe: Recipe,
    step_micros: u64,
    seed: i64,
    total_steps: u64,
    /// `Some` only with `--screenshots`.
    shots_dir: Option<PathBuf>,
    /// Pin every wall-clock animation to the lockstep time.
    pin_clock: bool,

    /// Whether the current draw advanced the simulation clock (decided by the same predicate
    /// `diagnostic_clock` used at the top of the draw).
    draw_is_step: bool,
    /// Lockstep frames so far: `n` of the current draw.
    step: u64,
    current_segment: Option<usize>,
    /// Successful presents so far (`after_render` calls).
    presents: u64,
    pending_shot: Option<u64>,

    tracker_seeded: bool,
    had_session: bool,
    frames_without_session: u64,
    entity_id_checked: bool,
    complete: bool,
    failure: Option<String>,
    replay_summary: Option<String>,

    prev_held: HeldSet,
    attacks_issued: u64,
    attacks_dropped: u64,
    approach_ticks: u64,
    aim_missing_ticks: u64,

    rows: Vec<Row>,
    shots_taken: Vec<String>,
}

impl Bench {
    fn fail(&mut self, message: String) {
        if self.failure.is_none() {
            tracing::error!(%message, "skywars bench self-check failed");
            self.failure = Some(message);
        }
    }

    fn session_open(&self) -> bool {
        !self.client.is_loading() && self.client.session().is_some()
    }

    /// Replace the session's entity tracker with a seeded one before the first poll.
    fn seed_tracker(&mut self) -> Result<(), String> {
        let seed = self.seed ^ ENTITY_SEED_SALT;
        let session = self
            .client
            .session_mut()
            .ok_or("no session to seed".to_owned())?;
        if !session.state().entities.is_empty() {
            return Err(
                "the entity tracker had entities before the first poll; it cannot be seeded"
                    .to_owned(),
            );
        }
        session.state_mut().entities = EntityTracker::with_random_seed(seed);
        let source = session
            .replay_source()
            .ok_or("the session is not a replay".to_owned())?;
        let summary = source.summary();
        self.replay_summary = Some(summary.line());
        if summary.decode_failures > 0 {
            return Err(format!(
                "the capture has {} frames that failed to decode",
                summary.decode_failures
            ));
        }
        Ok(())
    }

    /// Scripted input for the client tick about to run, which processes server tick `tick`.
    fn apply_tick_input(&mut self, ctx: &mut TickCtx<'_>, tick: u64) -> Result<(), String> {
        let Some(local) = self.recipe.local.as_ref() else {
            return Ok(());
        };
        match resolve_look(&self.client, local, tick, tick as f64, 1.0, ctx.camera.yaw) {
            LookOutcome::Set(yaw, pitch) => {
                // The shell camera feeds movement input; the predictor feeds this tick's
                // crosshair pick (`update_hit_result` reads `render_view_rotation(1.0)`).
                ctx.camera.yaw = yaw;
                ctx.camera.pitch = pitch;
                if let Some(predictor) = self
                    .client
                    .session_mut()
                    .and_then(|session| session.predictor_mut())
                {
                    predictor.state.yaw = yaw;
                    predictor.state.pitch = pitch;
                }
            }
            LookOutcome::TargetMissing => self.aim_missing_ticks += 1,
            LookOutcome::Unscripted => {}
        }

        if let Some(slots) = local.hotbar.get(&tick) {
            for &slot in slots {
                tap(&mut ctx.settings.keys, &bind::hotbar(slot + 1))?;
            }
        }
        if let Some(&count) = local.attacks.get(&tick) {
            for _ in 0..count {
                tap(&mut ctx.settings.keys, bind::ATTACK)?;
            }
            self.attacks_issued += count;
            // The client drains presses while a screen blocks game input.
            if !ShellApp::game_input_allowed(&self.client) {
                self.attacks_dropped += count;
            }
        }

        let mut held = HeldSet::default();
        for range in &local.keys {
            if range.covers(tick) {
                held.extend(range.value);
            }
        }
        if let Some(rule) = local.approach.iter().find(|range| range.covers(tick)) {
            let approach = rule.value;
            if horizontal_distance(&self.client, approach.entity)
                .is_some_and(|distance| distance > approach.keep_within)
            {
                held.insert(HeldKey::Forward);
                if approach.sprint {
                    held.insert(HeldKey::Sprint);
                }
                self.approach_ticks += 1;
            }
        }
        for key in HeldKey::ALL {
            if held.contains(key) {
                hold(
                    &mut ctx.settings.keys,
                    key.binding(),
                    self.prev_held.contains(key),
                )?;
            }
        }
        self.prev_held = held;
        Ok(())
    }

    /// Self-checks and the sidecar row, after a lockstep frame's ticks and frame preparation.
    fn after_step_frame(&mut self) {
        let step = self.step;
        let expected_micros = step * self.step_micros;
        let expected_ticks = expected_micros / TICK_MICROS;
        let ticks = self.client.tick_count();
        let observed = self.client.session().map(|session| {
            let source = session.replay_source();
            Observation {
                virtual_micros: source.map(|source| source.virtual_micros()),
                remaining: source.map_or(0, |source| source.remaining()),
                playing: matches!(session.phase(), Phase::Playing),
                local_id: session.state().local.entity_id,
                entities: session.state().entities.len(),
                pose: session.predictor().map(|predictor| {
                    (
                        predictor.state.yaw,
                        predictor.state.pitch,
                        predictor.state.pos,
                    )
                }),
            }
        });
        let Some(observed) = observed else {
            self.fail("the replay session closed during a lockstep frame".to_owned());
            return;
        };
        self.had_session = true;
        let Some(virtual_micros) = observed.virtual_micros else {
            self.fail("the session is not a replay".to_owned());
            return;
        };
        if virtual_micros != expected_micros {
            self.fail(format!(
                "lockstep broken at step {step}: the replay clock is at {virtual_micros} us, \
                 expected {step} x {} = {expected_micros}",
                self.step_micros
            ));
        }
        if ticks != expected_ticks {
            self.fail(format!(
                "lockstep broken at step {step}: the client has run {ticks} ticks, expected \
                 floor({step} x {} / {TICK_MICROS}) = {expected_ticks}",
                self.step_micros
            ));
        }
        if observed.playing && !self.entity_id_checked {
            self.entity_id_checked = true;
            if let Some(expected) = self.recipe.local.as_ref().map(|local| local.entity_id) {
                if observed.local_id != expected {
                    self.fail(format!(
                        "JoinGame gave the viewer entity id {}, the recipe says {expected}",
                        observed.local_id
                    ));
                }
            }
        }
        self.rows.push(Row {
            frame: self.presents as i64 - 1,
            step,
            virtual_micros,
            tick: ticks,
            segment: self.current_segment,
            particles: self.client.particles.len(),
            entities: observed.entities,
            packets_remaining: observed.remaining,
            in_game: matches!(self.client.ui, Ui::InGame),
            attacks: self.attacks_issued,
            hit: self.client.hit.entity().map_or(-1, |hit| hit.network_id),
            pose: observed.pose,
            shot: false,
        });
        // The frame on which tick T's client tick ran is the one whose clock reaches (T+1) ticks.
        if self.shots_dir.is_some() && virtual_micros.is_multiple_of(TICK_MICROS) {
            let tick = virtual_micros / TICK_MICROS - 1;
            if self.recipe.screenshots.contains_key(&tick) {
                self.pending_shot = Some(tick);
            }
        }
        if step >= self.total_steps {
            self.complete = true;
        }
    }

    /// Non-lockstep draws: loading, and the draw on which loading finishes.
    fn watch_session(&mut self) {
        if self.client.is_loading() || self.client.session().is_some() {
            return;
        }
        if self.had_session {
            self.fail("the replay session closed before the scene ended".to_owned());
            return;
        }
        self.frames_without_session += 1;
        if self.frames_without_session > SESSION_GRACE_FRAMES {
            self.fail(format!(
                "the replay session did not open within {SESSION_GRACE_FRAMES} frames of the \
                 assets loading (unreadable capture? see the log)"
            ));
        }
    }

    fn save_screenshot(&self, gpu: &Gpu, frame: &AcquiredFrame, tick: u64) -> Result<Vec<String>> {
        let dir = self
            .shots_dir
            .as_ref()
            .context("screenshots are disabled")?;
        let names = self
            .recipe
            .screenshots
            .get(&tick)
            .context("no screenshot is scheduled for this tick")?;
        let (width, height) = (gpu.config.width, gpu.config.height);
        let pixels = frame
            .read_pixels(gpu)
            .map_err(|error| anyhow::anyhow!("read back the frame: {error}"))?;
        let image = image::RgbaImage::from_raw(width, height, pixels)
            .context("the readback is not width x height RGBA8")?;
        let mut png = Vec::new();
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
            .context("encode PNG")?;
        for name in names {
            write_exclusive(&dir.join(format!("{name}.png")), &png)?;
        }
        Ok(names.clone())
    }
}

impl ShellApp for Bench {
    fn init(&mut self, gpu: &Gpu, phases: &mut PhaseList) {
        ShellApp::init(&mut self.client, gpu, phases);
    }

    fn before_ticks(&mut self, camera: &mut Camera) {
        self.draw_is_step = self.session_open();
        if self.draw_is_step {
            self.step += 1;
            let capture_tick = (self.step - 1) * self.step_micros / TICK_MICROS;
            self.current_segment = self.recipe.segment_index(capture_tick);
            if !self.tracker_seeded {
                self.tracker_seeded = true;
                if let Err(message) = self.seed_tracker() {
                    self.fail(message);
                }
            }
        }
        if self.pin_clock {
            // Every wall-clock animation follows the lockstep time, in whole ms (0 until the
            // session opens), before anything of this draw reads a clock.
            let millis = self.step * self.step_micros / 1_000;
            mc_render::entity::item::pin_animation_clock(Some(millis));
        }
        ShellApp::before_ticks(&mut self.client, camera);
    }

    fn tick(&mut self, ctx: &mut TickCtx<'_>) {
        if self.draw_is_step && self.failure.is_none() {
            // Ticks completed so far: the tick about to run processes server tick `input_tick`.
            let input_tick = self.client.tick_count();
            if let Err(message) = self.apply_tick_input(ctx, input_tick) {
                self.fail(message);
            }
        }
        ShellApp::tick(&mut self.client, ctx);
    }

    fn on_key_event(&mut self, event: &winit::event::KeyEvent, sequence: u64) {
        ShellApp::on_key_event(&mut self.client, event, sequence);
    }

    fn on_focus_changed(&mut self, focused: bool) {
        ShellApp::on_focus_changed(&mut self.client, focused);
    }

    fn game_input_allowed(&self) -> bool {
        ShellApp::game_input_allowed(&self.client)
    }

    fn effective_frame_cap(&self, settings: &GameSettings, focused: bool) -> Option<u32> {
        ShellApp::effective_frame_cap(&self.client, settings, focused)
    }

    fn frame(&mut self, ctx: &mut FrameCtx<'_>) {
        if self.draw_is_step && self.failure.is_none() {
            if let Some(local) = self.recipe.local.as_ref() {
                let virtual_micros = self.step * self.step_micros;
                // The rule of the tick this frame leads up to; on a tick frame, that tick.
                let rule_tick = if virtual_micros.is_multiple_of(TICK_MICROS) {
                    virtual_micros / TICK_MICROS - 1
                } else {
                    virtual_micros / TICK_MICROS
                };
                let time = virtual_micros as f64 / TICK_MICROS as f64 - 1.0;
                if let LookOutcome::Set(yaw, pitch) = resolve_look(
                    &self.client,
                    local,
                    rule_tick,
                    time,
                    ctx.partial_ticks,
                    ctx.camera.yaw,
                ) {
                    ctx.camera.yaw = yaw;
                    ctx.camera.pitch = pitch;
                }
            }
        }
        ShellApp::frame(&mut self.client, ctx);
        if self.draw_is_step {
            self.after_step_frame();
        } else {
            self.watch_session();
        }
    }

    fn diagnostic_phase(&self) -> &'static str {
        if self.draw_is_step {
            return self
                .current_segment
                .map_or("sw_overrun", |index| self.recipe.segments[index].name);
        }
        ShellApp::diagnostic_phase(&self.client)
    }

    fn diagnostic_clock(&self) -> Option<DiagnosticClock> {
        if let Some(DiagnosticClock::Invalid(reason)) = ShellApp::diagnostic_clock(&self.client) {
            return Some(DiagnosticClock::Invalid(reason));
        }
        Some(if self.session_open() {
            DiagnosticClock::Step {
                micros: self.step_micros,
            }
        } else {
            DiagnosticClock::Frozen
        })
    }

    fn begin_diagnostic_frame(&mut self) {
        ShellApp::begin_diagnostic_frame(&mut self.client);
    }

    fn diagnostic_work(&self) -> FrameWorkStats {
        ShellApp::diagnostic_work(&self.client)
    }

    fn finalize_camera(&mut self, camera: &mut Camera) {
        ShellApp::finalize_camera(&mut self.client, camera);
    }

    fn after_render(&mut self, gpu: &Gpu, frame: &AcquiredFrame) {
        ShellApp::after_render(&mut self.client, gpu, frame);
        if let Some(tick) = self.pending_shot.take() {
            match self.save_screenshot(gpu, frame, tick) {
                Ok(names) => {
                    if let Some(row) = self.rows.last_mut() {
                        row.shot = true;
                    }
                    self.shots_taken.extend(names);
                }
                Err(error) => self.fail(format!("screenshot for tick {tick}: {error:#}")),
            }
        }
        self.presents += 1;
    }

    fn keep_running(&self) -> bool {
        !self.complete && self.failure.is_none() && ShellApp::keep_running(&self.client)
    }

    fn wants_fullscreen(&self) -> Option<bool> {
        ShellApp::wants_fullscreen(&self.client)
    }

    fn resized(&mut self, gpu: &Gpu, width: u32, height: u32) {
        ShellApp::resized(&mut self.client, gpu, width, height);
    }

    fn shutdown(&mut self, settings: &GameSettings) {
        ShellApp::shutdown(&mut self.client, settings);
    }
}

// ---------------------------------------------------------------------------------------------
// Command line and run
// ---------------------------------------------------------------------------------------------

struct Options {
    script: PathBuf,
    lockstep_micros: u64,
    seed: Option<i64>,
    sidecar: Option<PathBuf>,
    screenshots: bool,
    client_args: Vec<String>,
}

/// Remove the wrapper's own flags. Anything left goes to `ClientConfig::parse_args`, which would
/// only warn about an unknown flag and then silently drop its value.
fn split_args(args: Vec<String>) -> Result<Options, String> {
    let mut script = None;
    let mut lockstep_micros = None;
    let mut seed = None;
    let mut sidecar = None;
    let mut screenshots = false;
    let mut client_args = Vec::new();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let name = arg.strip_prefix("--").or_else(|| arg.strip_prefix('-'));
        let mut value = |flag: &str| args.next().ok_or_else(|| format!("--{flag} needs a value"));
        match name {
            Some("script") => {
                if script.replace(PathBuf::from(value("script")?)).is_some() {
                    return Err("duplicate --script".to_owned());
                }
            }
            Some("lockstepMicros") => {
                let raw = value("lockstepMicros")?;
                let parsed = raw.parse::<u64>().map_err(|_| {
                    format!("--lockstepMicros must be a positive integer, got {raw:?}")
                })?;
                if lockstep_micros.replace(parsed).is_some() {
                    return Err("duplicate --lockstepMicros".to_owned());
                }
            }
            Some("seed") => {
                let raw = value("seed")?;
                let parsed = raw
                    .parse::<i64>()
                    .map_err(|_| format!("--seed must be an i64, got {raw:?}"))?;
                if seed.replace(parsed).is_some() {
                    return Err("duplicate --seed".to_owned());
                }
            }
            Some("sidecar") => {
                if sidecar.replace(PathBuf::from(value("sidecar")?)).is_some() {
                    return Err("duplicate --sidecar".to_owned());
                }
            }
            Some("screenshots") => screenshots = true,
            _ => client_args.push(arg),
        }
    }
    Ok(Options {
        script: script.ok_or("--script <recipe.json> is required")?,
        lockstep_micros: lockstep_micros.unwrap_or(1_000),
        seed,
        sidecar,
        screenshots,
        client_args,
    })
}

struct Setup {
    config: ClientConfig,
    recipe: Recipe,
    recipe_path: PathBuf,
    step_micros: u64,
    seed: i64,
    sidecar: PathBuf,
    shots_dir: Option<PathBuf>,
}

fn prepare() -> Result<Setup, String> {
    let options = split_args(std::env::args().skip(1).collect())?;
    let step_micros = options.lockstep_micros;
    if step_micros == 0 || step_micros > TICK_MICROS || !TICK_MICROS.is_multiple_of(step_micros) {
        return Err(format!(
            "--lockstepMicros {step_micros} must divide {TICK_MICROS}"
        ));
    }
    let config = ClientConfig::parse_args(&options.client_args)?.with_discovered_assets();
    if config.replay.is_none() {
        return Err("--replay <capture.mccap> is required".to_owned());
    }
    if !matches!(config.replay_camera, CameraScript::Fixed) {
        return Err(
            "--replayCamera fixed is required: the default orbit overwrites the scripted player \
             every frame"
                .to_owned(),
        );
    }
    if config.replay_step_micros != step_micros {
        return Err(format!(
            "--replayStep {} must equal --lockstepMicros {step_micros}",
            config.replay_step_micros
        ));
    }
    if config.diagnostic_simulation.is_some() {
        return Err(
            "--diagnosticSimulation must not be used: it forces synchronous meshing and turns \
             game input off"
                .to_owned(),
        );
    }

    let text = fs::read_to_string(&options.script)
        .map_err(|error| format!("cannot read {}: {error}", options.script.display()))?;
    let recipe = parse_recipe(&text)
        .map_err(|error| format!("recipe {}: {error}", options.script.display()))?;
    if recipe.lockstep_micros != step_micros {
        return Err(format!(
            "the recipe was generated for lockstep_micros {}, but --lockstepMicros is {step_micros}",
            recipe.lockstep_micros
        ));
    }
    let seed = options.seed.unwrap_or(recipe.seed);
    if seed != recipe.seed {
        eprintln!(
            "skywars_bench: note: --seed {seed} overrides the recipe seed {}",
            recipe.seed
        );
    }

    let sidecar = match options.sidecar {
        Some(path) => path,
        None => {
            let mut path: OsString = std::env::var_os("MC_FRAME_TRACE")
                .filter(|value| !value.is_empty())
                .ok_or("no sidecar path: pass --sidecar <csv> or set MC_FRAME_TRACE")?;
            path.push(".sidecar.csv");
            PathBuf::from(path)
        }
    };
    if sidecar.exists() {
        return Err(format!(
            "the sidecar {} already exists; refusing to overwrite it",
            sidecar.display()
        ));
    }
    let sidecar_dir = sidecar
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    if !sidecar_dir.is_dir() {
        return Err(format!(
            "the sidecar's directory {} does not exist",
            sidecar_dir.display()
        ));
    }
    let shots_dir = if options.screenshots {
        let dir = sidecar_dir.join("shots");
        for name in recipe.screenshots.values().flatten() {
            let path = dir.join(format!("{name}.png"));
            if path.exists() {
                return Err(format!(
                    "the screenshot {} already exists; refusing to overwrite it",
                    path.display()
                ));
            }
        }
        Some(dir)
    } else {
        None
    };
    Ok(Setup {
        config,
        recipe,
        recipe_path: options.script,
        step_micros,
        seed,
        sidecar,
        shots_dir,
    })
}

fn write_exclusive(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create {}", path.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Written once, after the run: nothing touches the disk while frames are being measured.
fn write_sidecar(path: &Path, bench: &Bench) -> Result<()> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create sidecar {}", path.display()))?;
    let mut out = BufWriter::with_capacity(1 << 20, file);
    writeln!(out, "{SIDECAR_HEADER}")?;
    for row in &bench.rows {
        let segment = row
            .segment
            .map_or("", |index| bench.recipe.segments[index].name);
        write!(
            out,
            "{},{},{},{},{},{},{},{},{},{},{},",
            row.frame,
            row.step,
            row.virtual_micros,
            row.tick,
            segment,
            row.particles,
            row.entities,
            row.packets_remaining,
            u8::from(row.in_game),
            row.attacks,
            row.hit,
        )?;
        match row.pose {
            Some((yaw, pitch, pos)) => {
                write!(out, "{yaw},{pitch},{},{},{},", pos.x, pos.y, pos.z)?;
            }
            None => write!(out, ",,,,,")?,
        }
        writeln!(out, "{}", u8::from(row.shot))?;
    }
    out.flush()
        .with_context(|| format!("write sidecar {}", path.display()))?;
    Ok(())
}

fn run(setup: Setup) -> Result<u8> {
    let Setup {
        config,
        recipe,
        recipe_path,
        step_micros,
        seed,
        sidecar,
        shots_dir,
    } = setup;
    let pin_clock = animation_clock_pinned();
    let skin_downloads = skin_downloads_allowed();
    eprintln!(
        "skywars_bench: allocator {}, animation clock {}, skin downloads {}",
        allocator_name(),
        if pin_clock { "lockstep" } else { "live" },
        if skin_downloads { "on" } else { "off" }
    );
    if let Some(dir) = &shots_dir {
        if !recipe.screenshots.is_empty() {
            fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        }
    }
    let mut client = Client::new_deferred(config).context("Initializing game")?;
    // Offline and deterministic: no skin or cape is ever fetched (see `skin_downloads_allowed`).
    client.set_skin_downloads(skin_downloads);
    // Seeded streams (study §2.5). The density is re-applied at every spawn site anyway; set it
    // here too so the replacement can never run one spawn at the constructor's `All`.
    client.particles = ParticleSystem::with_seed(seed ^ PARTICLE_SEED_SALT);
    client
        .particles
        .set_density(ParticleDensity::from_raw(client.settings.particles));
    let shell = ShellConfig {
        title: "mc-rs SkyWars lockstep bench".to_owned(),
        width: client.config.width,
        height: client.config.height,
        max_frames: client.config.max_frames,
        ..ShellConfig::default()
    };
    let settings = client.settings.clone();
    let total_steps = recipe.total_ticks() * recipe.k;
    let reserved = usize::try_from(total_steps.min(MAX_RESERVED_ROWS)).unwrap_or(0);
    let mut bench = Bench {
        client,
        recipe,
        step_micros,
        seed,
        total_steps,
        shots_dir,
        pin_clock,
        draw_is_step: false,
        step: 0,
        current_segment: None,
        presents: 0,
        pending_shot: None,
        tracker_seeded: false,
        had_session: false,
        frames_without_session: 0,
        entity_id_checked: false,
        complete: false,
        failure: None,
        replay_summary: None,
        prev_held: HeldSet::default(),
        attacks_issued: 0,
        attacks_dropped: 0,
        approach_ticks: 0,
        aim_missing_ticks: 0,
        rows: Vec::with_capacity(reserved),
        shots_taken: Vec::new(),
    };

    let summary = Shell::run(shell, settings, &mut bench);
    mc_core::profile::dump();
    if bench.client.shutdown_report().is_none() {
        let _ = bench.client.shutdown();
    }
    if let Some(report) = bench.client.shutdown_report() {
        for (step, error) in &report.failures {
            tracing::error!(?step, %error, "shutdown step failed");
        }
    }

    let sidecar_result = write_sidecar(&sidecar, &bench);
    let skipped: Vec<&String> = if bench.shots_dir.is_some() {
        Vec::new()
    } else {
        bench.recipe.screenshots.values().flatten().collect()
    };
    let receipt = json!({
        "schema": "mc-rs-skywars-bench-run/1",
        "recipe": recipe_path.display().to_string(),
        "lockstep_micros": step_micros,
        "k": bench.recipe.k,
        "seed": seed,
        "steps": bench.step,
        "total_steps": bench.total_steps,
        "client_ticks": bench.client.tick_count(),
        "complete": bench.complete,
        "failure": bench.failure,
        "attacks_issued": bench.attacks_issued,
        "attacks_dropped": bench.attacks_dropped,
        "approach_ticks": bench.approach_ticks,
        "aim_missing_ticks": bench.aim_missing_ticks,
        "screenshots_taken": bench.shots_taken,
        "screenshots_skipped": skipped,
        "sidecar": sidecar.display().to_string(),
        "sidecar_rows": bench.rows.len(),
        "replay": bench.replay_summary,
        "allocator": allocator_name(),
        "animation_clock": if bench.pin_clock { "lockstep" } else { "live" },
        "skin_downloads": skin_downloads,
        "skin_worker_started": bench.client.skin_worker_running(),
        "skins_requested_delivered": bench.client.skin_stats(),
    });
    if !skin_downloads && bench.client.skin_worker_running() {
        bench.fail("the skin download worker started in an offline benchmark".to_owned());
    }
    println!("skywars-bench receipt: {receipt}");

    let mut code = 0;
    if let Some(failure) = &bench.failure {
        eprintln!("skywars_bench: self-check failed: {failure}");
        code = EXIT_CHECK;
    }
    match &summary {
        Ok(summary) => {
            eprintln!("mc-rs frame time: {}", summary.frame_time);
            eprintln!("mc-rs input-to-present: {}", summary.input_to_present);
            // Per-phase GPU timestamps, as `mc-client` prints them. Empty
            // (and nothing printed) unless `MC_GPU_TIMING` is set and the adapter has timestamps.
            if !summary.gpu.is_empty() {
                eprint!("{}", summary.gpu.format_table());
            }
            tracing::info!(
                frames = summary.frames,
                ticks = summary.ticks,
                ms_per_frame = summary.frame_time.mean_ms,
                "stopped"
            );
        }
        Err(error) => {
            eprintln!("skywars_bench: the shell failed: {error}");
            if code == 0 {
                code = EXIT_RUNTIME;
            }
        }
    }
    if let Some(error) = bench.client.fatal_error() {
        eprintln!("skywars_bench: client fatal error: {error}");
        if code == 0 {
            code = EXIT_RUNTIME;
        }
    }
    if let Err(error) = &sidecar_result {
        eprintln!("skywars_bench: {error:#}");
        if code == 0 {
            code = EXIT_RUNTIME;
        }
    }
    if code == 0 && !bench.complete {
        eprintln!(
            "skywars_bench: stopped at lockstep frame {} of {} before the scene ended \
             (--frames too small, or the window closed)",
            bench.step, bench.total_steps
        );
        code = EXIT_INCOMPLETE;
    }
    Ok(code)
}

fn main() -> ExitCode {
    mc_core::init_tracing();
    // Drain the background log writer on every exit path, including a panic unwinding out of main.
    let _flush_logs = mc_core::FlushLogsOnDrop;
    let setup = match prepare() {
        Ok(setup) => setup,
        Err(message) => {
            eprintln!("skywars_bench: {message}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    match run(setup) {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("skywars_bench: {error:#}");
            ExitCode::from(EXIT_RUNTIME)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"{"schema": "mc-rs-skywars-scene/1", "k": 50, "lockstep_micros": 1000,
        "seed": 7, "segments": [{"name": "a", "start_tick": 0, "end_tick": 10},
                                 {"name": "b", "start_tick": 10, "end_tick": 30}]}"#;

    fn with_local(local: &str) -> String {
        MINIMAL.trim_end_matches('}').to_owned() + r#", "local": "# + local + "}"
    }

    #[test]
    fn parses_the_documented_example_and_ignores_unknown_keys() {
        let text = r#"{"schema": "mc-rs-skywars-scene/1", "k": 50, "lockstep_micros": 1000,
            "seed": 1, "map": "dwarf_fortress", "segments": [
              {"name": "sw_join", "start_tick": 0, "end_tick": 60},
              {"name": "sw_fight", "start_tick": 60, "end_tick": 1260, "note": "measured"},
              {"name": "sw_tail", "start_tick": 1260, "end_tick": 1360}],
            "screenshots": [[100, "fight_start"], [100, "fight_start_b"]],
            "local": {"entity_id": 1, "extra": true,
              "attack_presses": [[70, 1], [71, 2], [70, 1]],
              "keys": [{"from": 80, "to": 90, "down": ["forward", "sprint", "playerlist"]}],
              "hotbar": [[61, 0]],
              "look": [{"from": 0, "to": 60, "fixed": [90.0, 10.0]},
                       {"from": 60, "to": 1260, "aim_entity": 17}],
              "look_track": [[1300, 0.0, 0.0], [1260, 10.0, 5.0]],
              "approach": [{"from": 60, "to": 1260, "entity": 17, "keep_within": 2.6,
                            "sprint": true}]}}"#;
        let recipe = parse_recipe(text).expect("documented example parses");
        assert_eq!(recipe.total_ticks(), 1360);
        assert_eq!(recipe.segment_index(59), Some(0));
        assert_eq!(recipe.segment_index(60), Some(1));
        assert_eq!(recipe.segment_index(1360), None);
        assert_eq!(recipe.screenshots[&100].len(), 2);
        let local = recipe.local.expect("local");
        assert_eq!(local.attacks[&70], 2);
        assert_eq!(local.attacks[&71], 2);
        assert!(local.keys[0].value.contains(HeldKey::PlayerList));
        assert!(!local.keys[0].value.contains(HeldKey::Jump));
        assert_eq!(local.look_track[0].0, 1260, "keyframes are sorted");
        assert!(matches!(local.look[1].value, LookRule::Aim { entity: 17 }));
    }

    #[test]
    fn rejects_schema_cadence_and_tiling_errors() {
        let bad_schema = MINIMAL.replace("scene/1", "scene/2");
        assert!(parse_recipe(&bad_schema).unwrap_err().contains("schema"));
        let bad_k = MINIMAL.replace("\"k\": 50", "\"k\": 20");
        assert!(parse_recipe(&bad_k).unwrap_err().contains("must equal"));
        let gap = MINIMAL.replace("\"start_tick\": 10", "\"start_tick\": 11");
        assert!(parse_recipe(&gap).unwrap_err().contains("contiguously"));
        let late_start = MINIMAL.replace("\"start_tick\": 0", "\"start_tick\": 1");
        assert!(parse_recipe(&late_start).is_err());
    }

    #[test]
    fn rejects_script_errors_the_generator_could_make() {
        for (local, needle) in [
            (
                r#"{"entity_id": 1, "keys": [{"from": 0, "to": 5, "down": ["tab"]}]}"#,
                "unknown key",
            ),
            (r#"{"entity_id": 1, "hotbar": [[3, 9]]}"#, "0..=8"),
            (
                r#"{"entity_id": 1, "attack_presses": [[30, 1]]}"#,
                "not before",
            ),
            (
                r#"{"entity_id": 1, "look": [{"from": 0, "to": 5, "aim_entity": 1}]}"#,
                "own entity",
            ),
            (
                r#"{"entity_id": 1, "look": [{"from": 0, "to": 5, "fixed": [0, 0]},
                    {"from": 4, "to": 8, "aim_entity": 3}]}"#,
                "overlap",
            ),
            (
                r#"{"entity_id": 1, "look_track": [[2, 0, 0], [2, 1, 1]]}"#,
                "duplicate",
            ),
            (
                r#"{"entity_id": 1, "look": [{"from": 0, "to": 5}]}"#,
                "exactly one",
            ),
            (r#"{"attack_presses": []}"#, "entity_id"),
        ] {
            let error = parse_recipe(&with_local(local)).expect_err(local);
            assert!(error.contains(needle), "{local}: {error}");
        }
    }

    #[test]
    fn look_angles_invert_vanilla_look_vector() {
        // Same formula as `interact::look_vector`: x = -sin(yaw)cos(pitch), y = -sin(pitch),
        // z = cos(yaw)cos(pitch).
        for (yaw, pitch) in [
            (0.0_f32, 0.0_f32),
            (90.0, 10.0),
            (-135.0, -30.0),
            (179.0, 45.0),
        ] {
            let (y, p) = (yaw.to_radians() as f64, pitch.to_radians() as f64);
            let look = DVec3::new(-y.sin() * p.cos(), -p.sin(), y.cos() * p.cos());
            let (yaw2, pitch2) = look_angles(look * 3.0);
            assert!(wrap_degrees(yaw2 - yaw).abs() < 1e-3, "{yaw} -> {yaw2}");
            assert!((pitch2 - pitch).abs() < 1e-3, "{pitch} -> {pitch2}");
        }
    }

    #[test]
    fn look_track_interpolates_holds_and_takes_the_short_arc() {
        let track = [(10_u64, 170.0_f32, 0.0_f32), (20, -170.0, 10.0)];
        assert_eq!(track_at(&track, 8.0, 9), None);
        assert_eq!(track_at(&track, 9.5, 10), Some((170.0, 0.0)));
        assert_eq!(track_at(&track, 10.0, 10), Some((170.0, 0.0)));
        let (yaw, pitch) = track_at(&track, 15.0, 15).unwrap();
        assert!(
            (yaw - 180.0).abs() < 1e-4,
            "short arc through 180, got {yaw}"
        );
        assert!((pitch - 5.0).abs() < 1e-4);
        assert_eq!(track_at(&track, 99.0, 99), Some((-170.0, 10.0)));
        assert_eq!(wrap_degrees(190.0), -170.0);
        assert_eq!(wrap_degrees(-180.0), -180.0);
    }

    #[test]
    fn wrapper_flags_are_removed_and_client_flags_pass_through() {
        let args = [
            "--replay",
            "a.mccap",
            "--script",
            "r.json",
            "--lockstepMicros",
            "2000",
            "--seed",
            "-5",
            "--screenshots",
            "--frames",
            "10",
            "-sidecar",
            "s.csv",
        ]
        .map(str::to_owned)
        .to_vec();
        let options = split_args(args).expect("parses");
        assert_eq!(options.script, PathBuf::from("r.json"));
        assert_eq!(options.lockstep_micros, 2000);
        assert_eq!(options.seed, Some(-5));
        assert_eq!(options.sidecar, Some(PathBuf::from("s.csv")));
        assert!(options.screenshots);
        assert_eq!(
            options.client_args,
            ["--replay", "a.mccap", "--frames", "10"]
        );
        assert!(split_args(vec!["--frames".to_owned(), "1".to_owned()]).is_err());
    }
}
