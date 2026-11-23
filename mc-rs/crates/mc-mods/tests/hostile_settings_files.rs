//! The mod settings file, shared mod profiles and the waypoint file are JSON that players edit,
//! copy and pass around. Each loader is driven here with seeded random damage to a valid file:
//! no panic, no unbounded allocation, no super-linear time, and whatever loads saves again.
//!
//! The rounds are bounded (a few seconds). For a longer run by hand:
//!
//! ```text
//! SETTINGS_FUZZ_ROUNDS=20 SETTINGS_FUZZ_SEED=12345 cargo test -p mc-mods --test hostile_settings_files -- --nocapture
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use mc_mods::mods::waypoints::store::WaypointStore;
use mc_mods::profiles::ProfileEntry;
use mc_mods::settings::{parse_color, ModSettings};

struct Counting;

const REFUSED_ALLOCATION: usize = 1 << 30;

thread_local! {
    static LIVE: Cell<i64> = const { Cell::new(0) };
    static PEAK: Cell<i64> = const { Cell::new(0) };
}

fn grew(bytes: usize) {
    let _ = LIVE.try_with(|live| {
        live.set(live.get() + bytes as i64);
        let _ = PEAK.try_with(|peak| peak.set(peak.get().max(live.get())));
    });
}

fn shrank(bytes: usize) {
    let _ = LIVE.try_with(|live| live.set(live.get() - bytes as i64));
}

// SAFETY: forwards every call to the system allocator unchanged, except that a request over
// `REFUSED_ALLOCATION` gets a null pointer, which every caller must already handle.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > REFUSED_ALLOCATION {
            return std::ptr::null_mut();
        }
        grew(layout.size());
        // SAFETY: the caller's contract is passed through.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() > REFUSED_ALLOCATION {
            return std::ptr::null_mut();
        }
        grew(layout.size());
        // SAFETY: as above.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > REFUSED_ALLOCATION {
            return std::ptr::null_mut();
        }
        grew(new_size);
        shrank(layout.size());
        // SAFETY: as above.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        shrank(layout.size());
        // SAFETY: as above.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

const MIB: i64 = 1 << 20;

fn peak_during<T>(run: impl FnOnce() -> T) -> (T, i64) {
    let before = LIVE.with(Cell::get);
    PEAK.with(|peak| peak.set(before));
    let out = run();
    (out, PEAK.with(Cell::get) - before)
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
}

const TOKENS: &[&[u8]] = &[
    b"{",
    b"}",
    b"[",
    b"]",
    b"\"",
    b":",
    b",",
    b"null",
    b"true",
    b"false",
    b"1e309",
    b"-1",
    b"2147483648",
    b"-2147483649",
    b"9007199254740993",
    b"0.5",
    b"NaN",
    b"\"#FF00FF\"",
    b"\"0xFFFFFFFFF\"",
    b"\"enabled\"",
    b"\"options\"",
    b"\"version\"",
    b"\"mods\"",
    b"\"servers\"",
    b"\"name\"",
    b"\"x\"",
    b"\"y\"",
    b"\"z\"",
    b"\"dim\"",
    b"\"color\"",
    b"\"visible\"",
    b"\"death\"",
    b"\"profile\"",
    b"\"saved\"",
    b"\"\\ud800\"",
    b"\"\\u0000\"",
    b"\xC3\xA9",
    b"\xFF",
    b"\xEF\xBB\xBF",
    b"\"\xC2\xA7k\"",
];

const NUMBERS: &[&[u8]] = &[
    b"0",
    b"-1",
    b"1",
    b"255",
    b"65535",
    b"2147483647",
    b"2147483648",
    b"-2147483648",
    b"4294967296",
    b"9223372036854775807",
    b"18446744073709551616",
    b"1e309",
    b"-1e309",
    b"1e-400",
    b"0.5",
    b"-0",
];

fn mutate(rng: &mut Rng, data: &mut Vec<u8>) {
    for _ in 0..=rng.below(4) {
        let len = data.len();
        let token = TOKENS[rng.below(TOKENS.len())];
        match rng.below(9) {
            0 if len > 0 => {
                let at = rng.below(len);
                data[at] ^= 1 << rng.below(8);
            }
            1 => {
                let at = rng.below(len + 1);
                data.splice(at..at, token.iter().copied());
            }
            2 if len > 0 => {
                let at = rng.below(len);
                let end = (at + 1 + rng.below(16)).min(len);
                data.splice(at..end, token.iter().copied());
            }
            3 if len > 0 => {
                let at = rng.below(len);
                let end = (at + 1 + rng.below(64)).min(len);
                data.drain(at..end);
            }
            4 if len > 0 => {
                let at = rng.below(len);
                let end = (at + 1 + rng.below(256)).min(len);
                let piece = data[at..end].to_vec();
                let to = rng.below(data.len() + 1);
                data.splice(to..to, piece);
            }
            5 => {
                let times = 1 + rng.below(300);
                let at = rng.below(len + 1);
                let run: Vec<u8> = token
                    .iter()
                    .copied()
                    .cycle()
                    .take(token.len() * times)
                    .collect();
                data.splice(at..at, run);
            }
            6 if len > 0 => {
                let from = rng.below(len);
                if let Some(start) = (from..len).find(|&i| data[i].is_ascii_digit()) {
                    let end = (start..len)
                        .find(|&i| !data[i].is_ascii_digit())
                        .unwrap_or(len);
                    let number = NUMBERS[rng.below(NUMBERS.len())];
                    data.splice(start..end, number.iter().copied());
                }
            }
            7 if len > 1 => data.truncate(rng.below(len)),
            _ => {
                let at = rng.below(len + 1);
                let random: Vec<u8> = (0..1 + rng.below(6)).map(|_| rng.next() as u8).collect();
                data.splice(at..at, random);
            }
        }
    }
    data.truncate(256 << 10);
}

fn env_number(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// Runs `rounds` damaged copies of `seeds` through `run` on a 1 MiB stack; fails on a panic, an
/// input slower than 5 s or holding over 256 MiB, printing the input.
fn fuzz(
    target: &'static str,
    seeds: Vec<Vec<u8>>,
    rounds: usize,
    run: impl Fn(&str) + Send + 'static,
) {
    let base = env_number("SETTINGS_FUZZ_SEED", 0x00C0_FFEE_5E77);
    let seed = base
        ^ target
            .bytes()
            .fold(0u64, |h, b| h.rotate_left(5) ^ u64::from(b));
    let rounds = rounds * env_number("SETTINGS_FUZZ_ROUNDS", 1).max(1) as usize;
    let current = Arc::new(Mutex::new(Vec::new()));
    let watched = Arc::clone(&current);
    let (done, finished) = mpsc::channel();
    std::thread::Builder::new()
        .name(target.to_owned())
        .stack_size(1 << 20)
        .spawn(move || {
            let mut rng = Rng(seed);
            let started = Instant::now();
            for round in 0..rounds {
                let mut input = seeds[rng.below(seeds.len())].clone();
                mutate(&mut rng, &mut input);
                if let Ok(mut slot) = current.lock() {
                    slot.clone_from(&input);
                }
                // The loaders read text: lossy, as the files are read.
                let text = String::from_utf8_lossy(&input).into_owned();
                let began = Instant::now();
                let (outcome, peak) =
                    peak_during(|| std::panic::catch_unwind(AssertUnwindSafe(|| run(&text))));
                let took = began.elapsed();
                let shown = || {
                    let shown: String =
                        text.escape_debug().to_string().chars().take(2000).collect();
                    format!("{} bytes: \"{shown}\"", input.len())
                };
                assert!(
                    outcome.is_ok(),
                    "{target}: round {round} (seed {seed:#x}) panicked on {}",
                    shown()
                );
                assert!(
                    took < Duration::from_secs(5),
                    "{target}: round {round} (seed {seed:#x}) took {took:?} on {}",
                    shown()
                );
                assert!(
                    peak < 256 * MIB,
                    "{target}: round {round} (seed {seed:#x}) held {peak} bytes on {}",
                    shown()
                );
            }
            eprintln!(
                "{target}: {rounds} rounds, seed {seed:#x}, {:?}",
                started.elapsed()
            );
            let _ = done.send(());
        })
        .expect("spawn");
    match finished.recv_timeout(Duration::from_secs(600)) {
        Ok(()) => {}
        Err(mpsc::RecvTimeoutError::Timeout) => {
            let input = watched
                .lock()
                .map(|i| {
                    String::from_utf8_lossy(&i)
                        .chars()
                        .take(2000)
                        .collect::<String>()
                })
                .unwrap_or_default();
            panic!("{target}: an input never returned: {input:?}")
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("{target}: the fuzz worker failed"),
    }
}

fn settings_seeds() -> Vec<Vec<u8>> {
    let defaults = ModSettings::new().to_json_string();
    // A second seed with every value made unusual by plain text substitution.
    let altered = defaults
        .replace("true", "false")
        .replace(": 1", ": 2147483647");
    vec![defaults.into_bytes(), altered.into_bytes()]
}

#[test]
fn mod_settings_files_survive_damage() {
    fuzz("mod settings", settings_seeds(), 400, |text| {
        let (settings, _report) = ModSettings::from_json_str(text);
        let _ = settings.to_json_string();
    });
}

#[test]
fn shared_mod_profiles_survive_damage() {
    let mut seeds = settings_seeds();
    for seed in &mut seeds {
        let text = String::from_utf8_lossy(seed).into_owned();
        // A profile is a settings file with a header naming it.
        let with_header = text.replacen(
            '{',
            r#"{"profile":{"name":"Shared \u00a7k setup","saved":1700000000},"#,
            1,
        );
        *seed = with_header.into_bytes();
    }
    fuzz("mod profiles", seeds, 400, |text| {
        let entry = ProfileEntry::parse(Path::new("shared.json"), text, mc_mods::catalog::all());
        if let Some(setup) = &entry.setup {
            let _ = setup.to_json_string();
        }
    });
}

#[test]
fn waypoint_files_survive_damage() {
    let seed = br##"{"version":1,"servers":{"play.example.net:25565":[
        {"name":"Home \u00a7c base","x":120,"y":64,"z":-340,"dim":0,"color":"#FF5555","visible":true},
        {"name":"Death","x":-5,"y":12,"z":7,"dim":-1,"death":true}],
        "other.example.net":[{"name":"End","x":0,"y":60,"z":0,"dim":1,"color":16711935}]}}"##;
    fuzz("waypoints", vec![seed.to_vec()], 3000, |text| {
        if let Ok((store, _skipped)) = WaypointStore::from_json_str(text) {
            let _ = store.to_json_string();
        }
    });
}

#[test]
fn colour_strings_survive_damage() {
    let seeds = vec![
        b"#FF00FF".to_vec(),
        b"0x80FFFFFF".to_vec(),
        b"  #abcdef  ".to_vec(),
    ];
    fuzz("colours", seeds, 4000, |text| {
        let _ = parse_color(text);
    });
}
