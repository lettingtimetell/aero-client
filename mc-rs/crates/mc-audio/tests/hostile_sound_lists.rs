//! A resource pack's `sounds.json` (players share packs, and a server can offer one; a server
//! also chooses which sound events play) is driven here with a hostile event graph and with
//! seeded random damage to a valid file. Loading, prewarming and picking must stay bounded.
//!
//! A few seconds; for a longer run by hand:
//!
//! ```text
//! SOUNDS_FUZZ_ROUNDS=20 SOUNDS_FUZZ_SEED=12345 cargo test -p mc-audio --test hostile_sound_lists -- --nocapture
//! ```

use std::collections::BTreeMap;
use std::io::{Cursor, Read};
use std::panic::AssertUnwindSafe;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use mc_assets::{ResourceLocation, ResourceManager, ResourcePack};
use mc_audio::SoundRegistry;
use mc_math::JavaRandom;

#[derive(Clone, Default)]
struct MemPack {
    files: BTreeMap<(String, String), Vec<u8>>,
}

impl MemPack {
    fn with(mut self, path: &str, bytes: impl Into<Vec<u8>>) -> Self {
        self.files
            .insert(("minecraft".to_owned(), path.to_owned()), bytes.into());
        self
    }

    fn key(l: &ResourceLocation) -> (String, String) {
        (l.namespace.clone(), l.path.clone())
    }
}

impl ResourcePack for MemPack {
    fn open(&self, l: &ResourceLocation) -> Option<Box<dyn Read>> {
        self.files
            .get(&Self::key(l))
            .map(|bytes| Box::new(Cursor::new(bytes.clone())) as Box<dyn Read>)
    }

    fn contains(&self, l: &ResourceLocation) -> bool {
        self.files.contains_key(&Self::key(l))
    }

    fn lists(&self, l: &ResourceLocation) -> Option<bool> {
        Some(self.contains(l))
    }

    fn namespaces(&self) -> Vec<String> {
        let mut namespaces: Vec<String> = self.files.keys().map(|(ns, _)| ns.clone()).collect();
        namespaces.dedup();
        namespaces
    }

    fn list_files(&self, namespace: &str, dir_prefix: &str) -> Vec<String> {
        self.files
            .keys()
            .filter(|(ns, path)| ns == namespace && path.starts_with(dir_prefix))
            .map(|(_, path)| path.clone())
            .collect()
    }
}

fn registry(sounds_json: &[u8]) -> SoundRegistry {
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(
        MemPack::default()
            .with("sounds.json", sounds_json)
            .with("sounds/a.ogg", b"OggS".to_vec())
            .with("sounds/b.ogg", b"OggS".to_vec()),
    ));
    SoundRegistry::load(&rm).expect("load never fails")
}

/// Runs `work` on its own 1 MiB-stack thread; `None` if it has not returned after `limit`.
fn within<T: Send + 'static>(
    limit: Duration,
    work: impl FnOnce() -> T + Send + 'static,
) -> Option<T> {
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(move || {
            let _ = tx.send(work());
        })
        .expect("spawn");
    rx.recv_timeout(limit).ok()
}

/// One event whose ten entries all name itself: `"type": "event"` indirection with a fan-out.
fn fanned_cycle(fan_out: usize) -> Vec<u8> {
    let entries = vec![r#"{"name":"loop","type":"event"}"#; fan_out].join(",");
    format!(
        r#"{{"loop":{{"category":"master","sounds":[{entries}]}},
"click":{{"category":"master","sounds":["a",{{"name":"loop","type":"event"}}]}}}}"#
    )
    .into_bytes()
}

/// Indirection was capped at 16 levels, but each level
/// summed the weight of every entry again: one event naming itself ten times asked for 10^17
/// steps, the moment the sounds were prewarmed after the pack loaded, and again at each play of
/// it. (Vanilla recurses without a bound and overflows its stack on any cycle.)
#[test]
fn a_fanned_out_event_cycle_loads_and_plays_in_bounded_time() {
    let json = fanned_cycle(10);
    let outcome = within(Duration::from_secs(10), move || {
        let registry = registry(&json);
        let began = Instant::now();
        let files = registry.static_files_where(|_| true);
        let prewarm = began.elapsed();
        let mut rng = JavaRandom::new(7);
        let began = Instant::now();
        let looped = registry.pick_variant(&ResourceLocation::minecraft("loop"), &mut rng);
        let mut picks = Vec::new();
        for _ in 0..20 {
            picks.push(registry.pick_variant(&ResourceLocation::minecraft("click"), &mut rng));
        }
        (files, prewarm, looped, picks, began.elapsed())
    });
    let (files, prewarm, looped, picks, picking) = outcome.expect("finished within 10 s");
    eprintln!("prewarm list {prewarm:?}, 21 picks {picking:?}");
    assert_eq!(
        files,
        vec![ResourceLocation::minecraft("sounds/a.ogg")],
        "the prewarm list still finds the real file"
    );
    assert!(
        looped.is_none(),
        "a cycle of nothing but itself plays nothing"
    );
    assert!(
        picks
            .iter()
            .flatten()
            .all(|v| v.location == ResourceLocation::minecraft("sounds/a.ogg")),
        "{picks:?}"
    );
    assert!(
        picks.iter().any(Option::is_some),
        "the real file still plays"
    );
}

/// An ordinary chain resolves as before (vanilla's deepest real chain is two levels).
#[test]
fn ordinary_event_indirection_is_unchanged() {
    let json = br#"{"a.event":{"category":"block","sounds":[{"name":"b.event","type":"event","weight":3},"b"]},
"b.event":{"category":"block","sounds":["a",{"name":"c.event","type":"event"}]},
"c.event":{"category":"block","sounds":[{"name":"b","weight":2}]}}"#;
    let registry = registry(json);
    let mut rng = JavaRandom::new(1);
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..200 {
        let picked = registry
            .pick_variant(&ResourceLocation::minecraft("a.event"), &mut rng)
            .expect("resolves");
        seen.insert(picked.location.path);
    }
    assert_eq!(
        seen.into_iter().collect::<Vec<_>>(),
        vec!["sounds/a.ogg".to_owned(), "sounds/b.ogg".to_owned()]
    );
    let mut files = registry.static_files_where(|_| true);
    files.sort_by(|a, b| a.path.cmp(&b.path));
    assert_eq!(
        files,
        vec![
            ResourceLocation::minecraft("sounds/a.ogg"),
            ResourceLocation::minecraft("sounds/b.ogg")
        ]
    );
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
    b"-1",
    b"0",
    b"2147483648",
    b"4294967296",
    b"1e309",
    b"0.5",
    b"\"event\"",
    b"\"type\"",
    b"\"weight\"",
    b"\"name\"",
    b"\"stream\"",
    b"\"volume\"",
    b"\"pitch\"",
    b"\"replace\"",
    b"\"category\"",
    b"\"master\"",
    b"\"music\"",
    b"\"loop\"",
    b"\"click\"",
    b"\"a\"",
    b"{\"name\":\"loop\",\"type\":\"event\"},",
    b"\\u0000",
    b"\xC3\xA9",
    b"\xFF",
];

fn mutate(rng: &mut Rng, data: &mut Vec<u8>) {
    for _ in 0..=rng.below(4) {
        let len = data.len();
        let token = TOKENS[rng.below(TOKENS.len())];
        match rng.below(6) {
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
            4 => {
                let times = 1 + rng.below(60);
                let at = rng.below(len + 1);
                let run: Vec<u8> = token
                    .iter()
                    .copied()
                    .cycle()
                    .take(token.len() * times)
                    .collect();
                data.splice(at..at, run);
            }
            _ if len > 1 => data.truncate(rng.below(len)),
            _ => data.push(rng.next() as u8),
        }
    }
    data.truncate(64 << 10);
}

fn env_number(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[test]
fn damaged_sound_lists_load_prewarm_and_play_in_bounded_time() {
    let seeds = [
        fanned_cycle(3),
        br#"{"a.event":{"category":"block","replace":true,"sounds":[{"name":"b.event","type":"event","weight":3},"b",{"name":"a","volume":0.5,"pitch":2,"stream":true}]},
"b.event":{"category":"player","sounds":["a",{"name":"a.event","type":"event"}]}}"#
            .to_vec(),
    ];
    let rounds = 1500 * env_number("SOUNDS_FUZZ_ROUNDS", 1).max(1) as usize;
    let seed = env_number("SOUNDS_FUZZ_SEED", 0x0050_0DD5);
    let done = within(Duration::from_secs(300), move || {
        let mut rng = Rng(seed);
        let started = Instant::now();
        for round in 0..rounds {
            let mut input = seeds[rng.below(seeds.len())].clone();
            mutate(&mut rng, &mut input);
            let began = Instant::now();
            let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
                let registry = registry(&input);
                let _ = registry.static_files_where(|_| true);
                let mut java = JavaRandom::new(round as i64);
                let events: Vec<ResourceLocation> = registry
                    .events()
                    .map(|(event, _)| event.clone())
                    .take(50)
                    .collect();
                for event in events {
                    let _ = registry.pick_variant(&event, &mut java);
                }
            }));
            let took = began.elapsed();
            let shown = || {
                input
                    .escape_ascii()
                    .to_string()
                    .chars()
                    .take(1500)
                    .collect::<String>()
            };
            assert!(
                outcome.is_ok(),
                "round {round} (seed {seed:#x}) panicked on \"{}\"",
                shown()
            );
            assert!(
                took < Duration::from_secs(5),
                "round {round} (seed {seed:#x}) took {took:?} on \"{}\"",
                shown()
            );
        }
        eprintln!(
            "sound lists: {rounds} rounds, seed {seed:#x}, {:?}",
            started.elapsed()
        );
    });
    assert!(done.is_some(), "an input never returned");
}
