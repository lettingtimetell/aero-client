//! The updater reads release data from the network and its own state files from disk. Each
//! parser is driven here with seeded random damage to valid input: no panic, no unbounded
//! allocation, no super-linear time. A few seconds; for a longer run by hand:
//!
//! ```text
//! UPDATE_FUZZ_ROUNDS=20 UPDATE_FUZZ_SEED=12345 cargo test -p mc-update --test hostile_update_data -- --nocapture
//! ```

use std::panic::AssertUnwindSafe;
use std::time::{Duration, Instant};

use mc_update::github::{asset_name, choose_release, parse_release, release_page_ok};
use mc_update::net::HttpPolicy;
use mc_update::state::UpdateState;
use mc_update::verify::{
    parse_api_digest, parse_public_key, parse_sha256_sidecar, parse_signature,
};
use mc_update::version::parse_release_tag;

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
    b"18446744073709551615",
    b"18446744073709551616",
    b"1e309",
    b"\"v0.0.0\"",
    b"\"v99999999999999999999.0.0\"",
    b"\"v1.2.3-rc.1+build\"",
    b"\"sha256:\"",
    b"\"uploaded\"",
    b"\"https://github.com/\"",
    b"\"http://\"",
    b"\\u0000",
    b"\\ud800",
    b"..",
    b"%2e",
    b"\xC3\xA9",
    b"\xFF",
    b"\"tag_name\"",
    b"\"assets\"",
    b"\"digest\"",
    b"\"size\"",
    b"\"browser_download_url\"",
    b"\"html_url\"",
    b"\"draft\"",
    b"\"prerelease\"",
];

fn mutate(rng: &mut Rng, data: &mut Vec<u8>) {
    for _ in 0..=rng.below(4) {
        let len = data.len();
        let token = TOKENS[rng.below(TOKENS.len())];
        match rng.below(7) {
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
            5 if len > 1 => data.truncate(rng.below(len)),
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

fn fuzz(target: &str, seeds: &[Vec<u8>], rounds: usize, run: impl Fn(&[u8])) {
    let base = env_number("UPDATE_FUZZ_SEED", 0x0005_EED0_0D5E);
    let seed = base
        ^ target
            .bytes()
            .fold(0u64, |h, b| h.rotate_left(5) ^ u64::from(b));
    let rounds = rounds * env_number("UPDATE_FUZZ_ROUNDS", 1).max(1) as usize;
    let mut rng = Rng(seed);
    let started = Instant::now();
    for round in 0..rounds {
        let mut input = seeds[rng.below(seeds.len())].clone();
        mutate(&mut rng, &mut input);
        let began = Instant::now();
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| run(&input)));
        let took = began.elapsed();
        let shown = || {
            input
                .escape_ascii()
                .to_string()
                .chars()
                .take(2000)
                .collect::<String>()
        };
        assert!(
            outcome.is_ok(),
            "{target}: round {round} (seed {seed:#x}) panicked on \"{}\"",
            shown()
        );
        assert!(
            took < Duration::from_secs(5),
            "{target}: round {round} (seed {seed:#x}) took {took:?} on \"{}\"",
            shown()
        );
    }
    eprintln!(
        "{target}: {rounds} rounds, seed {seed:#x}, {:?}",
        started.elapsed()
    );
}

fn release_json() -> Vec<u8> {
    let version = parse_release_tag("v0.3.0").expect("a release tag");
    let zip = asset_name(&version);
    let base = "https://github.com/example/client/releases/download/v0.3.0";
    format!(
        r#"{{"tag_name":"v0.3.0","draft":false,"prerelease":false,
"html_url":"https://github.com/example/client/releases/tag/v0.3.0",
"assets":[{{"name":"{zip}","size":31457280,"browser_download_url":"{base}/{zip}","state":"uploaded",
"digest":"sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}},
{{"name":"{zip}.sha256","size":90,"browser_download_url":"{base}/{zip}.sha256"}},
{{"name":"{zip}.sig","size":88,"browser_download_url":"{base}/{zip}.sig"}}]}}"#
    )
    .into_bytes()
}

#[test]
fn release_data_survives_damage() {
    let current = parse_release_tag("v0.2.0").expect("a release tag");
    let policy = HttpPolicy::github();
    fuzz("release JSON", &[release_json()], 4000, |bytes| {
        if let Ok(release) = parse_release(bytes) {
            let _ = choose_release(&release, &current, &policy, false);
            let _ = choose_release(&release, &current, &policy, true);
        }
    });
}

/// Found by the fuzzer: a release asset URL that does not parse is kept
/// (cut to 160 bytes) for the refusal message, and the cut landed inside a two-byte character,
/// which panicked the updater's thread. Minimized: `https` and eighty U+00E9 (two bytes each).
#[test]
fn a_non_ascii_url_that_does_not_parse_is_refused_not_a_panic() {
    let url = format!("https{}", "\u{e9}".repeat(80));
    assert!(
        !url.is_char_boundary(160),
        "the cut falls inside a character"
    );
    let refused = HttpPolicy::github().check_url(&url);
    assert!(refused.is_err());
}

#[test]
fn verifier_inputs_survive_damage() {
    let seeds = vec![
        b"sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_vec(),
        b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  client.zip\n".to_vec(),
        b"untrusted comment: x\nRWQ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012345678901234==\n"
            .to_vec(),
        b"v1.20.300-beta.4+x".to_vec(),
        b"https://github.com/example/client/releases/tag/v0.3.0".to_vec(),
    ];
    fuzz("verifier inputs", &seeds, 6000, |bytes| {
        let text = String::from_utf8_lossy(bytes);
        let _ = parse_api_digest(&text);
        let _ = parse_sha256_sidecar(bytes, "client.zip");
        let _ = parse_signature(bytes);
        let _ = parse_public_key(&text);
        let _ = parse_release_tag(&text);
        let _ = release_page_ok(&text);
    });
}

#[test]
fn update_state_files_survive_damage() {
    let seed = serde_json::to_vec(&UpdateState::default()).expect("serialize");
    fuzz("update state", &[seed], 4000, |bytes| {
        if let Ok(state) = serde_json::from_slice::<UpdateState>(bytes) {
            let _ = state.may_offer("0.3.0");
            let _ = serde_json::to_string(&state);
        }
    });
}
