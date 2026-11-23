//! The server list file (`servers.dat`, NBT, carried by a shared game folder), the addresses in
//! it, and LAN game announcements (any device on the network can send one) are driven here with
//! seeded random damage to valid input. A few seconds; for a longer run by hand:
//!
//! ```text
//! SERVERS_FUZZ_ROUNDS=20 SERVERS_FUZZ_SEED=12345 cargo test -p mc-ui --test hostile_server_entries -- --nocapture
//! ```

use std::panic::AssertUnwindSafe;
use std::time::{Duration, Instant};

use mc_ui::screens::multiplayer::server_data::{lan_announcement, parse_address};
use mc_ui::screens::multiplayer::{ServerEntry, ServerList};

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
    b"\x00",
    b"\xFF",
    b"\x0A",
    b"\x09",
    b"\x08",
    b"\x7F\xFF",
    b"\xFF\xFF\xFF\xFF",
    b":",
    b"[",
    b"]",
    b"::1",
    b"[::1]:25565",
    b":65536",
    b":-1",
    b":99999999999",
    b"\xC3\xA9",
    b"xn--",
    b"[MOTD]",
    b"[/MOTD]",
    b"[AD]",
    b"[/AD]",
    b".",
    b"..",
    b"%",
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
                let end = (at + 1 + rng.below(32)).min(len);
                data.drain(at..end);
            }
            4 => {
                let times = 1 + rng.below(100);
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
                let random: Vec<u8> = (0..1 + rng.below(8)).map(|_| rng.next() as u8).collect();
                data.splice(at..at, random);
            }
        }
    }
}

fn env_number(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn fuzz(target: &str, seeds: &[Vec<u8>], rounds: usize, run: impl Fn(&[u8])) {
    let base = env_number("SERVERS_FUZZ_SEED", 0x0000_5E2F_E2D5);
    let seed = base
        ^ target
            .bytes()
            .fold(0u64, |h, b| h.rotate_left(5) ^ u64::from(b));
    let rounds = rounds * env_number("SERVERS_FUZZ_ROUNDS", 1).max(1) as usize;
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
                .take(1500)
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

#[test]
fn damaged_server_lists_load_and_save() {
    let mut list = ServerList::new();
    let mut entry = ServerEntry::new("A \u{a7}cserver", "play.example.net:25565");
    entry.icon_base64 = Some("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==".to_owned());
    list.add(entry);
    list.add(ServerEntry::new("IPv6", "[::1]:25566"));
    list.add(ServerEntry::new("IDN", "b\u{fc}cher.example"));
    let bytes = mc_nbt::write::to_vec("", &list.to_nbt()).expect("serialize");
    // Control: the undamaged file reads back whole, so the damage reaches the entries.
    let read = ServerList::read(&bytes).expect("the seed reads");
    assert_eq!(read.entries().len(), 3);
    fuzz("servers.dat", &[bytes], 3000, |bytes| {
        if let Ok(list) = ServerList::read(bytes) {
            for entry in list.entries() {
                let _ = entry.host_and_port();
                let _ = (entry.ping_bars(), entry.version_mismatch());
            }
            let _ = mc_nbt::write::to_vec("", &list.to_nbt());
        }
    });
}

#[test]
fn damaged_addresses_and_lan_announcements_never_panic() {
    let seeds = vec![
        b"play.example.net:25565".to_vec(),
        b"[2001:db8::1]:25566".to_vec(),
        b"[MOTD]A local world[/MOTD][AD]25566[/AD]".to_vec(),
    ];
    fuzz("addresses and LAN", &seeds, 6000, |bytes| {
        let text = String::from_utf8_lossy(bytes);
        let _ = parse_address(&text);
        let _ = lan_announcement(&text, "192.168.0.7");
        let _ = lan_announcement(&text, "fe80::1%3");
    });
}
