//! Two more inputs from outside the client: `options.txt` (a shared game folder carries one) and
//! skin and cape images (a profile's texture, or a cached file). Each is driven with seeded random
//! damage to valid input; PNG damage is re-sealed (each chunk's CRC recomputed) so it reaches the
//! image decoder instead of stopping at a checksum. A few seconds; for a longer run by hand:
//!
//! ```text
//! CLIENT_FILES_FUZZ_ROUNDS=20 CLIENT_FILES_FUZZ_SEED=12345 cargo test -p mc-render --test hostile_client_files -- --nocapture
//! ```

use std::io::Cursor;
use std::panic::AssertUnwindSafe;
use std::time::{Duration, Instant};

use mc_render::entity::skin::{decode_cape, decode_skin, SkinModel};
use mc_render::settings::decode_options_bytes;
use mc_render::GameSettings;

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

const OPTION_TOKENS: &[&[u8]] = &[
    b":",
    b"\n",
    b"\r\n",
    b"\r",
    b"\x00",
    b"\xEF\xBB\xBF",
    b"\xE9",
    b"\xFF",
    b"-",
    b"2147483647",
    b"-2147483648",
    b"2147483648",
    b"NaN",
    b"Infinity",
    b"-Infinity",
    b"1e309",
    b"0.5",
    b"true",
    b"false",
    b"[",
    b"]",
    b",",
    b"\"",
    b"renderDistance",
    b"guiScale",
    b"fov",
    b"lang",
    b"resourcePacks",
    b"key_key.jump",
    b"maxFps",
    b"mipmapLevels",
    b"soundCategory_master",
    b"lastServer",
    b"../",
    b"CON",
    b"incompatibleResourcePacks",
];

fn mutate(rng: &mut Rng, data: &mut Vec<u8>, tokens: &[&[u8]]) {
    for _ in 0..=rng.below(4) {
        let len = data.len();
        let token = tokens[rng.below(tokens.len())];
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
                let times = 1 + rng.below(200);
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

fn fuzz(target: &str, seeds: &[Vec<u8>], rounds: usize, mut run: impl FnMut(&mut Rng, &[u8])) {
    let base = env_number("CLIENT_FILES_FUZZ_SEED", 0x00F1_1E5F_00D5);
    let seed = base
        ^ target
            .bytes()
            .fold(0u64, |h, b| h.rotate_left(5) ^ u64::from(b));
    let rounds = rounds * env_number("CLIENT_FILES_FUZZ_ROUNDS", 1).max(1) as usize;
    let mut rng = Rng(seed);
    let started = Instant::now();
    for round in 0..rounds {
        let input = seeds[rng.below(seeds.len())].clone();
        let began = Instant::now();
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| run(&mut rng, &input)));
        let took = began.elapsed();
        assert!(
            outcome.is_ok(),
            "{target}: round {round} (seed {seed:#x}) panicked"
        );
        assert!(
            took < Duration::from_secs(5),
            "{target}: round {round} (seed {seed:#x}) took {took:?}"
        );
    }
    eprintln!(
        "{target}: {rounds} rounds, seed {seed:#x}, {:?}",
        started.elapsed()
    );
}

#[test]
fn damaged_options_files_load_and_save() {
    let defaults = GameSettings::default().to_options_txt().into_bytes();
    fuzz("options.txt", &[defaults], 3000, |rng, seed| {
        let mut bytes = seed.to_vec();
        mutate(rng, &mut bytes, OPTION_TOKENS);
        let settings = GameSettings::parse(&decode_options_bytes(&bytes));
        // What loads saves again and reads back.
        let saved = settings.to_options_txt();
        let _ = GameSettings::parse(&saved);
    });
}

fn png(width: u32, height: u32) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    image::RgbaImage::from_fn(width, height, |x, y| {
        image::Rgba([(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8, 255])
    })
    .write_to(&mut out, image::ImageFormat::Png)
    .expect("encode");
    out.into_inner()
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Recomputes every chunk's CRC, as far as the chunks parse.
fn reseal_png(bytes: &mut [u8]) {
    let mut at = 8;
    while at + 12 <= bytes.len() {
        let length = u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
        let Some(end) = (at + 8).checked_add(length as usize) else {
            return;
        };
        if end + 4 > bytes.len() {
            return;
        }
        let crc = crc32(&bytes[at + 4..end]);
        bytes[end..end + 4].copy_from_slice(&crc.to_be_bytes());
        at = end + 4;
    }
}

const PNG_TOKENS: &[&[u8]] = &[
    b"IHDR",
    b"IDAT",
    b"IEND",
    b"PLTE",
    b"tRNS",
    b"zTXt",
    b"\x00\x00\x00\x00",
    b"\xFF\xFF\xFF\xFF",
    b"\x00\x00\x40\x00",
    b"\x08\x06",
    b"\x10\x02",
];

#[test]
fn damaged_skins_and_capes_never_panic() {
    let seeds = vec![png(64, 64), png(64, 32), png(22, 17), png(128, 128)];
    fuzz("skins and capes", &seeds, 2000, |rng, seed| {
        let mut bytes = seed.to_vec();
        mutate(rng, &mut bytes, PNG_TOKENS);
        if rng.below(4) != 0 {
            reseal_png(&mut bytes);
        }
        let _ = decode_skin(&bytes, SkinModel::Classic);
        let _ = decode_skin(&bytes, SkinModel::Slim);
        let _ = decode_cape(&bytes);
    });
}
