//! Integration tests against the real 1.8.9 assets checked in under `MCP-919-main/jars/`.
//!
//! Mirrors `mc-assets`' own `tests/vanilla_assets.rs`: if the data isn't where expected, every
//! test prints a `SKIP:` line and passes rather than failing confusingly. Point `MC_JARS_DIR` at
//! a `jars/` directory to override.
//!
//! This is the "numeric verification, no speakers needed" the S19 section spec asks for: real
//! vanilla `sounds.json` parses into a registry that resolves real events, and the real `.ogg`
//! blobs those events point at decode to sane, real PCM (right sample rate, right channel count,
//! a plausible non-zero duration, and — for at least one clip — actual non-silent audio energy).

use std::path::{Path, PathBuf};

use mc_assets::{ObjectStorePack, ResourceLocation, ResourceManager};
use mc_audio::{ogg, SoundRegistry};

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    // crates/mc-audio -> crates -> mc-rs -> repo root
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn assets_manager() -> Option<ResourceManager> {
    let assets = jars_dir()?.join("assets");
    if !assets.join("indexes").join("1.8.json").is_file() {
        return None;
    }
    let pack = ObjectStorePack::open(&assets, "1.8").ok()?;
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(pack));
    Some(rm)
}

macro_rules! require {
    ($opt:expr, $what:literal) => {
        match $opt {
            Some(v) => v,
            None => {
                eprintln!(
                    "SKIP: {} not found; set MC_JARS_DIR to the jars/ directory",
                    $what
                );
                return;
            }
        }
    };
}

#[test]
fn real_sounds_json_loads_and_resolves_known_events() {
    let rm = require!(assets_manager(), "jars/assets");
    let registry = SoundRegistry::load(&rm).expect("real sounds.json should parse");
    assert!(
        registry.len() > 100,
        "expected hundreds of sound events, got {}",
        registry.len()
    );

    for name in ["random.click", "mob.chicken.step", "dig.stone", "note.harp"] {
        let event = ResourceLocation::minecraft(name);
        assert!(
            registry.contains(&event),
            "{name} should be a registered vanilla event"
        );
        assert!(registry.category_of(&event).is_some());

        let mut rng = mc_math::JavaRandom::new(1);
        let variant = registry
            .pick_variant(&event, &mut rng)
            .unwrap_or_else(|| panic!("{name} should resolve to a real file"));
        assert!(
            rm.contains(&variant.location),
            "{} should exist as an asset",
            variant.location
        );
    }
}

#[test]
fn a_real_resolved_ogg_decodes_to_sane_pcm() {
    let rm = require!(assets_manager(), "jars/assets");
    let registry = SoundRegistry::load(&rm).expect("real sounds.json should parse");

    let event = ResourceLocation::minecraft("random.click");
    let mut rng = mc_math::JavaRandom::new(1);
    let variant = registry
        .pick_variant(&event, &mut rng)
        .expect("random.click resolves");

    let bytes = rm.load(&variant.location).expect("asset bytes load");
    let decoded = ogg::decode_all(&bytes, &variant.location.to_string()).expect("valid ogg");

    assert!(
        decoded.sample_rate == 44100 || decoded.sample_rate == 22050,
        "unexpected sample rate {}",
        decoded.sample_rate
    );
    assert!(
        decoded.channels == 1 || decoded.channels == 2,
        "unexpected channel count {}",
        decoded.channels
    );
    assert!(decoded.num_frames() > 0, "decoded no audio at all");
    // A UI click is short; sanity-bound it rather than pinning an exact length.
    assert!(
        decoded.duration_secs() > 0.0 && decoded.duration_secs() < 5.0,
        "{}",
        decoded.duration_secs()
    );

    // Not silence: real click audio has non-trivial energy, unlike a bug that decodes to all
    // zeros (e.g. a channel/stride mistake) while still reporting a plausible frame count.
    let peak = decoded
        .samples
        .iter()
        .map(|&s| s.unsigned_abs())
        .max()
        .unwrap_or(0);
    assert!(
        peak > 1000,
        "decoded audio looks silent, peak sample = {peak}"
    );
}

#[test]
fn every_resolved_variant_across_common_events_decodes_without_error() {
    let rm = require!(assets_manager(), "jars/assets");
    let registry = SoundRegistry::load(&rm).expect("real sounds.json should parse");

    let events = [
        "random.click",
        "random.pop",
        "random.bow",
        "mob.chicken.step",
        "mob.cow.say",
        "dig.wood",
        "dig.grass",
        "note.harp",
        "note.bd",
    ];
    let mut checked = 0;
    for name in events {
        let event = ResourceLocation::minecraft(name);
        if !registry.contains(&event) {
            continue;
        }
        // Walk every seed 0..20 to touch multiple weighted variants where an event has more
        // than one, rather than only ever exercising whichever file seed 1 happens to pick.
        for seed in 0..20 {
            let mut rng = mc_math::JavaRandom::new(seed);
            let Some(variant) = registry.pick_variant(&event, &mut rng) else {
                continue;
            };
            let bytes = rm
                .load(&variant.location)
                .unwrap_or_else(|e| panic!("{} should load: {e}", variant.location));
            ogg::decode_all(&bytes, &variant.location.to_string())
                .unwrap_or_else(|e| panic!("{} should decode: {e}", variant.location));
        }
        checked += 1;
    }
    assert!(
        checked >= 5,
        "expected at least 5 of the sampled events to be registered, got {checked}"
    );
}

/// The last-page length that spares a full decode at stream start has to agree with the decoder,
/// on every real streamed asset, so nothing audible changes.
#[test]
fn the_last_ogg_page_gives_the_length_of_every_streamed_vanilla_track() {
    let assets = require!(jars_dir(), "jars").join("assets");
    let Ok(index) = std::fs::read_to_string(assets.join("indexes").join("1.8.json")) else {
        eprintln!("SKIP: no 1.8 asset index");
        return;
    };
    let index: serde_json::Value = serde_json::from_str(&index).expect("asset index parses");
    let objects = index["objects"].as_object().expect("asset index objects");
    let mut checked = 0;
    for (name, object) in objects {
        if !(name.starts_with("minecraft/sounds/music/")
            || name.starts_with("minecraft/sounds/records/"))
        {
            continue;
        }
        let hash = object["hash"].as_str().expect("asset hash");
        let path = assets.join("objects").join(&hash[..2]).join(hash);
        let bytes: std::sync::Arc<[u8]> = std::fs::read(&path).expect("asset exists").into();
        let from_page = ogg::frames_from_last_page(&bytes).expect("real asset has a last page");
        let decoded = ogg::count_frames(&bytes, name).expect("decodes");
        assert!(
            from_page == decoded,
            "{name}: last page {from_page}, decoder {decoded}"
        );
        checked += 1;
    }
    assert!(checked >= 20, "only {checked} streamed assets were checked");
}
