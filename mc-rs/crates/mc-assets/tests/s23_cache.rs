//! S23: the on-disk bake cache, end to end against the real 1.8.9 jar.
//!
//! "Startup-time optimisation (atlas + model baking cache on disk)".
//! `src/cache.rs`'s own tests cover the codec, the key and every rejection path with synthetic
//! data. This file covers the thing those cannot: that a cache written from a **real**
//! `bake_all` and read back reconstructs the same bake, and that installing it publishes a
//! complete registry rather than one with a hole in it.
//!
//! # The bar
//!
//! A stale or lossy cache that silently serves the wrong models is far worse than a slow start.
//! So the assertions here are equality over the *whole* bake — every one of the 1251 model
//! states, every sprite, every item model and the atlas byte for byte — not a spot check on
//! stone. `model_eq` compares `BakedModel` field by field, and its field list is exhaustive by
//! construction (see `cache::model_eq`), so a field added to `BakedModel` later fails to compile
//! rather than being silently dropped from the cache.

//! **These tests do not skip.** They used to print `skip: no vanilla jar in this checkout` and
//! `return`, which *passes*, and `cargo test` swallows a passing test's output — so on a
//! checkout without the jar this file reported three green tests of the on-disk bake cache
//! having cached nothing. They now panic naming the path. Its old locator also honoured no
//! `MC_JARS_DIR`; it uses `tests/common/mod.rs` now.

mod common;

use std::path::Path;

use mc_assets::cache::{self, BakeInputs, BakedSnapshot, CacheMiss, CacheRead};
use mc_assets::{bake_all, ResourceManager, ZipResourcePack};

const MAX_ATLAS: u32 = 2048;

fn vanilla_stack(jar: &Path) -> ResourceManager {
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(
        ZipResourcePack::open(jar).expect("open 1.8.9.jar"),
    ));
    rm
}

fn inputs_for(jar: &Path) -> BakeInputs {
    let mut inputs = BakeInputs::new(MAX_ATLAS);
    inputs.push_path(jar);
    assert!(inputs.is_stampable(), "the real jar must be stampable");
    inputs
}

/// The load-bearing test of the cache: a real bake, written and read back, is the same bake.
#[test]
fn a_real_bake_survives_a_round_trip_through_the_cache_without_losing_anything() {
    let jar = common::vanilla_jar("s23_cache (jars/versions/1.8.9/1.8.9.jar)");
    let dir = tempfile::tempdir().expect("scratch dir");
    let path = dir.path().join("baked.bin");
    let rm = vanilla_stack(&jar);
    let inputs = inputs_for(&jar);

    let (baked, report) = bake_all(&rm, MAX_ATLAS).expect("bake the real jar");
    assert!(report.is_clean(), "{:?}", report.failures);
    assert!(report.model_states_considered > 1_000);

    let snapshot = BakedSnapshot::of(&baked);
    // The snapshot really is the whole bake, not a sample.
    assert_eq!(snapshot.models.len(), mc_data::STATE_COUNT);
    assert!(snapshot.sprites.len() > 300, "{}", snapshot.sprites.len());
    assert!(
        snapshot.item_models.len() > 300,
        "{}",
        snapshot.item_models.len()
    );
    assert!(!snapshot.weighted.is_empty(), "1.8.9 has weighted variants");
    assert_eq!(
        snapshot.atlas.texture.len(),
        (snapshot.atlas.w * snapshot.atlas.h * 4) as usize
    );

    cache::save(&path, &inputs, &snapshot).expect("save");
    let back = match cache::load(&path, &inputs) {
        CacheRead::Hit(s) => *s,
        CacheRead::Miss(m) => panic!("the cache we just wrote must load: {m:?}"),
    };
    assert_eq!(
        back, snapshot,
        "the round trip must preserve every model, sprite, weighted variant and item model"
    );

    // And rebuilding a `BakedAssets` from it and re-snapshotting gives the same thing again, so
    // `from_parts` is genuinely the inverse of `to_parts` rather than merely lossy in a way the
    // codec happens to hide.
    let rebuilt = BakedSnapshot::of(&back.clone().into_assets());
    assert_eq!(
        rebuilt, snapshot,
        "from_parts must be the inverse of to_parts"
    );

    // Control: the same file read against a *different* key must not load, so the hit above is
    // the key check passing rather than the key check being absent.
    let mut other = BakeInputs::new(MAX_ATLAS);
    other.push_opaque("some other pack", b"not the jar");
    assert!(matches!(
        cache::load(&path, &other).miss(),
        Some(CacheMiss::Inputs { .. })
    ));

    println!(
        "cache file: {} bytes for {} states, {} sprites, {} item models, {}x{} atlas",
        std::fs::metadata(&path).expect("stat").len(),
        snapshot.models.len(),
        snapshot.sprites.len(),
        snapshot.item_models.len(),
        snapshot.atlas.w,
        snapshot.atlas.h
    );
}

/// A bake that came out of the cache must be **installable**, and everything reached through
/// S09's frozen entry points afterwards — `model_for`, `model_for_at`, `sprite` — must answer
/// exactly what a fresh bake would.
///
/// (`item_model_for` is deliberately *not* asserted here: `bake_all` publishes item models on its
/// own way out, first-wins, so by the time this test installs anything the item registry already
/// holds the fresh bake's copy and the assertion would pass whatever `install` did. That claim
/// gets its own binary, `s23_install.rs`, which never calls `bake_all`. What *is* asserted here
/// is that the cache carried the item models losslessly, compared directly rather than through
/// the global.)
///
/// This is the one test in this binary that touches the process-global registry, so it owns it.
#[test]
fn installing_a_cached_bake_publishes_exactly_what_a_fresh_bake_would() {
    let jar = common::vanilla_jar("s23_cache (jars/versions/1.8.9/1.8.9.jar)");
    let dir = tempfile::tempdir().expect("scratch dir");
    let path = dir.path().join("baked.bin");
    let rm = vanilla_stack(&jar);
    let inputs = inputs_for(&jar);

    let (fresh, report) = bake_all(&rm, MAX_ATLAS).expect("bake");
    assert!(report.is_clean());
    cache::save(&path, &inputs, &BakedSnapshot::of(&fresh)).expect("save");

    // What a fresh bake holds, kept for comparison before anything is installed.
    let (fresh_atlas, fresh_sprites, fresh_models, fresh_weighted, fresh_items) = fresh.to_parts();

    // `bake_all` publishes item models on its way out, which would mask the thing this test is
    // about. Assert the state we are actually testing from: the registry is untouched.
    assert_eq!(
        mc_assets::model::installed_generation(),
        0,
        "this binary must own the globals; \
         `bake_all` above must not have installed the block registry"
    );

    let cached = match cache::load(&path, &inputs) {
        CacheRead::Hit(s) => s.into_assets(),
        CacheRead::Miss(m) => panic!("unexpected miss: {m:?}"),
    };

    // Item models: compared on the reconstructed `BakedAssets` itself, not through the global,
    // for the reason in this test's doc comment.
    let (_, _, _, _, cached_items) = cached.to_parts();
    assert_eq!(cached_items.len(), fresh_items.len());
    for ((cid, cm), (fid, fm)) in cached_items.iter().zip(&fresh_items) {
        assert_eq!(cid, fid);
        assert!(cache::model_eq(cm, fm), "item model {cid} differs");
    }
    assert!(
        fresh_items.iter().any(|(_, m)| !m.general_quads.is_empty()),
        "control: at least one item model must have geometry"
    );

    drop(fresh);
    mc_assets::install(cached);
    assert_eq!(mc_assets::model::installed_generation(), 1);

    // Every legal block state's published model must equal the fresh bake's.
    let mut checked = 0usize;
    let mut non_empty = 0usize;
    for state in mc_data::legal_states() {
        let published = mc_assets::model_for(state);
        let expected = &fresh_models[state.id as usize];
        assert!(
            cache::model_eq(published, expected),
            "state {} differs after a cache round trip",
            state.id
        );
        checked += 1;
        if !published.is_empty() {
            non_empty += 1;
        }
    }
    assert!(checked > 1_000, "{checked}");
    assert!(
        non_empty > 1_000,
        "the published registry is nearly empty ({non_empty} of {checked}); \
         the comparison above would be vacuous"
    );

    // Sprites, including the four fluid textures S09b stitched, which are looked up by name.
    for sprite in &fresh_sprites {
        assert_eq!(
            mc_assets::sprite(&sprite.location),
            sprite,
            "sprite {} differs",
            sprite.location
        );
    }
    for name in [
        "textures/blocks/water_still",
        "textures/blocks/lava_still",
        "textures/blocks/stone",
    ] {
        let loc = mc_assets::ResourceLocation::minecraft(name);
        assert!(
            mc_assets::sprite(&loc).uv != mc_assets::sprite::UvRect::FULL,
            "{name} came back as the placeholder"
        );
    }

    // Weighted variants, reached through the position-aware selector.
    assert!(!fresh_weighted.is_empty());
    for (state_id, candidates) in &fresh_weighted {
        let state = mc_data::BlockState { id: *state_id };
        let mut seen_any = false;
        for x in 0..8i32 {
            for z in 0..8i32 {
                let picked = mc_assets::model_for_at(state, [x, 64, z]);
                assert!(
                    candidates.iter().any(|(_, m)| cache::model_eq(picked, m)),
                    "state {state_id} picked a weighted variant the cache does not hold"
                );
                seen_any = true;
            }
        }
        assert!(seen_any);
    }

    // Stone is the cheapest proof the published registry is the real thing rather than a
    // structurally-valid empty one.
    assert!(
        !mc_assets::model_for(mc_data::BlockState { id: 1 << 4 }).is_empty(),
        "stone must have geometry after installing from the cache"
    );
    assert_eq!(
        fresh_atlas.texture.len(),
        (fresh_atlas.w * fresh_atlas.h * 4) as usize
    );
    println!(
        "installed from cache: {checked} states ({non_empty} with geometry), \
         {} sprites, {} item models, {} weighted",
        fresh_sprites.len(),
        fresh_items.len(),
        fresh_weighted.len()
    );
}

/// The invalidation rule that matters most in the real client: the jar itself changing. Written
/// with a *copy* of the jar so the real one is never touched.
#[test]
fn a_changed_pack_file_invalidates_the_cache_and_a_matching_one_does_not() {
    let jar = common::vanilla_jar("s23_cache (jars/versions/1.8.9/1.8.9.jar)");
    let dir = tempfile::tempdir().expect("scratch dir");
    let copy = dir.path().join("1.8.9.jar");
    std::fs::copy(&jar, &copy).expect("copy the jar");
    let path = dir.path().join("baked.bin");

    let rm = vanilla_stack(&copy);
    let (baked, report) = bake_all(&rm, MAX_ATLAS).expect("bake");
    assert!(report.is_clean());
    let snapshot = BakedSnapshot::of(&baked);

    let inputs = inputs_for(&copy);
    cache::save(&path, &inputs, &snapshot).expect("save");
    assert!(
        cache::load(&path, &inputs_for(&copy)).hit().is_some(),
        "control: an unchanged pack must hit"
    );

    // Append a byte: a different length is caught even by the sampled stamp, and at 8 MiB the
    // default is a full hash anyway.
    let mut bytes = std::fs::read(&copy).expect("read");
    bytes.push(0);
    std::fs::write(&copy, &bytes).expect("write");
    assert!(matches!(
        cache::load(&path, &inputs_for(&copy)).miss(),
        Some(CacheMiss::Inputs { .. })
    ));

    // Restoring the *content* does **not** restore the hit, and that is deliberate: the stamp
    // carries the modification time as well as the content hash, so touching a pack at all
    // forces a re-bake. The conservative direction is the safe one — a needless 40 ms bake
    // against a silently wrong model set — so this is asserted rather than left to be discovered.
    std::fs::write(&copy, &bytes[..bytes.len() - 1]).expect("restore");
    assert!(
        cache::load(&path, &inputs_for(&copy)).hit().is_none(),
        "a rewritten pack must miss even when its bytes are back, because its mtime moved"
    );

    // A second pack stacked on top is a different question and must also miss...
    let restored = inputs_for(&copy);
    cache::save(&path, &restored, &snapshot).expect("re-save against the restored file");
    let mut stacked = inputs_for(&copy);
    stacked.push_opaque("user-pack", b"classic 32x");
    assert!(matches!(
        cache::load(&path, &stacked).miss(),
        Some(CacheMiss::Inputs { .. })
    ));

    // ...and the control that makes that miss mean something: the same file, keyed on the same
    // stack, still hits.
    assert!(
        cache::load(&path, &inputs_for(&copy)).hit().is_some(),
        "control: the unchanged one-pack stack must hit"
    );
}

// -------------------------------------------------------- the negative control

/// The control on "these tests do not skip". See `vanilla_assets.rs`' twin for the reasoning.
#[test]
fn the_cache_suite_fails_loudly_when_the_jar_is_absent() {
    let missing = common::definitely_missing();
    let err = common::locate_jar(&missing).expect_err("a missing directory must not yield a jar");
    assert!(err.contains("is not a directory"), "{err}");

    let empty = tempfile::tempdir().expect("tempdir");
    let err =
        common::locate_jar(empty.path()).expect_err("an empty directory must not yield a jar");
    assert!(err.ends_with("1.8.9.jar is not a file"), "{err}");

    let jars = common::jars_dir();
    assert!(
        common::locate_jar(&jars).is_ok(),
        "the real jar must be found under {}",
        jars.display()
    );
}
