//! S21 → S09: a second `bake_all` must be able to replace the first.
//!
//! java: `ModelManager.onResourceManagerReload` re-runs `ModelBakery` and overwrites
//! `modelRegistry` on every `refreshResources`. `mc_assets::install` was a `OnceLock`, so this
//! client could not do it; `mc_assets::model::reinstall` is the replacement, and `install` keeps
//! its first-wins behaviour untouched.
//!
//! **This file owns the process globals.** `model_for`/`sprite`/`item_model_for` read a
//! process-wide registry, so the assertions below are order-dependent and live in their own test
//! binary. Rust runs the tests in one binary on several threads, so everything that touches the
//! registry is sequenced inside a single `#[test]`. The one other `#[test]` here — the
//! jar-absent negative control — touches no global and only stats paths, so it is safe to run
//! alongside.
//!
//! **This test does not skip.** It used to print `skip: no vanilla jar in this checkout` and
//! `return`, which *passes*, and `cargo test` swallows a passing test's output — so on a
//! checkout without the jar it reported a green test of `install`/`reinstall` having installed
//! nothing. It now panics naming the path. Its old locator also hard-coded three candidate
//! paths and honoured no `MC_JARS_DIR`; it uses `tests/common/mod.rs` now.

mod common;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use mc_assets::{
    bake_all, BakedAssets, BakedModel, ResourceLocation, ResourceManager, ZipResourcePack,
};

const MAX_ATLAS: u32 = 2048;

fn vanilla_stack() -> ResourceManager {
    let jar = common::vanilla_jar("s21_reinstall (jars/versions/1.8.9/1.8.9.jar)");
    let mut manager = ResourceManager::new();
    manager.push_pack(Box::new(
        ZipResourcePack::open(&jar).expect("open 1.8.9.jar"),
    ));
    manager
}

/// A byte-level fingerprint of one baked model: every vertex position, every UV, the facing, the
/// tint index and the shade flag of every quad in every bucket.
///
/// Floats are hashed by their **bit pattern**, so this is a byte comparison and not a tolerance:
/// a UV that moved by one ULP fails it.
///
/// **Quads are sorted within their bucket before hashing.** That used to be load-bearing:
/// `bake_resolved_model` iterated `RawElement::faces`, a `HashMap<String, RawFace>`, straight off
/// the map, and two `HashMap`s built in the same process get different `RandomState` seeds, so
/// two runs of `bake_all` on identical input could emit the same *set* of quads in a different
/// order. The `S21 → S09` request closed that: faces are now sorted by `Facing` — vanilla's own
/// wire ordinal order — before baking, so the sequence is a property of the input, not of two
/// independently-seeded maps, and this function's own sort no longer changes what it hashes. It
/// stays anyway, as a redundant safety net that costs nothing here. Draw order inside a bucket
/// was never observable regardless — the depth test settles opaque geometry and `terrain::sort`
/// orders translucent quads by distance — which is why `s11_render.rs`'s committed snapshot has
/// always passed. The rendered-frame proof lives in `mc-render`'s `s21_pack_reload.rs`.
fn fingerprint(model: &BakedModel) -> u64 {
    let mut hasher = DefaultHasher::new();
    for bucket in 0..7usize {
        let quads = if bucket < 6 {
            &model.face_quads[bucket]
        } else {
            &model.general_quads
        };
        bucket.hash(&mut hasher);
        quads.len().hash(&mut hasher);
        let mut per_quad: Vec<u64> = quads
            .iter()
            .map(|quad| {
                let mut quad_hasher = DefaultHasher::new();
                for vertex in &quad.verts {
                    for value in vertex.pos {
                        value.to_bits().hash(&mut quad_hasher);
                    }
                    for value in vertex.uv {
                        value.to_bits().hash(&mut quad_hasher);
                    }
                }
                quad.face.map(mc_math::Facing::index).hash(&mut quad_hasher);
                quad.tint_index.hash(&mut quad_hasher);
                quad.shade.hash(&mut quad_hasher);
                quad_hasher.finish()
            })
            .collect();
        per_quad.sort_unstable();
        per_quad.hash(&mut hasher);
    }
    model.ambient_occlusion.hash(&mut hasher);
    hasher.finish()
}

/// The fingerprint of the whole installed registry: every legal block state, in id order.
fn registry_fingerprint() -> u64 {
    let mut hasher = DefaultHasher::new();
    for id in 0..u16::try_from(mc_data::STATE_COUNT).unwrap_or(u16::MAX) {
        fingerprint(mc_assets::model_for(mc_data::BlockState { id })).hash(&mut hasher);
    }
    hasher.finish()
}

/// A pack that repaints `blocks/stone.png` a flat, unmistakable colour.
fn repaint_stone(dir: &std::path::Path, rgb: [u8; 3]) -> std::path::PathBuf {
    let pack = dir.join("Repaint");
    let textures = pack.join("assets/minecraft/textures/blocks");
    std::fs::create_dir_all(&textures).expect("mkdir");
    std::fs::write(
        pack.join("pack.mcmeta"),
        br#"{"pack":{"pack_format":1,"description":"s21"}}"#,
    )
    .expect("mcmeta");
    let mut image = image::RgbaImage::new(16, 16);
    for pixel in image.pixels_mut() {
        *pixel = image::Rgba([rgb[0], rgb[1], rgb[2], 255]);
    }
    image
        .save(textures.join("stone.png"))
        .expect("write stone.png");
    pack
}

/// How many texels of `rgb` an atlas sheet contains. The repainted stone is 16×16 of one flat
/// colour, so "at least 256" is the signal and "zero" is the vanilla control.
fn texels_of(assets: &BakedAssets, rgb: [u8; 3]) -> usize {
    assets
        .atlas
        .texture
        .chunks_exact(4)
        .filter(|texel| texel[0] == rgb[0] && texel[1] == rgb[1] && texel[2] == rgb[2])
        .count()
}

/// The stone sprite's first source texel, read through the **installed** sprite table — the same
/// lookup `BlockFluidRenderer` and the crack overlay do by name.
fn installed_stone_texel() -> [u8; 3] {
    // `sprite`'s elided output lifetime is tied to its argument, so the location has to outlive
    // the borrow.
    let location = ResourceLocation::minecraft("textures/blocks/stone");
    let sprite = mc_assets::sprite(&location);
    let frame = &sprite.frames[0];
    [frame.rgba[0], frame.rgba[1], frame.rgba[2]]
}

#[test]
fn install_is_first_wins_and_reinstall_swaps_the_whole_registry() {
    let vanilla = vanilla_stack();

    // ---- 1. a cold install, exactly as every caller since S09 has done it ----
    assert_eq!(
        mc_assets::model::installed_generation(),
        0,
        "this binary must own the globals"
    );
    let (first, report) = bake_all(&vanilla, MAX_ATLAS).expect("bake the real jar");
    let first_atlas = (first.atlas.w, first.atlas.h);
    mc_assets::install(first);
    assert_eq!(mc_assets::model::installed_generation(), 1);
    assert!(report.model_states_considered > 1_000);

    let baseline = registry_fingerprint();
    let stone = mc_assets::model_for(mc_data::BlockState { id: 1 << 4 });
    assert!(!stone.is_empty(), "stone must have geometry");

    // ---- 2. `install` a second time must be refused, byte for byte ----
    let (second, _) = bake_all(&vanilla, MAX_ATLAS).expect("bake again");
    mc_assets::install(second);
    assert_eq!(
        mc_assets::model::installed_generation(),
        1,
        "install() is first-wins; this is the behaviour every existing caller relies on"
    );
    assert_eq!(
        registry_fingerprint(),
        baseline,
        "a refused install must leave every one of the {} states bit-identical",
        mc_data::STATE_COUNT
    );

    // ---- 3. reinstalling the *same* bake must be inert, byte for byte ----
    let (identical, _) = bake_all(&vanilla, MAX_ATLAS).expect("bake identically");
    assert_eq!((identical.atlas.w, identical.atlas.h), first_atlas);
    mc_assets::model::reinstall(identical);
    assert_eq!(mc_assets::model::installed_generation(), 2);
    assert_eq!(
        registry_fingerprint(),
        baseline,
        "reinstalling an identical bake must not change a single vertex — this is the \
         load-bearing inertness claim: a client that never swaps packs sees exactly what it \
         always saw"
    );

    // ---- 4. reinstalling a *different* bake must actually change things ----
    let dir = tempfile::tempdir().expect("scratch dir");
    let pack = repaint_stone(dir.path(), [255, 0, 255]);
    let mut stacked = vanilla_stack();
    stacked.push_pack(Box::new(
        mc_assets::FolderResourcePack::open(&pack).expect("folder pack"),
    ));
    let (repainted, _) = bake_all(&stacked, MAX_ATLAS).expect("bake with the pack");
    assert!(
        texels_of(&repainted, [255, 0, 255]) >= 256,
        "the re-stitched atlas must carry the pack's flat 16x16 stone"
    );
    mc_assets::model::reinstall(repainted);
    assert_eq!(mc_assets::model::installed_generation(), 3);
    assert_eq!(
        installed_stone_texel(),
        [255, 0, 255],
        "after reinstall, `sprite()` must answer from the NEW bake"
    );

    // ---- 5. and reverting must restore the original bytes exactly ----
    let (restored, _) = bake_all(&vanilla, MAX_ATLAS).expect("bake vanilla again");
    assert_eq!(
        texels_of(&restored, [255, 0, 255]),
        0,
        "the control: vanilla's own atlas has none of the pack's colour anywhere"
    );
    mc_assets::model::reinstall(restored);
    assert_eq!(
        registry_fingerprint(),
        baseline,
        "removing the pack must restore every baked quad byte for byte, not approximately"
    );
    assert_ne!(
        installed_stone_texel(),
        [255, 0, 255],
        "and the sprite table must come back with it"
    );
}

// -------------------------------------------------------- the negative control

/// The control on "this test does not skip". See `vanilla_assets.rs`' twin for the reasoning.
///
/// Touches no process global, so it does not violate this file's single-`#[test]` sequencing
/// rule for the registry.
#[test]
fn the_reinstall_suite_fails_loudly_when_the_jar_is_absent() {
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
