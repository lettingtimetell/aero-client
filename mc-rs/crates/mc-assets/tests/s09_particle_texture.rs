//! S09 → S09/S21: `bake_all` must also stitch each model's `particle` texture,
//! which vanilla adds in `ModelBakery.getTextureLocations`/`getItemsTextureLocations` —
//! both call `resolveTextureName("particle")` unconditionally, *alongside* the face-texture
//! scan, not only when some face happens to use the same texture.
//!
//! Before this closed, `bake_all`'s Pass 2 collected only face textures, so a model whose
//! `particle` texture was not also used by any face resolved to the missing-texture placeholder
//! wherever something looked it up by location directly — `sprite()`, which a block-break
//! particle system would call with exactly this kind of location.
//!
//! No block in the real 1.8.9 jar actually exercises the gap: every vanilla model's `particle`
//! texture is also used as a face texture *somewhere* in the asset set (checked programmatically
//! against every `block/`/`item/` model in `1.8.9.jar` while writing this test — zero
//! counterexamples), so the bug was real but silent against vanilla assets alone. This test
//! manufactures the case a resource pack (or a future vanilla-adjacent model) could hit: a
//! folder pack stacked over the real jar that overrides `models/block/dirt.json`'s `particle`
//! texture to a texture no face anywhere in the combined pack stack references.
//!
//! Like `s21_reinstall.rs`, this calls `mc_assets::install` and therefore owns the process
//! globals — it lives in its own test binary so it cannot race another file's bake.
//!
//! **This test does not skip.** It used to print `SKIP:` and `return`, which *passes*, and
//! `cargo test` swallows a passing test's output — so on a checkout without the jar it
//! reported a green test of the particle-stitching regression having stitched nothing. It now
//! panics naming the path. Its old locator also hard-coded three candidate paths and honoured
//! no `MC_JARS_DIR`; it uses `tests/common/mod.rs` now, like the rest of the directory.

mod common;

use mc_assets::{
    bake_all, sprite, Atlas, FolderResourcePack, ResourceLocation, ResourceManager, ZipResourcePack,
};

const MAX_ATLAS: u32 = 2048;

/// The colour the override pack paints its particle-only texture. Not used anywhere else, real
/// jar or synthetic pack, so any texel of it in the atlas can only have come from this pack's
/// `particle` texture actually being stitched.
const PARTICLE_ONLY_RGB: [u8; 3] = [11, 222, 137];

/// A pack overriding `block/dirt`'s `particle` texture to a texture that is **not** any face's
/// texture. Vanilla's `block/cube_all` (dirt's real parent) sets `"particle": "#all"` — the same
/// texture as every face — so plain dirt could never exercise this gap; this override keeps
/// every face exactly as vanilla bakes them (`down`/`up`/`north`/`south`/`west`/`east` all still
/// resolve `#all` -> `blocks/dirt`, inherited unchanged from `block/cube_all` -> `block/cube`)
/// and changes only `particle`, which is what manufactures the gap deliberately rather than by
/// accident.
fn particle_override_pack(dir: &std::path::Path) -> std::path::PathBuf {
    let pack = dir.join("ParticleOverride");
    let models = pack.join("assets/minecraft/models/block");
    let textures = pack.join("assets/minecraft/textures/blocks");
    std::fs::create_dir_all(&models).expect("mkdir models");
    std::fs::create_dir_all(&textures).expect("mkdir textures");
    std::fs::write(
        pack.join("pack.mcmeta"),
        br#"{"pack":{"pack_format":1,"description":"s09 particle texture"}}"#,
    )
    .expect("write pack.mcmeta");
    std::fs::write(
        models.join("dirt.json"),
        br#"{
            "parent": "block/cube_all",
            "textures": {
                "all": "blocks/dirt",
                "particle": "blocks/test_particle_only"
            }
        }"#,
    )
    .expect("write dirt.json override");

    let mut image = image::RgbaImage::new(16, 16);
    for pixel in image.pixels_mut() {
        *pixel = image::Rgba([
            PARTICLE_ONLY_RGB[0],
            PARTICLE_ONLY_RGB[1],
            PARTICLE_ONLY_RGB[2],
            255,
        ]);
    }
    image
        .save(textures.join("test_particle_only.png"))
        .expect("write test_particle_only.png");

    pack
}

/// How many texels of `rgb` an atlas sheet contains.
fn texels_of(atlas: &Atlas, rgb: [u8; 3]) -> usize {
    atlas
        .texture
        .chunks_exact(4)
        .filter(|texel| texel[0] == rgb[0] && texel[1] == rgb[1] && texel[2] == rgb[2])
        .count()
}

#[test]
fn a_particle_texture_used_by_no_face_still_gets_stitched_into_the_atlas() {
    let jar_path = common::vanilla_jar("s09_particle_texture (jars/versions/1.8.9/1.8.9.jar)");

    let dir = tempfile::tempdir().expect("tempdir");
    let pack_dir = particle_override_pack(dir.path());

    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(
        ZipResourcePack::open(&jar_path).expect("open 1.8.9.jar"),
    ));
    rm.push_pack(Box::new(
        FolderResourcePack::open(&pack_dir).expect("open override pack"),
    ));

    let (assets, report) = bake_all(&rm, MAX_ATLAS).expect("bake with the override pack");
    assert!(
        report.is_clean(),
        "the override must not break any legal state's bake ({} failures)",
        report.failures.len()
    );

    let found = texels_of(&assets.atlas, PARTICLE_ONLY_RGB);
    assert!(
        found >= 256,
        "the particle-only texture must be stitched into the atlas as a real 16x16 tile \
         (>= 256 texels of {PARTICLE_ONLY_RGB:?}); found {found}"
    );

    mc_assets::install(assets);
    let loc = ResourceLocation::minecraft("textures/blocks/test_particle_only");
    let s = sprite(&loc);
    assert_ne!(
        s.uv,
        mc_assets::sprite::UvRect::FULL,
        "the particle-only texture is still resolving to the placeholder sprite"
    );
    assert_eq!((s.width, s.height), (16, 16));
    let px = &s.frames[0].rgba;
    assert_eq!(
        (px[0], px[1], px[2]),
        (
            PARTICLE_ONLY_RGB[0],
            PARTICLE_ONLY_RGB[1],
            PARTICLE_ONLY_RGB[2]
        ),
        "sprite() must answer with the real decoded pixels, not the missing-texture checkerboard"
    );
}

// -------------------------------------------------------- the negative control

/// The control on "this test does not skip". See `vanilla_assets.rs`' twin for the reasoning.
#[test]
fn the_particle_suite_fails_loudly_when_the_jar_is_absent() {
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
