//! S09 "Done when" acceptance tests: every legal, `RenderType::Model` block state bakes
//! cleanly against the real vanilla 1.8.9 assets, and none of them resolve to the
//! missing-model placeholder. Hazard **H12** canaries (stairs, fences, doors) get their own
//! targeted checks in addition to the exhaustive sweep.
//!
//! **These tests do not skip.** They used to: `require!` printed `SKIP:` and `return`ed, which
//! *passes*, and `cargo test` swallows a passing test's output — so a checkout without the jar
//! reported ten green tests of S09's literal "Done when" criterion having baked nothing. If the
//! jar is absent, every test now panics naming the path it looked at. See `tests/common/mod.rs`.

mod common;

use mc_assets::{
    bake_all, item_model_for, item_model_for_damage, model_for, model_for_actual, sprite,
    ResourceLocation, ResourceManager, ZipResourcePack,
};
use mc_data::block::ActualBlockState;
use mc_data::{block_by_name, item_by_name, BlockState};

/// Was `let Some(x) = ... else { SKIP; return }`.
macro_rules! require {
    ($f:ident, $what:literal) => {
        $f(concat!("s09_model_bake (", $what, ")"))
    };
}

fn manager(what: &str) -> ResourceManager {
    let jar = common::vanilla_jar(what);
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(
        ZipResourcePack::open(&jar).expect("open 1.8.9.jar"),
    ));
    rm
}

/// The full sweep: bake every legal state against the real jar and check the report is
/// clean. This is S09's literal "Done when" criterion.
#[test]
fn every_legal_model_state_bakes_without_hitting_the_missing_placeholder() {
    let rm = require!(manager, "jars/versions/1.8.9/1.8.9.jar");

    let (assets, report) =
        bake_all(&rm, 2048).expect("bake_all should not hard-fail against real vanilla assets");

    if !report.is_clean() {
        for (state, reason) in report.failures.iter().take(20) {
            eprintln!(
                "FAIL state block={} meta={}: {reason}",
                state.block().0,
                state.meta()
            );
        }
        for failure in report.model_failures.iter().take(20) {
            eprintln!("FAIL model identity: {failure:?}");
        }
    }
    assert!(
        report.is_clean(),
        "{} of {} model states failed to bake (see stderr for the first 20)",
        report.failures.len(),
        report.model_states_considered
    );

    // Cross-check against the counts independently verified (S01's codegen validator, plus
    // the S09 state-mapper cross-check against the real jar): 1394 legal states total, 143 of
    // them are render_type != Model (air/fluids/tile-entity-rendered) and never reach this
    // bake pass at all, leaving exactly 1251 that must bake cleanly.
    assert_eq!(report.model_states_considered, 1251);
    assert_eq!(
        assets.actual_model_count(),
        2555,
        "every full blockstate variant key must reach the registry"
    );
    assert_eq!(
        assets.item_model_count(),
        515,
        "every vanilla ItemModelMesher registration must reach the registry"
    );
    assert_eq!(
        assets.item_frame_model_count(),
        2,
        "both manually registered item-frame variants must remain usable"
    );

    assert!(assets.atlas.w.is_power_of_two());
    assert!(assets.atlas.h.is_power_of_two());
    assert_eq!(
        assets.atlas.texture.len(),
        (assets.atlas.w * assets.atlas.h * 4) as usize
    );
}

/// H12 canary: stairs. Every legal rotation/half/shape combination must bake to a non-empty
/// model with plausible geometry.
#[test]
fn stairs_bake_with_geometry_in_every_orientation() {
    let rm = require!(manager, "jars/versions/1.8.9/1.8.9.jar");
    let (assets, report) = bake_all(&rm, 2048).unwrap();
    assert!(report.is_clean());
    mc_assets::install(assets);

    let stairs = block_by_name("oak_stairs").unwrap();
    let mut any_nonempty = 0;
    for meta in 0u8..8 {
        let state = BlockState::new(stairs, meta);
        if !state.is_legal() {
            continue;
        }
        let model = model_for(state);
        assert!(!model.is_empty(), "oak_stairs meta={meta} baked empty");
        any_nonempty += 1;
        // A stair has both cube-ish culled faces and the step's cut corner as a general quad.
        assert!(
            !model.general_quads.is_empty(),
            "oak_stairs meta={meta} has no general quads (missing the step geometry?)"
        );
    }
    assert_eq!(
        any_nonempty, 8,
        "all 8 oak_stairs metas (0..7) should be legal"
    );

    // The packed meta-0 state is the outer-right storage representative, while an isolated
    // stair's client-derived shape is straight. These models must not collapse to the same
    // geometry, or a stale raw-state lookup can pass the weaker non-empty canary above.
    let stored = BlockState::new(stairs, 0);
    let raw = model_for(stored);
    let mut actual = ActualBlockState::from_stored(stored);
    assert!(actual.set("shape", "straight"));
    let straight = model_for_actual(&actual);
    assert!(!straight.is_empty());
    assert_ne!(
        raw.general_quads, straight.general_quads,
        "outer-right and isolated-straight stairs must select distinct baked geometry"
    );
}

/// H12 canary: fences. `fence` needs no state-mapper rename (direct lookup); `fence_gate`
/// does (its `powered` property is ignored by vanilla's real mapper).
#[test]
fn fences_and_fence_gates_bake_with_geometry() {
    let rm = require!(manager, "jars/versions/1.8.9/1.8.9.jar");
    let (assets, report) = bake_all(&rm, 2048).unwrap();
    assert!(report.is_clean());
    mc_assets::install(assets);

    let fence = model_for(BlockState::new(block_by_name("fence").unwrap(), 0));
    assert!(!fence.is_empty(), "fence baked empty");

    let gate = model_for(BlockState::new(block_by_name("fence_gate").unwrap(), 0));
    assert!(!gate.is_empty(), "fence_gate baked empty");
}

/// Non-`Model` blocks (fluids, tile-entity-rendered, air) must bake to the *empty* model —
/// not the missing-texture placeholder, and not a crash.
#[test]
fn special_render_type_blocks_bake_empty_not_missing() {
    let rm = require!(manager, "jars/versions/1.8.9/1.8.9.jar");
    let (assets, report) = bake_all(&rm, 2048).unwrap();
    assert!(report.is_clean());
    mc_assets::install(assets);

    for name in [
        "air",
        "water",
        "flowing_lava",
        "chest",
        "standing_sign",
        "skull",
    ] {
        let block = block_by_name(name).unwrap();
        let model = model_for(BlockState::new(block, 0));
        assert!(
            model.is_empty(),
            "{name} should bake to the empty model, not a placeholder"
        );
    }
}

/// A representative block texture must be reachable through the atlas at a sane UV rect
/// after a real bake.
#[test]
fn sprite_lookup_resolves_into_the_stitched_atlas() {
    let rm = require!(manager, "jars/versions/1.8.9/1.8.9.jar");
    let (assets, report) = bake_all(&rm, 2048).unwrap();
    assert!(report.is_clean());
    mc_assets::install(assets);

    let loc = ResourceLocation::minecraft("textures/blocks/stone");
    let s = sprite(&loc);
    assert!(s.uv.min_u < s.uv.max_u);
    assert!(s.uv.min_v < s.uv.max_v);
    assert_eq!(s.width, 16);
    assert_eq!(s.height, 16);
    assert!(!s.frames.is_empty());
}

/// Pixel-diffs a stitched sprite's atlas region against the source PNG it came from. There is
/// no display in this environment to eyeball the atlas, so this is the closest available
/// substitute for the "Done when" clause about visual fidelity: every pixel the atlas placed
/// for `stone.png` must equal the corresponding pixel decoded directly from the same PNG.
#[test]
fn atlas_region_is_a_byte_exact_copy_of_the_source_png() {
    let rm = require!(manager, "jars/versions/1.8.9/1.8.9.jar");
    let (assets, report) = bake_all(&rm, 2048).unwrap();
    assert!(report.is_clean());

    let loc = ResourceLocation::minecraft("textures/blocks/stone");
    let direct =
        mc_assets::sprite::load_sprite(&rm, &loc).expect("stone.png should decode directly");

    // Copy out what's needed before `install` takes ownership of `assets`.
    let atlas_w = assets.atlas.w;
    let atlas_h = assets.atlas.h;
    let atlas_pixels = assets.atlas.texture.clone();
    mc_assets::install(assets);
    let placed = sprite(&loc);

    let x0 = (placed.uv.min_u * atlas_w as f32).round() as i64;
    let y0 = (placed.uv.min_v * atlas_h as f32).round() as i64;

    let mut checked = 0u32;
    for py in 0..direct.height as i64 {
        for px in 0..direct.width as i64 {
            let ax = x0 + px;
            let ay = y0 + py;
            if ax < 0 || ay < 0 || ax as u32 >= atlas_w || ay as u32 >= atlas_h {
                continue; // half-texel inset can round a border row/col out; not a fidelity bug
            }
            let atlas_idx = ((ay as u32 * atlas_w + ax as u32) * 4) as usize;
            let src_idx = ((py as u32 * direct.width + px as u32) * 4) as usize;
            let atlas_px = &atlas_pixels[atlas_idx..atlas_idx + 4];
            let src_px = &direct.frames[0].rgba[src_idx..src_idx + 4];
            assert_eq!(
                atlas_px, src_px,
                "pixel ({px},{py}) of stone.png differs in the stitched atlas"
            );
            checked += 1;
        }
    }
    assert!(
        checked > 200,
        "sanity: should have diffed most of a 16x16 texture, got {checked}"
    );
}

/// Closes the S11 -> S09 request: `bake_all` must also stitch the textures no JSON model
/// references (java: `ModelBakery.LOCATIONS_BUILTIN_TEXTURES`). Without them `sprite()` answers
/// with its placeholder and every fluid quad's UV is wrong.
#[test]
fn the_builtin_textures_no_model_references_are_stitched_into_the_atlas() {
    let rm = require!(manager, "jars/versions/1.8.9/1.8.9.jar");
    let (assets, report) = bake_all(&rm, 2048).unwrap();
    assert!(report.is_clean());
    eprintln!("atlas is {}x{}", assets.atlas.w, assets.atlas.h);
    mc_assets::install(assets);

    for path in mc_assets::model::BUILTIN_TEXTURES {
        let loc = ResourceLocation::minecraft(format!("textures/{path}"));
        let s = sprite(&loc);
        assert_ne!(
            s.uv,
            mc_assets::sprite::UvRect::FULL,
            "{path} is still resolving to the placeholder sprite"
        );
        assert!(s.uv.min_u < s.uv.max_u && s.uv.min_v < s.uv.max_v, "{path}");
        assert!(s.width > 0 && s.height > 0, "{path}");
    }

    // The four fluid sprites are animated, and their frame counts come straight out of the
    // jar: with no explicit `frames` list the count is strip height / width (water_still
    // 16x512 -> 32, water_flow 32x1024 -> 32, lava_flow 32x512 -> 16); `lava_still.png.mcmeta`
    // *does* list frames, 38 of them over a 20-frame strip, so the sequence repeats.
    for (path, side, frames) in [
        ("textures/blocks/water_still", 16u32, 32usize),
        ("textures/blocks/water_flow", 32, 32),
        ("textures/blocks/lava_still", 16, 38),
        ("textures/blocks/lava_flow", 32, 16),
    ] {
        let loc = ResourceLocation::minecraft(path);
        let s = sprite(&loc);
        assert_eq!((s.width, s.height), (side, side), "{path} frame size");
        assert_eq!(s.frames.len(), frames, "{path} frame count");
    }
}

/// The same byte-exact pixel diff `atlas_region_is_a_byte_exact_copy_of_the_source_png` does
/// for `stone`, but for `water_still` — a *builtin* (model-less), animated, 32-frame texture.
/// This is what makes "fluid UVs are right now" a measured fact rather than a claim: the UV
/// `terrain::fluid` interpolates through lands on texels that equal frame 0 of the real PNG.
#[test]
fn the_water_sprite_region_is_a_byte_exact_copy_of_frame_zero() {
    let rm = require!(manager, "jars/versions/1.8.9/1.8.9.jar");
    let (assets, report) = bake_all(&rm, 2048).unwrap();
    assert!(report.is_clean());

    let loc = ResourceLocation::minecraft("textures/blocks/water_still");
    let direct = mc_assets::sprite::load_sprite(&rm, &loc).expect("water_still.png decodes");
    assert!(direct.frames.len() > 1, "water_still must be animated");

    let atlas_w = assets.atlas.w;
    let atlas_h = assets.atlas.h;
    let atlas_pixels = assets.atlas.texture.clone();
    mc_assets::install(assets);
    let placed = sprite(&loc);

    let x0 = (placed.uv.min_u * atlas_w as f32).round() as i64;
    let y0 = (placed.uv.min_v * atlas_h as f32).round() as i64;

    let mut checked = 0u32;
    let mut nonzero = 0u32;
    for py in 0..direct.height as i64 {
        for px in 0..direct.width as i64 {
            let (ax, ay) = (x0 + px, y0 + py);
            if ax < 0 || ay < 0 || ax as u32 >= atlas_w || ay as u32 >= atlas_h {
                continue;
            }
            let a = ((ay as u32 * atlas_w + ax as u32) * 4) as usize;
            let s = ((py as u32 * direct.width + px as u32) * 4) as usize;
            assert_eq!(
                &atlas_pixels[a..a + 4],
                &direct.frames[0].rgba[s..s + 4],
                "pixel ({px},{py}) of water_still frame 0 differs in the stitched atlas"
            );
            checked += 1;
            if atlas_pixels[a + 3] != 0 {
                nonzero += 1;
            }
        }
    }
    assert!(checked > 200, "only diffed {checked} pixels");
    assert!(
        nonzero > 200,
        "the region is blank ({nonzero} opaque texels) — the UV points at empty atlas space"
    );
}

/// Closes the S12 -> S09 request: the model JSON's `display` block reaches `BakedModel`, with
/// vanilla's `1/16` translation scale, its clamps, and its **per-context** parent inheritance.
///
/// Expected values are read off the real jar's JSON by hand rather than round-tripped through
/// this crate's own parser, so a wrong scale factor or a wrong inheritance rule fails here.
#[test]
fn item_camera_transforms_are_carried_onto_the_baked_model() {
    use mc_assets::model::{ItemTransform, TransformType};
    use mc_data::item_by_name;

    let rm = require!(manager, "jars/versions/1.8.9/1.8.9.jar");
    let (assets, report) = bake_all(&rm, 2048).unwrap();
    assert!(report.is_clean());
    mc_assets::install(assets);

    let close = |a: [f32; 3], b: [f32; 3]| (0..3).all(|i| (a[i] - b[i]).abs() < 1e-6);

    // `models/item/diamond_sword.json` — `builtin/generated`, sets thirdperson + firstperson.
    // thirdperson translation [0, 1.25, -3.5] texels -> [0, 0.078125, -0.21875] blocks.
    let sword = item_model_for(item_by_name("diamond_sword").expect("diamond_sword"));
    let tp = sword.transforms.third_person;
    assert_eq!(tp.rotation, [0.0, 90.0, -35.0]);
    assert!(
        close(tp.translation, [0.0, 1.25 / 16.0, -3.5 / 16.0]),
        "{tp:?}"
    );
    assert_eq!(tp.scale, [0.85, 0.85, 0.85]);
    let fp = sword.transforms.first_person;
    assert_eq!(fp.rotation, [0.0, -135.0, 25.0]);
    assert!(
        close(fp.translation, [0.0, 4.0 / 16.0, 2.0 / 16.0]),
        "{fp:?}"
    );
    assert_eq!(fp.scale, [1.7, 1.7, 1.7]);
    // It sets neither, and nothing in `builtin/generated`'s chain does either.
    assert!(sword.transforms.gui.is_default());
    assert!(sword.transforms.ground.is_default());

    // `models/item/oak_stairs.json` parents to the *block* model and sets `gui` with a rotation
    // only — translation and scale must fall back to the identity, not to nothing.
    let stairs = item_model_for(item_by_name("oak_stairs").expect("oak_stairs"));
    assert_eq!(
        stairs.transforms.gui,
        ItemTransform {
            rotation: [0.0, 180.0, 0.0],
            translation: [0.0; 3],
            scale: [1.0; 3],
        }
    );
    assert_eq!(stairs.transforms.third_person.scale, [0.375, 0.375, 0.375]);

    // A plain block model has no `display` anywhere in its chain: every context stays identity.
    let stone_block = model_for(BlockState::new(block_by_name("stone").unwrap(), 0));
    assert!(!stone_block.is_empty());
    for ty in TransformType::VALUES {
        assert!(
            stone_block.transforms.get(ty).is_default(),
            "block/stone should have no {ty:?} transform"
        );
    }

    // ...but the *item* form of the same block does, through `models/item/stone.json`.
    let stone_item = item_model_for(item_by_name("stone").expect("stone item"));
    assert_eq!(
        stone_item.transforms.third_person.rotation,
        [10.0, -45.0, 170.0]
    );
    assert!(close(
        stone_item.transforms.third_person.translation,
        [0.0, 1.5 / 16.0, -2.75 / 16.0]
    ));

    // Honest scope note, asserted rather than written down: **no 1.8.9 model sets
    // `display.ground`** (the key arrives in 1.9), so with vanilla assets the dropped-item
    // context is always the identity. The seam is now carried and correct — it matters for a
    // resource pack and it removes S12's externally-supplied ground scale — but it changes
    // nothing about how vanilla 1.8.9 items look on the ground.
    let mut with_ground = 0usize;
    for props in mc_data::items() {
        if props.registered && !item_model_for(props.id).transforms.ground.is_default() {
            with_ground += 1;
        }
    }
    assert_eq!(
        with_ground, 0,
        "1.8.9 ships no display.ground; if this ever fires, the assets changed"
    );
}

#[test]
fn item_model_mesher_selects_vanillas_named_metadata_variants() {
    use mc_data::item_by_name;

    let rm = require!(manager, "jars/versions/1.8.9/1.8.9.jar");
    let (assets, report) = bake_all(&rm, 2048).unwrap();
    assert!(report.is_clean());
    mc_assets::install(assets);

    let door = item_model_for_damage(item_by_name("wooden_door").unwrap(), 0);
    let double_plant = item_model_for_damage(item_by_name("double_plant").unwrap(), 5);
    let black_dye = item_model_for_damage(item_by_name("dye").unwrap(), 0);
    let white_dye = item_model_for_damage(item_by_name("dye").unwrap(), 15);
    assert!(!door.is_empty(), "wooden_door selects item/oak_door");
    assert!(
        !double_plant.is_empty(),
        "double_plant metadata 5 selects paeonia"
    );
    assert!(!black_dye.is_empty() && !white_dye.is_empty());
    assert!(
        !mc_assets::cache::model_eq(black_dye, white_dye),
        "negative control: dye metadata endpoints must select distinct baked geometry"
    );

    let oak = item_model_for_damage(item_by_name("planks").unwrap(), 0);
    let dark_oak = item_model_for_damage(item_by_name("planks").unwrap(), 5);
    assert!(!oak.is_empty() && !dark_oak.is_empty());
    assert!(
        !mc_assets::cache::model_eq(oak, dark_oak),
        "negative control: registry-name-only selection would alias all six planks"
    );
    let missing = item_model_for_damage(item_by_name("planks").unwrap(), 99);
    assert_eq!(
        missing
            .face_quads
            .iter()
            .map(std::vec::Vec::len)
            .sum::<usize>(),
        6,
        "an unregistered metadata must use Java's shared builtin/missing cube"
    );
    let missing_uv = sprite(&ResourceLocation::minecraft("missingno")).uv;
    for quads in &missing.face_quads {
        assert_eq!(
            quads.len(),
            1,
            "missing cube has one cull-face quad per side"
        );
        for quad in quads {
            for vertex in &quad.verts {
                assert!(
                    vertex.uv[0] >= missing_uv.min_u - 1e-6
                        && vertex.uv[0] <= missing_uv.max_u + 1e-6
                        && vertex.uv[1] >= missing_uv.min_v - 1e-6
                        && vertex.uv[1] <= missing_uv.max_v + 1e-6,
                    "missing-model vertex {:?} escaped the stitched missingno UV {:?}",
                    vertex.uv,
                    missing_uv
                );
            }
        }
    }
    assert!(
        !mc_assets::cache::model_eq(missing, oak),
        "negative control: an unregistered metadata must not alias subtype zero"
    );

    let compass = item_model_for(item_by_name("compass").unwrap());
    let clock = item_model_for(item_by_name("clock").unwrap());
    assert!(
        !compass.is_empty(),
        "builtin/compass uses generated geometry"
    );
    assert!(!clock.is_empty(), "builtin/clock uses generated geometry");
}

/// `ItemPotion` is the one 1.8.9 item whose model location is not derived from its registry
/// name. Vanilla's `ItemMeshDefinition` selects `bottle_drinkable` or `bottle_splash` from bit
/// 14, and both generated models carry the liquid overlay as tint index 0 plus the white bottle
/// art as tint index 1. This test uses the real jar so a registry-name-only fallback (or a bake
/// that forgets the texture-only model names) cannot pass by returning an empty placeholder.
#[test]
fn potion_mesh_definition_bakes_both_models_and_preserves_tint_layers() {
    let rm = require!(manager, "jars/versions/1.8.9/1.8.9.jar");
    let (assets, report) = bake_all(&rm, 2048).unwrap();
    assert!(report.is_clean());
    mc_assets::install(assets);

    let potion = item_by_name("potion").expect("potion item");
    let drinkable = item_model_for_damage(potion, 8194);
    let splash = item_model_for_damage(potion, 16_386);
    assert!(
        !drinkable.is_empty(),
        "item/potion.json must not be the lookup path"
    );
    assert!(
        !splash.is_empty(),
        "bottle_splash must be baked and addressable"
    );
    assert!(
        !mc_assets::cache::model_eq(drinkable, splash),
        "drinkable and splash models must retain their distinct layer-1 textures"
    );

    for (name, model) in [("drinkable", drinkable), ("splash", splash)] {
        assert!(
            model
                .general_quads
                .iter()
                .any(|quad| quad.tint_index == Some(0)),
            "{name} model lost the tintable liquid overlay layer"
        );
        assert!(
            model
                .general_quads
                .iter()
                .any(|quad| quad.tint_index == Some(1)),
            "{name} model lost the un-tinted bottle layer"
        );
    }

    // ItemPotion.isSplash checks only bit 14; all other damage bits must leave the selected
    // model unchanged. The alternate values are biting controls for a raw-metadata lookup.
    assert!(mc_assets::cache::model_eq(
        drinkable,
        item_model_for_damage(potion, 8194 | 0x0200)
    ));
    assert!(mc_assets::cache::model_eq(
        splash,
        item_model_for_damage(potion, 16_386 | 0x0200)
    ));
}

/// Animated textures (lava/water/fire) decode into more than one frame.
#[test]
fn animated_textures_decode_multiple_frames() {
    let rm = require!(manager, "jars/versions/1.8.9/1.8.9.jar");
    let s = mc_assets::sprite::load_sprite(
        &rm,
        &ResourceLocation::minecraft("textures/blocks/lava_still"),
    )
    .expect("lava_still.png should decode");
    assert!(
        s.frames.len() > 1,
        "lava_still should be an animated multi-frame texture"
    );
    assert!(s.default_duration_ticks >= 1);
}

// -------------------------------------------------------- the negative control

/// The control on "these tests do not skip". See `vanilla_assets.rs`' twin for the reasoning.
#[test]
fn the_bake_suite_fails_loudly_when_the_jar_is_absent() {
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
