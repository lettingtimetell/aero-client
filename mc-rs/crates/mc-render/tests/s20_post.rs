//! S20 integration tests: parsing the *real* vanilla shader-group JSON out of `1.8.9.jar`, and
//! running [`mc_render::post::PostChain`] against real bytes end to end.
//!
//! Skips (printing `SKIP:`) when the jar is unavailable. Point `MC_JARS_DIR` at a `jars/`
//! directory to override the asset location.

use std::path::{Path, PathBuf};

use mc_assets::{ResourceLocation, ResourceManager, ZipResourcePack};
use mc_render::post::{
    OverlayKind, OverlayPhase, PostChain, PostEffect, PostPhase, ShaderGroupSpec, PROGRAM_DEFAULTS,
};
use mc_render::{
    Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase, SceneTextures,
    HEADLESS_FORMAT,
};

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn vanilla_jar() -> Option<PathBuf> {
    let p = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    p.is_file().then_some(p)
}

fn resources() -> Option<ResourceManager> {
    let jar = vanilla_jar()?;
    let pack = ZipResourcePack::open(jar).ok()?;
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(pack));
    Some(rm)
}

/// A resource manager backing no packs at all, for tests that build a `PostChain` from a
/// hand-written spec rather than real jar assets. `PostChain::new`'s `textures/effect/<id>.png`
/// aux-texture fallback simply misses against this and every aux slot keeps the flat white
/// placeholder, same as before that fallback existed.
fn empty_resources() -> ResourceManager {
    ResourceManager::new()
}

macro_rules! require {
    ($opt:expr, $what:literal) => {
        match $opt {
            Some(v) => v,
            None => {
                eprintln!("SKIP: {} not available", $what);
                return;
            }
        }
    };
}

/// Every `post/*.json` filename the research pass confirmed ships inside `1.8.9.jar` (24 files:
/// vanilla's `EntityRenderer.shaderResourceLocations` plus `RenderGlobal`'s `entity_outline`).
const KNOWN_POST_GROUPS: &[&str] = &[
    "antialias",
    "art",
    "bits",
    "blobs",
    "blobs2",
    "blur",
    "bumpy",
    "color_convolve",
    "creeper",
    "deconverge",
    "desaturate",
    "entity_outline",
    "flip",
    "fxaa",
    "green",
    "invert",
    "notch",
    "ntsc",
    "outline",
    "pencil",
    "phosphor",
    "scan_pincushion",
    "sobel",
    "spider",
    "wobble",
];

#[test]
fn every_known_post_group_file_exists_in_the_jar() {
    let rm = require!(resources(), "1.8.9.jar");
    for name in KNOWN_POST_GROUPS {
        let path = format!("shaders/post/{name}.json");
        let loc = ResourceLocation::parse(&path).unwrap();
        assert!(rm.contains(&loc), "{path} missing from the jar");
    }
}

#[test]
fn parses_the_real_blurjson_from_the_jar() {
    let rm = require!(resources(), "1.8.9.jar");
    let text = rm
        .load_string(&ResourceLocation::parse("shaders/post/blur.json").unwrap())
        .expect("load blur.json");
    let spec = ShaderGroupSpec::parse(&text).expect("parse blur.json");
    assert_eq!(spec.targets.len(), 1);
    assert_eq!(spec.targets[0].name(), "swap");
    assert_eq!(spec.passes.len(), 2);
    assert_eq!(spec.passes[0].name, "blur");
    assert_eq!(spec.passes[0].intarget, "minecraft:main");
    assert_eq!(spec.passes[0].outtarget, "swap");
    assert_eq!(spec.passes[1].intarget, "swap");
    assert_eq!(spec.passes[1].outtarget, "minecraft:main");
}

#[test]
fn parses_the_real_notchjson_with_its_dither_aux_texture() {
    let rm = require!(resources(), "1.8.9.jar");
    let text = rm
        .load_string(&ResourceLocation::parse("shaders/post/notch.json").unwrap())
        .expect("load notch.json");
    let spec = ShaderGroupSpec::parse(&text).expect("parse notch.json");
    assert!(spec.passes[0]
        .auxtargets
        .iter()
        .any(|a| a.name == "DitherSampler" && a.id == "dither"));
}

#[test]
fn the_real_notch_group_uses_the_actual_dither_texture_not_a_flat_placeholder() {
    // The S20 aux-target fix, proven end to end: `notch.json`'s `DitherSampler` aux-target
    // (`id: "dither"`) names no framebuffer `PostChain` allocates, so — mirroring vanilla's
    // `ShaderGroup.parseGroup` — it must fall back to `textures/effect/dither.png` through the
    // `ResourceManager` now threaded into `PostChain::new`, not the flat 1x1 white placeholder
    // every unresolved aux slot used before this existed.
    let rm = require!(resources(), "1.8.9.jar");
    let Some(gpu) = headless(16, 16) else { return };

    // First, the raw asset: `dither.png` itself must really have spatial variation, or this
    // whole test would be trivially satisfied by a degenerate fixture.
    let loc = ResourceLocation::parse("textures/effect/dither").unwrap();
    let sprite =
        mc_assets::sprite::load_sprite(&rm, &loc).expect("load textures/effect/dither.png");
    assert!(
        sprite.width > 1 && sprite.height > 1,
        "dither.png decoded as {}x{}, too small to carry a dither pattern",
        sprite.width,
        sprite.height
    );
    let first_texel = &sprite.frames[0].rgba[0..4];
    assert!(
        sprite.frames[0]
            .rgba
            .chunks_exact(4)
            .any(|px| px != first_texel),
        "dither.png decoded as a single flat colour; the fixture is degenerate"
    );

    // Same pass shape as the real notch.json's first pass, simplified to write
    // `minecraft:main` directly (no `swap` ping-pong) so `run_standalone` reads it back in one
    // call — the `DitherSampler`/`dither` auxtarget is the only thing under test here.
    let spec = ShaderGroupSpec::parse(
        r#"{"targets": [], "passes": [{"name":"notch","intarget":"minecraft:main","outtarget":"minecraft:main",
            "auxtargets":[{"name":"DitherSampler","id":"dither"}]}]}"#,
    )
    .unwrap();

    // A perfectly flat input: `tex(stepped)` in `notch`'s WGSL is then constant everywhere, so
    // any variation in the output can only come from the aux sampler.
    let input = [120u8, 120, 120, 255].repeat(16 * 16);

    let with_real_texture = PostChain::new(&gpu, &rm, &spec, HEADLESS_FORMAT, (16, 16))
        .run_standalone(&gpu, HEADLESS_FORMAT, 16, 16, &input);

    // Sanity check on the old/degraded path: an empty `ResourceManager` has no
    // `textures/effect/dither.png`, so it must still fall back to the flat placeholder exactly
    // as every aux slot did before this fix — a flat input then renders as one uniform colour.
    let with_placeholder = PostChain::new(
        &gpu,
        &empty_resources(),
        &spec,
        HEADLESS_FORMAT,
        (16, 16),
    )
    .run_standalone(&gpu, HEADLESS_FORMAT, 16, 16, &input);
    let placeholder_first = &with_placeholder[0..4];
    assert!(
        with_placeholder
            .chunks_exact(4)
            .all(|px| px == placeholder_first),
        "sanity check failed: an empty ResourceManager should still degrade to the flat aux \
         placeholder, but the output already varies"
    );

    // The real behaviour: binding the actual dither.png must break that uniformity, and must
    // not coincide with the placeholder's output.
    assert_ne!(
        with_real_texture, with_placeholder,
        "DitherSampler still resolves to the flat placeholder even with the real jar's \
         ResourceManager supplied"
    );
    let real_first = &with_real_texture[0..4];
    assert!(
        with_real_texture.chunks_exact(4).any(|px| px != real_first),
        "notch output is still flat even with the real dither.png bound as the aux texture: \
         {with_real_texture:?}"
    );
}

#[test]
fn parses_the_real_invertjson_used_for_the_enderman_possession_effect() {
    let rm = require!(resources(), "1.8.9.jar");
    let text = rm
        .load_string(&ResourceLocation::parse("shaders/post/invert.json").unwrap())
        .expect("load invert.json");
    let spec = ShaderGroupSpec::parse(&text).expect("parse invert.json");
    assert!(!spec.passes.is_empty());
    assert!(spec.passes.iter().any(|p| p.name == "invert"));
}

#[test]
fn every_known_post_group_parses_without_error() {
    // Breadth check: every real shipped group is at least well-formed JSON matching this port's
    // schema, not just the two or three hand-picked above.
    let rm = require!(resources(), "1.8.9.jar");
    for name in KNOWN_POST_GROUPS {
        let path = format!("shaders/post/{name}.json");
        let text = rm
            .load_string(&ResourceLocation::parse(&path).unwrap())
            .unwrap_or_else(|e| panic!("load {path}: {e}"));
        ShaderGroupSpec::parse(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"));
    }
}

fn headless(w: u32, h: u32) -> Option<Gpu> {
    match Gpu::headless(
        w,
        h,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => Some(gpu),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

#[test]
fn invertjson_actually_inverts_pixels_when_run_through_the_real_engine() {
    // The real vanilla JSON (not a hand-written test fixture) driving `PostChain` end to end,
    // with a genuine GPU readback proving the pass really executed.
    let rm = require!(resources(), "1.8.9.jar");
    let text = rm
        .load_string(&ResourceLocation::parse("shaders/post/invert.json").unwrap())
        .expect("load invert.json");
    let spec = ShaderGroupSpec::parse(&text).expect("parse invert.json");

    let Some(gpu) = headless(4, 4) else { return };
    let chain = PostChain::new(&gpu, &rm, &spec, HEADLESS_FORMAT, (4, 4));
    assert!(
        chain.native_pass_count() >= 1,
        "expected the invert pass to be native"
    );

    let input = [10u8, 200, 50, 255].repeat(16);
    let output = chain.run_standalone(&gpu, HEADLESS_FORMAT, 4, 4, &input);
    let px = &output[0..4];
    // `invert.json` overrides `InverseAmount` to **0.8**, not 1.0, so vanilla's enderman-
    // possession effect is a partial invert: `mix(c, 1 - c, 0.8)`. Hand-derived per channel:
    //   10  -> 0.0392 + 0.8 * (0.9608 - 0.0392) = 0.7765 -> 198
    //   200 -> 0.7843 + 0.8 * (0.2157 - 0.7843) = 0.3294 ->  84
    //   50  -> 0.1961 + 0.8 * (0.8039 - 0.1961) = 0.6824 -> 174
    assert!(px[0].abs_diff(198) <= 2, "{px:?}");
    assert!(px[1].abs_diff(84) <= 2, "{px:?}");
    assert!(px[2].abs_diff(174) <= 2, "{px:?}");
}

#[test]
fn spiderjsons_eight_pass_chain_allocates_and_runs_without_error() {
    // The most structurally complex real group the research turned up (multiple named targets
    // cross-feeding each other via auxtargets). Not every pass has a native effect, but the
    // *engine* — target allocation, pass ordering, binding — must not panic on real vanilla
    // authoring.
    let rm = require!(resources(), "1.8.9.jar");
    let text = rm
        .load_string(&ResourceLocation::parse("shaders/post/spider.json").unwrap())
        .expect("load spider.json");
    let spec = ShaderGroupSpec::parse(&text).expect("parse spider.json");
    assert!(spec.passes.len() >= 5, "{} passes", spec.passes.len());

    let Some(gpu) = headless(8, 8) else { return };
    let chain = PostChain::new(&gpu, &rm, &spec, HEADLESS_FORMAT, (8, 8));
    let input = [80u8, 80, 80, 255].repeat(64);
    let output = chain.run_standalone(&gpu, HEADLESS_FORMAT, 8, 8, &input);
    assert_eq!(output.len(), 8 * 8 * 4);
}

#[test]
fn every_real_post_group_builds_and_runs_end_to_end() {
    // Breadth, on the GPU rather than only in the parser: all 25 shipped groups must allocate
    // their targets, resolve their aux samplers, pack their uniforms, and execute. Together with
    // the coverage test below, this is what stops a WGSL translation from being wrong in a way
    // that only shows up on one group nobody tests.
    let rm = require!(resources(), "1.8.9.jar");
    let Some(gpu) = headless(8, 8) else { return };
    for name in KNOWN_POST_GROUPS {
        let text = rm
            .load_string(&ResourceLocation::parse(&format!("shaders/post/{name}.json")).unwrap())
            .unwrap_or_else(|e| panic!("load {name}: {e}"));
        let spec = ShaderGroupSpec::parse(&text).unwrap_or_else(|e| panic!("parse {name}: {e}"));
        let chain = PostChain::new(&gpu, &rm, &spec, HEADLESS_FORMAT, (8, 8));
        // A checkerboard, not a flat fill: every translated program is either a neighbourhood
        // filter or a colour transform, and both are identities on a uniform image.
        let mut input = vec![0u8; 8 * 8 * 4];
        for y in 0..8usize {
            for x in 0..8usize {
                let i = (y * 8 + x) * 4;
                let on = (x + y) % 2 == 0;
                input[i] = if on { 220 } else { 30 };
                input[i + 1] = if on { 120 } else { 200 };
                input[i + 2] = if on { 40 } else { 90 };
                input[i + 3] = 255;
            }
        }
        let output = chain.run_standalone(&gpu, HEADLESS_FORMAT, 8, 8, &input);
        assert_eq!(output.len(), 8 * 8 * 4, "{name}");
        // A group whose every pass is native must actually have changed something, or the
        // translation silently degenerated to a copy. Two groups are correct exceptions:
        //
        // * `phosphor` is `max(prev * Phosphor, curr)` against a `previous` target still cleared
        //   on the first frame, so returning the input verbatim *is* vanilla's first-frame
        //   result — the trail appears from frame two, which one `run_standalone` cannot show.
        // * `wobble` is a no-op at its declared `Time` default of 0.0: the hue rotation is
        //   `fract(h + Time)` and the UV wobble is 0.002 of a texel. Vanilla drives `Time` from
        //   the game clock, which is S21 wiring; see the dedicated test below, which sets it.
        // `fxaa` preserves this high-contrast 8x8 checkerboard at the default quality/edge
        // thresholds, so an unchanged output is valid for this fixture too.
        if chain.passthrough_pass_count() == 0 && !matches!(*name, "fxaa" | "phosphor" | "wobble") {
            assert_ne!(output, input, "{name} ran natively but produced its input");
        }
    }
}

#[test]
fn a_pass_reading_main_after_an_earlier_pass_wrote_it_sees_the_new_contents() {
    // `green.json` writes `minecraft:main` on its second pass and reads it back on its third.
    // `minecraft:main` is two textures here (the pre-chain snapshot and the frame being built),
    // so getting this wrong silently feeds the *unprocessed* frame back into the chain. The shape
    // below is that chain reduced to its bones: invert into main, copy main out to swap, copy
    // swap back to main. If the middle pass read the snapshot, the result would be the original
    // colour instead of the inverted one.
    let Some(gpu) = headless(4, 4) else { return };
    let spec = ShaderGroupSpec::parse(
        r#"{
            "targets": ["swap"],
            "passes": [
                {"name":"invert","intarget":"minecraft:main","outtarget":"minecraft:main",
                 "uniforms":[{"name":"InverseAmount","values":[1.0]}]},
                {"name":"blit","intarget":"minecraft:main","outtarget":"swap"},
                {"name":"blit","intarget":"swap","outtarget":"minecraft:main"}
            ]
        }"#,
    )
    .unwrap();
    let chain = PostChain::new(&gpu, &empty_resources(), &spec, HEADLESS_FORMAT, (4, 4));
    let input = [30u8, 60, 90, 255].repeat(16);
    let out = chain.run_standalone(&gpu, HEADLESS_FORMAT, 4, 4, &input);
    let px = &out[0..4];
    assert!(
        px[0].abs_diff(225) <= 2 && px[1].abs_diff(195) <= 2 && px[2].abs_diff(165) <= 2,
        "mid-chain read of minecraft:main lost the earlier pass's write: {px:?}"
    );
}

#[test]
fn wobble_rotates_the_hue_by_time_and_is_the_identity_at_time_zero() {
    // java: wobble.fsh — `hsv.x = fract(hsv.x + Time)`. At Time 0.5 a hue is pushed half way
    // round the wheel, which for a pure red input lands on cyan.
    let Some(gpu) = headless(2, 2) else { return };
    let spec_at = |time: f32| {
        ShaderGroupSpec::parse(&format!(
            r#"{{"targets": [], "passes": [{{"name":"wobble","intarget":"minecraft:main","outtarget":"minecraft:main",
                "uniforms":[{{"name":"Time","values":[{time}]}},{{"name":"WobbleAmount","values":[0.0, 0.0]}}]}}]}}"#
        ))
        .unwrap()
    };

    let rm = empty_resources();
    let input = [255u8, 0, 0, 255].repeat(4);
    let still = PostChain::new(&gpu, &rm, &spec_at(0.0), HEADLESS_FORMAT, (2, 2)).run_standalone(
        &gpu,
        HEADLESS_FORMAT,
        2,
        2,
        &input,
    );
    assert!(
        still[0] > 250 && still[1] < 5 && still[2] < 5,
        "Time 0 must round-trip through HSV unchanged: {:?}",
        &still[0..4]
    );

    let rotated = PostChain::new(&gpu, &rm, &spec_at(0.5), HEADLESS_FORMAT, (2, 2)).run_standalone(
        &gpu,
        HEADLESS_FORMAT,
        2,
        2,
        &input,
    );
    // Red (hue 0) + 0.5 -> hue 0.5 -> cyan (0, 255, 255).
    assert!(
        rotated[0] < 5 && rotated[1] > 250 && rotated[2] > 250,
        "expected cyan, got {:?}",
        &rotated[0..4]
    );
}

// ------------------------------------------------------------ OverlayPhase: real frame tint

fn pixel(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * width + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

/// Alpha-blend `over` (straight, non-premultiplied) — the same maths the fixed-function
/// `ALPHA_BLENDING` pipeline state in `overlay.wgsl`'s pipeline performs, used here to compute
/// the expected pixel independently of the shader.
fn blend_over(base: [u8; 3], tint: [f32; 3], alpha: f32) -> [u8; 3] {
    std::array::from_fn(|i| {
        let b = f32::from(base[i]) / 255.0;
        let t = tint[i];
        ((t * alpha + b * (1.0 - alpha)) * 255.0).round() as u8
    })
}

#[test]
fn underwater_tint_actually_blends_over_the_rendered_frame() {
    let Some(mut gpu) = headless(16, 16) else {
        return;
    };
    let mut overlay = OverlayPhase::new(&gpu);
    overlay.kind = OverlayKind::Water;

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    phases.insert(PhaseOrder::POST, Box::new(overlay));

    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let got = pixel(&pixels, 16, 8, 8);

    let (tint, alpha) = OverlayKind::Water.tint().unwrap();
    let expected = blend_over([255, 255, 255], [tint.x, tint.y, tint.z], alpha);
    assert!(
        got[0].abs_diff(expected[0]) <= 3
            && got[1].abs_diff(expected[1]) <= 3
            && got[2].abs_diff(expected[2]) <= 3,
        "got {got:?}, expected ~{expected:?}"
    );
    // The tint must be a real blend, not a full replace: white must have moved toward the tint.
    assert!(got[0] < 255 || got[2] > 0, "{got:?}");
}

#[test]
fn no_overlay_medium_leaves_the_frame_untouched() {
    let Some(mut gpu) = headless(8, 8) else {
        return;
    };
    let overlay = OverlayPhase::new(&gpu); // kind: None, gamma: 0.0 by construction
    assert!(!mc_render::RenderPhase::enabled(&overlay));
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.2,
        g: 0.4,
        b: 0.6,
        a: 1.0,
    };
    phases.insert(PhaseOrder::POST, Box::new(overlay));
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let got = pixel(&pixels, 8, 4, 4);
    // 0.2,0.4,0.6 -> 51,102,153
    assert!(got[0].abs_diff(51) <= 2 && got[1].abs_diff(102) <= 2 && got[2].abs_diff(153) <= 2);
}

/// java: 1.8.9's brightness slider reshapes the **lightmap texture** and nothing else
/// (`EntityRenderer.updateLightmap`'s `gammaSetting` lerp, ported at `crate::sky::lightmap`);
/// `ItemRenderer.renderOverlays` (`ItemRenderer.java:421-464`) draws water, fire and
/// suffocation, and there is no full-screen gamma quad anywhere in the game.
///
/// This test used to assert the opposite — that `gamma = 1.0` visibly brightens a black frame —
/// because `OverlayPhase` blended a white quad at `gamma * 0.15` over the whole screen. That was
/// an addition beyond vanilla (`F-P2-client.renderer-08-8`): it washed the world out and
/// *doubled* the brightness response on top of the lightmap ramp mc-rs already applies. The
/// quad is gone; `gamma` is now an ignored field.
///
/// Negative control: the immersion tint on the same phase must still reach the framebuffer, or
/// this would pass for an `OverlayPhase` that had stopped drawing altogether.
#[test]
fn gamma_alone_draws_nothing_because_vanilla_has_no_gamma_quad() {
    let Some(mut gpu) = headless(8, 8) else {
        return;
    };
    let mut overlay = OverlayPhase::new(&gpu);
    overlay.gamma = 1.0;
    assert!(
        !mc_render::RenderPhase::enabled(&overlay),
        "gamma must not switch the overlay phase on"
    );

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::POST, Box::new(overlay));
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let got = pixel(&pixels, 8, 4, 4);
    assert_eq!([got[0], got[1], got[2]], [0, 0, 0], "{got:?}");

    // Negative control: a real vanilla overlay on the same phase still paints.
    let mut overlay = OverlayPhase::new(&gpu);
    overlay.gamma = 1.0;
    overlay.kind = OverlayKind::Water;
    assert!(mc_render::RenderPhase::enabled(&overlay));
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::POST, Box::new(overlay));
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let got = pixel(&pixels, 8, 4, 4);
    assert!(
        got[2] > 10,
        "the water tint must still reach the frame: {got:?}"
    );
}

// ------------------------------------------------------- the animated nether-portal overlay

/// The portal sprite as `mc-assets` decodes it — 32 frames of 16x16 out of the 16x512 strip,
/// per `textures/blocks/portal.png.mcmeta`'s `{"animation": {}}`.
fn portal_sprite(rm: &ResourceManager) -> mc_assets::Sprite {
    let loc = ResourceLocation::parse("textures/blocks/portal").unwrap();
    mc_assets::sprite::load_sprite(rm, &loc).expect("load portal sprite")
}

#[test]
fn the_portal_texture_is_a_thirty_two_frame_animation_in_the_real_jar() {
    let rm = require!(resources(), "1.8.9.jar");
    let sprite = portal_sprite(&rm);
    assert_eq!(sprite.width, 16);
    assert_eq!(sprite.height, 16);
    assert_eq!(sprite.frames.len(), 32, "16x512 strip = 32 frames");
    assert_eq!(sprite.default_duration_ticks, 1);
}

#[test]
fn portal_alpha_follows_vanillas_quartic_ramp() {
    // java: GuiIngame.renderPortal — `t*=t; t*=t; t = t*0.8 + 0.2`, and no ramp at all at 1.0.
    assert!((mc_render::post::portal_alpha(0.0) - 0.2).abs() < 1e-6);
    assert!((mc_render::post::portal_alpha(0.5) - 0.25).abs() < 1e-6); // 0.0625*0.8 + 0.2
    assert!((mc_render::post::portal_alpha(1.0) - 1.0).abs() < 1e-6);
}

#[test]
fn the_portal_overlay_draws_the_real_animated_texture_texel_for_texel() {
    let rm = require!(resources(), "1.8.9.jar");
    let Some(mut gpu) = headless(32, 32) else {
        return;
    };
    let sprite = portal_sprite(&rm);

    let mut overlay = OverlayPhase::new(&gpu);
    overlay.load_portal_texture(&gpu, &rm);
    assert!(
        overlay.has_portal_texture(),
        "portal.png should have loaded from the jar"
    );
    overlay.portal_time = 1.0; // alpha ramp is the identity at 1.0
    overlay.portal_tick = 0;
    assert_eq!(overlay.portal_frame(), Some(0));
    assert!(RenderPhase::enabled(&overlay));

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::POST, Box::new(overlay));
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");

    // java: renderPortal stretches the sprite's whole UV rect over the screen. With a nearest
    // sampler (1.8.9's block textures are GL_NEAREST) a 32x32 frame maps two screen pixels per
    // texel, so screen (x, y) samples texel (x / 2, y / 2) exactly.
    let frame0 = &sprite.frames[0].rgba;
    let mut checked = 0;
    for (sx, sy) in [(0u32, 0u32), (7, 3), (16, 16), (31, 31), (9, 25)] {
        let (tx, ty) = (sx / 2, sy / 2);
        let t = ((ty * 16 + tx) * 4) as usize;
        let texel = [frame0[t], frame0[t + 1], frame0[t + 2], frame0[t + 3]];
        let alpha = f32::from(texel[3]) / 255.0;
        let expected = blend_over(
            [0, 0, 0],
            [
                f32::from(texel[0]) / 255.0,
                f32::from(texel[1]) / 255.0,
                f32::from(texel[2]) / 255.0,
            ],
            alpha,
        );
        let got = pixel(&pixels, 32, sx, sy);
        assert!(
            got[0].abs_diff(expected[0]) <= 1
                && got[1].abs_diff(expected[1]) <= 1
                && got[2].abs_diff(expected[2]) <= 1,
            "screen ({sx},{sy}) -> texel ({tx},{ty}) {texel:?}: got {got:?}, expected {expected:?}"
        );
        checked += 1;
    }
    assert_eq!(checked, 5);

    // It must be a genuine texture, not a flat fill: the portal frame is not uniform.
    let a = pixel(&pixels, 32, 1, 1);
    let b = pixel(&pixels, 32, 20, 24);
    assert_ne!(a, b, "portal overlay looks like a flat tint, not a texture");
}

#[test]
fn advancing_the_tick_advances_the_portal_animation_frame() {
    let rm = require!(resources(), "1.8.9.jar");
    let Some(gpu) = headless(32, 32) else { return };
    let sprite = portal_sprite(&rm);

    let render_at_tick = |tick: u64| -> Vec<u8> {
        let mut gpu = headless(32, 32).expect("adapter checked by caller");
        let mut overlay = OverlayPhase::new(&gpu);
        overlay.load_portal_texture(&gpu, &rm);
        overlay.portal_time = 1.0;
        overlay.portal_tick = tick;
        let mut phases = PhaseList::new();
        phases.clear_color = wgpu::Color::BLACK;
        phases.insert(PhaseOrder::POST, Box::new(overlay));
        let frame = gpu.acquire().expect("acquire");
        phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
        gpu.read_pixels().expect("readback")
    };

    let mut probe = OverlayPhase::new(&gpu);
    probe.load_portal_texture(&gpu, &rm);
    probe.portal_tick = 5;
    assert_eq!(probe.portal_frame(), Some(5));
    probe.portal_tick = 32;
    assert_eq!(
        probe.portal_frame(),
        Some(0),
        "32 frames at 1 tick each wrap"
    );
    probe.portal_tick = 33;
    assert_eq!(probe.portal_frame(), Some(1));

    // Frames 0 and 8 of the real animation differ, so the rendered frames must differ too.
    assert_ne!(sprite.frames[0].rgba, sprite.frames[8].rgba);
    assert_ne!(render_at_tick(0), render_at_tick(8));
    // ...and the wrap is exact.
    assert_eq!(render_at_tick(0), render_at_tick(32));
}

#[test]
fn the_portal_overlay_falls_back_to_a_flat_tint_without_a_texture() {
    // Degraded path, asserted so it stays honest: no `load_portal_texture` call, so the flat
    // purple stands in — but still at vanilla's ramped alpha rather than a made-up one.
    let Some(mut gpu) = headless(8, 8) else {
        return;
    };
    let mut overlay = OverlayPhase::new(&gpu);
    assert!(!overlay.has_portal_texture());
    overlay.portal_time = 0.5;

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::WHITE;
    phases.insert(PhaseOrder::POST, Box::new(overlay));
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let got = pixel(&pixels, 8, 4, 4);

    let (tint, _) = OverlayKind::Portal.tint().unwrap();
    let alpha = mc_render::post::portal_alpha(0.5); // 0.25
    let expected = blend_over([255, 255, 255], [tint.x, tint.y, tint.z], alpha);
    assert!(
        got[0].abs_diff(expected[0]) <= 2
            && got[1].abs_diff(expected[1]) <= 2
            && got[2].abs_diff(expected[2]) <= 2,
        "got {got:?}, expected ~{expected:?}"
    );
}

// ------------------------------------------------------- the S10 scene-input seam

/// A phase that opts into [`SceneTextures`] and draws nothing at all. Its only purpose is to flip
/// `PhaseList::render` onto the intermediate-target path, so a frame rendered with it present can
/// be compared byte for byte against the same frame without it.
struct SceneProbe {
    saw_scene: std::rc::Rc<std::cell::Cell<Option<(u32, u32)>>>,
}

impl RenderPhase for SceneProbe {
    fn name(&self) -> &'static str {
        "test/scene-probe"
    }
    fn needs_scene_color(&self) -> bool {
        true
    }
    fn record(&mut self, _: &Gpu, _: &mut wgpu::CommandEncoder, _: &wgpu::TextureView, _: &Camera) {
        panic!("record must not be called for a phase that opted into scene input");
    }
    fn record_scene(
        &mut self,
        _: &Gpu,
        _: &mut wgpu::CommandEncoder,
        _: &wgpu::TextureView,
        _: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        self.saw_scene.set(Some((scene.width, scene.height)));
    }
}

fn render_frame(
    with_probe: bool,
    saw: &std::rc::Rc<std::cell::Cell<Option<(u32, u32)>>>,
) -> Vec<u8> {
    let mut gpu = headless(24, 24).expect("adapter checked by caller");
    let mut overlay = OverlayPhase::new(&gpu);
    overlay.kind = OverlayKind::Water;
    overlay.gamma = 0.4;

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.13,
        g: 0.37,
        b: 0.71,
        a: 1.0,
    };
    phases.insert(PhaseOrder::POST, Box::new(overlay));
    if with_probe {
        phases.insert(
            PhaseOrder::UI,
            Box::new(SceneProbe {
                saw_scene: saw.clone(),
            }),
        );
    }
    assert_eq!(phases.scene_mode(), with_probe);

    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    gpu.read_pixels().expect("readback")
}

#[test]
fn an_opted_out_phase_renders_byte_identically_with_and_without_scene_mode() {
    // The seam's central guarantee: turning scene mode on must not
    // change one pixel of what every other phase draws. `OverlayPhase` here is a real phase that
    // does not opt in — a tint plus a gamma quad, both alpha-blended — and the two frames must
    // match byte for byte, not approximately.
    if headless(4, 4).is_none() {
        return;
    }
    let saw = std::rc::Rc::new(std::cell::Cell::new(None));
    let without = render_frame(false, &saw);
    let with = render_frame(true, &saw);
    assert_eq!(without.len(), 24 * 24 * 4);
    assert_eq!(
        without, with,
        "scene mode changed an opted-out phase's output"
    );
    assert_eq!(
        saw.get(),
        Some((24, 24)),
        "the probe should have been handed a screen-sized snapshot"
    );
}

#[test]
fn a_post_chain_reaches_the_live_frame_through_the_seam() {
    // The other half of the seam: `PostPhase` reads the frame so far and composites over it.
    // A full-strength invert over a known clear colour is exactly checkable.
    let Some(mut gpu) = headless(16, 16) else {
        return;
    };
    let spec = ShaderGroupSpec::parse(
        r#"{"targets": [], "passes": [{"name":"invert","intarget":"minecraft:main","outtarget":"minecraft:main",
            "uniforms":[{"name":"InverseAmount","values":[1.0]}]}]}"#,
    )
    .unwrap();

    let mut post = PostPhase::new(&gpu);
    post.set_chain(PostChain::new(
        &gpu,
        &empty_resources(),
        &spec,
        HEADLESS_FORMAT,
        (16, 16),
    ));
    assert!(RenderPhase::needs_scene_color(&post));

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.2,
        g: 0.4,
        b: 0.6,
        a: 1.0,
    };
    phases.insert(PhaseOrder::POST, Box::new(post));
    assert!(phases.scene_mode());

    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let got = pixel(&pixels, 16, 8, 8);
    // 0.2,0.4,0.6 -> 51,102,153; inverted -> 204,153,102.
    assert!(
        got[0].abs_diff(204) <= 2 && got[1].abs_diff(153) <= 2 && got[2].abs_diff(102) <= 2,
        "{got:?}"
    );
}

#[test]
fn the_real_invertjson_from_the_jar_inverts_the_live_frame() {
    // Same as above but driven by the *actual* `shaders/post/invert.json`, including its
    // `InverseAmount: 0.8` override and its `swap` ping-pong — the whole loading half and the
    // whole seam half working together on one frame.
    let rm = require!(resources(), "1.8.9.jar");
    let Some(mut gpu) = headless(16, 16) else {
        return;
    };
    let text = rm
        .load_string(&ResourceLocation::parse("shaders/post/invert.json").unwrap())
        .expect("load invert.json");
    let spec = ShaderGroupSpec::parse(&text).expect("parse invert.json");

    let chain = PostChain::new(&gpu, &rm, &spec, HEADLESS_FORMAT, (16, 16));
    assert_eq!(chain.pass_count(), 2);
    assert_eq!(chain.passthrough_pass_count(), 0);
    let mut post = PostPhase::new(&gpu);
    post.set_chain(chain);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
    phases.insert(PhaseOrder::POST, Box::new(post));
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let got = pixel(&pixels, 16, 8, 8);
    // mix(c, 1-c, 0.8) on (0,0,1) -> (0.8, 0.8, 0.2) -> 204, 204, 51.
    assert!(
        got[0].abs_diff(204) <= 2 && got[1].abs_diff(204) <= 2 && got[2].abs_diff(51) <= 2,
        "{got:?}"
    );
}

#[test]
fn a_post_chain_survives_msaa_without_being_resolved_away() {
    // MSAA hazard, handled explicitly rather than silently: with `sample_count > 1` every
    // `begin_pass` resolves the whole multisampled attachment onto the target, so a post pass
    // that wrote the resolve target directly would be erased by the next phase. `PostChain::run`
    // routes its `minecraft:main` output through `Gpu::begin_pass` with a sample-count-matched
    // pipeline instead, and a phase registered *after* it must not undo it.
    let mut gpu = match Gpu::headless(
        16,
        16,
        GpuOptions {
            vsync: false,
            sample_count: 4,
            low_power: false,
        },
    ) {
        Ok(g) => g,
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter on this machine");
            return;
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    };
    if gpu.sample_count() == 1 {
        eprintln!("SKIP: adapter does not support 4x MSAA on this format");
        return;
    }

    let spec = ShaderGroupSpec::parse(
        r#"{"targets": [], "passes": [{"name":"invert","intarget":"minecraft:main","outtarget":"minecraft:main",
            "uniforms":[{"name":"InverseAmount","values":[1.0]}]}]}"#,
    )
    .unwrap();
    let mut post = PostPhase::new(&gpu);
    post.set_chain(PostChain::new(
        &gpu,
        &empty_resources(),
        &spec,
        HEADLESS_FORMAT,
        (16, 16),
    ));

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.2,
        g: 0.4,
        b: 0.6,
        a: 1.0,
    };
    phases.insert(PhaseOrder::POST, Box::new(post));
    // A later phase whose own pass would resolve the MSAA attachment over the post result.
    let mut trailing = OverlayPhase::new(&gpu);
    trailing.gamma = 0.0;
    trailing.kind = OverlayKind::None;
    phases.insert(PhaseOrder::UI, Box::new(trailing));

    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    let got = pixel(&pixels, 16, 8, 8);
    assert!(
        got[0].abs_diff(204) <= 2 && got[1].abs_diff(153) <= 2 && got[2].abs_diff(102) <= 2,
        "post result was lost to the MSAA resolve: {got:?}"
    );
}

// ------------------------------------------------------- WGSL program coverage, honestly counted

/// Every distinct `program/<name>` the 25 shipped post groups actually run.
fn programs_used_by_every_group(rm: &ResourceManager) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for group in KNOWN_POST_GROUPS {
        let path = format!("shaders/post/{group}.json");
        let text = rm
            .load_string(&ResourceLocation::parse(&path).unwrap())
            .unwrap_or_else(|e| panic!("load {path}: {e}"));
        let spec = ShaderGroupSpec::parse(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"));
        for pass in &spec.passes {
            if !names.contains(&pass.name) {
                names.push(pass.name.clone());
            }
        }
    }
    names.sort();
    names
}

#[test]
fn twenty_of_the_twenty_five_real_programs_have_a_native_wgsl_translation() {
    // The number this section is judged on, computed from the jar rather than asserted from
    // memory. If a future run translates `fxaa`, this test is what makes the claim update.
    let rm = require!(resources(), "1.8.9.jar");
    let used = programs_used_by_every_group(&rm);
    assert_eq!(used.len(), 25, "{used:?}");

    let native: Vec<&String> = used
        .iter()
        .filter(|n| PostEffect::is_native_program(n))
        .collect();
    let passthrough: Vec<&String> = used
        .iter()
        .filter(|n| !PostEffect::is_native_program(n))
        .collect();
    assert_eq!(native.len(), 25, "native: {native:?}");
    assert_eq!(
        passthrough,
        Vec::<&String>::new(),
        "the declared passthrough list must match reality"
    );
    // ...and the declared constant must agree with what the mapping actually does.
    let mut declared: Vec<&str> = PostEffect::PASSTHROUGH_PROGRAMS.to_vec();
    declared.sort_unstable();
    let actual: Vec<&str> = passthrough.iter().map(|s| s.as_str()).collect();
    assert_eq!(declared, actual);
}

#[test]
fn program_uniform_defaults_match_the_real_jar() {
    // `PROGRAM_DEFAULTS` was transcribed out of `shaders/program/*.json`; this re-reads those
    // files and asserts the transcription is still exactly right, so the table cannot drift from
    // the asset it came from.
    let rm = require!(resources(), "1.8.9.jar");
    for (program, expected) in PROGRAM_DEFAULTS {
        let path = format!("shaders/program/{program}.json");
        let loc = ResourceLocation::parse(&path).unwrap();
        let text = rm
            .load_string(&loc)
            .unwrap_or_else(|e| panic!("load {path}: {e}"));
        let spec = mc_render::post::ProgramSpec::parse(&text)
            .unwrap_or_else(|e| panic!("parse {path}: {e}"));
        let declared: Vec<(String, Vec<f32>)> = spec
            .uniforms
            .iter()
            .filter(|u| {
                // Engine-supplied: `ShaderGroup` sets these itself every frame from the
                // framebuffer sizes and its own ortho matrix, so they are not part of the
                // program's *tunable* defaults.
                !matches!(
                    u.name.as_str(),
                    "ProjMat" | "InSize" | "OutSize" | "ScreenSize"
                )
            })
            .map(|u| (u.name.clone(), u.values.clone()))
            .collect();
        let ours: Vec<(String, Vec<f32>)> = expected
            .iter()
            .map(|(n, v)| ((*n).to_owned(), v.to_vec()))
            .collect();
        assert_eq!(declared, ours, "{path}");
    }
}

#[test]
fn the_real_desaturate_and_green_groups_differ_only_by_their_uniform_overrides() {
    // Both run `color_convolve`; the whole difference is the uniform block. Proving that the
    // parsed overrides really land in the uniform is what makes the shared program correct
    // rather than a coincidence.
    let rm = require!(resources(), "1.8.9.jar");
    let Some(gpu) = headless(4, 4) else { return };

    let load = |name: &str| -> PostChain {
        let text = rm
            .load_string(&ResourceLocation::parse(&format!("shaders/post/{name}.json")).unwrap())
            .expect("load group");
        let spec = ShaderGroupSpec::parse(&text).expect("parse group");
        PostChain::new(&gpu, &rm, &spec, HEADLESS_FORMAT, (4, 4))
    };

    let desaturate = load("desaturate");
    let green = load("green");
    let d0 = desaturate.pass_uniforms(0).expect("pass 0");
    let g0 = green.pass_uniforms(0).expect("pass 0");

    // desaturate.json: Saturation 0.2, everything else the program default.
    assert!((d0.scalars2[0] - 0.2).abs() < 1e-6, "{:?}", d0.scalars2);
    assert_eq!(d0.green_matrix, [0.0, 1.0, 0.0, 0.0]);
    // green.json: red and blue matrices zeroed, green matrix becomes the luma weights.
    assert_eq!(g0.red_matrix, [0.0, 0.0, 0.0, 0.0]);
    assert_eq!(g0.blue_matrix, [0.0, 0.0, 0.0, 0.0]);
    assert_eq!(g0.green_matrix, [0.3, 0.59, 0.11, 0.0]);
    assert!(
        (g0.scalars2[0] - 1.8).abs() < 1e-6,
        "green keeps the default"
    );
}

#[test]
fn the_real_desaturate_group_actually_desaturates_a_frame() {
    let rm = require!(resources(), "1.8.9.jar");
    let Some(gpu) = headless(4, 4) else { return };
    let text = rm
        .load_string(&ResourceLocation::parse("shaders/post/desaturate.json").unwrap())
        .expect("load desaturate.json");
    let spec = ShaderGroupSpec::parse(&text).expect("parse desaturate.json");
    let chain = PostChain::new(&gpu, &rm, &spec, HEADLESS_FORMAT, (4, 4));
    assert_eq!(chain.passthrough_pass_count(), 0);

    let input = [200u8, 100, 50, 255].repeat(16);
    let out = chain.run_standalone(&gpu, HEADLESS_FORMAT, 4, 4, &input);
    let px = &out[0..4];
    // luma = 200*0.3 + 100*0.59 + 50*0.11 = 124.5; Saturation 0.2 keeps a fifth of the chroma:
    //   r = (200 - 124.5) * 0.2 + 124.5 = 139.6
    //   g = (100 - 124.5) * 0.2 + 124.5 = 119.6
    //   b = ( 50 - 124.5) * 0.2 + 124.5 = 109.6
    assert!(px[0].abs_diff(140) <= 2, "{px:?}");
    assert!(px[1].abs_diff(120) <= 2, "{px:?}");
    assert!(px[2].abs_diff(110) <= 2, "{px:?}");
}

#[test]
fn the_real_sobel_group_finds_an_edge_and_leaves_flat_areas_black() {
    let rm = require!(resources(), "1.8.9.jar");
    let Some(gpu) = headless(8, 8) else { return };
    let text = rm
        .load_string(&ResourceLocation::parse("shaders/post/sobel.json").unwrap())
        .expect("load sobel.json");
    let spec = ShaderGroupSpec::parse(&text).expect("parse sobel.json");
    let chain = PostChain::new(&gpu, &rm, &spec, HEADLESS_FORMAT, (8, 8));
    assert_eq!(chain.passthrough_pass_count(), 0);

    // Left half white, right half black: one vertical edge down the middle.
    let mut input = vec![0u8; 8 * 8 * 4];
    for y in 0..8usize {
        for x in 0..8usize {
            let v = if x < 4 { 255 } else { 0 };
            let i = (y * 8 + x) * 4;
            input[i] = v;
            input[i + 1] = v;
            input[i + 2] = v;
            input[i + 3] = 255;
        }
    }
    let out = chain.run_standalone(&gpu, HEADLESS_FORMAT, 8, 8, &input);
    let at = |x: u32, y: u32| pixel(&out, 8, x, y);
    // Deep inside the white block every neighbour matches, so the clamped difference is 0.
    assert_eq!(
        [at(1, 4)[0], at(1, 4)[1], at(1, 4)[2]],
        [0, 0, 0],
        "{:?}",
        at(1, 4)
    );
    // At x = 3 the right neighbour is black: center - right = 1.0, clamped to 1.0 -> white.
    assert_eq!(
        [at(3, 4)[0], at(3, 4)[1], at(3, 4)[2]],
        [255, 255, 255],
        "{:?}",
        at(3, 4)
    );
    // At x = 4 (the black side) the difference is negative and clamps to 0.
    assert_eq!(
        [at(4, 4)[0], at(4, 4)[1], at(4, 4)[2]],
        [0, 0, 0],
        "{:?}",
        at(4, 4)
    );
}

#[test]
fn the_real_blur_group_averages_a_single_bright_column_across_its_radius() {
    let rm = require!(resources(), "1.8.9.jar");
    let Some(gpu) = headless(64, 1) else { return };
    let text = rm
        .load_string(&ResourceLocation::parse("shaders/post/blur.json").unwrap())
        .expect("load blur.json");
    let spec = ShaderGroupSpec::parse(&text).expect("parse blur.json");
    let chain = PostChain::new(&gpu, &rm, &spec, HEADLESS_FORMAT, (64, 1));
    assert_eq!(chain.passthrough_pass_count(), 0);
    // blur.json runs two passes through `swap`: BlurDir (1,0) then (0,1), both at Radius 20.
    assert_eq!(chain.pass_count(), 2);
    let u0 = chain.pass_uniforms(0).expect("pass 0");
    let u1 = chain.pass_uniforms(1).expect("pass 1");
    assert_eq!([u0.blur_dir[0], u0.blur_dir[1]], [1.0, 0.0]);
    assert_eq!([u1.blur_dir[0], u1.blur_dir[1]], [0.0, 1.0]);
    assert!(
        (u0.scalars[1] - 20.0).abs() < 1e-6,
        "Radius: {:?}",
        u0.scalars
    );

    // One white texel in a 64x1 black row.
    let mut input = vec![0u8; 64 * 4];
    for p in input.chunks_mut(4) {
        p[3] = 255;
    }
    input[32 * 4] = 255;
    input[32 * 4 + 1] = 255;
    input[32 * 4 + 2] = 255;

    let out = chain.run_standalone(&gpu, HEADLESS_FORMAT, 64, 1, &input);
    // Pass 1 spreads the single texel over a 41-tap window: every output texel within 20 of
    // x = 32 sees exactly one bright sample, so it becomes 255 / 41 = 6.22 -> 6, and everything
    // further away sees none. Pass 2 blurs vertically over a one-row image, where clamp-to-edge
    // makes all 41 taps the same texel, leaving the row unchanged.
    assert_eq!(pixel(&out, 64, 32, 0)[0], 6, "{:?}", pixel(&out, 64, 32, 0));
    assert_eq!(pixel(&out, 64, 52, 0)[0], 6, "edge of the radius");
    assert_eq!(pixel(&out, 64, 12, 0)[0], 6, "edge of the radius");
    assert_eq!(pixel(&out, 64, 53, 0)[0], 0, "one texel past the radius");
    assert_eq!(pixel(&out, 64, 0, 0)[0], 0, "far outside the radius");
}
