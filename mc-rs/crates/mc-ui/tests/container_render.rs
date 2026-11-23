//! Headless GPU test for item-in-slot rendering (S17), mirroring S15's
//! `tests/headless_ui.rs` technique: render through the real `UI 800` phase and read the
//! framebuffer back, so "the icon actually landed in the slot" is a checkable fact.
//!
//! This covers the **flat 2D icon** path ([`mc_ui::screens::container::render::draw_flat_icon`])
//! end to end against the real vanilla jar's baked item model and atlas — the isometric 3D block
//! path is covered by CPU-only geometry tests in `render.rs` itself (see that module's doc for
//! why: a `BakedQuad`'s winding order is not guaranteed, so the iso path builds its own
//! geometry rather than consuming baked quads directly, and what is being tested there is the
//! projection math, not a texture landing on screen).
//!
//! Skips (loudly) when there is no wgpu adapter or the vanilla jar is absent, exactly like
//! `headless_ui.rs`.

use std::path::{Path, PathBuf};

use mc_assets::{
    bake_all, install, item_model_for, ResourceLocation, ResourceManager, ZipResourcePack,
};
use mc_data::item_by_name;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::screens::container::render::draw_flat_icon;
use mc_ui::widget::{DrawList, ScaledResolution, UiPhase, UiTexture};

const W: u32 = 64;
const H: u32 = 64;

fn gpu() -> Option<Gpu> {
    match Gpu::headless(
        W,
        H,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(g) => Some(g),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

fn jars_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

/// Loads the vanilla jar and bakes+installs its models/atlas, exactly once per test process
/// (`mc_assets::install`'s registry is a `OnceLock` — later calls are harmless no-ops).
fn baked_assets() -> Option<mc_assets::Atlas> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    let (baked, _report) = bake_all(&rm, 4096).expect("bake_all against the real jar");
    let atlas = baked.atlas.clone();
    install(baked);
    Some(atlas)
}

fn pixel(px: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2], px[i + 3]]
}

#[test]
fn a_flat_item_icon_actually_renders_non_background_pixels_in_its_slot() {
    let Some(mut gpu) = gpu() else { return };
    let Some(atlas) = baked_assets() else {
        eprintln!("SKIP: vanilla jar not found");
        return;
    };

    // A stick is a plain `Item` (not `ItemBlock`), so vanilla bakes it through the flat
    // `builtin/generated` path — exactly what `draw_flat_icon` is for.
    let stick = item_by_name("stick").expect("stick is a real 1.8.9 item");
    let model = item_model_for(stick);
    assert!(
        !model.is_empty(),
        "the real jar must have baked a stick model"
    );

    let atlas_key = UiTexture::Resource(ResourceLocation::minecraft("mc-ui/container-test-atlas"));
    let mut phase = UiPhase::new(&gpu, W, H, 1, false);
    phase
        .renderer
        .upload(&gpu, atlas_key.clone(), &atlas.texture, atlas.w, atlas.h);

    let mut list = DrawList::new();
    // Slot drawn at (16, 16): a 16x16 icon box away from every framebuffer edge.
    draw_flat_icon(&mut list, &atlas_key, 16, 16, model);
    assert!(!list.is_empty(), "the flat icon must have emitted geometry");

    let resolution = ScaledResolution::new(W, H, 1, false);
    phase.set_resolution(resolution);
    phase.submit(list);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    let px = gpu.read_pixels().expect("readback");

    // The icon must have painted *something* inside its 16x16 box...
    let mut painted = 0;
    for dy in 0..16 {
        for dx in 0..16 {
            if pixel(&px, 16 + dx, 16 + dy) != [0, 0, 0, 255] {
                painted += 1;
            }
        }
    }
    assert!(
        painted > 0,
        "the icon box is entirely background — nothing was drawn"
    );

    // ...and must not have painted the untouched corner of the framebuffer.
    assert_eq!(
        pixel(&px, 1, 1),
        [0, 0, 0, 255],
        "geometry leaked outside the slot"
    );
}

#[test]
fn every_registered_item_bakes_to_a_model_without_panicking() {
    let Some(_atlas) = baked_assets() else {
        eprintln!("SKIP: vanilla jar not found");
        return;
    };
    // Not a rendering assertion (no GPU needed) — a coverage sweep proving
    // `item_model_for`/`draw_flat_icon`/`draw_iso_block_icon`'s geometry code paths tolerate
    // every real registered item, including the ones with no baked quads at all (air-like/
    // unrenderable ids), without panicking.
    use mc_ui::screens::container::render::{draw_flat_icon, draw_iso_block_icon};
    let tex = UiTexture::Resource(ResourceLocation::minecraft("mc-ui/container-test-atlas"));
    for props in mc_data::items() {
        if !props.registered {
            continue;
        }
        let model = item_model_for(props.id);
        let mut list = DrawList::new();
        if props.block.is_some() {
            draw_iso_block_icon(&mut list, &tex, 0, 0, model);
        } else {
            draw_flat_icon(&mut list, &tex, 0, 0, model);
        }
    }
}
