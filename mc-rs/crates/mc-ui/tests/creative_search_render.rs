//! Headless GPU tests for the creative screen's **item grid**, which needs the real baked item
//! models and the stitched block/item atlas installed.
//!
//! This lives in its own test binary on purpose: `mc_assets::install` writes a process-global
//! `OnceLock`, and `tests/container_screens.rs` asserts that tab plates and panel art land
//! texel-for-texel on the source PNG — which only holds while nothing is drawing item icons on
//! top of them. Separate binaries means separate processes, so the two cannot interfere.
//!
//! Skips (loudly) with no wgpu adapter or no vanilla jar, like every other GPU test here.

use std::path::{Path, PathBuf};

use mc_assets::{Atlas, ResourceLocation, ResourceManager, ZipResourcePack};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::FontRenderer;
use mc_ui::screen_stack::{Screen, ScreenCtx};
use mc_ui::screens::container::creative::CreativeTab;
use mc_ui::screens::container::creative_screen::CreativeScreen;
use mc_ui::widget::{keys, DrawList, Modifiers, ScaledResolution, UiPhase, UiTexture};

const W: u32 = 256;
const H: u32 = 256;

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

fn assets() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(rm)
}

/// Bake the real jar's models and install them, handing back the stitched atlas.
fn baked_atlas() -> Option<Atlas> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    let (baked, _report) = mc_assets::bake_all(&rm, 4096).expect("bake_all against the real jar");
    let atlas = baked.atlas.clone();
    mc_assets::install(baked);
    Some(atlas)
}

fn pixel(px: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2], px[i + 3]]
}

fn resolution() -> ScaledResolution {
    let r = ScaledResolution::new(W, H, 1, false);
    assert_eq!(r.scale_factor(), 1);
    r
}

fn ctx() -> ScreenCtx {
    ScreenCtx::new(resolution(), 0.0)
}

fn atlas_key() -> UiTexture {
    UiTexture::Resource(ResourceLocation::minecraft("mc-ui/creative-test-atlas"))
}

/// Draw `screen` and read the framebuffer back, with the baked atlas uploaded alongside the
/// jar's own GUI textures.
fn shoot(
    gpu: &mut Gpu,
    rm: &ResourceManager,
    atlas: &Atlas,
    screen: &mut CreativeScreen,
    font: &mut FontRenderer,
) -> Vec<u8> {
    let mut list = DrawList::new();
    // The mouse is parked in the far corner so nothing draws a hover highlight or tooltip.
    screen.draw(&mut list, font, &ctx(), (W as i32 - 1, H as i32 - 1));

    let mut phase = UiPhase::new(gpu, W, H, 1, false);
    phase
        .renderer
        .upload(gpu, atlas_key(), &atlas.texture, atlas.w, atlas.h);
    let missing = phase.renderer.ensure_list_textures(gpu, rm, &list);
    assert!(missing.is_empty(), "missing textures: {missing:?}");
    phase.set_resolution(resolution());
    phase.submit(list);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    gpu.read_pixels().expect("readback")
}

/// How many pixels differ between two shots inside a 16x16 cell at window offset `(x, y)`.
fn cell_diff(a: &[u8], b: &[u8], screen: &CreativeScreen, x: i32, y: i32) -> usize {
    let (cx, cy) = (
        (screen.gui_left() + x) as u32,
        (screen.gui_top() + y) as u32,
    );
    (0..16u32)
        .flat_map(|dy| (0..16u32).map(move |dx| (dx, dy)))
        .filter(|(dx, dy)| pixel(a, cx + dx, cy + dy) != pixel(b, cx + dx, cy + dy))
        .count()
}

/// Typing into the search box has to change what the grid *renders*, not just what a `Vec`
/// holds. This is the end-to-end proof for the search half of the creative task.
#[test]
fn typing_a_search_query_changes_the_pixels_in_the_item_grid() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: vanilla jar not found");
        return;
    };
    let Some(atlas) = baked_atlas() else {
        eprintln!("SKIP: vanilla jar not found");
        return;
    };
    let mut font = FontRenderer::load(&rm, false, false).expect("font loads");

    let mut screen = CreativeScreen::new();
    screen.atlas = atlas_key();
    screen.init(&ctx());
    screen.set_tab(CreativeTab::Search.index().expect("search is registered"));
    screen.apply_pending_input(&font);
    // java: `setCurrentCreativeTab` clears the box and re-runs the search, and an empty needle
    // matches every tooltip line -- so the tab opens showing the whole pool, not blank.
    assert!(
        !screen.item_list().is_empty(),
        "an empty query lists the whole pool"
    );

    let empty_px = shoot(&mut gpu, &rm, &atlas, &mut screen, &mut font);

    for c in "sword".chars() {
        screen.key_typed_with_font(Some(c), 0, Modifiers::NONE, &font);
    }
    assert_eq!(screen.search.text(), "sword");
    assert!(
        !screen.item_list().is_empty(),
        "searching for 'sword' must find items"
    );
    let filled_px = shoot(&mut gpu, &rm, &atlas, &mut screen, &mut font);

    // The first grid cell is at window (9, 18): empty before, an item icon after.
    let painted = cell_diff(&empty_px, &filled_px, &screen, 9, 18);
    assert!(
        painted > 20,
        "the first grid cell has to gain an item icon: only {painted} pixels changed"
    );

    // The typed query itself has to reach the framebuffer inside the 89x9 search box.
    let (bx, by) = (
        (screen.gui_left() + 82) as u32,
        (screen.gui_top() + 6) as u32,
    );
    let text_pixels = (0..9u32)
        .flat_map(|dy| (0..89u32).map(move |dx| (dx, dy)))
        .filter(|(dx, dy)| {
            pixel(&empty_px, bx + dx, by + dy) != pixel(&filled_px, bx + dx, by + dy)
        })
        .count();
    assert!(
        text_pixels > 10,
        "the search text has to be drawn: only {text_pixels} pixels changed in the box"
    );

    // Clearing the query restores the *whole pool*, in pixels and not just in a `Vec` -- java
    // re-runs the search with an empty needle, which matches everything.
    for _ in 0..5 {
        screen.key_typed_with_font(None, keys::BACKSPACE, Modifiers::NONE, &font);
    }
    assert_eq!(screen.search.text(), "");
    assert!(!screen.item_list().is_empty());
    let cleared_px = shoot(&mut gpu, &rm, &atlas, &mut screen, &mut font);
    assert_eq!(
        cell_diff(&empty_px, &cleared_px, &screen, 9, 18),
        0,
        "clearing the query has to put the same grid back on screen"
    );
}

/// Scrolling has to move real icons, not just an index: the row that was below the fold has to
/// end up rendered in the top row.
#[test]
fn scrolling_the_creative_grid_repaints_it_with_the_next_rows_icons() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: vanilla jar not found");
        return;
    };
    let Some(atlas) = baked_atlas() else {
        eprintln!("SKIP: vanilla jar not found");
        return;
    };
    let mut font = FontRenderer::load(&rm, false, false).expect("font loads");

    let mut screen = CreativeScreen::new();
    screen.atlas = atlas_key();
    screen.init(&ctx());
    screen.set_tab(0);
    assert!(
        screen.needs_scroll_bars(),
        "the building-blocks tab must overflow one page"
    );

    let top_px = shoot(&mut gpu, &rm, &atlas, &mut screen, &mut font);
    let top_row_before: Vec<_> = screen.grid()[..9].to_vec();

    screen.scroll_by_wheel(-1.0);
    assert!(screen.scroll() > 0.0, "the wheel has to scroll down");
    let scrolled_px = shoot(&mut gpu, &rm, &atlas, &mut screen, &mut font);
    let top_row_after: Vec<_> = screen.grid()[..9].to_vec();
    assert_ne!(
        top_row_before, top_row_after,
        "a scroll has to change which items the top row holds"
    );

    // At least one of the nine top-row cells has to look different on screen.
    let changed_cells = (0..9)
        .filter(|column| cell_diff(&top_px, &scrolled_px, &screen, 9 + column * 18, 18) > 0)
        .count();
    assert!(
        changed_cells >= 1,
        "scrolling repainted none of the top row's nine cells"
    );

    // Scrolling all the way back restores the original pixels exactly.
    for _ in 0..500 {
        screen.scroll_by_wheel(1.0);
    }
    assert_eq!(screen.scroll(), 0.0);
    let back_px = shoot(&mut gpu, &rm, &atlas, &mut screen, &mut font);
    for column in 0..9 {
        assert_eq!(
            cell_diff(&top_px, &back_px, &screen, 9 + column * 18, 18),
            0,
            "cell {column} did not come back to its original pixels"
        );
    }
}
