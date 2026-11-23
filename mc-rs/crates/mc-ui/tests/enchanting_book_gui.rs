//! "Book page pokes out when book is closed when inside enchanting
//! table inventory".
//!
//! `GuiEnchantment.drawGuiContainerBackgroundLayer` (`:113-169`) draws `ModelBook` through its
//! own `gluPerspective` with the depth test on — `GuiContainer.drawScreen` only disables depth
//! *after* the background layer, and `EntityRenderer.updateCameraAndRender` cleared the depth
//! buffer just before `drawScreen` — and with `GL_CULL_FACE`/`GL_BACK` inherited from the world
//! pass (`EntityRenderer.renderWorldPass` leaves `enableCull()` set). A closed book is two
//! cover quads enclosing the page blocks, so vanilla shows nothing but cover.
//!
//! These render the real `ContainerScreen` through the real UI phase on a headless GPU, with
//! the book texture replaced by a colour key — covers red, spine blue, the side faces of the
//! page blocks and both flipping pages green — and read the framebuffer back.
//!
//! Skips (loudly) with no wgpu adapter or no vanilla jar, like its siblings.

use std::path::{Path, PathBuf};

use mc_assets::{ResourceLocation, ResourceManager, ZipResourcePack};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::FontRenderer;
use mc_ui::screen_stack::{Screen, ScreenCtx};
use mc_ui::screens::container::screen::ContainerScreen;
use mc_ui::widget::{DrawList, ScaledResolution, UiPhase, UiTexture};

const W: u32 = 256;
const H: u32 = 256;

/// `TileEntityEnchantmentTableRenderer`/`GuiEnchantment`'s shared book sheet.
const BOOK_TEXTURE: &str = "textures/entity/enchanting_table_book.png";

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

fn resolution() -> ScaledResolution {
    let r = ScaledResolution::new(W, H, 1, false);
    assert_eq!(r.scale_factor(), 1);
    r
}

/// The 64x32 `ModelBook` sheet, colour-keyed by the `ModelBook` texture offsets:
/// `coverRight (0, 0)` and `coverLeft (16, 0)` are 6x10x0 boxes, `bookSpine (12, 0)` 2x10x0,
/// `pagesRight (0, 10)`/`pagesLeft (12, 10)` 5x8x1 and the flipping pages `(24, 10)` 5x8x0.
///
/// A zero-depth box has only its two Z faces: `ModelBox` quad 4 (`-Z`) samples
/// `u = texU..texU+6` and quad 5 (`+Z`) samples `u = texU+6..texU+12`. Quad-4 cover texels are
/// magenta and quad-5 cover texels red, so the colour also says *which side* of a cover won —
/// the thing a back-face cull decides.
///
/// Row `v = 10` holds only the page blocks' 5x1 top and bottom faces; it is left white so a
/// cover edge sampled exactly on its `v = 10` boundary cannot read as a page.
fn keyed_book_texture() -> Vec<u8> {
    let mut rgba = vec![0u8; 64 * 32 * 4];
    for v in 0..32usize {
        for u in 0..64usize {
            let colour = if v < 10 && (u < 6 || (16..22).contains(&u)) {
                [255, 0, 255, 255]
            } else if v < 10 && ((6..12).contains(&u) || (22..28).contains(&u)) {
                [255, 0, 0, 255]
            } else if v < 10 && (12..16).contains(&u) {
                [0, 0, 255, 255]
            } else if (11..19).contains(&v) && u < 34 {
                [0, 255, 0, 255]
            } else {
                [255, 255, 255, 255]
            };
            let i = (v * 64 + u) * 4;
            rgba[i..i + 4].copy_from_slice(&colour);
        }
    }
    rgba
}

/// Render one `DrawList`, loading its textures from the jar, then swapping the book sheet for
/// [`keyed_book_texture`].
fn render(gpu: &mut Gpu, rm: &ResourceManager, list: DrawList) -> Vec<u8> {
    let mut phase = UiPhase::new(gpu, W, H, 1, false);
    let missing = phase.renderer.ensure_list_textures(gpu, rm, &list);
    assert!(
        missing.is_empty(),
        "textures missing from the jar: {missing:?}"
    );
    phase.renderer.upload(
        gpu,
        UiTexture::Resource(ResourceLocation::minecraft(BOOK_TEXTURE)),
        &keyed_book_texture(),
        64,
        32,
    );
    phase.set_resolution(resolution());
    phase.submit(list);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    gpu.read_pixels().expect("readback")
}

/// Pixel counts of each key colour inside the panel's book corner.
#[derive(Debug, Default)]
struct BookPixels {
    /// `ModelBox` quad 5 of either cover (red).
    cover_quad5: usize,
    /// `ModelBox` quad 4 of either cover (magenta).
    cover_quad4: usize,
    /// Page-block sides and flipping pages (green).
    page: usize,
}

/// Lighting only scales the key colours, so the dominant channels identify the texel's region.
fn count_book_pixels(px: &[u8], left: i32, top: i32) -> BookPixels {
    let mut counts = BookPixels::default();
    // The closed and the open book both project inside GUI (17..51, 16..42); the panel title
    // above and the item slots below stay outside this window.
    for y in (top + 8)..(top + 46) {
        for x in (left + 10)..(left + 58) {
            let i = ((y as u32 * W + x as u32) * 4) as usize;
            let (r, g, b) = (u32::from(px[i]), u32::from(px[i + 1]), u32::from(px[i + 2]));
            if r > 40 && r > 2 * g && r > 2 * b {
                counts.cover_quad5 += 1;
            } else if r > 40 && b > 40 && r > 2 * g && b > 2 * g {
                counts.cover_quad4 += 1;
            } else if g > 40 && g > 2 * r && g > 2 * b {
                counts.page += 1;
            }
        }
    }
    counts
}

#[test]
fn the_closed_enchanting_book_shows_only_its_cover() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: vanilla jar not found");
        return;
    };
    let mut font = FontRenderer::load(&rm, false, false).expect("font loads");
    let ctx = ScreenCtx::new(resolution(), 0.0);

    // A fresh screen: no enchant offers, so `field_147080_z`/`field_147076_A` are both 0 and the
    // book is fully closed (`GuiEnchantment.func_147068_g` only opens it for a non-zero offer).
    let mut screen = ContainerScreen::enchanting_table(1, "Enchant");
    screen.init(&ctx);
    let pose = screen
        .enchanting_book_pose(0.0)
        .expect("the enchanting screen owns a book");
    assert_eq!(pose.spread, 0.0, "the book starts closed");

    let mut list = DrawList::new();
    screen.draw(&mut list, &mut font, &ctx, (0, 0));
    let px = render(&mut gpu, &rm, list);
    let counts = count_book_pixels(&px, screen.gui_left(), screen.gui_top());
    eprintln!("closed book: {counts:?}");

    // A reference z-buffer rasterisation of vanilla's state for this pose (cull on, LEQUAL over
    // the panel's 0.5 depth; scratch `book_raster.py` in the L7 report) puts ~366 GUI pixels of
    // `coverLeft`'s quad 5 in front, ~32 of `coverRight`'s quad 4 peeking past its far edge, and
    // no page at all.
    assert_eq!(counts.page, 0, "a closed book shows no page: {counts:?}");
    assert!(
        counts.cover_quad5 > 250,
        "the camera-facing cover side is quad 5 of coverLeft: {counts:?}"
    );
    assert!(
        counts.cover_quad4 * 4 < counts.cover_quad5,
        "quad 4 faces the pages, so it is culled on the near cover: {counts:?}"
    );
}
