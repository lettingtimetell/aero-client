//! Found: the scoreboard sidebar's text colour is
//! `553648127` = `0x20FFFFFF`, a 12.5% alpha.
//!
//! Vanilla still shows it opaque. In `GuiIngame.renderScoreboard` every `drawString` follows a
//! `Gui.drawRect`, and `drawRect` ends with `GlStateManager.disableBlend()`. `FontRenderer` never
//! touches the blend state: `drawString` only calls `enableAlpha()`, and `renderString` keeps the
//! 0x20 alpha (`(color & 0xFC000000) != 0`) in `GlStateManager.color`. So the glyphs are drawn
//! with blending off under `alphaFunc(GL_GREATER, 0.1F)`: a glyph texel's fragment alpha is
//! `1.0 * 0.125 > 0.1`, it passes, and its colour is written as is, opaque. A clear texel
//! (`0.0`) is discarded.
//!
//! These tests render the real `Hud::draw_scoreboard` (the shared `paint_sidebar`) through
//! `UiPhase` with the jar's font and read the frame back. Vanilla's output is binary: every pixel
//! of the board is either the plate over the background or the text colour. Skips (loudly)
//! without a wgpu adapter or the jar; set `MC_JARS_DIR`.

use std::path::{Path, PathBuf};

use mc_assets::{ResourceManager, ZipResourcePack};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::FontRenderer;
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::hud::{Hud, Sidebar};
use mc_ui::widget::{DrawList, ScaledResolution, UiPhase};

const W: u32 = 640;
const H: u32 = 480;
const SCALE: i32 = 2;

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
        Ok(gpu) => Some(gpu),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            None
        }
        Err(error) => panic!("headless context creation failed: {error}"),
    }
}

fn resources() -> Option<ResourceManager> {
    let dir = match std::env::var("MC_JARS_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)?
            .join("MCP-919-main")
            .join("jars"),
    };
    let jar = dir.join("versions").join("1.8.9").join("1.8.9.jar");
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(rm)
}

fn resolution() -> ScaledResolution {
    let resolution = ScaledResolution::new(W, H, SCALE, false);
    assert_eq!(resolution.scale_factor(), SCALE);
    resolution
}

/// A full-screen opaque background, then whatever `paint` draws with the jar's font.
fn render(
    gpu: &mut Gpu,
    rm: &ResourceManager,
    background: [u8; 3],
    paint: impl FnOnce(&mut DrawList, &mut FontRenderer, &ScreenCtx),
) -> Vec<u8> {
    let ctx = ScreenCtx::new(resolution(), 0.0);
    let mut font = FontRenderer::load(rm, false, false).expect("the jar's font");
    let mut d = DrawList::new();
    let [r, g, b] = background;
    let argb = 0xFF00_0000 | (u32::from(r) << 16) | (u32::from(g) << 8) | u32::from(b);
    d.rect(0, 0, ctx.width, ctx.height, argb);
    paint(&mut d, &mut font, &ctx);

    let mut phase = UiPhase::new(gpu, W, H, SCALE, false);
    let missing = phase.renderer.ensure_list_textures(gpu, rm, &d);
    assert!(missing.is_empty(), "textures did not load: {missing:?}");
    phase.set_resolution(resolution());
    phase.submit(d);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    gpu.read_pixels().expect("readback")
}

fn rgb(px: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * W + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2]]
}

/// `Gui.drawRect` of black at `alpha` over `background`: `dst * (1 - a)`, rounded as unorm8.
fn plate_over(background: [u8; 3], alpha: u8) -> [u8; 3] {
    let keep = 1.0 - f32::from(alpha) / 255.0;
    background.map(|c| (f32::from(c) * keep).round() as u8)
}

fn near(a: [u8; 3], b: [u8; 3]) -> bool {
    a.iter().zip(b).all(|(&x, y)| x.abs_diff(y) <= 1)
}

/// What fraction of the way from `plate` to `ink` a pixel sits, over its most separated
/// channel: 1.0 is opaque text, 0.125 is the 0x20 alpha blended in.
fn opacity(pixel: [u8; 3], plate: [u8; 3], ink: [u8; 3]) -> f32 {
    let c = (0..3)
        .max_by_key(|&c| ink[c].abs_diff(plate[c]))
        .expect("three channels");
    (f32::from(pixel[c]) - f32::from(plate[c])) / (f32::from(ink[c]) - f32::from(plate[c]))
}

/// White names and title, and `§c` scores: `EnumChatFormatting.RED` is `0xFF5555`.
const WHITE: [u8; 3] = [255, 255, 255];
const RED: [u8; 3] = [255, 85, 85];

#[test]
fn sidebar_text_is_opaque_like_vanillas_blend_off_draw() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found; set MC_JARS_DIR");
        return;
    };
    let sidebar = Sidebar::from_scores(
        "Board",
        &[("Alice".to_owned(), 3), ("Bob".to_owned(), 7)],
        str::to_owned,
    );
    for background in [[51, 102, 153], [128, 128, 128], [0, 0, 0], [240, 240, 240]] {
        let px = render(&mut gpu, &rm, background, |d, font, ctx| {
            Hud::new().draw_scoreboard(d, font, ctx, &sidebar);
        });
        let row_plate = plate_over(background, 0x50);
        let header_plate = plate_over(background, 0x60);
        let mut ink = [0usize; 2];
        let mut worst: Option<(u32, u32, [u8; 3], f32)> = None;
        // The board hugs the right edge; scan the right third of the screen.
        for y in 0..H {
            for x in W * 2 / 3..W {
                let p = rgb(&px, x, y);
                if near(p, background) || near(p, row_plate) || near(p, header_plate) {
                    continue;
                }
                if near(p, WHITE) {
                    ink[0] += 1;
                    continue;
                }
                if near(p, RED) {
                    ink[1] += 1;
                    continue;
                }
                // Neither plate nor ink: blended text. Measure how opaque it came out.
                let target = if p[1] < p[0] && p[2] < p[0] {
                    RED
                } else {
                    WHITE
                };
                let alpha = opacity(p, row_plate, target);
                if worst.is_none_or(|(.., a)| alpha < a) {
                    worst = Some((x, y, p, alpha));
                }
            }
        }
        if let Some((x, y, p, alpha)) = worst {
            panic!(
                "background {background:?}: text pixel ({x}, {y}) is {p:?}, between plate \
                 {row_plate:?} and ink, at {:.1}% opacity; vanilla writes the 0x20FFFFFF \
                 glyphs opaque (blending off)",
                alpha * 100.0
            );
        }
        // "Board", "Alice", "Bob" in white and "3", "7" in red, at 2x2 pixels per texel.
        assert!(ink[0] > 200, "white text drawn: {ink:?} on {background:?}");
        assert!(ink[1] > 20, "red scores drawn: {ink:?} on {background:?}");
    }
}

/// Negative control, and the confirmation of the finding: the same colour through the
/// ordinary blended text sink (how `paint_sidebar` drew it before this fix) lands at its 0x20
/// alpha, 12.5%.
#[test]
fn the_blended_text_sink_draws_the_sidebar_colour_at_an_eighth() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found; set MC_JARS_DIR");
        return;
    };
    let background = [51, 102, 153];
    let px = render(&mut gpu, &rm, background, |d, font, _| {
        font.draw_string(d, "Board", 40.0, 40.0, 0x20FF_FFFF, false);
    });
    let mut text = 0;
    for y in 0..H {
        for x in 0..W {
            let p = rgb(&px, x, y);
            if near(p, background) {
                continue;
            }
            text += 1;
            let alpha = opacity(p, background, WHITE);
            assert!(
                (alpha - 32.0 / 255.0).abs() < 0.01,
                "pixel ({x}, {y}) is {p:?}: {:.1}% opacity, not the 12.5% blend",
                alpha * 100.0
            );
        }
    }
    eprintln!("blended 0x20FFFFFF text: {text} pixels, all at 12.5% opacity");
    assert!(text > 100, "the control string was drawn: {text} pixels");
}
