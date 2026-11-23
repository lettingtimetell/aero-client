//! "Lines over O's with small letters in minecraft
//! font (ex. chat, spotify integration)".
//!
//! In `ascii.png` the `_` glyph (0x5F) sits directly above `o` (0x6F), and `_` is ink only in its
//! cell's bottom row. A glyph quad's top edge maps to the first texel row of its cell; when that
//! edge lands exactly on a pixel centre, the rasteriser covers the pixel but the interpolated
//! `v` can come out a hair *above* the cell, and nearest filtering then samples the bottom row of
//! the glyph above. Integer GUI scales never put an edge on a pixel centre; fractional ones
//! (chat scale, the Spotify widget's scale, HUD scale) do. The user's own settings (guiScale 3,
//! chatScale 0.6690141 = 95/142) put the 8th chat line's top edge at exactly 853.5 px.
//!
//! These tests draw `o` through the real font renderer and the real `UI 800` phase, read the
//! frame back, and check the rows that sample the `o` cell's empty top rows: any lit pixel there
//! is ink from another glyph.
//!
//! Skips (loudly) when there is no wgpu adapter or the vanilla jar is absent.

use std::path::{Path, PathBuf};

use mc_assets::{ResourceLocation, ResourceManager, ZipResourcePack};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::FontRenderer;
use mc_ui::widget::{DrawList, ScaledResolution, Transform, UiPhase};

fn gpu(width: u32, height: u32) -> Option<Gpu> {
    match Gpu::headless(
        width,
        height,
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

fn resources() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(rm)
}

/// How many rows at the top of `ch`'s 8x8 cell carry no ink, read from the real `ascii.png`.
fn empty_top_rows(rm: &ResourceManager, ch: u8) -> u32 {
    let (pixels, width, height, _) = rm
        .load_png_rgba(&ResourceLocation::minecraft("textures/font/ascii.png"))
        .expect("ascii.png");
    assert_eq!((width, height), (128, 128), "the vanilla sheet");
    let (cx, cy) = (u32::from(ch % 16) * 8, u32::from(ch / 16) * 8);
    let clear =
        |row: u32, col: u32| pixels[(((cy + row) * width + cx + col) * 4 + 3) as usize] == 0;
    (0..8)
        .take_while(|&row| (0..8).all(|col| clear(row, col)))
        .count() as u32
}

/// Whether the bottom row of `ch`'s cell has ink (what bleeds into the cell below).
fn bottom_row_inked(rm: &ResourceManager, ch: u8) -> bool {
    let (pixels, width, _, _) = rm
        .load_png_rgba(&ResourceLocation::minecraft("textures/font/ascii.png"))
        .expect("ascii.png");
    let (cx, cy) = (u32::from(ch % 16) * 8, u32::from(ch / 16) * 8 + 7);
    (0..8).any(|col| pixels[((cy * width + cx + col) * 4 + 3) as usize] != 0)
}

/// One drawn `o`: where its quad's top-left lands in physical pixels and how many physical
/// pixels one font pixel spans.
#[derive(Debug, Clone, Copy)]
struct Placed {
    x: f32,
    y: f32,
    px_per_unit: f32,
    label: (f32, f32),
}

/// Render `draws` (GUI-space transforms applied to an `o` at the origin) and return the frame.
fn render(
    gpu: &mut Gpu,
    rm: &ResourceManager,
    size: (u32, u32),
    gui_scale: i32,
    text: &str,
    draws: &[Transform],
) -> Vec<u8> {
    let mut font = FontRenderer::load(rm, false, false).expect("font");
    let mut list = DrawList::new();
    for transform in draws {
        list.with_transform(*transform, |d| {
            font.draw_string(d, text, 0.0, 0.0, 0x00FF_FFFF, false);
        });
    }
    let mut phase = UiPhase::new(gpu, size.0, size.1, gui_scale, false);
    let missing = phase.renderer.ensure_list_textures(gpu, rm, &list);
    assert!(missing.is_empty(), "textures did not load: {missing:?}");
    phase.set_resolution(ScaledResolution::new(size.0, size.1, gui_scale, false));
    phase.submit(list);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let target = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, target.view(), &Camera::default(), 0.0);
    gpu.read_pixels().expect("readback")
}

/// Every lit pixel whose centre samples one of the `o` cell's `empty_rows` top rows.
fn bleeding_pixels(frame: &[u8], width: u32, placed: &[Placed], empty_rows: u32) -> Vec<String> {
    let height = (frame.len() / 4) as u32 / width;
    let mut out = Vec::new();
    // 'o' is 6 wide (5 px of ink plus spacing): the quad spans 4.99 font pixels.
    // Glyphs at a fractional scale are box-filtered (the
    // `fs_text` sharp sampler), so the `o`'s own first inked row reaches up to half a pixel
    // above its texel seam. That is the `o`'s ink, not another glyph's: stop the band there.
    let own_ink = if mc_ui::widget::scaled::sharp_text() {
        0.5
    } else {
        0.0
    };
    for p in placed {
        let band_top = p.y;
        // Stop a little short of the first inked row so a texel-boundary rounding question on
        // the far side of the band can never count.
        let band_bottom = p.y + (empty_rows as f32 - 0.05) * p.px_per_unit - own_ink;
        let x0 = p.x.floor().max(0.0) as u32;
        let x1 = ((p.x + 4.99 * p.px_per_unit).ceil() as u32).min(width);
        let y0 = band_top.floor().max(0.0) as u32;
        let y1 = (band_bottom.ceil() as u32).min(height);
        for py in y0..y1 {
            let centre_y = py as f32 + 0.5;
            if centre_y < band_top || centre_y >= band_bottom {
                continue;
            }
            for px in x0..x1 {
                let centre_x = px as f32 + 0.5;
                if centre_x < p.x || centre_x > p.x + 4.99 * p.px_per_unit {
                    continue;
                }
                let i = ((py * width + px) * 4) as usize;
                let lum = u32::from(frame[i]) + u32::from(frame[i + 1]) + u32::from(frame[i + 2]);
                if lum > 3 * 40 {
                    out.push(format!(
                        "scale {:.4} offset {:.5}: pixel ({px},{py}) lit {:?} in the empty band \
                         y {:.4}..{:.4}",
                        p.label.0,
                        p.label.1,
                        &frame[i..i + 3],
                        band_top,
                        band_bottom
                    ));
                }
            }
        }
    }
    out
}

/// The sweep: `o` at many sub-pixel vertical offsets (dense around the half-pixel point) and a
/// spread of fractional scales, drawn at GUI scale 1 so GUI units are physical pixels.
#[test]
fn an_o_never_shows_the_underscore_above_it_at_any_scale_or_offset() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found; set MC_JARS_DIR to the jars/ directory");
        return;
    };
    assert!(bottom_row_inked(&rm, b'_'), "'_' is ink in its bottom row");
    let empty = empty_top_rows(&rm, b'o');
    assert!(empty >= 1, "the 'o' cell starts with an empty row");

    const W: u32 = 1024;
    const H: u32 = 1024;
    let Some(mut gpu) = gpu(W, H) else { return };

    // 285/142 is the user's chat (guiScale 3 x chatScale 95/142); 2.4 the Spotify widget at
    // 0.8 on guiScale 3; the rest a spread of HUD/chat scales.
    let scales = [
        0.75f32,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        285.0 / 142.0,
        2.25,
        2.4,
        2.5,
        3.0,
        3.5,
        4.0,
    ];
    let mut offsets: Vec<f32> = (0..32).map(|k| k as f32 / 32.0).collect();
    for k in 1..=16 {
        offsets.push(0.5 + k as f32 / 2048.0);
        offsets.push(0.5 - k as f32 / 2048.0);
    }

    let mut failures = Vec::new();
    let mut drawn = 0usize;
    for &scale in &scales {
        let cell = (9.0 * scale).ceil() + 3.0;
        let per_row = ((W as f32 - 8.0) / cell) as usize;
        let mut transforms = Vec::new();
        let mut placed = Vec::new();
        for (n, &offset) in offsets.iter().enumerate() {
            let x = 4.0 + (n % per_row) as f32 * cell;
            let y = 4.0 + (n / per_row) as f32 * cell + offset;
            transforms.push(Transform::translate(x, y).then_scale(scale));
            placed.push(Placed {
                x,
                y,
                px_per_unit: scale,
                label: (scale, offset),
            });
        }
        let frame = render(&mut gpu, &rm, (W, H), 1, "o", &transforms);
        drawn += placed.len();
        failures.extend(bleeding_pixels(&frame, W, &placed, empty));
    }
    // One line per lit pixel; the glyph is the "scale .. offset .." prefix.
    let mut glyphs: Vec<&str> = failures
        .iter()
        .map(|line| line.split(": pixel").next().unwrap_or(line))
        .collect();
    glyphs.dedup();
    assert!(
        failures.is_empty(),
        "{} of {drawn} 'o's show ink from another glyph ({} lit pixels):\n{}",
        glyphs.len(),
        failures.len(),
        failures.join("\n")
    );
}

/// The user's exact geometry: a 1920x1080 window at guiScale 3, chat scale 0.6690141, the chat
/// transform `GuiNewChat.drawChat` uses (`translate(2, height - 28)`, `scale(chatScale)`, line
/// `row` at `y = -9 * row - 8`), twenty lines of `o`. Line 7 is the one whose top edge lands on
/// 853.5 px.
#[test]
fn the_users_chat_scale_draws_every_line_of_os_cleanly() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found; set MC_JARS_DIR to the jars/ directory");
        return;
    };
    let empty = empty_top_rows(&rm, b'o');
    const W: u32 = 1920;
    const H: u32 = 1080;
    const GUI: i32 = 3;
    // The 320x240 floor caps the factor on a narrow target; 1920x1080 really gets 3.
    let resolution = ScaledResolution::new(W, H, GUI, false);
    assert_eq!(resolution.scale_factor(), GUI);
    let Some(mut gpu) = gpu(W, H) else { return };
    let chat_scale = 0.669_014_1_f32;
    let gui_height = resolution.scaled_height() as f32;
    let mut transforms = Vec::new();
    let mut placed = Vec::new();
    for row in 0..20 {
        let line_y = (-9 * row - 8) as f32;
        let chat = Transform::translate(2.0, gui_height - 28.0).then_scale(chat_scale);
        transforms.push(chat.then_translate(0.0, line_y));
        // Physical position of the quad's top-left corner.
        let [gx, gy] = chat.apply([0.0, line_y]);
        placed.push(Placed {
            x: gx * GUI as f32,
            y: gy * GUI as f32,
            px_per_unit: chat_scale * GUI as f32,
            label: (chat_scale, row as f32),
        });
    }
    let frame = render(&mut gpu, &rm, (W, H), GUI, "oooooooooo", &transforms);
    // Ten glyphs per line: check each glyph cell along the line (6 font pixels apart).
    let mut glyphs = Vec::new();
    for p in &placed {
        for k in 0..10 {
            glyphs.push(Placed {
                x: p.x + 6.0 * k as f32 * p.px_per_unit,
                ..*p
            });
        }
    }
    let failures = bleeding_pixels(&frame, W, &glyphs, empty);
    assert!(
        failures.is_empty(),
        "{} lit pixels above the 'o's (label = chat scale, chat row):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
