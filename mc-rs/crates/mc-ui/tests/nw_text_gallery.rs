//! Bug report: "some minecraft text gets mildly smushed/clipped at
//! top/bottom".
//!
//! Three instruments, all through the real font renderer and the real `UI 800` phase on a
//! headless wgpu device, read back:
//!
//! * **Vanilla parity at integer scales** ([`integer_scales_match_vanillas_nearest_picture`]):
//!   lines of every printable ASCII glyph at the GUI positions screens use (top-left text, the
//!   chat transform, right-aligned text), at window sizes 1280x720, 1366x768, 1600x900,
//!   1920x1080, 2560x1440 and two windowed (maximised) client areas, GUI scales 1-4. Each frame
//!   is compared pixel for pixel with a CPU model of vanilla 1.8.9's picture: vanilla's glyph
//!   quads (`FontRenderer.renderDefaultChar`) under vanilla's projection
//!   (`EntityRenderer.setupOverlayRendering`: `ortho(0, getScaledWidth_double(), ...)`, exactly
//!   `scaleFactor` pixels per GUI pixel) and nearest sampling.
//! * **Even rows at fractional scales** ([`fractional_scales_draw_every_row_evenly`]): a
//!   synthetic font whose glyphs are stripes, drawn through the HUD element scales (0.5-1.5) at
//!   GUI scales 1-4. Every interior stripe must integrate to the same thickness, the scale.
//! * **The gallery** ([`gallery`]): real screens (HUD with chat, sidebar, mod HUD elements at the
//!   user's scales, a tooltip; the tab list; the open chat; the title screen; the fractional
//!   scale cases) at every window size and GUI scale, saved as PNGs when
//!   `MC_NW_TEXT_PNG_DIR=<dir>` is set (`<dir>/<scene>-<w>x<h>-gui<n>.png`).
//!
//! Skips (loudly) when there is no wgpu adapter or the vanilla jar is absent (`MC_JARS_DIR`).

use std::path::{Path, PathBuf};

use mc_assets::{Language, ResourceLocation, ResourceManager, ZipResourcePack};
use mc_mods::{HudAnchor, HudPlacement, ModSettings};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::component::ChatComponent;
use mc_ui::font::{FontData, FontRenderer, GlyphQuad, TextMesh};
use mc_ui::modhud::{HudCtx, HudData, HudKeys, ModHud};
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::chat::{ChatLayout, ChatLog, ChatScreen};
use mc_ui::screens::hud::{Hud, Sidebar};
use mc_ui::screens::menu::MainMenuScreen;
use mc_ui::screens::tab_list::{TabListOverlay, TabRow};
use mc_ui::widget::{DrawList, ScaledResolution, Transform, UiPhase};

/// Window sizes of the matrix: the four the plan names, a common laptop size, and two windowed
/// client areas (a maximised window on a 1080p and on a 1440p monitor: title bar and taskbar
/// taken off), whose heights are not multiples of every GUI scale.
const SIZES: [(u32, u32); 7] = [
    (1280, 720),
    (1366, 768),
    (1600, 900),
    (1920, 1080),
    (2560, 1440),
    (1920, 1017),
    (2560, 1377),
];

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

fn png_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MC_NW_TEXT_PNG_DIR")?);
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

fn save_png(dir: &Path, name: &str, frame: &[u8], width: u32, height: u32) {
    let path = dir.join(name);
    image::save_buffer(&path, frame, width, height, image::ColorType::Rgba8)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// Render `list` at `resolution` (custom ascii sheet uploaded first, when given) on black.
fn render(
    gpu: &mut Gpu,
    rm: &ResourceManager,
    resolution: ScaledResolution,
    gui_option: i32,
    sheet: Option<&FontData>,
    list: DrawList,
) -> Vec<u8> {
    render_counted(gpu, rm, resolution, gui_option, sheet, list).0
}

/// [`render`], plus how many `set_pipeline` calls and batches the list cost.
fn render_counted(
    gpu: &mut Gpu,
    rm: &ResourceManager,
    resolution: ScaledResolution,
    gui_option: i32,
    sheet: Option<&FontData>,
    list: DrawList,
) -> (Vec<u8>, u64, usize) {
    let (w, h) = resolution.display_size();
    let (w, h) = (w as u32, h as u32);
    gpu.resize(w, h);
    let mut phase = UiPhase::new(gpu, w, h, gui_option, false);
    if let Some(data) = sheet {
        phase.renderer.upload_ascii_font(gpu, data);
    }
    let missing = phase.renderer.ensure_list_textures(gpu, rm, &list);
    assert!(missing.is_empty(), "textures did not load: {missing:?}");
    let binds = phase.renderer.pipeline_binds(&list);
    let batches = list.batches().len();
    phase.set_resolution(resolution);
    phase.submit(list);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let target = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, target.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    assert_eq!(pixels.len(), (w * h * 4) as usize, "readback size");
    (pixels, binds, batches)
}

/// Every printable character of `ascii.png`'s ASCII rows, in lines that fit 300 GUI pixels.
fn ascii_lines(font: &FontRenderer) -> Vec<String> {
    let all: String = (33u8..127).map(char::from).collect();
    font.wrap_to_lines(&all, 300)
}

// ---------------------------------------------------------------------------------------------
// 1. Vanilla parity at integer scales.

/// One string drawn for the parity probe: where, and its glyph quads in GUI space.
struct Placed {
    quads: Vec<GlyphQuad>,
}

/// Vanilla's GL picture of `quads` (GUI space): a pixel is ink when its centre, mapped through
/// vanilla's exact projection (`factor` pixels per GUI pixel), falls inside a quad (top-left
/// rule) and the texel nearest-sampled there is inked. The texel mapping is vanilla's own: the
/// quad spans `f - 1` by `7.99` texels from its cell origin (`renderDefaultChar`), so this model
/// ignores mc-rs's UV inset on purpose — at integer scales it must not change a single pixel.
fn vanilla_ink(quads: &[GlyphQuad], sheet: &FontData, w: u32, h: u32, factor: f32) -> Vec<bool> {
    let (sheet_w, sheet_h) = sheet.ascii_size();
    let rgba = sheet.ascii_rgba();
    // `renderDefaultChar` works in 128-unit texel space whatever the sheet's real size.
    let to_px = sheet_w as f32 / 128.0;
    let mut ink = vec![false; (w * h) as usize];
    for q in quads {
        let (x0, y0) = (q.pos[0][0], q.pos[0][1]);
        let (x1, y1) = (q.pos[3][0], q.pos[3][1]);
        let cx = (q.uv[0][0] * 128.0 / 8.0).floor() * 8.0;
        let cy = (q.uv[0][1] * 128.0 / 8.0).floor() * 8.0;
        let span_x = q.uv[3][0] * 128.0 - cx;
        let span_y = q.uv[3][1] * 128.0 - cy;
        let px0 = ((x0 * factor).floor() - 1.0).max(0.0) as u32;
        let py0 = ((y0 * factor).floor() - 1.0).max(0.0) as u32;
        let px1 = ((x1 * factor).ceil() as u32 + 1).min(w);
        let py1 = ((y1 * factor).ceil() as u32 + 1).min(h);
        for py in py0..py1 {
            let gy = (py as f32 + 0.5) / factor;
            if gy < y0 || gy >= y1 {
                continue;
            }
            let ty = cy + ((gy - y0) / (y1 - y0) * span_y).floor();
            for px in px0..px1 {
                let gx = (px as f32 + 0.5) / factor;
                if gx < x0 || gx >= x1 {
                    continue;
                }
                let tx = cx + ((gx - x0) / (x1 - x0) * span_x).floor();
                let sx = ((tx * to_px) as u32).min(sheet_w - 1);
                let sy = ((ty * to_px) as u32).min(sheet_h - 1);
                if rgba[((sy * sheet_w + sx) * 4 + 3) as usize] != 0 {
                    ink[(py * w + px) as usize] = true;
                }
            }
        }
    }
    ink
}

/// The parity probe's text: ASCII lines at the left, the chat's transform, right-aligned, and
/// centred (integer division, as `drawCenteredString`).
fn parity_scene(font: &mut FontRenderer, ctx: &ScreenCtx) -> (DrawList, Vec<Placed>) {
    let mut list = DrawList::new();
    let mut placed = Vec::new();
    let lines = ascii_lines(font);
    let mut draw = |list: &mut DrawList, font: &mut FontRenderer, t: Transform, text: &str| {
        let mut mesh = TextMesh::new();
        font.draw_string(&mut mesh, text, 0.0, 0.0, 0x00FF_FFFF, false);
        let quads = mesh
            .glyphs
            .iter()
            .map(|q| GlyphQuad {
                pos: q.pos.map(|p| t.apply(p)),
                ..*q
            })
            .collect();
        placed.push(Placed { quads });
        list.with_transform(t, |d| {
            font.draw_string(d, text, 0.0, 0.0, 0x00FF_FFFF, false);
        });
    };
    for (i, line) in lines.iter().enumerate() {
        let t = Transform::translate(2.0, 2.0 + 10.0 * i as f32);
        draw(&mut list, font, t, line);
    }
    // GuiNewChat: translate(2, height - 48) . translate(0, 20) . scale(1), rows at -9 * row - 8.
    for (row, line) in lines.iter().enumerate() {
        let t = Transform::translate(2.0, (ctx.height - 28) as f32)
            .then_translate(0.0, (-9 * row as i32 - 8) as f32);
        draw(&mut list, font, t, line);
    }
    // Right-aligned (the sidebar's scores, F3's right column), where a projection error is
    // largest, and centred.
    for (i, line) in lines.iter().enumerate() {
        let x = ctx.width - 2 - font.string_width(line);
        let t = Transform::translate(x as f32, 2.0 + 10.0 * i as f32);
        draw(&mut list, font, t, line);
        let x = ctx.width / 2 - font.string_width(line) / 2;
        let t = Transform::translate(x as f32, (ctx.height / 2 + 10 * i as i32) as f32);
        draw(&mut list, font, t, line);
    }
    (list, placed)
}

#[test]
fn integer_scales_match_vanillas_nearest_picture() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found; set MC_JARS_DIR to the jars/ directory");
        return;
    };
    let Some(mut gpu) = gpu(1280, 720) else {
        return;
    };
    let sheet = FontData::load(&rm).expect("ascii.png");
    let mut font = FontRenderer::load(&rm, false, false).expect("font");
    let mut report = Vec::new();
    let mut failures = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (w, h) in SIZES {
        for gui in 1..=4 {
            let res = ScaledResolution::new(w, h, gui, false);
            if !seen.insert((w, h, res.scale_factor())) {
                continue; // a capped option draws the same frame as the smaller one
            }
            let ctx = ScreenCtx::new(res, 0.0);
            let (list, placed) = parity_scene(&mut font, &ctx);
            let frame = render(&mut gpu, &rm, res, gui, None, list);
            let quads: Vec<GlyphQuad> = placed.into_iter().flat_map(|p| p.quads).collect();
            let factor = res.scale_factor() as f32;
            let ink = vanilla_ink(&quads, &sheet, w, h, factor);
            let mut wrong = 0usize;
            let mut first = None;
            for (i, &want) in ink.iter().enumerate() {
                let got = frame[i * 4] > 127;
                if got != want {
                    wrong += 1;
                    first.get_or_insert(((i as u32) % w, (i as u32) / w, want));
                }
            }
            let line = format!(
                "{w}x{h} gui{gui} (factor {}, scaled {}x{} = {:.3}x{:.3}): {wrong} pixels differ{}",
                res.scale_factor(),
                res.scaled_width(),
                res.scaled_height(),
                res.scaled_width_f64(),
                res.scaled_height_f64(),
                first.map_or(String::new(), |(x, y, want)| format!(
                    ", first at ({x},{y}) vanilla ink={want}"
                ))
            );
            if wrong != 0 {
                failures.push(line.clone());
            }
            report.push(line);
        }
    }
    eprintln!("{}", report.join("\n"));
    assert!(
        failures.is_empty(),
        "integer-scale text differs from vanilla's picture:\n{}",
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------------------------
// 2. Even rows at fractional scales.

const STRIPES_H: u8 = b'H';
const STRIPES_V: u8 = b'V';

/// A sheet of `16 * cell` pixels square (8: vanilla's 128; 16: an HD pack's 256, like a 16x
/// resource pack): `H` is horizontal stripes on every even texel row, `V` vertical stripes
/// on every even texel column. Every other cell is empty. The inked columns are chosen so both
/// measure 8 wide (`readFontTexture`), like vanilla's widest letters.
fn stripe_sheet(cell: u32) -> FontData {
    let mut img = image::RgbaImage::new(16 * cell, 16 * cell);
    let origin = |ch: u8| (u32::from(ch % 16) * cell, u32::from(ch / 16) * cell);
    let (h_last, v_last) = if cell == 8 {
        (6, 6)
    } else {
        (cell - 3, cell - 4)
    };
    let (hx, hy) = origin(STRIPES_H);
    for row in (0..cell).step_by(2) {
        for col in 0..=h_last {
            img.put_pixel(hx + col, hy + row, image::Rgba([255, 255, 255, 255]));
        }
    }
    let (vx, vy) = origin(STRIPES_V);
    for col in (0..=v_last).step_by(2) {
        for row in 0..cell {
            img.put_pixel(vx + col, vy + row, image::Rgba([255, 255, 255, 255]));
        }
    }
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut png, image::ImageFormat::Png)
        .expect("encode");
    FontData::from_parts(&png.into_inner(), &vec![0u8; 0x1_0000]).expect("stripe font")
}

/// Integrated ink (in pixels, alpha/255 summed) of each interior stripe of every drawn glyph,
/// measured across the stripes through the glyph's middle (`middle` texels in). Ideal: `s` for
/// every stripe.
fn stripe_sums(
    frame: &[u8],
    w: u32,
    h: u32,
    glyphs: &[(f32, f32, f32, bool)],
    stripes: &[f32],
    middle: f32,
) -> Vec<f32> {
    let lum = |x: i64, y: i64| -> f32 {
        if x < 0 || y < 0 || x >= i64::from(w) || y >= i64::from(h) {
            return 0.0;
        }
        f32::from(frame[((y as u32 * w + x as u32) * 4) as usize]) / 255.0
    };
    let mut out = Vec::new();
    // (left, top, s, horizontal): screen-space origin of the glyph and pixels per texel.
    for &(left, top, s, horizontal) in glyphs {
        for &stripe in stripes {
            let (lo, hi) = (stripe - 0.5, stripe + 1.5);
            let mut sum = 0.0;
            if horizontal {
                let x = (left + middle * s).floor() as i64;
                let (a, b) = (top + lo * s, top + hi * s);
                for y in (a.floor() as i64)..=(b.ceil() as i64) {
                    let c = y as f32 + 0.5;
                    if c >= a && c < b {
                        sum += lum(x, y);
                    }
                }
            } else {
                let y = (top + middle * s).floor() as i64;
                let (a, b) = (left + lo * s, left + hi * s);
                for x in (a.floor() as i64)..=(b.ceil() as i64) {
                    let c = x as f32 + 0.5;
                    if c >= a && c < b {
                        sum += lum(x, y);
                    }
                }
            }
            out.push(sum);
        }
    }
    out
}

#[test]
fn fractional_scales_draw_every_row_evenly() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found; set MC_JARS_DIR to the jars/ directory");
        return;
    };
    let (w, h) = (1920u32, 1080u32);
    let Some(mut gpu) = gpu(w, h) else {
        return;
    };
    let mut report = Vec::new();
    let mut failures = Vec::new();
    // The HUD element scale slider's range (0.5-2.0) with the user's 0.6/0.7/0.8, the
    // keystrokes CPS half scale, and integer controls.
    let scales = [0.5f32, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.2, 1.25, 1.5, 2.0];
    // Sub-pixel placements: a scaled element's inner text and chat rows under chat scale land
    // anywhere. Screens place text on whole (or half) GUI pixels: offset 0.
    let offsets = [0.0f32, 0.13, 0.37, 0.5, 0.71, 0.94];
    // 8: vanilla's 128x128 sheet. 16: a 256x256 HD sheet, like the reference pack's, where even GUI
    // scale 3 at 100% is 1.5 pixels per texel.
    for cell in [8u32, 16] {
        let mut font = FontRenderer::new(stripe_sheet(cell), false, false);
        let texels_per_unit = cell as f32 / 8.0;
        // Interior stripes only: not the first (on the quad's top/left edge) and not one on the
        // quad's right edge (`V`'s column 6 in an 8-texel cell). A glyph's outermost texels sit
        // on its quad's edges, where the rasteriser, not the sampler, decides coverage: those
        // stay within half a pixel (nearest: a whole pixel). See the TEXT report.
        let last = if cell == 8 { 4 } else { cell - 4 };
        let stripes: Vec<f32> = (1..=last / 2).map(|i| 2.0 * i as f32).collect();
        let middle = cell as f32 / 2.0 - 0.5;
        for gui in 1..=4 {
            let res = ScaledResolution::new(w, h, gui, false);
            let g = res.scale_factor() as f32;
            let mut list = DrawList::new();
            // (scale index, offset index) of each glyph, then the glyph itself.
            let mut glyphs = Vec::new();
            let mut tags = Vec::new();
            let gui_width = res.scaled_width() as f32;
            let (mut x, mut y, mut row_h) = (4.0f32, 4.0f32, 0.0f32);
            for (si, &k) in scales.iter().enumerate() {
                for (oi, &off) in offsets.iter().enumerate() {
                    for (text, horizontal) in [("HHH", true), ("VVV", false)] {
                        let advance = k * 8.0 * 3.0 + 4.0;
                        if x + advance > gui_width - 4.0 {
                            (x, y, row_h) = (4.0, y + row_h + 4.0, 0.0);
                        }
                        let (ox, oy) = (x + off, y + off);
                        list.with_transform(Transform::translate(ox, oy).then_scale(k), |d| {
                            font.draw_string(d, text, 0.0, 0.0, 0x00FF_FFFF, false);
                        });
                        for n in 0..3 {
                            let left = (ox + k * 8.0 * n as f32) * g;
                            glyphs.push((left, oy * g, k * g / texels_per_unit, horizontal));
                            tags.push((si, oi));
                        }
                        x += advance;
                        row_h = row_h.max(8.0 * k + 1.0);
                    }
                }
            }
            assert!(
                y + row_h < res.scaled_height() as f32,
                "gui{gui}: the probe must fit"
            );
            let sheet = stripe_sheet(cell);
            let frame = render(&mut gpu, &rm, res, gui, Some(&sheet), list);
            let sums = stripe_sums(&frame, w, h, &glyphs, &stripes, middle);
            let per_glyph = stripes.len();
            for (si, &k) in scales.iter().enumerate() {
                let s = k * g / texels_per_unit;
                let integer = (s - s.round()).abs() < 1e-4;
                // Worst stripe error over the asserted placements, and over all of them.
                let (mut asserted, mut all) = (0.0f32, 0.0f32);
                let (mut lo, mut hi) = (f32::MAX, f32::MIN);
                for (gi, &(gsi, goi)) in tags.iter().enumerate() {
                    if gsi != si {
                        continue;
                    }
                    for &v in &sums[gi * per_glyph..(gi + 1) * per_glyph] {
                        let e = (v - s).abs();
                        all = all.max(e);
                        // An integer scale is vanilla's nearest picture by design, so it is held
                        // to evenness only where screens put text; elsewhere the font's UV inset
                        // can still tip one tie (reported, pre-existing).
                        if !integer || goi == 0 {
                            asserted = asserted.max(e);
                        }
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
                let line = format!(
                    "sheet {}: gui{gui} element scale {k:>4}: {s:.2} px per texel, stripes \
                     {lo:.3}..{hi:.3} px, worst error {asserted:.3} px ({all:.3} at any offset)",
                    16 * cell
                );
                // A few quantisation steps of 8-bit alpha per stripe is "even"; nearest
                // sampling's floor/ceil rows miss by up to a whole pixel. Below one pixel per
                // texel (GUI 1 with a shrunk element: 4-7 px tall text) a pixel spans more than
                // two texels and the two-tap filter is approximate: reported, not asserted.
                if asserted > 0.03 && s >= 1.0 {
                    failures.push(line.clone());
                }
                report.push(line);
            }
            if let Some(dir) = png_dir() {
                let name = format!("stripes-{}-gui{gui}.png", 16 * cell);
                save_png(&dir, &name, &frame, w, h);
            }
        }
    }
    eprintln!("{}", report.join("\n"));
    assert!(
        failures.is_empty(),
        "uneven rows at fractional scales:\n{}",
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------------------------
// 3. The gallery.

/// A dim, varied backdrop so shadows and plates read as they do over a world.
fn backdrop(d: &mut DrawList, w: i32, h: i32) {
    let bands = 12;
    for band in 0..bands {
        let b = band as u32;
        let shade = 0xFF00_0000u32 | ((30 + b * 9) << 16) | ((60 + b * 8) << 8) | (90 + b * 6);
        d.rect(0, band * h / bands, w, (band + 1) * h / bands, shade);
    }
    d.rect(0, h * 2 / 3, w, h, 0xFF2E_5A24);
}

fn chat_log(font: &FontRenderer, layout: &ChatLayout, lang: &Language, n: usize) -> ChatLog {
    let mut log = ChatLog::new();
    let samples = [
        "\u{a7}6[MVP\u{a7}c+\u{a7}6] Sky_Builder\u{a7}f: gg",
        "\u{a7}aYou have \u{a7}e5 \u{a7}aseconds to choose a kit!",
        "\u{a7}7Cages open in: \u{a7}c3 \u{a7}7seconds!",
        "\u{a7}eThe quick brown fox jumps over the lazy dog. gjpqy_",
        "\u{a7}b[VIP] \u{a7}bAlex\u{a7}f: anyone want to team? Qq Gg Jj Yy",
        "\u{a7}cYou were killed by Steve! \u{a7}7(\u{a7}c+0 coins\u{a7}7)",
        "\u{a7}fABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz 0123456789",
        "\u{a7}d\u{a7}lMEGA \u{a7}r\u{a7}fSkyWars \u{a7}8- \u{a7}7\u{a7}oitalic \u{a7}nunderline",
    ];
    for i in 0..n {
        let text = samples[i % samples.len()];
        log.add_message(
            ChatComponent::text(text),
            0,
            1000,
            layout,
            font,
            lang,
            false,
        );
    }
    log
}

fn sidebar() -> Sidebar {
    Sidebar::from_scores(
        "\u{a7}e\u{a7}lSKYWARS",
        &[
            ("\u{a7}ewww.example.net".to_owned(), 1),
            (" ".to_owned(), 2),
            ("Mode: \u{a7}aNormal".to_owned(), 3),
            ("Map: \u{a7}aOnionring".to_owned(), 4),
            ("  ".to_owned(), 5),
            ("Kills: \u{a7}a3".to_owned(), 6),
            ("Players left: \u{a7}a7".to_owned(), 7),
            ("   ".to_owned(), 8),
            ("Next Event:".to_owned(), 9),
            ("\u{a7}aRefill 1:52".to_owned(), 10),
        ],
        str::to_owned,
    )
}

/// The user's own HUD mod scales (`mc-rs-mods.json`): fps 0.7, cps 0.7, potion 0.8; plus
/// coordinates and keystrokes (with its half-scale CPS line) at 1.0.
fn mod_hud_settings() -> ModSettings {
    let mut settings = ModSettings::new();
    for (id, element, anchor, x, y, scale) in [
        ("fps", "fps", HudAnchor::TopLeft, 2.0, 2.0, 0.7),
        ("cps", "cps", HudAnchor::TopLeft, 2.0, 12.0, 0.7),
        (
            "coordinates",
            "coordinates",
            HudAnchor::TopLeft,
            2.0,
            22.0,
            1.0,
        ),
        (
            "keystrokes",
            "keystrokes",
            HudAnchor::TopLeft,
            2.0,
            70.0,
            1.0,
        ),
    ] {
        settings.set_enabled(id, true);
        let mut p = HudPlacement::new(anchor, x, y);
        p.scale = scale;
        assert!(
            settings.set_placement(id, element, p) || scale == 1.0,
            "{id} placement"
        );
    }
    settings
}

fn hud_data() -> HudData {
    HudData {
        now_ms: 10_000,
        tick: 200,
        fps: 487,
        cps_left: 12,
        cps_right: 4,
        position: [128.7, 64.0, -301.2],
        keys: HudKeys {
            forward: true,
            attack: true,
            ..HudKeys::default()
        },
        ..HudData::default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scene {
    /// Chat (closed), the sidebar, mod HUD elements at the user's scales, a tooltip.
    Hud,
    /// The tab list.
    Tab,
    /// The open chat with typed text.
    ChatOpen,
    /// The vanilla title screen's widgets over the dirt background.
    Title,
    /// Chat scale 0.67 (`chatScale` 95/142, the user's old setting) and HUD scales 0.6/1.2/1.5.
    Scaled,
    /// The vanilla Options screen.
    Options,
}

impl Scene {
    const ALL: [Scene; 6] = [
        Scene::Hud,
        Scene::Tab,
        Scene::ChatOpen,
        Scene::Title,
        Scene::Scaled,
        Scene::Options,
    ];

    fn name(self) -> &'static str {
        match self {
            Scene::Hud => "hud",
            Scene::Tab => "tab",
            Scene::ChatOpen => "chat-open",
            Scene::Title => "title",
            Scene::Scaled => "scaled",
            Scene::Options => "options",
        }
    }
}

fn build_scene(
    scene: Scene,
    font: &mut FontRenderer,
    ctx: &ScreenCtx,
    lang: &Language,
) -> DrawList {
    let mut d = DrawList::new();
    let (w, h) = (ctx.width, ctx.height);
    let gui = ctx.resolution.scale_factor() as f32;
    match scene {
        Scene::Hud | Scene::Scaled => {
            backdrop(&mut d, w, h);
            let mut layout = ChatLayout::from_settings(&mc_render::GameSettings::default());
            if scene == Scene::Scaled {
                layout.scale = 95.0 / 142.0;
            }
            let log = chat_log(font, &layout, lang, 10);
            log.draw(&mut d, font, ctx, 1010, &layout, false);
            Hud::new().draw_scoreboard(&mut d, font, ctx, &sidebar());
            let mut settings = mod_hud_settings();
            if scene == Scene::Scaled {
                // Stacked so the scaled elements do not overlap one another.
                for (id, y, scale) in [
                    ("fps", 2.0, 0.6),
                    ("cps", 12.0, 1.2),
                    ("coordinates", 34.0, 1.5),
                    ("keystrokes", 110.0, 1.0),
                ] {
                    let mut p = HudPlacement::new(HudAnchor::TopLeft, 2.0, y);
                    p.scale = scale;
                    settings.set_placement(id, id, p);
                }
            }
            let mut hud = ModHud::new();
            hud.sync(&settings);
            let data = hud_data();
            let hud_ctx = HudCtx {
                data: &data,
                settings: &settings,
                screen: (w as f32, h as f32),
                pixel_scale: gui,
                editing: false,
            };
            hud.draw(&hud_ctx, font, &mut d);
            if scene == Scene::Hud {
                let lines = [
                    "\u{a7}bDiamond Sword".to_owned(),
                    "\u{a7}7Sharpness II".to_owned(),
                    "".to_owned(),
                    "\u{a7}9+7 Attack Damage".to_owned(),
                ];
                mc_ui::screens::container::tooltip_box::draw(
                    &mut d,
                    font,
                    &lines,
                    w / 2 + 20,
                    h / 2 - 40,
                    w,
                    h,
                );
            }
        }
        Scene::Tab => {
            backdrop(&mut d, w, h);
            let mut rows: Vec<TabRow> = [
                "Sky_Builder",
                "Notch",
                "jeb_",
                "Dinnerbone",
                "Grumm",
                "Jumpy",
                "Steve",
                "Alex",
                "xX_Gamer_Xx",
                "qwertyuiopgj",
                "Player_With_Long_Name",
                "Yy_Jj_Gg",
            ]
            .iter()
            .enumerate()
            .map(|(i, name)| TabRow::new(*name, 20 + 37 * i as i32))
            .collect();
            mc_ui::screens::tab_list::sort_rows(&mut rows);
            let mut tab = TabListOverlay::new();
            tab.update(true, 0);
            tab.draw(
                &mut d,
                font,
                w,
                &rows,
                Some("\u{a7}bYou are playing on \u{a7}e\u{a7}lMC.EXAMPLE.NET"),
                Some("\u{a7}aRanks, Boosters & MORE! \u{a7}c\u{a7}lSTORE.EXAMPLE.NET"),
                None,
                false,
            );
        }
        Scene::ChatOpen => {
            backdrop(&mut d, w, h);
            let layout = ChatLayout::from_settings(&mc_render::GameSettings::default());
            let log = chat_log(font, &layout, lang, 30);
            log.draw(&mut d, font, ctx, 1010, &layout, true);
            let mut screen = ChatScreen::new("/msg Notch Hello! gjpqy_ \u{a7}", 0, font);
            screen.init(ctx);
            screen.draw(&mut d, font, ctx);
        }
        Scene::Title => {
            let mut menu = MainMenuScreen::new("Also try Terraria!");
            menu.init(ctx, lang);
            menu.draw(&mut d, font, ctx, (0, 0), 0, 0);
        }
        Scene::Options => {
            let mut screen = mc_ui::screens::options::OptionsScreen::new(false);
            screen.init(ctx, &mc_render::GameSettings::default(), lang);
            screen.draw(&mut d, font, ctx, (0, 0), lang);
        }
    }
    d
}

#[test]
fn gallery() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found; set MC_JARS_DIR to the jars/ directory");
        return;
    };
    let Some(mut gpu) = gpu(1280, 720) else {
        return;
    };
    let dir = png_dir();
    let lang = Language::load(&rm, &["en_US"]);
    let mut font = FontRenderer::load(&rm, false, false).expect("font");
    let mut frames = 0;
    for (w, h) in SIZES {
        let mut seen = std::collections::HashSet::new();
        for gui in 1..=4 {
            let res = ScaledResolution::new(w, h, gui, false);
            if !seen.insert(res.scale_factor()) {
                continue;
            }
            let ctx = ScreenCtx::new(res, 0.0);
            for scene in Scene::ALL {
                let list = build_scene(scene, &mut font, &ctx, &lang);
                let (frame, binds, batches) = render_counted(&mut gpu, &rm, res, gui, None, list);
                if (w, h) == (1920, 1080) {
                    eprintln!(
                        "counts {} {w}x{h} gui{}: {batches} batches, {binds} pipeline binds",
                        scene.name(),
                        res.scale_factor()
                    );
                }
                frames += 1;
                // Something was drawn: not every pixel is the clear colour.
                assert!(
                    frame
                        .chunks_exact(4)
                        .any(|p| p[0] > 200 && p[1] > 200 && p[2] > 200),
                    "{} {w}x{h} gui{gui}: no white text",
                    scene.name()
                );
                if let Some(dir) = &dir {
                    let name = format!("{}-{w}x{h}-gui{}.png", scene.name(), res.scale_factor());
                    save_png(dir, &name, &frame, w, h);
                }
            }
        }
    }
    eprintln!("gallery: {frames} frames");
}

/// The gallery's HUD and open-chat scenes with a resource pack's `ascii.png` instead of the
/// jar's, when `MC_NW_TEXT_FONT_PNG` names one (read-only; e.g. an HD pack's 256x256 sheet, where
/// GUI scale 3 is 1.5 pixels per texel even at 100%). Saved as `pack-<scene>-<w>x<h>-gui<n>.png`.
#[test]
fn gallery_with_a_pack_font() {
    let Some(path) = std::env::var_os("MC_NW_TEXT_FONT_PNG") else {
        eprintln!("SKIP: set MC_NW_TEXT_FONT_PNG to a resource pack's font/ascii.png");
        return;
    };
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found; set MC_JARS_DIR to the jars/ directory");
        return;
    };
    let Some(mut gpu) = gpu(1920, 1080) else {
        return;
    };
    let png = std::fs::read(&path).expect("the pack's ascii.png");
    let sizes = rm
        .load(&ResourceLocation::minecraft("font/glyph_sizes.bin"))
        .expect("glyph_sizes.bin");
    let mut font = FontRenderer::new(
        FontData::from_parts(&png, &sizes).expect("pack font"),
        false,
        false,
    );
    let sheet = FontData::from_parts(&png, &sizes).expect("pack font");
    let lang = Language::load(&rm, &["en_US"]);
    let dir = png_dir();
    for (w, h) in [(1920u32, 1080u32), (1366, 768)] {
        for gui in 2..=4 {
            let res = ScaledResolution::new(w, h, gui, false);
            let ctx = ScreenCtx::new(res, 0.0);
            for scene in [Scene::Hud, Scene::ChatOpen] {
                let list = build_scene(scene, &mut font, &ctx, &lang);
                let frame = render(&mut gpu, &rm, res, gui, Some(&sheet), list);
                if let Some(dir) = &dir {
                    let name = format!(
                        "pack-{}-{w}x{h}-gui{}.png",
                        scene.name(),
                        res.scale_factor()
                    );
                    save_png(dir, &name, &frame, w, h);
                }
            }
        }
    }
}
