//! Headless GPU tests for the `UI 800` phase.
//!
//! These render into an offscreen target through [`mc_render::Gpu::headless`] and read the
//! pixels back, so "the glyph landed on the right texels" is a checkable fact rather than
//! something verified by looking at a window. They skip (loudly) when the machine has no wgpu
//! adapter, and when the vanilla jar is absent.
//!
//! The scaled resolution is pinned to a scale factor of **1** throughout, so one GUI pixel is
//! one framebuffer pixel and every assertion below is a direct statement about vanilla's
//! coordinates.

use std::path::{Path, PathBuf};

use mc_assets::{ResourceManager, ZipResourcePack};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::{FontData, FontRenderer};
use mc_ui::widget::{DrawList, ScaledResolution, UiPhase, WIDGETS_TEXTURE};

const W: u32 = 320;
const H: u32 = 240;

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

fn pixel(px: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2], px[i + 3]]
}

/// `guiScale = 1`, so scaled pixels and framebuffer pixels coincide.
fn resolution() -> ScaledResolution {
    let r = ScaledResolution::new(W, H, 1, false);
    assert_eq!(r.scale_factor(), 1);
    assert_eq!((r.scaled_width(), r.scaled_height()), (W as i32, H as i32));
    r
}

/// Render one `DrawList` and hand back the framebuffer.
fn render(gpu: &mut Gpu, mut phase: UiPhase, list: DrawList) -> Vec<u8> {
    phase.submit(list);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    gpu.read_pixels().expect("readback")
}

#[test]
fn a_solid_rect_lands_on_exactly_its_own_pixels() {
    let Some(mut gpu) = gpu() else { return };
    let phase = UiPhase::new(&gpu, W, H, 1, false);
    let mut list = DrawList::new();
    list.rect(10, 20, 30, 40, 0xFF00_FF00);
    let px = render(&mut gpu, phase, list);

    assert_eq!(
        pixel(&px, 10, 20),
        [0, 255, 0, 255],
        "top-left corner is inside"
    );
    assert_eq!(
        pixel(&px, 29, 39),
        [0, 255, 0, 255],
        "bottom-right pixel is inside"
    );
    assert_eq!(
        pixel(&px, 30, 39),
        [0, 0, 0, 255],
        "the right edge is exclusive"
    );
    assert_eq!(
        pixel(&px, 29, 40),
        [0, 0, 0, 255],
        "the bottom edge is exclusive"
    );
    assert_eq!(pixel(&px, 9, 20), [0, 0, 0, 255]);
    assert_eq!(pixel(&px, 10, 19), [0, 0, 0, 255]);
}

#[test]
fn a_half_transparent_rect_blends_over_what_is_under_it() {
    let Some(mut gpu) = gpu() else { return };
    let phase = UiPhase::new(&gpu, W, H, 1, false);
    let mut list = DrawList::new();
    list.rect(0, 0, 50, 50, 0xFFFF_FFFF);
    list.rect(0, 0, 50, 50, 0x8000_0000); // 50% black over white
    let px = render(&mut gpu, phase, list);
    let [r, g, b, _] = pixel(&px, 10, 10);
    assert!(
        (100..=140).contains(&r) && r == g && g == b,
        "expected mid grey, got {r},{g},{b}"
    );
}

/// The gradient really interpolates down the rectangle, which is what every vanilla screen's
/// dimmed background and every tooltip border relies on.
#[test]
fn a_gradient_rect_interpolates_down_the_screen() {
    let Some(mut gpu) = gpu() else { return };
    let phase = UiPhase::new(&gpu, W, H, 1, false);
    let mut list = DrawList::new();
    list.gradient_rect(0, 0, 100, 100, 0xFFFF_0000, 0xFF00_00FF);
    let px = render(&mut gpu, phase, list);
    let top = pixel(&px, 50, 1);
    let mid = pixel(&px, 50, 50);
    let bottom = pixel(&px, 50, 98);
    assert!(top[0] > 240 && top[2] < 15, "top is red: {top:?}");
    assert!(
        bottom[2] > 240 && bottom[0] < 15,
        "bottom is blue: {bottom:?}"
    );
    assert!(
        (100..=155).contains(&mid[0]) && (100..=155).contains(&mid[2]),
        "middle is purple: {mid:?}"
    );
}

/// The load-bearing one: a glyph drawn through the whole pipeline has to land on exactly the
/// texels `ascii.png` holds for it, at exactly the position vanilla's quad maths puts them.
#[test]
fn a_glyph_reaches_the_framebuffer_texel_for_texel() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: jars/versions/1.8.9/1.8.9.jar not found");
        return;
    };
    let data = FontData::load(&rm).expect("load font");
    let source: Vec<u8> = data.ascii_rgba().to_vec();
    let mut font = FontRenderer::new(data, false, false);

    let mut phase = UiPhase::new(&gpu, W, H, 1, false);
    phase.renderer.upload_ascii_font(&gpu, font.data());

    let mut list = DrawList::new();
    // 'A' is cell 65 of the sheet: column 1, row 4, i.e. texel origin (8, 32).
    let end = font.draw_string(&mut list, "A", 100.0, 100.0, -1, false);
    assert_eq!(end, 106, "charWidth['A'] is 6");

    let px = render(&mut gpu, phase, list);

    // The quad is `charWidth - 0.01 - 1` = 4.99 wide and 7.99 tall, so it covers five columns
    // and eight rows of framebuffer pixels, sampling texels (8..13, 32..40) one for one.
    let mut inked = 0;
    for j in 0..8u32 {
        for k in 0..5u32 {
            let src = {
                let i = (((32 + j) * 128 + (8 + k)) * 4) as usize;
                source[i + 3]
            };
            let got = pixel(&px, 100 + k, 100 + j);
            if src == 255 {
                inked += 1;
                assert_eq!(
                    got,
                    [255, 255, 255, 255],
                    "texel ({}, {}) should be lit at screen ({}, {})",
                    8 + k,
                    32 + j,
                    100 + k,
                    100 + j
                );
            } else {
                assert_eq!(
                    got,
                    [0, 0, 0, 255],
                    "texel ({}, {}) should be clear at screen ({}, {})",
                    8 + k,
                    32 + j,
                    100 + k,
                    100 + j
                );
            }
        }
    }
    assert_eq!(
        inked, 18,
        "the 'A' glyph lights 18 of the 40 texels it covers"
    );

    // Nothing spills past the glyph's own five columns.
    for j in 0..8u32 {
        assert_eq!(pixel(&px, 99, 100 + j), [0, 0, 0, 255], "one column left");
        assert_eq!(pixel(&px, 105, 100 + j), [0, 0, 0, 255], "one column right");
    }
}

/// The unicode-page path, end to end: with the flag on, a CJK ideograph is fetched from
/// `unicode_page_4e.png` and drawn at **half scale** — sixteen texels squeezed into eight
/// framebuffer pixels — so every framebuffer pixel picks the odd-numbered texel under it.
#[test]
fn a_unicode_page_glyph_renders_at_half_scale_from_its_own_texture() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: jars/versions/1.8.9/1.8.9.jar not found");
        return;
    };
    let mut font = FontRenderer::load(&rm, true, true).expect("load font");
    assert!(font.unicode_flag());

    // U+4E00 lives in cell (0, 0) of page 0x4E, and its glyph_sizes byte is 0x0F: the full
    // 16-column cell, which measures 9 and draws 8 pixels wide.
    let page = mc_ui::font::unicode_page_location(0x4e);
    let source = image::load_from_memory_with_format(
        &rm.load(&page).expect("load page"),
        image::ImageFormat::Png,
    )
    .expect("decode page")
    .to_rgba8();
    assert_eq!((source.width(), source.height()), (256, 256));

    let mut phase = UiPhase::new(&gpu, W, H, 1, true);
    let mut list = DrawList::new();
    let end = font.draw_string(&mut list, "\u{4e00}", 40.0, 40.0, -1, false);
    assert_eq!(
        font.char_width('\u{4e00}'),
        9,
        "measured from glyph_sizes.bin"
    );
    // The draw advance is (end + 1 - start) / 2 + 1 = 9.0 exactly, so measurement and drawing
    // agree here. They only diverge when that division is not exact — see the `0x0E` case in
    // the unit tests.
    assert_eq!(end, 49);

    let missing = phase.renderer.ensure_list_textures(&gpu, &rm, &list);
    assert!(missing.is_empty(), "unicode page did not load: {missing:?}");
    assert!(phase.renderer.has_texture(&mc_ui::widget::UiTexture::Font(
        mc_ui::font::FontTexture::UnicodePage(0x4e)
    )));

    let px = render(&mut gpu, phase, list);

    let mut lit = 0;
    for j in 0..8u32 {
        for k in 0..8u32 {
            // Nearest sampling over a 2:1 minification lands on texel (2k+1, 2j+1).
            let src = source.get_pixel(2 * k + 1, 2 * j + 1)[3];
            let got = pixel(&px, 40 + k, 40 + j);
            if src == 255 {
                lit += 1;
                assert_eq!(
                    got,
                    [255, 255, 255, 255],
                    "texel ({}, {})",
                    2 * k + 1,
                    2 * j + 1
                );
            } else if src == 0 {
                assert_eq!(got, [0, 0, 0, 255], "texel ({}, {})", 2 * k + 1, 2 * j + 1);
            }
        }
    }
    assert!(
        lit > 4,
        "only {lit} lit pixels; the ideograph should be dense"
    );
}

/// Hazard **H9**, checked on the GPU: the ink of a real string stops inside the width the baked
/// tables predict. A port that measured widths from the texture instead would drift here.
#[test]
fn rendered_text_stays_inside_the_width_the_tables_predict() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: jars/versions/1.8.9/1.8.9.jar not found");
        return;
    };
    let mut font = FontRenderer::load(&rm, false, false).expect("load font");
    let mut phase = UiPhase::new(&gpu, W, H, 1, false);
    phase.renderer.upload_ascii_font(&gpu, font.data());

    let text = "Multiplayer";
    let width = font.string_width(text);
    assert_eq!(width, 54);

    let mut list = DrawList::new();
    font.draw_string(&mut list, text, 20.0, 100.0, -1, false);
    let px = render(&mut gpu, phase, list);

    let lit = |x: u32| (100..108).any(|y| pixel(&px, x, y) != [0, 0, 0, 255]);
    let first = (0..W).find(|x| lit(*x)).expect("some ink");
    let last = (0..W).rev().find(|x| lit(*x)).expect("some ink");

    assert_eq!(first, 20, "the first glyph starts at the pen position");
    // The last glyph's ink stops one or two pixels short of the advance, because every width
    // includes vanilla's one-pixel inter-glyph gap.
    assert!(
        last < 20 + width as u32 && last >= 20 + width as u32 - 3,
        "ink ends at {last}, advance predicts {}",
        20 + width
    );
    // And nothing at all beyond the advance.
    for x in (20 + width as u32)..W {
        assert!(!lit(x), "ink at x={x}, past the advance");
    }
}

#[test]
fn a_drop_shadow_appears_one_pixel_down_and_right_at_a_quarter_brightness() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: jars/versions/1.8.9/1.8.9.jar not found");
        return;
    };
    let mut font = FontRenderer::load(&rm, false, false).expect("load font");
    let mut phase = UiPhase::new(&gpu, W, H, 1, false);
    phase.renderer.upload_ascii_font(&gpu, font.data());

    let mut list = DrawList::new();
    // The 'A' glyph's top row lights texel column 1..=3 of its cell.
    font.draw_string_with_shadow(&mut list, "A", 50.0, 50.0, 0x00FF_FFFF);
    let px = render(&mut gpu, phase, list);

    // The glyph body covers x 50..=54, y 50..=57; its shadow the same block offset by (1, 1).
    // Body pixel: 'A' row 0 lights columns 1..=3, so (51, 50) is pure white.
    assert_eq!(pixel(&px, 51, 50), [255, 255, 255, 255]);
    // (55, 57) is *only* reachable by the shadow — one column right of the body — and maps to
    // glyph row 6 column 4, which is lit. 255 >> 2 = 63.
    assert_eq!(
        pixel(&px, 55, 57),
        [63, 63, 63, 255],
        "shadow-only pixel should be a quarter-brightness copy"
    );
    // And the same texel in the body is white, one pixel up-left.
    assert_eq!(pixel(&px, 54, 56), [255, 255, 255, 255]);
    // The shadow never reaches further than one pixel past the body.
    for y in 50..60u32 {
        assert_eq!(pixel(&px, 56, y), [0, 0, 0, 255], "spill at x=56, y={y}");
    }
}

#[test]
fn button_artwork_blits_from_the_widgets_texture() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: jars/versions/1.8.9/1.8.9.jar not found");
        return;
    };
    let mut font = FontRenderer::load(&rm, false, false).expect("load font");
    let mut phase = UiPhase::new(&gpu, W, H, 1, false);
    phase.renderer.upload_ascii_font(&gpu, font.data());

    let mut list = DrawList::new();
    let mut button = mc_ui::widget::Button::sized(0, 20, 20, 100, 20, "Done");
    button.draw(&mut list, &mut font, 25, 25);
    assert!(button.is_mouse_over());

    let missing = phase.renderer.ensure_list_textures(&gpu, &rm, &list);
    assert!(missing.is_empty(), "missing textures: {missing:?}");
    assert!(phase
        .renderer
        .has_texture(&mc_ui::widget::UiTexture::gui(WIDGETS_TEXTURE)));

    let px = render(&mut gpu, phase, list);
    // Hovered, so the button samples the third 20-pixel row of widgets.png (v = 86). Screen
    // (20 + k, 20 + j) is texel (k, 86 + j) for the left half.
    assert_eq!(
        pixel(&px, 21, 21),
        [190, 199, 255, 255],
        "the hover highlight texel (1, 87)"
    );
    assert_eq!(pixel(&px, 22, 22), [126, 136, 191, 255], "texel (2, 88)");
    // Vanilla's button has a one-pixel black border, so the outermost row really is black.
    assert_eq!(pixel(&px, 20, 20), [0, 0, 0, 255], "the border, not a hole");
    // The right half starts at x = 70 and samples from u = 200 - 50 = 150.
    assert_eq!(pixel(&px, 118, 21), [126, 136, 191, 255], "texel (198, 87)");
    assert_eq!(
        pixel(&px, 120, 30),
        [0, 0, 0, 255],
        "and it stops at its edge"
    );
    // The label is drawn on top in 0xFFFFA0, brighter than any button texel.
    let bright = (20..120).any(|x| pixel(&px, x, 27)[0] > 220);
    assert!(bright, "no label drawn on the button");
}

#[test]
fn a_missing_texture_degrades_to_flat_colour_instead_of_a_hole() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = assets() else {
        eprintln!("SKIP: jars/versions/1.8.9/1.8.9.jar not found");
        return;
    };
    let mut phase = UiPhase::new(&gpu, W, H, 1, false);
    let mut list = DrawList::new();
    let bogus = mc_ui::widget::UiTexture::gui("textures/gui/does_not_exist.png");
    list.blit(&bogus, 10, 10, 0, 0, 20, 20);
    let missing = phase.renderer.ensure_list_textures(&gpu, &rm, &list);
    assert_eq!(missing.len(), 1, "the miss is reported");

    let px = render(&mut gpu, phase, list);
    assert_eq!(
        pixel(&px, 15, 15),
        [255, 255, 255, 255],
        "falls back to the white 1x1"
    );
}

#[test]
fn nothing_is_drawn_for_an_empty_list() {
    let Some(mut gpu) = gpu() else { return };
    let phase = UiPhase::new(&gpu, W, H, 1, false);
    let px = render(&mut gpu, phase, DrawList::new());
    assert!(px.chunks(4).all(|p| p == [0, 0, 0, 255]));
}

/// At `guiScale = 2` one GUI pixel really does become a 2×2 block of framebuffer pixels, and
/// the top-left of the scaled surface still coincides with the top-left of the window.
#[test]
fn gui_scaling_turns_one_gui_pixel_into_a_block() {
    let mut gpu = match Gpu::headless(
        640,
        480,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(g) => g,
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            return;
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    };

    let res = ScaledResolution::new(640, 480, 2, false);
    assert_eq!(res.scale_factor(), 2);
    assert_eq!((res.scaled_width(), res.scaled_height()), (320, 240));

    let mut phase = UiPhase::new(&gpu, 640, 480, 2, false);
    assert_eq!(phase.resolution().scale_factor(), 2);
    let mut list = DrawList::new();
    list.rect(0, 0, 1, 1, 0xFFFF_0000); // one GUI pixel at the origin
    list.rect(319, 239, 320, 240, 0xFF00_FF00); // and one at the far corner
    phase.submit(list);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
    let px = gpu.read_pixels().expect("readback");

    let at = |x: u32, y: u32| -> [u8; 4] {
        let i = ((y * 640 + x) * 4) as usize;
        [px[i], px[i + 1], px[i + 2], px[i + 3]]
    };
    for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        assert_eq!(
            at(x, y),
            [255, 0, 0, 255],
            "({x}, {y}) is part of the 2x2 block"
        );
    }
    assert_eq!(at(2, 0), [0, 0, 0, 255], "and the block is only two wide");
    assert_eq!(at(0, 2), [0, 0, 0, 255]);
    for (x, y) in [(638, 478), (639, 479)] {
        assert_eq!(
            at(x, y),
            [0, 255, 0, 255],
            "the far corner reaches the far corner"
        );
    }
    let _ = resolution();
}
