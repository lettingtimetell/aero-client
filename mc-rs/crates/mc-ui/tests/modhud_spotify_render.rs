//! Headless GPU renders of the Spotify widget: both styles,
//! with a fake track and a procedural cover, over a busy synthetic "scene", drawn by the real
//! `UI 800` phase at GUI scale 3 (the user's) and read back.
//!
//! What the frame must show:
//! * the Minecraft style's cover lands where the layout says, in the cover's colours, inside a
//!   panel that darkens the scene like the chat backdrop (alpha 0x7F);
//! * the Aero style's panel has rounded corners (the scene shows in the corner), white title
//!   text, and the cover clipped to rounded corners too;
//! * neither style asks for the scene (`PhaseList::scene_mode()` stays false): one UI pass.
//!
//! Set `MC_W9_RENDER_DIR=<dir>` to save every variant as a PNG for a human look.
//!
//! Skips (loudly) when there is no wgpu adapter or the vanilla jar is absent.

use std::path::{Path, PathBuf};

use mc_assets::{ResourceManager, TextureMetadata, ZipResourcePack};
use mc_mods::ModSettings;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::FontRenderer;
use mc_ui::modhud::elements::spotify::{
    cover_texture, AeroText, SpotifyControl, SpotifyElement, SpotifyHud, MOD_ID,
};
use mc_ui::modhud::{HudCtx, HudData, HudElement};
use mc_ui::screens::modmenu::text::{atlas_key, MenuText, ATLAS_SIZE};
use mc_ui::widget::{DrawList, ScaledResolution, Transform, UiPhase};

const SCALE: u32 = 3;
const W: u32 = 1200;
const H: u32 = 760;
const COVER: u32 = 128;

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

fn resources() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(rm)
}

/// A fake album cover: a magenta-to-orange diagonal with a pale disc in the middle.
fn cover_pixels() -> Vec<u8> {
    let mut out = Vec::with_capacity((COVER * COVER * 4) as usize);
    for y in 0..COVER {
        for x in 0..COVER {
            let t = (x + y) as f32 / (2 * COVER - 2) as f32;
            let (dx, dy) = (x as f32 - 63.5, y as f32 - 63.5);
            let disc = (dx * dx + dy * dy).sqrt() < 30.0;
            let (r, g, b) = if disc {
                (250, 240, 220)
            } else {
                (
                    (200.0 + 55.0 * t) as u8,
                    (40.0 + 120.0 * t) as u8,
                    (180.0 - 150.0 * t) as u8,
                )
            };
            out.extend_from_slice(&[r, g, b, 255]);
        }
    }
    out
}

/// The scene behind the HUD: sky to grass, with saturated blocks the panels sit across.
fn scene(d: &mut DrawList, gui: (i32, i32)) {
    d.gradient_rect(0, 0, gui.0, gui.1, 0xFF6F_A8FF, 0xFF4E_9A3A);
    for (i, color) in [0xFFFF_D23Fu32, 0xFFE8_3F6F, 0xFF3F_E8D9, 0xFFFF_FFFF]
        .into_iter()
        .enumerate()
    {
        let x = 20 + i as i32 * 95;
        d.rect(x, 30, x + 40, gui.1 - 30, color);
    }
}

struct Variant {
    name: &'static str,
    aero: bool,
    data: SpotifyHud,
    scale: f32,
    at: (f32, f32),
}

fn track() -> SpotifyHud {
    SpotifyHud {
        present: true,
        text_revision: 1,
        title: "Shining Home".to_owned(),
        artist: "Namitape".to_owned(),
        playing: true,
        position_ms: Some(87_133),
        duration_ms: Some(182_978),
        volume: Some(0.65),
        cover: true,
        can_play_pause: true,
        can_next: true,
        can_previous: true,
        ..SpotifyHud::default()
    }
}

fn variants() -> Vec<Variant> {
    let long = SpotifyHud {
        title: "A Title Far Too Long For The Widget (Extended Club Remix)".to_owned(),
        artist: "Somebody, Somebody Else & Their Orchestra".to_owned(),
        text_revision: 2,
        ..track()
    };
    let paused_hover = SpotifyHud {
        playing: false,
        interactive: true,
        hover: Some(SpotifyControl::PlayPause),
        ..track()
    };
    let no_cover = SpotifyHud {
        cover: false,
        can_previous: false,
        interactive: true,
        hover: Some(SpotifyControl::Volume),
        pressed: Some(SpotifyControl::Volume),
        volume: Some(0.3),
        ..track()
    };
    let mut out = Vec::new();
    for aero in [false, true] {
        let x = if aero { 205.0 } else { 15.0 };
        out.push(Variant {
            name: if aero { "aero_playing" } else { "mc_playing" },
            aero,
            data: track(),
            scale: 1.0,
            at: (x, 8.0),
        });
        out.push(Variant {
            name: if aero {
                "aero_paused_hover"
            } else {
                "mc_paused_hover"
            },
            aero,
            data: paused_hover.clone(),
            scale: 1.0,
            at: (x, 72.0),
        });
        out.push(Variant {
            name: if aero { "aero_no_cover" } else { "mc_no_cover" },
            aero,
            data: no_cover.clone(),
            scale: 1.0,
            at: (x, 136.0),
        });
        out.push(Variant {
            name: if aero {
                "aero_long_scale08"
            } else {
                "mc_long_scale08"
            },
            aero,
            data: long.clone(),
            scale: 0.8,
            at: (x, 200.0),
        });
    }
    out
}

fn settings(aero: bool) -> ModSettings {
    let mut settings = ModSettings::new();
    settings.set_enabled(MOD_ID, true);
    settings.set_choice(MOD_ID, "style", usize::from(aero));
    settings
}

fn rgb(frame: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * W + x) * 4) as usize;
    [frame[i], frame[i + 1], frame[i + 2]]
}

fn near(a: [u8; 3], b: [u8; 3], tolerance: u8) -> bool {
    a.iter().zip(b).all(|(x, y)| x.abs_diff(y) <= tolerance)
}

#[test]
fn both_styles_render_with_a_cover_over_a_scene() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let mut font = FontRenderer::load(&rm, false, false).expect("font");
    let gui = ((W / SCALE) as i32, (H / SCALE) as i32);
    let aero_text = AeroText::new(MenuText::load());

    let mut list = DrawList::new();
    list.enable_solid_merging();
    scene(&mut list, gui);
    let mut drawn = Vec::new();
    for variant in variants() {
        let settings = settings(variant.aero);
        let mut element = SpotifyElement::default();
        element.sync(&settings);
        let data = HudData {
            // 6.5 s after the text appeared: the long lines are mid-scroll.
            now_ms: 6_500,
            spotify: SpotifyHud {
                aero_text: Some(aero_text.clone()),
                ..variant.data.clone()
            },
            ..HudData::default()
        };
        let ctx = HudCtx {
            data: &data,
            settings: &settings,
            screen: (gui.0 as f32, gui.1 as f32),
            pixel_scale: SCALE as f32,
            editing: false,
        };
        // The marquee clock starts on the first layout; lay out once at t = 0.
        let first = HudData {
            now_ms: 0,
            ..data.clone()
        };
        element.layout(
            &HudCtx {
                data: &first,
                ..ctx
            },
            &font,
        );
        let size = element.layout(&ctx, &font).expect("a track is playing");
        list.with_transform(
            Transform::translate(variant.at.0, variant.at.1).then_scale(variant.scale),
            |d| element.draw(&ctx, &mut font, d),
        );
        if variant.aero {
            let (shapes, glyphs) = element.aero_geometry();
            eprintln!(
                "{}: {shapes} shape triangles, {glyphs} glyph quads",
                variant.name
            );
        }
        drawn.push((variant, size, *element.current_layout()));
    }
    eprintln!(
        "frame list: {} vertices, {} batches",
        list.vertices().len(),
        list.batches().len()
    );

    let mut phase = UiPhase::new(&gpu, W, H, SCALE as i32, false);
    let linear = TextureMetadata {
        blur: true,
        clamp: true,
        ..TextureMetadata::default()
    };
    // What the client does in `build_ui`: the cover, then the Segoe atlas if it has new glyphs.
    phase.renderer.upload_with_options(
        &gpu,
        cover_texture().clone(),
        &cover_pixels(),
        COVER,
        COVER,
        &linear,
    );
    if let Some(pixels) = aero_text.0.borrow_mut().take_upload() {
        phase.renderer.upload_with_options(
            &gpu,
            atlas_key(),
            pixels,
            ATLAS_SIZE,
            ATLAS_SIZE,
            &linear,
        );
    }
    let missing = phase.renderer.ensure_list_textures(&gpu, &rm, &list);
    assert!(missing.is_empty(), "textures did not load: {missing:?}");
    phase.set_resolution(ScaledResolution::new(W, H, SCALE as i32, false));
    phase.submit(list);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    assert!(!phases.scene_mode(), "the widget never asks for the scene");
    let target = gpu.acquire().expect("acquire offscreen frame");
    phases.render(&gpu, target.view(), &Camera::default(), 0.0);
    let frame = gpu.read_pixels().expect("readback");

    if let Ok(dir) = std::env::var("MC_W9_RENDER_DIR") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("render dir");
        let whole = image::RgbaImage::from_raw(W, H, frame.clone()).expect("frame size");
        whole.save(dir.join("spotify_all.png")).expect("save");
        for (variant, size, _) in &drawn {
            let x = ((variant.at.0 - 4.0) * SCALE as f32) as u32;
            let y = ((variant.at.1 - 4.0) * SCALE as f32) as u32;
            let w = (((size.0 * variant.scale) + 8.0) * SCALE as f32) as u32;
            let h = (((size.1 * variant.scale) + 8.0) * SCALE as f32) as u32;
            let crop = image::imageops::crop_imm(&whole, x, y, w.min(W - x), h.min(H - y));
            crop.to_image()
                .save(dir.join(format!("spotify_{}.png", variant.name)))
                .expect("save crop");
        }
        eprintln!("saved to {}", dir.display());
    }

    let s = SCALE as f32;
    let px = |v: f32| v.round() as u32;
    for (variant, _, layout) in &drawn {
        let (ox, oy, k) = (variant.at.0, variant.at.1, variant.scale);
        let at = |x: f32, y: f32| (px((ox + x * k) * s), px((oy + y * k) * s));
        // The cover's middle is the pale disc; a point a quarter in is the gradient.
        if variant.data.cover && variant.scale == 1.0 {
            let c = layout.cover.expect("cover shown");
            let (x, y) = at(c.x + c.w * 0.5, c.y + c.h * 0.5);
            assert!(
                near(rgb(&frame, x, y), [250, 240, 220], 6),
                "{}: cover centre {:?}",
                variant.name,
                rgb(&frame, x, y)
            );
            let (x, y) = at(c.x + 3.0, c.y + c.h * 0.5);
            let edge = rgb(&frame, x, y);
            assert!(
                edge[0] > 150 && edge[2] > 60 && edge[1] < 140,
                "{}: cover left edge {edge:?}",
                variant.name
            );
        }
        // The title starts at the text column: some bright (white) pixel in its first glyphs.
        let (x0, y0) = at(layout.text_x, layout.title_y);
        let (x1, y1) = at(layout.text_x + 24.0, layout.title_y + 8.0);
        let bright = (y0..y1)
            .flat_map(|y| (x0..x1).map(move |x| (x, y)))
            .filter(|&(x, y)| rgb(&frame, x, y).iter().all(|&c| c > 215))
            .count();
        assert!(
            bright > 20,
            "{}: title is not drawn ({bright})",
            variant.name
        );
    }

    // The Minecraft panel darkens the scene by the chat's 0x7F alpha. The blue sky at the top
    // of the panel's right edge, between the text and the border.
    let (mc, _, l) = &drawn[0];
    let x = px((mc.at.0 + l.size.0 - 2.0) * s);
    let y = px((mc.at.1 + 2.0) * s);
    let under = rgb(&frame, x, y);
    assert!(
        under[2] < 140 && under[2] > 90,
        "the panel darkens the sky by half: {under:?}"
    );

    // The Aero panel's top-left corner pixel is outside its rounded corner: pure scene.
    let (aero, _, _) = &drawn[4];
    let (cx, cy) = (px(aero.at.0 * s), px(aero.at.1 * s));
    let corner = rgb(&frame, cx, cy);
    let outside = rgb(&frame, cx - 3, cy);
    assert!(
        near(corner, outside, 12),
        "rounded corner: {corner:?} vs the scene {outside:?}"
    );
    // And its body (right of the text, between the info block and the controls) is tinted
    // darker than the scene just right of the widget, at the same height.
    let y = px((aero.at.1 + 37.0) * s);
    let inside = rgb(&frame, px((aero.at.0 + 157.0) * s), y);
    let beside = rgb(&frame, px((aero.at.0 + 163.0) * s), y);
    assert!(
        inside.iter().zip(beside).all(|(a, b)| *a < b),
        "the glass tints the scene: {inside:?} vs {beside:?}"
    );
}

/// "Background opacity" fades the panel only, on screen, in
/// both styles. One column per style, four rows: the old behaviour at 50% (the whole widget
/// faded, emulated with `DrawList::with_opacity` at 100%, exactly what the old draw did), then
/// the new one at 100%, 50% and 0%. The title stays fully bright at 50% and 0% (it was dimmed
/// in the old row: the negative control), and at 0% the scene shows through the panel unchanged.
///
/// `MC_B3_PNG_DIR=<dir>` saves the frame and one crop per row.
#[test]
fn background_opacity_fades_only_the_panel_on_screen() {
    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let mut font = FontRenderer::load(&rm, false, false).expect("font");
    let gui = ((W / SCALE) as i32, (H / SCALE) as i32);
    let aero_text = AeroText::new(MenuText::load());
    // (name, old whole-widget fade, Background opacity %)
    let rows: [(&str, Option<f32>, f32); 4] = [
        ("before_whole_50", Some(0.5), 100.0),
        ("after_100", None, 100.0),
        ("after_50", None, 50.0),
        ("after_0", None, 0.0),
    ];

    let mut list = DrawList::new();
    list.enable_solid_merging();
    scene(&mut list, gui);
    let mut drawn = Vec::new();
    for aero in [false, true] {
        let x = if aero { 205.0 } else { 15.0 };
        for (row, (name, old_fade, background)) in rows.iter().enumerate() {
            let mut settings = settings(aero);
            settings.set_f32(MOD_ID, "opacity", *background);
            let mut element = SpotifyElement::default();
            element.sync(&settings);
            let data = HudData {
                now_ms: 0,
                spotify: SpotifyHud {
                    aero_text: Some(aero_text.clone()),
                    ..track()
                },
                ..HudData::default()
            };
            let ctx = HudCtx {
                data: &data,
                settings: &settings,
                screen: (gui.0 as f32, gui.1 as f32),
                pixel_scale: SCALE as f32,
                editing: false,
            };
            let size = element.layout(&ctx, &font).expect("a track is playing");
            let at = (x, 8.0 + 64.0 * row as f32);
            list.with_transform(Transform::translate(at.0, at.1), |d| match old_fade {
                Some(fade) => d.with_opacity(*fade, |d| element.draw(&ctx, &mut font, d)),
                None => element.draw(&ctx, &mut font, d),
            });
            let style = if aero { "aero" } else { "mc" };
            drawn.push((
                format!("{style}_{name}"),
                at,
                size,
                *element.current_layout(),
            ));
        }
    }

    let mut phase = UiPhase::new(&gpu, W, H, SCALE as i32, false);
    let linear = TextureMetadata {
        blur: true,
        clamp: true,
        ..TextureMetadata::default()
    };
    phase.renderer.upload_with_options(
        &gpu,
        cover_texture().clone(),
        &cover_pixels(),
        COVER,
        COVER,
        &linear,
    );
    if let Some(pixels) = aero_text.0.borrow_mut().take_upload() {
        phase.renderer.upload_with_options(
            &gpu,
            atlas_key(),
            pixels,
            ATLAS_SIZE,
            ATLAS_SIZE,
            &linear,
        );
    }
    let missing = phase.renderer.ensure_list_textures(&gpu, &rm, &list);
    assert!(missing.is_empty(), "textures did not load: {missing:?}");
    phase.set_resolution(ScaledResolution::new(W, H, SCALE as i32, false));
    phase.submit(list);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let target = gpu.acquire().expect("acquire offscreen frame");
    phases.render(&gpu, target.view(), &Camera::default(), 0.0);
    let frame = gpu.read_pixels().expect("readback");

    if let Ok(dir) = std::env::var("MC_B3_PNG_DIR") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("png dir");
        let whole = image::RgbaImage::from_raw(W, H, frame.clone()).expect("frame size");
        whole.save(dir.join("b3_spotify_all.png")).expect("save");
        for (name, at, size, _) in &drawn {
            let x = ((at.0 - 4.0) * SCALE as f32) as u32;
            let y = ((at.1 - 4.0) * SCALE as f32) as u32;
            let w = ((size.0 + 8.0) * SCALE as f32) as u32;
            let h = ((size.1 + 8.0) * SCALE as f32) as u32;
            image::imageops::crop_imm(&whole, x, y, w.min(W - x), h.min(H - y))
                .to_image()
                .save(dir.join(format!("b3_spotify_{name}.png")))
                .expect("save crop");
        }
        eprintln!("saved to {}", dir.display());
    }

    let s = SCALE as f32;
    let px = |v: f32| v.round() as u32;
    let bright_title =
        |at: (f32, f32), layout: &mc_ui::modhud::elements::spotify::SpotifyLayout| {
            let (x0, y0) = (
                px((at.0 + layout.text_x) * s),
                px((at.1 + layout.title_y) * s),
            );
            let (x1, y1) = (
                px((at.0 + layout.text_x + 24.0) * s),
                px((at.1 + layout.title_y + 8.0) * s),
            );
            (y0..y1)
                .flat_map(|y| (x0..x1).map(move |x| (x, y)))
                .filter(|&(x, y)| rgb(&frame, x, y).iter().all(|&c| c > 215))
                .count()
        };
    for (column, style) in [(0, "mc"), (4, "aero")] {
        let before = &drawn[column];
        let full = &drawn[column + 1];
        let half = &drawn[column + 2];
        let none = &drawn[column + 3];
        // The title: dimmed by the old whole-widget fade, fully bright at 50% and 0% now.
        assert!(
            bright_title(before.1, &before.3) < 5,
            "{style}: the old fade dimmed the title (negative control)"
        );
        for row in [full, half, none] {
            let lit = bright_title(row.1, &row.3);
            assert!(lit > 20, "{}: the title is fully bright ({lit})", row.0);
        }
        // A panel pixel with no content (Minecraft: its top-right corner; Aero: right of the
        // text, as above) against the scene just right of the widget at the same height.
        let probe = |row: &(String, (f32, f32), (f32, f32), _)| {
            let (dx, dy) = if style == "mc" {
                (row.2 .0 - 2.0, 2.0)
            } else {
                (157.0, 37.0)
            };
            let y = px((row.1 .1 + dy) * s);
            (
                rgb(&frame, px((row.1 .0 + dx) * s), y),
                rgb(&frame, px((row.1 .0 + row.2 .0 + 6.0) * s), y),
            )
        };
        let (inside, beside) = probe(full);
        assert!(
            inside.iter().zip(beside).all(|(a, b)| *a + 20 < b),
            "{style}: 100% is the full panel: {inside:?} vs {beside:?}"
        );
        let (inside_half, _) = probe(half);
        assert!(
            inside_half.iter().zip(inside).all(|(a, b)| *a > b),
            "{style}: 50% is lighter than 100%: {inside_half:?} vs {inside:?}"
        );
        let (inside, beside) = probe(none);
        assert!(
            near(inside, beside, 6),
            "{style}: 0% shows the scene unchanged: {inside:?} vs {beside:?}"
        );
    }
}
