//! The Nametags mod's style, through the production
//! label path (`nameplate::submit_cached` into a real `NameplatePhase`) on a headless GPU with the
//! vanilla `ascii.png`.
//!
//! * **Text shadow**: the shadow is drawn and the text stays whole on top of it. The text sits at
//!   the shadow's depth, so shadows depth-test without writing over their own foreground.
//!   The negative control shows the old depth-writing shadow losing the text's overlap.
//! * **Background opacity** and **scale** change only what they say.
//! * **The vanilla style** is the old frame, pixel for pixel.
//!
//! Skips (loudly) without a wgpu adapter or the vanilla jar (`MC_JARS_DIR`).

use std::path::{Path, PathBuf};

use glam::Vec3;
use mc_assets::{ResourceManager, ZipResourcePack};
use mc_render::entity::nameplate::{NameplateStyle, NAMEPLATE_SCALE};
use mc_render::entity::{Nameplate, NameplatePhase};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::{FontData, FontRenderer};
use mc_ui::nameplate::{self, NameplateCache, NameplateRequest};

const W: u32 = 256;
const H: u32 = 256;
/// Framebuffer pixels per label pixel at the plate distance.
const SCALE_PX: f32 = 6.0;
/// No red in the clear colour: red marks glyph ink (white text, grey shadow).
const CLEAR: wgpu::Color = wgpu::Color {
    r: 0.0,
    g: 0.0,
    b: 0.4,
    a: 1.0,
};

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

fn assets() -> Option<ResourceManager> {
    let dir = match std::env::var("MC_JARS_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)?
            .join("MCP-919-main")
            .join("jars"),
    };
    let jar = dir.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(rm)
}

fn camera() -> Camera {
    let mut cam = Camera::default();
    cam.set_viewport(W, H);
    cam.update();
    cam
}

/// A label whose anchor is the middle of the frame, `SCALE_PX` pixels per label pixel.
fn centred_plate(text: &str, cam: &Camera) -> Nameplate {
    let distance = cam.proj.x_axis.x * W as f32 * NAMEPLATE_SCALE / (2.0 * SCALE_PX);
    Nameplate {
        text: text.to_owned(),
        rel: Vec3::new(0.0, -0.5, distance),
        height: 0.0,
        child: false,
        sneaking: false,
    }
}

/// One frame of `text` drawn through the cache in `cache_style`, the phase told `phase_style`.
fn render(
    gpu: &mut Gpu,
    font: &mut FontRenderer,
    text: &str,
    cache_style: NameplateStyle,
    phase_style: NameplateStyle,
) -> Vec<u8> {
    render_at(gpu, font, text, cache_style, phase_style, &camera())
}

fn render_at(
    gpu: &mut Gpu,
    font: &mut FontRenderer,
    text: &str,
    cache_style: NameplateStyle,
    phase_style: NameplateStyle,
    cam: &Camera,
) -> Vec<u8> {
    let mut plate = centred_plate(text, cam);
    let distance = plate.rel.z;
    let anchor = cam
        .view
        .inverse()
        .transform_point3(Vec3::new(0.0, 0.0, -distance));
    plate.rel = anchor - Vec3::new(0.0, 0.5, 0.0);
    let mut phase = NameplatePhase::new(gpu);
    nameplate::upload_ascii_font(&mut phase, gpu, font.data());
    phase.set_style(phase_style);
    let mut cache = NameplateCache::new();
    cache.set_style(cache_style);
    cache.begin_frame();
    phase.begin();
    assert!(nameplate::submit_cached(
        font,
        &mut cache,
        &NameplateRequest {
            plate: &plate,
            view_yaw: cam.yaw,
            view_pitch: cam.pitch,
            distance_sq: 4.0,
            score: None,
        },
        &mut phase,
    ));
    phase.finish(gpu);
    let mut phases = PhaseList::new();
    phases.clear_color = CLEAR;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &cam, 0.0);
    frame.present(&gpu.queue);
    gpu.read_pixels().expect("read back")
}

fn pixels_where(px: &[u8], f: impl Fn([u8; 4]) -> bool) -> usize {
    px.chunks_exact(4)
        .filter(|p| f([p[0], p[1], p[2], p[3]]))
        .count()
}

fn white(p: [u8; 4]) -> bool {
    p[0] > 200 && p[1] > 200 && p[2] > 200
}

/// The in-front shadow colour, `(0xFFFFFF & 0xFCFCFC) >> 2` = 63 grey.
fn shadow_grey(p: [u8; 4]) -> bool {
    (45..=80).contains(&p[0]) && (45..=80).contains(&p[1])
}

/// Horizontal extent of the white ink, in framebuffer pixels.
fn ink_width(px: &[u8]) -> u32 {
    let mut min = W;
    let mut max = 0;
    for y in 0..H {
        for x in 0..W {
            let i = ((y * W + x) * 4) as usize;
            if white([px[i], px[i + 1], px[i + 2], px[i + 3]]) {
                min = min.min(x);
                max = max.max(x);
            }
        }
    }
    max.saturating_sub(min)
}

macro_rules! setup {
    () => {{
        let Some(rm) = assets() else {
            eprintln!("SKIP: vanilla jar not found; set MC_JARS_DIR");
            return;
        };
        let Some(gpu) = gpu() else {
            return;
        };
        let font = match FontData::load(&rm) {
            Ok(data) => FontRenderer::new(data, false, false),
            Err(e) => {
                eprintln!("SKIP: vanilla font unavailable: {e}");
                return;
            }
        };
        (gpu, font)
    }};
}

#[test]
fn the_shadow_draws_under_whole_text() {
    let (mut gpu, mut font) = setup!();
    let shadowed = NameplateStyle {
        text_shadow: true,
        ..NameplateStyle::VANILLA
    };
    let plain = render(
        &mut gpu,
        &mut font,
        "Mosaic",
        NameplateStyle::VANILLA,
        NameplateStyle::VANILLA,
    );
    let with_shadow = render(&mut gpu, &mut font, "Mosaic", shadowed, shadowed);
    let (plain_white, plain_grey) = (
        pixels_where(&plain, white),
        pixels_where(&plain, shadow_grey),
    );
    let (shadow_white, shadow_grey_px) = (
        pixels_where(&with_shadow, white),
        pixels_where(&with_shadow, shadow_grey),
    );
    assert!(plain_white > 500, "the label is drawn: {plain_white}");
    assert_eq!(plain_grey, 0, "vanilla: no shadow");
    assert!(
        shadow_grey_px > plain_white / 4,
        "the shadow shows below/right of the strokes: {shadow_grey_px}"
    );
    assert!(
        shadow_white.abs_diff(plain_white) * 50 <= plain_white,
        "the text is whole over its shadow: {shadow_white} vs {plain_white} white pixels"
    );

    // Negative control: the shadow recorded but the phase left on vanilla's `Less` pipeline.
    let broken = render(
        &mut gpu,
        &mut font,
        "Mosaic",
        shadowed,
        NameplateStyle::VANILLA,
    );
    let broken_white = pixels_where(&broken, white);
    assert!(
        broken_white * 10 < plain_white * 9,
        "with a depth-writing shadow the text loses its overlap: {broken_white} vs {plain_white}"
    );
}

#[test]
fn background_opacity_and_scale_do_what_they_say() {
    let (mut gpu, mut font) = setup!();
    let vanilla = render(
        &mut gpu,
        &mut font,
        "Mosaic",
        NameplateStyle::VANILLA,
        NameplateStyle::VANILLA,
    );
    let none = NameplateStyle {
        background_alpha: 0.0,
        ..NameplateStyle::VANILLA
    };
    let no_backdrop = render(&mut gpu, &mut font, "Mosaic", none, none);
    let dark = NameplateStyle {
        background_alpha: 0.75,
        ..NameplateStyle::VANILLA
    };
    let dark_backdrop = render(&mut gpu, &mut font, "Mosaic", dark, dark);
    // Through-walls glyph ghost aside, the backdrop is all that darkens the clear blue (102).
    let (v, n, d) = (
        pixels_where(&vanilla, |p| p[2] < 95 && p[0] < 30),
        pixels_where(&no_backdrop, |p| p[2] < 95 && p[0] < 30),
        pixels_where(&dark_backdrop, |p| p[2] < 40 && p[0] < 30),
    );
    assert!(v > 1000, "vanilla's 25% backdrop darkens a box: {v}");
    assert_eq!(n, 0, "0% removes the backdrop");
    assert!(d > 1000, "75% is darker than vanilla's 25%: {d}");

    // A short name, so the doubled label still fits the frame.
    let big = NameplateStyle {
        scale: 2.0,
        ..NameplateStyle::VANILLA
    };
    let single = render(
        &mut gpu,
        &mut font,
        "Hi",
        NameplateStyle::VANILLA,
        NameplateStyle::VANILLA,
    );
    let doubled = render(&mut gpu, &mut font, "Hi", big, big);
    let (w1, w2) = (ink_width(&single), ink_width(&doubled));
    assert!(
        w2.abs_diff(2 * w1) <= 2 * SCALE_PX as u32,
        "scale 2 doubles the label: {w1} -> {w2}"
    );
}

#[test]
fn the_vanilla_style_is_the_old_frame() {
    let (mut gpu, mut font) = setup!();
    let cam = camera();
    let plate = centred_plate("Mosaic", &cam);
    // The uncached reference path (`build`) never reads a style.
    let mut phase = NameplatePhase::new(&gpu);
    nameplate::upload_ascii_font(&mut phase, &gpu, font.data());
    phase.begin();
    nameplate::build(
        &mut font,
        &NameplateRequest {
            plate: &plate,
            view_yaw: cam.yaw,
            view_pitch: cam.pitch,
            distance_sq: 4.0,
            score: None,
        },
    )
    .expect("in range")
    .submit(&mut phase);
    phase.finish(&gpu);
    let mut phases = PhaseList::new();
    phases.clear_color = CLEAR;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    frame.present(&gpu.queue);
    let reference = gpu.read_pixels().expect("read back");
    let styled = render(
        &mut gpu,
        &mut font,
        "Mosaic",
        NameplateStyle::VANILLA,
        NameplateStyle::VANILLA,
    );
    assert!(
        reference == styled,
        "the vanilla style draws the old frame bit for bit"
    );
}

#[test]
fn shadowed_scaled_text_stays_whole_while_the_camera_turns() {
    let (mut gpu, mut font) = setup!();
    for (yaw, pitch, scale) in [
        (31.7, -18.3, 1.35),
        (119.4, 27.1, 1.85),
        (249.3, -42.6, 0.75),
    ] {
        let mut cam = camera();
        cam.yaw = yaw;
        cam.pitch = pitch;
        cam.update();
        let plain_style = NameplateStyle {
            scale,
            background_alpha: 0.7,
            ..NameplateStyle::VANILLA
        };
        let shadow_style = NameplateStyle {
            text_shadow: true,
            ..plain_style
        };
        let plain = render_at(
            &mut gpu,
            &mut font,
            "Mosaic",
            plain_style,
            plain_style,
            &cam,
        );
        let shadowed = render_at(
            &mut gpu,
            &mut font,
            "Mosaic",
            shadow_style,
            shadow_style,
            &cam,
        );
        let n = pixels_where(&plain, white);
        let missing = plain
            .chunks_exact(4)
            .zip(shadowed.chunks_exact(4))
            .filter(|(a, b)| white([a[0], a[1], a[2], a[3]]) && !white([b[0], b[1], b[2], b[3]]))
            .count();
        eprintln!("yaw={yaw} pitch={pitch} scale={scale}: {missing}/{n} foreground pixels lost");
        assert!(n > 300, "label must draw");
        assert!(missing * 100 <= n, "shadow punched holes in foreground at yaw={yaw} pitch={pitch} scale={scale}: {missing}/{n}");
    }
}
