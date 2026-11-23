//! The replay viewer's Options button (the game's own options over
//! the replay; the client side is `mc-client/src/app/mods/nw_replay_options.rs`).
//!
//! * Headless logic with a deterministic 5-pixel font: the cog button (found by its tooltip, the
//!   way a player finds it) and Ctrl+, send `ViewerEvent::Options`; a plain `,` still steps one
//!   tick back; the bar still lays out without overlaps at common sizes.
//! * A real render (wgpu, the Aero glass phase, the vanilla font from the jar) of the bar over a
//!   SkyWars frame with the pointer on the cog and its tooltip up. With `MC_RD_PNG_DIR=<dir>` the
//!   frames are saved there. Needs the vanilla jar (`MC_JARS_DIR`) and a GPU; skips without.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mc_assets::{ResourceLocation, TextureMetadata};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::{FontData, FontRenderer};
use mc_ui::screens::modmenu::glass::{GlassFrame, ModMenuPhase};
use mc_ui::screens::modmenu::paint::{white, Painter, Rect};
use mc_ui::screens::modmenu::text::{MenuText, ATLAS_SIZE};
use mc_ui::screens::replay::{
    key, PlaybackStatus, ReplayEnv, ReplayInfo, ReplayViewerOverlay, ViewerEvent,
};
use mc_ui::widget::keys::Modifiers;
use mc_ui::widget::{DrawList, UiPhase, UiTexture};

const TIP: &str = "Options: video, chat, controls...";

/// The Minecraft font with every glyph 5 pixels wide (as the logic tests build it).
fn uniform_font() -> FontRenderer {
    let mut img = image::RgbaImage::new(128, 128);
    for cell in 0..256u32 {
        let cx = (cell % 16) * 8;
        let cy = (cell / 16) * 8;
        for col in 0..4 {
            img.put_pixel(cx + col, cy + 2, image::Rgba([255, 255, 255, 255]));
        }
    }
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut png, image::ImageFormat::Png)
        .expect("encode");
    let data = FontData::from_parts(&png.into_inner(), &[0u8; 0x1_0000]).expect("font data");
    FontRenderer::new(data, false, false)
}

fn info(size: (u32, u32)) -> ReplayInfo {
    ReplayInfo {
        title: "SkyWars \u{b7} sky.example.org".to_owned(),
        server: "sky.example.org".to_owned(),
        started_unix_ms: 1_794_626_723_000,
        clip: false,
        has_mic: false,
        markers: vec![(41_000_000, "Kill".to_owned())],
        window_size: size,
    }
}

fn status() -> PlaybackStatus {
    PlaybackStatus {
        duration_us: 168_000_000,
        time_us: 41_200_000,
        playing: false,
        speed: 1.0,
        ..PlaybackStatus::default()
    }
}

fn env_at<'a>(
    text: &'a mut MenuText,
    font: &'a mut FontRenderer,
    screen: (f32, f32),
    t0: Instant,
    ms: u64,
) -> ReplayEnv<'a> {
    ReplayEnv {
        text,
        mc_font: font,
        screen,
        now: t0 + Duration::from_millis(ms),
        time_s: ms as f32 / 1000.0,
    }
}

/// The first point (on a 4-pixel grid) whose control's tooltip says `label`.
fn find_tip(overlay: &ReplayViewerOverlay, screen: (f32, f32), label: &str) -> Option<(f32, f32)> {
    let mut y = 0.0;
    while y < screen.1 {
        let mut x = 0.0;
        while x < screen.0 {
            if overlay.tooltip_at(x, y) == Some(label) {
                return Some((x, y));
            }
            x += 4.0;
        }
        y += 4.0;
    }
    None
}

#[test]
fn the_cog_and_ctrl_comma_open_the_options_and_comma_still_steps() {
    let screen = (1920.0, 1080.0);
    let mut text = MenuText::minecraft_only();
    let mut font = uniform_font();
    let t0 = Instant::now();
    let mut overlay = ReplayViewerOverlay::new(info((1920, 1080)), t0);
    overlay.set_status(status());
    let mut list = DrawList::new();
    for ms in (0..=600).step_by(16) {
        list.clear();
        let _ = overlay.build(&mut env_at(&mut text, &mut font, screen, t0, ms), &mut list);
    }
    let (x, y) = find_tip(&overlay, screen, TIP).expect("an Options button with its tooltip");
    let now = t0 + Duration::from_millis(700);
    overlay.pointer_moved(x, y, now);
    overlay.mouse_pressed(x, y, 0, now);
    overlay.mouse_released(x, y, 0, now);
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Options]
    );
    let ctrl = Modifiers {
        ctrl: true,
        ..Modifiers::NONE
    };
    overlay.key(key::COMMA, None, ctrl, now);
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Options]
    );
    // Without Ctrl, `,` is still one tick back.
    overlay.key(key::COMMA, None, Modifiers::NONE, now);
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Step(-1)]
    );
}

#[test]
fn the_bar_still_fits_with_the_options_button() {
    for size in [
        (854.0, 480.0),
        (1024.0, 768.0),
        (1280.0, 720.0),
        (1920.0, 1080.0),
        (2560.0, 1080.0),
    ] {
        let mut text = MenuText::minecraft_only();
        let mut font = uniform_font();
        let t0 = Instant::now();
        let mut overlay = ReplayViewerOverlay::new(info((size.0 as u32, size.1 as u32)), t0);
        overlay.set_status(status());
        overlay.set_selection(Some(1_000_000), Some(9_000_000));
        let mut list = DrawList::new();
        let _ = overlay.build(&mut env_at(&mut text, &mut font, size, t0, 800), &mut list);
        assert!(
            find_tip(&overlay, size, TIP).is_some(),
            "{size:?}: the Options button is there"
        );
        let rects = overlay.control_rects();
        for (i, a) in rects.iter().enumerate() {
            assert!(
                a.x >= 0.0 && a.right() <= size.0,
                "{size:?}: {a:?} off screen"
            );
            for b in &rects[i + 1..] {
                let overlap = a.intersect(b);
                assert!(
                    overlap.w <= 1.0 || overlap.h <= 1.0,
                    "{size:?}: {a:?} overlaps {b:?}"
                );
            }
        }
    }
}

// ---- the real render ------------------------------------------------------------------------

/// wgpu's native error-scope futures are ready at once; poll without an executor.
fn ready<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    for _ in 0..1000 {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
        std::thread::yield_now();
    }
    panic!("the error scope never resolved");
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("the project folder")
        .to_path_buf()
}

fn jar_assets() -> Option<mc_assets::ResourceManager> {
    let jars = PathBuf::from(std::env::var_os("MC_JARS_DIR")?);
    let jar = jars.join("versions").join("1.8.9").join("1.8.9.jar");
    let pack = mc_assets::ZipResourcePack::open(&jar).ok()?;
    let mut rm = mc_assets::ResourceManager::new();
    rm.push_pack(Box::new(pack));
    Some(rm)
}

/// A SkyWars frame from earlier benchmark pictures, or `None` (a gradient instead).
fn skywars() -> Option<(Vec<u8>, u32, u32)> {
    let path = root().join("featwave-2026-11-09/scratch/X9/bench/off/trace/shots/fight_start.png");
    let image = image::open(path).ok()?.to_rgba8();
    let (w, h) = image.dimensions();
    Some((image.into_raw(), w, h))
}

fn gpu_for(size: (u32, u32)) -> Option<Gpu> {
    match Gpu::headless(
        size.0,
        size.1,
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
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

#[test]
fn the_options_button_renders_with_its_tooltip() {
    let Some(rm) = jar_assets() else {
        eprintln!("SKIP: set MC_JARS_DIR to the folder holding versions/1.8.9/1.8.9.jar");
        return;
    };
    let backdrop = skywars();
    for size in [(1280u32, 720u32), (1920, 1080)] {
        let Some(mut gpu) = gpu_for(size) else {
            return;
        };
        let sz = (size.0 as f32, size.1 as f32);
        let mut text = MenuText::load();
        let mut font = FontRenderer::load(&rm, false, false).expect("vanilla font");
        let t0 = Instant::now();
        let mut overlay = ReplayViewerOverlay::new(info(size), t0);
        overlay.set_status(status());
        let (mut list, mut glass) = (DrawList::new(), GlassFrame::default());
        let _ = overlay.build(&mut env_at(&mut text, &mut font, sz, t0, 50), &mut list);
        let (x, y) = find_tip(&overlay, sz, TIP).expect("the Options button");
        overlay.pointer_moved(x, y, t0 + Duration::from_millis(60));
        // Long enough for the tooltip to fade in.
        for ms in (100..=1400).step_by(50) {
            list = DrawList::new();
            overlay.set_status(status());
            glass = overlay.build(&mut env_at(&mut text, &mut font, sz, t0, ms), &mut list);
        }

        let (w, h) = size;
        let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut ui = UiPhase::new(&gpu, w, h, 1, false);
        let mut scene = DrawList::new();
        let key = UiTexture::Resource(ResourceLocation::new("mcrs", "test/rd_backdrop"));
        match &backdrop {
            Some((rgba, bw, bh)) => {
                ui.renderer.upload_with_options(
                    &gpu,
                    key.clone(),
                    rgba,
                    *bw,
                    *bh,
                    &TextureMetadata {
                        blur: true,
                        clamp: true,
                        ..TextureMetadata::default()
                    },
                );
                let mut painter = Painter::new(&mut scene);
                painter.textured(
                    &key,
                    Rect::new(0.0, 0.0, w as f32, h as f32),
                    [0.0, 0.0, 1.0, 1.0],
                    white(1.0),
                );
            }
            None => scene.gradient_rect(0, 0, w as i32, h as i32, 0xFF6F_B8EA, 0xFF2C_6E31),
        }
        ui.submit(scene);
        let mut phases = PhaseList::new();
        phases.clear_color = wgpu::Color::BLACK;
        phases.insert(PhaseOrder::UI, Box::new(ui));
        let mut phase = ModMenuPhase::new(&gpu);
        if let Some(pixels) = text.take_upload() {
            phase.upload_atlas(&gpu, pixels, ATLAS_SIZE);
        }
        if !text.is_segoe() {
            let _ = phase.renderer.ensure_list_textures(&gpu, &rm, &list);
        }
        phase.submit(list, glass);
        // The client registers the replay UI at UI + 85.
        phases.insert(PhaseOrder(PhaseOrder::UI.0 + 85), Box::new(phase));
        let target = gpu.acquire().expect("acquire offscreen frame");
        phases.render(&gpu, target.view(), &Camera::default(), 0.0);
        if let Some(error) = ready(scope.pop()) {
            panic!("wgpu validation error drawing the viewer: {error}");
        }
        let px = gpu.read_pixels().expect("readback");
        // The cog itself: near-white strokes around the button's centre (the middle of the area
        // whose tooltip is the button's).
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, 0.0f32, 0.0f32);
        let mut py = 0.0;
        while py < sz.1 {
            let mut px_ = 0.0;
            while px_ < sz.0 {
                if overlay.tooltip_at(px_, py) == Some(TIP) {
                    (x0, y0, x1, y1) = (x0.min(px_), y0.min(py), x1.max(px_), y1.max(py));
                }
                px_ += 2.0;
            }
            py += 2.0;
        }
        let (cx, cy) = (((x0 + x1) * 0.5) as u32, ((y0 + y1) * 0.5) as u32);
        let mut brightest = 0u8;
        for yy in cy.saturating_sub(10)..(cy + 10).min(h) {
            for xx in cx.saturating_sub(10)..(cx + 10).min(w) {
                let i = ((yy * w + xx) * 4) as usize;
                brightest = brightest.max(px[i].min(px[i + 1]).min(px[i + 2]));
            }
        }
        assert!(brightest > 180, "{w}x{h}: the cog is drawn ({brightest})");
        if let Some(dir) = std::env::var_os("MC_RD_PNG_DIR") {
            std::fs::create_dir_all(&dir).expect("png dir");
            let path = Path::new(&dir).join(format!("viewer-options-button-{w}x{h}.png"));
            image::save_buffer(&path, &px, w, h, image::ExtendedColorType::Rgba8)
                .expect("write the PNG");
            eprintln!("saved {}", path.display());
        }
    }
}
