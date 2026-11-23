//! Pictures of the replay viewer's Sound menu (each channel's
//! mute and volume) and of the export window's Sound row (game, microphone, desktop audio and
//! separate tracks), rendered for real (wgpu, the Aero glass phase, the vanilla font from the jar)
//! over the SkyWars backdrop the render test uses. The harness is the (`nw_ra_replay_render.rs`).
//!
//! Each frame runs under a wgpu validation scope and must put near-white text where the window's
//! title is. With `MC_RE_PNG_DIR=<dir>` every frame is saved there (`<state>-<w>x<h>.png`).
//! Needs the vanilla jar (`MC_JARS_DIR`) and a GPU adapter; skips loudly without either.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mc_assets::{ResourceLocation, TextureMetadata};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::FontRenderer;
use mc_ui::screens::modmenu::glass::{GlassFrame, ModMenuPhase};
use mc_ui::screens::modmenu::paint::{white, Painter, Rect};
use mc_ui::screens::modmenu::text::{MenuText, ATLAS_SIZE};
use mc_ui::screens::replay::{
    key, PlaybackStatus, ReplayEnv, ReplayInfo, ReplayViewerOverlay, ViewerEvent,
};
use mc_ui::widget::keys::Modifiers;
use mc_ui::widget::{DrawList, UiPhase, UiTexture};

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

fn picture(relative: &str) -> Option<(Vec<u8>, u32, u32)> {
    let image = image::open(root().join(relative)).ok()?.to_rgba8();
    let (w, h) = image.dimensions();
    Some((image.into_raw(), w, h))
}

const SKYWARS: &str = "featwave-2026-11-09/scratch/X9/bench/off/trace/shots/fight_start.png";

struct Harness {
    rm: mc_assets::ResourceManager,
    text: MenuText,
    font: FontRenderer,
    t0: Instant,
}

impl Harness {
    fn at(&self, ms: u64) -> Instant {
        self.t0 + Duration::from_millis(ms)
    }

    fn env(&mut self, size: (u32, u32), ms: u64) -> ReplayEnv<'_> {
        ReplayEnv {
            text: &mut self.text,
            mc_font: &mut self.font,
            screen: (size.0 as f32, size.1 as f32),
            now: self.t0 + Duration::from_millis(ms),
            time_s: 1.5 + ms as f32 / 1000.0,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn render(
    gpu: &mut Gpu,
    h: &mut Harness,
    size: (u32, u32),
    backdrop: Option<&(Vec<u8>, u32, u32)>,
    list: DrawList,
    glass: GlassFrame,
    state: &str,
    title_at: Rect,
) {
    let (w, hgt) = size;
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut ui = UiPhase::new(gpu, w, hgt, 1, false);
    let mut scene = DrawList::new();
    let key = UiTexture::Resource(ResourceLocation::new("mcrs", "test/re_backdrop"));
    match backdrop {
        Some((rgba, bw, bh)) => {
            ui.renderer.upload_with_options(
                gpu,
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
                Rect::new(0.0, 0.0, w as f32, hgt as f32),
                [0.0, 0.0, 1.0, 1.0],
                white(1.0),
            );
        }
        None => scene.gradient_rect(0, 0, w as i32, hgt as i32, 0xFF6F_B8EA, 0xFF2C_6E31),
    }
    ui.submit(scene);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(ui));
    let mut phase = ModMenuPhase::new(gpu);
    if let Some(pixels) = h.text.take_upload() {
        phase.upload_atlas(gpu, pixels, ATLAS_SIZE);
    }
    if !h.text.is_segoe() {
        let _ = phase.renderer.ensure_list_textures(gpu, &h.rm, &list);
    }
    phase.submit(list, glass);
    phases.insert(PhaseOrder(PhaseOrder::UI.0 + 85), Box::new(phase));
    let target = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, target.view(), &Camera::default(), 0.0);
    if let Some(error) = ready(scope.pop()) {
        panic!("wgpu validation error drawing {state}: {error}");
    }
    let px = gpu.read_pixels().expect("readback");
    let mut brightest = 0u8;
    let (x0, y0) = (title_at.x.max(0.0) as u32, title_at.y.max(0.0) as u32);
    let (x1, y1) = (
        (title_at.right() as u32).min(w),
        (title_at.bottom() as u32).min(hgt),
    );
    for y in y0..y1 {
        for x in x0..x1 {
            let i = ((y * w + x) * 4) as usize;
            brightest = brightest.max(px[i].min(px[i + 1]).min(px[i + 2]));
        }
    }
    assert!(
        brightest > 200,
        "{state} {w}x{hgt}: title text drawn ({brightest})"
    );
    h.text.mark_dirty();
    if let Some(dir) = std::env::var_os("MC_RE_PNG_DIR") {
        std::fs::create_dir_all(&dir).expect("png dir");
        let path = Path::new(&dir).join(format!("{state}-{w}x{hgt}.png"));
        image::save_buffer(&path, &px, w, hgt, image::ExtendedColorType::Rgba8)
            .expect("write the PNG");
        eprintln!("saved {}", path.display());
    }
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

fn info(size: (u32, u32)) -> ReplayInfo {
    ReplayInfo {
        title: "sky.example.org \u{b7} Nov 13, 9:30 PM".to_owned(),
        server: "sky.example.org".to_owned(),
        started_unix_ms: 1_794_623_439_617,
        clip: false,
        has_mic: true,
        markers: Vec::new(),
        window_size: size,
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn the_sound_menu_and_the_export_sound_row_render() {
    let Some(rm) = jar_assets() else {
        eprintln!("SKIP: no vanilla jar (set MC_JARS_DIR)");
        return;
    };
    let skywars = picture(SKYWARS);
    for size in [(1920, 1080), (1280, 720)] {
        let Some(mut gpu) = gpu_for(size) else {
            return;
        };
        let mut h = Harness {
            font: FontRenderer::load(&rm, false, false).expect("vanilla font"),
            rm: jar_assets().expect("jar"),
            text: MenuText::load(),
            t0: Instant::now(),
        };
        let sz = (size.0 as f32, size.1 as f32);
        let s = mc_ui::screens::replay::style::scale(sz);
        let status = PlaybackStatus {
            duration_us: 162_230_000,
            time_us: 21_000_000,
            playing: true,
            speed: 1.0,
            ..PlaybackStatus::default()
        };

        // ---- the Sound menu: the game at 100%, the microphone at 80%, desktop audio muted ----
        let mut viewer = ReplayViewerOverlay::new(info(size), h.t0);
        viewer.set_has_desktop(true);
        viewer.set_sound_channels([(false, 1.0), (false, 0.8), (true, 1.0)]);
        viewer.set_status(status);
        let (mut list, mut glass) = (DrawList::new(), GlassFrame::default());
        let _ = viewer.build(&mut h.env(size, 50), &mut list);
        viewer.open_sound_menu();
        let bar = ReplayViewerOverlay::bar_rect(sz);
        for ms in (100..=900).step_by(50) {
            if ms == 300 {
                // Hover the bar so it stays shown.
                viewer.pointer_moved(bar.x + bar.w * 0.5, bar.y + 10.0 * s, h.at(ms));
            }
            list = DrawList::new();
            viewer.set_status(status);
            glass = viewer.build(&mut h.env(size, ms), &mut list);
        }
        render(
            &mut gpu,
            &mut h,
            size,
            skywars.as_ref(),
            list,
            glass,
            "viewer-sound-menu",
            Rect::new(20.0 * s, 18.0 * s, 260.0 * s, 40.0 * s),
        );
        // The buttons still fit without overlapping (the Sound button joined the right group).
        let rects = viewer.control_rects();
        for (i, a) in rects.iter().enumerate() {
            assert!(
                a.x >= 0.0 && a.right() <= sz.0 + 0.5 && a.y >= 0.0 && a.bottom() <= sz.1 + 0.5,
                "a control leaves the screen at {size:?}: {a:?}"
            );
            for b in &rects[i + 1..] {
                let overlap = a.x < b.right() - 0.5
                    && b.x < a.right() - 0.5
                    && a.y < b.bottom() - 0.5
                    && b.y < a.bottom() - 0.5;
                assert!(!overlap, "controls overlap at {size:?}: {a:?} {b:?}");
            }
        }
        // Esc closes the menu first (the replay stays open).
        viewer.key(key::ESCAPE, None, Modifiers::NONE, h.at(950));
        assert!(
            !viewer
                .drain_events()
                .any(|e| matches!(e, ViewerEvent::Exit)),
            "Esc closed the Sound menu, not the replay"
        );

        // ---- the export window's Sound row: Game, Mic, Desktop and Tracks ----
        let mut viewer = ReplayViewerOverlay::new(info(size), h.t0);
        viewer.set_has_desktop(true);
        viewer.set_status(status);
        let (mut list, mut glass) = (DrawList::new(), GlassFrame::default());
        let _ = viewer.build(&mut h.env(size, 50), &mut list);
        viewer.key(key::E, None, Modifiers::NONE, h.at(60));
        for ms in (100..=800).step_by(50) {
            list = DrawList::new();
            viewer.set_status(status);
            glass = viewer.build(&mut h.env(size, ms), &mut list);
        }
        render(
            &mut gpu,
            &mut h,
            size,
            skywars.as_ref(),
            list,
            glass,
            "export-sound-row",
            Rect::new(
                sz.0 * 0.5 - 300.0 * s,
                sz.1 * 0.5 - 360.0 * s,
                300.0 * s,
                80.0 * s,
            ),
        );
    }
}
