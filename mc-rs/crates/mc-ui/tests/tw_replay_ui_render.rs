//! The replay browser, the viewer's controls and the export window
//! rendered for real (wgpu, Segoe UI, the Aero glass phase, the vanilla font from the jar) over
//! real backdrops: the title screen's greenhouse for the browser, a SkyWars frame for the viewer.
//! Also the title screen itself with its new Replays button.
//!
//! Each frame runs under a wgpu validation scope and must put near-white text where the screen's
//! title is. With `MC_RC_PNG_DIR=<dir>` every frame is saved there (`<state>-<w>x<h>.png`) for a
//! human look. Backdrops come from earlier waves' pictures; a plain gradient stands in when they are missing.
//!
//! Needs the vanilla jar (`MC_JARS_DIR`) and a GPU adapter; skips loudly without either.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mc_assets::{ResourceLocation, TextureMetadata};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::FontRenderer;
use mc_ui::screens::aero_title::font::TitleFont;
use mc_ui::screens::aero_title::phase::AeroTitlePhase;
use mc_ui::screens::aero_title::{AeroButton, AeroTitleEnv, AeroTitleScreen};
use mc_ui::screens::modmenu::glass::{GlassFrame, ModMenuPhase};
use mc_ui::screens::modmenu::paint::{white, Painter, Rect};
use mc_ui::screens::modmenu::text::{MenuText, ATLAS_SIZE};
use mc_ui::screens::replay::browser::TOOLBAR_H;
use mc_ui::screens::replay::export::{ExportProgress, ExportSettings, ExportState};
use mc_ui::screens::replay::{
    key, LibraryView, PlaybackStatus, ReplayBrowserScreen, ReplayEntry, ReplayEnv, ReplayInfo,
    ReplayViewerOverlay,
};
use mc_ui::widget::keys::Modifiers;
use mc_ui::widget::{DrawList, UiPhase, UiTexture};

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
    let jars = match std::env::var("MC_JARS_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => root().join("MCP-919-main").join("jars"),
    };
    let jar = jars.join("versions").join("1.8.9").join("1.8.9.jar");
    let pack = mc_assets::ZipResourcePack::open(&jar).ok()?;
    let mut rm = mc_assets::ResourceManager::new();
    rm.push_pack(Box::new(pack));
    Some(rm)
}

/// A backdrop picture as RGBA, or `None` (a gradient stands in).
fn picture(relative: &str) -> Option<(Vec<u8>, u32, u32)> {
    let path = root().join(relative);
    let image = image::open(&path).ok()?.to_rgba8();
    let (w, h) = image.dimensions();
    Some((image.into_raw(), w, h))
}

const GREENHOUSE: &str = "aero-title-2026-11-07/scratch/B/final-vhs-t012.0.png";
const SKYWARS: &str = "featwave-2026-11-09/scratch/X9/bench/off/trace/shots/fight_start.png";

fn backdrop_key() -> UiTexture {
    UiTexture::Resource(ResourceLocation::new("mcrs", "test/rc_backdrop"))
}

struct Harness {
    rm: mc_assets::ResourceManager,
    text: MenuText,
    font: FontRenderer,
    title: TitleFont,
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

/// Draw `backdrop` (or a gradient) under `list` + `glass`, read the frame back, check it, and
/// save it when asked.
#[allow(clippy::too_many_arguments)]
fn render(
    gpu: &mut Gpu,
    h: &mut Harness,
    size: (u32, u32),
    backdrop: Option<&(Vec<u8>, u32, u32)>,
    list: DrawList,
    glass: GlassFrame,
    title_phase: Option<(AeroTitlePhase, DrawList, GlassFrame)>,
    state: &str,
    title_at: Rect,
) {
    let (w, hgt) = size;
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut ui = UiPhase::new(gpu, w, hgt, 1, false);
    let mut scene = DrawList::new();
    match backdrop {
        Some((rgba, bw, bh)) => {
            ui.renderer.upload_with_options(
                gpu,
                backdrop_key(),
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
                &backdrop_key(),
                Rect::new(0.0, 0.0, w as f32, hgt as f32),
                [0.0, 0.0, 1.0, 1.0],
                white(1.0),
            );
        }
        None => {
            scene.gradient_rect(0, 0, w as i32, hgt as i32, 0xFF6F_B8EA, 0xFF2C_6E31);
        }
    }
    ui.submit(scene);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(ui));
    if let Some((mut title, title_list, title_glass)) = title_phase {
        if let Some(pixels) = h.text.take_upload() {
            title.upload_text_atlas(gpu, pixels);
        }
        if let Some((pixels, tw, th)) = h.title.take_upload() {
            title.upload_title(gpu, pixels, tw, th);
        }
        if !h.text.is_segoe() {
            let _ = title
                .renderer()
                .ensure_list_textures(gpu, &h.rm, &title_list);
        }
        title.submit(title_list, title_glass);
        phases.insert(PhaseOrder(PhaseOrder::UI.0 + 40), Box::new(title));
    } else {
        let mut phase = ModMenuPhase::new(gpu);
        if let Some(pixels) = h.text.take_upload() {
            phase.upload_atlas(gpu, pixels, ATLAS_SIZE);
        }
        if !h.text.is_segoe() {
            let _ = phase.renderer.ensure_list_textures(gpu, &h.rm, &list);
        }
        phase.submit(list, glass);
        // The client registers the replay UI at UI + 85.
        phases.insert(PhaseOrder(PhaseOrder::UI.0 + 85), Box::new(phase));
    }
    let target = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, target.view(), &Camera::default(), 0.0);
    if let Some(error) = ready(scope.pop()) {
        panic!("wgpu validation error drawing {state}: {error}");
    }
    let px = gpu.read_pixels().expect("readback");
    // Near-white text where the title is.
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
    // The mod menu is reset before every new frame's text atlas upload.
    h.text.mark_dirty();
    h.title.mark_dirty();
    if let Some(dir) = std::env::var_os("MC_RC_PNG_DIR") {
        std::fs::create_dir_all(&dir).expect("png dir");
        let path = Path::new(&dir).join(format!("{state}-{w}x{hgt}.png"));
        image::save_buffer(&path, &px, w, hgt, image::ExtendedColorType::Rgba8)
            .expect("write the PNG");
        eprintln!("saved {}", path.display());
    }
}

fn entry(id: u64, title: &str, minutes_ago: u64, seconds: u64, mb: f32, clip: bool) -> ReplayEntry {
    ReplayEntry {
        id,
        title: title.to_owned(),
        server: "mc.example.net".to_owned(),
        started_unix_ms: NOW_MS - minutes_ago * 60_000,
        duration_us: seconds * 1_000_000,
        size_bytes: (mb * 1024.0 * 1024.0) as u64,
        clip,
        has_mic: id == 2,
        problem: None,
        bookmarks: 0,
        folder: String::new(),
        player: String::new(),
    }
}

/// 2026-11-12 21:41 EST.
const NOW_MS: u64 = 1_794_537_660_000;

fn library() -> LibraryView {
    LibraryView {
        recordings: vec![
            entry(1, "SkyWars \u{b7} Dwarf Fortress", 12, 761, 9.8, false),
            entry(2, "BedWars \u{b7} Lighthouse", 101, 1450, 17.2, false),
            entry(3, "Duels \u{b7} Classic", 60 * 24 + 40, 192, 2.4, false),
            entry(4, "SkyWars \u{b7} Elven", 60 * 24 * 3, 535, 7.1, false),
            // A recording keeps the default title (its server) until it is renamed.
            entry(5, "mc.example.net", 60 * 24 * 10, 65, 0.9, false),
            entry(6, "SkyWars \u{b7} Shire", 60 * 24 * 12, 702, 8.8, false),
        ],
        clips: vec![
            entry(10, "4 kills off spawn", 20, 30, 1.6, true),
            entry(11, "Bow shot across the map", 60 * 26, 30, 1.5, true),
            entry(12, "Clutch at the void", 60 * 24 * 4, 45, 2.3, true),
        ],
        // The client's 20 GB default (`replay` mod's storage limit, 10^9 bytes per GB).
        cap_bytes: 20_000_000_000,
        recording_on: true,
        clip_key_set: true,
        ..LibraryView::default()
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

#[test]
#[allow(clippy::too_many_lines)]
fn the_replay_screens_render_at_720p_and_1080p() {
    let Some(rm) = jar_assets() else {
        eprintln!("SKIP: jars/versions/1.8.9/1.8.9.jar not found (set MC_JARS_DIR)");
        return;
    };
    let greenhouse = picture(GREENHOUSE);
    let skywars = picture(SKYWARS);
    if greenhouse.is_none() || skywars.is_none() {
        eprintln!("note: a backdrop picture is missing; a gradient stands in");
    }
    for size in [(1920, 1080), (1280, 720)] {
        let Some(mut gpu) = gpu_for(size) else {
            return;
        };
        let mut h = Harness {
            font: FontRenderer::load(&rm, false, false).expect("vanilla font"),
            rm: jar_assets().expect("jar"),
            text: MenuText::load(),
            title: TitleFont::embedded(),
            t0: Instant::now(),
        };
        let sz = (size.0 as f32, size.1 as f32);
        // The replay screens' own scale (0.8 at 720p); the title screen uses its own.
        let s = mc_ui::screens::replay::style::scale(sz);
        let title_s = sz.1 / 1080.0;

        // ---- the title screen with its Replays button, hovered ----
        {
            let labels = AeroButton::ALL.map(|b| b.english().to_owned());
            let mut screen = AeroTitleScreen::new(h.t0);
            let mut list = DrawList::new();
            let mut glass = GlassFrame::default();
            for ms in [0u64, 400, 1200, 1500, 1800, 2000] {
                if ms == 1200 {
                    let r = screen.button_rect(AeroButton::Replays);
                    screen.pointer_moved(r.x + r.w * 0.6, r.y + r.h * 0.5);
                }
                list = DrawList::new();
                let now = h.at(ms);
                let mut env = AeroTitleEnv {
                    text: &mut h.text,
                    title_font: &mut h.title,
                    mc_font: &mut h.font,
                    labels: &labels,
                    version: "Minecraft 1.8.9",
                    copyright: "Copyright Mojang AB. Do not distribute!",
                    screen: sz,
                    now,
                    time_s: ms as f32 / 1000.0,
                };
                glass = screen.build(&mut env, &mut list);
            }
            // The buttons and the link pill in the bottom-right corner.
            assert_eq!(glass.rects.len(), AeroButton::COUNT + 1);
            let replays = screen.button_rect(AeroButton::Replays);
            let phase = AeroTitlePhase::new(&gpu);
            render(
                &mut gpu,
                &mut h,
                size,
                greenhouse.as_ref(),
                DrawList::new(),
                GlassFrame::default(),
                Some((phase, list, glass)),
                "title-replays-hover",
                Rect::new(
                    replays.x + 50.0 * title_s,
                    replays.y,
                    130.0 * title_s,
                    replays.h,
                ),
            );
        }

        // ---- the browser: recordings, a row hovered with its tooltip ----
        let est = chrono::FixedOffset::west_opt(5 * 3600).unwrap();
        let mut browser = ReplayBrowserScreen::new(h.t0);
        browser.set_library_in_zone(library(), NOW_MS, &est);
        let window = ReplayBrowserScreen::window_rect(sz);
        let mut list = DrawList::new();
        let _ = browser.build(&mut h.env(size, 100), &mut list);
        browser.key(key::DOWN, None, Modifiers::NONE, h.at(150));
        // Hover the second row's Watch button (right side of the row), long enough for its
        // tooltip.
        // (Under the toolbar and the day's 34 px header.)
        let row2_y = window.y + (78.0 + TOOLBAR_H + 12.0 + 34.0 + 86.0 + 40.0) * s;
        let watch_x =
            window.right() - (18.0 + 10.0 + 12.0 + 34.0 * 3.0 + 6.0 * 2.0 + 12.0 + 52.0) * s;
        browser.pointer_moved(watch_x, row2_y);
        let mut glass = GlassFrame::default();
        for ms in (200..=1400).step_by(50) {
            list = DrawList::new();
            glass = browser.build(&mut h.env(size, ms), &mut list);
        }
        render(
            &mut gpu,
            &mut h,
            size,
            greenhouse.as_ref(),
            list,
            glass,
            None,
            "browser",
            Rect::new(
                window.x + 60.0 * s,
                window.y + 10.0 * s,
                140.0 * s,
                40.0 * s,
            ),
        );

        // ---- the browser: clips, one armed for deletion, a message ----
        browser.key(key::TAB, None, Modifiers::NONE, h.at(1500));
        browser.key(key::DOWN, None, Modifiers::NONE, h.at(1500));
        browser.key(key::DELETE, None, Modifiers::NONE, h.at(1550));
        browser.show_file_message(
            "Saved \u{201c}4 kills off spawn.mp4\u{201d} (9.4 MB)",
            "C:/Users/Player/Videos/Aero Client/Clips/4 kills off spawn.mp4",
            h.at(1600),
        );
        browser.pointer_moved(window.x + window.w * 0.35, window.y + 200.0 * s);
        let (mut list, mut glass) = (DrawList::new(), GlassFrame::default());
        for ms in (1600..=2000).step_by(50) {
            list = DrawList::new();
            glass = browser.build(&mut h.env(size, ms), &mut list);
        }
        render(
            &mut gpu,
            &mut h,
            size,
            greenhouse.as_ref(),
            list,
            glass,
            None,
            "browser-clips",
            Rect::new(
                window.x + 60.0 * s,
                window.y + 10.0 * s,
                140.0 * s,
                40.0 * s,
            ),
        );

        // ---- the viewer: paused, a selection, hovering the timeline ----
        let info = ReplayInfo {
            title: "SkyWars \u{b7} Dwarf Fortress".to_owned(),
            server: "mc.example.net".to_owned(),
            started_unix_ms: NOW_MS,
            clip: false,
            has_mic: true,
            markers: vec![
                (190_000_000, "Kill".to_owned()),
                (465_000_000, "Kill".to_owned()),
            ],
            window_size: size,
        };
        let mut viewer = ReplayViewerOverlay::new(info.clone(), h.t0);
        let status = PlaybackStatus {
            duration_us: 761_000_000,
            time_us: 133_400_000,
            playing: false,
            speed: 1.0,
            ..PlaybackStatus::default()
        };
        viewer.set_status(status);
        viewer.set_selection(Some(118_000_000), Some(151_500_000));
        let bar = ReplayViewerOverlay::bar_rect(sz);
        let track_y = bar.y + 37.0 * s;
        let hover_x = bar.x + 26.0 * s + (bar.w - 52.0 * s) * 0.42;
        viewer.pointer_moved(hover_x, track_y, h.at(100));
        let (mut list, mut glass) = (DrawList::new(), GlassFrame::default());
        for ms in (100..=900).step_by(50) {
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
            None,
            "viewer-selection",
            Rect::new(20.0 * s, 18.0 * s, 260.0 * s, 40.0 * s),
        );

        // ---- the viewer: playing at 2x, the speed list open, a tooltip ----
        let mut viewer = ReplayViewerOverlay::new(info.clone(), h.t0);
        let playing = PlaybackStatus {
            playing: true,
            speed: 2.0,
            camera: mc_ui::screens::replay::CameraMode::Follow,
            ..status
        };
        viewer.set_status(playing);
        viewer.set_selection(Some(118_000_000), Some(151_500_000));
        let (mut list, mut glass) = (DrawList::new(), GlassFrame::default());
        let _ = viewer.build(&mut h.env(size, 50), &mut list);
        // The speed list open, the pointer resting on the Follow camera (its tooltip).
        viewer.open_speed_menu();
        let cam_x = bar.x + bar.w * 0.5;
        let cy = bar.y + 88.0 * s;
        viewer.pointer_moved(cam_x, cy, h.at(60));
        for ms in (100..=900).step_by(50) {
            list = DrawList::new();
            viewer.set_status(playing);
            glass = viewer.build(&mut h.env(size, ms), &mut list);
        }
        render(
            &mut gpu,
            &mut h,
            size,
            skywars.as_ref(),
            list,
            glass,
            None,
            "viewer-speed",
            Rect::new(20.0 * s, 18.0 * s, 260.0 * s, 40.0 * s),
        );

        // ---- the export window ----
        let mut viewer = ReplayViewerOverlay::new(info.clone(), h.t0);
        viewer.set_status(status);
        viewer.set_selection(Some(118_000_000), Some(151_500_000));
        viewer.set_export_defaults(ExportSettings {
            max_size: 2,
            ..ExportSettings::default()
        });
        let (mut list, mut glass) = (DrawList::new(), GlassFrame::default());
        let _ = viewer.build(&mut h.env(size, 50), &mut list);
        viewer.key(key::E, None, Modifiers::NONE, h.at(60));
        for ms in (100..=700).step_by(50) {
            list = DrawList::new();
            viewer.set_status(status);
            glass = viewer.build(&mut h.env(size, ms), &mut list);
        }
        let form_title = Rect::new(
            sz.0 * 0.5 - 300.0 * s,
            sz.1 * 0.5 - 360.0 * s,
            300.0 * s,
            80.0 * s,
        );
        render(
            &mut gpu,
            &mut h,
            size,
            skywars.as_ref(),
            list,
            glass,
            None,
            "export-form",
            form_title,
        );

        // ---- the export running, then done ----
        viewer.key(key::ENTER, None, Modifiers::NONE, h.at(800));
        assert!(viewer.export_running());
        let _ = viewer.drain_events().count();
        for (state, progress) in [
            (
                "export-progress",
                ExportProgress {
                    frames_done: 834,
                    frames_total: 2010,
                    elapsed_s: 9.8,
                    state: ExportState::Running,
                },
            ),
            (
                "export-done",
                ExportProgress {
                    frames_done: 2010,
                    frames_total: 2010,
                    elapsed_s: 23.4,
                    state: ExportState::Done {
                        path: "C:\\Users\\Player\\Videos\\Aero Client\\SkyWars.mp4".to_owned(),
                        bytes: 9_400_000,
                    },
                },
            ),
        ] {
            let (mut list, mut glass) = (DrawList::new(), GlassFrame::default());
            for ms in (900..=1700).step_by(50) {
                if let Some(dialog) = viewer.export_dialog() {
                    dialog.set_progress(progress.clone());
                }
                list = DrawList::new();
                glass = viewer.build(&mut h.env(size, ms), &mut list);
            }
            let panel_title = Rect::new(
                sz.0 * 0.5 - 320.0 * s,
                sz.1 - (176.0 + 36.0) * s,
                260.0 * s,
                60.0 * s,
            );
            render(
                &mut gpu,
                &mut h,
                size,
                skywars.as_ref(),
                list,
                glass,
                None,
                state,
                panel_title,
            );
        }
    }
}
