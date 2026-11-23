//! Pictures of the replay browser grouped by day (short
//! recordings folded, then shown, bookmark counts on the rows) and of the viewer stepping
//! between bookmarks, rendered for real (wgpu, the Aero glass phase, the vanilla font from the
//! jar) over the backdrops the render test uses. The library is a night of Nov 13 shaped
//! like a real one (titles and lengths), with bookmarks added to two games.
//!
//! Each frame runs under a wgpu validation scope and must put near-white text where the window's
//! title is. With `MC_RA_PNG_DIR=<dir>` every frame is saved there (`<state>-<w>x<h>.png`).
//! Needs the vanilla jar (`MC_JARS_DIR`) and a GPU adapter; skips loudly without either.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use chrono::{FixedOffset, TimeZone};
use mc_assets::{ResourceLocation, TextureMetadata};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::FontRenderer;
use mc_ui::screens::modmenu::glass::{GlassFrame, ModMenuPhase};
use mc_ui::screens::modmenu::paint::{white, Painter, Rect};
use mc_ui::screens::modmenu::text::{MenuText, ATLAS_SIZE};
use mc_ui::screens::replay::{
    key, LibraryView, PlaybackStatus, ReplayBrowserScreen, ReplayEntry, ReplayEnv, ReplayInfo,
    ReplayViewerOverlay,
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

const GREENHOUSE: &str = "aero-title-2026-11-07/scratch/B/final-vhs-t012.0.png";
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
    let key = UiTexture::Resource(ResourceLocation::new("mcrs", "test/ra_backdrop"));
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
    if let Some(dir) = std::env::var_os("MC_RA_PNG_DIR") {
        std::fs::create_dir_all(&dir).expect("png dir");
        let path = Path::new(&dir).join(format!("{state}-{w}x{hgt}.png"));
        image::save_buffer(&path, &px, w, hgt, image::ExtendedColorType::Rgba8)
            .expect("write the PNG");
        eprintln!("saved {}", path.display());
    }
}

fn est() -> FixedOffset {
    FixedOffset::west_opt(5 * 3600).unwrap()
}

/// Nov `d`, `h:mi:s` EST, as Unix milliseconds.
fn at(d: u32, h: u32, mi: u32, s: u32) -> u64 {
    est()
        .with_ymd_and_hms(2026, 11, d, h, mi, s)
        .unwrap()
        .timestamp_millis() as u64
}

fn entry(id: u64, title: &str, server: &str, started: u64, seconds: u64, mb: f32) -> ReplayEntry {
    ReplayEntry {
        id,
        title: title.to_owned(),
        server: server.to_owned(),
        started_unix_ms: started,
        duration_us: seconds * 1_000_000,
        size_bytes: (mb * 1_000_000.0) as u64,
        clip: false,
        has_mic: false,
        problem: None,
        bookmarks: 0,
        folder: String::new(),
        player: String::new(),
    }
}

/// A night of replays shaped like a real library (titles, lengths and sizes), with a recording
/// made in another time zone among them and bookmarks on two games.
fn library() -> LibraryView {
    const SKY: &str = "sky.example.org";
    const EXAMPLE: &str = "EXAMPLE.NET";
    let mut recordings = vec![
        entry(
            1,
            "sky.example.org \u{b7} Nov 13, 11:26 PM",
            SKY,
            at(13, 23, 26, 56),
            1798,
            18.2,
        ),
        entry(
            2,
            "sky.example.org \u{b7} Nov 13, 11:09 PM",
            SKY,
            at(13, 23, 9, 9),
            1065,
            6.8,
        ),
        entry(
            3,
            "sky.example.org \u{b7} Nov 13, 10:12 PM",
            SKY,
            at(13, 22, 12, 51),
            3377,
            32.4,
        ),
        entry(
            4,
            "sky.example.org \u{b7} Nov 13, 9:49 PM",
            SKY,
            at(13, 21, 49, 3),
            1405,
            10.0,
        ),
        entry(
            5,
            "sky.example.org \u{b7} Nov 13, 9:41 PM",
            SKY,
            at(13, 21, 41, 39),
            443,
            3.9,
        ),
        entry(
            6,
            "sky.example.org \u{b7} Nov 13, 9:30 PM",
            SKY,
            at(13, 21, 30, 39),
            162,
            1.5,
        ),
        entry(
            7,
            "SkyWars \u{b7} EXAMPLE.NET \u{b7} Nov 13, 9:30 PM",
            EXAMPLE,
            at(13, 21, 30, 25),
            9,
            0.5,
        ),
        entry(
            8,
            "SkyWars \u{b7} EXAMPLE.NET \u{b7} Nov 13, 9:30 PM",
            EXAMPLE,
            at(13, 21, 30, 15),
            10,
            1.1,
        ),
        // The recording from another time zone: its default title shows this computer's time.
        entry(
            9,
            "sky.example.org \u{b7} Nov 13, 9:05 PM",
            SKY,
            1_794_621_923_186,
            208,
            2.1,
        ),
        entry(
            10,
            "Bed Wars \u{b7} EXAMPLE.NET \u{b7} Nov 13, 8:59 PM",
            EXAMPLE,
            at(13, 20, 59, 25),
            14,
            1.4,
        ),
        entry(
            11,
            "Bed Wars \u{b7} EXAMPLE.NET \u{b7} Nov 13, 8:56 PM",
            EXAMPLE,
            at(13, 20, 56, 16),
            189,
            2.7,
        ),
        entry(
            12,
            "Bed Wars \u{b7} EXAMPLE.NET \u{b7} Nov 13, 8:54 PM",
            EXAMPLE,
            at(13, 20, 54, 30),
            106,
            1.1,
        ),
        entry(
            13,
            "Bed Wars \u{b7} EXAMPLE.NET \u{b7} Nov 13, 8:51 PM",
            EXAMPLE,
            at(13, 20, 51, 45),
            165,
            6.3,
        ),
        entry(
            14,
            "EXAMPLE.NET \u{b7} Nov 13, 8:49 PM",
            EXAMPLE,
            at(13, 20, 49, 27),
            1,
            0.4,
        ),
        entry(
            15,
            "Bed Wars \u{b7} EXAMPLE.NET \u{b7} Nov 13, 8:48 PM",
            EXAMPLE,
            at(13, 20, 48, 46),
            38,
            2.5,
        ),
        entry(
            16,
            "sky.example.org \u{b7} Nov 13, 9:33 AM",
            SKY,
            at(13, 9, 33, 30),
            13,
            0.3,
        ),
        entry(
            17,
            "sky.example.org \u{b7} Nov 12, 8:17 PM",
            SKY,
            at(12, 20, 17, 11),
            247,
            2.4,
        ),
    ];
    recordings[2].bookmarks = 3;
    recordings[4].bookmarks = 1;
    LibraryView {
        recordings,
        clips: Vec::new(),
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
fn the_browser_by_day_and_the_viewers_bookmarks_render() {
    let Some(rm) = jar_assets() else {
        eprintln!("SKIP: no vanilla jar (set MC_JARS_DIR)");
        return;
    };
    let greenhouse = picture(GREENHOUSE);
    let skywars = picture(SKYWARS);
    // 01:30 on the 14th: the user's games were "Yesterday".
    let now_ms = at(14, 1, 30, 0);
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
        let window = ReplayBrowserScreen::window_rect(sz);
        let title_at = Rect::new(
            window.x + 60.0 * s,
            window.y + 10.0 * s,
            140.0 * s,
            40.0 * s,
        );

        // ---- the browser: by day, short recordings folded away ----
        let mut browser = ReplayBrowserScreen::new(h.t0);
        browser.set_library_in_zone(library(), now_ms, &est());
        browser.pointer_moved(window.x + window.w * 0.5, window.bottom() - 10.0 * s);
        let (mut list, mut glass) = (DrawList::new(), GlassFrame::default());
        for ms in (100..=1300).step_by(50) {
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
            "browser-by-day",
            title_at,
        );

        // ---- the same with the short ones shown, scrolled to the 12th ----
        browser.key(key::S, Some('s'), Modifiers::NONE, h.at(1400));
        browser.key(key::END, None, Modifiers::NONE, h.at(1400));
        let (mut list, mut glass) = (DrawList::new(), GlassFrame::default());
        for ms in (1450..=2600).step_by(50) {
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
            "browser-short-shown",
            title_at,
        );

        // ---- the viewer: bookmarks on the timeline, `]` just pressed, a diamond hovered ----
        let info = ReplayInfo {
            title: "sky.example.org \u{b7} Nov 13, 10:12 PM".to_owned(),
            server: "sky.example.org".to_owned(),
            started_unix_ms: at(13, 22, 12, 51),
            clip: false,
            has_mic: false,
            markers: vec![
                (412_000_000, "Bookmark".to_owned()),
                (1_390_500_000, "Bookmark".to_owned()),
                (2_051_000_000, "Clip".to_owned()),
                (2_930_000_000, "Bookmark".to_owned()),
            ],
            window_size: size,
        };
        let mut viewer = ReplayViewerOverlay::new(info, h.t0);
        let status = PlaybackStatus {
            duration_us: 3_377_000_000,
            time_us: 420_000_000,
            playing: false,
            speed: 1.0,
            ..PlaybackStatus::default()
        };
        viewer.set_status(status);
        let (mut list, mut glass) = (DrawList::new(), GlassFrame::default());
        let _ = viewer.build(&mut h.env(size, 50), &mut list);
        viewer.key(key::RBRACKET, None, Modifiers::NONE, h.at(60));
        let seek = viewer
            .drain_events()
            .find_map(|e| match e {
                mc_ui::screens::replay::ViewerEvent::Seek(t) => Some(t),
                _ => None,
            })
            .expect("] seeks");
        assert_eq!(seek, 1_390_500_000, "the next bookmark");
        let playing = PlaybackStatus {
            time_us: seek,
            ..status
        };
        // Hover the third diamond (the clip marker) for its tooltip.
        let bar = ReplayViewerOverlay::bar_rect(sz);
        let track_x0 = bar.x + 26.0 * s;
        let track_w = bar.w - 52.0 * s;
        let marker_x = track_x0 + track_w * (2_051_000_000.0 / 3_377_000_000.0);
        let marker_y = bar.y + 37.0 * s - 9.0 * s;
        for ms in (100..=1200).step_by(50) {
            if ms == 400 {
                viewer.pointer_moved(marker_x, marker_y, h.at(ms));
            }
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
            "viewer-bookmarks",
            Rect::new(20.0 * s, 18.0 * s, 260.0 * s, 40.0 * s),
        );
    }
}
