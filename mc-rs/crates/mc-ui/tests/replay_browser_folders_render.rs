//! Pictures of the replay browser's Folders view and sort orders, rendered for real (wgpu, the Aero glass phase, the vanilla font from the jar) over the title
//! screen's greenhouse, as the other replay pictures are: the folders of the recordings' root, a
//! day folder with its path, a deep path that does not fit, an empty folder, the Sort menu open,
//! the list by server and by account, and the clips' folders. Server and account names are made
//! up.
//!
//! Each frame runs under a wgpu validation scope and must put near-white text where the window's
//! title is. With `MC_BROWSER_PNG_DIR=<dir>` every frame is saved there (`<state>-<w>x<h>.png`).
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
use mc_ui::screens::replay::arrange::{Arrangement, ListView, SortOrder};
use mc_ui::screens::replay::browser::TOOLBAR_H;
use mc_ui::screens::replay::{key, LibraryView, ReplayBrowserScreen, ReplayEntry, ReplayEnv};
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
    let key = UiTexture::Resource(ResourceLocation::new("mcrs", "test/rfl_backdrop"));
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
    if let Some(dir) = std::env::var_os("MC_BROWSER_PNG_DIR") {
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

fn at(d: u32, h: u32, mi: u32) -> u64 {
    est()
        .with_ymd_and_hms(2026, 11, d, h, mi, 0)
        .unwrap()
        .timestamp_millis() as u64
}

#[allow(clippy::too_many_arguments)]
fn entry(
    id: u64,
    title: &str,
    folder: &str,
    started: u64,
    server: &str,
    player: &str,
    seconds: u64,
    mb: f32,
) -> ReplayEntry {
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
        folder: folder.to_owned(),
        player: player.to_owned(),
    }
}

const DEEP: &str = "Tournaments/Season three, the long one/Finals weekend/Round of sixteen";

/// A library shaped like a real one: date folders the recorder made, folders of the player's
/// own (one deep, one empty, one with an odd name), and a replay left in the root.
fn library() -> LibraryView {
    let sky = "sky.example.org";
    let pvp = "pvp.example.com";
    let mc = "mc.example.net";
    let day13 = "2026/11 November/13";
    let day12 = "2026/11 November/12";
    let mut recordings = vec![
        entry(
            1,
            "SkyWars \u{b7} Dwarf Fortress",
            day13,
            at(13, 23, 26),
            sky,
            "PlayerOne",
            761,
            9.8,
        ),
        entry(
            2,
            "Bridge Duels \u{b7} Urban",
            day13,
            at(13, 22, 5),
            pvp,
            "PlayerOne",
            402,
            4.1,
        ),
        entry(
            3,
            "BedWars \u{b7} Lighthouse",
            day13,
            at(13, 21, 30),
            mc,
            "AltAccount",
            1450,
            17.2,
        ),
        entry(
            4,
            "SkyWars \u{b7} Elven",
            day12,
            at(12, 20, 51),
            sky,
            "PlayerOne",
            535,
            7.1,
        ),
        entry(
            5,
            "Duels \u{b7} Classic",
            day12,
            at(12, 19, 2),
            pvp,
            "AltAccount",
            192,
            2.4,
        ),
        entry(
            6,
            "SkyWars \u{b7} Shire",
            "2026/10 October/30",
            at(9, 18, 0),
            sky,
            "PlayerOne",
            702,
            8.8,
        ),
        entry(
            7,
            "Final round",
            DEEP,
            at(11, 20, 15),
            pvp,
            "PlayerOne",
            1205,
            14.9,
        ),
        entry(
            8,
            "Quarter final",
            DEEP,
            at(11, 19, 40),
            pvp,
            "PlayerOne",
            980,
            11.3,
        ),
        entry(
            9,
            "Clutch with one heart",
            "Best games",
            at(10, 21, 0),
            sky,
            "PlayerOne",
            640,
            7.7,
        ),
        entry(
            10,
            "Comeback",
            "Best games",
            at(8, 22, 30),
            mc,
            "AltAccount",
            1820,
            21.0,
        ),
        entry(
            11,
            "Odd folder test",
            "copied\u{202e}gnp.exe",
            at(7, 12, 0),
            "",
            "",
            300,
            3.0,
        ),
        entry(
            12,
            "mc.example.net",
            "",
            at(6, 16, 45),
            mc,
            "PlayerOne",
            65,
            0.9,
        ),
    ];
    recordings[0].bookmarks = 3;
    recordings[8].bookmarks = 1;
    recordings[1].has_mic = true;
    let mut clips = vec![
        entry(
            20,
            "4 kills off spawn",
            day13,
            at(13, 23, 40),
            sky,
            "PlayerOne",
            30,
            1.6,
        ),
        entry(
            21,
            "Bow shot across the map",
            day12,
            at(12, 21, 0),
            sky,
            "PlayerOne",
            30,
            1.5,
        ),
        entry(
            22,
            "Clutch at the void",
            "",
            at(9, 20, 0),
            pvp,
            "PlayerOne",
            45,
            2.3,
        ),
    ];
    for clip in &mut clips {
        clip.clip = true;
    }
    let mut recording_folders: Vec<String> = [
        "2026",
        "2026/11 November",
        day13,
        day12,
        "2026/10 October",
        "2026/10 October/30",
        "Tournaments",
        "Tournaments/Season three, the long one",
        "Tournaments/Season three, the long one/Finals weekend",
        DEEP,
        "Best games",
        "To sort later",
        "copied\u{202e}gnp.exe",
    ]
    .iter()
    .map(|k| (*k).to_owned())
    .collect();
    recording_folders.sort();
    LibraryView {
        recordings,
        clips,
        recording_folders,
        clip_folders: ["2026", "2026/11 November", day13, day12]
            .iter()
            .map(|k| (*k).to_owned())
            .collect(),
        cap_bytes: 20_000_000_000,
        recording_on: true,
        clip_key_set: true,
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

/// Frames from `from` to `to` ms (the pop-in, the uploads, tooltips): the last frame's list.
fn frames(
    browser: &mut ReplayBrowserScreen,
    h: &mut Harness,
    size: (u32, u32),
    from: u64,
    to: u64,
) -> (DrawList, GlassFrame) {
    let (mut list, mut glass) = (DrawList::new(), GlassFrame::default());
    for ms in (from..=to).step_by(50) {
        list = DrawList::new();
        glass = browser.build(&mut h.env(size, ms), &mut list);
    }
    (list, glass)
}

#[test]
#[allow(clippy::too_many_lines)]
fn the_folders_view_and_the_sort_menu_render() {
    let Some(rm) = jar_assets() else {
        eprintln!("SKIP: no vanilla jar (set MC_JARS_DIR)");
        return;
    };
    let greenhouse = picture(GREENHOUSE);
    if greenhouse.is_none() {
        eprintln!("note: the greenhouse picture is missing; a gradient stands in");
    }
    let now_ms = at(14, 1, 30);
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
        let row_y = |n: f32| window.y + (78.0 + TOOLBAR_H + 12.0 + 40.0 + 86.0 * n) * s;
        let none = Modifiers::NONE;
        let mut shot = |browser: &mut ReplayBrowserScreen,
                        h: &mut Harness,
                        from: u64,
                        to: u64,
                        state: &str| {
            let (list, glass) = frames(browser, h, size, from, to);
            render(
                &mut gpu,
                h,
                size,
                greenhouse.as_ref(),
                list,
                glass,
                state,
                title_at,
            );
        };

        // ---- the recordings' root in the Folders view, the second folder hovered ----
        let mut browser = ReplayBrowserScreen::new(h.t0);
        browser.set_arrangement(Arrangement {
            sort: SortOrder::Newest,
            view: ListView::Folders,
        });
        browser.set_library_in_zone(library(), now_ms, &est());
        browser.pointer_moved(window.x + window.w * 0.45, row_y(1.0));
        shot(&mut browser, &mut h, 100, 1300, "browser-folders-root");

        // ---- a day folder: the path, the day's replays ----
        browser.key(key::ENTER, None, none, h.at(1400));
        browser.key(key::ENTER, None, none, h.at(1410));
        browser.key(key::ENTER, None, none, h.at(1420));
        browser.pointer_moved(window.x + window.w * 0.45, window.bottom() - 10.0 * s);
        shot(&mut browser, &mut h, 1450, 2000, "browser-folders-day");

        // ---- a deep folder whose path does not fit: the tab, then the last parts ----
        browser.key(key::HOME, None, none, h.at(2100));
        for _ in 0..3 {
            browser.key(key::BACKSPACE, None, none, h.at(2110));
        }
        browser.key(key::HOME, None, none, h.at(2120));
        browser.key(key::DOWN, None, none, h.at(2130));
        for _ in 0..4 {
            browser.key(key::ENTER, None, none, h.at(2140));
        }
        shot(&mut browser, &mut h, 2150, 2700, "browser-folders-deep");

        // ---- an empty folder ----
        for _ in 0..4 {
            browser.key(key::BACKSPACE, None, none, h.at(2800));
        }
        browser.key(key::END, None, none, h.at(2810));
        browser.key(key::UP, None, none, h.at(2820));
        browser.key(key::ENTER, None, none, h.at(2830));
        shot(&mut browser, &mut h, 2850, 3400, "browser-folder-empty");

        // ---- the List view with the Sort menu open (the keyboard on Server) ----
        browser.key(key::F, Some('f'), none, h.at(3500));
        browser.key(key::O, Some('o'), none, h.at(3510));
        for _ in 0..3 {
            browser.key(key::DOWN, None, none, h.at(3520));
        }
        shot(&mut browser, &mut h, 3550, 4100, "browser-sort-menu");

        // ---- sorted by server, under a header per server ----
        browser.key(key::ENTER, None, none, h.at(4200));
        shot(&mut browser, &mut h, 4250, 4800, "browser-by-server");

        // ---- by account ----
        browser.set_arrangement(Arrangement {
            sort: SortOrder::Account,
            view: ListView::List,
        });
        shot(&mut browser, &mut h, 4850, 5400, "browser-by-account");

        // ---- largest first: no headers, each row with its date ----
        browser.set_arrangement(Arrangement {
            sort: SortOrder::Largest,
            view: ListView::List,
        });
        shot(&mut browser, &mut h, 5450, 6000, "browser-largest");

        // ---- the clips in the Folders view, with a tooltip on the Up button ----
        browser.set_arrangement(Arrangement {
            sort: SortOrder::Newest,
            view: ListView::Folders,
        });
        browser.key(key::TAB, None, none, h.at(6100));
        browser.key(key::ENTER, None, none, h.at(6110));
        let up_x = window.x + (18.0 + 236.0 + 14.0 + 17.0) * s;
        let up_y = window.y + (78.0 + 8.0 + (TOOLBAR_H - 4.0) * 0.5) * s;
        browser.pointer_moved(up_x, up_y);
        shot(&mut browser, &mut h, 6250, 7200, "browser-clips-folders");
    }
}
