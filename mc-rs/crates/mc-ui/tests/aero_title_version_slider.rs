//! The title screen's version slider rendered for real (wgpu, Segoe UI, the vanilla font from the
//! jar) over the title screen's plain backdrop, at 1920x1080 and 1280x720: 1.8.9 picked (the
//! default), the pointer over 1.7.10, a press on it, the knob halfway through its slide, and
//! 1.7.10 picked. Each frame runs under a wgpu validation scope, and the knob must be the bright
//! fill in the picked version's half of the row.
//!
//! With `MC_VERSION_SLIDER_PNG_DIR=<dir>` every frame is saved there
//! (`version-<state>-<w>x<h>.png`, and a crop of the column, `...-column.png`) for a look at it.
//!
//! Needs the vanilla jar (`MC_JARS_DIR`) and a GPU adapter; skips loudly without either.

use std::future::Future;
use std::time::{Duration, Instant};

use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::FontRenderer;
use mc_ui::screens::aero_title::font::TitleFont;
use mc_ui::screens::aero_title::phase::AeroTitlePhase;
use mc_ui::screens::aero_title::{
    draw_stand_in_backdrop, AeroButton, AeroTitleEnv, AeroTitleScreen,
};
use mc_ui::screens::modmenu::glass::GlassFrame;
use mc_ui::screens::modmenu::paint::Rect;
use mc_ui::screens::modmenu::text::MenuText;
use mc_ui::widget::{DrawList, UiPhase};
use mc_version::GameVersion;

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

fn jar_assets() -> Option<mc_assets::ResourceManager> {
    let jars = match std::env::var("MC_JARS_DIR") {
        Ok(dir) => std::path::PathBuf::from(dir),
        Err(_) => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)?
            .join("MCP-919-main")
            .join("jars"),
    };
    let jar = jars.join("versions").join("1.8.9").join("1.8.9.jar");
    let pack = mc_assets::ZipResourcePack::open(&jar).ok()?;
    let mut rm = mc_assets::ResourceManager::new();
    rm.push_pack(Box::new(pack));
    Some(rm)
}

/// One title screen and what the client would hand it.
struct Title {
    text: MenuText,
    title: TitleFont,
    font: FontRenderer,
    labels: [String; AeroButton::COUNT],
    game_version: GameVersion,
    screen: AeroTitleScreen,
    size: (u32, u32),
    t0: Instant,
}

impl Title {
    fn new(rm: &mc_assets::ResourceManager, size: (u32, u32), game_version: GameVersion) -> Title {
        let t0 = Instant::now();
        Title {
            text: MenuText::load(),
            title: TitleFont::embedded(),
            font: FontRenderer::load(rm, false, false).expect("vanilla font"),
            labels: AeroButton::ALL.map(|b| b.english().to_owned()),
            game_version,
            screen: AeroTitleScreen::new(t0),
            size,
            t0,
        }
    }

    fn build(&mut self, ms: u64) -> (DrawList, GlassFrame) {
        let mut list = DrawList::new();
        let mut env = AeroTitleEnv {
            text: &mut self.text,
            title_font: &mut self.title,
            mc_font: &mut self.font,
            labels: &self.labels,
            version: self.game_version.title_line(),
            game_version: self.game_version,
            copyright: "Copyright Mojang AB. Do not distribute!",
            screen: (self.size.0 as f32, self.size.1 as f32),
            now: self.t0 + Duration::from_millis(ms),
            time_s: ms as f32 / 1000.0,
        };
        let glass = self.screen.build(&mut env, &mut list);
        (list, glass)
    }

    /// Frames every 16 ms from `from` to `to`, returning the last.
    fn settle(&mut self, from: u64, to: u64) -> (DrawList, GlassFrame) {
        let mut last = self.build(from);
        for ms in (from + 16..=to).step_by(16) {
            last = self.build(ms);
        }
        last
    }

    fn slot(&self, version: GameVersion) -> Rect {
        self.screen.version_slider.version_rect(version)
    }
}

fn render(
    gpu: &mut Gpu,
    rm: &mc_assets::ResourceManager,
    title: &mut Title,
    (list, glass): (DrawList, GlassFrame),
) -> Vec<u8> {
    let (w, h) = title.size;
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut ui = UiPhase::new(gpu, w, h, 1, false);
    let mut backdrop = DrawList::new();
    draw_stand_in_backdrop(&mut backdrop, w as i32, h as i32);
    ui.submit(backdrop);
    // A new phase has no atlases yet: upload both whole.
    let mut phase = AeroTitlePhase::new(gpu);
    title.text.mark_dirty();
    if let Some(pixels) = title.text.take_upload() {
        phase.upload_text_atlas(gpu, pixels);
    }
    title.title.mark_dirty();
    if let Some((pixels, tw, th)) = title.title.take_upload() {
        phase.upload_title(gpu, pixels, tw, th);
    }
    if !title.text.is_segoe() {
        let _ = phase.renderer().ensure_list_textures(gpu, rm, &list);
    }
    phase.submit(list, glass);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(ui));
    phases.insert(PhaseOrder(PhaseOrder::UI.0 + 40), Box::new(phase));
    let target = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, target.view(), &Camera::default(), 0.0);
    if let Some(error) = ready(scope.pop()) {
        panic!("wgpu validation error drawing the title screen: {error}");
    }
    gpu.read_pixels().expect("readback")
}

/// The darkest channel of the pixel at `(x, y)`: high only where the pixel is near white.
fn whiteness(px: &[u8], (w, _): (u32, u32), (x, y): (f32, f32)) -> u8 {
    let i = ((y as u32 * w + x as u32) * 4) as usize;
    px[i].min(px[i + 1]).min(px[i + 2])
}

/// A point inside the knob's fill when it rests on `slot`, clear of the label.
fn knob_point(slot: Rect) -> (f32, f32) {
    (slot.x + slot.w * 0.12, slot.y + slot.h * 0.3)
}

/// Check that the knob rests on `picked`, then save the frame (and a crop of the column).
fn check_and_save(px: &[u8], title: &Title, picked: GameVersion, state: &str) {
    let other = GameVersion::ALL
        .into_iter()
        .find(|version| *version != picked)
        .expect("two versions");
    let on = whiteness(px, title.size, knob_point(title.slot(picked)));
    let off = whiteness(px, title.size, knob_point(title.slot(other)));
    assert!(on > 180, "{state}: the knob is bright ({on})");
    assert!(
        on > off.saturating_add(60),
        "{state}: and only on {picked} ({on} vs {off})"
    );
    save(px, title, state);
}

fn save(px: &[u8], title: &Title, state: &str) {
    let Some(dir) = std::env::var_os("MC_VERSION_SLIDER_PNG_DIR") else {
        return;
    };
    let (w, h) = title.size;
    let dir = std::path::Path::new(&dir);
    std::fs::create_dir_all(dir).expect("the PNG folder");
    let image = image::RgbaImage::from_raw(w, h, px.to_vec()).expect("frame size");
    let path = dir.join(format!("version-{state}-{w}x{h}.png"));
    image.save(&path).expect("write the PNG");
    // The column (title, slider, the first buttons) at twice the size.
    let quit = title.screen.button_rect(AeroButton::Quit);
    let crop_w = (quit.right() + quit.x).min(w as f32) as u32;
    let crop_h = (quit.bottom() + 24.0).min(h as f32) as u32;
    let column = image::imageops::crop_imm(&image, 0, 0, crop_w, crop_h).to_image();
    let column = image::imageops::resize(
        &column,
        crop_w * 2,
        crop_h * 2,
        image::imageops::FilterType::Nearest,
    );
    let path = dir.join(format!("version-{state}-{w}x{h}-column.png"));
    column.save(&path).expect("write the PNG");
    eprintln!("saved {}", path.display());
}

#[test]
fn the_version_slider_renders_both_choices_hovered_pressed_and_sliding() {
    let Some(rm) = jar_assets() else {
        eprintln!("SKIP: jars/versions/1.8.9/1.8.9.jar not found (set MC_JARS_DIR)");
        return;
    };
    for size in [(1920, 1080), (1280, 720)] {
        let mut gpu = match Gpu::headless(
            size.0,
            size.1,
            GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(gpu) => gpu,
            Err(RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter available on this machine");
                return;
            }
            Err(error) => panic!("headless context creation failed: {error}"),
        };

        // 1.8.9, the default, the pointer away.
        let mut title = Title::new(&rm, size, GameVersion::V1_8_9);
        let frame = title.settle(0, 1200);
        let px = render(&mut gpu, &rm, &mut title, frame);
        check_and_save(&px, &title, GameVersion::V1_8_9, "1.8.9");

        // The knob dragged half-way: it covers half of each label, cut at its edges.
        let (from_x, y) = title.slot(GameVersion::V1_8_9).center();
        let half = (from_x - title.slot(GameVersion::V1_7_10).center().0) * 0.5;
        title.screen.pointer_moved(from_x, y);
        assert!(title.screen.version_slider.mouse_pressed(from_x, y, 0));
        title.screen.pointer_moved(from_x - half, y);
        let frame = title.settle(1216, 1300);
        let px = render(&mut gpu, &rm, &mut title, frame);
        save(&px, &title, "dragging");
        // Dragged back and released: still 1.8.9, nothing picked.
        title.screen.pointer_moved(from_x, y);
        assert_eq!(
            title.screen.version_slider.mouse_released(from_x, y, 0),
            None
        );

        // The pointer over 1.7.10.
        let (x, y) = title.slot(GameVersion::V1_7_10).center();
        title.screen.pointer_moved(x, y);
        let frame = title.settle(1316, 1600);
        let px = render(&mut gpu, &rm, &mut title, frame);
        check_and_save(&px, &title, GameVersion::V1_8_9, "1.8.9-hover");

        // Pressed on 1.7.10.
        assert!(title.screen.version_slider.mouse_pressed(x, y, 0));
        let frame = title.settle(1616, 1900);
        let px = render(&mut gpu, &rm, &mut title, frame);
        check_and_save(&px, &title, GameVersion::V1_8_9, "1.8.9-pressed");

        // Released: 1.7.10 is picked and the knob is on its way.
        let picked = title.screen.version_slider.mouse_released(x, y, 0);
        assert_eq!(picked, Some(GameVersion::V1_7_10));
        title.game_version = GameVersion::V1_7_10;
        let frame = title.settle(1916, 1964);
        let px = render(&mut gpu, &rm, &mut title, frame);
        save(&px, &title, "sliding");

        // Settled on 1.7.10, the pointer away.
        title.screen.pointer_moved(-1.0, -1.0);
        let frame = title.settle(1980, 2600);
        let px = render(&mut gpu, &rm, &mut title, frame);
        check_and_save(&px, &title, GameVersion::V1_7_10, "1.7.10");

        // A title screen opened with 1.7.10 already picked, the pointer over 1.8.9.
        let mut title = Title::new(&rm, size, GameVersion::V1_7_10);
        let _ = title.settle(0, 1200);
        let (x, y) = title.slot(GameVersion::V1_8_9).center();
        title.screen.pointer_moved(x, y);
        let frame = title.settle(1216, 1600);
        let px = render(&mut gpu, &rm, &mut title, frame);
        check_and_save(&px, &title, GameVersion::V1_7_10, "1.7.10-hover");
    }
}
