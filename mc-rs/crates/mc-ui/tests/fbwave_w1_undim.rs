//! "Having the Color grading tab open should undim the
//! background so the person can see the raw color they are configuring."
//!
//! The client's phase order on the real GPU: a uniform "world" (a `UiPhase` early in the list,
//! at `PhaseOrder::TERRAIN_SOLID`), the real colour grade (`MotionGradePhase` at
//! `PhaseOrder::POST`, where `install_m7_postfx` registers it) and the menu glass
//! (`ModMenuPhase` at `PhaseOrder::UI + 50`, where `install_mod_phases` registers it), driven by
//! the `GlassFrame` a real `ModMenuScreen` builds after a right click opened a mod's settings.
//!
//! Outside the glass panes the frame must be:
//! * with Color Grading's settings open: the graded colour itself (`CdlParams::apply`);
//! * with another mod's settings open: that graded colour dimmed by the menu's 0.35.
//!
//! So the glass samples the frame *after* the grade, and the undim shows the grade as it is.
//! Everything runs under a wgpu validation error scope.

use std::future::Future;
use std::time::{Duration, Instant};

use mc_mods::{DynamicRows, ModSettings};
use mc_render::post::motion_grade::{CdlParams, MotionGradePhase};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::{FontData, FontRenderer, FontTexture};
use mc_ui::modhud::ModHud;
use mc_ui::screens::modmenu::glass::{GlassFrame, ModMenuPhase};
use mc_ui::screens::modmenu::text::MenuText;
use mc_ui::screens::modmenu::{MenuEnv, MenuInput, ModMenuScreen};
use mc_ui::widget::{DrawList, UiPhase};

const W: u32 = 1280;
const H: u32 = 720;
/// The "world": one uniform colour, `0xAARRGGBB`.
const WORLD: u32 = 0xFFDC_B43C;
const WORLD_RGB: [f32; 3] = [220.0, 180.0, 60.0];
/// A pixel outside every glass pane: at 1280x720 the windows start about 125 px down.
const PROBE: (u32, u32) = (12, 12);
/// glass.js `uDim`, the menu's dim.
const MENU_DIM: f32 = 0.35;

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
        Ok(gpu) => Some(gpu),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            None
        }
        Err(error) => panic!("headless context creation failed: {error}"),
    }
}

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

/// A deterministic font (every glyph 5 px wide), so the menu lays out without the vanilla jar.
fn font() -> FontRenderer {
    let mut sheet = image::RgbaImage::new(128, 128);
    for cell in 0..256_u32 {
        let (cx, cy) = ((cell % 16) * 8, (cell / 16) * 8);
        for column in 0..4 {
            sheet.put_pixel(cx + column, cy + 2, image::Rgba([255, 255, 255, 255]));
        }
    }
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(sheet)
        .write_to(&mut png, image::ImageFormat::Png)
        .expect("encode the font sheet");
    let data = FontData::from_parts(&png.into_inner(), &[0_u8; 0x1_0000]).expect("font data");
    FontRenderer::new_with_texture(data, FontTexture::Ascii, false, false)
}

/// A grade that moves every channel a clearly different way.
fn grade() -> CdlParams {
    CdlParams {
        slope: [0.8, 1.1, 1.3],
        offset: [0.02, -0.03, 0.0],
        power: [1.0, 1.2, 0.9],
        saturation: 1.1,
    }
}

/// The world colour after [`grade`], from the CPU reference, in 0..=255.
fn graded_world() -> [f32; 3] {
    grade()
        .apply(WORLD_RGB.map(|c| c / 255.0))
        .map(|c| c * 255.0)
}

struct Menu {
    screen: ModMenuScreen,
    settings: ModSettings,
    rows: Vec<DynamicRows>,
    hud: ModHud,
    text: MenuText,
    font: FontRenderer,
    t0: Instant,
}

impl Menu {
    fn new() -> Menu {
        let settings = ModSettings::new();
        let t0 = Instant::now();
        Menu {
            screen: ModMenuScreen::new(settings.catalog(), t0),
            rows: vec![DynamicRows::default(); settings.catalog().len()],
            settings,
            hud: ModHud::new(),
            text: MenuText::minecraft_only(),
            font: font(),
            t0,
        }
    }

    fn at(&self, ms: u64) -> Instant {
        self.t0 + Duration::from_millis(ms)
    }

    fn build(&mut self, ms: u64) -> GlassFrame {
        let now = self.at(ms);
        let mut list = DrawList::new();
        let mut env = MenuEnv {
            text: &mut self.text,
            mc_font: &mut self.font,
            settings: &self.settings,
            rows: &self.rows,
            hud: &self.hud,
            screen: (W as f32, H as f32),
            gui_scale: 2.0,
            now,
            time_s: 1.5,
            game: mc_ui::screens::modmenu::MenuGame::default(),
        };
        self.screen.build(&mut env, &mut list)
    }

    /// A left click at `(x, y)`.
    fn click(&mut self, (x, y): (f32, f32), ms: u64) {
        let now = self.at(ms);
        let mut events = Vec::new();
        let mut input = MenuInput {
            settings: &mut self.settings,
            hud: &mut self.hud,
            events: &mut events,
            gui_screen: (640.0, 360.0),
            gui_scale: 2.0,
            now,
            alt: false,
            shift: false,
        };
        self.screen.mouse_pressed(x, y, 0, &mut input);
        self.screen.mouse_released(x, y, 0, &mut input);
    }

    /// Open a mod's settings as the user does (the three-window menu):
    /// its category in window 1, then its row in window 2; window 3 shows the settings.
    fn open_settings(&mut self, mod_id: &str, ms: u64) {
        let m = self.settings.index_of(mod_id).expect("in the catalog");
        let home = mc_mods::menu_layout::CATEGORIES
            .iter()
            .position(|category| {
                category
                    .sections
                    .iter()
                    .any(|section| section.mods.contains(&mod_id))
            })
            .and_then(|c| u8::try_from(c).ok())
            .expect("placed in a category");
        let nav = self
            .screen
            .nav_rect(mc_ui::screens::modmenu::View::Category(home))
            .expect("its category is in window 1");
        self.click(nav.center(), ms);
        self.build(ms);
        let row = self.screen.row_rect(m).expect("its row is on screen");
        self.click((row.x + 40.0, row.y + row.h * 0.5), ms);
        assert_eq!(self.screen.selected(), Some(m), "{mod_id} open");
    }
}

/// The glass a real menu draws once `mod_id`'s settings have been open for 400 ms.
fn glass_with_open(mod_id: &str) -> GlassFrame {
    let mut menu = Menu::new();
    menu.build(1000);
    menu.open_settings(mod_id, 1000);
    let mut frame = menu.build(1016);
    for ms in (1032..=1400).step_by(16) {
        frame = menu.build(ms);
    }
    assert_eq!(
        frame.rects.len(),
        4,
        "three windows and the taskbar's Edit HUD pane"
    );
    frame
}

/// Render the world, the grade (when `Some`) and the menu glass (when `Some`) in the client's
/// order; return the probe pixel.
fn render(gpu: &mut Gpu, grade: Option<CdlParams>, glass: Option<GlassFrame>) -> [u8; 4] {
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut world = UiPhase::new(gpu, W, H, 1, false);
    let mut scene = DrawList::new();
    scene.rect(0, 0, W as i32, H as i32, WORLD);
    world.submit(scene);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::TERRAIN_SOLID, Box::new(world));
    if let Some(grade) = grade {
        let mut post = MotionGradePhase::new(gpu);
        post.set_grade(grade);
        post.set_in_world(true);
        assert!(post.is_active(), "the grade runs");
        phases.insert(PhaseOrder::POST, Box::new(post));
    }
    if let Some(glass) = glass {
        let mut menu = ModMenuPhase::new(gpu);
        menu.submit(DrawList::new(), glass);
        phases.insert(PhaseOrder(PhaseOrder::UI.0 + 50), Box::new(menu));
    }
    let target = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, target.view(), &Camera::default(), 0.0);
    if let Some(error) = ready(scope.pop()) {
        panic!("wgpu validation error: {error}");
    }
    let px = gpu.read_pixels().expect("readback");
    let i = ((PROBE.1 * W + PROBE.0) * 4) as usize;
    [px[i], px[i + 1], px[i + 2], px[i + 3]]
}

fn assert_rgb(got: [u8; 4], want: [f32; 3], tolerance: f32, what: &str) {
    for (channel, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (f32::from(*g) - w).abs() <= tolerance,
            "{what}: channel {channel} is {g}, want {w:.1} (pixel {got:?}, want {want:?})"
        );
    }
}

/// The vanilla jar's resources (the Minecraft font), or `None` (skip) without the jar.
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

/// The whole menu, text included, over a colourful graded scene, once with Color Grading's
/// settings open and once with Zoom's. Outside the windows the first is the graded scene and
/// the second that scene dimmed. With `MC_W1_PNG_DIR=<dir>` both frames are saved there for a
/// human look (`w1-color-grading-open.png`, `w1-zoom-open.png`).
#[test]
fn the_whole_menu_over_a_graded_scene_dims_only_without_color_grading_open() {
    use mc_ui::screens::modmenu::text::ATLAS_SIZE;

    let Some(mut gpu) = gpu() else { return };
    let Some(rm) = jar_assets() else {
        eprintln!("SKIP: jars/versions/1.8.9/1.8.9.jar not found (set MC_JARS_DIR)");
        return;
    };
    // Sky-like bands, "terrain" and bright posts, so the grade and the dim are easy to see.
    let mut scene = DrawList::new();
    for band in 0..12_u32 {
        let shade = 0xFF00_0000 | ((40 + band * 14) << 16) | ((90 + band * 10) << 8) | 230;
        scene.rect(0, band as i32 * 40, W as i32, band as i32 * 40 + 40, shade);
    }
    scene.rect(0, 480, W as i32, H as i32, 0xFF3C_7A2E);
    for post in 0..16 {
        let x = post * 80 + 20;
        scene.rect(x, 300, x + 12, 600, 0xFFFF_E080);
    }
    let mut corners = Vec::new();
    for (mod_id, file) in [
        ("color_grading", "w1-color-grading-open.png"),
        ("zoom", "w1-zoom-open.png"),
    ] {
        let mut menu = Menu::new();
        menu.text = MenuText::load();
        menu.font = FontRenderer::load(&rm, false, false).expect("vanilla font");
        menu.build(1000);
        menu.open_settings(mod_id, 1000);
        let mut list = DrawList::new();
        let mut frame = menu.build(1016);
        for ms in (1032..=1400).step_by(16) {
            list = DrawList::new();
            let now = menu.at(ms);
            let mut env = MenuEnv {
                text: &mut menu.text,
                mc_font: &mut menu.font,
                settings: &menu.settings,
                rows: &menu.rows,
                hud: &menu.hud,
                screen: (W as f32, H as f32),
                gui_scale: 2.0,
                now,
                time_s: 1.5,
                game: mc_ui::screens::modmenu::MenuGame::default(),
            };
            frame = menu.screen.build(&mut env, &mut list);
        }
        let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut world = UiPhase::new(&gpu, W, H, 1, false);
        world.submit(scene.clone());
        let mut post = MotionGradePhase::new(&gpu);
        post.set_grade(grade());
        post.set_in_world(true);
        let mut glass = ModMenuPhase::new(&gpu);
        if let Some(pixels) = menu.text.take_upload() {
            glass.upload_atlas(&gpu, pixels, ATLAS_SIZE);
        }
        if !menu.text.is_segoe() {
            let _ = glass.renderer.ensure_list_textures(&gpu, &rm, &list);
        }
        glass.submit(list, frame);
        let mut phases = PhaseList::new();
        phases.clear_color = wgpu::Color::BLACK;
        phases.insert(PhaseOrder::TERRAIN_SOLID, Box::new(world));
        phases.insert(PhaseOrder::POST, Box::new(post));
        phases.insert(PhaseOrder(PhaseOrder::UI.0 + 50), Box::new(glass));
        let target = gpu.acquire().expect("acquire offscreen frame");
        phases.render(&gpu, target.view(), &Camera::default(), 0.0);
        if let Some(error) = ready(scope.pop()) {
            panic!("wgpu validation error drawing the menu: {error}");
        }
        let px = gpu.read_pixels().expect("readback");
        if let Some(dir) = std::env::var_os("MC_W1_PNG_DIR") {
            let path = std::path::Path::new(&dir).join(file);
            image::save_buffer(&path, &px, W, H, image::ExtendedColorType::Rgba8)
                .expect("write the PNG");
            eprintln!("saved {}", path.display());
        }
        corners.push([px[0], px[1], px[2]]);
    }
    // The top-left corner is outside every window: the first sky band, graded, then dimmed.
    let band0 = grade()
        .apply([40.0 / 255.0, 90.0 / 255.0, 230.0 / 255.0])
        .map(|c| c * 255.0);
    eprintln!(
        "corner: Color Grading open {:?}, Zoom open {:?}",
        corners[0], corners[1]
    );
    for channel in 0..3 {
        let (open, zoom) = (
            f32::from(corners[0][channel]),
            f32::from(corners[1][channel]),
        );
        assert!(
            (open - band0[channel]).abs() <= 2.0,
            "{corners:?} vs {band0:?}"
        );
        assert!(
            (zoom - band0[channel] * (1.0 - MENU_DIM)).abs() <= 2.0,
            "{corners:?} vs {band0:?}"
        );
    }
}

#[test]
fn color_grading_open_shows_the_graded_world_undimmed() {
    let Some(mut gpu) = gpu() else { return };
    let graded = graded_world();
    // The grade really changes the colour, so the checks below are not vacuous.
    assert!(
        graded
            .iter()
            .zip(WORLD_RGB)
            .any(|(g, w)| (g - w).abs() > 20.0),
        "{graded:?}"
    );
    let plain = render(&mut gpu, None, None);
    assert_rgb(plain, WORLD_RGB, 1.0, "the world alone");
    let graded_alone = render(&mut gpu, Some(grade()), None);
    assert_rgb(
        graded_alone,
        graded,
        2.0,
        "the grade alone matches the CPU reference",
    );

    // Color Grading's settings open: the glass leaves the graded world exactly as it is.
    let undimmed = glass_with_open("color_grading");
    assert_eq!(undimmed.dim, 0.0);
    let px = render(&mut gpu, Some(grade()), Some(undimmed));
    eprintln!(
        "world {plain:?}, graded alone {graded_alone:?} (CPU {graded:.1?}), \
         Color Grading open {px:?}"
    );
    assert_rgb(
        px,
        graded,
        2.0,
        "Color Grading open: the graded world, undimmed",
    );
    assert_eq!(px[3], 255);

    // Another mod's settings open: the same graded world under the menu's dim.
    let dimmed = glass_with_open("zoom");
    assert!((dimmed.dim - MENU_DIM).abs() < 1e-4, "{}", dimmed.dim);
    let px = render(&mut gpu, Some(grade()), Some(dimmed));
    eprintln!("Zoom open {px:?}");
    assert_rgb(
        px,
        graded.map(|c| c * (1.0 - MENU_DIM)),
        2.0,
        "Zoom open: the graded world, dimmed",
    );
}
