//! Headless GPU tests for the mod menu's glass phase (`screens::modmenu::glass`), the WGSL port
//! of the Aero web prototype's `glass.js` `FRAGMENT` shader.
//!
//! The scene behind the glass is drawn by a real `UiPhase` at `PhaseOrder::UI`: a uniform band
//! across the top and one-pixel black/white stripes below it. One-pixel stripes make the frost
//! measurable exactly: mip level 1 (a 2x2 box) of a one-pixel stripe pattern is uniform grey, so
//! wherever glass.js samples with its `uFrost` bias (1.5 levels) the stripes vanish into grey,
//! while the un-frosted `base` sample (bias 0.4) outside the glass keeps most of their contrast.
//!
//! Everything runs inside a wgpu validation error scope, so these tests are also the WGSL
//! validation of both shaders (glass and mip chain) on the real device.

use std::future::Future;

use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::screens::modmenu::glass::{GlassFrame, GlassRect, ModMenuPhase};
use mc_ui::widget::{DrawList, UiPhase};

const W: u32 = 320;
const H: u32 = 240;
/// The uniform band at the top of the scene, `0xAARRGGBB`.
const BAND: u32 = 0xFFDC_B43C;
const BAND_RGB: [f32; 3] = [220.0, 180.0, 60.0];
const BAND_BOTTOM: u32 = 40;
/// The glass pane: well inside the striped area, radius 16.
const PANE: [f32; 4] = [80.0, 80.0, 160.0, 120.0];

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

fn scene() -> DrawList {
    let mut list = DrawList::new();
    list.rect(0, 0, W as i32, BAND_BOTTOM as i32, BAND);
    // White at even x, the black clear colour at odd x.
    for x in (0..W as i32).step_by(2) {
        list.rect(x, BAND_BOTTOM as i32, x + 1, H as i32, 0xFFFF_FFFF);
    }
    list
}

fn glass(dim: f32) -> GlassFrame {
    GlassFrame {
        rects: vec![GlassRect {
            rect: PANE,
            radius: 16.0,
            alpha: 1.0,
        }],
        dim,
        strength: 1.0,
        frost: 1.5,
        px_per_dip: 1.0,
        time: 0.0,
    }
}

/// Render the scene, then the menu phase with `frame` (or the menu phase closed when `None`)
/// and `content` on top, all under a validation error scope. Returns the RGBA framebuffer.
fn render(gpu: &mut Gpu, frame: Option<GlassFrame>, content: DrawList) -> Vec<u8> {
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut ui = UiPhase::new(gpu, W, H, 1, false);
    ui.submit(scene());
    let mut menu = ModMenuPhase::new(gpu);
    if let Some(frame) = frame {
        menu.submit(content, frame);
    }
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(ui));
    phases.insert(PhaseOrder(PhaseOrder::UI.0 + 50), Box::new(menu));
    let target = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, target.view(), &Camera::default(), 0.0);
    if let Some(error) = ready(scope.pop()) {
        panic!("wgpu validation error in the menu phase: {error}");
    }
    gpu.read_pixels().expect("readback")
}

fn pixel(px: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2], px[i + 3]]
}

fn close(actual: u8, expected: f32, tolerance: f32) -> bool {
    (f32::from(actual) - expected).abs() <= tolerance
}

#[test]
fn closed_menu_leaves_the_frame_untouched() {
    let Some(mut gpu) = gpu() else { return };
    let px = render(&mut gpu, None, DrawList::new());
    assert_eq!(pixel(&px, 10, 10), [220, 180, 60, 255]);
    assert_eq!(
        pixel(&px, 160, 140),
        [255, 255, 255, 255],
        "even x is white"
    );
    assert_eq!(pixel(&px, 161, 140), [0, 0, 0, 255], "odd x is black");
}

#[test]
fn outside_the_glass_the_frame_is_dimmed_by_u_dim() {
    let Some(mut gpu) = gpu() else { return };
    let px = render(&mut gpu, Some(glass(0.35)), DrawList::new());
    // glass.js: `gl_FragColor = vec4(base * shade, 1.0)` with `shade = 1 - uDim`.
    let [r, g, b, a] = pixel(&px, 10, 10);
    for (got, want) in [r, g, b].into_iter().zip(BAND_RGB) {
        assert!(
            close(got, want * 0.65, 2.0),
            "{:?} vs {:?} x 0.65",
            [r, g, b],
            BAND_RGB
        );
    }
    assert_eq!(a, 255);
    // The un-frosted base sample keeps the stripes outside the glass (60 % mip 0 at bias 0.4).
    let (white, black) = (pixel(&px, 20, 140), pixel(&px, 21, 140));
    assert!(
        i32::from(white[0]) - i32::from(black[0]) > 60,
        "stripes outside the pane stay sharp: {white:?} {black:?}"
    );
}

#[test]
fn inside_the_glass_the_scene_is_frosted_and_still_dimmed() {
    let Some(mut gpu) = gpu() else { return };
    let px = render(&mut gpu, Some(glass(0.35)), DrawList::new());
    // Frost: at bias >= 1.5 the one-pixel stripes are mip-1/2 grey (127.5), so neighbouring
    // pixels in the middle of the pane agree, where the scene alternates 255 / 0.
    for x in 150..170 {
        let (a, b) = (pixel(&px, x, 140), pixel(&px, x + 1, 140));
        for channel in 0..3 {
            let step = (i32::from(a[channel]) - i32::from(b[channel])).abs();
            assert!(
                step <= 4,
                "x={x}: frosted glass still shows stripes: {a:?} {b:?}"
            );
        }
    }
    // Dimmed inside too, as glass.js does (`mix(base, refracted, glassMask) * shade`): grey
    // 127.5 x 0.65 = 83, plus at most the caustic's 0.018 (about 3 after the dim).
    let [r, g, b, _] = pixel(&px, 160, 140);
    assert!(
        close(r, 84.0, 5.0) && close(g, 84.0, 5.0) && close(b, 84.0, 5.0),
        "{r} {g} {b}"
    );
}

#[test]
fn without_the_dim_the_glass_keeps_the_scene_brightness() {
    let Some(mut gpu) = gpu() else { return };
    let px = render(&mut gpu, Some(glass(0.0)), DrawList::new());
    let outside = pixel(&px, 10, 10);
    for (got, want) in outside.into_iter().zip(BAND_RGB) {
        assert!(
            close(got, want, 1.0),
            "uDim 0 outside is the scene itself: {outside:?}"
        );
    }
    let [r, _, _, _] = pixel(&px, 160, 140);
    assert!(close(r, 128.0, 6.0), "undimmed frost is mid grey: {r}");
}

#[test]
fn the_glass_edge_is_tinted_and_bent_differently_from_the_middle() {
    let Some(mut gpu) = gpu() else { return };
    let px = render(&mut gpu, Some(glass(0.0)), DrawList::new());
    // Just inside the left edge (x = 82), `edge` is near 1: glass.js adds
    // `edge * vec3(.02, .03, .045)` and frosts harder (`bias = uFrost + edge * .8`). Blue gains
    // the most, so the edge is bluer than the middle.
    let edge = pixel(&px, 82, 140);
    let middle = pixel(&px, 160, 140);
    let blue_lift = |p: [u8; 4]| i32::from(p[2]) - i32::from(p[0]);
    assert!(
        blue_lift(edge) > blue_lift(middle),
        "edge {edge:?} should be bluer than middle {middle:?}"
    );
}

#[test]
fn menu_content_draws_over_the_glass() {
    let Some(mut gpu) = gpu() else { return };
    let mut content = DrawList::new();
    content.rect(150, 130, 170, 150, 0xFFFF_0000);
    let px = render(&mut gpu, Some(glass(0.35)), content);
    assert_eq!(
        pixel(&px, 160, 140),
        [255, 0, 0, 255],
        "content is not dimmed or refracted"
    );
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

/// The whole menu at 1280x720 over a colourful scene: seven windows, an expanded mod, real text.
/// Asserts the frame changed where the windows are; with `MC_MODMENU_PNG=<path>` it also
/// saves the frame for a human look.
#[test]
fn the_whole_menu_renders_over_a_scene() {
    use mc_mods::{DynamicRows, ModSettings};
    use mc_ui::font::FontRenderer;
    use mc_ui::modhud::ModHud;
    use mc_ui::screens::modmenu::text::{MenuText, ATLAS_SIZE};
    use mc_ui::screens::modmenu::{MenuEnv, MenuInput, ModMenuScreen};
    use std::time::{Duration, Instant};

    const MW: u32 = 1280;
    const MH: u32 = 720;
    let mut gpu = match Gpu::headless(
        MW,
        MH,
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
    let Some(rm) = jar_assets() else {
        eprintln!("SKIP: jars/versions/1.8.9/1.8.9.jar not found (set MC_JARS_DIR)");
        return;
    };
    let mut mc_font = FontRenderer::load(&rm, false, false).expect("vanilla font");
    let mut settings = ModSettings::new();
    for id in [
        "fps",
        "keystrokes",
        "toggle_sprint",
        "fullbright",
        "crosshair",
    ] {
        settings.set_enabled(id, true);
    }
    let rows = vec![DynamicRows::default(); settings.catalog().len()];
    let mut hud = ModHud::new();
    let mut events = Vec::new();
    let mut text = MenuText::load();
    let t0 = Instant::now();
    let at = |ms: u64| t0 + Duration::from_millis(ms);
    let mut menu = ModMenuScreen::new(settings.catalog(), t0);
    let mut build = |menu: &mut ModMenuScreen,
                     text: &mut MenuText,
                     settings: &ModSettings,
                     hud: &ModHud,
                     ms: u64| {
        let mut list = DrawList::new();
        let mut env = MenuEnv {
            text,
            mc_font: &mut mc_font,
            settings,
            rows: &rows,
            hud,
            screen: (MW as f32, MH as f32),
            gui_scale: 2.0,
            now: at(ms),
            time_s: 1.5,
            game: mc_ui::screens::modmenu::MenuGame::default(),
        };
        let frame = menu.build(&mut env, &mut list);
        (list, frame)
    };
    // Open Zoom as the user does (three windows): Controls in window 1,
    // then Zoom's row in window 2; window 3 shows its settings.
    let _ = build(&mut menu, &mut text, &settings, &hud, 1000);
    let zoom = settings.index_of("zoom").expect("zoom");
    let controls = mc_mods::menu_layout::CATEGORIES
        .iter()
        .position(|category| category.id == "controls")
        .and_then(|c| u8::try_from(c).ok())
        .expect("a Controls category");
    let nav = menu
        .nav_rect(mc_ui::screens::modmenu::View::Category(controls))
        .expect("Controls is in window 1");
    for step in 0..2 {
        let (x, y) = if step == 0 {
            nav.center()
        } else {
            let _ = build(&mut menu, &mut text, &settings, &hud, 1000);
            let row = menu.row_rect(zoom).expect("zoom row drawn");
            (row.x + row.w * 0.3, row.y + row.h * 0.5)
        };
        let mut input = MenuInput {
            settings: &mut settings,
            hud: &mut hud,
            events: &mut events,
            gui_screen: (640.0, 360.0),
            gui_scale: 2.0,
            now: at(1000),
            alt: false,
            shift: false,
        };
        menu.mouse_pressed(x, y, 0, &mut input);
        menu.mouse_released(x, y, 0, &mut input);
    }
    assert_eq!(menu.selected(), Some(zoom));
    let (list, frame) = build(&mut menu, &mut text, &settings, &hud, 1100);
    assert_eq!(
        frame.rects.len(),
        4,
        "three windows and the taskbar's Edit HUD pane"
    );
    let first = frame.rects[0].rect;

    // A colourful scene: sky gradient bands, "terrain", and bright posts behind the glass.
    let mut scene = DrawList::new();
    for band in 0..12_i32 {
        let b = band as u32;
        let shade = 0xFF00_0000 | ((40 + b * 14) << 16) | ((90 + b * 10) << 8) | 230;
        scene.rect(0, band * 40, MW as i32, band * 40 + 40, shade);
    }
    scene.rect(0, 480, MW as i32, MH as i32, 0xFF3C_7A2E);
    for post in 0..16 {
        let x = post * 80 + 20;
        scene.rect(x, 300, x + 12, 600, 0xFFFF_E080);
    }
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut ui = UiPhase::new(&gpu, MW, MH, 1, false);
    ui.submit(scene);
    let mut phase = ModMenuPhase::new(&gpu);
    if let Some(pixels) = text.take_upload() {
        phase.upload_atlas(&gpu, pixels, ATLAS_SIZE);
    }
    if !text.is_segoe() {
        let _ = phase.renderer.ensure_list_textures(&gpu, &rm, &list);
    }
    phase.submit(list, frame);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(ui));
    phases.insert(PhaseOrder(PhaseOrder::UI.0 + 50), Box::new(phase));
    let target = gpu.acquire().expect("acquire offscreen frame");
    phases.render(&gpu, target.view(), &Camera::default(), 0.0);
    if let Some(error) = ready(scope.pop()) {
        panic!("wgpu validation error drawing the menu: {error}");
    }
    let px = gpu.read_pixels().expect("readback");
    if let Ok(path) = std::env::var("MC_MODMENU_PNG") {
        image::save_buffer(&path, &px, MW, MH, image::ExtendedColorType::Rgba8)
            .expect("write the PNG");
        eprintln!("saved {path}");
    }
    // Far outside every window (the top-left corner) the scene is only dimmed.
    let corner = &px[0..4];
    assert!(
        close(corner[2], 230.0 * 0.65, 3.0),
        "dimmed sky: {corner:?}"
    );
    // Inside the first window's title bar something was drawn: the text is near white.
    let (x0, y0) = (first[0] as u32 + 10, first[1] as u32 + 10);
    let mut brightest = 0u8;
    for y in y0..y0 + 40 {
        for x in x0..x0 + 120 {
            let i = ((y * MW + x) * 4) as usize;
            brightest = brightest.max(px[i].min(px[i + 1]).min(px[i + 2]));
        }
    }
    assert!(
        brightest > 200,
        "the window title should be drawn in near-white: {brightest}"
    );
}
