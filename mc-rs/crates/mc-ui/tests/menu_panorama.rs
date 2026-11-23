//! The main-menu backdrop through the live pipeline (`MainMenuScreen::draw_panorama_seed`, then
//! `UiPhase::submit_with_panorama`: the sharp seed pass, the `rotateAndBlurSkybox` passes and the
//! final crop), read back from a real GPU and compared with a real-OpenGL run of 1.8.9's
//! `GuiMainMenu.renderSkybox`.
//!
//! Skips (printing `SKIP:`) without a wgpu adapter or the vanilla jar. Set `MC_JARS_DIR` to point
//! at a `jars/` directory, and `MC_RS_PANORAMA_DUMP` to a directory to also write the frames as
//! PNGs.
//!
//! # Where the reference numbers come from
//!
//! `javaref/PanoramaRef.java` transcribes `drawPanorama`, `rotateAndBlurSkybox` and
//! `renderSkybox` (`GuiMainMenu.java:374-526`) call for call onto LWJGL 2.9.4, the 1.8.9 client's
//! own GL binding, and draws the jar's six panorama faces into an offscreen Pbuffer;
//! `javaref/gen_panorama_golden.ps1` runs it and prints the numbers below. Each is the mean colour
//! of one cell of a coarse grid. The blur leaves the cells smooth, so the means are stable against
//! the small differences between that GL run and this renderer (whose seed is a subdivided CPU
//! projection with affine texturing), while a sideways, mirrored or wrongly panned backdrop moves
//! them by tens of levels: before the 2026-11-06 fix this renderer drew the backdrop sideways and
//! was off by 33 to 145 levels in the first frame below.

use std::path::{Path, PathBuf};

use mc_assets::{ResourceManager, ZipResourcePack};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::menu::MainMenuScreen;
use mc_ui::widget::gpu::PanoramaFrameSubmission;
use mc_ui::widget::{DrawList, ScaledResolution, UiPhase};

/// Every reference window uses GUI scale 2.
const GUI_SCALE: i32 = 2;

/// The driver's `PANORAMA_STEPS`.
const PANORAMA_STEPS: u32 = 8;

/// Largest difference allowed between a cell mean here and in the GL reference, per channel.
const TOLERANCE: f64 = 10.0;

/// One final frame of the GL reference: the backdrop alone, before the menu's gradients.
struct Backdrop {
    timer: i32,
    partial_ticks: f32,
    width: u32,
    height: u32,
    /// Mean RGB of a 4x3 grid, row-major from the top-left.
    cells: [[u8; 3]; 12],
}

const BACKDROPS: [Backdrop; 3] = [
    // Upright: sky across the top, water and sand below.
    Backdrop {
        timer: 100,
        partial_ticks: 0.0,
        width: 854,
        height: 480,
        cells: [
            [131, 143, 169],
            [144, 156, 191],
            [155, 173, 214],
            [151, 169, 208],
            [51, 61, 132],
            [61, 64, 110],
            [93, 91, 89],
            [124, 120, 91],
            [59, 77, 104],
            [52, 64, 134],
            [88, 102, 49],
            [111, 110, 96],
        ],
    },
    // 113 degrees of pan later: another face, which pins the pan direction and speed.
    Backdrop {
        timer: 1234,
        partial_ticks: 0.37,
        width: 854,
        height: 480,
        cells: [
            [164, 181, 233],
            [148, 165, 212],
            [88, 104, 101],
            [49, 71, 32],
            [121, 117, 88],
            [95, 96, 60],
            [73, 95, 36],
            [74, 96, 37],
            [127, 124, 101],
            [83, 97, 63],
            [76, 98, 38],
            [75, 98, 38],
        ],
    },
    // A narrower window: the same view, cropped by `renderSkybox`'s aspect terms.
    Backdrop {
        timer: 100,
        partial_ticks: 0.0,
        width: 640,
        height: 480,
        cells: [
            [147, 164, 210],
            [152, 170, 231],
            [162, 182, 247],
            [159, 179, 237],
            [56, 66, 120],
            [63, 65, 110],
            [93, 95, 83],
            [123, 120, 91],
            [59, 76, 109],
            [64, 78, 102],
            [87, 103, 45],
            [96, 101, 101],
        ],
    },
];

/// The sharp 256x256 seed (`drawPanorama` alone) at timer 100: mean RGB of a 4x4 grid,
/// row-major from the top-left. Vanilla draws it turned 90 degrees clockwise (sky on the
/// right); the final quad turns it back.
const SEED_T100: [[u8; 3]; 16] = [
    [67, 83, 64],
    [55, 71, 113],
    [101, 114, 143],
    [155, 174, 248],
    [82, 99, 51],
    [51, 61, 142],
    [117, 123, 147],
    [145, 166, 255],
    [90, 109, 44],
    [89, 99, 70],
    [135, 145, 163],
    [149, 169, 253],
    [86, 99, 79],
    [120, 117, 95],
    [141, 151, 164],
    [157, 175, 247],
];

fn gpu(width: u32, height: u32) -> Option<Gpu> {
    match Gpu::headless(
        width,
        height,
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

fn assets() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(rm)
}

fn dump(name: &str, pixels: &[u8], width: u32, height: u32) {
    if let Some(dir) = std::env::var_os("MC_RS_PANORAMA_DUMP") {
        let path = PathBuf::from(dir).join(name);
        image::save_buffer(
            &path,
            pixels,
            width,
            height,
            image::ExtendedColorType::Rgba8,
        )
        .expect("write the panorama dump");
    }
}

/// Mean RGB of each cell of a `cols` x `rows` grid, row-major from the top-left, over the same
/// integer cell bounds `PanoramaRef.cells` uses.
fn cell_means(pixels: &[u8], width: u32, height: u32, cols: u32, rows: u32) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let (x0, x1) = (c * width / cols, (c + 1) * width / cols);
            let (y0, y1) = (r * height / rows, (r + 1) * height / rows);
            let mut sum = [0.0_f64; 3];
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = ((y * width + x) * 4) as usize;
                    for (ch, total) in sum.iter_mut().enumerate() {
                        *total += f64::from(pixels[i + ch]);
                    }
                }
            }
            let n = f64::from((x1 - x0) * (y1 - y0));
            out.push(sum.map(|total| total / n));
        }
    }
    out
}

/// The cell whose mean is furthest from the reference, and that largest per-channel difference.
fn worst_cell(got: &[[f64; 3]], want: &[[u8; 3]]) -> (usize, f64) {
    got.iter()
        .zip(want)
        .map(|(g, w)| {
            (0..3)
                .map(|ch| (g[ch] - f64::from(w[ch])).abs())
                .fold(0.0, f64::max)
        })
        .enumerate()
        .fold(
            (0, 0.0),
            |worst, (i, d)| if d > worst.1 { (i, d) } else { worst },
        )
}

/// Render the menu backdrop alone (no foreground widgets) exactly as the driver submits it.
fn render_backdrop(gpu: &mut Gpu, rm: &ResourceManager, frame: &Backdrop) -> Vec<u8> {
    let (w, h) = (frame.width, frame.height);
    let mut phase = UiPhase::new(gpu, w, h, GUI_SCALE, false);
    let ctx = ScreenCtx::new(
        ScaledResolution::new(w, h, GUI_SCALE, false),
        frame.partial_ticks,
    );
    let mut menu = MainMenuScreen::with_update_counter("", 0.5);
    menu.panorama.timer = frame.timer;
    let mut seed = DrawList::new();
    let plan = menu.draw_panorama_seed(&mut seed, &ctx, PANORAMA_STEPS);
    let missing = phase.submit_with_panorama(
        gpu,
        rm,
        DrawList::new(),
        PanoramaFrameSubmission::new(seed, plan.composite_uv),
    );
    assert!(missing.is_empty(), "panorama faces missing: {missing:?}");
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let target = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, target.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    dump(&format!("mcrs_{w}x{h}_t{}.png", frame.timer), &pixels, w, h);
    pixels
}

/// Render only the sharp first stage, `drawPanorama`, into a plain 256x256 target.
fn render_seed(gpu: &mut Gpu, rm: &ResourceManager, timer: i32) -> Vec<u8> {
    let mut phase = UiPhase::new(gpu, 256, 256, 1, false);
    // The seed does not depend on the window, only on the timer and partial ticks.
    let ctx = ScreenCtx::new(ScaledResolution::new(854, 480, GUI_SCALE, false), 0.0);
    let mut menu = MainMenuScreen::with_update_counter("", 0.5);
    menu.panorama.timer = timer;
    let mut seed = DrawList::new();
    let _ = menu.draw_panorama_seed(&mut seed, &ctx, PANORAMA_STEPS);
    let missing = phase.renderer.ensure_list_textures(gpu, rm, &seed);
    assert!(missing.is_empty(), "panorama faces missing: {missing:?}");
    phase.submit(seed);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let target = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, target.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    dump(&format!("mcrs_seed_t{timer}.png"), &pixels, 256, 256);
    pixels
}

/// The backdrop must be upright, pan the way vanilla pans and crop the way vanilla crops: before
/// the fix it came out sideways and mirrored, because the blur and crop read vanilla's
/// bottom-up `glCopyTexSubImage2D` texture coordinates top-down and the crop showed the 7th blur
/// pass instead of the 6th.
#[test]
fn the_menu_backdrop_matches_a_real_opengl_run_of_vanilla_render_skybox() {
    let Some(rm) = assets() else {
        eprintln!("SKIP: 1.8.9.jar not available");
        return;
    };
    for frame in &BACKDROPS {
        let Some(mut gpu) = gpu(frame.width, frame.height) else {
            return;
        };
        let pixels = render_backdrop(&mut gpu, &rm, frame);
        let cells = cell_means(&pixels, frame.width, frame.height, 4, 3);
        let (cell, delta) = worst_cell(&cells, &frame.cells);
        assert!(
            delta <= TOLERANCE,
            "timer {} + {}, {}x{}: cell {cell} is {:?}, GL has {:?} (off by {delta:.1})",
            frame.timer,
            frame.partial_ticks,
            frame.width,
            frame.height,
            cells[cell],
            frame.cells[cell]
        );
    }
}

/// The first stage on its own: vanilla's square 120 degree projection, `rotate(180, X)`,
/// `rotate(90, Z)`, the animated pitch and yaw and the per-face rotations.
#[test]
fn the_sharp_seed_matches_vanilla_draw_panorama() {
    let Some(rm) = assets() else {
        eprintln!("SKIP: 1.8.9.jar not available");
        return;
    };
    let Some(mut gpu) = gpu(256, 256) else { return };
    let pixels = render_seed(&mut gpu, &rm, 100);
    let cells = cell_means(&pixels, 256, 256, 4, 4);
    let (cell, delta) = worst_cell(&cells, &SEED_T100);
    assert!(
        delta <= TOLERANCE,
        "cell {cell} is {:?}, the GL reference has {:?} (off by {delta:.1})",
        cells[cell],
        SEED_T100[cell]
    );
}
