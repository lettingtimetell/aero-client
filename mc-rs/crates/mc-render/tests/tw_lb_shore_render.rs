//! The harness for the neon shore loading screen.
//!
//! The rule test runs by default (it skips without a GPU): it reads back top-down diagrams of a
//! wave's neon ribbon over time and checks the user's colour-push rule on the pixels. The others
//! are ignored: they render PNGs for a human to look at.
//!
//! ```text
//! cargo test -p mc-render --test tw_lb_shore_render
//! cargo test -p mc-render --test tw_lb_shore_render -- --ignored --nocapture render_pngs
//! cargo test -p mc-render --test tw_lb_shore_render -- --ignored --nocapture diagram_pngs
//! MC_GPU_TIMING=1 cargo test -p mc-render --test tw_lb_shore_render -- --ignored --nocapture gpu_cost
//! ```
//!
//! PNGs go to `SHORE_OUT`, default `<repo>/taskwave-2026-11-12/scratch/LB/png/`. Knobs:
//! `SHORE_TIMES` (`0,1.9,3`: seconds since the first frame), `SHORE_SIZE` (`1920x1080`),
//! `SHORE_SEED` (`0`), `SHORE_TAG` (file prefix), `SHORE_LOOK` (`exposure=2.4,grain=0.08,bloom=1`).

use std::path::{Path, PathBuf};

use mc_render::loading_scene::shore::{rest_time, Diagram, ShoreScene, ShoreTextures};
use mc_render::loading_scene::{LoadingCurtain, SceneFrame, CURTAIN_ORDER};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, RenderError, HEADLESS_FORMAT};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("repository root")
        .to_path_buf()
}

fn jar() -> Option<PathBuf> {
    let jars = std::env::var_os("MC_JARS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root().join("MCP-919-main").join("jars"));
    let jar = jars.join("versions").join("1.8.9").join("1.8.9.jar");
    jar.is_file().then_some(jar)
}

fn out_dir() -> PathBuf {
    let dir = std::env::var_os("SHORE_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            root()
                .join("taskwave-2026-11-12")
                .join("scratch")
                .join("LB")
                .join("png")
        });
    std::fs::create_dir_all(&dir).expect("create the output folder");
    dir
}

fn env_size(default: (u32, u32)) -> (u32, u32) {
    std::env::var("SHORE_SIZE")
        .ok()
        .and_then(|s| {
            let (w, h) = s.split_once('x')?;
            Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
        })
        .unwrap_or(default)
}

fn seed() -> u32 {
    std::env::var("SHORE_SEED")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

fn gpu(size: (u32, u32)) -> Option<Gpu> {
    match Gpu::headless(
        size.0,
        size.1,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(g) => {
            g.device.on_uncaptured_error(std::sync::Arc::new(|e| {
                panic!("wgpu validation error: {e}")
            }));
            Some(g)
        }
        Err(RenderError::NoAdapter) => None,
        Err(e) => panic!("headless context: {e}"),
    }
}

fn textures() -> ShoreTextures {
    match jar() {
        Some(jar) => {
            let t = ShoreTextures::from_jar(&jar);
            assert_eq!(
                t.vanilla_count(),
                3,
                "sand, gravel and clay come from the jar"
            );
            t
        }
        None => {
            eprintln!("note: no 1.8.9 jar, using the stand-in textures");
            ShoreTextures::fallback()
        }
    }
}

/// `SHORE_LOOK="exposure=2.4,grain=0.08,bloom=1"` overrides look fields without a rebuild.
fn apply_look_env(scene: &mut ShoreScene) {
    let Ok(spec) = std::env::var("SHORE_LOOK") else {
        return;
    };
    let mut look = scene.look();
    for item in spec.split(',').filter(|s| !s.trim().is_empty()) {
        let (key, value) = item.split_once('=').expect("key=value");
        let v: f32 = value.trim().parse().expect("a number");
        match key.trim() {
            "exposure" => look.exposure = v,
            "grain" => look.grain = v,
            "bloom" => look.bloom = v,
            "grain_hz" => look.grain_hz = v,
            "logo_shade" => look.logo_shade = v,
            other => panic!("unknown look key {other}"),
        }
    }
    eprintln!("look: {look:?}");
    scene.set_look(look);
}

/// Render one frame at `t` seconds since the first frame and read it back (RGBA8).
fn frame(gpu: &mut Gpu, scene: &mut ShoreScene, t: f64) -> Vec<u8> {
    let (w, h) = (gpu.config.width, gpu.config.height);
    scene.update(t);
    let target = gpu.acquire().expect("offscreen frame");
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("shore"),
        });
    scene.record_into(gpu, &mut enc, target.view(), (w, h), (HEADLESS_FORMAT, 1));
    gpu.queue.submit([enc.finish()]);
    drop(target);
    gpu.read_pixels().expect("readback")
}

fn save(path: &Path, pixels: &[u8], (w, h): (u32, u32)) {
    image::save_buffer(path, pixels, w, h, image::ExtendedColorType::Rgba8).expect("write PNG");
    eprintln!("wrote {}", path.display());
}

/// A grid of images, each scaled to `cell` (aspect kept, letterboxed in black).
fn contact_sheet(images: &[image::RgbaImage], cols: u32, cell: (u32, u32)) -> image::RgbaImage {
    let rows = (images.len() as u32).div_ceil(cols);
    let mut sheet =
        image::RgbaImage::from_pixel(cols * cell.0, rows * cell.1, image::Rgba([0, 0, 0, 255]));
    for (k, img) in images.iter().enumerate() {
        let s = (cell.0 as f32 / img.width() as f32).min(cell.1 as f32 / img.height() as f32);
        let (w, h) = (
            (img.width() as f32 * s) as u32,
            (img.height() as f32 * s) as u32,
        );
        let small = image::imageops::resize(img, w, h, image::imageops::FilterType::Triangle);
        let (cx, cy) = ((k as u32 % cols) * cell.0, (k as u32 / cols) * cell.1);
        image::imageops::overlay(
            &mut sheet,
            &small,
            i64::from(cx + (cell.0 - w) / 2),
            i64::from(cy + (cell.1 - h) / 2),
        );
    }
    sheet
}

/// The scene at the times in `SHORE_TIMES`, 1080p by default, plus a contact sheet of them after
/// the two reference photos (`<tag>-sheet.png`).
#[test]
#[ignore = "writes review PNGs; run explicitly"]
fn render_pngs() {
    let size = env_size((1920, 1080));
    let Some(mut gpu) = gpu(size) else {
        eprintln!("skip: no GPU");
        return;
    };
    let started = std::time::Instant::now();
    let mut scene = ShoreScene::with_textures(&gpu, &textures(), seed());
    eprintln!("scene built in {:?}", started.elapsed());
    apply_look_env(&mut scene);
    let times: Vec<f64> = std::env::var("SHORE_TIMES")
        .unwrap_or_else(|_| "0,0.9,1.5,1.9,2.3,2.8,3.6,5,8".into())
        .split(',')
        .map(|s| s.trim().parse().expect("a time"))
        .collect();
    let tag = std::env::var("SHORE_TAG").unwrap_or_else(|_| "shore".into());
    let out = out_dir();
    let refs = root().join("taskwave-2026-11-12").join("refs");
    let mut sheet: Vec<image::RgbaImage> = ["image2-foggy-shore.png", "image3-neon-trails.png"]
        .iter()
        .filter_map(|name| image::open(refs.join(name)).ok().map(|i| i.to_rgba8()))
        .collect();
    for t in times {
        let pixels = frame(&mut gpu, &mut scene, t);
        save(&out.join(format!("{tag}-t{t:05.2}.png")), &pixels, size);
        sheet.push(image::RgbaImage::from_raw(size.0, size.1, pixels).expect("a frame"));
    }
    let path = out.join(format!("{tag}-sheet.png"));
    contact_sheet(&sheet, 3, (640, 360))
        .save(&path)
        .expect("write the sheet");
    eprintln!("wrote {}", path.display());
}

/// The diagram window: the swash terrace and the inner surf, 48 blocks of shore.
fn window(wave: i32, test_channels: bool, clamp: bool) -> Diagram {
    Diagram {
        wave,
        test_channels,
        clamp,
        u_top: -14.0,
        u_span: 22.0,
        v_left: -26.0,
        v_span: 48.0,
    }
}

const DIAG: (u32, u32) = (720, 330);

/// What the rule check saw over one wave.
struct PushReport {
    /// Pushing-colour pixels strictly beyond the border the front had drawn by then.
    violations: usize,
    /// Pixels where a pushing colour covered the cyan front band's place (it took the colour).
    taken: usize,
    /// Pushing-colour pixels ahead of where the cyan front was at that moment (past it).
    past_front: usize,
    /// Frames rendered.
    frames: usize,
}

/// Render wave `wave`'s test channels every 50 ms over 11 seconds around its rest, tracking per
/// column the farthest row the cyan front band ever reached (the border it drew).
fn check_push(gpu: &mut Gpu, scene: &mut ShoreScene, wave: i32, clamp: bool) -> PushReport {
    let (w, h) = DIAG;
    scene.set_diagram(Some(window(wave, true, clamp)));
    // The rest time here is mid-window; columns vary by a couple of seconds either side.
    let rest = rest_time(wave, -2.0) - scene.start_clock();
    let mut border = vec![u32::MAX; w as usize];
    let mut report = PushReport {
        violations: 0,
        taken: 0,
        past_front: 0,
        frames: 0,
    };
    let steps = 220;
    for step in 0..steps {
        let t = rest - 5.5 + f64::from(step) * 0.05;
        let px = frame(gpu, scene, t.max(0.0));
        report.frames += 1;
        // This frame's cyan front band, per column: its topmost (most shoreward) row.
        let mut front_now = vec![u32::MAX; w as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                if px[i] > 127 {
                    let c = x as usize;
                    front_now[c] = front_now[c].min(y);
                    border[c] = border[c].min(y);
                }
            }
        }
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                if px[i + 1] <= 127 {
                    continue;
                }
                let c = x as usize;
                // One row of slack for the time sampling.
                if border[c] == u32::MAX || y + 1 < border[c] {
                    report.violations += 1;
                }
                if px[i] > 127 {
                    report.taken += 1;
                }
                if front_now[c] != u32::MAX && y + 1 < front_now[c] {
                    report.past_front += 1;
                }
            }
        }
    }
    scene.set_diagram(None);
    report
}

/// The user's rule: at the run-up the pushing colours go past the cyan front band (taking its
/// colour where they arrive) but never past the border it drew. Checked on read-back pixels for
/// three waves (three different pairs of pushing colours), with the clamp switched off as a
/// negative control.
#[test]
fn the_colour_push_never_passes_the_border_the_front_drew() {
    let Some(mut gpu) = gpu(DIAG) else {
        eprintln!("skip: no GPU");
        return;
    };
    let mut scene = ShoreScene::with_textures(&gpu, &ShoreTextures::fallback(), 0);
    let first = scene.focus_wave() + 1;
    for wave in [first, first + 1, first + 2] {
        let on = check_push(&mut gpu, &mut scene, wave, true);
        eprintln!(
            "wave {wave}: {} frames, {} violations, {} pixels taken from cyan, {} past the front",
            on.frames, on.violations, on.taken, on.past_front
        );
        assert_eq!(on.violations, 0, "wave {wave}: a colour passed the border");
        assert!(
            on.taken > 200,
            "wave {wave}: the push never took the cyan's place"
        );
        assert!(
            on.past_front > 200,
            "wave {wave}: nothing went past the draining front"
        );
        // Negative control: without the clamp the same frames do cross the border.
        let off = check_push(&mut gpu, &mut scene, wave, false);
        eprintln!(
            "wave {wave} without the clamp: {} violations",
            off.violations
        );
        assert!(
            off.violations > 200,
            "wave {wave}: the check cannot see a violation"
        );
    }
}

/// The band diagram over one rest, for the report: twelve top-down frames of the swash's top,
/// 30 pixels to a block both ways, from 2.4 s before the rest (at mid-window) to 2.6 s after,
/// in a 3 x 4 grid (`diagram-grid.png`, read left to right, top to bottom). The grey lines are the
/// border the cyan front draws (the run-up, top) and, when in view, the terrace edge.
#[test]
#[ignore = "writes review PNGs; run explicitly"]
fn diagram_pngs() {
    let size = (720_u32, 330_u32);
    let Some(mut gpu) = gpu(size) else {
        eprintln!("skip: no GPU");
        return;
    };
    let mut scene = ShoreScene::with_textures(&gpu, &ShoreTextures::fallback(), seed());
    let wave = scene.focus_wave() + 1;
    let rest = rest_time(wave, -2.0) - scene.start_clock();
    // The window: 24 blocks of shore by 11 blocks up and down the beach, the run-up near its top.
    scene.set_diagram(Some(Diagram {
        u_top: -11.6,
        u_span: 11.0,
        v_left: -14.0,
        v_span: 24.0,
        ..window(wave, false, true)
    }));
    let out = out_dir();
    let offsets = [
        -2.4, -1.2, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2, 1.6, 2.0, 2.6,
    ];
    let mut frames = Vec::new();
    for (k, dt) in offsets.iter().enumerate() {
        let px = frame(&mut gpu, &mut scene, rest + dt);
        save(
            &out.join(format!("diagram-{k:02}-rest{dt:+.1}s.png")),
            &px,
            size,
        );
        frames.push(image::RgbaImage::from_raw(size.0, size.1, px).expect("a frame"));
    }
    let grid = contact_sheet(&frames, 3, (size.0 + 8, size.1 + 8));
    let path = out.join("diagram-grid.png");
    grid.save(&path).expect("write the grid");
    eprintln!("wrote {}", path.display());
}

/// GPU time per frame at 1080p through the real curtain phase (the frame), with
/// `MC_GPU_TIMING=1`.
#[test]
#[ignore = "measures; run explicitly with MC_GPU_TIMING=1"]
fn gpu_cost() {
    let size = env_size((1920, 1080));
    let Some(mut gpu) = gpu(size) else {
        eprintln!("skip: no GPU");
        return;
    };
    if !gpu.supports_gpu_timing() {
        eprintln!("skip: set MC_GPU_TIMING=1 (and the adapter must support timestamps)");
        return;
    }
    let scene = ShoreScene::with_textures(&gpu, &textures(), seed());
    let mut curtain = LoadingCurtain::with_scene(&gpu, Box::new(scene), seed());
    let mut phases = PhaseList::new();
    phases.insert(CURTAIN_ORDER, curtain.phase());
    let cam = Camera::default();
    let frames = 360;
    for i in 0..frames {
        let t = f64::from(i) / 144.0;
        curtain.update(
            SceneFrame {
                time_s: t as f32,
                dt_s: 1.0 / 144.0,
                opacity: 1.0,
                ..SceneFrame::default()
            },
            1.0,
        );
        let target = gpu.acquire().expect("offscreen frame");
        phases.render(&gpu, target.view(), &cam, 0.0);
        if i % 60 == 59 {
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        }
    }
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let target = gpu.acquire().expect("offscreen frame");
    phases.render(&gpu, target.view(), &cam, 0.0);
    eprintln!("{}", phases.gpu_timings().format_table());
}
