//! The render harness for the Aero title screen background.
//!
//! Ignored: these render real 1920x1080 frames on the GPU and write PNGs for a human to look at.
//!
//! ```text
//! cargo test -p mc-render --test menu_scene_render -- --ignored --nocapture render_pngs
//! MC_GPU_TIMING=1 cargo test -p mc-render --test menu_scene_render -- --ignored --nocapture gpu_cost
//! ```
//!
//! PNGs go to `MENU_SCENE_OUT`, default `<repo>/aero-title-2026-11-07/scratch/B/`. The vanilla
//! jar comes from `MC_JARS_DIR` (the `jars` folder) or the repository's own copy. Other knobs:
//! `MENU_SCENE_TAG` (file prefix), `MENU_SCENE_TIMES` (`6,31,0.9`), `MENU_SCENE_SIZE`
//! (`2560x1440`), `MENU_SCENE_LOOK` (`exposure=0.8,debug=1`, see `apply_look_env`).

use std::path::{Path, PathBuf};

use mc_render::menu_scene::{self, Look, MenuScene};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, RenderError};

/// The frame size: 1920x1080, or `MENU_SCENE_SIZE=2560x1440` for another one.
fn size() -> (u32, u32) {
    std::env::var("MENU_SCENE_SIZE")
        .ok()
        .and_then(|s| {
            let (w, h) = s.split_once('x')?;
            Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
        })
        .unwrap_or((1920, 1080))
}

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
    let dir = std::env::var_os("MENU_SCENE_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            root()
                .join("aero-title-2026-11-07")
                .join("scratch")
                .join("B")
        });
    std::fs::create_dir_all(&dir).expect("create the output folder");
    dir
}

fn gpu() -> Option<Gpu> {
    let (w, h) = size();
    match Gpu::headless(
        w,
        h,
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

/// `MENU_SCENE_LOOK="exposure=0.8,haze=0.004,debug=1,sun=2.6:2.4:2.1"` overrides [`Look`]
/// fields for tuning without a rebuild.
fn apply_look_env(scene: &mut MenuScene) {
    let Ok(spec) = std::env::var("MENU_SCENE_LOOK") else {
        return;
    };
    let mut look: Look = scene.look();
    for item in spec.split(',').filter(|s| !s.trim().is_empty()) {
        let (key, value) = item.split_once('=').expect("key=value");
        let num = |v: &str| v.trim().parse::<f32>().expect("a number");
        match key.trim() {
            "exposure" => look.exposure = num(value),
            "haze" => look.haze = num(value),
            "haze_mix" => look.haze_mix = num(value),
            "shafts" => look.shafts = num(value),
            "bloom" => look.bloom = num(value),
            "glow" => look.glow = num(value),
            "debug" => look.debug = num(value),
            "scrim" => look.scrim = num(value),
            "sun" => {
                let parts: Vec<f32> = value.split(':').map(num).collect();
                look.sun = [parts[0], parts[1], parts[2]];
            }
            other => panic!("unknown look key {other}"),
        }
    }
    eprintln!("look: {look:?}");
    scene.set_look(look);
}

fn frame(gpu: &mut Gpu, phases: &mut PhaseList, scene: &mut MenuScene, t: f64) -> Vec<u8> {
    let (w, h) = size();
    scene.update(t, w, h);
    let cam = Camera::default();
    let target = gpu.acquire().expect("offscreen frame");
    phases.render(gpu, target.view(), &cam, 0.0);
    gpu.read_pixels().expect("readback")
}

#[test]
#[ignore = "renders 1920x1080 PNGs for review; run by hand"]
fn render_pngs() {
    let Some(jar) = jar() else {
        eprintln!("skip: no 1.8.9 jar");
        return;
    };
    let Some(mut gpu) = gpu() else {
        eprintln!("skip: no GPU");
        return;
    };
    let started = std::time::Instant::now();
    let mut scene = MenuScene::new(&gpu, &jar).expect("menu scene");
    apply_look_env(&mut scene);
    eprintln!(
        "MenuScene::new took {:.1} ms; prepare: {:?}",
        started.elapsed().as_secs_f64() * 1000.0,
        menu_scene::prepare_stats(&jar).expect("stats")
    );
    let dir = out_dir();
    let tag = std::env::var("MENU_SCENE_TAG").unwrap_or_else(|_| "v".to_owned());
    let times: Vec<f64> = std::env::var("MENU_SCENE_TIMES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![6.0, 31.0, 0.9]);

    let (w, h) = size();
    // Background alone.
    let mut phases = PhaseList::new();
    phases.insert(menu_scene::BACKGROUND_ORDER, scene.background_phase());
    for &t in &times {
        let px = frame(&mut gpu, &mut phases, &mut scene, t);
        let path = dir.join(format!("{tag}-bg-t{t:05.1}.png"));
        image::save_buffer(&path, &px, w, h, image::ColorType::Rgba8).expect("png");
        eprintln!("wrote {}", path.display());
    }
    // With the VHS finish over it.
    phases.insert(menu_scene::FINISH_ORDER, scene.finish_phase());
    for &t in times.iter().take(1) {
        let px = frame(&mut gpu, &mut phases, &mut scene, t);
        let path = dir.join(format!("{tag}-vhs-t{t:05.1}.png"));
        image::save_buffer(&path, &px, w, h, image::ColorType::Rgba8).expect("png");
        eprintln!("wrote {}", path.display());
    }
}

/// Not ignored: one small frame through a 4x-MSAA game context with both phases, the way the
/// client inserts them, must render without a validation error and must not be a flat colour.
/// Skips without the jar or a GPU.
#[test]
fn a_small_msaa_frame_renders_with_both_phases() {
    let Some(jar) = jar() else {
        eprintln!("skip: no 1.8.9 jar");
        return;
    };
    let mut gpu = match Gpu::headless(
        320,
        180,
        GpuOptions {
            vsync: false,
            sample_count: 4,
            low_power: false,
        },
    ) {
        Ok(g) => g,
        Err(RenderError::NoAdapter) => {
            eprintln!("skip: no GPU");
            return;
        }
        Err(e) => panic!("headless context: {e}"),
    };
    gpu.device.on_uncaptured_error(std::sync::Arc::new(|e| {
        panic!("wgpu validation error: {e}")
    }));
    let mut scene = MenuScene::new(&gpu, &jar).expect("menu scene");
    let mut phases = PhaseList::new();
    phases.insert(menu_scene::BACKGROUND_ORDER, scene.background_phase());
    phases.insert(menu_scene::FINISH_ORDER, scene.finish_phase());
    assert_eq!(
        phases.names().collect::<Vec<_>>(),
        [
            menu_scene::BACKGROUND_PHASE_NAME,
            menu_scene::FINISH_PHASE_NAME
        ]
    );
    for t in [0.5, 7.0] {
        scene.update(t, 320, 180);
        let target = gpu.acquire().expect("offscreen frame");
        phases.render(&gpu, target.view(), &Camera::default(), 0.0);
    }
    let px = gpu.read_pixels().expect("readback");
    let first = [px[0], px[1], px[2]];
    let varied = px.chunks_exact(4).filter(|p| p[..3] != first).count();
    assert!(
        varied > px.len() / 8,
        "the frame is flat: {varied} differing pixels"
    );
    // Dropping the phases and the scene must release everything without a validation error.
    assert_eq!(phases.remove_named(menu_scene::BACKGROUND_PHASE_NAME), 1);
    assert_eq!(phases.remove_named(menu_scene::FINISH_PHASE_NAME), 1);
    drop(scene);
    let target = gpu.acquire().expect("offscreen frame");
    phases.render(&gpu, target.view(), &Camera::default(), 0.0);
}

#[test]
#[ignore = "GPU cost of the background + finish; run by hand with MC_GPU_TIMING=1"]
fn gpu_cost() {
    let Some(jar) = jar() else {
        eprintln!("skip: no 1.8.9 jar");
        return;
    };
    let Some(mut gpu) = gpu() else {
        eprintln!("skip: no GPU");
        return;
    };
    if !gpu.supports_gpu_timing() {
        eprintln!("skip: set MC_GPU_TIMING=1 (and the adapter must support timestamps)");
        return;
    }
    let mut scene = MenuScene::new(&gpu, &jar).expect("menu scene");
    let mut phases = PhaseList::new();
    phases.insert(menu_scene::BACKGROUND_ORDER, scene.background_phase());
    phases.insert(menu_scene::FINISH_ORDER, scene.finish_phase());
    let cam = Camera::default();
    let frames = 600;
    let (w, h) = size();
    for i in 0..frames {
        scene.update(5.0 + f64::from(i) / 360.0, w, h);
        let target = gpu.acquire().expect("offscreen frame");
        phases.render(&gpu, target.view(), &cam, 0.0);
        if i % 60 == 59 {
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        }
    }
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    // One more frame so the last readbacks are harvested.
    let target = gpu.acquire().expect("offscreen frame");
    phases.render(&gpu, target.view(), &cam, 0.0);
    eprintln!("{}", phases.gpu_timings().format_table());
}
