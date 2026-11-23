//! The Aero title scene's volumetric clouds, rendered through
//! the real menu scene on a real GPU and read back.
//!
//! Needs the vanilla jar (the scene bakes vanilla); skips without it or a GPU.
//!
//! ```text
//! cargo test -p mc-render --test x11_title_clouds
//! MC_GPU_TIMING=1 cargo test -p mc-render --test x11_title_clouds -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::time::Duration;

use mc_render::menu_scene::{self, MenuScene};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, RenderError};

const READY: Duration = Duration::from_secs(60);

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
        Ok(gpu) => {
            gpu.device.on_uncaptured_error(std::sync::Arc::new(|e| {
                panic!("wgpu validation error: {e}")
            }));
            Some(gpu)
        }
        Err(RenderError::NoAdapter) => None,
        Err(e) => panic!("headless context: {e}"),
    }
}

/// `frames` frames of the background at scene time `t` (the clouds accumulate over frames),
/// read back.
fn render(gpu: &mut Gpu, scene: &mut MenuScene, t: f64, frames: usize) -> Vec<u8> {
    let mut phases = PhaseList::new();
    phases.insert(menu_scene::BACKGROUND_ORDER, scene.background_phase());
    let (w, h) = (gpu.config.width, gpu.config.height);
    for i in 0..frames {
        // A few milliseconds apart, as at a high frame rate.
        scene.update(t + i as f64 * 0.004, w, h);
        let target = gpu.acquire().expect("offscreen frame");
        phases.render(gpu, target.view(), &Camera::default(), 0.0);
    }
    gpu.read_pixels().expect("readback")
}

/// The sky seen through the greenhouse's roof changes when the volumetric clouds replace the
/// painted ones, and nothing else does (the greenhouse covers the rest).
#[test]
fn the_title_sky_gets_volumetric_clouds() {
    let Some(jar) = jar() else {
        eprintln!("skip: no 1.8.9 jar");
        return;
    };
    let Some(mut gpu) = gpu(480, 270) else {
        eprintln!("skip: no GPU");
        return;
    };
    let mut painted = MenuScene::new(&gpu, &jar).expect("menu scene");
    painted.set_volumetric_clouds(false);
    let before = render(&mut gpu, &mut painted, 12.0, 12);
    drop(painted);

    let mut scene = MenuScene::new(&gpu, &jar).expect("menu scene");
    assert!(scene.wait_for_clouds(&gpu, READY), "the clouds were built");
    let after = render(&mut gpu, &mut scene, 12.0, 12);
    let changed = before
        .chunks(4)
        .zip(after.chunks(4))
        .filter(|(a, b)| {
            a.iter()
                .zip(b.iter())
                .take(3)
                .any(|(x, y)| x.abs_diff(*y) > 4)
        })
        .count();
    let total = before.len() / 4;
    eprintln!("title pixels changed by the volumetric clouds: {changed} of {total}");
    assert!(
        changed > total / 50,
        "the sky through the roof changed ({changed} px)"
    );
    assert!(
        changed < total * 3 / 4,
        "only the sky changed ({changed} px)"
    );

    // Released with the scene's big memory, and rebuilt (no new shader work) on the next draw.
    scene.release_gpu_memory();
    let again = render(&mut gpu, &mut scene, 12.0, 12);
    assert_eq!(again.len(), after.len());
}

fn out_dir() -> PathBuf {
    let dir = root()
        .join("featwave-2026-11-09")
        .join("scratch")
        .join("X11");
    std::fs::create_dir_all(&dir).expect("create the output folder");
    dir
}

/// The title background at 1920x1080 with the painted clouds and with the volumetric ones, a
/// few moments apart; with `MC_GPU_TIMING=1` also the GPU cost per phase.
#[test]
#[ignore = "writes 1920x1080 PNGs for review; run by hand"]
fn render_title_pngs() {
    let Some(jar) = jar() else {
        eprintln!("skip: no 1.8.9 jar");
        return;
    };
    let Some(mut gpu) = gpu(1920, 1080) else {
        eprintln!("skip: no GPU");
        return;
    };
    let dir = out_dir();
    let tag = std::env::var("X11_TAG").unwrap_or_else(|_| "title".to_owned());
    let mut painted = MenuScene::new(&gpu, &jar).expect("menu scene");
    painted.set_volumetric_clouds(false);
    let px = render(&mut gpu, &mut painted, 12.0, 4);
    let path = dir.join(format!("{tag}-painted-t12.png"));
    image::save_buffer(&path, &px, 1920, 1080, image::ColorType::Rgba8).expect("png");
    drop(painted);

    let mut scene = MenuScene::new(&gpu, &jar).expect("menu scene");
    assert!(scene.wait_for_clouds(&gpu, READY));
    for t in [12.0, 31.0, 60.0] {
        let px = render(&mut gpu, &mut scene, t, 40);
        let path = dir.join(format!("{tag}-volumetric-t{t:02.0}.png"));
        image::save_buffer(&path, &px, 1920, 1080, image::ColorType::Rgba8).expect("png");
        eprintln!("wrote {}", path.display());
    }
    if std::env::var_os("MC_GPU_TIMING").is_some() {
        // The clouds trace inside the background phase: their share is the difference.
        for volumetric in [true, false] {
            scene.set_volumetric_clouds(volumetric);
            if volumetric {
                assert!(scene.wait_for_clouds(&gpu, READY));
            }
            let mut phases = PhaseList::new();
            phases.insert(menu_scene::BACKGROUND_ORDER, scene.background_phase());
            for i in 0..240 {
                scene.update(20.0 + f64::from(i) / 360.0, 1920, 1080);
                let target = gpu.acquire().expect("offscreen frame");
                phases.render(&gpu, target.view(), &Camera::default(), 0.0);
            }
            let _ = gpu.read_pixels();
            eprintln!(
                "volumetric clouds {volumetric}:\n{}",
                phases.gpu_timings().format_table()
            );
        }
    }
}
