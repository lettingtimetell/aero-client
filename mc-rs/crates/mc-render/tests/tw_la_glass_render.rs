//! The render harness for the loading screen's glass scene and its curtain.
//!
//! The ignored tests render real frames on the GPU and write PNGs for a human to look at:
//!
//! ```text
//! cargo test -p mc-render --test tw_la_glass_render -- --ignored --nocapture render_pngs
//! cargo test -p mc-render --test tw_la_glass_render -- --ignored --nocapture render_sequence
//! MC_GPU_TIMING=1 cargo test -p mc-render --test tw_la_glass_render -- --ignored --nocapture gpu_cost
//! ```
//!
//! The shader is read from `src/loading_scene/glass.wgsl` **at run time**, so an edit to it shows
//! on the next run without a rebuild. Knobs: `LA_OUT` (folder, default
//! `<repo>/taskwave-2026-11-12/scratch/LA/png`), `LA_TAG` (file prefix), `LA_TIMES` (`0.5,3,7`),
//! `LA_SIZE` (`2560x1440`), `LA_SEED`, `LA_LOOK` (`frost=1.2,wire=0.8,debug=1`, see
//! `apply_look`), `LA_SCENE` (`glass` or `shore`, for the curtain tests). The vanilla jar comes
//! from `MC_JARS_DIR` (the `jars` folder) or the repository's own copy.

use std::path::{Path, PathBuf};

use mc_render::loading_scene::{
    self, GlassLook, JarTextures, LoadingCurtain, SceneFrame, SceneInit, SceneKind,
};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, RenderError};

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

fn size() -> (u32, u32) {
    std::env::var("LA_SIZE")
        .ok()
        .and_then(|s| {
            let (w, h) = s.split_once('x')?;
            Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
        })
        .unwrap_or((1920, 1080))
}

fn out_dir() -> PathBuf {
    let dir = std::env::var_os("LA_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            root()
                .join("taskwave-2026-11-12")
                .join("scratch")
                .join("LA")
                .join("png")
        });
    std::fs::create_dir_all(&dir).expect("create the output folder");
    dir
}

fn seed() -> u32 {
    std::env::var("LA_SEED")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(20_261_112)
}

fn gpu_at(size: (u32, u32), samples: u32) -> Option<Gpu> {
    match Gpu::headless(
        size.0,
        size.1,
        GpuOptions {
            vsync: false,
            sample_count: samples,
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

/// `LA_LOOK="frost=1.2,wire=0.8,spot=0.4:0.1"` overrides [`GlassLook`] fields.
fn apply_look(mut look: GlassLook) -> GlassLook {
    let Ok(spec) = std::env::var("LA_LOOK") else {
        return look;
    };
    for item in spec.split(',').filter(|s| !s.trim().is_empty()) {
        let (key, value) = item.split_once('=').expect("key=value");
        let num = |v: &str| v.trim().parse::<f32>().expect("a number");
        match key.trim() {
            "exposure" => look.exposure = num(value),
            "frost" => look.frost = num(value),
            "refraction" => look.refraction = num(value),
            "wire" => look.wire = num(value),
            "sparkle" => look.sparkle = num(value),
            "fan" => look.fan = num(value),
            "glow" => look.glow = num(value),
            "fan_speed" => look.fan_speed = num(value),
            "dimples" => look.dimples = num(value),
            "dimple_depth" => look.dimple_depth = num(value),
            "fog" => look.fog = num(value),
            "sway" => look.sway = num(value),
            "spot_size" => look.spot_size = num(value),
            "debug" => look.debug = num(value),
            "spot" => {
                let (x, y) = value.split_once(':').expect("spot=x:y");
                look.spot = [num(x), num(y)];
            }
            other => panic!("unknown look key {other}"),
        }
    }
    eprintln!("look: {look:?}");
    look
}

/// The glass shader as it is on disk now (so edits show without a rebuild).
fn glass_wgsl() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("loading_scene")
        .join("glass.wgsl");
    std::fs::read_to_string(&path).expect("read glass.wgsl")
}

/// A curtain around the glass scene built from the on-disk shader, or around `LA_SCENE`.
fn curtain(gpu: &Gpu, textures: &JarTextures) -> LoadingCurtain {
    let seed = seed();
    let init = SceneInit { textures, seed };
    match std::env::var("LA_SCENE").as_deref() {
        Ok("shore") => LoadingCurtain::new(gpu, SceneKind::Shore, textures, seed),
        _ => {
            let wgsl = glass_wgsl();
            let scene = loading_scene::glass_scene(
                gpu,
                &init,
                apply_look(GlassLook::default()),
                Some(&wgsl),
            );
            LoadingCurtain::with_scene(gpu, scene, seed)
        }
    }
}

fn frame_at(t: f32, opacity: f32) -> SceneFrame {
    let (w, h) = size();
    SceneFrame {
        time_s: t,
        dt_s: 1.0 / 60.0,
        width: w,
        height: h,
        progress: (t / 2.0).min(1.0),
        loaded: t > 2.0,
        opacity,
        logo: logo_rect(w, h),
        seed: seed(),
    }
}

/// Where the client's overlay puts the logo at this size (kept in step with `tw_loading.rs`).
fn logo_rect(w: u32, h: u32) -> [f32; 4] {
    let s = h as f32 / 1080.0;
    let (lw, lh) = (560.0 * s, 110.0 * s);
    [(w as f32 - lw) * 0.5, h as f32 * 0.43 - lh * 0.5, lw, lh]
}

fn render(gpu: &mut Gpu, phases: &mut PhaseList, curtain: &mut LoadingCurtain, t: f32) -> Vec<u8> {
    curtain.update(frame_at(t, 1.0), 1.0);
    let target = gpu.acquire().expect("offscreen frame");
    phases.render(gpu, target.view(), &Camera::default(), 0.0);
    gpu.read_pixels().expect("readback")
}

#[test]
#[ignore = "renders full-size PNGs for review; run by hand"]
fn render_pngs() {
    let Some(jar) = jar() else {
        eprintln!("skip: no 1.8.9 jar");
        return;
    };
    let (w, h) = size();
    let Some(mut gpu) = gpu_at((w, h), 1) else {
        eprintln!("skip: no GPU");
        return;
    };
    let kind = match std::env::var("LA_SCENE").as_deref() {
        Ok("shore") => SceneKind::Shore,
        _ => SceneKind::Glass,
    };
    let started = std::time::Instant::now();
    let textures = JarTextures::load(&jar, kind.jar_textures());
    eprintln!(
        "jar: {} textures in {:.1} ms",
        textures.len(),
        started.elapsed().as_secs_f64() * 1000.0
    );
    let started = std::time::Instant::now();
    let mut curtain = curtain(&gpu, &textures);
    eprintln!(
        "scene built in {:.1} ms",
        started.elapsed().as_secs_f64() * 1000.0
    );
    let mut phases = PhaseList::new();
    phases.insert(loading_scene::CURTAIN_ORDER, curtain.phase());
    let tag = std::env::var("LA_TAG").unwrap_or_else(|_| "glass".to_owned());
    let times: Vec<f32> = std::env::var("LA_TIMES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![3.0]);
    let dir = out_dir();
    for &t in &times {
        let px = render(&mut gpu, &mut phases, &mut curtain, t);
        let path = dir.join(format!("{tag}-t{t:04.1}.png"));
        image::save_buffer(&path, &px, w, h, image::ColorType::Rgba8).expect("png");
        eprintln!("wrote {}", path.display());
    }
}

#[test]
#[ignore = "renders a short frame sequence (motion review); run by hand"]
fn render_sequence() {
    let Some(jar) = jar() else {
        eprintln!("skip: no 1.8.9 jar");
        return;
    };
    let (w, h) = size();
    let Some(mut gpu) = gpu_at((w, h), 1) else {
        eprintln!("skip: no GPU");
        return;
    };
    let textures = JarTextures::load(&jar, SceneKind::Glass.jar_textures());
    let mut curtain = curtain(&gpu, &textures);
    let mut phases = PhaseList::new();
    phases.insert(loading_scene::CURTAIN_ORDER, curtain.phase());
    let tag = std::env::var("LA_TAG").unwrap_or_else(|_| "seq".to_owned());
    let start: f32 = std::env::var("LA_START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3.0);
    let frames: u32 = std::env::var("LA_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let step: f32 = std::env::var("LA_STEP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0 / 30.0);
    let dir = out_dir().join(&tag);
    std::fs::create_dir_all(&dir).expect("sequence folder");
    for i in 0..frames {
        let t = start + step * i as f32;
        let px = render(&mut gpu, &mut phases, &mut curtain, t);
        let path = dir.join(format!("{tag}-{i:03}.png"));
        image::save_buffer(&path, &px, w, h, image::ColorType::Rgba8).expect("png");
    }
    eprintln!("wrote {frames} frames to {}", dir.display());
}

#[test]
#[ignore = "GPU cost of the scene + curtain; run by hand with MC_GPU_TIMING=1"]
fn gpu_cost() {
    let Some(jar) = jar() else {
        eprintln!("skip: no 1.8.9 jar");
        return;
    };
    let (w, h) = size();
    let Some(mut gpu) = gpu_at((w, h), 1) else {
        eprintln!("skip: no GPU");
        return;
    };
    if !gpu.supports_gpu_timing() {
        eprintln!("skip: set MC_GPU_TIMING=1 (and the adapter must support timestamps)");
        return;
    }
    let kind = match std::env::var("LA_SCENE").as_deref() {
        Ok("shore") => SceneKind::Shore,
        _ => SceneKind::Glass,
    };
    let textures = JarTextures::load(&jar, kind.jar_textures());
    let mut curtain = LoadingCurtain::new(&gpu, kind, &textures, seed());
    let mut phases = PhaseList::new();
    phases.insert(loading_scene::CURTAIN_ORDER, curtain.phase());
    let cam = Camera::default();
    for i in 0..600 {
        curtain.update(frame_at(2.0 + i as f32 / 360.0, 1.0), 1.0);
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

/// Not ignored: each scene renders through the curtain into a small 4x-MSAA frame (the way the
/// client inserts it) without a validation error and not as a flat colour; at opacity 0 the
/// curtain's phase is disabled and the frame is byte-identical to a frame without it; dropping
/// the curtain releases everything cleanly. Skips without a GPU (the jar is optional: the scenes
/// draw stand-ins without it).
#[test]
fn every_scene_renders_through_the_curtain_and_vanishes_at_zero_opacity() {
    let Some(mut gpu) = gpu_at((320, 180), 4) else {
        eprintln!("skip: no GPU");
        return;
    };
    for kind in SceneKind::ALL {
        let textures = jar()
            .map(|jar| JarTextures::load(&jar, kind.jar_textures()))
            .unwrap_or_default();
        let mut curtain = LoadingCurtain::new(&gpu, kind, &textures, 7);
        assert_eq!(curtain.scene_name(), kind.name());
        let mut phases = PhaseList::new();
        phases.insert(loading_scene::CURTAIN_ORDER, curtain.phase());
        let cam = Camera::default();
        for t in [0.2, 2.5] {
            let frame = SceneFrame {
                time_s: t,
                dt_s: 1.0 / 60.0,
                width: 320,
                height: 180,
                progress: 0.5,
                loaded: false,
                opacity: 1.0,
                logo: [0.0; 4],
                seed: 7,
            };
            curtain.update(frame, 1.0);
            let target = gpu.acquire().expect("offscreen frame");
            phases.render(&gpu, target.view(), &cam, 0.0);
        }
        assert_eq!(curtain.frames_drawn(), 2, "{kind:?}");
        assert_eq!(curtain.last_size(), (320, 180));
        let px = gpu.read_pixels().expect("readback");
        let first = [px[0], px[1], px[2]];
        let varied = px.chunks_exact(4).filter(|p| p[..3] != first).count();
        assert!(
            varied > px.len() / 16,
            "{kind:?}: the frame is flat ({varied} differing pixels)"
        );

        // Opacity 0: the phase is disabled, and the frame is exactly what it is without it.
        let mut bare = PhaseList::new();
        let target = gpu.acquire().expect("offscreen frame");
        bare.render(&gpu, target.view(), &cam, 0.0);
        let without = gpu.read_pixels().expect("readback");
        curtain.update(
            SceneFrame {
                opacity: 0.0,
                ..SceneFrame::default()
            },
            1.0,
        );
        let target = gpu.acquire().expect("offscreen frame");
        phases.render(&gpu, target.view(), &cam, 0.0);
        let with_transparent = gpu.read_pixels().expect("readback");
        assert_eq!(
            curtain.frames_drawn(),
            2,
            "a transparent curtain draws nothing"
        );
        assert!(
            without == with_transparent,
            "{kind:?}: a transparent curtain changed the frame"
        );

        // Dropping the curtain and its phase frees everything without a validation error.
        assert_eq!(phases.remove_named(loading_scene::CURTAIN_PHASE_NAME), 1);
        drop(curtain);
        let target = gpu.acquire().expect("offscreen frame");
        phases.render(&gpu, target.view(), &cam, 0.0);
    }
}

/// Not ignored: half opacity really blends (the frame is between the curtain's and the frame's
/// own colour), so the fade into the title screen is a cross-fade, not a cut.
#[test]
fn half_opacity_blends_over_the_frame() {
    let Some(mut gpu) = gpu_at((160, 90), 1) else {
        eprintln!("skip: no GPU");
        return;
    };
    let textures = JarTextures::empty();
    let mut curtain = LoadingCurtain::new(&gpu, SceneKind::Glass, &textures, 3);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 1.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
    phases.insert(loading_scene::CURTAIN_ORDER, curtain.phase());
    let cam = Camera::default();
    let mut shot = |opacity: f32| {
        curtain.update(
            SceneFrame {
                time_s: 1.0,
                opacity,
                ..SceneFrame::default()
            },
            1.0,
        );
        let target = gpu.acquire().expect("offscreen frame");
        phases.render(&gpu, target.view(), &cam, 0.0);
        gpu.read_pixels().expect("readback")
    };
    let full = shot(1.0);
    let half = shot(0.5);
    // Magenta underneath: at half opacity red and blue rise toward 255 by about half the way.
    let mean = |px: &[u8], c: usize| {
        px.chunks_exact(4).map(|p| f64::from(p[c])).sum::<f64>() / (px.len() / 4) as f64
    };
    let (r_full, r_half) = (mean(&full, 0), mean(&half, 0));
    let expected = r_full + (255.0 - r_full) * 0.5;
    assert!(
        (r_half - expected).abs() < 6.0,
        "red: full {r_full:.1}, half {r_half:.1}, expected about {expected:.1}"
    );
    let (g_full, g_half) = (mean(&full, 1), mean(&half, 1));
    assert!(
        (g_half - g_full * 0.5).abs() < 6.0,
        "green: full {g_full:.1}, half {g_half:.1}"
    );
}
