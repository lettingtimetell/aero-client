//! Lightning bolts render like vanilla's
//! `RenderLightningBolt`, pixel for pixel.
//!
//! The reference is vanilla, not a re-derivation: `javaref/LightningRef.java`
//! (`generate-lightning.ps1`) runs the unmodified recompiled 1.8.9 `RenderLightningBolt.doRender`
//! in an LWJGL Pbuffer under `RenderGlobal.renderEntities`' state — back faces **culled** (the
//! world pass's `enableCull()`, which nothing before the weather-effect loop undoes), depth
//! `GL_LEQUAL` with depth writes **on**, an identity (all-white) lightmap and no fog — and saved
//! each frame as `tests/snapshots/lightning_gl-<case>.png`.
//!
//! mc-rs used to draw the bolt two-sided with depth writes off, so every ray through a tube added
//! the near **and** the far wall, and every tube added on top of the others: twice vanilla's
//! layers, a saturated white core instead of vanilla's pale lavender bands. This test renders
//! the same four cases through the real [`WeatherPhase`] and compares every pixel.
//!
//! Skips (loudly) without a wgpu adapter. Set `MC_LIGHTNING_DUMP=<dir>` to write mc-rs's frames.

use std::path::{Path, PathBuf};

use glam::DVec3;
use mc_render::sky::{LightningStrike, WeatherPhase};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

const W: u32 = 256;
const H: u32 = 256;

/// `LightningRef.CASES`: name, `boltVertex`, camera-relative strike position, yaw, pitch.
const CASES: [(&str, i64, [f64; 3], f32, f32); 4] = [
    ("base", 0x0BAD_C0DE, [-0.5, -4.0, 6.0], 0.0, -15.0),
    ("middle", 12345, [-0.5, -24.0, 30.0], 0.0, -10.0),
    ("side", -1, [30.0, -40.0, -0.5], -90.0, -5.0),
    ("below", 7, [-3.0, -1.0, -3.0], 0.0, -80.0),
];

fn headless() -> Option<Gpu> {
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
            eprintln!("SKIP: no wgpu adapter on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

fn snapshot(name: &str) -> Vec<[u8; 3]> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/snapshots")
        .join(format!("lightning_gl-{name}.png"));
    let image = image::open(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .to_rgb8();
    assert_eq!(image.dimensions(), (W, H), "{}", path.display());
    image.pixels().map(|p| p.0).collect()
}

fn render(gpu: &mut Gpu, bolt_vertex: i64, rel: [f64; 3], yaw: f32, pitch: f32) -> Vec<[u8; 3]> {
    let mut cam = Camera {
        pos: DVec3::ZERO,
        yaw,
        pitch,
        fov: 70.0,
        near: 0.05,
        far: 256.0,
        aspect: W as f32 / H as f32,
        ..Camera::default()
    };
    cam.update();

    let mut weather = WeatherPhase::new(gpu);
    weather.add_lightning_strike(LightningStrike {
        position: DVec3::from_array(rel),
        bolt_vertex,
    });
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::WEATHER, Box::new(weather));
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), &cam, 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    pixels.chunks(4).map(|p| [p[0], p[1], p[2]]).collect()
}

fn dump(dir: &Path, name: &str, pixels: &[[u8; 3]]) {
    let mut image = image::RgbImage::new(W, H);
    for (i, p) in pixels.iter().enumerate() {
        image.put_pixel(i as u32 % W, i as u32 / W, image::Rgb(*p));
    }
    let path: PathBuf = dir.join(format!("lightning_rs-{name}.png"));
    image.save(&path).expect("dump png");
}

/// Per-pixel comparison of every case against vanilla's frame.
///
/// Both sides rasterise the same bolt on the same GPU, but through different APIs (fixed-function
/// GL vs wgpu) from vertices computed in different precisions (Java offsets in `double` through
/// the GL matrix stack, mc-rs camera-relative `f32`), so a pixel whose centre lies within a
/// rounding error of a tube edge may land on either side. Everything else must be identical:
/// the lit area to within 1%, and at least 99% of lit pixels exactly equal. (On the RTX 5070 Ti
/// this was written on, every lit pixel of all four frames is exactly equal; before the fix,
/// none was, and the core saturated at blue 255 against vanilla's 152.)
#[test]
fn lightning_bolts_match_vanilla_render_lightning_bolt_pixel_for_pixel() {
    let Some(mut gpu) = headless() else {
        return;
    };
    let dump_dir = std::env::var_os("MC_LIGHTNING_DUMP").map(PathBuf::from);
    let mut failures = Vec::new();
    for (name, bolt_vertex, rel, yaw, pitch) in CASES {
        let java = snapshot(name);
        let rust = render(&mut gpu, bolt_vertex, rel, yaw, pitch);
        if let Some(dir) = &dump_dir {
            dump(dir, name, &rust);
        }
        let lit = |p: &[u8; 3]| p.iter().any(|&c| c > 0);
        let java_lit = java.iter().filter(|p| lit(p)).count();
        let rust_lit = rust.iter().filter(|p| lit(p)).count();
        let union_lit = java
            .iter()
            .zip(&rust)
            .filter(|(j, r)| lit(j) || lit(r))
            .count();
        let exact = java
            .iter()
            .zip(&rust)
            .filter(|(j, r)| (lit(j) || lit(r)) && j == r)
            .count();
        let java_max = java.iter().map(|p| p[2]).max().unwrap_or(0);
        let rust_max = rust.iter().map(|p| p[2]).max().unwrap_or(0);
        let exact_share = exact as f64 / union_lit.max(1) as f64;
        let area_ratio = rust_lit as f64 / java_lit.max(1) as f64;
        eprintln!(
            "{name}: java lit {java_lit} max blue {java_max}; rust lit {rust_lit} max blue \
             {rust_max}; exact {exact}/{union_lit} ({:.2}%)",
            exact_share * 100.0
        );
        assert!(
            java_lit > 1_000,
            "{name}: the reference frame shows too little bolt"
        );
        if !(0.99..=1.01).contains(&area_ratio) || exact_share < 0.99 || java_max != rust_max {
            failures.push(format!(
                "{name}: lit area ratio {area_ratio:.4}, exact share {:.4}, max blue java \
                 {java_max} rust {rust_max}",
                exact_share
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
