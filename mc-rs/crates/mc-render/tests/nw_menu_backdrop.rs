//! The Glass and Shore scenes as the title screen's background,
//! through `LoadingCurtain` at opacity 1 with a `BackdropStyle`.
//!
//! * The mirror flips the scene exactly (every pixel is its mirror image's pixel).
//! * The scrim darkens the left of the frame, where the title's logo and buttons sit, and leaves
//!   the right untouched.
//! * Giving the screen-sized GPU memory back and drawing again gives the same picture.
//!
//! Headless GPU (skips without one); the vanilla jar is optional (the scenes draw stand-ins
//! without it; set `MC_JARS_DIR`).

use std::path::{Path, PathBuf};

use mc_render::loading_scene::{BackdropStyle, JarTextures, LoadingCurtain, SceneFrame, SceneKind};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

const SIZE: (u32, u32) = (320, 180);

fn jar() -> Option<PathBuf> {
    let jars = std::env::var_os("MC_JARS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .ancestors()
                .nth(3)
                .expect("repository root")
                .join("MCP-919-main")
                .join("jars")
        });
    let jar = jars.join("versions").join("1.8.9").join("1.8.9.jar");
    jar.is_file().then_some(jar)
}

fn gpu() -> Option<Gpu> {
    match Gpu::headless(
        SIZE.0,
        SIZE.1,
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

/// One frame of `curtain` at `time` (the title screen's background slot's order), read back.
fn shoot(
    gpu: &mut Gpu,
    phases: &mut PhaseList,
    curtain: &mut LoadingCurtain,
    time: f32,
) -> Vec<u8> {
    curtain.update(
        SceneFrame {
            time_s: time,
            dt_s: 0.0,
            width: SIZE.0,
            height: SIZE.1,
            progress: 1.0,
            loaded: true,
            opacity: 1.0,
            logo: [0.0; 4],
            seed: 7,
        },
        1.0,
    );
    let target = gpu.acquire().expect("offscreen frame");
    phases.render(gpu, target.view(), &Camera::default(), 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    drop(target);
    pixels
}

fn texel(pixels: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * SIZE.0 + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2]]
}

fn luma(p: [u8; 3]) -> f32 {
    0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2])
}

#[test]
fn the_mirror_flips_exactly_the_scrim_darkens_only_the_left_and_a_release_redraws_the_same() {
    let Some(mut gpu) = gpu() else {
        eprintln!("skip: no GPU");
        return;
    };
    for kind in SceneKind::ALL {
        let textures = jar()
            .map(|jar| JarTextures::load(&jar, kind.jar_textures()))
            .unwrap_or_default();
        let mut curtain = LoadingCurtain::new(&gpu, kind, &textures, 7);
        assert_eq!(curtain.style(), BackdropStyle::default());
        let mut phases = PhaseList::new();
        // Where the title screen puts it: its background slot, under the UI.
        phases.insert(PhaseOrder(PhaseOrder::UI.0 - 10), curtain.phase());
        let time = 2.5;
        let plain = shoot(&mut gpu, &mut phases, &mut curtain, time);
        let again = shoot(&mut gpu, &mut phases, &mut curtain, time);
        assert!(
            plain == again,
            "{kind:?}: a frame at a fixed time is deterministic"
        );

        // The mirror: every pixel is its mirror image's.
        curtain.set_style(BackdropStyle {
            mirror: true,
            scrim: 0.0,
        });
        let mirrored = shoot(&mut gpu, &mut phases, &mut curtain, time);
        let mut asymmetric = 0;
        for y in 0..SIZE.1 {
            for x in 0..SIZE.0 {
                let flipped = texel(&plain, SIZE.0 - 1 - x, y);
                assert_eq!(texel(&mirrored, x, y), flipped, "{kind:?} at ({x}, {y})");
                if texel(&plain, x, y) != flipped {
                    asymmetric += 1;
                }
            }
        }
        // The control: the scene is not symmetric, so the check above could fail.
        assert!(
            asymmetric > (SIZE.0 * SIZE.1 / 4) as usize,
            "{kind:?}: only {asymmetric} pixels differ from their mirror image"
        );

        // The scrim: darker on the left (where the logo and buttons go), the right untouched.
        curtain.set_style(BackdropStyle {
            mirror: false,
            scrim: 0.2,
        });
        let scrimmed = shoot(&mut gpu, &mut phases, &mut curtain, time);
        let (mut left_before, mut left_after) = (0.0_f32, 0.0_f32);
        for y in 0..SIZE.1 {
            for x in 0..SIZE.0 {
                let (a, b) = (texel(&plain, x, y), texel(&scrimmed, x, y));
                let u = (x as f32 + 0.5) / SIZE.0 as f32;
                if u >= 0.9 {
                    for k in 0..3 {
                        assert!(
                            a[k].abs_diff(b[k]) <= 1,
                            "{kind:?} at ({x}, {y}): the right edge changed: {a:?} -> {b:?}"
                        );
                    }
                } else if u < 0.3 {
                    for k in 0..3 {
                        assert!(b[k] <= a[k].saturating_add(1), "{kind:?}: never brighter");
                    }
                    left_before += luma(a);
                    left_after += luma(b);
                }
            }
        }
        assert!(
            left_after < left_before * 0.9,
            "{kind:?}: the left is darker: {left_before} -> {left_after}"
        );

        // A release gives the screen-sized memory back; the next frame is the same picture.
        curtain.set_style(BackdropStyle::default());
        assert!(!curtain.gpu_memory_released());
        curtain.release_gpu_memory();
        assert!(curtain.gpu_memory_released());
        let redrawn = shoot(&mut gpu, &mut phases, &mut curtain, time);
        assert!(!curtain.gpu_memory_released());
        assert!(
            plain == redrawn,
            "{kind:?}: the same picture after a release"
        );
    }
}
