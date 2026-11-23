//! Two sky behaviours `SkyPhase` had implemented behind
//! switches that were never turned on, now vanilla by default.
//!
//! * **The horizon plane is culled from below.** java: `EntityRenderer.renderWorldPass` opens
//!   with `GlStateManager.enableCull()`, nothing before `RenderGlobal.renderSky` turns it off,
//!   and `renderSky` never touches it. Its last draw is `glSkyList2` (reversed winding, so it
//!   faces up) translated to `World.getHorizon()`. With the eye below the horizon that plane is
//!   above the eye and back-facing, so vanilla culls it and the sky plane stays visible overhead.
//!   mc-rs drew it unculled, last, over the whole upper sky in the dark horizon colour.
//! * **Stars wheel with the moon.** java: `starGLCallList` is drawn inside the same
//!   `rotate(-90, Y)`, `rotate(getCelestialAngle * 360, X)` block as the sun and moon.
//!   mc-rs drew the star field fixed.
//!
//! Skips (loudly) without a wgpu adapter.

use glam::DVec3;
use mc_render::sky::{Dimension, Environment, SkyPhase, Weather};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

/// Stars are 0.3-0.5 blocks across at distance 100, so they need a real resolution to raster.
const SIZE: u32 = 256;

fn headless() -> Option<Gpu> {
    match Gpu::headless(
        SIZE,
        SIZE,
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

/// The dome alone (no textures: the sun and moon stay out of frame), cleared to magenta.
fn render(
    gpu: &mut Gpu,
    world_time: i64,
    eye_y: f64,
    pitch: f32,
    configure: impl FnOnce(&mut SkyPhase),
) -> Vec<[u8; 3]> {
    let mut sky = SkyPhase::new(gpu);
    sky.environment = Environment {
        world_time,
        dimension: Dimension::Overworld,
        weather: Weather::default(),
    };
    sky.set_frame_origin_y(eye_y);
    configure(&mut sky);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 1.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
    phases.insert(PhaseOrder::SKY, Box::new(sky));
    let mut cam = Camera {
        pos: DVec3::new(0.0, eye_y, 0.0),
        pitch,
        ..Camera::default()
    };
    cam.update();
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), &cam, 0.0);
    gpu.read_pixels()
        .expect("readback")
        .chunks(4)
        .map(|p| [p[0], p[1], p[2]])
        .collect()
}

#[test]
fn looking_up_from_below_the_horizon_shows_the_sky_plane_not_the_horizon_plane() {
    let Some(mut gpu) = headless() else {
        return;
    };
    // Mid-morning: the sun is 45 degrees up in the east, nothing celestial is overhead, stars
    // are off. The sky plane is camera-relative, so straight up it must look the same from
    // y = 70 (above the horizon at 63) and from y = 40 (below it).
    let above = render(&mut gpu, 3000, 70.0, -90.0, |_| {});
    let below = render(&mut gpu, 3000, 40.0, -90.0, |_| {});
    let differing = above.iter().zip(&below).filter(|(a, b)| a != b).count();
    assert_eq!(
        differing,
        0,
        "{differing} of {} overhead pixels change when the eye drops below the horizon",
        above.len()
    );

    // Negative control: the pre-H7 unculled planes paint the horizon plane over the sky.
    let unculled = render(&mut gpu, 3000, 40.0, -90.0, |sky| sky.cull_planes = false);
    let centre = (SIZE / 2 * SIZE + SIZE / 2) as usize;
    assert_ne!(
        unculled[centre], above[centre],
        "the unculled horizon plane must cover the zenith from below (control)"
    );
    // ... in its dark `sky * 0.2 + 0.04` / `sky * 0.6 + 0.1` colour (lightly fogged), well
    // under two thirds of the sky's brightness.
    let luma = |p: [u8; 3]| u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2]);
    assert!(
        luma(unculled[centre]) * 3 < luma(above[centre]) * 2,
        "control: horizon plane {:?} vs sky {:?}",
        unculled[centre],
        above[centre]
    );
}

/// java: `EntityRenderer.renderWorldPass` (`:1346`) draws the sky only when
/// `renderDistanceChunks >= 4`. At 3 chunks only the fog-coloured backdrop (vanilla's clear
/// colour) may show; at 4 the sky plane is back overhead.
#[test]
fn render_distance_below_four_draws_no_sky_geometry() {
    let Some(mut gpu) = headless() else {
        return;
    };
    let at = |gpu: &mut Gpu, chunks: u32| {
        let mut sky = SkyPhase::new(gpu);
        sky.environment = Environment {
            world_time: 3000,
            dimension: Dimension::Overworld,
            weather: Weather::default(),
        };
        sky.set_frame_origin_y(70.0);
        let mut phases = PhaseList::new();
        phases.clear_color = wgpu::Color {
            r: 1.0,
            g: 0.0,
            b: 1.0,
            a: 1.0,
        };
        phases.insert(PhaseOrder::SKY, Box::new(sky));
        let mut cam = Camera {
            pos: DVec3::new(0.0, 70.0, 0.0),
            pitch: -45.0,
            ..Camera::default()
        };
        cam.set_view_distance_chunks(chunks);
        cam.update();
        let frame = gpu.acquire().expect("acquire");
        phases.render(gpu, frame.view(), &cam, 0.0);
        let pixels: Vec<[u8; 3]> = gpu
            .read_pixels()
            .expect("readback")
            .chunks(4)
            .map(|p| [p[0], p[1], p[2]])
            .collect();
        pixels
    };
    let three = at(&mut gpu, 3);
    let distinct = |pixels: &[[u8; 3]]| {
        let mut seen: Vec<[u8; 3]> = pixels.to_vec();
        seen.sort_unstable();
        seen.dedup();
        seen.len()
    };
    assert_eq!(
        distinct(&three),
        1,
        "3 chunks: only the backdrop may show, got {} colours",
        distinct(&three)
    );
    assert_ne!(
        three[0],
        [255, 0, 255],
        "the backdrop still paints the fog colour"
    );
    let four = at(&mut gpu, 4);
    assert!(
        distinct(&four) > 1,
        "4 chunks: the sky plane must draw over the backdrop (control)"
    );
}

#[test]
fn the_star_field_turns_with_the_celestial_angle() {
    let Some(mut gpu) = headless() else {
        return;
    };
    // Two night times with the same star brightness (`getStarBrightness` saturates through the
    // middle of the night), looking at the same patch of sky.
    let lit_mask = |pixels: &[[u8; 3]]| -> Vec<bool> {
        pixels
            .iter()
            .map(|p| p.iter().map(|&c| u32::from(c)).sum::<u32>() > 60)
            .collect()
    };
    let early = lit_mask(&render(&mut gpu, 17_000, 70.0, -60.0, |_| {}));
    let late = lit_mask(&render(&mut gpu, 19_000, 70.0, -60.0, |_| {}));
    assert!(
        early.iter().filter(|&&lit| lit).count() > 5,
        "no stars in view"
    );
    assert_ne!(
        early, late,
        "the stars must move between 17000 and 19000 ticks"
    );

    // Negative control: with the pre-H7 fixed star field, the same view is the same stars.
    let mut fixed = |time| {
        lit_mask(&render(&mut gpu, time, 70.0, -60.0, |sky| {
            sky.rotate_stars = false;
        }))
    };
    let fixed_early = fixed(17_000);
    let fixed_late = fixed(19_000);
    assert_eq!(
        fixed_early, fixed_late,
        "control: a fixed star field does not move"
    );
}
