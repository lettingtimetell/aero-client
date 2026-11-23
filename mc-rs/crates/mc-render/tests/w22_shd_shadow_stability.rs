//! The shader packs' shadow map holds still when it should.
//!
//! The terrain variant draws the shadow map only when its key changes and reuses it on every other
//! frame, while receivers rebuild their matrix every frame. Shadows stay put only if every frame
//! that shares a key projects a world point exactly where the cached map stored it, and if a held
//! clock (the Time Changer, or a server that stops the daylight cycle) never changes the key. These
//! checks drive `PackFrame::compute` and `ShadowFrame::compute_on_path` the way a 60 fps client
//! does: three frames per tick, the sun from vanilla's celestial angle, turned by the packs'
//! sun-path tilt.

use glam::{DVec3, Mat4, Vec3};
use mc_render::shaderpack::frame::{sun_direction, sun_path_tilt};
use mc_render::shaderpack::{EyeMedium, FrameInputs, PackFrame, ShadowFrame, ShadowSettings};
use mc_render::sky::fog::{FogDistance, FogMode};
use mc_render::sky::Environment;
use mc_render::Camera;

/// Frames per tick (60 fps).
const FRAMES_PER_TICK: u32 = 3;

fn project(m: Mat4, p: Vec3) -> Vec3 {
    let c = m * p.extend(1.0);
    c.truncate() / c.w
}

/// Toward the light that casts shadows (the sun by day, the moon by night) at a partial tick,
/// as `PackFrame` computes it for a running clock.
fn light_dir(world_time: i64, partial_ticks: f32, tilt: f32) -> Vec3 {
    let env = Environment {
        world_time,
        ..Environment::default()
    };
    let disc = sun_direction(env.celestial_angle(partial_ticks));
    let sun = if tilt == 0.0 {
        disc
    } else {
        (sun_path_tilt(tilt) * disc).normalize_or(disc)
    };
    if sun.y >= 0.0 {
        sun
    } else {
        -sun
    }
}

/// This frame's `PackFrame`, as the client builds it.
fn pack_frame(world_time: i64, partial_ticks: f32, tilt: f32, clock_held: bool) -> PackFrame {
    let env = Environment {
        world_time,
        ..Environment::default()
    };
    let camera = Camera::default();
    PackFrame::compute(&FrameInputs {
        environment: &env,
        biome_temperature: 0.8,
        fog_color: Vec3::ONE,
        fog: FogMode::CLEAR,
        fog_distance: FogDistance::Planar,
        eye_medium: EyeMedium::Air,
        render_distance_chunks: 8,
        camera: &camera,
        width: 64,
        height: 64,
        partial_ticks,
        clock_held,
        seconds: 0.0,
        frame_index: 0,
        in_world: true,
        sun_tilt: tilt,
    })
}

/// World points around the eye: the ground and the tops of things, out to 60 blocks.
fn sample_points(eye: DVec3) -> Vec<DVec3> {
    let mut points = Vec::new();
    for i in -6..=6 {
        for k in -6..=6 {
            for h in [-1.6, 3.0, 9.0] {
                let x = f64::from(i) * 10.0 + 0.3;
                let z = f64::from(k) * 10.0 - 0.7;
                points.push(eye + DVec3::new(x, h, z));
            }
        }
    }
    points
}

/// The largest disagreement between where receivers read a point and where the cached map stored
/// it, over `ticks` ticks of a running clock from `start`: (texels across the map, blocks along
/// the light), and how many times the map was drawn.
fn drift(
    settings: ShadowSettings,
    tilt: f32,
    start: i64,
    ticks: i64,
    eye: DVec3,
) -> (f32, f32, u32) {
    let points = sample_points(eye);
    let depth_range = 320.0 + f64::from(settings.distance) + 48.0;
    let mut cached: Option<ShadowFrame> = None;
    let mut renders = 0;
    let mut worst_texels = 0.0_f32;
    let mut worst_depth = 0.0_f32;
    for tick in 0..ticks {
        for f in 0..FRAMES_PER_TICK {
            let pt = f as f32 / FRAMES_PER_TICK as f32;
            let light = light_dir(start + tick, pt, tilt);
            let now = ShadowFrame::compute_on_path(settings, light, tilt, eye);
            if cached.is_none_or(|c| c.key != now.key) {
                cached = Some(now);
                renders += 1;
            }
            let map = cached.expect("set above");
            for p in &points {
                let stored = project(map.caster_matrix, (*p - map.caster_origin).as_vec3());
                let read = project(now.receiver_matrix, (*p - eye).as_vec3());
                let texels = (read.truncate() - stored.truncate()).length()
                    * 0.5
                    * settings.resolution as f32;
                let depth = (read.z - stored.z).abs() * depth_range as f32;
                worst_texels = worst_texels.max(texels);
                worst_depth = worst_depth.max(depth);
            }
        }
    }
    (worst_texels, worst_depth, renders)
}

/// Every frame that shares the cached map's key reads the map where it was drawn: no slide
/// between redraws and no snap back at the next one. The packs light along a path tilted 25
/// degrees (Classic's default, Dreamy's fixed tilt), so the tilt must not leak into the matrices
/// unquantised. The tilted path costs no extra redraws.
#[test]
fn receivers_read_the_cached_map_where_it_was_drawn() {
    let eye = DVec3::new(-3.47, 81.62, 21.31);
    let mut report = Vec::new();
    let mut worst = 0.0_f32;
    let mut draws = std::collections::BTreeMap::new();
    for resolution in [2048, 4096] {
        let settings = ShadowSettings {
            resolution,
            ..ShadowSettings::default()
        };
        for tilt in [0.0_f32, 25.0] {
            // Two minutes of game time in the morning, then two in the afternoon.
            for start in [2_000, 8_500] {
                let (texels, depth, renders) = drift(settings, tilt, start, 2_400, eye);
                draws.insert((resolution, tilt.to_bits(), start), renders);
                report.push(format!(
                    "res {resolution} tilt {tilt:>4} start {start:>5}: receivers off the map by up \
                     to {texels:.3} texels and {depth:.4} blocks of depth; {renders} map draws in \
                     {} frames",
                    2_400 * FRAMES_PER_TICK
                ));
                worst = worst.max(texels);
            }
        }
    }
    eprintln!("{}", report.join("\n"));
    assert!(
        worst < 0.01,
        "receivers slide against the cached shadow map by up to {worst:.3} texels"
    );
    for ((resolution, tilt, start), renders) in &draws {
        let flat = draws[&(*resolution, 0.0_f32.to_bits(), *start)];
        assert!(
            *renders <= flat + 2,
            "tilt {}: {renders} map draws against {flat} untilted",
            f32::from_bits(*tilt)
        );
    }
}

/// The settings (hours) whose map key changes from frame to frame over ten ticks of a clock held
/// at that time, at `tilt`.
fn flipping_settings(tilt: f32, clock_held: bool) -> Vec<f32> {
    let eye = DVec3::new(-3.47, 81.62, 21.31);
    let settings = ShadowSettings::default();
    let mut flipping = Vec::new();
    for quarter in 0..96 {
        let hours = quarter as f32 * 0.25;
        let ticks = ((f64::from(hours) - 6.0) * 1000.0).round() as i64;
        let held = ticks.rem_euclid(24_000);
        let mut keys = std::collections::BTreeSet::new();
        for _tick in 0..10 {
            for f in 0..FRAMES_PER_TICK {
                let pt = f as f32 / FRAMES_PER_TICK as f32;
                let frame = pack_frame(held, pt, tilt, clock_held);
                if frame.light_strength <= 0.0 {
                    continue;
                }
                keys.insert(ShadowFrame::compute_on_path(settings, frame.light_dir, tilt, eye).key);
            }
        }
        if keys.len() > 1 {
            flipping.push(hours);
        }
    }
    flipping
}

/// The Time Changer holds the clock (so does a server that stops the daylight cycle); vanilla
/// still adds the partial tick to the celestial angle, which swings the sun back and forth by one
/// tick 20 times a second. A held clock's sun stands still, so the map's key never changes, at
/// any Time Changer setting. Negative control: following the partial tick, some settings flip.
#[test]
fn a_held_clock_never_redraws_the_map() {
    for tilt in [0.0_f32, 25.0] {
        let held = flipping_settings(tilt, true);
        assert!(
            held.is_empty(),
            "tilt {tilt}: a held clock redraws the shadow map 20 times a second at {held:?} h"
        );
        let swinging = flipping_settings(tilt, false);
        eprintln!("tilt {tilt}: following the partial tick, the map flips at {swinging:?} h");
        assert!(
            !swinging.is_empty(),
            "the control finds no flipping setting: the check can't see the fault"
        );
    }
}

/// A held clock's sun is exactly where the clock holds it at every partial tick; a running
/// clock's sun moves on smoothly between ticks.
#[test]
fn a_held_clock_holds_the_sun_still() {
    for tilt in [0.0_f32, 25.0] {
        let first = pack_frame(3_250, 0.0, tilt, true);
        for pt in [0.25_f32, 0.5, 0.75, 0.99] {
            let held = pack_frame(3_250, pt, tilt, true);
            assert_eq!(held.light_dir, first.light_dir, "held, partial tick {pt}");
            assert_eq!(held.sun_disc, first.sun_disc, "held, partial tick {pt}");
            let running = pack_frame(3_250, pt, tilt, false);
            assert_ne!(
                running.light_dir, first.light_dir,
                "running, partial tick {pt}"
            );
        }
    }
}
