//! The Hitboxes mod's wide lines.
//!
//! wgpu has no wide lines, so `mc_render::entity::hitbox::HitboxPhase` expands every segment into
//! a screen-facing quad. These tests draw through the real phase on a headless GPU and read the
//! frame back: the width is in physical pixels, terrain/bodies in front hide the lines (depth
//! test, as F3+B), a segment crossing the near plane draws only its visible part, and an empty
//! phase records nothing and owns no GPU objects. Skips (printing `SKIP:`) without an adapter.

use glam::{Vec3, Vec4};
use mc_render::entity::hitbox::{HitboxPhase, HITBOX_PHASE_ORDER};
use mc_render::entity::layers::LayerBlend;
use mc_render::entity::render::EntityPhase;
use mc_render::entity::EntityVertex;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase};

const SIZE: u32 = 128;
const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

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

fn camera() -> Camera {
    let mut cam = Camera::default();
    cam.set_viewport(SIZE, SIZE);
    cam.update();
    cam
}

fn pixel_of(cam: &Camera, point: Vec3) -> (u32, u32) {
    let clip = cam.view_proj() * Vec4::new(point.x, point.y, point.z, 1.0);
    let ndc = clip.truncate() / clip.w;
    let x = ((ndc.x * 0.5 + 0.5) * SIZE as f32) as u32;
    let y = ((0.5 - ndc.y * 0.5) * SIZE as f32) as u32;
    (x.min(SIZE - 1), y.min(SIZE - 1))
}

fn is_red(pixels: &[u8], x: u32, y: u32) -> bool {
    let i = ((y * SIZE + x) * 4) as usize;
    pixels[i] > 200 && pixels[i + 1] < 60 && pixels[i + 2] < 60
}

/// Render `hitboxes` (and optionally an entity-pass occluder) and read the frame back.
fn render(gpu: &mut Gpu, hitboxes: HitboxPhase, occluder: Option<Vec<EntityVertex>>) -> Vec<u8> {
    let cam = camera();
    let mut list = PhaseList::new();
    list.clear_color = wgpu::Color::BLACK;
    if let Some(verts) = occluder {
        let mut phase = EntityPhase::new(gpu);
        phase.begin();
        phase.push_blended_lit(
            EntityPhase::SOLID_TEXTURE,
            LayerBlend::OpaqueUnlit,
            mc_render::entity::render::FULL_BRIGHT_COMBINED_LIGHT,
            &verts,
        );
        phase.finish(gpu);
        list.insert(PhaseOrder::ENTITIES, Box::new(phase));
    }
    list.insert(HITBOX_PHASE_ORDER, Box::new(hitboxes));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    list.render(gpu, frame.view(), &cam, 0.0);
    frame.present(&gpu.queue);
    gpu.read_pixels().expect("read back the frame")
}

/// Red rows in column `x`.
fn red_run(pixels: &[u8], x: u32) -> u32 {
    (0..SIZE).filter(|&y| is_red(pixels, x, y)).count() as u32
}

#[test]
fn an_empty_phase_records_nothing_and_owns_no_gpu_objects() {
    let Some(gpu) = headless() else {
        return;
    };
    let mut phase = HitboxPhase::new();
    phase.begin();
    phase.finish(&gpu);
    assert!(
        !phase.enabled(),
        "nothing queued: the frame skips the phase"
    );
    assert!(
        !phase.has_gpu_resources(),
        "no pipeline until a segment is drawn"
    );
    phase.push_line(Vec3::ZERO, Vec3::X, 2.0, RED);
    phase.finish(&gpu);
    assert!(phase.enabled() && phase.has_gpu_resources());
    phase.release();
    assert!(
        !phase.enabled() && !phase.has_gpu_resources(),
        "mod off drops them again"
    );
}

#[test]
fn the_line_width_is_in_physical_pixels_at_any_distance() {
    let Some(mut gpu) = headless() else {
        return;
    };
    let cam = camera();
    for (width, depth) in [(1.0f32, 3.0f32), (4.0, 3.0), (4.0, 12.0), (6.0, 30.0)] {
        let mut phase = HitboxPhase::new();
        phase.push_line(
            Vec3::new(-10.0, 0.0, depth),
            Vec3::new(10.0, 0.0, depth),
            width,
            RED,
        );
        phase.finish(&gpu);
        let pixels = render(&mut gpu, phase, None);
        let (x, _) = pixel_of(&cam, Vec3::new(0.0, 0.0, depth));
        let run = red_run(&pixels, x);
        assert!(
            run.abs_diff(width.round() as u32) <= 1,
            "width {width} at depth {depth}: {run} red rows"
        );
    }
}

#[test]
fn bodies_in_front_hide_the_lines_like_f3b() {
    let Some(mut gpu) = headless() else {
        return;
    };
    let cam = camera();
    // An opaque square at depth 2 covering the left half of the screen.
    let quad = |x0: f32, x1: f32| {
        let v = |x: f32, y: f32| EntityVertex {
            pos: [x, y, 2.0],
            uv: [0.5, 0.5],
            normal: [0.0, 0.0, -1.0],
            color: [0.0, 0.0, 1.0, 1.0],
        };
        vec![
            v(x0, -2.0),
            v(x1, -2.0),
            v(x1, 2.0),
            v(x0, -2.0),
            v(x1, 2.0),
            v(x0, 2.0),
        ]
    };
    let left = pixel_of(&cam, Vec3::new(-1.0, 0.0, 2.0)).0 < SIZE / 2;
    let occluder = if left {
        quad(-3.0, 0.0)
    } else {
        quad(0.0, 3.0)
    };
    let mut phase = HitboxPhase::new();
    phase.push_line(
        Vec3::new(-10.0, 0.0, 5.0),
        Vec3::new(10.0, 0.0, 5.0),
        3.0,
        RED,
    );
    phase.finish(&gpu);
    let pixels = render(&mut gpu, phase, Some(occluder));
    if let Some(path) = std::env::var_os("MC_X6_HITBOX_PNG") {
        image::save_buffer(path, &pixels, SIZE, SIZE, image::ExtendedColorType::Rgba8)
            .expect("write the PNG");
    }
    // Probe on the occluder's side of the screen and on the open side.
    let side = if left { -1.0 } else { 1.0 };
    let hidden_x = pixel_of(&cam, Vec3::new(side, 0.0, 5.0)).0;
    let shown_x = pixel_of(&cam, Vec3::new(-side, 0.0, 5.0)).0;
    assert!(
        hidden_x < SIZE / 2 && shown_x > SIZE / 2,
        "{hidden_x} / {shown_x}"
    );
    assert_eq!(red_run(&pixels, hidden_x), 0, "behind the body: hidden");
    assert!(red_run(&pixels, shown_x) >= 2, "in the open: drawn");
}

#[test]
fn a_segment_through_the_near_plane_draws_only_its_visible_part() {
    let Some(mut gpu) = headless() else {
        return;
    };
    let mut phase = HitboxPhase::new();
    // From behind the camera to well in front, below the eye: a floor edge running under you.
    phase.push_line(
        Vec3::new(0.0, -1.0, -3.0),
        Vec3::new(0.0, -1.0, 8.0),
        3.0,
        RED,
    );
    // Entirely behind the camera: must not draw anything at all.
    phase.push_line(
        Vec3::new(-1.0, 0.0, -2.0),
        Vec3::new(1.0, 0.0, -2.0),
        3.0,
        RED,
    );
    phase.finish(&gpu);
    let pixels = render(&mut gpu, phase, None);
    let red = (0..SIZE)
        .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
        .filter(|&(x, y)| is_red(&pixels, x, y))
        .count();
    let top_half_red = (0..SIZE / 2)
        .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
        .filter(|&(x, y)| is_red(&pixels, x, y))
        .count();
    assert!(
        red > 50,
        "the visible part of the floor line is drawn: {red}"
    );
    assert_eq!(
        top_half_red, 0,
        "nothing above the horizon: no wrapped or inverted quad"
    );
    assert!(
        red < (SIZE * SIZE / 4) as usize,
        "no runaway quad: {red} red pixels"
    );
}
