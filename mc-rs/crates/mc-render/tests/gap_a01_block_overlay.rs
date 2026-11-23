//! The block selection outline lands where the block is.
//!
//! `s11_render.rs`'s `the_block_overlay_phase_builds_and_records` already proves the phase can
//! be constructed and recorded without a validation error, but it only asserts that the readback
//! has the right *length* — it would pass just as happily if the outline drew nothing at all, or
//! drew in the wrong place. That is exactly the failure mode A01-02 describes on the client side
//! (`selection` stays `None` forever), so the wiring fix deserves a test that can tell the
//! difference.
//!
//! This renders the real [`BlockOverlayPhase`] through the real [`PhaseList`] into a headless
//! target twice — once with no selection, once with one — and checks that
//!
//! 1. a selection changes pixels at all, and no selection changes none;
//! 2. the pixels it changes surround the projection of the block's own centre, computed here
//!    from the camera matrices rather than from the phase; and
//! 3. moving the block moves the outline with it, in the direction the projection says.
//!
//! No atlas is needed: `BlockOverlayPhase::new(gpu, None)` builds the line pipeline only, which
//! is the half A01-02 is about.

use mc_math::{BlockPos, DVec3};
use mc_render::terrain::{fixture, BlockOverlayPhase, SelectionBox};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 192;

fn gpu() -> Option<Gpu> {
    match Gpu::headless(
        WIDTH,
        HEIGHT,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(g) => {
            // Turn wgpu validation errors into test failures instead of silent no-draws.
            g.device.on_uncaptured_error(std::sync::Arc::new(|e| {
                panic!("wgpu validation error: {e}")
            }));
            Some(g)
        }
        Err(RenderError::NoAdapter) => {
            eprintln!("skip: no wgpu adapter available on this machine");
            None
        }
        Err(e) => panic!("headless context: {e}"),
    }
}

/// Eye at the centre of a block face, looking down `+Z` — vanilla yaw `0` is south.
fn camera() -> Camera {
    let mut cam = Camera {
        pos: DVec3::new(8.5, 17.5, 0.5),
        yaw: 0.0,
        pitch: 0.0,
        ..Camera::default()
    };
    cam.set_viewport(WIDTH, HEIGHT);
    cam.update();
    cam
}

/// Project a world point to framebuffer pixels through the same two matrices the phase's vertex
/// shader multiplies by, so a mismatch is the renderer's and not this test's arithmetic.
fn project(cam: &Camera, world: DVec3) -> (f32, f32) {
    let rel = cam.relative(world);
    let ndc = (cam.proj * cam.view).project_point3(rel);
    // wgpu NDC is Y-up over `-1..1`; the framebuffer is Y-down over `0..height`.
    (
        (ndc.x * 0.5 + 0.5) * WIDTH as f32,
        (0.5 - ndc.y * 0.5) * HEIGHT as f32,
    )
}

/// Render one frame and hand back the RGBA bytes.
fn frame_with(gpu: &mut Gpu, cam: &Camera, selection: Option<SelectionBox>) -> Vec<u8> {
    let mut overlay = BlockOverlayPhase::new(gpu, None);
    overlay.selection = selection;
    let mut phases = PhaseList::new();
    phases.insert(PhaseOrder::BLOCK_OVERLAY, Box::new(overlay));
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), cam, 0.0);
    drop(frame);
    gpu.read_pixels().expect("readback")
}

/// Every pixel that differs from `base`, as `(x, y)`.
fn changed_pixels(base: &[u8], other: &[u8]) -> Vec<(u32, u32)> {
    assert_eq!(base.len(), other.len());
    let mut out = Vec::new();
    for i in (0..base.len()).step_by(4) {
        if base[i..i + 4] != other[i..i + 4] {
            let px = (i / 4) as u32;
            out.push((px % WIDTH, px / WIDTH));
        }
    }
    out
}

#[test]
fn the_selection_outline_draws_around_the_block_it_names() {
    let Some(mut gpu) = gpu() else { return };
    let cam = camera();
    let stone = fixture::state_of("minecraft:stone");

    // The block eight metres straight ahead of the eye.
    let pos = BlockPos::new(8, 17, 8);
    let centre = DVec3::new(
        f64::from(pos.x) + 0.5,
        f64::from(pos.y) + 0.5,
        f64::from(pos.z) + 0.5,
    );

    let empty = frame_with(&mut gpu, &cam, None);
    let outlined = frame_with(&mut gpu, &cam, Some(SelectionBox::for_block(pos, stone)));

    // (1) `None` really is a no-op, and a `Some` really is not. This is the exact difference the
    // client-side wiring makes: `selection` used to be `None` on every frame of real play.
    assert_eq!(
        changed_pixels(&empty, &frame_with(&mut gpu, &cam, None)).len(),
        0,
        "two selection-free frames must be byte-identical"
    );
    let drawn = changed_pixels(&empty, &outlined);
    assert!(
        !drawn.is_empty(),
        "a selection box in front of the camera drew nothing"
    );

    // (2) The outline is a hollow 12-edge wireframe, so the *bounding box* of what it drew is
    // what should agree with the projected block, not its centroid density. Vanilla expands the
    // box by 0.002 before drawing, which is sub-pixel at this distance.
    let (min_x, max_x) = (
        drawn.iter().map(|p| p.0).min().expect("non-empty"),
        drawn.iter().map(|p| p.0).max().expect("non-empty"),
    );
    let (min_y, max_y) = (
        drawn.iter().map(|p| p.1).min().expect("non-empty"),
        drawn.iter().map(|p| p.1).max().expect("non-empty"),
    );
    let (cx, cy) = project(&cam, centre);
    let box_cx = f32::midpoint(min_x as f32, max_x as f32);
    let box_cy = f32::midpoint(min_y as f32, max_y as f32);
    assert!(
        (box_cx - cx).abs() <= 2.0 && (box_cy - cy).abs() <= 2.0,
        "the outline is centred on ({box_cx}, {box_cy}) but the block projects to ({cx}, {cy})"
    );

    // The far face of a 1 m cube 7.5 m away subtends roughly `height / (2 tan(fov/2) * d)`
    // pixels; the near face is bigger. Just assert it is a plausible cube and not the whole
    // screen or a single pixel — the placement assertion above carries the precision.
    let width_px = max_x - min_x;
    let height_px = max_y - min_y;
    assert!(
        (10..=90).contains(&width_px) && (10..=90).contains(&height_px),
        "outline extent {width_px}x{height_px} px is not a 1 m cube at 8 m"
    );

    // (3) The outline follows the block, in the direction the projection says it should.
    //
    // Note the sign: the camera faces south (vanilla yaw `0` is `+Z`) and `right = forward × up`
    // is `-X`, so moving the block two metres *east* moves its outline to the **left** of the
    // screen. This assertion is written against `project`'s answer rather than an intuition
    // about which way east is, and then the intuition is stated separately so that a future
    // handedness flip fails loudly instead of silently agreeing with itself.
    let east = BlockPos::new(pos.x + 2, pos.y, pos.z);
    let moved = frame_with(&mut gpu, &cam, Some(SelectionBox::for_block(east, stone)));
    let moved_px = changed_pixels(&empty, &moved);
    assert!(!moved_px.is_empty(), "the shifted selection drew nothing");
    let moved_cx = f32::midpoint(
        moved_px.iter().map(|p| p.0).min().expect("non-empty") as f32,
        moved_px.iter().map(|p| p.0).max().expect("non-empty") as f32,
    );
    let expected_cx = project(&cam, DVec3::new(centre.x + 2.0, centre.y, centre.z)).0;
    assert!(
        (moved_cx - expected_cx).abs() <= 2.0,
        "the shifted outline is at {moved_cx}, the projection says {expected_cx}"
    );
    assert!(
        expected_cx < cx && moved_cx < box_cx,
        "facing +Z, +X is screen-left: the outline should have moved left, \
         {box_cx} -> {moved_cx} (projection {cx} -> {expected_cx})"
    );
}

#[test]
fn a_selection_behind_the_camera_draws_nothing() {
    // The line pipeline has no near-plane guard of its own; this is the check that the
    // projection is doing the clipping, i.e. that the wiring cannot paint a stripe across the
    // screen when the player turns around with a block still named in `HitResult`.
    let Some(mut gpu) = gpu() else { return };
    let cam = camera();
    let stone = fixture::state_of("minecraft:stone");

    let empty = frame_with(&mut gpu, &cam, None);
    let behind = frame_with(
        &mut gpu,
        &cam,
        Some(SelectionBox::for_block(BlockPos::new(8, 17, -8), stone)),
    );
    assert_eq!(
        changed_pixels(&empty, &behind).len(),
        0,
        "a block behind the camera must be clipped away entirely"
    );
}
