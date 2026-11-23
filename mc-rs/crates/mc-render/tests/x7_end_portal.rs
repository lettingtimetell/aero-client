//! Bug report: "you shouldn't be able to see end portal void
//! blocks when viewed from the bottom".
//!
//! java: `EntityRenderer.renderWorldPass` enables back-face culling and
//! `TileEntityEndPortalRenderer` never disables it; its single quad at `y + 0.75` is wound
//! counter-clockwise seen from above. So the starfield shows from above and nothing from below.

use mc_render::entity::TilePhase;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

const SIDE: u32 = 96;

fn headless() -> Option<Gpu> {
    match Gpu::headless(
        SIDE,
        SIDE,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => Some(gpu),
        Err(RenderError::NoAdapter) => {
            eprintln!("skip: no wgpu adapter available on this machine");
            None
        }
        Err(error) => panic!("headless context creation failed: {error}"),
    }
}

/// Pixels that are not the clear colour after drawing one end portal tile whose block origin
/// is `rel` from a camera with `pitch` (vanilla degrees: +90 looks straight down).
fn drawn_pixels(gpu: &mut Gpu, rel: glam::Vec3, pitch: f32) -> usize {
    let mut phase = TilePhase::new(gpu);
    // No pack files: both layers bind vanilla's missing texture, which is plenty to see.
    assert!(phase.load_end_portal_textures(gpu, &mc_assets::ResourceManager::new()));
    phase.begin();
    assert!(phase.push_end_portal(rel, 123_456));
    phase.finish(gpu);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::ENTITIES, Box::new(phase));
    let mut camera = Camera {
        pitch,
        ..Camera::default()
    };
    camera.set_viewport(SIDE, SIDE);
    camera.update();
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &camera, 0.0);
    frame.present(&gpu.queue);
    let pixels = gpu.read_pixels().expect("read back the frame");
    pixels
        .chunks_exact(4)
        .filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0)
        .count()
}

#[test]
fn the_end_portal_shows_from_above_and_not_from_below() {
    let Some(mut gpu) = headless() else {
        return;
    };
    // Eye 2.25 blocks above the surface, looking down: the tile fills the middle of the view.
    let above = drawn_pixels(&mut gpu, glam::Vec3::new(-0.5, -3.0, -0.5), 90.0);
    // Eye 2.25 blocks below the same surface, looking up at it.
    let below = drawn_pixels(&mut gpu, glam::Vec3::new(-0.5, 1.5, -0.5), -90.0);
    eprintln!("end portal pixels: above {above}, below {below}");
    assert!(
        above > (SIDE * SIDE / 20) as usize,
        "seen from above the starfield covers the tile ({above} px)"
    );
    assert_eq!(below, 0, "seen from below there is nothing to see");
}
