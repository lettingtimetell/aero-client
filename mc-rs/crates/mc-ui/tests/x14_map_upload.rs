//! A GUI map texture (`dynamic/map_<id>`, drawn for maps in an
//! open container or the creative inventory) is uploaded only when its pixels change.
//!
//! The client calls `UiRenderer::upload_dynamic_map` for every visible map every frame; it used
//! to copy the image twice and re-upload it each time. A repeat of the resident bytes is now a
//! no-op. This checks the upload counters and that the skipped uploads leave exactly the texels a
//! fresh renderer shows, including after a live anaglyph toggle (which replays the retained
//! bytes through the new transform).

use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder};
use mc_ui::map::{dynamic_texture_key, MAP_TEXTURE_BYTES, MAP_TEXTURE_SIZE};
use mc_ui::widget::{DrawList, ScaledResolution, UiPhase};

const W: u32 = 160;
const H: u32 = 160;
const MAP_ID: i32 = 7;

/// An opaque 128x128 image (map pixels are opaque once explored).
fn image(seed: u32) -> Vec<u8> {
    (0..MAP_TEXTURE_BYTES as u32)
        .map(|i| {
            if i % 4 == 3 {
                255
            } else {
                (i.wrapping_mul(2_654_435_761)
                    .wrapping_add(seed.wrapping_mul(40_503))
                    >> 13) as u8
            }
        })
        .collect()
}

fn upload_calls(phase: &UiPhase) -> u64 {
    phase
        .diagnostic_stats()
        .expect("diagnostics enabled")
        .texture_upload_calls
}

/// Draw the map at 1:1 in the top-left corner and read the frame back.
fn render(gpu: &mut Gpu, mut phase: UiPhase) -> Vec<u8> {
    let mut list = DrawList::new();
    let size = MAP_TEXTURE_SIZE as i32;
    list.blit_scaled(
        &dynamic_texture_key(MAP_ID),
        0,
        0,
        0.0,
        0.0,
        size,
        size,
        size,
        size,
        size as f32,
        size as f32,
    );
    phase.set_resolution(ScaledResolution::new(W, H, 1, false));
    phase.submit(list);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::UI, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    gpu.read_pixels().expect("readback")
}

#[test]
fn a_gui_map_uploads_only_when_its_pixels_change() {
    let mut gpu = Gpu::headless(W, H, GpuOptions::default()).expect("needs a GPU");
    let (a, b) = (image(1), image(2));

    // The renderer that sees repeated uploads, as in a container held open for many frames.
    let mut repeated = UiPhase::new(&gpu, W, H, 1, false);
    repeated.set_diagnostics_enabled(true);
    repeated.begin_diagnostic_frame();
    assert!(repeated.renderer.upload_dynamic_map(&gpu, MAP_ID, &a));
    assert_eq!(
        upload_calls(&repeated),
        1,
        "first sight creates and uploads the texture"
    );
    for _ in 0..3 {
        assert!(repeated.renderer.upload_dynamic_map(&gpu, MAP_ID, &a));
    }
    assert_eq!(
        upload_calls(&repeated),
        1,
        "unchanged pixels are not uploaded again"
    );
    assert!(repeated.renderer.upload_dynamic_map(&gpu, MAP_ID, &b));
    assert_eq!(upload_calls(&repeated), 2, "an S34 change uploads once");
    assert!(repeated.renderer.upload_dynamic_map(&gpu, MAP_ID, &b));
    assert_eq!(upload_calls(&repeated), 2);
    repeated.set_anaglyph_with_gpu(&gpu, true);
    let after_toggle = upload_calls(&repeated);
    assert!(repeated.renderer.upload_dynamic_map(&gpu, MAP_ID, &b));
    assert_eq!(
        upload_calls(&repeated),
        after_toggle,
        "the toggle already replayed the map through the new transform"
    );
    assert!(!repeated
        .renderer
        .upload_dynamic_map(&gpu, MAP_ID, &b[..MAP_TEXTURE_BYTES - 4]));
    let shown = render(&mut gpu, repeated);

    // A fresh renderer that uploads `b` once under the same transform.
    let mut fresh = UiPhase::new_with_anaglyph(&gpu, W, H, 1, false, true);
    assert!(fresh.renderer.upload_dynamic_map(&gpu, MAP_ID, &b));
    let expected = render(&mut gpu, fresh);
    assert!(
        shown == expected,
        "skipped uploads must leave the fresh renderer's texels"
    );

    // The comparison discriminates: the other image, or the untransformed one, looks different.
    let mut stale = UiPhase::new_with_anaglyph(&gpu, W, H, 1, false, true);
    assert!(stale.renderer.upload_dynamic_map(&gpu, MAP_ID, &a));
    assert!(
        render(&mut gpu, stale) != expected,
        "image a must render differently from b"
    );
    let mut plain = UiPhase::new(&gpu, W, H, 1, false);
    assert!(plain.renderer.upload_dynamic_map(&gpu, MAP_ID, &b));
    assert!(
        render(&mut gpu, plain) != expected,
        "the anaglyph transform must be visible"
    );
}
