//! Scene mode drawing straight into a copyable frame target.
//!
//! `PhaseList` used to render every scene-mode frame (any enabled phase asking for
//! `SceneTextures`: TAA, the shader packs, menu blur, motion blur, Super Secret Settings) into an
//! intermediate target and blit it onto the frame target at the end. Where the frame target can
//! be copied from (the window's swap chain requests `COPY_SRC` for screenshots; headless targets
//! always have it), it now renders into the target itself and snapshots it by copy: no
//! intermediate target, no end-of-frame blit. These tests prove the frame is the same bytes either
//! way, over several scene phases in one frame and over several frames, and that the
//! intermediate target is only built when the direct path is off.

use mc_assets::ResourceManager;
use mc_render::post::{OverlayKind, OverlayPhase, PostChain, PostPhase, ShaderGroupSpec};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, HEADLESS_FORMAT};

const SIZE: (u32, u32) = (40, 24);

fn headless() -> Option<Gpu> {
    match Gpu::headless(
        SIZE.0,
        SIZE.1,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => Some(gpu),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

/// A one-pass post chain over `minecraft:main` (reads the snapshot, writes the whole frame).
fn chain(gpu: &Gpu, program: &str, uniforms: &str) -> PostChain {
    let spec = ShaderGroupSpec::parse(&format!(
        r#"{{"targets": [], "passes": [{{"name":"{program}","intarget":"minecraft:main",
            "outtarget":"minecraft:main","uniforms":[{uniforms}]}}]}}"#
    ))
    .expect("spec parses");
    PostChain::new(gpu, &ResourceManager::new(), &spec, HEADLESS_FORMAT, SIZE)
}

/// Render `frames` frames of a list with three blending phases and two scene phases between
/// them, and return the last frame's bytes and whether the intermediate target was built.
fn render(direct: bool, frames: usize) -> (Vec<u8>, bool) {
    let mut gpu = headless().expect("adapter checked by caller");
    let mut phases = PhaseList::new();
    phases.set_direct_scene(direct);
    phases.clear_color = wgpu::Color {
        r: 0.13,
        g: 0.37,
        b: 0.71,
        a: 1.0,
    };
    // Before the first scene phase: a blended tint.
    let mut water = OverlayPhase::new(&gpu);
    water.kind = OverlayKind::Water;
    water.gamma = 0.4;
    phases.insert(PhaseOrder::POST, Box::new(water));
    // Scene phase 1: a partial invert of the frame so far.
    let mut invert = PostPhase::new(&gpu);
    invert.set_chain(chain(
        &gpu,
        "invert",
        r#"{"name":"InverseAmount","values":[0.8]}"#,
    ));
    phases.insert(PhaseOrder::POST, Box::new(invert));
    // Between the scene phases: another blended tint (it must land on the frame phase 1 wrote):
    // the flat portal stand-in.
    let mut portal = OverlayPhase::new(&gpu);
    portal.kind = OverlayKind::Portal;
    portal.portal_time = 0.6;
    phases.insert(PhaseOrder::UI, Box::new(portal));
    // Scene phase 2: reads a snapshot that must include everything above.
    let mut invert_again = PostPhase::new(&gpu);
    invert_again.set_chain(chain(
        &gpu,
        "invert",
        r#"{"name":"InverseAmount","values":[0.3]}"#,
    ));
    phases.insert(PhaseOrder::UI, Box::new(invert_again));
    assert!(phases.scene_mode());

    let mut pixels = Vec::new();
    for _ in 0..frames {
        let frame = gpu.acquire().expect("acquire");
        phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
        pixels = gpu.read_pixels().expect("readback");
    }
    (pixels, phases.scene_intermediate_allocated())
}

#[test]
fn drawing_straight_into_the_target_gives_the_same_bytes_as_the_intermediate_path() {
    if headless().is_none() {
        return;
    }
    let (via_intermediate, built_old) = render(false, 3);
    let (direct, built_new) = render(true, 3);
    assert_eq!(via_intermediate.len(), (SIZE.0 * SIZE.1 * 4) as usize);
    assert!(
        built_old,
        "with the direct path off the intermediate target is used"
    );
    assert!(
        !built_new,
        "a copyable target must not get an intermediate target or a blit"
    );
    assert_eq!(
        direct, via_intermediate,
        "drawing straight into the target changed the frame"
    );
    // Not vacuous: the scene phases really changed the frame (an untouched clear would also
    // compare equal between the two paths).
    let clear = [33u8, 94, 181];
    let first = &direct[..3];
    assert!(
        first.iter().zip(clear).any(|(a, b)| a.abs_diff(b) > 8),
        "the post phases did not run: {first:?}"
    );
}

#[test]
fn a_target_without_copy_src_keeps_the_intermediate_path() {
    // The negative control for the gate: the same list rendered into a texture that cannot be
    // copied from must take the old path (and still render).
    let Some(gpu) = headless() else {
        return;
    };
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("nw-f4 render-only target"),
        size: wgpu::Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut phases = PhaseList::new();
    phases.set_direct_scene(true);
    let mut invert = PostPhase::new(&gpu);
    invert.set_chain(chain(&gpu, "invert", ""));
    phases.insert(PhaseOrder::POST, Box::new(invert));
    phases.render(&gpu, &view, &Camera::default(), 0.0);
    assert!(phases.scene_intermediate_allocated());
}
