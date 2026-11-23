//! Headless GPU readback for waypoint labels.
//!
//! A waypoint label has to read through walls. The client draws it in
//! `NameplatePass::ThroughWalls` of its own `NameplatePhase`, registered after the world at
//! order 645. This file draws an opaque wall at `PhaseOrder::ENTITIES`, a red label behind the
//! wall, and reads the frame back with `Gpu::read_pixels`. The control is a red entity quad at
//! the label's depth, which the same wall hides.
//!
//! The font is synthetic (every glyph cell fully inked), so no vanilla jar is needed. Skips
//! loudly when there is no wgpu adapter.

use glam::DVec3;
use mc_render::entity::{EntityPhase, EntityVertex, NameplatePhase};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase};
use mc_ui::font::{FontData, FontRenderer};
use mc_ui::waypoint_label::{label_matrix, LabelScratch, WaypointLabel};

const W: u32 = 256;
const H: u32 = 256;

/// Where the client registers waypoint labels: after the fog phase (640), before the hand (650).
const LABEL_ORDER: PhaseOrder = PhaseOrder(645);

/// No red at all, so "red pixels" can only come from the label or the control quad.
const CLEAR: wgpu::Color = wgpu::Color {
    r: 0.0,
    g: 0.0,
    b: 0.4,
    a: 1.0,
};

fn gpu() -> Option<Gpu> {
    match Gpu::headless(
        W,
        H,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => Some(gpu),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            None
        }
        Err(error) => panic!("headless context creation failed: {error}"),
    }
}

/// A font whose every glyph is a solid 8x8 block, so each drawn character is a filled box.
fn inked_font() -> FontRenderer {
    let sheet = image::RgbaImage::from_pixel(128, 128, image::Rgba([255, 255, 255, 255]));
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(sheet)
        .write_to(&mut png, image::ImageFormat::Png)
        .expect("encode the synthetic sheet");
    let data = FontData::from_parts(&png.into_inner(), &[0u8; 0x1_0000]).expect("synthetic font");
    FontRenderer::new(data, false, false)
}

/// Looking along +Z from the origin.
fn camera() -> Camera {
    let mut camera = Camera::default();
    camera.set_viewport(W, H);
    camera.update();
    camera
}

/// A camera-facing square of entity geometry at depth `z`, `half` blocks each way, one colour.
fn quad_phase(gpu: &Gpu, z: f32, half: f32, rgba: [u8; 4]) -> EntityPhase {
    let mut phase = EntityPhase::new(gpu);
    phase.upload_texture(gpu, "solid", &rgba, 1, 1);
    let corners = [
        (-half, -half),
        (half, -half),
        (half, half),
        (-half, -half),
        (half, half),
        (-half, half),
    ];
    let verts: Vec<EntityVertex> = corners
        .into_iter()
        .map(|(x, y)| EntityVertex {
            pos: [x, y, z],
            uv: [0.5, 0.5],
            normal: [0.0, 0.0, -1.0],
            ..EntityVertex::default()
        })
        .collect();
    phase.begin();
    phase.push("solid", &verts);
    phase.finish(gpu);
    phase
}

/// A label phase holding one red waypoint label anchored at `anchor` (camera-relative).
fn label_phase(
    gpu: &Gpu,
    font: &mut FontRenderer,
    camera: &Camera,
    anchor: DVec3,
    distance: bool,
) -> NameplatePhase {
    let mut phase = NameplatePhase::new(gpu);
    mc_ui::nameplate::upload_ascii_font(&mut phase, gpu, font.data());
    let mut label = WaypointLabel::default();
    label.update(font, "Base", 0xFF_0000, distance.then(|| anchor.length()));
    // 200%, so the text is comfortably more than a few pixels tall at 256x256.
    let matrix = label_matrix(anchor, camera.yaw, camera.pitch, 2.0).expect("far enough");
    phase.begin();
    label.emit(matrix, &mut LabelScratch::default(), &mut phase);
    phase.finish(gpu);
    phase
}

fn render(
    gpu: &mut Gpu,
    camera: &Camera,
    phases: Vec<(PhaseOrder, Box<dyn RenderPhase>)>,
) -> Vec<u8> {
    let mut list = PhaseList::new();
    list.clear_color = CLEAR;
    for (order, phase) in phases {
        list.insert(order, phase);
    }
    let frame = gpu.acquire().expect("acquire offscreen frame");
    list.render(gpu, frame.view(), camera, 0.0);
    frame.present(&gpu.queue);
    gpu.read_pixels().expect("read back")
}

fn is_red(p: &[u8]) -> bool {
    p[0] >= 100 && p[1] < 50 && p[2] < 50
}

fn red_pixels(px: &[u8]) -> usize {
    px.chunks(4).filter(|p| is_red(p)).count()
}

/// The bounding box of the red pixels, `(x0, y0, x1, y1)`.
fn red_box(px: &[u8]) -> Option<(u32, u32, u32, u32)> {
    let mut found: Option<(u32, u32, u32, u32)> = None;
    for (i, p) in px.chunks(4).enumerate() {
        if is_red(p) {
            let (x, y) = (i as u32 % W, i as u32 / W);
            found = Some(match found {
                None => (x, y, x, y),
                Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
            });
        }
    }
    found
}

fn pixel(px: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2], px[i + 3]]
}

/// The load-bearing one: a label 30 blocks out, behind an opaque wall 10 blocks out, still
/// shows its red text and its red marker on the exact spot, while a red entity quad at the same
/// depth is hidden by the same wall.
#[test]
fn a_waypoint_label_shows_through_a_wall_that_hides_an_entity() {
    let Some(mut gpu) = gpu() else { return };
    let mut font = inked_font();
    let camera = camera();
    let anchor = DVec3::new(0.0, 0.0, 30.0);
    let wall = |gpu: &Gpu| -> Box<dyn RenderPhase> {
        Box::new(quad_phase(gpu, 10.0, 8.0, [0, 0, 255, 255]))
    };

    // Control, positive half: the red quad at the label's depth is visible with no wall.
    let control: Box<dyn RenderPhase> = Box::new(quad_phase(&gpu, 30.0, 2.0, [255, 0, 0, 255]));
    let px = render(&mut gpu, &camera, vec![(PhaseOrder::ENTITIES, control)]);
    let control_open = red_pixels(&px);
    assert!(
        control_open > 200,
        "the control quad must be visible on its own: {control_open}"
    );

    // Control, negative half: the wall in front of it depth-rejects every red pixel.
    let phases = vec![
        (PhaseOrder::ENTITIES, wall(&gpu)),
        (
            PhaseOrder::ENTITIES,
            Box::new(quad_phase(&gpu, 30.0, 2.0, [255, 0, 0, 255])) as Box<dyn RenderPhase>,
        ),
    ];
    let px = render(&mut gpu, &camera, phases);
    assert_eq!(
        red_pixels(&px),
        0,
        "the wall must hide an ordinary entity at that depth"
    );
    assert!(
        pixel(&px, 4, 4)[2] > 150,
        "the wall fills the frame: {:?}",
        pixel(&px, 4, 4)
    );

    // The label at the same depth, behind the same wall: drawn anyway.
    let labels = label_phase(&gpu, &mut font, &camera, anchor, true);
    let phases = vec![
        (PhaseOrder::ENTITIES, wall(&gpu)),
        (LABEL_ORDER, Box::new(labels) as Box<dyn RenderPhase>),
    ];
    let px = render(&mut gpu, &camera, phases);
    let through = red_pixels(&px);
    assert!(
        through > 100,
        "the label's red name must show through the wall: {through}"
    );
    // Opaque, not a ghost: the ink is pure waypoint red over the blue wall.
    assert!(
        px.chunks(4).any(|p| p[..3] == [255, 0, 0]),
        "vertex alpha 1 draws the waypoint colour exactly"
    );
    // The marker diamond sits on the anchor, the centre of the frame.
    assert!(
        is_red(&pixel(&px, 128, 128)),
        "marker at the anchor: {:?}",
        pixel(&px, 128, 128)
    );
    // ...and the label is stacked above it: red text well above the centre row, and nothing
    // red below the diamond (about 4 px tall each way at this size).
    let (_, top, _, bottom) = red_box(&px).expect("red pixels");
    assert!(
        top < 120 && bottom <= 134,
        "label above the marker: rows {top}..={bottom}"
    );
}

/// Constant size on screen: a label 5000 blocks away (far past this camera's 256-block far
/// plane) draws exactly where and as large as one 10 blocks away on the same ray.
#[test]
fn a_far_label_is_not_clipped_and_keeps_its_size() {
    let Some(mut gpu) = gpu() else { return };
    let mut font = inked_font();
    let camera = camera();
    let direction = DVec3::new(0.2, 0.1, 1.0).normalize();
    let boxes: Vec<_> = [10.0, 5000.0]
        .into_iter()
        .map(|d| {
            let labels: Box<dyn RenderPhase> =
                Box::new(label_phase(&gpu, &mut font, &camera, direction * d, false));
            let px = render(&mut gpu, &camera, vec![(LABEL_ORDER, labels)]);
            red_box(&px).unwrap_or_else(|| panic!("the label at {d} blocks was not drawn"))
        })
        .collect();
    let (near, far) = (boxes[0], boxes[1]);
    let close = |a: u32, b: u32| a.abs_diff(b) <= 1;
    assert!(
        close(near.0, far.0)
            && close(near.1, far.1)
            && close(near.2, far.2)
            && close(near.3, far.3),
        "10 blocks: {near:?}, 5000 blocks: {far:?}"
    );
    assert!(near.2 - near.0 > 20, "a real label, not a speck: {near:?}");
}
