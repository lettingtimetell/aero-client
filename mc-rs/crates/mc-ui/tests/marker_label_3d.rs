//! Headless GPU readback for marker pins.
//!
//! A marker has to read through walls at any distance. The client draws it in
//! `NameplatePass::ThroughWalls` of its own `NameplatePhase` at order 645, after the world. This
//! file draws an opaque wall at `PhaseOrder::ENTITIES`, a red marker behind it, and reads the
//! frame back with `Gpu::read_pixels`; the control is a red entity quad at the marker's depth,
//! which the same wall hides.
//!
//! The font is synthetic (every glyph cell fully inked), so no vanilla jar is needed. Skips
//! loudly when there is no wgpu adapter.

use glam::DVec3;
use mc_render::entity::{EntityPhase, EntityVertex, NameplatePhase};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase};
use mc_ui::font::{FontData, FontRenderer};
use mc_ui::marker_label::MarkerLabel;
use mc_ui::waypoint_label::{label_matrix, LabelScratch};

const W: u32 = 256;
const H: u32 = 256;

/// Where the client registers marker pins: the waypoint labels' slot, after the fog (640).
const LABEL_ORDER: PhaseOrder = PhaseOrder(645);

/// No red at all, so "red pixels" can only come from the pin or the control quad.
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

/// A phase holding one red marker anchored at `anchor` (camera-relative), at 200% size.
fn marker_phase(
    gpu: &Gpu,
    font: &mut FontRenderer,
    camera: &Camera,
    anchor: DVec3,
) -> NameplatePhase {
    let mut phase = NameplatePhase::new(gpu);
    mc_ui::nameplate::upload_ascii_font(&mut phase, gpu, font.data());
    let mut label = MarkerLabel::default();
    label.update(font, anchor.length());
    let matrix = label_matrix(anchor, camera.yaw, camera.pitch, 2.0).expect("far enough");
    phase.begin();
    label.emit(matrix, 0xFF_0000, &mut LabelScratch::default(), &mut phase);
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

fn is_white(p: &[u8]) -> bool {
    p[0] > 200 && p[1] > 200 && p[2] > 200
}

fn count(px: &[u8], test: fn(&[u8]) -> bool) -> usize {
    px.chunks(4).filter(|p| test(p)).count()
}

/// The bounding box of the pixels passing `test`, `(x0, y0, x1, y1)`.
fn bounds(px: &[u8], test: fn(&[u8]) -> bool) -> Option<(u32, u32, u32, u32)> {
    let mut found: Option<(u32, u32, u32, u32)> = None;
    for (i, p) in px.chunks(4).enumerate() {
        if test(p) {
            let (x, y) = (i as u32 % W, i as u32 / W);
            found = Some(match found {
                None => (x, y, x, y),
                Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
            });
        }
    }
    found
}

/// The load-bearing one: a marker 30 blocks out, behind an opaque wall 10 blocks out, still
/// shows its red pin pointing at the exact spot and its white distance above, while a red entity
/// quad at the same depth is hidden by the same wall.
#[test]
fn a_marker_shows_through_a_wall_that_hides_an_entity() {
    let Some(mut gpu) = gpu() else { return };
    let mut font = inked_font();
    let camera = camera();
    let anchor = DVec3::new(0.0, 0.0, 30.0);
    let wall = |gpu: &Gpu| -> Box<dyn RenderPhase> {
        Box::new(quad_phase(gpu, 10.0, 8.0, [0, 0, 255, 255]))
    };

    // Control: the red quad at the marker's depth shows alone and hides behind the wall.
    let control: Box<dyn RenderPhase> = Box::new(quad_phase(&gpu, 30.0, 2.0, [255, 0, 0, 255]));
    let px = render(&mut gpu, &camera, vec![(PhaseOrder::ENTITIES, control)]);
    assert!(
        count(&px, is_red) > 200,
        "the control quad is visible alone"
    );
    let phases = vec![
        (PhaseOrder::ENTITIES, wall(&gpu)),
        (
            PhaseOrder::ENTITIES,
            Box::new(quad_phase(&gpu, 30.0, 2.0, [255, 0, 0, 255])) as Box<dyn RenderPhase>,
        ),
    ];
    let px = render(&mut gpu, &camera, phases);
    assert_eq!(count(&px, is_red), 0, "the wall hides an ordinary entity");

    // The marker at the same depth, behind the same wall: drawn anyway, at full strength.
    let markers = marker_phase(&gpu, &mut font, &camera, anchor);
    let phases = vec![
        (PhaseOrder::ENTITIES, wall(&gpu)),
        (LABEL_ORDER, Box::new(markers) as Box<dyn RenderPhase>),
    ];
    let px = render(&mut gpu, &camera, phases);
    let red = count(&px, is_red);
    assert!(red > 40, "the pin shows through the wall: {red} red pixels");
    assert!(
        px.chunks(4).any(|p| p[..3] == [255, 0, 0]),
        "vertex alpha 1 draws the marker colour exactly"
    );
    // The pin points down at the anchor (the frame centre): it ends at the centre row and sits
    // right above it, centred.
    let (x0, y0, x1, y1) = bounds(&px, is_red).expect("red pixels");
    assert!(
        (126..=130).contains(&((x0 + x1) / 2)),
        "centred on the anchor: columns {x0}..={x1}"
    );
    assert!(
        (126..=130).contains(&y1) && y0 < y1 && y0 > 100,
        "above the anchor, its tip on it: rows {y0}..={y1}"
    );
    // The distance is drawn in white above the pin.
    let (_, _, _, text_bottom) = bounds(&px, is_white).expect("white distance text");
    assert!(
        text_bottom < y0,
        "text rows end at {text_bottom}, the pin starts at {y0}"
    );
    assert!(count(&px, is_white) > 50, "the text is readable");
}

/// Constant size on screen: a marker 5000 blocks away (far past this camera's far plane) draws
/// exactly where and as large as one 10 blocks away on the same ray.
#[test]
fn a_far_marker_is_not_clipped_and_keeps_its_size() {
    let Some(mut gpu) = gpu() else { return };
    let mut font = inked_font();
    let camera = camera();
    let direction = DVec3::new(0.2, 0.1, 1.0).normalize();
    let boxes: Vec<_> = [10.0, 5000.0]
        .into_iter()
        .map(|d| {
            let markers: Box<dyn RenderPhase> =
                Box::new(marker_phase(&gpu, &mut font, &camera, direction * d));
            let px = render(&mut gpu, &camera, vec![(LABEL_ORDER, markers)]);
            bounds(&px, is_red).unwrap_or_else(|| panic!("the pin at {d} blocks was not drawn"))
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
}
