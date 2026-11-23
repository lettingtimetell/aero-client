//! Headless GPU tests for **entity nameplate text**.
//!
//! S15 proved a glyph lands on exactly the `ascii.png` texels vanilla samples in 2-D screen
//! space (`tests/headless_ui.rs`). This file does the same thing in **3-D**: the glyphs go
//! through `mc_ui::nameplate` → `Nameplate::glyph_vertices` → `NameplatePhase`'s billboard
//! pipelines, are rasterised by a real adapter, and are read back with
//! `mc_render::Gpu::read_pixels`.
//!
//! # How "texel-exact in 3-D" is made checkable
//!
//! At `view_yaw = view_pitch = 0` the billboard is a plane of constant camera-relative `z`, so
//! all four corners of a glyph quad share one projected `w` and the perspective divide is a
//! single uniform scale. That makes the label-pixel → framebuffer-pixel map *affine*, and it
//! is derived here from the projected corners of the geometry the seam actually produced —
//! not assumed. Every framebuffer pixel strictly inside the quad is then inverse-mapped to a
//! texel of `ascii.png` and compared against that texel's alpha.
//!
//! The plate's distance is solved for so one label pixel is exactly [`SCALE_PX`] framebuffer
//! pixels, which keeps the boundary-straddling pixels to a small, explicitly skipped band.
//!
//! Skips (loudly) when there is no wgpu adapter or no vanilla jar.

use std::path::{Path, PathBuf};

use glam::{Vec3, Vec4};
use mc_assets::{ResourceManager, ZipResourcePack};
use mc_render::entity::nameplate::NAMEPLATE_SCALE;
use mc_render::entity::{EntityPhase, EntityVertex, Nameplate, NameplatePhase};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use mc_ui::font::{FontData, FontRenderer};
use mc_ui::nameplate::{self, NameplateRequest};

/// Square, so the projection's x and y scales coincide and one number describes both.
const W: u32 = 256;
const H: u32 = 256;

/// Framebuffer pixels per label pixel. The plate's distance is solved for this.
const SCALE_PX: f32 = 8.0;

/// Clear colour: no red at all, so the red channel is a clean "is this glyph ink?" probe —
/// `ascii.png` is white ink, the backdrop is black, and the wall used by the depth tests is
/// blue.
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
        Ok(g) => Some(g),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

fn jars_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn assets() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(rm)
}

fn vanilla_font() -> Option<FontRenderer> {
    let rm = assets()?;
    match FontData::load(&rm) {
        Ok(data) => Some(FontRenderer::new(data, false, false)),
        Err(e) => {
            eprintln!("SKIP: vanilla font unavailable: {e}");
            None
        }
    }
}

fn camera() -> Camera {
    let mut cam = Camera::default();
    cam.set_viewport(W, H);
    cam.update();
    cam
}

/// Distance at which one label pixel projects to exactly `SCALE_PX` framebuffer pixels.
///
/// A point `x` blocks off-axis at depth `z` lands at ndc `proj00 * x / z`, i.e.
/// `proj00 * x * W / (2z)` pixels from centre; one label pixel is [`NAMEPLATE_SCALE`] blocks.
fn plate_distance(cam: &Camera) -> f32 {
    let proj00 = cam.proj.x_axis.x;
    proj00 * W as f32 * NAMEPLATE_SCALE / (2.0 * SCALE_PX)
}

/// A label whose anchor — label-space `(0, 0)` — is exactly the middle of the framebuffer.
///
/// `height = 0` and `rel.y = -NAMEPLATE_LIFT` cancel the `y + height + 0.5` vanilla applies,
/// which puts the anchor on the camera's forward ray.
fn centred_plate(text: &str, cam: &Camera) -> Nameplate {
    Nameplate {
        text: text.to_owned(),
        rel: Vec3::new(0.0, -0.5, plate_distance(cam)),
        height: 0.0,
        child: false,
        sneaking: false,
    }
}

/// Project a camera-relative position to framebuffer pixels, exactly as the vertex shader plus
/// the rasteriser does: `view_proj`, perspective divide, NDC → viewport (y flipped).
fn project(cam: &Camera, pos: [f32; 3]) -> (f32, f32, f32) {
    let clip = cam.view_proj() * Vec4::new(pos[0], pos[1], pos[2], 1.0);
    let ndc = clip.truncate() / clip.w;
    (
        (ndc.x * 0.5 + 0.5) * W as f32,
        (0.5 - ndc.y * 0.5) * H as f32,
        clip.w,
    )
}

fn pixel(px: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2], px[i + 3]]
}

/// One frame: an optional opaque wall at `wall_z`, then the nameplate phase.
fn render(
    gpu: &mut Gpu,
    cam: &Camera,
    mut plate_phase: NameplatePhase,
    wall_z: Option<f32>,
) -> Vec<u8> {
    plate_phase.finish(gpu);
    let mut phases = PhaseList::new();
    phases.clear_color = CLEAR;
    if let Some(z) = wall_z {
        let mut wall = EntityPhase::new(gpu);
        wall.upload_texture(gpu, "wall", &[0u8, 0, 255, 255], 1, 1);
        // Big enough to fill the frame at this depth: the visible half-width is z / proj00.
        let r = 4.0f32;
        let quad = [(-r, -r), (r, -r), (r, r), (-r, -r), (r, r), (-r, r)];
        let verts: Vec<EntityVertex> = quad
            .into_iter()
            .map(|(x, y)| EntityVertex {
                pos: [x, y, z],
                uv: [0.5, 0.5],
                normal: [0.0, 0.0, -1.0],
                // A02-07 added `EntityVertex::color`; `Default` is opaque white.
                ..EntityVertex::default()
            })
            .collect();
        wall.begin();
        wall.push("wall", &verts);
        wall.finish(gpu);
        phases.insert(PhaseOrder::ENTITIES, Box::new(wall));
    }
    phases.insert(PhaseOrder::UI, Box::new(plate_phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), cam, 0.0);
    frame.present(&gpu.queue);
    gpu.read_pixels().expect("read back")
}

/// Build a label's geometry and push it into a fresh phase with the real `ascii.png` bound.
fn phase_for(
    gpu: &Gpu,
    font: &mut FontRenderer,
    plate: &Nameplate,
    cam: &Camera,
    distance_sq: f32,
) -> (NameplatePhase, mc_ui::nameplate::NameplateGeometry) {
    let mut phase = NameplatePhase::new(gpu);
    nameplate::upload_ascii_font(&mut phase, gpu, font.data());
    let geometry = nameplate::build(
        font,
        &NameplateRequest {
            plate,
            view_yaw: cam.yaw,
            view_pitch: cam.pitch,
            distance_sq,
            score: None,
        },
    )
    .expect("label is in range");
    phase.begin();
    geometry.submit(&mut phase);
    (phase, geometry)
}

/// The load-bearing one. Every framebuffer pixel strictly inside the projected `'A'` quad must
/// carry the colour the `ascii.png` texel it samples calls for — white where the sheet is
/// inked, backdrop-black where it is not.
#[test]
fn a_nameplate_glyph_lands_on_exactly_the_ascii_texels_vanilla_samples() {
    let Some(mut gpu) = gpu() else { return };
    let Some(mut font) = vanilla_font() else {
        return;
    };
    let sheet: Vec<u8> = font.data().ascii_rgba().to_vec();
    let (sheet_w, sheet_h) = font.data().ascii_size();
    assert_eq!((sheet_w, sheet_h), (128, 128), "vanilla's ascii.png");

    let cam = camera();
    let plate = centred_plate("A", &cam);
    let (phase, geometry) = phase_for(&gpu, &mut font, &plate, &cam, 25.0);
    // The opaque `InFront` pass is the one whose ink must read as pure white.
    let batch = geometry
        .glyphs
        .iter()
        .find(|b| b.pass == mc_render::entity::NameplatePass::InFront)
        .expect("a depth-tested glyph pass");
    assert_eq!(batch.verts.len(), 6, "one glyph, one quad, two triangles");
    let px = render(&mut gpu, &cam, phase, None);

    // `glyph_vertices` emits the GL_TRIANGLE_STRIP expansion [TL, BL, TR, TR, BL, BR].
    let (tl, br) = (batch.verts[0], batch.verts[5]);
    let (x0, y0, w0) = project(&cam, tl.pos);
    let (x1, y1, w1) = project(&cam, br.pos);
    assert!(
        (w0 - w1).abs() < 1e-4,
        "the billboard must be parallel to the image plane for this map to be affine: {w0} vs {w1}"
    );
    assert!(
        x1 > x0 && y1 > y0,
        "TL is up-left of BR: {x0},{y0} -> {x1},{y1}"
    );
    // Correct scale: 7.99 label pixels tall, SCALE_PX framebuffer pixels each.
    let got_scale = (y1 - y0) / 7.99;
    assert!(
        (got_scale - SCALE_PX).abs() < 0.05,
        "expected {SCALE_PX} framebuffer px per label px, got {got_scale}"
    );

    let (u0, v0) = (tl.uv[0], tl.uv[1]);
    let (u1, v1) = (br.uv[0], br.uv[1]);
    let mut inked = 0u32;
    let mut blank = 0u32;
    for py in (y0.floor() as u32)..=(y1.ceil() as u32) {
        for px_x in (x0.floor() as u32)..=(x1.ceil() as u32) {
            let fx = (px_x as f32 + 0.5 - x0) / (x1 - x0);
            let fy = (py as f32 + 0.5 - y0) / (y1 - y0);
            if !(0.0..=1.0).contains(&fx) || !(0.0..=1.0).contains(&fy) {
                continue;
            }
            let u = (u0 + fx * (u1 - u0)) * sheet_w as f32;
            let v = (v0 + fy * (v1 - v0)) * sheet_h as f32;
            // Skip the sub-pixel band where a texel boundary falls inside this framebuffer
            // pixel: which side the GPU lands on there is a rounding question, not a
            // correctness one. At SCALE_PX = 8 this excludes well under a pixel per texel edge.
            if (u.fract() < 0.05 || u.fract() > 0.95) || (v.fract() < 0.05 || v.fract() > 0.95) {
                continue;
            }
            let (tx, ty) = (u.floor() as u32, v.floor() as u32);
            let alpha = sheet[((ty * sheet_w + tx) * 4 + 3) as usize];
            let got = pixel(&px, px_x, py);
            if alpha == 255 {
                inked += 1;
                assert_eq!(
                    got,
                    [255, 255, 255, 255],
                    "texel ({tx}, {ty}) is inked; screen ({px_x}, {py}) should be white"
                );
            } else {
                blank += 1;
                assert_eq!(
                    got[0], 0,
                    "texel ({tx}, {ty}) is empty; screen ({px_x}, {py}) must have no ink: {got:?}"
                );
            }
        }
    }
    // 'A' is 5 texels wide by 8 tall at SCALE_PX = 8, so both counts are in the hundreds.
    assert!(inked > 400, "only {inked} inked pixels checked");
    assert!(blank > 400, "only {blank} blank pixels checked");
}

/// Where vanilla puts the label: centred on `-getStringWidth(str) / 2` over the entity, with
/// the backdrop one label pixel proud of the text on every side.
#[test]
fn the_label_is_centred_over_the_entity_and_the_backdrop_frames_it() {
    let Some(mut gpu) = gpu() else { return };
    let Some(mut font) = vanilla_font() else {
        return;
    };
    let cam = camera();
    let plate = centred_plate("Steve", &cam);
    let width = font.string_width(&plate.text);
    // S=6 t=4 e=6 v=6 e=6 from the real `ascii.png` width table, so `j` is 14 and the backdrop
    // runs from label x -15 to +15.
    assert_eq!(width, 28, "vanilla's ascii metrics for \"Steve\"");

    let (phase, _) = phase_for(&gpu, &mut font, &plate, &cam, 25.0);
    let px = render(&mut gpu, &cam, phase, None);

    // The anchor projects to the exact middle of the frame.
    let (ax, ay, _) = project(&cam, plate.local_to_world(cam.yaw, cam.pitch, 0.0, 0.0));
    assert!(
        (ax - 128.0).abs() < 0.01 && (ay - 128.0).abs() < 0.01,
        "{ax},{ay}"
    );

    // Ink extent: white pixels only, since the backdrop has no red.
    let (mut lo, mut hi) = (u32::MAX, 0u32);
    let mut ink = 0u32;
    for y in 0..H {
        for x in 0..W {
            if pixel(&px, x, y) == [255, 255, 255, 255] {
                lo = lo.min(x);
                hi = hi.max(x);
                ink += 1;
            }
        }
    }
    assert!(ink > 500, "the name did not render: {ink} lit pixels");
    let centre = (lo + hi) as f32 / 2.0;
    // java: `-getStringWidth(str) / 2` is integer division, so an odd width sits half a pixel
    // left; one label pixel of slack covers that and the glyph quads' -0.01 shrink.
    assert!(
        (centre - ax).abs() <= SCALE_PX,
        "ink spans {lo}..{hi} (centre {centre}), anchor at {ax}"
    );
    // Width: the drawn string is `width - 1` label pixels of ink at most.
    let span = (hi - lo) as f32 / SCALE_PX;
    assert!(
        (span - (width - 1) as f32).abs() <= 2.5,
        "ink is {span} label px wide, expected about {}",
        width - 1
    );

    // The backdrop reaches one label pixel past the text on each side and stops there:
    // java: `(-j - 1, -1 + i)` to `(j + 1, 8 + i)` with `j = getStringWidth(str) / 2` = 14.
    let backdrop_at = |label_x: f32| {
        let (sx, sy, _) = project(&cam, plate.local_to_world(cam.yaw, cam.pitch, label_x, 4.0));
        pixel(&px, sx.round() as u32, sy.round() as u32)
    };
    let inside = backdrop_at(-14.5); // between the backdrop's edge (-15) and the text (-14)
    let outside = backdrop_at(-15.5); // one label pixel further out, clear of the backdrop
    assert!(
        inside[2] < outside[2],
        "the backdrop must darken the clear colour: inside {inside:?} outside {outside:?}"
    );
    assert_eq!(outside, [0, 0, 102, 255], "clear colour, untouched");
}

/// (b) of the acceptance bar: with real glyphs present, `ThroughWalls` still shows through an
/// occluder and `InFront` is still occluded by it.
#[test]
fn the_two_pass_depth_policy_still_holds_with_real_glyphs() {
    let Some(mut gpu) = gpu() else { return };
    let Some(mut font) = vanilla_font() else {
        return;
    };
    let cam = camera();
    let plate = centred_plate("Steve", &cam);
    let wall_z = plate.rel.z * 0.5;

    let max_red = |gpu: &mut Gpu, font: &mut FontRenderer, plate: &Nameplate, wall: Option<f32>| {
        let (phase, _) = phase_for(gpu, font, plate, &cam, 25.0);
        let px = render(gpu, &cam, phase, wall);
        px.chunks(4).map(|p| p[0]).max().unwrap_or(0)
    };

    let clear = max_red(&mut gpu, &mut font, &plate, None);
    assert_eq!(
        clear, 255,
        "with line of sight the InFront pass paints opaque white ink"
    );

    let occluded = max_red(&mut gpu, &mut font, &plate, Some(wall_z));
    // The wall depth-rejects the opaque pass, so the brightest ink left is the ghost: white at
    // alpha 32/255 over a backdrop with no red in it. HEADLESS_FORMAT is Rgba8Unorm, so that
    // blend is exact — 0.1254902 * 255 = 32.
    assert!(
        (20..=45).contains(&occluded),
        "expected the 32/255 ghost and nothing brighter, got {occluded}"
    );
}

/// Sneaking is a third behaviour, not a dimmer variant of the first: one depth-tested pass, so
/// a sneaking name is invisible behind a wall rather than faint.
#[test]
fn a_sneaking_name_never_shows_through_a_wall_and_is_dim_in_front_of_one() {
    let Some(mut gpu) = gpu() else { return };
    let Some(mut font) = vanilla_font() else {
        return;
    };
    let cam = camera();
    let mut plate = centred_plate("Steve", &cam);
    plate.sneaking = true;
    let wall_z = plate.rel.z * 0.5;

    let (phase, geometry) = phase_for(&gpu, &mut font, &plate, &cam, 25.0);
    assert_eq!(geometry.glyphs.len(), 1, "one pass while sneaking");
    let visible = render(&mut gpu, &cam, phase, None);
    let front = visible.chunks(4).map(|p| p[0]).max().unwrap_or(0);
    assert!(
        (20..=45).contains(&front),
        "a sneaking name in plain sight is drawn at 32/255, got {front}"
    );

    let (phase, _) = phase_for(&gpu, &mut font, &plate, &cam, 25.0);
    let hidden = render(&mut gpu, &cam, phase, Some(wall_z));
    assert_eq!(
        hidden.chunks(4).map(|p| p[0]).max().unwrap_or(0),
        0,
        "a sneaking name behind a wall is not drawn at all"
    );
}

/// A `§` colour code reaches the GPU as a real tint, and costs no geometry of its own.
#[test]
fn a_formatting_code_recolours_the_rendered_glyphs() {
    let Some(mut gpu) = gpu() else { return };
    let Some(mut font) = vanilla_font() else {
        return;
    };
    let cam = camera();
    // §c is vanilla's "red", 0xFF5555.
    let plate = centred_plate("§cA", &cam);
    let (phase, geometry) = phase_for(&gpu, &mut font, &plate, &cam, 25.0);
    assert_eq!(
        geometry.glyph_vertex_count(),
        12,
        "one glyph in each of two passes; the code itself is not drawn"
    );
    let px = render(&mut gpu, &cam, phase, None);

    let lit: Vec<[u8; 4]> = px
        .chunks(4)
        .map(|p| [p[0], p[1], p[2], p[3]])
        .filter(|p| p[0] > 200)
        .collect();
    assert!(!lit.is_empty(), "no red ink was drawn");
    for p in &lit {
        assert_eq!(
            [p[0], p[1], p[2]],
            [255, 85, 85],
            "expected vanilla's §c (0xFF5555)"
        );
    }
}
