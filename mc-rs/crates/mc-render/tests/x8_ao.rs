//! Screen-space ambient occlusion ([`AoPhase`]) on a real GPU.
//!
//! Every test renders whole frames through a real [`PhaseList`] on a headless context and reads
//! them back. The scene is drawn by a small test phase at `TERRAIN_SOLID` that writes depth like
//! terrain does: a floor, a wall standing on it (an inside corner along the floor line), a block
//! sitting on the open floor, and sky above the wall. A stand-in for the first-person hand clears
//! depth and draws a quad at `FIRST_PERSON`, exactly like the real hand phase.
//!
//! * Off (switch off, or no world): the frame is byte-identical to a frame without the phase,
//!   and the phase owns no GPU memory.
//! * On: the inside corner darkens, the open floor and the wall's middle stay (almost) as they
//!   were, the sky and the hand keep their exact bytes. At half and full resolution, and under
//!   4x MSAA.
//! * Toggling on and off again returns to the exact off frame; a resize keeps working.
//! * `gpu_cost_at_1080p` (ignored): GPU time at 1920x1080 with `MC_GPU_TIMING=1`.

use std::cell::RefCell;
use std::rc::Rc;

use glam::{DVec3, Vec3, Vec4};
use mc_render::post::ao::{AoPhase, AoQuality, AoSettings};
use mc_render::{
    Camera, DepthAttachment, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase,
};
use wgpu::util::DeviceExt;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 200;

// ------------------------------------------------------------------------------- harness

fn headless(width: u32, height: u32, sample_count: u32) -> Option<Gpu> {
    match Gpu::headless(
        width,
        height,
        GpuOptions {
            vsync: false,
            sample_count,
            low_power: false,
        },
    ) {
        Ok(gpu) => {
            gpu.device.on_uncaptured_error(std::sync::Arc::new(|e| {
                panic!("wgpu validation error: {e}")
            }));
            Some(gpu)
        }
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

/// The client's `Shared`: the list owns one handle, the test keeps the other.
#[derive(Clone)]
struct Handle(Rc<RefCell<AoPhase>>);

impl Handle {
    fn new(on: bool, settings: AoSettings) -> Handle {
        let mut phase = AoPhase::new();
        phase.set_settings(settings);
        phase.set_enabled(on);
        phase.set_in_world(true);
        Handle(Rc::new(RefCell::new(phase)))
    }

    fn with<R>(&self, f: impl FnOnce(&mut AoPhase) -> R) -> R {
        f(&mut self.0.borrow_mut())
    }
}

impl RenderPhase for Handle {
    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        self.0.borrow_mut().record(gpu, enc, view, cam);
    }

    fn name(&self) -> &'static str {
        AoPhase::NAME
    }

    fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        self.0.borrow_mut().resize(gpu, width, height);
    }

    fn enabled(&self) -> bool {
        self.0.borrow().enabled()
    }

    fn needs_scene_color(&self) -> bool {
        self.0.borrow().needs_scene_color()
    }
}

const SCENE_WGSL: &str = r#"
struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen:     vec4<f32>,
    frame:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs(@location(0) pos: vec3<f32>, @location(1) color: vec3<f32>) -> VsOut {
    var out: VsOut;
    out.clip = camera.view_proj * vec4<f32>(pos, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
    color: [f32; 3],
}

/// Flat-coloured, depth-writing triangles in camera-relative space (the camera sits at the
/// origin), opened with `depth` (`Shared` for the world, `Clear` for the hand stand-in).
struct Geometry {
    label: &'static str,
    depth: DepthAttachment,
    pipeline: wgpu::RenderPipeline,
    vertices: wgpu::Buffer,
    count: u32,
}

impl Geometry {
    fn new(gpu: &Gpu, label: &'static str, depth: DepthAttachment, verts: &[Vertex]) -> Geometry {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("x8 scene"),
                source: wgpu::ShaderSource::Wgsl(SCENE_WGSL.into()),
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("x8 scene"),
                bind_group_layouts: &[Some(gpu.camera_layout())],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                    })],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: mc_render::DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: gpu.sample_count(),
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        let vertices = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
        Geometry {
            label,
            depth,
            pipeline,
            vertices,
            count: verts.len() as u32,
        }
    }
}

impl RenderPhase for Geometry {
    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        let mut pass = gpu.begin_pass(enc, view, self.label, self.depth);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.draw(0..self.count, 0..1);
    }

    fn name(&self) -> &'static str {
        self.label
    }
}

/// One axis-aligned quad as two triangles.
fn quad(out: &mut Vec<Vertex>, corners: [[f32; 3]; 4], color: [f32; 3]) {
    for i in [0, 1, 2, 0, 2, 3] {
        out.push(Vertex {
            pos: corners[i],
            color,
        });
    }
}

/// Eye height above the floor, as in vanilla.
const EYE: f32 = 1.62;
/// The wall's plane.
const WALL_Z: f32 = 6.0;
const FLOOR_COLOR: [f32; 3] = [0.8, 0.8, 0.8];
const WALL_COLOR: [f32; 3] = [0.7, 0.75, 0.8];
const BLOCK_COLOR: [f32; 3] = [0.6, 0.5, 0.4];
const HAND_COLOR: [f32; 3] = [0.9, 0.6, 0.5];

/// Floor (y = -EYE), a wall at z = WALL_Z rising 2.2 blocks, and a unit block on the floor at
/// x in [2, 3], z in [3, 4].
fn world_vertices() -> Vec<Vertex> {
    let mut v = Vec::new();
    let y0 = -EYE;
    quad(
        &mut v,
        [
            [-12.0, y0, 0.2],
            [12.0, y0, 0.2],
            [12.0, y0, WALL_Z],
            [-12.0, y0, WALL_Z],
        ],
        FLOOR_COLOR,
    );
    let top = y0 + 2.2;
    quad(
        &mut v,
        [
            [-12.0, y0, WALL_Z],
            [12.0, y0, WALL_Z],
            [12.0, top, WALL_Z],
            [-12.0, top, WALL_Z],
        ],
        WALL_COLOR,
    );
    // The block's four sides and top (the bottom sits on the floor and is never seen).
    let (x0, x1, z0, z1, h) = (2.0, 3.0, 3.0, 4.0, y0 + 1.0);
    quad(
        &mut v,
        [[x0, y0, z0], [x1, y0, z0], [x1, h, z0], [x0, h, z0]],
        BLOCK_COLOR,
    );
    quad(
        &mut v,
        [[x0, y0, z1], [x1, y0, z1], [x1, h, z1], [x0, h, z1]],
        BLOCK_COLOR,
    );
    quad(
        &mut v,
        [[x0, y0, z0], [x0, y0, z1], [x0, h, z1], [x0, h, z0]],
        BLOCK_COLOR,
    );
    quad(
        &mut v,
        [[x1, y0, z0], [x1, y0, z1], [x1, h, z1], [x1, h, z0]],
        BLOCK_COLOR,
    );
    quad(
        &mut v,
        [[x0, h, z0], [x1, h, z0], [x1, h, z1], [x0, h, z1]],
        BLOCK_COLOR,
    );
    v
}

/// A quad half a block in front of the eye, low and to the right: where the held item sits. It
/// covers the floor and the wall's crease behind it, so AO would show on it if it leaked.
fn hand_vertices() -> Vec<Vertex> {
    let mut v = Vec::new();
    quad(
        &mut v,
        [
            [-0.25, -0.40, 0.5],
            [-0.45, -0.40, 0.5],
            [-0.45, -0.06, 0.5],
            [-0.25, -0.06, 0.5],
        ],
        HAND_COLOR,
    );
    v
}

fn camera(width: u32, height: u32) -> Camera {
    let mut cam = Camera {
        pos: DVec3::ZERO,
        yaw: 0.0,
        pitch: 12.0,
        fov: 70.0,
        near: 0.05,
        far: 256.0,
        aspect: width as f32 / height as f32,
        ..Camera::default()
    };
    cam.update();
    cam
}

/// Where a camera-relative point lands on screen, in pixels.
fn project(cam: &Camera, p: Vec3, width: u32, height: u32) -> (u32, u32) {
    let clip = cam.view_proj() * Vec4::new(p.x, p.y, p.z, 1.0);
    let ndc = clip / clip.w;
    let x = (ndc.x + 1.0) * 0.5 * width as f32;
    let y = (1.0 - ndc.y) * 0.5 * height as f32;
    (x as u32, y as u32)
}

const SKY: wgpu::Color = wgpu::Color {
    r: 0.3,
    g: 0.5,
    b: 0.9,
    a: 1.0,
};

/// The world, optionally AO, and the hand stand-in, in the client's order.
fn list(gpu: &Gpu, ao: Option<&Handle>, hand: bool) -> PhaseList {
    let mut phases = PhaseList::new();
    phases.clear_color = SKY;
    phases.insert(
        PhaseOrder::TERRAIN_SOLID,
        Box::new(Geometry::new(
            gpu,
            "x8 world",
            DepthAttachment::Shared,
            &world_vertices(),
        )),
    );
    if let Some(ao) = ao {
        phases.insert(PhaseOrder::AMBIENT_OCCLUSION, Box::new(ao.clone()));
    }
    if hand {
        phases.insert(
            PhaseOrder::FIRST_PERSON,
            Box::new(Geometry::new(
                gpu,
                "x8 hand",
                DepthAttachment::Clear,
                &hand_vertices(),
            )),
        );
    }
    phases
}

fn frame(gpu: &mut Gpu, phases: &mut PhaseList, cam: &Camera) -> Vec<u8> {
    let target = gpu.acquire().expect("acquire");
    phases.render(gpu, target.view(), cam, 0.0);
    drop(target);
    gpu.read_pixels().expect("readback")
}

fn pixel(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * width + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

/// Mean brightness (sum of RGB) over a small box around `(x, y)`.
fn brightness(pixels: &[u8], width: u32, (x, y): (u32, u32)) -> f32 {
    let mut sum = 0.0;
    let mut n = 0.0;
    for dy in 0..3 {
        for dx in 0..3 {
            let p = pixel(pixels, width, x + dx - 1, y + dy - 1);
            sum += f32::from(p[0]) + f32::from(p[1]) + f32::from(p[2]);
            n += 1.0;
        }
    }
    sum / n
}

/// The probe points of the scene, in pixels.
struct Probes {
    /// On the floor, just in front of the wall.
    corner_floor: (u32, u32),
    /// On the wall, just above the floor.
    corner_wall: (u32, u32),
    /// Open floor far from the wall and the block.
    open_floor: (u32, u32),
    /// The middle of the wall, far from the floor.
    open_wall: (u32, u32),
    /// On the floor right beside the block's left face.
    block_contact: (u32, u32),
    /// Above the wall.
    sky: (u32, u32),
}

/// Probe points well inside the frame, off the hand (which spans 27-42 degrees right of centre,
/// from 5 degrees above centre to 27 below, over the darkened crease), at least a radius (1.5
/// blocks) from any crease they must not see.
fn probes(cam: &Camera, width: u32, height: u32) -> Probes {
    let y0 = -EYE;
    let p = |x: f32, y: f32, z: f32| {
        let (px, py) = project(cam, Vec3::new(x, y, z), width, height);
        assert!(
            (8..width - 8).contains(&px) && (8..height - 8).contains(&py),
            "probe ({x}, {y}, {z}) lands at ({px}, {py}), outside the frame"
        );
        (px, py)
    };
    Probes {
        corner_floor: p(-1.0, y0, WALL_Z - 0.12),
        corner_wall: p(-1.0, y0 + 0.1, WALL_Z),
        open_floor: p(-0.5, y0, 3.0),
        open_wall: p(-2.5, y0 + 1.8, WALL_Z),
        block_contact: p(1.9, y0, 3.5),
        sky: (width / 2, 4),
    }
}

/// The mod's default settings, at half or full resolution.
fn settings(half: bool) -> AoSettings {
    AoSettings {
        quality: AoQuality::Medium,
        radius: 1.5,
        strength: 1.0,
        half_resolution: half,
    }
}

// --------------------------------------------------------------------------------- tests

#[test]
fn off_or_outside_a_world_changes_no_byte_and_allocates_nothing() {
    let Some(mut gpu) = headless(WIDTH, HEIGHT, 1) else {
        return;
    };
    let cam = camera(WIDTH, HEIGHT);
    let mut plain = list(&gpu, None, true);
    let reference = frame(&mut gpu, &mut plain, &cam);

    // Registered but switched off.
    let off = Handle::new(false, settings(true));
    let mut phases = list(&gpu, Some(&off), true);
    assert_eq!(frame(&mut gpu, &mut phases, &cam), reference);
    assert_eq!(off.with(|p| p.passes_recorded()), 0);
    assert!(!off.with(|p| p.allocated()));
    assert!(!phases.scene_mode());

    // Switched on, but no world on screen.
    let no_world = Handle::new(true, settings(true));
    no_world.with(|p| p.set_in_world(false));
    let mut phases = list(&gpu, Some(&no_world), true);
    assert_eq!(frame(&mut gpu, &mut phases, &cam), reference);
    assert_eq!(no_world.with(|p| p.passes_recorded()), 0);
    assert!(!no_world.with(|p| p.allocated()));

    // On at 0 % strength.
    let zero = Handle::new(
        true,
        AoSettings {
            strength: 0.0,
            ..settings(true)
        },
    );
    let mut phases = list(&gpu, Some(&zero), true);
    assert_eq!(frame(&mut gpu, &mut phases, &cam), reference);
    assert_eq!(zero.with(|p| p.passes_recorded()), 0);
}

/// The core look: corners darken, open surfaces barely change, sky and hand are untouched.
fn assert_ao_look(gpu: &mut Gpu, width: u32, height: u32, half: bool) {
    let cam = camera(width, height);
    let probes = probes(&cam, width, height);
    let mut plain = list(gpu, None, true);
    let off = frame(gpu, &mut plain, &cam);
    let ao = Handle::new(true, settings(half));
    let mut phases = list(gpu, Some(&ao), true);
    let on = frame(gpu, &mut phases, &cam);
    assert_eq!(ao.with(|p| p.passes_recorded()), 1);
    assert!(
        !phases.scene_mode(),
        "AO must not switch the frame to scene mode"
    );

    let ratio = |at: (u32, u32)| brightness(&on, width, at) / brightness(&off, width, at);
    let corner_floor = ratio(probes.corner_floor);
    let corner_wall = ratio(probes.corner_wall);
    let open_floor = ratio(probes.open_floor);
    let open_wall = ratio(probes.open_wall);
    let contact = ratio(probes.block_contact);
    eprintln!(
        "{width}x{height} half={half}: corner floor {corner_floor:.3}, corner wall \
         {corner_wall:.3}, block contact {contact:.3}, open floor {open_floor:.3}, open wall \
         {open_wall:.3}"
    );
    assert!(
        corner_floor < 0.85,
        "the floor by the wall should darken: {corner_floor}"
    );
    assert!(
        corner_wall < 0.85,
        "the wall by the floor should darken: {corner_wall}"
    );
    assert!(
        contact < 0.9,
        "the floor beside the block should darken: {contact}"
    );
    assert_eq!(open_floor, 1.0, "open floor must be untouched");
    assert_eq!(open_wall, 1.0, "the open wall must be untouched");

    // Sky: every pixel whose off colour is the clear colour keeps its exact bytes.
    let sky = pixel(&off, width, probes.sky.0, probes.sky.1);
    let mut sky_pixels = 0;
    for (a, b) in off.chunks_exact(4).zip(on.chunks_exact(4)) {
        if a == sky {
            sky_pixels += 1;
            assert_eq!(a, b, "a sky pixel changed");
        }
    }
    assert!(
        sky_pixels > (width * height / 20) as usize,
        "the scene shows sky"
    );

    // The hand stand-in: the pixels it covers (where the frame differs from one without it) are
    // byte-identical with AO on, and they include darkened crease behind it. Under MSAA a hand
    // edge pixel resolves hand samples with world samples behind it, which AO did darken, so only
    // pixels whose four neighbours are hand too must be identical.
    let mut handless = list(gpu, None, false);
    let no_hand = frame(gpu, &mut handless, &cam);
    let is_hand = |x: u32, y: u32| pixel(&off, width, x, y) != pixel(&no_hand, width, x, y);
    let mut hand_pixels = 0;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let interior = is_hand(x, y)
                && is_hand(x - 1, y)
                && is_hand(x + 1, y)
                && is_hand(x, y - 1)
                && is_hand(x, y + 1);
            if interior {
                hand_pixels += 1;
                assert_eq!(
                    pixel(&off, width, x, y),
                    pixel(&on, width, x, y),
                    "hand pixel ({x}, {y}) changed"
                );
            }
        }
    }
    assert!(hand_pixels > 200, "the hand covers {hand_pixels} pixels");
    // Behind the hand, AO is there (so the identity above is not vacuous).
    let ao_handless = Handle::new(true, settings(half));
    let mut handless_ao = list(gpu, Some(&ao_handless), false);
    let on_no_hand = frame(gpu, &mut handless_ao, &cam);
    let darkened_behind_hand = off
        .chunks_exact(4)
        .zip(no_hand.chunks_exact(4))
        .zip(on_no_hand.chunks_exact(4))
        .filter(|((a, c), d)| a != c && d != c)
        .count();
    assert!(
        darkened_behind_hand > 20,
        "AO would darken {darkened_behind_hand} pixels under the hand"
    );
}

#[test]
fn corners_darken_open_surfaces_sky_and_hand_do_not_at_half_resolution() {
    let Some(mut gpu) = headless(WIDTH, HEIGHT, 1) else {
        return;
    };
    assert_ao_look(&mut gpu, WIDTH, HEIGHT, true);
}

#[test]
fn corners_darken_open_surfaces_sky_and_hand_do_not_at_full_resolution() {
    let Some(mut gpu) = headless(WIDTH, HEIGHT, 1) else {
        return;
    };
    assert_ao_look(&mut gpu, WIDTH, HEIGHT, false);
}

#[test]
fn msaa_4x_is_supported() {
    let Some(mut gpu) = headless(WIDTH, HEIGHT, 4) else {
        return;
    };
    if gpu.sample_count() != 4 {
        eprintln!("SKIP: 4x MSAA unavailable ({}x)", gpu.sample_count());
        return;
    }
    assert_ao_look(&mut gpu, WIDTH, HEIGHT, true);
}

/// Nothing but a floor, seen from eye height down to its horizon: every quality at both
/// resolutions leaves the frame byte-identical. A textbook GTAO shades an open plane by a few
/// percent off the screen centre (uneven slice angles, samples rounded beside the slice line);
/// this pins the two corrections in `ao.wgsl`.
#[test]
fn an_open_floor_is_untouched_at_every_quality() {
    let Some(mut gpu) = headless(WIDTH, HEIGHT, 1) else {
        return;
    };
    let cam = camera(WIDTH, HEIGHT);
    let floor = |gpu: &Gpu, ao: Option<&Handle>| {
        let mut v = Vec::new();
        quad(
            &mut v,
            [
                [-40.0, -EYE, 0.2],
                [40.0, -EYE, 0.2],
                [40.0, -EYE, 80.0],
                [-40.0, -EYE, 80.0],
            ],
            FLOOR_COLOR,
        );
        let mut phases = PhaseList::new();
        phases.clear_color = SKY;
        phases.insert(
            PhaseOrder::TERRAIN_SOLID,
            Box::new(Geometry::new(gpu, "x8 floor", DepthAttachment::Shared, &v)),
        );
        if let Some(ao) = ao {
            phases.insert(PhaseOrder::AMBIENT_OCCLUSION, Box::new(ao.clone()));
        }
        phases
    };
    let mut plain = floor(&gpu, None);
    let off = frame(&mut gpu, &mut plain, &cam);
    for quality in AoQuality::ALL {
        for half in [true, false] {
            let ao = Handle::new(
                true,
                AoSettings {
                    quality,
                    strength: 2.0,
                    ..settings(half)
                },
            );
            let mut phases = floor(&gpu, Some(&ao));
            let on = frame(&mut gpu, &mut phases, &cam);
            assert_eq!(ao.with(|p| p.passes_recorded()), 1);
            let changed = off
                .chunks_exact(4)
                .zip(on.chunks_exact(4))
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                changed, 0,
                "{quality:?} half={half}: {changed} floor pixels shaded"
            );
        }
    }
}

#[test]
fn every_quality_darkens_the_corner_and_more_strength_darkens_more() {
    let Some(mut gpu) = headless(WIDTH, HEIGHT, 1) else {
        return;
    };
    let cam = camera(WIDTH, HEIGHT);
    let probes = probes(&cam, WIDTH, HEIGHT);
    let mut plain = list(&gpu, None, false);
    let off = frame(&mut gpu, &mut plain, &cam);
    let base = brightness(&off, WIDTH, probes.corner_floor);
    let ao = Handle::new(true, settings(true));
    let mut phases = list(&gpu, Some(&ao), false);
    for quality in AoQuality::ALL {
        let mut previous = f32::INFINITY;
        for strength in [0.5, 1.0, 2.0] {
            ao.with(|p| {
                p.set_settings(AoSettings {
                    quality,
                    strength,
                    ..settings(true)
                })
            });
            let on = frame(&mut gpu, &mut phases, &cam);
            let r = brightness(&on, WIDTH, probes.corner_floor) / base;
            eprintln!("{quality:?} strength {strength}: corner {r:.3}");
            assert!(
                r < 0.95,
                "{quality:?} at {strength} should darken the corner: {r}"
            );
            assert!(
                r < previous,
                "more strength must darken more ({r} vs {previous})"
            );
            previous = r;
        }
    }
}

#[test]
fn toggling_back_off_restores_the_exact_frame_and_frees_the_targets() {
    let Some(mut gpu) = headless(WIDTH, HEIGHT, 1) else {
        return;
    };
    let cam = camera(WIDTH, HEIGHT);
    let ao = Handle::new(false, settings(true));
    let mut phases = list(&gpu, Some(&ao), true);
    let before = frame(&mut gpu, &mut phases, &cam);
    ao.with(|p| p.set_enabled(true));
    let on = frame(&mut gpu, &mut phases, &cam);
    assert_ne!(on, before, "AO on changes the frame");
    assert!(ao.with(|p| p.allocated()));
    ao.with(|p| p.set_enabled(false));
    assert!(
        !ao.with(|p| p.allocated()),
        "off drops the screen-sized targets"
    );
    assert_eq!(frame(&mut gpu, &mut phases, &cam), before);
    // Leaving the world does the same.
    ao.with(|p| p.set_enabled(true));
    let _ = frame(&mut gpu, &mut phases, &cam);
    ao.with(|p| p.set_in_world(false));
    assert!(!ao.with(|p| p.allocated()));
    assert_eq!(frame(&mut gpu, &mut phases, &cam), before);
}

#[test]
fn a_resize_and_a_resolution_switch_keep_working() {
    let Some(mut gpu) = headless(WIDTH, HEIGHT, 1) else {
        return;
    };
    let ao = Handle::new(true, settings(true));
    let mut phases = list(&gpu, Some(&ao), true);
    let cam = camera(WIDTH, HEIGHT);
    let _ = frame(&mut gpu, &mut phases, &cam);

    let (w, h) = (257, 161);
    assert!(gpu.resize(w, h));
    phases.resize(&gpu, w, h);
    let cam = camera(w, h);
    let probes = probes(&cam, w, h);
    let off = {
        let mut plain = list(&gpu, None, true);
        frame(&mut gpu, &mut plain, &cam)
    };
    for half in [true, false, true] {
        ao.with(|p| p.set_settings(settings(half)));
        let on = frame(&mut gpu, &mut phases, &cam);
        let r = brightness(&on, w, probes.corner_floor) / brightness(&off, w, probes.corner_floor);
        assert!(
            r < 0.85,
            "after a resize (half={half}) the corner still darkens: {r}"
        );
    }
}

/// A busier scene for looking at: steps, a pillar and a ledge on the wall, plus the test scene.
fn rich_world_vertices() -> Vec<Vertex> {
    let mut v = world_vertices();
    let y0 = -EYE;
    let mut cube = |x0: f32, y: f32, z0: f32, x1: f32, y1: f32, z1: f32, c: [f32; 3]| {
        quad(
            &mut v,
            [[x0, y, z0], [x1, y, z0], [x1, y1, z0], [x0, y1, z0]],
            c,
        );
        quad(
            &mut v,
            [[x0, y, z1], [x1, y, z1], [x1, y1, z1], [x0, y1, z1]],
            c,
        );
        quad(
            &mut v,
            [[x0, y, z0], [x0, y, z1], [x0, y1, z1], [x0, y1, z0]],
            c,
        );
        quad(
            &mut v,
            [[x1, y, z0], [x1, y, z1], [x1, y1, z1], [x1, y1, z0]],
            c,
        );
        quad(
            &mut v,
            [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],
            c,
        );
        quad(
            &mut v,
            [[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]],
            c,
        );
    };
    // Steps against the wall.
    for i in 0..3 {
        let x = -1.0 - i as f32;
        cube(
            x - 1.0,
            y0,
            WALL_Z - 1.0,
            x,
            y0 + 1.0 + i as f32 * 0.5,
            WALL_Z,
            [0.55, 0.6, 0.65],
        );
    }
    // A pillar in the open.
    cube(-0.5, y0, 2.5, 0.0, y0 + 2.0, 3.0, [0.7, 0.6, 0.5]);
    // A ledge sticking out of the wall at head height.
    cube(
        -1.0,
        y0 + 1.5,
        WALL_Z - 1.2,
        2.0,
        y0 + 1.7,
        WALL_Z,
        [0.65, 0.65, 0.6],
    );
    v
}

/// Writes off/on images of the busier scene to `X8_DUMP_DIR` for a human to look at.
/// `X8_DUMP_DIR=... cargo test -p mc-render --test x8_ao dump -- --ignored`
#[test]
#[ignore = "writes PNGs for a visual check; needs X8_DUMP_DIR"]
fn dump_images_for_a_visual_check() {
    let Some(dir) = std::env::var_os("X8_DUMP_DIR") else {
        eprintln!("SKIP: set X8_DUMP_DIR");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).expect("dump dir");
    let (width, height) = (960, 600);
    let Some(mut gpu) = headless(width, height, 1) else {
        return;
    };
    let cam = camera(width, height);
    let rich = |gpu: &Gpu, ao: Option<&Handle>| {
        let mut phases = PhaseList::new();
        phases.clear_color = SKY;
        phases.insert(
            PhaseOrder::TERRAIN_SOLID,
            Box::new(Geometry::new(
                gpu,
                "x8 world",
                DepthAttachment::Shared,
                &rich_world_vertices(),
            )),
        );
        if let Some(ao) = ao {
            phases.insert(PhaseOrder::AMBIENT_OCCLUSION, Box::new(ao.clone()));
        }
        phases
    };
    let save = |name: &str, pixels: &[u8]| {
        image::RgbaImage::from_raw(width, height, pixels.to_vec())
            .expect("size")
            .save(dir.join(name))
            .expect("save png");
    };
    // Nothing but the floor: any AO here is false self-occlusion.
    {
        let floor_only = |gpu: &Gpu, ao: Option<&Handle>| {
            let mut v = Vec::new();
            quad(
                &mut v,
                [
                    [-40.0, -EYE, 0.2],
                    [40.0, -EYE, 0.2],
                    [40.0, -EYE, 80.0],
                    [-40.0, -EYE, 80.0],
                ],
                FLOOR_COLOR,
            );
            let mut phases = PhaseList::new();
            phases.clear_color = SKY;
            phases.insert(
                PhaseOrder::TERRAIN_SOLID,
                Box::new(Geometry::new(gpu, "x8 floor", DepthAttachment::Shared, &v)),
            );
            if let Some(ao) = ao {
                phases.insert(PhaseOrder::AMBIENT_OCCLUSION, Box::new(ao.clone()));
            }
            phases
        };
        let mut plain = floor_only(&gpu, None);
        let off = frame(&mut gpu, &mut plain, &cam);
        for quality in AoQuality::ALL {
            for half in [true, false] {
                let ao = Handle::new(
                    true,
                    AoSettings {
                        quality,
                        ..settings(half)
                    },
                );
                let mut phases = floor_only(&gpu, Some(&ao));
                let on = frame(&mut gpu, &mut phases, &cam);
                // Bottom-centre box and a bottom-left box of the floor.
                let mean = |x0: u32, y0: u32| {
                    let mut sum = 0.0;
                    for y in y0..y0 + 40 {
                        for x in x0..x0 + 40 {
                            let i = ((y * width + x) * 4 + 1) as usize;
                            sum += f32::from(on[i]) / f32::from(off[i]);
                        }
                    }
                    sum / 1600.0
                };
                eprintln!(
                    "floor only {quality:?} half={half}: bottom centre {:.4}, bottom left {:.4}",
                    mean(width / 2 - 20, height - 60),
                    mean(60, height - 60)
                );
            }
        }
        let ao = Handle::new(true, settings(true));
        let mut phases = floor_only(&gpu, Some(&ao));
        let on = frame(&mut gpu, &mut phases, &cam);
        let mut worst = (1.0f32, 0u32, 0u32);
        let amplified: Vec<u8> = on
            .chunks_exact(4)
            .zip(off.chunks_exact(4))
            .enumerate()
            .flat_map(|(i, (a, b))| {
                let r = if b[1] == 0 {
                    1.0
                } else {
                    (f32::from(a[1]) / f32::from(b[1])).min(1.0)
                };
                if r < worst.0 {
                    worst = (r, i as u32 % width, i as u32 / width);
                }
                let v = (255.0 - (1.0 - r) * 255.0 * 20.0).clamp(0.0, 255.0) as u8;
                [v, v, v, 255]
            })
            .collect();
        eprintln!(
            "floor only: darkest {:.4} at ({}, {})",
            worst.0, worst.1, worst.2
        );
        save("amp_x8_ao_floor_only.png", &amplified);
    }
    // Yawed 25 degrees: whatever follows the objects is AO, whatever stays put is the screen.
    let mut yawed = cam;
    yawed.yaw = 25.0;
    yawed.update();
    {
        let mut plain = rich(&gpu, None);
        let off = frame(&mut gpu, &mut plain, &yawed);
        let ao = Handle::new(true, settings(true));
        let mut phases = rich(&gpu, Some(&ao));
        let on = frame(&mut gpu, &mut phases, &yawed);
        let amplified: Vec<u8> = on
            .chunks_exact(4)
            .zip(off.chunks_exact(4))
            .flat_map(|(a, b)| {
                let r = if b[1] == 0 {
                    1.0
                } else {
                    (f32::from(a[1]) / f32::from(b[1])).min(1.0)
                };
                let v = (255.0 - (1.0 - r) * 255.0 * 20.0).clamp(0.0, 255.0) as u8;
                [v, v, v, 255]
            })
            .collect();
        save("amp_x8_ao_yawed.png", &amplified);
    }
    let mut plain = rich(&gpu, None);
    let off = frame(&mut gpu, &mut plain, &cam);
    save("x8_ao_off.png", &off);
    for (name, s) in [
        ("x8_ao_half_medium.png", settings(true)),
        ("x8_ao_full_medium.png", settings(false)),
        (
            "x8_ao_half_ultra.png",
            AoSettings {
                quality: AoQuality::Ultra,
                ..settings(true)
            },
        ),
        (
            "x8_ao_half_low.png",
            AoSettings {
                quality: AoQuality::Low,
                ..settings(true)
            },
        ),
    ] {
        let ao = Handle::new(true, s);
        let mut phases = rich(&gpu, Some(&ao));
        let on = frame(&mut gpu, &mut phases, &cam);
        save(name, &on);
        // The AO term itself: on / off, as grey.
        let ratio: Vec<u8> = on
            .chunks_exact(4)
            .zip(off.chunks_exact(4))
            .flat_map(|(a, b)| {
                let r = if b[1] == 0 {
                    255
                } else {
                    ((f32::from(a[1]) / f32::from(b[1])).min(1.0) * 255.0) as u8
                };
                [r, r, r, 255]
            })
            .collect();
        save(&format!("term_{name}"), &ratio);
        // The same, 20x amplified: shows any faint darkening (self-occlusion, banding).
        let amplified: Vec<u8> = ratio
            .chunks_exact(4)
            .flat_map(|p| {
                let a = 255u32.saturating_sub((255 - u32::from(p[0])) * 20) as u8;
                [a, a, a, 255]
            })
            .collect();
        save(&format!("amp_{name}"), &amplified);
    }
}

/// The same check on real 1.8.9 textures and the real terrain renderer: a small grass yard with a
/// wall, a ledge, a tree, a pillar and some stones (AO off/on), and a long grass field seen at a
/// grazing angle (AF off/16x). Needs the vanilla jar (`MC_JARS_DIR`) and `X8_DUMP_DIR`.
#[test]
#[ignore = "writes PNGs for a visual check; needs X8_DUMP_DIR and the vanilla jar"]
fn dump_vanilla_scene_for_a_visual_check() {
    use mc_math::{BlockPos, ChunkSectionPos};
    use mc_render::terrain::{fixture, mesh_section, TerrainPhase};

    let Some(dir) = std::env::var_os("X8_DUMP_DIR") else {
        eprintln!("SKIP: set X8_DUMP_DIR");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let Some(jars) = std::env::var_os("MC_JARS_DIR") else {
        eprintln!("SKIP: set MC_JARS_DIR");
        return;
    };
    let jar = std::path::PathBuf::from(jars)
        .join("versions")
        .join("1.8.9")
        .join("1.8.9.jar");
    if !jar.is_file() {
        eprintln!("SKIP: no 1.8.9.jar");
        return;
    }
    let mut rm = mc_assets::ResourceManager::new();
    rm.push_pack(Box::new(
        mc_assets::ZipResourcePack::open(&jar).expect("open 1.8.9.jar"),
    ));
    let (assets, _report) = mc_assets::bake_all(&rm, 2048).expect("bake_all");
    let atlas = assets.atlas.clone();
    mc_assets::model::install(assets);
    mc_render::terrain::tint::load_colormaps(&rm).expect("colormaps");

    let mut w = fixture::blank_world(2);
    let s = fixture::state_of;
    let at = BlockPos::new;
    fixture::fill(
        &mut w,
        at(-32, 16, -32),
        at(47, 16, 47),
        s("minecraft:grass"),
    );
    fixture::fill(&mut w, at(2, 17, 12), at(13, 19, 12), s("minecraft:stone"));
    fixture::fill(&mut w, at(3, 19, 11), at(8, 19, 11), s("minecraft:planks"));
    fixture::fill(&mut w, at(11, 17, 6), at(11, 20, 6), s("minecraft:log"));
    fixture::fill(&mut w, at(9, 21, 4), at(13, 22, 8), s("minecraft:leaves"));
    fixture::fill(
        &mut w,
        at(7, 17, 3),
        at(7, 19, 3),
        s("minecraft:cobblestone"),
    );
    w.set_block(at(4, 17, 5), s("minecraft:stone"));
    w.set_block(at(5, 17, 5), s("minecraft:stone"));
    w.set_block(at(5, 18, 5), s("minecraft:stone"));
    w.set_block(at(9, 17, 8), s("minecraft:glass"));

    let (width, height) = (960, 600);
    let Some(mut gpu) = headless(width, height, 1) else {
        return;
    };
    let view = |pos: DVec3, pitch: f32| {
        let mut cam = Camera {
            pos,
            yaw: 0.0,
            pitch,
            fov: 70.0,
            near: 0.05,
            far: 256.0,
            aspect: width as f32 / height as f32,
            ..Camera::default()
        };
        cam.update();
        cam
    };
    let save = |name: &str, pixels: Vec<u8>| {
        image::RgbaImage::from_raw(width, height, pixels)
            .expect("size")
            .save(dir.join(name))
            .expect("save png");
    };
    let yard = view(DVec3::new(8.0, 20.0, -4.0), 22.0);
    let close = view(DVec3::new(6.5, 18.62, 4.5), 25.0);
    let field = view(DVec3::new(0.5, 17.62, -30.5), 4.0);
    let shots: [(&str, Camera, Option<AoSettings>, u32); 6] = [
        ("x8_vanilla_off.png", yard, None, 0),
        ("x8_vanilla_ao.png", yard, Some(settings(true)), 0),
        ("x8_vanilla_close_off.png", close, None, 0),
        ("x8_vanilla_close_ao.png", close, Some(settings(true)), 0),
        ("x8_vanilla_af_off.png", field, None, 0),
        ("x8_vanilla_af_16x.png", field, None, 16),
    ];
    let mut previous: Option<Vec<u8>> = None;
    for (name, cam, ao, aniso) in shots {
        let mut phases = PhaseList::new();
        phases.clear_color = SKY;
        let terrain = TerrainPhase::install(&gpu, &atlas, &mut phases);
        {
            let mut r = terrain.borrow_mut();
            r.render_distance = 8;
            r.set_anisotropy(aniso);
            for cx in -2..=2 {
                for cz in -2..=2 {
                    r.upload(
                        &gpu,
                        mesh_section(&w, ChunkSectionPos::new(cx, 1, cz)),
                        cam.pos,
                    );
                }
            }
        }
        if let Some(s) = ao {
            phases.insert(
                PhaseOrder::AMBIENT_OCCLUSION,
                Box::new(Handle::new(true, s)),
            );
        }
        let pixels = frame(&mut gpu, &mut phases, &cam);
        // Each AO shot follows its off shot: also write the AO term (on / off, as grey).
        if let (Some(_), Some(off)) = (ao, previous.as_ref()) {
            let term: Vec<u8> = pixels
                .chunks_exact(4)
                .zip(off.chunks_exact(4))
                .flat_map(|(a, b)| {
                    let lum = |p: &[u8]| u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2]);
                    let r = if lum(b) == 0 {
                        255
                    } else {
                        ((lum(a) as f32 / lum(b) as f32).min(1.0) * 255.0) as u8
                    };
                    [r, r, r, 255]
                })
                .collect();
            let amplified: Vec<u8> = term
                .chunks_exact(4)
                .flat_map(|p| {
                    let a = 255u32.saturating_sub((255 - u32::from(p[0])) * 20) as u8;
                    [a, a, a, 255]
                })
                .collect();
            save(&format!("term_{name}"), term);
            save(&format!("amp_{name}"), amplified);
        }
        previous = Some(pixels.clone());
        save(name, pixels);
    }
}

/// `MC_GPU_TIMING=1 cargo test -p mc-render --test x8_ao -- --ignored --nocapture`
#[test]
#[ignore = "timing: needs MC_GPU_TIMING=1; numbers are provisional while other lanes build"]
fn gpu_cost_at_1080p() {
    const FRAMES: u32 = 400;
    let (width, height) = (1920, 1080);
    let Some(mut gpu) = headless(width, height, 1) else {
        return;
    };
    if !gpu.supports_gpu_timing() {
        eprintln!("SKIP: set MC_GPU_TIMING=1 (and the adapter needs timestamp queries)");
        return;
    }
    // The test scene (a third of it sky), and the same scene looked down on so geometry fills
    // the whole frame (every pixel runs the full search: the worst case).
    let scene_cam = camera(width, height);
    let mut down_cam = scene_cam;
    down_cam.pitch = 40.0;
    down_cam.update();
    eprintln!("{width}x{height}, {FRAMES} frames per row, mean GPU ms of the post/ao phase:");
    let views = [("scene", scene_cam), ("all geometry", down_cam)];
    let rows = views.iter().flat_map(|&(view, cam)| {
        [true, false].into_iter().flat_map(move |half| {
            AoQuality::ALL
                .into_iter()
                .map(move |quality| (view, cam, half, quality))
        })
    });
    for (view, cam, half, quality) in rows {
        let ao = Handle::new(
            true,
            AoSettings {
                quality,
                ..settings(half)
            },
        );
        let mut phases = list(&gpu, Some(&ao), true);
        for _ in 0..FRAMES {
            let target = gpu.acquire().expect("acquire");
            phases.render(&gpu, target.view(), &cam, 0.0);
        }
        let _ = gpu.read_pixels();
        let timings = phases.gpu_timings();
        let mean = |name: &str| {
            timings
                .scopes
                .iter()
                .find(|(scope, _)| *scope == name)
                .map_or(f64::NAN, |(_, stats)| stats.mean_ns() / 1e6)
        };
        eprintln!(
            "{view:<12} half={half:<5} {quality:<6?}: post/ao {:.3} ms (frame {:.3} ms, {} frames)",
            mean(AoPhase::NAME),
            timings.frame_ns.mean_ns() / 1e6,
            timings.frames
        );
    }
}

/// X15 (review L1): the hitboxes and the client's nameplates
/// record after AO, at `HITBOX_PHASE_ORDER` (342) and `NAMEPLATE_ORDER` (345), instead of 305 and
/// 320 before it. AO then neither greys a hitbox line nor reads label glyph depth (a halo on the
/// wall behind a label), and with AO off the move changes no byte.
mod x15_overlays_after_ao {
    use super::*;
    use mc_render::entity::hitbox::{HitboxPhase, HITBOX_PHASE_ORDER};
    use mc_render::entity::{NameplateGlyphVertex, NameplatePass, NameplatePhase};

    /// The client's nameplate slot (`mc-client` `driver::NAMEPLATE_ORDER`; its driver test pins
    /// it after AO and after the hitboxes).
    const NAMEPLATE_ORDER: PhaseOrder = PhaseOrder(345);
    /// The orders before the move.
    const OLD_HITBOX_ORDER: PhaseOrder = PhaseOrder(305);
    const OLD_NAMEPLATE_ORDER: PhaseOrder = PhaseOrder(320);
    const LINE: [f32; 4] = [1.0, 0.0, 1.0, 1.0];
    const LABEL: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
    /// A label-sized quad facing the camera, standing on the open floor in front of the wall (clear
    /// of the hand stand-in and the block), so it makes a crease with the floor: `x0, x1, y0, y1,
    /// z`.
    const QUAD: (f32, f32, f32, f32, f32) = (-0.4, 0.4, -EYE, -EYE + 0.4, 4.5);

    /// The scene, `ao` at 340, a hitbox line along the floor/wall crease and a label quad drawn
    /// as one in-front glyph (the pass that writes depth; the unknown sheet falls back to white),
    /// at the given orders. `label: false` leaves the quad out.
    fn render(
        gpu: &mut Gpu,
        cam: &Camera,
        ao: &Handle,
        (hitbox_order, label_order): (PhaseOrder, PhaseOrder),
        label: bool,
    ) -> Vec<u8> {
        let mut phases = list(gpu, Some(ao), true);
        let mut hitboxes = HitboxPhase::new();
        hitboxes.push_line(
            Vec3::new(-2.0, -EYE, WALL_Z),
            Vec3::new(0.5, -EYE, WALL_Z),
            3.0,
            LINE,
        );
        hitboxes.finish(gpu);
        phases.insert(hitbox_order, Box::new(hitboxes));
        let mut labels = NameplatePhase::new(gpu);
        labels.begin();
        if label {
            let (x0, x1, y0, y1, z) = QUAD;
            let v = |x: f32, y: f32| NameplateGlyphVertex {
                pos: [x, y, z],
                uv: [0.5, 0.5],
                color: LABEL,
            };
            labels.push_glyphs(
                NameplatePass::InFront,
                "x15/no-such-sheet",
                &[
                    v(x0, y0),
                    v(x1, y0),
                    v(x1, y1),
                    v(x0, y0),
                    v(x1, y1),
                    v(x0, y1),
                ],
            );
        }
        labels.finish(gpu);
        phases.insert(label_order, Box::new(labels));
        frame(gpu, &mut phases, cam)
    }

    fn rgb(pixels: &[u8], (x, y): (u32, u32)) -> [u8; 3] {
        let p = pixel(pixels, WIDTH, x, y);
        [p[0], p[1], p[2]]
    }

    /// The quad's pixel bounds: `(x0, y0, x1, y1)`, inclusive.
    fn quad_bounds(cam: &Camera) -> (u32, u32, u32, u32) {
        let (x0, x1, y0, y1, z) = QUAD;
        let corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            .map(|(x, y)| project(cam, Vec3::new(x, y, z), WIDTH, HEIGHT));
        let xs = corners.map(|c| c.0);
        let ys = corners.map(|c| c.1);
        (
            *xs.iter().min().unwrap(),
            *ys.iter().min().unwrap(),
            *xs.iter().max().unwrap(),
            *ys.iter().max().unwrap(),
        )
    }

    /// Wall pixels in a 6-pixel ring around the quad (one pixel of slack for its edges).
    fn ring(cam: &Camera) -> Vec<(u32, u32)> {
        let (x0, y0, x1, y1) = quad_bounds(cam);
        let mut out = Vec::new();
        for y in y0 - 7..=y1 + 7 {
            for x in x0 - 7..=x1 + 7 {
                let inside = (x0 - 1..=x1 + 1).contains(&x) && (y0 - 1..=y1 + 1).contains(&y);
                if !inside {
                    out.push((x, y));
                }
            }
        }
        out
    }

    #[test]
    fn ao_off_is_byte_identical_and_ao_on_leaves_hitboxes_and_labels_alone() {
        let Some(mut gpu) = headless(WIDTH, HEIGHT, 1) else {
            return;
        };
        let cam = camera(WIDTH, HEIGHT);
        let old = (OLD_HITBOX_ORDER, OLD_NAMEPLATE_ORDER);
        let new = (HITBOX_PHASE_ORDER, NAMEPLATE_ORDER);
        assert!(new.0 > PhaseOrder::AMBIENT_OCCLUSION && new.0 < new.1);
        assert!(new.1 < PhaseOrder::BLOCK_OVERLAY);

        // AO off: the move changes no byte, and the frame really holds the line and the label.
        let off = Handle::new(false, settings(true));
        let before = render(&mut gpu, &cam, &off, old, true);
        let after = render(&mut gpu, &cam, &off, new, true);
        assert!(
            before == after,
            "AO off: hitboxes 342 / labels 345 draw the 305 / 320 frame"
        );
        assert_eq!(off.with(|p| p.passes_recorded()), 0);
        let line_px = project(&cam, Vec3::new(-1.0, -EYE, WALL_Z), WIDTH, HEIGHT);
        let (x0, y0, x1, y1) = quad_bounds(&cam);
        let label_px = ((x0 + x1) / 2, (y0 + y1) / 2);
        let line = rgb(&after, line_px);
        assert!(
            line[0] > 200 && line[1] < 50 && line[2] > 200,
            "line on the crease: {line:?}"
        );
        let green = rgb(&after, label_px);
        assert!(
            green[0] < 50 && green[1] > 200 && green[2] < 50,
            "label: {green:?}"
        );

        // AO on, old order: AO multiplies the line by the crease's occlusion.
        let on = Handle::new(true, settings(true));
        let shaded = render(&mut gpu, &cam, &on, old, true);
        let grey = rgb(&shaded, line_px);
        assert!(
            u32::from(grey[0]) + 15 < u32::from(line[0]),
            "control: AO before the hitboxes greys the line, {grey:?} vs {line:?}"
        );
        // ...and the label quad where it meets the floor.
        let foot = project(&cam, Vec3::new(0.0, -EYE + 0.03, QUAD.4), WIDTH, HEIGHT);
        let (lit, dark) = (rgb(&after, foot), rgb(&shaded, foot));
        assert!(
            u32::from(dark[1]) + 10 < u32::from(lit[1]),
            "control: AO after the labels shades them, {dark:?} vs {lit:?}"
        );
        // AO on, new order: the line and the label keep their AO-off colour exactly.
        let clean = render(&mut gpu, &cam, &on, new, true);
        assert_eq!(rgb(&clean, foot), lit);
        assert!(on.with(|p| p.passes_recorded()) > 0, "AO really ran");
        assert_eq!(rgb(&clean, line_px), line);
        for y in y0 + 1..y1 {
            for x in x0 + 1..x1 {
                assert_eq!(
                    rgb(&clean, (x, y)),
                    rgb(&after, (x, y)),
                    "label pixel ({x}, {y})"
                );
            }
        }

        // Around the label: with the labels after AO, AO never sees the quad's depth, so the ring
        // is the frame without the label byte for byte; before AO the quad shades its surroundings.
        let no_label = render(&mut gpu, &cam, &on, new, false);
        let ring = ring(&cam);
        assert!(ring.iter().all(|&p| rgb(&clean, p) == rgb(&no_label, p)));
        let halo = ring
            .iter()
            .filter(|&&p| rgb(&shaded, p) != rgb(&no_label, p))
            .count();
        assert!(
            halo > 0,
            "control: AO after the labels darkens the floor around them"
        );
        eprintln!(
            "x15: line {grey:?} and label foot {dark:?} with AO first; {halo}/{} ring pixels \
             haloed",
            ring.len()
        );
    }
}
