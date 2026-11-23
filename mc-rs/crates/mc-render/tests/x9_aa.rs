//! Anti-aliasing, motion vectors and per-pixel motion blur on
//! a real GPU.
//!
//! Every test renders whole frames through a real [`PhaseList`] on a headless context and reads
//! them back. They skip (printing `SKIP:`) without an adapter.
//!
//! * Off: the three X9 phases installed but off leave every frame byte-identical to a frame
//!   without them, and allocate nothing.
//! * The jitter reaches only the world: a phase at or after `PhaseOrder::TEMPORAL_AA` reads the
//!   unjittered camera (the in-encoder restore), one before it the jittered one.
//! * TAA turns a hard slanted edge into a stable gradient and leaves flat areas and post-world
//!   phases alone.
//! * FXAA smooths a geometric edge; with "Texture edges" off it leaves texture detail on a flat
//!   surface byte-identical.
//! * The motion target: zero for a still camera (even jittered), the projected angle for a turn,
//!   and an entity box's own motion only on the entity's pixels.
//! * The per-pixel blur smears a turning view and leaves a still one untouched.
//! * `gpu_cost_at_1080p` (ignored): per-mode GPU time at 1920x1080 with `MC_GPU_TIMING=1`.

use std::cell::RefCell;
use std::rc::Rc;

use glam::{DVec3, Vec2, Vec3};
use mc_render::post::taa::{AaMode, AntiAliasPhase, JITTER_PHASES};
use mc_render::post::velocity::{ObjectMotion, VelocityConsumers, VelocityPhase};
use mc_render::post::velocity_blur::VelocityBlurPhase;
use mc_render::{
    Camera, DepthAttachment, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase,
    SceneTextures, DEPTH_FORMAT, HEADLESS_FORMAT,
};
use wgpu::util::DeviceExt;

const W: u32 = 96;
const H: u32 = 96;

// ------------------------------------------------------------------------------- harness

fn headless(width: u32, height: u32) -> Option<Gpu> {
    match Gpu::headless(
        width,
        height,
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

/// The client's `Shared`: the list owns one handle, the test keeps the other.
struct Handle<T>(Rc<RefCell<T>>);

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Handle(Rc::clone(&self.0))
    }
}

impl<T> Handle<T> {
    fn new(value: T) -> Handle<T> {
        Handle(Rc::new(RefCell::new(value)))
    }

    fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        f(&mut self.0.borrow_mut())
    }
}

impl<T: RenderPhase> RenderPhase for Handle<T> {
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
        self.0.borrow().name()
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

    fn record_scene(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        self.0.borrow_mut().record_scene(gpu, enc, view, cam, scene);
    }
}

/// Test geometry drawn through the shared camera uniform, like every world phase.
const WORLD_WGSL: &str = r#"
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
    @location(0) color: vec4<f32>,
    @location(1) world: vec3<f32>,
};

@vertex
fn vs(@location(0) pos: vec3<f32>, @location(1) color: vec4<f32>) -> VsOut {
    var out: VsOut;
    out.clip = camera.view_proj * vec4<f32>(pos, 1.0);
    out.color = color;
    out.world = pos;
    return out;
}

@fragment
fn fs_flat(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}

// Texture-like detail on a flat surface: a 2-block checker.
@fragment
fn fs_checker(in: VsOut) -> @location(0) vec4<f32> {
    let cell = (i32(floor(in.world.x * 0.5)) + i32(floor(in.world.y * 0.5))) & 1;
    return select(vec4<f32>(0.2, 0.2, 0.2, 1.0), in.color, cell == 1);
}

// Vertical stripes, 2 blocks wide.
@fragment
fn fs_stripes(in: VsOut) -> @location(0) vec4<f32> {
    let cell = i32(floor(in.world.x * 0.5)) & 1;
    return select(vec4<f32>(0.0, 0.0, 0.0, 1.0), in.color, cell == 1);
}
"#;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
    color: [f32; 4],
}

fn tri(a: [f32; 3], b: [f32; 3], c: [f32; 3], color: [f32; 4]) -> [Vertex; 3] {
    [a, b, c].map(|pos| Vertex { pos, color })
}

/// An axis-aligned quad facing the camera (which looks along +z) at depth `z`.
fn quad(x0: f32, y0: f32, x1: f32, y1: f32, z: f32, color: [f32; 4]) -> Vec<Vertex> {
    let mut v = Vec::new();
    v.extend(tri([x0, y0, z], [x1, y0, z], [x1, y1, z], color));
    v.extend(tri([x0, y0, z], [x1, y1, z], [x0, y1, z], color));
    v
}

struct World {
    name: &'static str,
    pipeline: wgpu::RenderPipeline,
    vertices: wgpu::Buffer,
    count: u32,
    depth: DepthAttachment,
}

impl World {
    fn new(
        gpu: &Gpu,
        name: &'static str,
        entry: &'static str,
        vertices: &[Vertex],
        depth: DepthAttachment,
    ) -> World {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("x9 test world"),
                source: wgpu::ShaderSource::Wgsl(WORLD_WGSL.into()),
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("x9 test world"),
                bind_group_layouts: &[Some(gpu.camera_layout())],
                immediate_size: 0,
            });
        let attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("x9 test world"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &attributes,
                    })],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HEADLESS_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        let buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("x9 test world"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        World {
            name,
            pipeline,
            vertices: buffer,
            count: vertices.len() as u32,
            depth,
        }
    }
}

impl RenderPhase for World {
    fn name(&self) -> &'static str {
        self.name
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        let mut pass = gpu.begin_pass(enc, view, self.name, self.depth);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.draw(0..self.count, 0..1);
    }
}

const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const GREY: [f32; 4] = [0.5, 0.5, 0.5, 1.0];
const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

/// A black background far away and a white triangle with a slanted edge across the middle.
fn edge_scene(gpu: &Gpu) -> World {
    let mut v = quad(-40.0, -40.0, 40.0, 40.0, 20.0, [0.0, 0.0, 0.0, 1.0]);
    v.extend(tri(
        [-10.0, -10.0, 5.0],
        [10.0, -10.0, 5.0],
        [-10.0, 5.0, 5.0],
        WHITE,
    ));
    World::new(gpu, "test/world", "fs_flat", &v, DepthAttachment::Shared)
}

/// A slanted red triangle in the bottom-left corner, drawn like the hand: depth cleared first.
fn late_triangle(gpu: &Gpu) -> World {
    let v = tri([0.6, -1.2, 3.0], [1.4, -1.8, 3.0], [0.8, -2.0, 3.0], RED);
    World::new(gpu, "test/late", "fs_flat", &v, DepthAttachment::Clear)
}

/// Which pixels the late (hand-like) phase covered: the only red ones.
fn red_mask(pixels: &[u8]) -> Vec<bool> {
    pixels
        .chunks_exact(4)
        .map(|p| p[0] > 200 && p[1] < 50)
        .collect()
}

/// The camera-relative point where pixel `(x, y)`'s centre ray meets the plane `z`.
fn on_plane(cam: &Camera, x: u32, y: u32, z: f32) -> Vec3 {
    let ndc = Vec2::new(
        (x as f32 + 0.5) / W as f32 * 2.0 - 1.0,
        1.0 - (y as f32 + 0.5) / H as f32 * 2.0,
    );
    let inv = cam.view_proj().inverse();
    let a = inv.project_point3(ndc.extend(0.0));
    let b = inv.project_point3(ndc.extend(1.0));
    a + (b - a) * ((z - a.z) / (b.z - a.z))
}

/// Where camera-relative `p` lands, in UV.
fn uv_of(cam: &Camera, p: Vec3) -> Vec2 {
    let c = cam.view_proj() * p.extend(1.0);
    let n = c.truncate().truncate() / c.w;
    Vec2::new(n.x * 0.5 + 0.5, 0.5 - n.y * 0.5)
}

fn camera(yaw: f32) -> Camera {
    let mut cam = Camera {
        yaw,
        pos: DVec3::new(0.0, 64.0, 0.0),
        ..Camera::default()
    };
    cam.set_viewport(W, H);
    cam.far = 128.0;
    cam.update();
    cam
}

fn draw(gpu: &mut Gpu, phases: &mut PhaseList, cam: &Camera) {
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), cam, 0.0);
}

fn frame(gpu: &mut Gpu, phases: &mut PhaseList, cam: &Camera) -> Vec<u8> {
    draw(gpu, phases, cam);
    gpu.read_pixels().expect("readback")
}

fn px(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

/// The three X9 phases, wired the way the client wires them.
struct X9 {
    velocity: Handle<VelocityPhase>,
    aa: Handle<AntiAliasPhase>,
    blur: Handle<VelocityBlurPhase>,
}

impl X9 {
    fn new(gpu: &Gpu) -> X9 {
        let mut velocity = VelocityPhase::new(gpu);
        velocity.set_fixed_dt(Some(1.0 / 240.0));
        let output = velocity.output();
        let mut aa = AntiAliasPhase::new(gpu);
        aa.set_velocity(output.clone());
        let mut blur = VelocityBlurPhase::new(gpu);
        blur.set_velocity(output);
        X9 {
            velocity: Handle::new(velocity),
            aa: Handle::new(aa),
            blur: Handle::new(blur),
        }
    }

    fn install(&self, phases: &mut PhaseList) {
        phases.insert(PhaseOrder::TEMPORAL_AA, Box::new(self.velocity.clone()));
        phases.insert(PhaseOrder::TEMPORAL_AA, Box::new(self.aa.clone()));
        phases.insert(PhaseOrder::VELOCITY_BLUR, Box::new(self.blur.clone()));
    }

    fn in_world(&self, on: bool) {
        self.velocity.with(|p| p.set_in_world(on));
        self.aa.with(|p| p.set_in_world(on));
        self.blur.with(|p| p.set_in_world(on));
    }
}

fn list(world: Vec<World>, x9: Option<&X9>) -> PhaseList {
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    for w in world {
        let order = if w.name == "test/late" {
            PhaseOrder::FIRST_PERSON
        } else {
            PhaseOrder::TERRAIN_SOLID
        };
        phases.insert(order, Box::new(w));
    }
    if let Some(x9) = x9 {
        x9.install(&mut phases);
    }
    phases
}

/// Pixels strictly between the two flat colours of the edge scene.
fn intermediate(pixels: &[u8]) -> usize {
    pixels
        .chunks_exact(4)
        .filter(|p| p[0] > 12 && p[0] < 243)
        .count()
}

/// Rgba16Float to f32.
fn f16(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    sign * match exponent {
        0 => mantissa * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + mantissa / 1024.0) * 2f32.powi(e - 15),
    }
}

/// The motion target, as `[r, g, b, a]` per pixel.
fn read_velocity(gpu: &Gpu, velocity: &Handle<VelocityPhase>) -> Vec<[f32; 4]> {
    let output = velocity.with(|p| p.output());
    let frame = output.get();
    let texture = &frame.as_ref().expect("the velocity pass ran").texture;
    let row = W * 8;
    assert_eq!(row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT, 0);
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("x9 velocity readback"),
        size: u64::from(row * H),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(Some(enc.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    rx.recv().expect("map callback").expect("map");
    let bytes = staging.get_mapped_range(..).expect("mapped").to_vec();
    bytes
        .chunks_exact(8)
        .map(|t| {
            let h = |i: usize| f16(u16::from_le_bytes([t[i], t[i + 1]]));
            [h(0), h(2), h(4), h(6)]
        })
        .collect()
}

// ---------------------------------------------------------------------------------- tests

/// The phases installed but off (and on with no mode) draw nothing, allocate nothing and leave
/// the frame byte-identical, and the camera uniform is today's (the world draws through it).
#[test]
fn installed_but_off_is_byte_identical() {
    let Some(mut gpu) = headless(W, H) else {
        return;
    };
    let cam = camera(10.0);
    let mut plain = list(vec![edge_scene(&gpu), late_triangle(&gpu)], None);
    let reference = frame(&mut gpu, &mut plain, &cam);
    assert!(intermediate(&reference) == 0, "the raw edge is hard");

    let x9 = X9::new(&gpu);
    let mut phases = list(vec![edge_scene(&gpu), late_triangle(&gpu)], Some(&x9));
    // Not in a world.
    assert_eq!(frame(&mut gpu, &mut phases, &cam), reference);
    // In a world, every mode off.
    x9.in_world(true);
    assert!(!phases.scene_mode());
    assert_eq!(frame(&mut gpu, &mut phases, &cam), reference);
    // A mode chosen but not in a world.
    x9.in_world(false);
    x9.aa.with(|p| p.set_mode(AaMode::Taa));
    x9.velocity.with(|p| {
        p.set_consumers(VelocityConsumers {
            taa: true,
            blur: true,
            blur_objects: true,
        })
    });
    x9.blur.with(|p| p.set_on(true));
    assert!(!phases.scene_mode());
    assert_eq!(x9.aa.with(|p| p.next_jitter(W, H)), Vec2::ZERO);
    assert_eq!(frame(&mut gpu, &mut phases, &cam), reference);
    assert_eq!(x9.aa.with(|p| p.passes_recorded()), 0);
    assert_eq!(x9.velocity.with(|p| p.frames_recorded()), 0);
    assert_eq!(x9.blur.with(|p| p.passes_recorded()), 0);
    assert!(!x9.velocity.with(|p| p.target_allocated()));
    assert!(!x9.aa.with(|p| p.history_allocated()));
}

/// With a non-zero jitter the world moves and a phase at or after `TEMPORAL_AA` does not: the
/// list restores the unjittered camera in the encoder. The negative control puts the late phase
/// just before `TEMPORAL_AA`, where it is jittered.
#[test]
fn the_jitter_reaches_only_the_world_phases() {
    let Some(mut gpu) = headless(W, H) else {
        return;
    };
    let still = camera(10.0);
    let mut jittered = still;
    jittered.jitter = Vec2::new(1.0 / W as f32, -1.0 / H as f32); // half a pixel each way
    let count = |mask: &[bool]| mask.iter().filter(|&&red| red).count();

    // The late phase at FIRST_PERSON (650 >= TEMPORAL_AA): restored.
    let mut phases = list(vec![edge_scene(&gpu), late_triangle(&gpu)], None);
    let a = frame(&mut gpu, &mut phases, &still);
    let b = frame(&mut gpu, &mut phases, &jittered);
    assert!(count(&red_mask(&a)) > 20, "the late triangle is on screen");
    assert_eq!(red_mask(&a), red_mask(&b), "the post-world phase moved");
    assert_ne!(
        a, b,
        "the world phase did not move: the jitter never reached it"
    );

    // Negative control: the same triangle at 641, before the restore, moves.
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::TERRAIN_SOLID, Box::new(edge_scene(&gpu)));
    phases.insert(
        PhaseOrder(PhaseOrder::TEMPORAL_AA.0 - 1),
        Box::new(late_triangle(&gpu)),
    );
    let a = frame(&mut gpu, &mut phases, &still);
    let b = frame(&mut gpu, &mut phases, &jittered);
    assert_ne!(
        red_mask(&a),
        red_mask(&b),
        "a phase before TEMPORAL_AA must see the jitter"
    );
}

/// TAA over 40 frames: the slanted edge becomes a gradient that no longer changes from one
/// jitter cycle to the next; flat areas keep their exact colour; the late (hand) phase stays
/// byte-identical to a frame without anti-aliasing.
#[test]
fn taa_turns_a_hard_edge_into_a_stable_gradient() {
    let Some(mut gpu) = headless(W, H) else {
        return;
    };
    let mut cam = camera(10.0);
    let mut plain = list(vec![edge_scene(&gpu), late_triangle(&gpu)], None);
    let reference = frame(&mut gpu, &mut plain, &cam);

    let x9 = X9::new(&gpu);
    let mut phases = list(vec![edge_scene(&gpu), late_triangle(&gpu)], Some(&x9));
    x9.in_world(true);
    x9.aa.with(|p| p.set_mode(AaMode::Taa));
    x9.velocity.with(|p| {
        p.set_consumers(VelocityConsumers {
            taa: true,
            ..VelocityConsumers::default()
        })
    });
    const SETTLE: usize = 80;
    let cycle = JITTER_PHASES as usize;
    let mut frames = Vec::new();
    for _ in 0..(SETTLE + cycle) {
        cam.jitter = x9.aa.with(|p| p.next_jitter(W, H));
        assert_ne!(cam.jitter, Vec2::ZERO);
        frames.push(frame(&mut gpu, &mut phases, &cam));
    }
    let settled = &frames[SETTLE - 1];
    let cycle_later = &frames[SETTLE - 1 + cycle];
    let soft = intermediate(settled);
    assert!(soft >= 40, "only {soft} edge pixels are in between");
    let worst = settled
        .iter()
        .zip(cycle_later.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    assert!(
        worst <= 4,
        "the resolve still moves by {worst} levels per cycle"
    );
    // Frame-to-frame change inside a settled cycle (the exponential history's ripple).
    let ripple = (SETTLE..SETTLE + cycle)
        .flat_map(|i| {
            frames[i]
                .iter()
                .zip(frames[i - 1].iter())
                .map(|(a, b)| a.abs_diff(*b))
        })
        .max()
        .unwrap();
    eprintln!("TAA: {soft} soft edge pixels, cycle drift {worst}, per-frame ripple {ripple}");
    // Flat areas: deep inside the triangle and the background.
    assert_eq!(px(settled, 88, 88)[..3], [255, 255, 255]);
    assert_eq!(px(settled, 8, 8)[..3], [0, 0, 0]);
    // The hand-like phase is drawn after the resolve, unjittered.
    assert_eq!(red_mask(settled), red_mask(&reference));
    for (i, (a, b)) in settled
        .chunks_exact(4)
        .zip(reference.chunks_exact(4))
        .enumerate()
    {
        if b[0] > 200 && b[1] < 50 {
            assert_eq!(a, b, "late pixel {i} changed");
        }
    }
    assert!(x9.aa.with(|p| p.history_allocated()));

    // Sharpness: still stable, and the edge stays soft.
    x9.aa.with(|p| p.set_sharpness(0.5));
    let mut last = Vec::new();
    for _ in 0..(2 * JITTER_PHASES) {
        cam.jitter = x9.aa.with(|p| p.next_jitter(W, H));
        last = frame(&mut gpu, &mut phases, &cam);
    }
    assert!(intermediate(&last) >= 30);

    // Off again: the history goes and the frame is today's.
    x9.aa.with(|p| p.set_mode(AaMode::Off));
    x9.velocity
        .with(|p| p.set_consumers(VelocityConsumers::default()));
    cam.jitter = x9.aa.with(|p| p.next_jitter(W, H));
    assert_eq!(cam.jitter, Vec2::ZERO);
    assert_eq!(frame(&mut gpu, &mut phases, &cam), reference);
    assert!(!x9.aa.with(|p| p.history_allocated()));
    assert!(!x9.velocity.with(|p| p.target_allocated()));
}

/// FXAA softens the triangle's silhouette. With "Texture edges" off the checker on the flat
/// background is untouched to the byte; with it on, classic FXAA changes the checker too.
#[test]
fn fxaa_smooths_geometry_and_leaves_flat_textures_alone() {
    let Some(mut gpu) = headless(W, H) else {
        return;
    };
    let cam = camera(10.0);
    let scene = |gpu: &Gpu| {
        let back = World::new(
            gpu,
            "test/world-back",
            "fs_checker",
            &quad(-40.0, -40.0, 40.0, 40.0, 20.0, WHITE),
            DepthAttachment::Shared,
        );
        let front = World::new(
            gpu,
            "test/world",
            "fs_flat",
            &tri(
                [-10.0, -10.0, 5.0],
                [10.0, -10.0, 5.0],
                [-10.0, 5.0, 5.0],
                GREY,
            ),
            DepthAttachment::Shared,
        );
        vec![back, front]
    };
    let mut plain = list(scene(&gpu), None);
    let reference = frame(&mut gpu, &mut plain, &cam);

    let x9 = X9::new(&gpu);
    let mut phases = list(scene(&gpu), Some(&x9));
    x9.in_world(true);
    x9.aa.with(|p| p.set_mode(AaMode::Fxaa));
    let geometry = frame(&mut gpu, &mut phases, &cam);
    // The checker region far from the triangle: the top right corner.
    let corner = |pixels: &[u8]| -> Vec<[u8; 4]> {
        (0..30)
            .flat_map(|y| (60..W).map(move |x| (x, y)))
            .map(|(x, y)| px(pixels, x, y))
            .collect()
    };
    assert_eq!(
        corner(&geometry),
        corner(&reference),
        "flat texture detail changed"
    );
    let changed = geometry
        .chunks_exact(4)
        .zip(reference.chunks_exact(4))
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        changed >= 20,
        "FXAA changed only {changed} pixels along the silhouette"
    );

    x9.aa.with(|p| p.set_texture_edges(true));
    let classic = frame(&mut gpu, &mut phases, &cam);
    assert_ne!(
        corner(&classic),
        corner(&reference),
        "classic FXAA leaves the checker alone"
    );
}

/// The motion target: zero for a still (jittered) camera, the projected angle for a 1-degree
/// turn, and an entity's own motion on its pixels only.
#[test]
fn the_motion_target_measures_camera_and_object_motion() {
    let Some(mut gpu) = headless(W, H) else {
        return;
    };
    let x9 = X9::new(&gpu);
    let mut phases = list(vec![edge_scene(&gpu)], Some(&x9));
    x9.in_world(true);
    x9.velocity.with(|p| {
        p.set_consumers(VelocityConsumers {
            taa: true,
            ..VelocityConsumers::default()
        })
    });
    let mut cam = camera(10.0);
    draw(&mut gpu, &mut phases, &cam);
    cam.jitter = Vec2::new(0.6 / W as f32, 0.2 / H as f32);
    draw(&mut gpu, &mut phases, &cam);
    let still = read_velocity(&gpu, &x9.velocity);
    assert!(
        still.iter().all(|v| v[0] == 0.0 && v[1] == 0.0),
        "a still camera has motion"
    );

    // Turn right by one degree. Pixel (48, 48) sees the background plane at z = 20.
    let before = cam;
    let turned = camera(11.0);
    draw(&mut gpu, &mut phases, &turned);
    let motion = read_velocity(&gpu, &x9.velocity);
    let centre = motion[(48 * W + 48) as usize];
    let p = on_plane(&turned, 48, 48, 20.0);
    let expected = uv_of(&turned, p) - uv_of(&before, p);
    assert!(expected.x < 0.0, "turning right moves the world left");
    assert!(
        (centre[0] - expected.x).abs() < 0.01 * expected.x.abs()
            && (centre[1] - expected.y).abs() < 1e-4,
        "{centre:?} vs {expected}"
    );

    // An entity box around the white triangle's lower right part, moving east at 2 blocks/s.
    let velocity = Vec3::new(2.0, 0.0, 0.0);
    x9.velocity.with(|p| {
        p.set_objects(&[ObjectMotion {
            min: Vec3::new(-10.0, -10.0, 4.5),
            max: Vec3::new(0.0, 0.0, 5.5),
            frame_delta: velocity / 240.0,
            velocity,
        }])
    });
    draw(&mut gpu, &mut phases, &turned); // the camera holds still now
    let motion = read_velocity(&gpu, &x9.velocity);
    // Pixel (80, 80): on the triangle (z = 5), inside the box. Pixel (60, 20): background.
    let on = motion[(80 * W + 80) as usize];
    let off = motion[(20 * W + 60) as usize];
    let p = on_plane(&turned, 80, 80, 5.0);
    assert!(p.x < 0.0 && p.x > -10.0 && p.y < 0.0 && p.y > -10.0, "{p}");
    let moved = velocity / 240.0;
    let expected = uv_of(&turned, p) - uv_of(&turned, p - moved);
    assert!(expected.x < 0.0, "east is screen-left when facing south");
    assert!(
        (on[0] - expected.x).abs() < 0.03 * expected.x.abs() && (on[1] - expected.y).abs() < 1e-4,
        "{on:?} vs {expected}"
    );
    assert_eq!(
        [off[0], off[1]],
        [0.0, 0.0],
        "the background got the entity's motion"
    );
}

/// Camera motion blur: a turning view of stripes gets soft stripe edges along the turn; once
/// the view stops (and the trail decays) the frame is untouched.
#[test]
fn per_pixel_blur_smears_a_turning_view_only() {
    let Some(mut gpu) = headless(W, H) else {
        return;
    };
    let stripes = |gpu: &Gpu| {
        vec![World::new(
            gpu,
            "test/world",
            "fs_stripes",
            &quad(-60.0, -60.0, 60.0, 60.0, 10.0, WHITE),
            DepthAttachment::Shared,
        )]
    };
    let x9 = X9::new(&gpu);
    let mut phases = list(stripes(&gpu), Some(&x9));
    x9.in_world(true);
    x9.velocity.with(|p| {
        p.set_consumers(VelocityConsumers {
            blur: true,
            ..VelocityConsumers::default()
        });
        p.set_exposure(0.05);
    });
    x9.blur.with(|p| p.set_on(true));
    let mut yaw = 0.0;
    let mut turning = Vec::new();
    for _ in 0..60 {
        yaw += 0.25; // 60 degrees per second at 240 FPS
        turning = frame(&mut gpu, &mut phases, &camera(yaw));
    }
    assert!(
        intermediate(&turning) > 200,
        "turning: only {} soft pixels",
        intermediate(&turning)
    );
    assert!(x9.blur.with(|p| p.passes_recorded()) > 0);
    // Stop and let the trail decay (0.05 s exposure, 1/240 s frames).
    let mut still = Vec::new();
    for _ in 0..120 {
        still = frame(&mut gpu, &mut phases, &camera(yaw));
    }
    let mut plain = list(stripes(&gpu), None);
    assert_eq!(still, frame(&mut gpu, &mut plain, &camera(yaw)));
}

/// GPU time of each mode at 1920x1080 (`MC_GPU_TIMING=1`). Prints a table; provisional while
/// other work shares the GPU.
#[test]
#[ignore = "timing run: MC_GPU_TIMING=1 cargo test -p mc-render --test x9_aa -- --ignored"]
fn gpu_cost_at_1080p() {
    const FRAMES: u32 = 400;
    let (width, height) = (1920, 1080);
    let Some(mut gpu) = headless(width, height) else {
        return;
    };
    if !gpu.supports_gpu_timing() {
        eprintln!("SKIP: set MC_GPU_TIMING=1 (and the adapter needs timestamp queries)");
        return;
    }
    let scene = |gpu: &Gpu| {
        let mut v = quad(-60.0, -60.0, 60.0, 60.0, 30.0, WHITE);
        for i in 0..12 {
            let x = -18.0 + 3.0 * i as f32;
            v.extend(tri(
                [x, -8.0, 6.0 + i as f32],
                [x + 2.5, -8.0, 6.0 + i as f32],
                [x + 0.7, 8.0, 6.0 + i as f32],
                GREY,
            ));
        }
        vec![World::new(
            gpu,
            "test/world",
            "fs_checker",
            &v,
            DepthAttachment::Shared,
        )]
    };
    #[derive(Clone, Copy)]
    struct Config {
        label: &'static str,
        mode: AaMode,
        texture_edges: bool,
        sharpness: f32,
        blur: bool,
        objects: bool,
    }
    let base = Config {
        label: "",
        mode: AaMode::Off,
        texture_edges: false,
        sharpness: 0.0,
        blur: false,
        objects: false,
    };
    let configs = [
        Config {
            label: "off",
            ..base
        },
        Config {
            label: "FXAA (geometry)",
            mode: AaMode::Fxaa,
            ..base
        },
        Config {
            label: "FXAA (textures)",
            mode: AaMode::Fxaa,
            texture_edges: true,
            ..base
        },
        Config {
            label: "TAA",
            mode: AaMode::Taa,
            ..base
        },
        Config {
            label: "TAA sharp 50%",
            mode: AaMode::Taa,
            sharpness: 0.5,
            ..base
        },
        Config {
            label: "blur camera",
            blur: true,
            ..base
        },
        Config {
            label: "blur cam+obj",
            blur: true,
            objects: true,
            ..base
        },
        Config {
            label: "TAA + blur",
            mode: AaMode::Taa,
            blur: true,
            objects: true,
            ..base
        },
    ];
    eprintln!("{width}x{height}, {FRAMES} frames per row, turning camera, mean ms per frame:");
    eprintln!(
        "config           | post/velocity | post/aa | post/velocity-blur | scene_blit | gpu.frame"
    );
    for config in configs {
        let x9 = X9::new(&gpu);
        let mut phases = list(scene(&gpu), Some(&x9));
        x9.in_world(true);
        x9.aa.with(|p| {
            p.set_mode(config.mode);
            p.set_texture_edges(config.texture_edges);
            p.set_sharpness(config.sharpness);
        });
        x9.velocity.with(|p| {
            p.set_consumers(VelocityConsumers {
                taa: config.mode == AaMode::Taa,
                blur: config.blur,
                blur_objects: config.objects,
            });
            p.set_exposure(0.012);
            if config.objects {
                let boxes: Vec<ObjectMotion> = (0..16)
                    .map(|i| ObjectMotion {
                        min: Vec3::new(-16.0 + 2.0 * i as f32, -1.0, 8.0),
                        max: Vec3::new(-15.4 + 2.0 * i as f32, 0.8, 8.6),
                        frame_delta: Vec3::new(4.0 / 240.0, 0.0, 0.0),
                        velocity: Vec3::new(4.0, 0.0, 0.0),
                    })
                    .collect();
                p.set_objects(&boxes);
            }
        });
        x9.blur.with(|p| p.set_on(config.blur));
        let mut cam = Camera {
            pos: DVec3::new(0.0, 64.0, 0.0),
            ..Camera::default()
        };
        cam.set_viewport(width, height);
        for i in 0..FRAMES {
            cam.yaw = (i as f32 * 0.2).sin() * 10.0;
            cam.update();
            cam.jitter = x9.aa.with(|p| p.next_jitter(width, height));
            let f = gpu.acquire().expect("acquire");
            phases.render(&gpu, f.view(), &cam, 0.0);
            drop(f);
            if i % 50 == 49 {
                let _ = gpu.read_pixels();
            }
        }
        let timings = phases.gpu_timings();
        let mean_ms = |name: &str| {
            timings
                .scopes
                .iter()
                .find(|(scope, _)| *scope == name)
                .map_or(0.0, |(_, stats)| stats.mean_ns() / 1e6)
        };
        eprintln!(
            "{:<16} | {:>13.4} | {:>7.4} | {:>18.4} | {:>10.4} | {:>9.4}",
            config.label,
            mean_ms(VelocityPhase::NAME),
            mean_ms(AntiAliasPhase::NAME),
            mean_ms(VelocityBlurPhase::NAME),
            mean_ms("gpu.scene_blit"),
            timings.frame_ns.mean_ns() / 1e6,
        );
        assert!(timings.frames > 0, "{}: no frame was timed", config.label);
    }
}
