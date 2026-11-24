//! Vanilla's cloud depth. `EntityRenderer.renderWorldPass` draws the clouds at one of two points,
//! picked by the render-view entity's eye height against a literal 128 (`:1364`, `:1474`), with
//! depth writes on both times:
//!
//! * **eye below 128:** after the sky and before terrain, so terrain behind a cloud is
//!   depth-rejected (the cloud hides it) and terrain in front of a cloud draws over it;
//! * **eye at or above 128:** after the world, depth-tested against it, so the clouds blend over
//!   the ground beneath them and anything rising in front of them hides them.
//!
//! Fast and fancy clouds, without and with 4x MSAA. A synthetic `clouds.png` (every texel opaque,
//! or every texel clear as a control) and solid-colour boxes standing in for blocks, so nothing
//! here needs the vanilla jar. Skips (printing `SKIP:`) without a GPU adapter.
//!
//! ```text
//! cargo test -p mc-render --test clouds_vanilla_depth -- --nocapture
//! ```

use std::cell::{Cell, RefCell};
use std::io::{Cursor, Read};
use std::rc::Rc;

use glam::{DVec3, Vec3};
use image::ImageEncoder;
use mc_assets::{ResourceLocation, ResourceManager, ResourcePack};
use mc_render::pipeline::PassTarget;
use mc_render::sky::clouds::{CloudMode, CloudsPhase, ABOVE_CLOUDS_EYE_Y, ABOVE_CLOUDS_ORDER};
use mc_render::sky::{
    Dimension, Environment, FogDistance, FogInputs, FogPhase, FogState, SkyPhase, Weather,
};
use mc_render::{
    Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase, DEPTH_FORMAT,
};

const W: u32 = 192;
const H: u32 = 144;

/// A pack holding only `clouds.png`.
struct CloudsPng(Vec<u8>);

impl ResourcePack for CloudsPng {
    fn open(&self, l: &ResourceLocation) -> Option<Box<dyn Read>> {
        self.contains(l)
            .then(|| Box::new(Cursor::new(self.0.clone())) as Box<dyn Read>)
    }

    fn contains(&self, l: &ResourceLocation) -> bool {
        l.namespace == "minecraft" && l.path == "textures/environment/clouds.png"
    }
}

/// A 256x256 white `clouds.png` whose every texel has alpha `alpha`: 255 is cloud everywhere,
/// 1 (vanilla's "empty" texel) is cloud nowhere, since the clouds' alpha test kills it.
fn clouds_texture(alpha: u8) -> ResourceManager {
    let mut rgba = vec![255_u8; 256 * 256 * 4];
    for px in rgba.chunks_exact_mut(4) {
        px[3] = alpha;
    }
    let mut png = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png)
        .write_image(&rgba, 256, 256, image::ExtendedColorType::Rgba8)
        .expect("encode clouds.png");
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(CloudsPng(png)));
    rm
}

fn headless(sample_count: u32) -> Option<Gpu> {
    match Gpu::headless(
        W,
        H,
        GpuOptions {
            vsync: false,
            sample_count,
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

// ---- boxes: opaque, depth-writing stand-ins for blocks ------------------------------------

const BOX_WGSL: &str = r"
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
};

@vertex
fn vs(@location(0) pos: vec3<f32>, @location(1) color: vec4<f32>) -> VsOut {
    var out: VsOut;
    out.clip = camera.view_proj * vec4<f32>(pos, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
";

/// An axis-aligned box in world space, one flat colour.
#[derive(Debug, Clone, Copy)]
struct Cuboid {
    min: DVec3,
    max: DVec3,
    rgba: [f32; 4],
}

/// Opaque boxes drawn through the terrain projection with `GL_LEQUAL` and depth writes, as solid
/// terrain is. They join the shared world pass, as the opaque terrain layers do.
struct Boxes {
    cuboids: Vec<Cuboid>,
    pipeline: wgpu::RenderPipeline,
    buffer: wgpu::Buffer,
}

impl Boxes {
    fn new(gpu: &Gpu, cuboids: Vec<Cuboid>) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("test boxes"),
                source: wgpu::ShaderSource::Wgsl(BOX_WGSL.into()),
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("test boxes"),
                bind_group_layouts: &[Some(gpu.camera_layout())],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("test boxes"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: 7 * 4,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
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
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test boxes"),
            size: (cuboids.len().max(1) * 36 * 7 * 4) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            cuboids,
            pipeline,
            buffer,
        }
    }

    fn record_to(&mut self, gpu: &Gpu, target: PassTarget<'_>, cam: &Camera) {
        if self.cuboids.is_empty() {
            return;
        }
        let mut data: Vec<f32> = Vec::with_capacity(self.cuboids.len() * 36 * 7);
        for c in &self.cuboids {
            let corner = |i: usize| -> Vec3 {
                cam.relative(DVec3::new(
                    if i & 1 == 0 { c.min.x } else { c.max.x },
                    if i & 2 == 0 { c.min.y } else { c.max.y },
                    if i & 4 == 0 { c.min.z } else { c.max.z },
                ))
            };
            for face in [
                [0, 1, 3, 2],
                [4, 6, 7, 5],
                [0, 4, 5, 1],
                [2, 3, 7, 6],
                [0, 2, 6, 4],
                [1, 5, 7, 3],
            ] {
                for i in [0, 1, 2, 0, 2, 3] {
                    data.extend_from_slice(&corner(face[i]).to_array());
                    data.extend_from_slice(&c.rgba);
                }
            }
        }
        gpu.queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(&data));
        let mut own = None;
        let pass = target.begin(gpu, "test/boxes", &mut own);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.set_vertex_buffer(0, self.buffer.slice(..));
        pass.draw(0..(self.cuboids.len() * 36) as u32, 0..1);
    }
}

impl RenderPhase for Boxes {
    fn name(&self) -> &'static str {
        "test/boxes"
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        self.record_to(gpu, PassTarget::Own { enc, view }, cam);
    }

    fn shares_world_pass(&self) -> bool {
        true
    }

    fn record_into_pass(&mut self, gpu: &Gpu, pass: &mut wgpu::RenderPass<'static>, cam: &Camera) {
        self.record_to(gpu, PassTarget::Shared(pass), cam);
    }
}

// ---- scenes -----------------------------------------------------------------------------

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

/// Below the clouds: the eye at y=100 looks 30 degrees up at the layer (128.33 to 132.33).
fn below_camera() -> Camera {
    camera(DVec3::new(0.0, 100.0, 0.0), -30.0)
}

/// A red block above the layer, seen through it.
const BEHIND: Cuboid = Cuboid {
    min: DVec3::new(-4.0, 140.0, 60.0),
    max: DVec3::new(4.0, 148.0, 68.0),
    rgba: RED,
};
/// A green block under the layer, between the eye and the clouds.
const IN_FRONT: Cuboid = Cuboid {
    min: DVec3::new(6.0, 112.0, 30.0),
    max: DVec3::new(10.0, 120.0, 38.0),
    rgba: GREEN,
};

/// Above the clouds: the eye at y=150 looks 35 degrees down through the layer at the ground.
fn above_camera() -> Camera {
    camera(DVec3::new(0.0, 150.0, 0.0), 35.0)
}

/// Red ground at y=64, far below the layer.
const GROUND: Cuboid = Cuboid {
    min: DVec3::new(-400.0, 63.0, -400.0),
    max: DVec3::new(400.0, 64.0, 400.0),
    rgba: RED,
};
/// A blue pillar rising from the ground through the layer to y=140.
const PILLAR: Cuboid = Cuboid {
    min: DVec3::new(-3.0, 64.0, 20.0),
    max: DVec3::new(3.0, 140.0, 26.0),
    rgba: BLUE,
};

fn camera(eye: DVec3, pitch: f32) -> Camera {
    let mut cam = Camera {
        pos: eye,
        pitch,
        ..Camera::default()
    };
    cam.set_viewport(W, H);
    cam.set_view_distance_chunks(12);
    cam.update();
    cam
}

fn noon() -> Environment {
    Environment {
        world_time: 6000,
        dimension: Dimension::Overworld,
        weather: Weather::default(),
    }
}

fn clouds_phase(gpu: &Gpu, mode: CloudMode, alpha: u8, eye_y: Option<f64>) -> CloudsPhase {
    let mut clouds = CloudsPhase::new(gpu);
    clouds.load_texture(gpu, &clouds_texture(alpha));
    clouds.mode = mode;
    clouds.environment = noon();
    clouds.view_eye_y = eye_y;
    clouds
}

/// Render `clouds` (installed at both of vanilla's points, as the client does) and `cuboids`
/// over black; returns the frame and whether the clouds drew after the world.
fn render(
    gpu: &mut Gpu,
    clouds: CloudsPhase,
    cam: &Camera,
    cuboids: Vec<Cuboid>,
) -> (Vec<u8>, bool) {
    let clouds = Rc::new(RefCell::new(clouds));
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    CloudsPhase::install(&clouds, &mut phases);
    phases.insert(
        PhaseOrder::TERRAIN_SOLID,
        Box::new(Boxes::new(gpu, cuboids)),
    );
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), cam, 0.0);
    drop(frame);
    let above = clouds.borrow().draws_above();
    (gpu.read_pixels().expect("readback"), above)
}

/// The pixel `world` lands on.
fn project(cam: &Camera, world: DVec3) -> (u32, u32) {
    let clip = cam.view_proj() * cam.relative(world).extend(1.0);
    let ndc = clip.truncate() / clip.w;
    assert!(
        ndc.x.abs() < 1.0 && ndc.y.abs() < 1.0,
        "{world:?} is off screen"
    );
    (
        ((ndc.x * 0.5 + 0.5) * W as f32) as u32,
        ((0.5 - ndc.y * 0.5) * H as f32) as u32,
    )
}

fn rgb(pixels: &[u8], (x, y): (u32, u32)) -> [u8; 3] {
    let i = ((y * W + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2]]
}

fn count(pixels: &[u8], rgb: [u8; 3]) -> usize {
    pixels.chunks_exact(4).filter(|p| p[..3] == rgb).count()
}

/// A grey: the white clouds (shaded by face, 80 % opaque) over the black clear colour.
fn is_cloud_over_black(p: [u8; 3]) -> bool {
    p[0] == p[1] && p[1] == p[2] && p[0] > 100
}

/// With `PNG_DUMP_DIR` set, store the frame there for a look.
fn dump(name: &str, pixels: &[u8]) {
    if let Some(dir) = std::env::var_os("PNG_DUMP_DIR") {
        let path = std::path::Path::new(&dir).join(format!("{name}.png"));
        image::save_buffer(&path, pixels, W, H, image::ExtendedColorType::Rgba8).expect("png");
    }
}

// ---- eye below 128 ------------------------------------------------------------------------

/// java: `renderWorldPass:1364` — below 128 the clouds draw before terrain with depth writes on,
/// and keep their depth. A block behind a cloud is depth-rejected wherever the cloud covers it
/// (the old port drew it over the cloud), and a block between the eye and the cloud draws over
/// the cloud.
#[test]
fn below_the_clouds_a_block_behind_a_cloud_is_hidden_and_a_block_in_front_shows() {
    for samples in [1, 4] {
        let Some(mut gpu) = headless(samples) else {
            return;
        };
        let cam = below_camera();
        let behind_px = project(&cam, DVec3::new(0.0, 144.0, 60.0));
        let front_px = project(&cam, DVec3::new(8.0, 116.0, 30.0));
        for mode in [CloudMode::Fast, CloudMode::Fancy] {
            let label = format!("{mode:?} x{}", gpu.sample_count());
            let clouds = clouds_phase(&gpu, mode, 255, Some(cam.pos.y));
            let (frame, above) = render(&mut gpu, clouds, &cam, vec![BEHIND, IN_FRONT]);
            dump(&format!("below-{mode:?}-x{}", gpu.sample_count()), &frame);
            assert!(
                !above,
                "{label}: an eye at y=100 draws the clouds before terrain"
            );
            assert_eq!(
                count(&frame, [255, 0, 0]),
                0,
                "{label}: the block behind the cloud layer showed through it"
            );
            assert!(
                is_cloud_over_black(rgb(&frame, behind_px)),
                "{label}: cloud expected over the hidden block, got {:?}",
                rgb(&frame, behind_px)
            );
            assert_eq!(
                rgb(&frame, front_px),
                [0, 255, 0],
                "{label}: the block in front of the cloud must draw over it"
            );
            assert!(
                count(&frame, [0, 255, 0]) > 100,
                "{label}: green block too small"
            );

            // Controls. Clear texels write no depth (the alpha test discards them before the
            // depth write), so the block behind them shows; and with the clouds off it is there.
            let clear = clouds_phase(&gpu, mode, 1, Some(cam.pos.y));
            let (frame, _) = render(&mut gpu, clear, &cam, vec![BEHIND, IN_FRONT]);
            assert_eq!(rgb(&frame, behind_px), [255, 0, 0], "{label}: clear texels");
            assert_eq!(rgb(&frame, front_px), [0, 255, 0], "{label}: clear texels");
            let mut off = clouds_phase(&gpu, mode, 255, Some(cam.pos.y));
            off.mode = CloudMode::Off;
            let (frame, _) = render(&mut gpu, off, &cam, vec![BEHIND, IN_FRONT]);
            assert_eq!(rgb(&frame, behind_px), [255, 0, 0], "{label}: clouds off");
        }
    }
}

// ---- eye at or above 128 ------------------------------------------------------------------

/// java: `renderWorldPass:1474` — at or above 128 the clouds draw after the world, depth-tested
/// against it: they blend over the ground beneath them (the old port drew the ground over them,
/// so from above they vanished), and a block rising through the layer hides them where it is in
/// front of them.
#[test]
fn above_the_clouds_they_cover_the_ground_and_a_block_rising_through_them_shows() {
    for samples in [1, 4] {
        let Some(mut gpu) = headless(samples) else {
            return;
        };
        let cam = above_camera();
        // Off to the side, so the pillar is not in the way.
        let ground_px = project(&cam, DVec3::new(40.0, 64.0, 100.0));
        let top_px = project(&cam, DVec3::new(0.0, 140.0, 23.0));
        let under_px = project(&cam, DVec3::new(0.0, 100.0, 20.0));
        for mode in [CloudMode::Fast, CloudMode::Fancy] {
            let label = format!("{mode:?} x{}", gpu.sample_count());
            let clouds = clouds_phase(&gpu, mode, 255, Some(cam.pos.y));
            let (frame, above) = render(&mut gpu, clouds, &cam, vec![GROUND, PILLAR]);
            dump(&format!("above-{mode:?}-x{}", gpu.sample_count()), &frame);
            assert!(
                above,
                "{label}: an eye at y=150 draws the clouds after the world"
            );
            // White cloud tops at 80 % over the red ground: red stays full, green and blue are
            // the cloud's 0.8 share (204), not 0 (no cloud) and not 255.
            let ground = rgb(&frame, ground_px);
            assert!(
                ground[0] >= 250 && (190..=215).contains(&ground[1]) && ground[1] == ground[2],
                "{label}: clouds blended over the ground expected, got {ground:?}"
            );
            assert_eq!(
                count(&frame, [255, 0, 0]),
                0,
                "{label}: bare ground showed where the cloud layer covers it"
            );
            assert_eq!(
                rgb(&frame, top_px),
                [0, 0, 255],
                "{label}: the pillar's top, above the layer, must hide the clouds"
            );
            let under = rgb(&frame, under_px);
            assert!(
                under[2] >= 250 && (190..=215).contains(&under[0]) && under[0] == under[1],
                "{label}: the pillar under the layer must be behind the clouds, got {under:?}"
            );

            // The same eye fed as below 128: the clouds draw before terrain and keep their
            // depth, so the ground behind them is rejected and they blend over the black clear
            // colour instead. This tells the two slots apart.
            let early = clouds_phase(&gpu, mode, 255, Some(ABOVE_CLOUDS_EYE_Y - 0.001));
            let (frame, above) = render(&mut gpu, early, &cam, vec![GROUND, PILLAR]);
            assert!(!above, "{label}: an eye fed just under 128");
            assert!(
                is_cloud_over_black(rgb(&frame, ground_px)),
                "{label}: before-terrain clouds over the clear colour, got {:?}",
                rgb(&frame, ground_px)
            );
            assert_eq!(rgb(&frame, top_px), [0, 0, 255], "{label}: early slot");
        }
    }
}

// ---- the slot choice ----------------------------------------------------------------------

/// java: the test is `entity.posY + entity.getEyeHeight() < 128.0D`, a literal: the cloud height
/// does not move it, the camera does not decide it (F5's orbit can sit on the other side), and an
/// undriven phase draws before terrain.
#[test]
fn the_eye_height_fed_by_the_client_picks_the_slot_at_a_literal_128() {
    let Some(gpu) = headless(1) else {
        return;
    };
    let mut clouds = clouds_phase(&gpu, CloudMode::Fast, 255, None);
    assert!(!clouds.draws_above(), "undriven: before terrain");
    for (eye, above) in [
        (127.999_99, false),
        (128.0, true),
        (64.0, false),
        (300.0, true),
    ] {
        clouds.view_eye_y = Some(eye);
        assert_eq!(clouds.draws_above(), above, "eye at {eye}");
    }
    clouds.cloud_height = 200.0;
    clouds.view_eye_y = Some(150.0);
    assert!(
        clouds.draws_above(),
        "the cloud height does not move the split"
    );
    assert!(ABOVE_CLOUDS_ORDER > FogPhase::ORDER);
    assert!(ABOVE_CLOUDS_ORDER > PhaseOrder::TERRAIN_TRANSLUCENT);
    assert!(ABOVE_CLOUDS_ORDER < PhaseOrder::TEMPORAL_AA);
}

/// Exactly one of the two registrations draws a frame, named as the frame graph documents.
#[test]
fn install_registers_both_points_and_one_draws_per_frame() {
    let Some(gpu) = headless(1) else {
        return;
    };
    let clouds = Rc::new(RefCell::new(clouds_phase(
        &gpu,
        CloudMode::Fancy,
        255,
        None,
    )));
    let mut phases = PhaseList::new();
    CloudsPhase::install(&clouds, &mut phases);
    assert_eq!(
        phases.names().collect::<Vec<_>>(),
        ["sky/clouds", "sky/clouds-above"]
    );
}

// ---- frame cost ---------------------------------------------------------------------------

/// Forwards to a shared `CloudsPhase` and counts how it recorded: a pass of its own, or into
/// the shared world pass.
struct Counting {
    clouds: Rc<RefCell<CloudsPhase>>,
    own: Rc<Cell<u32>>,
    shared: Rc<Cell<u32>>,
}

impl RenderPhase for Counting {
    fn name(&self) -> &'static str {
        "sky/clouds"
    }

    fn enabled(&self) -> bool {
        self.clouds.borrow().enabled()
    }

    fn frame_clear_handoff(&self) -> mc_render::pipeline::FrameClearHandoff {
        self.clouds.borrow().frame_clear_handoff()
    }

    fn shares_world_pass(&self) -> bool {
        self.clouds.borrow().shares_world_pass()
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        self.own.set(self.own.get() + 1);
        self.clouds.borrow_mut().record(gpu, enc, view, cam);
    }

    fn record_into_pass(&mut self, gpu: &Gpu, pass: &mut wgpu::RenderPass<'static>, cam: &Camera) {
        self.shared.set(self.shared.get() + 1);
        self.clouds.borrow_mut().record_into_pass(gpu, pass, cam);
    }
}

/// The clouds now open no pass of their own: with nothing to hand back (no depth restore, no
/// private depth target) their draws join the shared world pass, which the opaque world
/// recorded after them continues. The frame is the same bytes with the world pass shared or
/// not, with and without MSAA.
#[test]
fn clouds_join_the_world_pass_and_draw_the_same_frame_either_way() {
    for samples in [1, 4] {
        let Some(mut gpu) = headless(samples) else {
            return;
        };
        let cam = below_camera();
        for mode in [CloudMode::Fast, CloudMode::Fancy] {
            let mut frames = Vec::new();
            for share in [true, false] {
                let clouds = Rc::new(RefCell::new(clouds_phase(&gpu, mode, 255, Some(cam.pos.y))));
                let (own, shared) = (Rc::new(Cell::new(0)), Rc::new(Cell::new(0)));
                let mut phases = PhaseList::new();
                phases.clear_color = wgpu::Color::BLACK;
                phases.set_shared_world_pass(share);
                phases.insert(
                    PhaseOrder::SKY,
                    Box::new(Counting {
                        clouds: Rc::clone(&clouds),
                        own: Rc::clone(&own),
                        shared: Rc::clone(&shared),
                    }),
                );
                phases.insert(
                    PhaseOrder::TERRAIN_SOLID,
                    Box::new(Boxes::new(&gpu, vec![BEHIND, IN_FRONT])),
                );
                let frame = gpu.acquire().expect("acquire");
                phases.render(&gpu, frame.view(), &cam, 0.0);
                drop(frame);
                frames.push(gpu.read_pixels().expect("readback"));
                let label = format!("{mode:?} x{} shared={share}", gpu.sample_count());
                if share {
                    assert_eq!((own.get(), shared.get()), (0, 1), "{label}");
                } else {
                    assert_eq!((own.get(), shared.get()), (1, 0), "{label}");
                }
            }
            assert!(
                frames[0] == frames[1],
                "{mode:?} x{}: sharing the world pass changed the frame",
                gpu.sample_count()
            );
        }
    }
}

// ---- the screen-space fog ---------------------------------------------------------------

/// The world's screen-space fog pass (registered before translucent terrain, as the client does)
/// now sees cloud depth below the clouds. The clouds fog themselves (vanilla's `setupFog(0)`
/// around `renderClouds`), and the fog pass reconstructs distance with the terrain projection,
/// which puts a cloud nearer than it is (`1/d = 1/z + 1/far_terrain - 1/far_clouds`), so it adds
/// fog only to clouds past about 1.14x the render distance, which their own fog has already
/// made fully fog-coloured: what changes is the 20 % of the sky showing through them, pulled
/// toward the fog colour. Pinned: clouds nearer than the fog start are untouched, and the change
/// elsewhere stays small, with the client's clear colour (the fog colour) and sky.
#[test]
fn the_world_fog_pass_leaves_near_clouds_alone_and_far_ones_nearly_so() {
    let Some(mut gpu) = headless(1) else {
        return;
    };
    let env = noon();
    let inputs = FogInputs {
        render_distance_chunks: 8,
        ..FogInputs::default()
    };
    let state = FogState::compute(&env, inputs, 0.0);
    let mut cam = Camera {
        pos: DVec3::new(0.0, 100.0, 0.0),
        pitch: -12.0,
        ..Camera::default()
    };
    cam.set_viewport(W, H);
    cam.set_view_distance_chunks(8);
    cam.update();
    let mut frames = Vec::new();
    for with_fog in [false, true] {
        let mut sky = SkyPhase::new(&gpu);
        sky.environment = env;
        sky.fog_inputs = inputs;
        let mut clouds = clouds_phase(&gpu, CloudMode::Fancy, 255, Some(cam.pos.y));
        clouds.set_fog(Some((state, FogDistance::Radial)));
        let clouds = Rc::new(RefCell::new(clouds));
        let mut phases = PhaseList::new();
        phases.clear_color = wgpu::Color {
            r: f64::from(state.color.x),
            g: f64::from(state.color.y),
            b: f64::from(state.color.z),
            a: 1.0,
        };
        phases.insert(PhaseOrder::SKY, Box::new(sky));
        CloudsPhase::install(&clouds, &mut phases);
        if with_fog {
            let mut fog = FogPhase::new(&gpu);
            fog.environment = env;
            fog.inputs = inputs;
            fog.set_distance(FogDistance::Radial);
            phases.insert(FogPhase::BEFORE_TRANSLUCENT_ORDER, Box::new(fog));
        }
        let frame = gpu.acquire().expect("acquire");
        phases.render(&gpu, frame.view(), &cam, 0.0);
        drop(frame);
        frames.push(gpu.read_pixels().expect("readback"));
    }
    dump("fog-off", &frames[0]);
    dump("fog-on", &frames[1]);
    // Rows whose cloud is nearer than the fog start (96 blocks at 8 chunks): the slab's
    // underside, 28.33 blocks up, is within 96 blocks (radially) above about 17 degrees.
    let mut changed = 0;
    let mut worst = 0_u8;
    let mut worst_row = 0;
    for y in 0..H {
        for x in 0..W {
            let i = ((y * W + x) * 4) as usize;
            let d = (0..3)
                .map(|c| frames[0][i + c].abs_diff(frames[1][i + c]))
                .max()
                .unwrap_or(0);
            if d > 0 {
                changed += 1;
                if d > worst {
                    worst = d;
                    worst_row = y;
                }
            }
        }
    }
    let top_rows = H / 4;
    for y in 0..top_rows {
        for x in 0..W {
            let i = ((y * W + x) * 4) as usize;
            assert_eq!(
                frames[0][i..i + 3],
                frames[1][i..i + 3],
                "near cloud at {x},{y} changed by the fog pass"
            );
        }
    }
    eprintln!(
        "fog pass over clouds: {changed} of {} pixels changed, worst {worst}/255 (row {worst_row})",
        W * H
    );
    // Measured at 2/255 on the development machine.
    assert!(
        worst <= 4,
        "the fog pass moved a cloud pixel by {worst}/255 (row {worst_row})"
    );
}
