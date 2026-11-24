//! Fancy-cloud face culling must draw the same frame, byte for byte, as drawing every face, and must leave
//! the same cloud depth in the shared depth buffer. The clouds keep their depth there, as vanilla
//! does, so a depth-tested probe recorded after them shows it.
//!
//! Real `clouds.png` from the vanilla jar, real GPU readback; skips (printing `SKIP:`) without
//! either. Point `MC_JARS_DIR` at a `jars/` directory to override the asset location.
//!
//! ```text
//! MC_JARS_DIR=<jars> cargo test -p mc-render --test nw_f2_cloud_cull -- --nocapture
//! ```

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use glam::DVec3;
use mc_assets::{ResourceManager, ZipResourcePack};
use mc_render::sky::clouds::{CloudMode, CloudsPhase};
use mc_render::sky::{Dimension, Environment, FogDistance, FogInputs, FogState, Weather};
use mc_render::{
    Camera, DepthAttachment, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase,
    DEPTH_FORMAT,
};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 180;

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn resources() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    let pack = ZipResourcePack::open(jar).ok()?;
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(pack));
    Some(rm)
}

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
        Ok(g) => Some(g),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

/// Lets the test keep a handle on the phase it hands to the list.
struct Shared(Rc<RefCell<CloudsPhase>>);

impl RenderPhase for Shared {
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
        "sky/clouds"
    }

    fn enabled(&self) -> bool {
        self.0.borrow().enabled()
    }
}

/// Stands in for terrain recorded after the sky: an opaque red, depth-tested, depth-writing
/// quad over the left half of the frame, 400 blocks out. The cloud depth left in the shared
/// buffer rejects it wherever a cloud is nearer, so the left half shows that depth.
struct DepthProbe {
    pipeline: wgpu::RenderPipeline,
}

impl DepthProbe {
    fn new(gpu: &Gpu) -> Self {
        // Clip-space z of a surface 400 blocks out, as `s20_sky.rs`'s probe uses.
        let z = (1.0_f32 - 0.05 / 400.0).clamp(0.0, 0.999_9);
        let src = format!(
            "@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {{
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, {z}, 1.0);
}}
@fragment
fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {{
    if (p.x > {half}.0) {{ discard; }}
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}}",
            half = WIDTH / 2
        );
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("nw-f2 depth probe"),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("nw-f2 depth probe"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("nw-f2 depth probe"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
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
        Self { pipeline }
    }
}

impl RenderPhase for DepthProbe {
    fn name(&self) -> &'static str {
        "test/depth-probe"
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        let mut pass = gpu.begin_pass(enc, view, "nw-f2 depth probe", DepthAttachment::Shared);
        pass.set_pipeline(&self.pipeline);
        pass.draw(0..3, 0..1);
    }
}

/// One pose: eye, yaw, pitch (positive looks down), cloud ticks, world time, fogged.
type Pose = (DVec3, f32, f32, f32, i64, bool);

const POSES: [Pose; 8] = [
    // Straight up from an island.
    (DVec3::new(0.0, 64.0, 0.0), 0.0, -90.0, 0.0, 6000, false),
    // Toward the horizon from a SkyWars height, away from the origin.
    (
        DVec3::new(37.5, 96.0, -12.25),
        30.0,
        -12.0,
        1234.0,
        6000,
        true,
    ),
    (
        DVec3::new(-905.3, 110.0, 2210.7),
        145.0,
        -3.0,
        40_000.0,
        1000,
        true,
    ),
    // Just under the slab, inside it, and above it: every gate combination.
    (
        DVec3::new(0.0, 126.0, 0.0),
        300.0,
        -60.0,
        500.5,
        6000,
        false,
    ),
    (DVec3::new(12.0, 129.5, 7.0), 80.0, 10.0, 77.0, 6000, true),
    (
        DVec3::new(500.0, 150.0, -500.0),
        200.0,
        35.0,
        9_999.0,
        13_000,
        false,
    ),
    // Far out, where the cloud coordinate wraps.
    (
        DVec3::new(29_000.25, 90.0, -24_576.5),
        10.0,
        -25.0,
        123_456.0,
        18_000,
        true,
    ),
    (
        DVec3::new(-3.0, 70.0, 5.0),
        225.0,
        -40.0,
        0.25,
        23_000,
        false,
    ),
];

struct Frame {
    pixels: Vec<u8>,
    cull: Option<(u32, u32)>,
}

/// Render one fancy-cloud frame followed by the depth probe.
fn render(gpu: &mut Gpu, rm: &ResourceManager, pose: Pose, cull: bool) -> Frame {
    let (eye, yaw, pitch, ticks, world_time, fogged) = pose;
    let mut clouds = CloudsPhase::new(gpu);
    clouds.load_texture(gpu, rm);
    clouds.mode = CloudMode::Fancy;
    clouds.cloud_tick_counter = ticks;
    clouds.partial_ticks = 0.37;
    clouds.environment = Environment {
        world_time,
        dimension: Dimension::Overworld,
        weather: Weather::default(),
    };
    if fogged {
        let inputs = FogInputs {
            render_distance_chunks: 8,
            ..FogInputs::default()
        };
        let state = FogState::compute(&clouds.environment, inputs, 0.37);
        clouds.set_fog(Some((state, FogDistance::Radial)));
    }
    assert_eq!(clouds.set_fancy_face_culling(cull), cull, "culling switch");
    let clouds = Rc::new(RefCell::new(clouds));

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.1,
        g: 0.2,
        b: 0.4,
        a: 1.0,
    };
    phases.insert(PhaseOrder::SKY, Box::new(Shared(Rc::clone(&clouds))));
    phases.insert(PhaseOrder::TERRAIN_SOLID, Box::new(DepthProbe::new(gpu)));
    let mut cam = Camera {
        pos: eye,
        yaw,
        pitch,
        ..Camera::default()
    };
    cam.update();
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), &cam, 0.37);
    let pixels = gpu.read_pixels().expect("readback");
    let phase = clouds.borrow();
    Frame {
        pixels,
        cull: phase.fancy_face_cull_stats(),
    }
}

fn differing(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(a, b)| a != b)
        .count()
}

#[test]
fn culled_clouds_match_every_face_byte_for_byte_depth_included() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: jar unavailable");
        return;
    };
    let Some(mut gpu) = headless(WIDTH, HEIGHT) else {
        return;
    };
    let (mut drawn, mut total, mut cloud_pixels, mut held_back) = (0_u64, 0_u64, 0_usize, 0_usize);
    for (i, pose) in POSES.into_iter().enumerate() {
        // Every face.
        let all = render(&mut gpu, &rm, pose, false);
        assert!(all.cull.is_none(), "pose {i}: every face");
        // The clear colour (0.1, 0.2, 0.4) is about (26, 51, 102) in the unorm target.
        let is_clear = |p: &[u8]| {
            (i32::from(p[0]) - 26).abs() <= 1
                && (i32::from(p[1]) - 51).abs() <= 1
                && (i32::from(p[2]) - 102).abs() <= 1
        };
        // The probe covers the left half except where a nearer cloud's depth kept it out, so
        // nothing there is left at the clear colour.
        for y in 0..HEIGHT {
            for x in 0..WIDTH / 2 {
                let at = ((y * WIDTH + x) * 4) as usize;
                let p = &all.pixels[at..at + 3];
                assert!(!is_clear(p), "pose {i}: nothing drew at {x},{y}");
                held_back += usize::from(p != [255, 0, 0]);
            }
        }
        cloud_pixels += all
            .pixels
            .chunks_exact(4)
            .enumerate()
            .filter(|(n, p)| (*n as u32 % WIDTH) >= WIDTH / 2 && !is_clear(p))
            .count();
        let culled = render(&mut gpu, &rm, pose, true);
        let diff = differing(&all.pixels, &culled.pixels);
        assert_eq!(diff, 0, "pose {i}: culling changed {diff} px");
        let (n, lattice) = culled.cull.expect("culling ran");
        assert!(n <= lattice);
        drawn += u64::from(n);
        total += u64::from(lattice);
        eprintln!("pose {i}: {n} of {lattice} lattice indices drawn, identical");
    }
    // Not vacuous: clouds were on screen, their depth held the probe back, and culling removed
    // faces.
    assert!(
        cloud_pixels > 5_000,
        "only {cloud_pixels} cloud pixels across all poses"
    );
    assert!(
        held_back > 1_000,
        "cloud depth held the probe back at only {held_back} px"
    );
    assert!(
        drawn < total,
        "culling removed nothing ({drawn} of {total})"
    );
    eprintln!(
        "fancy clouds drew {drawn} of {total} lattice indices ({:.1}% culled)",
        100.0 * (total - drawn) as f64 / total as f64
    );
}
