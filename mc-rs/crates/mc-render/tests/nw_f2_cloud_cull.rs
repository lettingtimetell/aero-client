//! Fancy-cloud face culling and the
//! clouds' own depth target (both in `sky/clouds.rs`) must draw the same
//! frame, byte for byte, as the old path (every face, shared depth buffer plus a restore pass),
//! and must hand the shared depth buffer back untouched.
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
/// quad over the left half of the frame, far behind every cloud. A cloud depth left in the
/// shared buffer would reject it there.
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
    own_depth: bool,
}

/// Render one fancy-cloud frame followed by the depth probe.
fn render(gpu: &mut Gpu, rm: &ResourceManager, pose: Pose, cull: bool, own_depth: bool) -> Frame {
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
    clouds.set_fancy_own_depth(own_depth);
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
        own_depth: phase.fancy_own_depth_in_use(),
    }
}

fn differing(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(a, b)| a != b)
        .count()
}

#[test]
fn culled_clouds_on_their_own_depth_match_the_old_path_byte_for_byte() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: jar unavailable");
        return;
    };
    let Some(mut gpu) = headless(WIDTH, HEIGHT) else {
        return;
    };
    let (mut drawn, mut total, mut cloud_pixels) = (0_u64, 0_u64, 0_usize);
    for (i, pose) in POSES.into_iter().enumerate() {
        // The old path: every face, shared depth buffer, restore pass.
        let old = render(&mut gpu, &rm, pose, false, false);
        assert!(old.cull.is_none() && !old.own_depth, "pose {i}: old path");
        // The probe must have drawn over the whole left half: the shared depth was pristine.
        for y in 0..HEIGHT {
            for x in 0..WIDTH / 2 {
                let at = ((y * WIDTH + x) * 4) as usize;
                assert_eq!(
                    &old.pixels[at..at + 3],
                    &[255, 0, 0],
                    "pose {i} probe at {x},{y}"
                );
            }
        }
        // The clear colour (0.1, 0.2, 0.4) is about (26, 51, 102) in the unorm target.
        let is_clear = |p: &[u8]| {
            (i32::from(p[0]) - 26).abs() <= 1
                && (i32::from(p[1]) - 51).abs() <= 1
                && (i32::from(p[2]) - 102).abs() <= 1
        };
        cloud_pixels += old
            .pixels
            .chunks_exact(4)
            .enumerate()
            .filter(|(n, p)| (*n as u32 % WIDTH) >= WIDTH / 2 && !is_clear(p))
            .count();
        for (cull, own_depth) in [(true, true), (true, false), (false, true)] {
            let new = render(&mut gpu, &rm, pose, cull, own_depth);
            assert_eq!(
                new.own_depth, own_depth,
                "pose {i}: own depth target in use"
            );
            let diff = differing(&old.pixels, &new.pixels);
            assert_eq!(
                diff, 0,
                "pose {i} cull={cull} own_depth={own_depth}: {diff} px differ"
            );
            if cull {
                let (n, lattice) = new.cull.expect("culling ran");
                assert!(n <= lattice);
                if own_depth {
                    drawn += u64::from(n);
                    total += u64::from(lattice);
                    eprintln!("pose {i}: {n} of {lattice} lattice indices drawn, identical");
                }
            }
        }
    }
    // Not vacuous: clouds were on screen, and culling removed faces.
    assert!(
        cloud_pixels > 5_000,
        "only {cloud_pixels} cloud pixels across all poses"
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
