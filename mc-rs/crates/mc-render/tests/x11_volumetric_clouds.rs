//! The volumetric clouds through the real phase list on a real
//! GPU, read back.
//!
//! * With the clouds off (installed, never requested, or released) the frame is byte-identical to
//!   a frame without them, and nothing is allocated.
//! * On, they appear in the sky; terrain in front of them keeps its exact pixels; they work under
//!   MSAA; above the layer they draw from the late slot, over the ground.
//!
//! Everything skips (printing `SKIP:`) without a GPU adapter.
//!
//! The ignored `render_review_pngs` writes 1920x1080 PNGs for a human to look at:
//!
//! ```text
//! cargo test -p mc-render --test x11_volumetric_clouds -- --ignored --nocapture render_review
//! MC_GPU_TIMING=1 cargo test -p mc-render --test x11_volumetric_clouds -- --ignored gpu_cost
//! ```

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Duration;

use glam::{DVec3, Vec4};
use mc_render::sky::dome::SkyPhase;
use mc_render::sky::volumetric::{
    self, in_game_look, CloudLayer, CloudQuality, VolumetricCloudsPhase,
};
use mc_render::sky::{Dimension, Environment, Weather};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

const READY: Duration = Duration::from_secs(60);

fn headless(width: u32, height: u32, samples: u32) -> Option<Gpu> {
    match Gpu::headless(
        width,
        height,
        GpuOptions {
            vsync: false,
            sample_count: samples,
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

fn environment(world_time: i64, rain: f32) -> Environment {
    Environment {
        world_time,
        dimension: Dimension::Overworld,
        weather: Weather {
            rain_strength: rain,
            thunder_strength: rain * 0.5,
            last_lightning_bolt: 0,
        },
    }
}

fn camera(eye: DVec3, yaw: f32, pitch: f32, size: (u32, u32)) -> Camera {
    let mut cam = Camera {
        pos: eye,
        yaw,
        pitch,
        ..Camera::default()
    };
    cam.set_viewport(size.0, size.1);
    cam.set_view_distance_chunks(16);
    cam.update();
    cam
}

/// The NDC depth of a point `distance` blocks straight ahead of `cam`.
fn ndc_depth(cam: &Camera, distance: f32) -> f32 {
    let clip = cam.proj * Vec4::new(0.0, 0.0, -distance, 1.0);
    clip.z / clip.w
}

/// Stand-in terrain: an opaque, depth-writing rectangle (NDC `rect` = x0, y0, x1, y1) at a fixed
/// depth.
mod probe {
    use mc_render::{Camera, DepthAttachment, Gpu, RenderPhase, DEPTH_FORMAT};

    pub struct Probe {
        pipeline: wgpu::RenderPipeline,
    }

    impl Probe {
        pub fn new(gpu: &Gpu, rect: [f32; 4], depth: f32, color: [f32; 3]) -> Self {
            let source = format!(
                "@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {{
    var xs = array<f32, 6>({x0}, {x1}, {x1}, {x0}, {x1}, {x0});
    var ys = array<f32, 6>({y0}, {y0}, {y1}, {y0}, {y1}, {y1});
    return vec4<f32>(xs[i], ys[i], {depth}, 1.0);
}}
@fragment
fn fs() -> @location(0) vec4<f32> {{
    return vec4<f32>({r}, {g}, {b}, 1.0);
}}",
                x0 = rect[0],
                y0 = rect[1],
                x1 = rect[2],
                y1 = rect[3],
                depth = depth,
                r = color[0],
                g = color[1],
                b = color[2],
            );
            let module = gpu
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("x11 probe"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            let layout = gpu
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("x11 probe"),
                    bind_group_layouts: &[],
                    immediate_size: 0,
                });
            let pipeline = gpu
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("x11 probe"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
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
                        module: &module,
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
            Probe { pipeline }
        }
    }

    impl RenderPhase for Probe {
        fn name(&self) -> &'static str {
            "x11 probe"
        }

        fn record(
            &mut self,
            gpu: &Gpu,
            enc: &mut wgpu::CommandEncoder,
            view: &wgpu::TextureView,
            _cam: &Camera,
        ) {
            let mut pass = gpu.begin_pass(enc, view, "x11 probe", DepthAttachment::Shared);
            pass.set_pipeline(&self.pipeline);
            pass.draw(0..6, 0..1);
        }
    }
}

/// A vanilla sky (dome without textures) plus an optional terrain probe.
fn world(gpu: &Gpu, env: Environment, probe: Option<probe::Probe>) -> PhaseList {
    let mut phases = PhaseList::new();
    let fog = env.base_fog_color(0.0);
    phases.clear_color = wgpu::Color {
        r: f64::from(fog.x),
        g: f64::from(fog.y),
        b: f64::from(fog.z),
        a: 1.0,
    };
    let mut sky = SkyPhase::new(gpu);
    sky.environment = env;
    phases.insert(PhaseOrder::SKY, Box::new(sky));
    if let Some(probe) = probe {
        phases.insert(PhaseOrder::TERRAIN_SOLID, Box::new(probe));
    }
    phases
}

fn render(gpu: &mut Gpu, phases: &mut PhaseList, cam: &Camera) -> Vec<u8> {
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), cam, 0.0);
    gpu.read_pixels().expect("readback")
}

/// Turn the clouds on for a frame drawn from `cam` (`coverage` 0..1, at world time `time`).
fn drive(
    clouds: &Rc<RefCell<VolumetricCloudsPhase>>,
    env: &Environment,
    cam: &Camera,
    coverage: f32,
    quality: CloudQuality,
) -> bool {
    let mut phase = clouds.borrow_mut();
    phase.look = in_game_look(env, 0.8, 0.0, coverage);
    phase.quality = quality;
    phase.set_frame(true, cam.pos.y)
}

fn pixel(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * width + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

/// How many pixels in rows `rows` differ from `base` by more than `threshold` in some channel.
fn changed(a: &[u8], b: &[u8], width: u32, rows: std::ops::Range<u32>, threshold: u8) -> usize {
    let mut n = 0;
    for y in rows {
        for x in 0..width {
            let (p, q) = (pixel(a, width, x, y), pixel(b, width, x, y));
            if p.iter()
                .zip(q.iter())
                .any(|(u, v)| u.abs_diff(*v) > threshold)
            {
                n += 1;
            }
        }
    }
    n
}

#[test]
fn off_the_frame_is_byte_identical_and_nothing_is_allocated() {
    let size = (320, 180);
    let Some(mut gpu) = headless(size.0, size.1, 1) else {
        return;
    };
    let env = environment(6000, 0.0);
    let cam = camera(DVec3::new(10.5, 70.0, -3.25), 30.0, -35.0, size);
    let rect = [-1.0, -1.0, -0.2, 0.3];
    let depth = ndc_depth(&cam, 25.0);
    let terrain = |gpu: &Gpu| probe::Probe::new(gpu, rect, depth, [0.42, 0.31, 0.2]);

    let mut plain = world(&gpu, env, Some(terrain(&gpu)));
    let without = render(&mut gpu, &mut plain, &cam);

    let mut with = world(&gpu, env, Some(terrain(&gpu)));
    let clouds = VolumetricCloudsPhase::install(&mut with);
    let names: Vec<_> = with.names().collect();
    assert!(names.contains(&volumetric::EARLY_PHASE_NAME));
    assert!(names.contains(&volumetric::LATE_PHASE_NAME));
    // Installed, never requested: asking to draw does nothing.
    for _ in 0..3 {
        assert!(!drive(&clouds, &env, &cam, 0.6, CloudQuality::Medium));
        let frame = render(&mut gpu, &mut with, &cam);
        assert!(frame == without, "an idle clouds phase changed the frame");
    }
    assert!(!clouds.borrow().allocated(), "nothing allocated while off");

    // On, then off again (the mod toggled): released, and the frame is exactly vanilla again.
    clouds.borrow_mut().request(&gpu);
    assert!(clouds.borrow_mut().wait_ready(READY), "pipelines built");
    assert!(drive(&clouds, &env, &cam, 0.6, CloudQuality::Medium));
    let on = render(&mut gpu, &mut with, &cam);
    assert!(on != without, "the clouds drew something");
    clouds.borrow_mut().release();
    assert!(!clouds.borrow().allocated(), "released");
    assert!(!clouds.borrow().drawing());
    let off = render(&mut gpu, &mut with, &cam);
    assert!(off == without, "released clouds left the frame changed");
}

#[test]
fn on_they_fill_the_sky_and_terrain_in_front_keeps_its_pixels() {
    let size = (320, 180);
    let Some(mut gpu) = headless(size.0, size.1, 1) else {
        return;
    };
    let env = environment(5000, 0.0);
    // Looking up from y = 72: the cloud bases (y = 128) are 56 blocks above the eye.
    let cam = camera(DVec3::new(-120.0, 72.0, 455.5), 200.0, -32.0, size);
    // Terrain 30 blocks ahead, in front of every cloud, covering the left of the frame.
    let rect = [-1.0, -1.0, -0.3, 1.0];
    let depth = ndc_depth(&cam, 30.0);
    let terrain = |gpu: &Gpu| probe::Probe::new(gpu, rect, depth, [0.42, 0.31, 0.2]);

    let mut plain = world(&gpu, env, Some(terrain(&gpu)));
    let without = render(&mut gpu, &mut plain, &cam);

    let mut with = world(&gpu, env, Some(terrain(&gpu)));
    let clouds = VolumetricCloudsPhase::install(&mut with);
    clouds.borrow_mut().request(&gpu);
    assert!(clouds.borrow_mut().wait_ready(READY));
    let mut frame = Vec::new();
    for _ in 0..12 {
        assert!(drive(&clouds, &env, &cam, 0.65, CloudQuality::Medium));
        assert_eq!(
            clouds.borrow().drawing_order(),
            Some(volumetric::EARLY_ORDER),
            "below the layer the clouds draw before translucent terrain"
        );
        frame = render(&mut gpu, &mut with, &cam);
    }

    // The terrain's pixels (left 35 % of the frame; the probe ends at 35 %) are untouched.
    let terrain_cols = (size.0 as f32 * 0.34) as u32;
    for y in 0..size.1 {
        for x in 0..terrain_cols {
            assert_eq!(
                pixel(&frame, size.0, x, y),
                pixel(&without, size.0, x, y),
                "terrain pixel ({x}, {y}) changed: the clouds drew over terrain in front of them"
            );
        }
    }
    // The sky (right of the probe) has clouds: many pixels changed, and some are much brighter
    // and whiter than the blue sky behind them.
    let sky_rows = 0..size.1;
    let mut sky_changed = 0;
    let mut white = 0;
    for y in sky_rows {
        for x in (size.0 * 2 / 3)..size.0 {
            let (p, q) = (pixel(&frame, size.0, x, y), pixel(&without, size.0, x, y));
            if p.iter()
                .zip(q.iter())
                .take(3)
                .any(|(u, v)| u.abs_diff(*v) > 6)
            {
                sky_changed += 1;
            }
            let lift = i32::from(p[0]) - i32::from(q[0]);
            if lift > 25 && p[0] > 160 {
                white += 1;
            }
        }
    }
    let sky_pixels = (size.0 - size.0 * 2 / 3) * size.1;
    eprintln!("sky pixels changed: {sky_changed} of {sky_pixels}, bright cloud pixels: {white}");
    assert!(
        sky_changed as u32 > sky_pixels / 5,
        "too few sky pixels changed ({sky_changed} of {sky_pixels})"
    );
    assert!(white > 50, "no bright cloud pixels ({white})");
}

#[test]
fn under_msaa_they_render_without_validation_errors() {
    let size = (256, 144);
    let Some(mut gpu) = headless(size.0, size.1, 4) else {
        return;
    };
    if gpu.sample_count() < 2 {
        eprintln!("SKIP: no MSAA on this adapter");
        return;
    }
    let env = environment(6000, 0.0);
    let cam = camera(DVec3::new(0.0, 70.0, 0.0), 0.0, -30.0, size);
    let rect = [-1.0, -1.0, 1.0, -0.4];
    let depth = ndc_depth(&cam, 20.0);
    let mut plain = world(
        &gpu,
        env,
        Some(probe::Probe::new(&gpu, rect, depth, [0.3; 3])),
    );
    let without = render(&mut gpu, &mut plain, &cam);
    let mut with = world(
        &gpu,
        env,
        Some(probe::Probe::new(&gpu, rect, depth, [0.3; 3])),
    );
    let clouds = VolumetricCloudsPhase::install(&mut with);
    clouds.borrow_mut().request(&gpu);
    assert!(clouds.borrow_mut().wait_ready(READY));
    let mut frame = Vec::new();
    for _ in 0..6 {
        assert!(drive(&clouds, &env, &cam, 0.7, CloudQuality::High));
        frame = render(&mut gpu, &mut with, &cam);
    }
    let n = changed(&frame, &without, size.0, 0..size.1 / 2, 6);
    assert!(n > 200, "clouds visible under MSAA ({n} pixels changed)");
}

#[test]
fn above_the_layer_they_draw_late_over_the_ground() {
    let size = (320, 180);
    let Some(mut gpu) = headless(size.0, size.1, 1) else {
        return;
    };
    let env = environment(6000, 0.0);
    // 300 blocks up, looking down at the ground through the cloud layer (128..224).
    let cam = camera(DVec3::new(40.0, 300.0, 40.0), 10.0, 70.0, size);
    let ground = |gpu: &Gpu| {
        probe::Probe::new(
            gpu,
            [-1.0, -1.0, 1.0, 1.0],
            ndc_depth(&cam, 340.0),
            [0.2, 0.45, 0.15],
        )
    };
    let mut plain = world(&gpu, env, Some(ground(&gpu)));
    let without = render(&mut gpu, &mut plain, &cam);
    let mut with = world(&gpu, env, Some(ground(&gpu)));
    let clouds = VolumetricCloudsPhase::install(&mut with);
    clouds.borrow_mut().request(&gpu);
    assert!(clouds.borrow_mut().wait_ready(READY));
    let mut frame = Vec::new();
    for _ in 0..8 {
        assert!(drive(&clouds, &env, &cam, 0.6, CloudQuality::Medium));
        assert_eq!(
            clouds.borrow().drawing_order(),
            Some(volumetric::LATE_ORDER),
            "in or above the layer the clouds draw after translucent terrain"
        );
        frame = render(&mut gpu, &mut with, &cam);
    }
    let n = changed(&frame, &without, size.0, 0..size.1, 8);
    assert!(
        n > (size.0 * size.1 / 10) as usize,
        "clouds between the eye and the ground ({n} pixels changed)"
    );
}

#[test]
fn the_slots_sit_where_vanilla_draws_clouds() {
    assert!(volumetric::EARLY_ORDER > PhaseOrder::ENTITIES);
    assert!(volumetric::EARLY_ORDER < PhaseOrder::PARTICLES);
    assert!(volumetric::EARLY_ORDER < PhaseOrder::TERRAIN_TRANSLUCENT);
    assert!(volumetric::LATE_ORDER > PhaseOrder::TERRAIN_TRANSLUCENT);
    assert!(volumetric::LATE_ORDER > mc_render::sky::FogPhase::ORDER);
    assert!(volumetric::LATE_ORDER < PhaseOrder::FIRST_PERSON);
    let layer = CloudLayer::in_game(128.0);
    assert_eq!(layer.bottom, 128.0, "the vanilla cloud height");
}

// ------------------------------------------------------------------------------- for humans

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("repository root")
        .to_path_buf()
}

fn out_dir() -> PathBuf {
    let dir = std::env::var_os("X11_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            root()
                .join("featwave-2026-11-09")
                .join("scratch")
                .join("X11")
        });
    std::fs::create_dir_all(&dir).expect("create the output folder");
    dir
}

/// In-game views at 1920x1080 for review: a clear morning, a sunset, rain, a night and a view
/// from above the clouds, each with and without the clouds. `X11_TAG` prefixes the names.
#[test]
#[ignore = "writes 1920x1080 PNGs for review; run by hand"]
fn render_review_pngs() {
    let size = (1920, 1080);
    let Some(mut gpu) = headless(size.0, size.1, 1) else {
        return;
    };
    let dir = out_dir();
    let tag = std::env::var("X11_TAG").unwrap_or_else(|_| "v".to_owned());
    let quality = std::env::var("X11_QUALITY")
        .ok()
        .and_then(|q| q.parse::<usize>().ok())
        .map_or(CloudQuality::Medium, CloudQuality::from_index);
    let coverage: f32 = std::env::var("X11_COVERAGE")
        .ok()
        .and_then(|c| c.parse().ok())
        .unwrap_or(0.5);
    let shots: [(&str, i64, f32, DVec3, f32, f32); 6] = [
        (
            "morning",
            2500,
            0.0,
            DVec3::new(0.0, 75.0, 0.0),
            300.0,
            -18.0,
        ),
        (
            "noon-up",
            6000,
            0.0,
            DVec3::new(300.0, 70.0, 50.0),
            140.0,
            -55.0,
        ),
        (
            "sunset",
            11_900,
            0.0,
            DVec3::new(-500.0, 80.0, 900.0),
            95.0,
            -12.0,
        ),
        ("rain", 6000, 1.0, DVec3::new(0.0, 75.0, 0.0), 300.0, -18.0),
        (
            "night",
            17_000,
            0.0,
            DVec3::new(0.0, 75.0, 0.0),
            120.0,
            -25.0,
        ),
        ("above", 4000, 0.0, DVec3::new(0.0, 260.0, 0.0), 300.0, 25.0),
    ];
    for (name, time, rain, eye, yaw, pitch) in shots {
        let env = environment(time, rain);
        let cam = camera(eye, yaw, pitch, size);
        let ground = probe::Probe::new(
            &gpu,
            [-1.0, -1.0, 1.0, if name == "above" { 1.0 } else { -0.55 }],
            ndc_depth(&cam, if name == "above" { 330.0 } else { 60.0 }),
            [0.24, 0.42, 0.16],
        );
        let mut phases = world(&gpu, env, Some(ground));
        let clouds = VolumetricCloudsPhase::install(&mut phases);
        let base = render(&mut gpu, &mut phases, &cam);
        let path = dir.join(format!("{tag}-{name}-off.png"));
        image::save_buffer(&path, &base, size.0, size.1, image::ColorType::Rgba8).expect("png");
        clouds.borrow_mut().request(&gpu);
        assert!(clouds.borrow_mut().wait_ready(READY));
        let mut frame = Vec::new();
        for _ in 0..40 {
            assert!(drive(&clouds, &env, &cam, coverage, quality));
            frame = render(&mut gpu, &mut phases, &cam);
        }
        let path = dir.join(format!("{tag}-{name}.png"));
        image::save_buffer(&path, &frame, size.0, size.1, image::ColorType::Rgba8).expect("png");
        eprintln!("wrote {}", path.display());
    }
}

/// GPU cost at 1920x1080 per quality preset, with `MC_GPU_TIMING=1` (provisional: the numbers are noisy).
#[test]
#[ignore = "GPU timing at 1080p; run by hand with MC_GPU_TIMING=1"]
fn gpu_cost() {
    let size = (1920, 1080);
    let Some(mut gpu) = headless(size.0, size.1, 1) else {
        return;
    };
    let env = environment(5000, 0.0);
    let cam = camera(DVec3::new(0.0, 72.0, 0.0), 300.0, -20.0, size);
    for quality in [CloudQuality::Low, CloudQuality::Medium, CloudQuality::High] {
        let mut phases = world(&gpu, env, None);
        let clouds = VolumetricCloudsPhase::install(&mut phases);
        clouds.borrow_mut().request(&gpu);
        assert!(clouds.borrow_mut().wait_ready(READY));
        for _ in 0..240 {
            assert!(drive(&clouds, &env, &cam, 0.5, quality));
            let frame = gpu.acquire().expect("acquire");
            phases.render(&gpu, frame.view(), &cam, 0.0);
        }
        let _ = gpu.read_pixels();
        eprintln!("{quality:?}\n{}", phases.gpu_timings().format_table());
    }
}

// ------------------------------------------- Close to the layer

/// One vanilla jump as eye heights above the take-off, one per tick until it lands
/// (`EntityLivingBase.jump`: 0.42 up; `moveEntityWithHeading`: gravity 0.08, drag 0.98).
fn jump_heights() -> Vec<f64> {
    let (mut y, mut vy) = (0.0f64, 0.42f64);
    let mut out = vec![0.0];
    loop {
        y += vy;
        vy = (vy - 0.08) * 0.98;
        if y <= 0.0 {
            out.push(0.0);
            return out;
        }
        out.push(y);
    }
}

/// The eye heights of a jump drawn at `per_tick` frames per tick (interpolated, as the game
/// draws between ticks), starting from `ground`, plus a few still frames after landing.
fn jump_frames(ground: f64, per_tick: usize) -> Vec<f64> {
    let ticks = jump_heights();
    let mut out = Vec::new();
    for pair in ticks.windows(2) {
        for f in 0..per_tick {
            let t = f as f64 / per_tick as f64;
            out.push(ground + pair[0] + (pair[1] - pair[0]) * t);
        }
    }
    out.extend(std::iter::repeat_n(ground, per_tick * 2));
    out
}

/// How many pixels of `a` differ from `b` by more than `threshold` in some colour channel.
fn off_by(a: &[u8], b: &[u8], threshold: u8) -> usize {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(p, q)| {
            p.iter()
                .zip(q.iter())
                .take(3)
                .any(|(u, v)| u.abs_diff(*v) > threshold)
        })
        .count()
}

/// One B5 scenario: a jump (`per_tick` frames per tick) from eye height `ground`, looking along
/// `yaw`/`pitch`, drawn through the real phase with its history, every frame compared with a
/// converged frame from a second phase that sat still at the same pose (`settle` frames, its own
/// history). Returns, per frame, the eye height and how many pixels are off by more than
/// `threshold`, plus the worst frame and its reference.
struct JumpReport {
    rows: Vec<(f64, usize)>,
    worst: (usize, Vec<u8>, Vec<u8>),
    frames: Vec<Vec<u8>>,
}

#[allow(clippy::too_many_arguments)]
fn jump_through(
    gpu: &mut Gpu,
    size: (u32, u32),
    env: &Environment,
    layer: CloudLayer,
    ground: f64,
    (yaw, turn): (f32, f32),
    pitch: f32,
    per_tick: usize,
    settle: usize,
    threshold: u8,
) -> JumpReport {
    let coverage = 0.75;
    let mut moving = world(gpu, *env, None);
    let clouds = VolumetricCloudsPhase::install(&mut moving);
    clouds.borrow_mut().layer = layer;
    clouds.borrow_mut().request(gpu);
    assert!(clouds.borrow_mut().wait_ready(READY));
    let mut still = world(gpu, *env, None);
    let reference = VolumetricCloudsPhase::install(&mut still);
    reference.borrow_mut().layer = layer;
    reference.borrow_mut().request(gpu);
    assert!(reference.borrow_mut().wait_ready(READY));

    let xz = (10.5, -3.25);
    let mut cam = camera(DVec3::new(xz.0, ground, xz.1), yaw, pitch, size);
    // Standing still first, so the history is converged when the jump starts.
    for _ in 0..settle {
        assert!(drive(&clouds, env, &cam, coverage, CloudQuality::Medium));
        render(gpu, &mut moving, &cam);
    }
    let mut report = JumpReport {
        rows: Vec::new(),
        worst: (0, Vec::new(), Vec::new()),
        frames: Vec::new(),
    };
    for y in jump_frames(ground, per_tick) {
        cam.pos.y = y;
        cam.yaw += turn;
        cam.update();
        drive(&clouds, env, &cam, coverage, CloudQuality::Medium);
        let frame = render(gpu, &mut moving, &cam);
        reference.borrow_mut().reset_history();
        let mut truth = Vec::new();
        for _ in 0..settle {
            drive(&reference, env, &cam, coverage, CloudQuality::Medium);
            truth = render(gpu, &mut still, &cam);
        }
        let bad = off_by(&frame, &truth, threshold);
        if bad >= report.worst.0 {
            report.worst = (bad, frame.clone(), truth);
        }
        report.rows.push((y, bad));
        report.frames.push(frame);
    }
    report
}

/// Where the B5 pictures go: `TW_B_PNG_DIR`, else nowhere.
fn b5_png_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("TW_B_PNG_DIR")?);
    std::fs::create_dir_all(&dir).expect("create the PNG folder");
    Some(dir)
}

/// `frames` side by side in rows of `per_row`, for a contact sheet.
fn sheet(frames: &[&[u8]], size: (u32, u32), per_row: u32) -> image::RgbaImage {
    let rows = (frames.len() as u32).div_ceil(per_row);
    let mut out = image::RgbaImage::new(size.0 * per_row, size.1 * rows);
    for (i, frame) in frames.iter().enumerate() {
        let tile = image::RgbaImage::from_raw(size.0, size.1, frame.to_vec()).expect("frame size");
        let (x, y) = ((i as u32 % per_row) * size.0, (i as u32 / per_row) * size.1);
        image::imageops::replace(&mut out, &tile, i64::from(x), i64::from(y));
    }
    out
}

/// Jumping right under the clouds, into them and inside them, as the
/// user saw it ("they glitch out and stretch to the top or bottom of the screen, usually when
/// jumping"). Each frame of a 60 fps jump is compared with a converged still frame at the same
/// eye: a correct accumulation stays within the trace's noise of it. Writes contact sheets to
/// `TW_B_PNG_DIR` when set.
///
/// Before the fix the jump through the bases put up to 23,020 of the 57,600 pixels more than 48
/// levels off (the cloud ceiling smeared over the sky below the horizon); the other cases were
/// already within noise. After it every frame of every case is.
#[test]
fn b5_jumping_close_to_the_clouds_matches_a_still_camera() {
    let size = (320, 180);
    let Some(mut gpu) = headless(size.0, size.1, 1) else {
        return;
    };
    let env = environment(6000, 0.0);
    let layer = CloudLayer::in_game(128.0);
    let tag = std::env::var("TW_B_TAG").unwrap_or_else(|_| "now".to_owned());
    // (name, eye height on the ground, (yaw, turn per frame), pitch): 3 blocks under the bases
    // looking up; just under them (the jump's top stays below); a jump that takes the eye
    // through the bases, still and while turning; inside the layer; just above the tops
    // looking down.
    let scenarios: [(&str, f64, (f32, f32), f32); 6] = [
        ("under", 125.0, (30.0, 0.0), -25.0),
        ("just-under", 126.6, (30.0, 0.0), -15.0),
        ("through", 127.4, (30.0, 0.0), -8.0),
        ("through-turning", 127.4, (30.0, 1.5), -8.0),
        ("inside", 150.0, (30.0, 0.0), 4.0),
        ("over", 249.0, (30.0, 0.0), 30.0),
    ];
    let mut worst_total = 0;
    for (name, ground, yaw, pitch) in scenarios {
        let report = jump_through(&mut gpu, size, &env, layer, ground, yaw, pitch, 3, 48, 48);
        let total: usize = report.rows.iter().map(|(_, bad)| bad).sum();
        let peak = report.rows.iter().map(|(_, bad)| *bad).max().unwrap_or(0);
        eprintln!(
            "B5 {name}: {} frames, {total} pixel-frames off by >48, worst frame {peak} of {} px",
            report.rows.len(),
            size.0 * size.1
        );
        for (y, bad) in &report.rows {
            eprint!("{y:.2}:{bad} ");
        }
        eprintln!();
        if let Some(dir) = b5_png_dir() {
            let frames: Vec<&[u8]> = report.frames.iter().map(Vec::as_slice).collect();
            sheet(&frames, size, 6)
                .save(dir.join(format!("b5-{tag}-{name}-jump.png")))
                .expect("png");
            sheet(&[&report.worst.1, &report.worst.2], size, 2)
                .save(dir.join(format!("b5-{tag}-{name}-worst-vs-still.png")))
                .expect("png");
        }
        worst_total = worst_total.max(peak);
        assert!(
            peak < (size.0 * size.1 / 200) as usize,
            "B5 {name}: a jump frame strays from the still camera's clouds ({peak} px)"
        );
    }
    eprintln!("B5 worst frame overall: {worst_total} px");
}

/// B5 "Above you": a layer moved up with `CloudLayer::in_game_at` carries its clouds with it.
/// The view from 60 blocks higher of the layer 60 blocks higher is the same picture (float
/// rounding apart); the same move without the noise shift (the negative control) shows other
/// clouds, because the 3D noise would be read 60 blocks higher. The camera looks steeply up, where
/// vanilla's sky itself does not depend on the eye's height (checked first).
#[test]
fn b5_a_moved_layer_carries_its_clouds() {
    let size = (320, 180);
    let Some(mut gpu) = headless(size.0, size.1, 1) else {
        return;
    };
    let env = environment(6000, 0.0);
    let pitch = -55.0;
    let sky = |gpu: &mut Gpu, y: f64| {
        let mut phases = world(gpu, env, None);
        let cam = camera(DVec3::new(10.5, y, -3.25), 30.0, pitch, size);
        render(gpu, &mut phases, &cam)
    };
    let (sky_low, sky_high) = (sky(&mut gpu, 72.0), sky(&mut gpu, 132.0));
    assert_eq!(
        off_by(&sky_low, &sky_high, 0),
        0,
        "the sky alone is the same up there"
    );
    let mut render_at = |layer: CloudLayer, eye_y: f64| {
        let mut phases = world(&gpu, env, None);
        let clouds = VolumetricCloudsPhase::install(&mut phases);
        clouds.borrow_mut().layer = layer;
        clouds.borrow_mut().request(&gpu);
        assert!(clouds.borrow_mut().wait_ready(READY));
        let cam = camera(DVec3::new(10.5, eye_y, -3.25), 30.0, pitch, size);
        let mut frame = Vec::new();
        for _ in 0..24 {
            assert!(drive(&clouds, &env, &cam, 0.6, CloudQuality::Medium));
            frame = render(&mut gpu, &mut phases, &cam);
        }
        frame
    };
    let low = render_at(CloudLayer::in_game(128.0), 72.0);
    let moved = CloudLayer::in_game_at(188.0, 128.0);
    assert_eq!(moved.noise_shift, 60.0);
    let high = render_at(moved, 132.0);
    let unshifted = render_at(
        CloudLayer {
            noise_shift: 0.0,
            ..moved
        },
        132.0,
    );
    let pixels = (size.0 * size.1) as usize;
    let same = off_by(&low, &high, 12);
    let other = off_by(&low, &unshifted, 12);
    eprintln!("B5 moved layer: {same} px differ with the shift, {other} without (of {pixels})");
    if let Some(dir) = b5_png_dir() {
        sheet(&[&low, &high, &unshifted], size, 3)
            .save(dir.join("b5-moved-layer-low-high-unshifted.png"))
            .expect("png");
    }
    assert!(
        same < pixels / 200,
        "the moved layer shows other clouds ({same} px)"
    );
    assert!(
        other > pixels / 10,
        "the negative control must differ ({other} px)"
    );
}

/// A phase switched on and off from outside: an occluder that moves away.
struct Toggled {
    inner: probe::Probe,
    on: Rc<std::cell::Cell<bool>>,
}

impl mc_render::RenderPhase for Toggled {
    fn name(&self) -> &'static str {
        "x11 toggled probe"
    }

    fn enabled(&self) -> bool {
        self.on.get()
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        mc_render::RenderPhase::record(&mut self.inner, gpu, enc, view, cam);
    }
}

/// B5 (found in the SkyWars scene: a cloud-free copy of a nametag hung in the clouds above it):
/// something in front of the clouds (a player, a nametag) that moves away must leave no
/// cloud-free ghost of itself. The clouds converge, an occluder 6 blocks away covers a patch of
/// sky for 12 frames, then goes: the uncovered patch must show the clouds at once, as the frame
/// from before it came did. Before the fix the occluded texels wrote "no cloud" into the history,
/// so the patch came back as clear sky fading in over the next frames.
#[test]
fn b5_an_occluder_moving_away_leaves_no_ghost_in_the_clouds() {
    let size = (320, 180);
    let Some(mut gpu) = headless(size.0, size.1, 1) else {
        return;
    };
    let env = environment(6000, 0.0);
    let cam = camera(DVec3::new(10.5, 74.0, -3.25), 30.0, -20.0, size);
    let rect = [-0.3, 0.2, 0.3, 0.8];
    let on = Rc::new(std::cell::Cell::new(false));
    let mut phases = world(&gpu, env, None);
    phases.insert(
        PhaseOrder::TERRAIN_SOLID,
        Box::new(Toggled {
            inner: probe::Probe::new(&gpu, rect, ndc_depth(&cam, 6.0), [0.42, 0.31, 0.2]),
            on: Rc::clone(&on),
        }),
    );
    let clouds = VolumetricCloudsPhase::install(&mut phases);
    clouds.borrow_mut().request(&gpu);
    assert!(clouds.borrow_mut().wait_ready(READY));
    let coverage = 0.85;
    let mut frame = Vec::new();
    for _ in 0..40 {
        assert!(drive(&clouds, &env, &cam, coverage, CloudQuality::Medium));
        frame = render(&mut gpu, &mut phases, &cam);
    }
    let before = frame;
    on.set(true);
    for _ in 0..12 {
        drive(&clouds, &env, &cam, coverage, CloudQuality::Medium);
        render(&mut gpu, &mut phases, &cam);
    }
    on.set(false);
    drive(&clouds, &env, &cam, coverage, CloudQuality::Medium);
    let after = render(&mut gpu, &mut phases, &cam);
    // The patch, a few pixels inside the occluder's edges.
    let (x0, x1) = (
        ((rect[0] + 1.0) * 0.5 * size.0 as f32) as u32 + 4,
        ((rect[2] + 1.0) * 0.5 * size.0 as f32) as u32 - 4,
    );
    let (y0, y1) = (
        ((1.0 - rect[3]) * 0.5 * size.1 as f32) as u32 + 4,
        ((1.0 - rect[1]) * 0.5 * size.1 as f32) as u32 - 4,
    );
    let mut ghost = 0;
    let mut cloudy = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            let (p, q) = (pixel(&after, size.0, x, y), pixel(&before, size.0, x, y));
            if p.iter()
                .zip(q.iter())
                .take(3)
                .any(|(u, v)| u.abs_diff(*v) > 24)
            {
                ghost += 1;
            }
            // A cloudy pixel is grey-white: its blue channel is not far above its red.
            if i32::from(q[2]) - i32::from(q[0]) < 90 {
                cloudy += 1;
            }
        }
    }
    let patch = ((x1 - x0) * (y1 - y0)) as usize;
    eprintln!("B5 occluder: {ghost} of {patch} patch pixels off after it moved ({cloudy} cloudy)");
    if let Some(dir) = b5_png_dir() {
        sheet(&[&before, &after], size, 2)
            .save(dir.join("b5-occluder-before-after.png"))
            .expect("png");
    }
    assert!(
        cloudy > patch / 3,
        "the patch must be cloudy for this to test anything"
    );
    assert!(
        ghost < patch / 20,
        "a ghost of the occluder in the clouds ({ghost} px)"
    );
}

/// Jumping must not reproject a cloud onto rays that point away from the layer.
#[test]
fn jumping_keeps_cloud_history_above_the_horizon() {
    let size = (192, 128);
    let Some(mut gpu) = headless(size.0, size.1, 1) else {
        return;
    };
    let env = environment(6000, 0.0);
    let mut plain = world(&gpu, env, None);
    let mut with = world(&gpu, env, None);
    let clouds = VolumetricCloudsPhase::install(&mut with);
    clouds.borrow_mut().request(&gpu);
    assert!(clouds.borrow_mut().wait_ready(READY));
    let mut cam = camera(DVec3::new(10.5, 72.0, -3.25), 30.0, 0.0, size);
    for _ in 0..16 {
        assert!(drive(&clouds, &env, &cam, 0.85, CloudQuality::Medium));
        render(&mut gpu, &mut with, &cam);
    }
    let before = render(&mut gpu, &mut with, &cam);
    let base = render(&mut gpu, &mut plain, &cam);
    assert!(
        changed(&before, &base, size.0, 0..size.1 / 2, 3) > 100,
        "cloud history must contain clouds"
    );
    for y in [
        72.42, 72.7532, 73.0013, 73.1661, 73.2492, 73.17, 72.9, 72.5, 72.0,
    ] {
        cam.pos.y = y;
        cam.update();
        let base = render(&mut gpu, &mut plain, &cam);
        assert!(drive(&clouds, &env, &cam, 0.85, CloudQuality::Medium));
        let frame = render(&mut gpu, &mut with, &cam);
        assert_eq!(
            changed(&frame, &base, size.0, size.1 / 2 + 2..size.1, 0),
            0,
            "cloud history leaked below the horizon at eye height {y}"
        );
    }
}
