//! GPU readbacks for the **Fog** and **Vignette** mods.
//!
//! Both mods default **on**, which is vanilla. The claims tested here:
//!
//! * with the mods on (the default inputs) every frame is today's frame byte for byte: the
//!   `w4_golden_*` capture/compare test (see [`golden`]) checks that across builds, and the
//!   in-binary tests check every on/off pair;
//! * Fog off removes exactly the render-distance fog (`setupFog`'s air branch, Nether/End
//!   included) from the world pass and the clouds, while blindness, in-cloud, water and lava fog
//!   and the sky pass's horizon fade stay;
//! * Vignette off removes exactly `GuiIngame.renderVignette`'s darkening, while the world-border
//!   warning tint stays.
//!
//! Same technique as `m4_visual_readback.rs`: the real phases through the real `PhaseList` into
//! a headless target. Tests that need `1.8.9.jar` (clouds, vignette texture) skip with `SKIP:`
//! when it is missing; point `MC_JARS_DIR` at a `jars/` directory to override its location.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use glam::DVec3;
use mc_assets::{ResourceManager, ZipResourcePack};
use mc_render::post::OverlayPhase;
use mc_render::sky::clouds::{CloudMode, CloudsPhase};
use mc_render::sky::dome::SkyPhase;
use mc_render::sky::{
    CameraMedium, Dimension, Environment, FogDistance, FogInputs, FogPhase, FogState, Weather,
};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError, RenderPhase};

const SIZE: u32 = 64;
/// The clear colour of the overlay scenes: mid grey, so the vignette's darkening and the
/// border warning's cyan suppression both show.
const GREY: wgpu::Color = wgpu::Color {
    r: 0.5,
    g: 0.5,
    b: 0.5,
    a: 1.0,
};
/// The clear colour of the fog scenes.
const BLACK: wgpu::Color = wgpu::Color {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};
/// Eye-space distance of column 0 of [`DepthRamp`], in blocks.
const RAMP_START: f32 = 2.0;
/// Blocks per column of [`DepthRamp`]: 64 columns span 2..254 blocks.
const RAMP_STEP: f32 = 4.0;
/// The depth ramp's surface colour.
const RAMP_COLOR: [f32; 3] = [1.0, 0.6, 0.2];

fn gpu() -> Option<Gpu> {
    gpu_sized(SIZE, SIZE)
}

fn gpu_sized(width: u32, height: u32) -> Option<Gpu> {
    match Gpu::headless(
        width,
        height,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(g) => {
            // Turn wgpu validation errors into test failures instead of silent no-draws.
            g.device.on_uncaptured_error(std::sync::Arc::new(|e| {
                panic!("wgpu validation error: {e}")
            }));
            Some(g)
        }
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            None
        }
        Err(e) => panic!("headless context: {e}"),
    }
}

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

/// A phase the test keeps a handle on (the client's `Shared`).
struct Handle<T>(Rc<RefCell<T>>);

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

    fn enabled(&self) -> bool {
        self.0.borrow().enabled()
    }
}

/// One phase for [`render`]'s list.
fn boxed<T: RenderPhase + 'static>(phase: T) -> Box<dyn RenderPhase> {
    Box::new(phase)
}

/// A fog-scene configuration for [`render_fog_ramp`].
type FogScene = Box<dyn Fn(&mut FogPhase)>;

fn fog_scene(configure: impl Fn(&mut FogPhase) + 'static) -> FogScene {
    Box::new(configure)
}

/// An overlay-scene configuration for [`render_overlay`].
type OverlayScene = Box<dyn Fn(&mut OverlayPhase)>;

fn overlay_scene(configure: impl Fn(&mut OverlayPhase) + 'static) -> OverlayScene {
    Box::new(configure)
}

/// Render `phases` over `clear` and read the frame back.
fn render(
    gpu: &mut Gpu,
    cam: &Camera,
    clear: wgpu::Color,
    phases: Vec<(PhaseOrder, Box<dyn RenderPhase>)>,
) -> Vec<u8> {
    let mut list = PhaseList::new();
    list.clear_color = clear;
    for (order, phase) in phases {
        list.insert(order, phase);
    }
    let frame = gpu.acquire().expect("acquire");
    list.render(gpu, frame.view(), cam, 0.0);
    drop(frame);
    gpu.read_pixels().expect("readback")
}

fn at(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * SIZE + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

/// How many pixels differ between two frames.
fn differing(a: &[u8], b: &[u8]) -> usize {
    assert_eq!(a.len(), b.len());
    a.chunks(4).zip(b.chunks(4)).filter(|(p, q)| p != q).count()
}

// ---- the depth ramp: one surface whose eye distance grows column by column ----------------

/// A full-screen surface that writes depth so column `x` sits `RAMP_START + x * RAMP_STEP`
/// blocks from the eye (planar), in [`RAMP_COLOR`]. One frame therefore samples the whole fog
/// curve, from clear to fully fogged.
struct DepthRamp {
    pipeline: wgpu::RenderPipeline,
    buffer: wgpu::Buffer,
    group: wgpu::BindGroup,
}

const RAMP_WGSL: &str = r#"
struct P {
    // x = proj[2][2], y = proj[3][2], z = blocks per column, w = distance of column 0.
    proj: vec4<f32>,
    color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> p: P;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var c = array<vec2<f32>, 3>(vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0));
    return vec4<f32>(c[i], 0.5, 1.0);
}

struct Out {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> Out {
    // Planar eye distance d maps to depth (b - a*d) / d under the right-handed 0..1 projection;
    // `sky/fog.wgsl` inverts exactly this.
    let d = p.proj.w + floor(pos.x) * p.proj.z;
    var out: Out;
    out.depth = (p.proj.y - p.proj.x * d) / d;
    out.color = p.color;
    return out;
}
"#;

impl DepthRamp {
    fn new(gpu: &Gpu) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("w4 depth ramp"),
                source: wgpu::ShaderSource::Wgsl(RAMP_WGSL.into()),
            });
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("w4 depth ramp"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: mc_render::DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
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
                    compilation_options: Default::default(),
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
            label: None,
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self {
            pipeline,
            buffer,
            group,
        }
    }
}

impl RenderPhase for DepthRamp {
    fn name(&self) -> &'static str {
        "test/w4-depth-ramp"
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        let uniform = [
            cam.proj.z_axis.z,
            cam.proj.w_axis.z,
            RAMP_STEP,
            RAMP_START,
            RAMP_COLOR[0],
            RAMP_COLOR[1],
            RAMP_COLOR[2],
            1.0,
        ];
        gpu.queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(&uniform));
        let mut pass = gpu.begin_pass(
            enc,
            view,
            "test/w4-depth-ramp",
            mc_render::DepthAttachment::Shared,
        );
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// The fog scenes' camera: straight ahead, far plane at 32 chunks so the whole ramp is inside
/// the depth range.
fn ramp_camera() -> Camera {
    let mut cam = Camera::default();
    cam.set_viewport(SIZE, SIZE);
    cam.set_view_distance_chunks(32);
    cam.update();
    cam
}

/// Noon in the overworld, the environment every fog scene uses unless it says otherwise.
fn noon(dimension: Dimension) -> Environment {
    Environment {
        world_time: 6_000,
        dimension,
        weather: Weather::default(),
    }
}

/// Vanilla's inputs at `render_distance_chunks`, everything else default.
fn inputs(render_distance_chunks: u32) -> FogInputs {
    FogInputs {
        render_distance_chunks,
        ..FogInputs::default()
    }
}

/// The depth ramp with a [`FogPhase`] over it, the client's way (radial distance), configured
/// by `configure`. Returns the frame and whether the fog phase was enabled.
fn render_fog_ramp(gpu: &mut Gpu, configure: impl FnOnce(&mut FogPhase)) -> (Vec<u8>, bool) {
    let mut fog = FogPhase::new(gpu);
    fog.environment = noon(Dimension::Overworld);
    fog.inputs = inputs(8);
    fog.set_distance(FogDistance::Radial);
    configure(&mut fog);
    let fog = Rc::new(RefCell::new(fog));
    let enabled = fog.borrow().enabled();
    let phases = vec![
        (PhaseOrder::TERRAIN_SOLID, boxed(DepthRamp::new(gpu))),
        (FogPhase::ORDER, boxed(Handle(fog))),
    ];
    let frame = render(gpu, &ramp_camera(), BLACK, phases);
    (frame, enabled)
}

/// The depth ramp alone: what every fog scene looks like with no fog at all.
fn render_unfogged_ramp(gpu: &mut Gpu) -> Vec<u8> {
    let phases = vec![(PhaseOrder::TERRAIN_SOLID, boxed(DepthRamp::new(gpu)))];
    render(gpu, &ramp_camera(), BLACK, phases)
}

/// The vanilla fog scenes, by name: every branch of `setupFog(0, ..)`.
fn vanilla_fog_scenes() -> Vec<(&'static str, FogScene)> {
    vec![
        ("fog-air-rd8", fog_scene(|_| {})),
        (
            "fog-air-rd8-planar",
            fog_scene(|f| f.set_distance(FogDistance::Planar)),
        ),
        (
            "fog-air-rd16",
            fog_scene(|f| f.inputs.render_distance_chunks = 16),
        ),
        (
            "fog-air-rd8-rain-dusk",
            fog_scene(|f| {
                f.environment.world_time = 12_500;
                f.environment.weather.rain_strength = 0.6;
                f.inputs.sunset_look_dot = 0.5;
            }),
        ),
        (
            "fog-nether-rd8",
            fog_scene(|f| {
                f.environment = noon(Dimension::Nether);
                f.inputs.dimension_always_fogs = true;
            }),
        ),
        (
            "fog-end-rd8",
            fog_scene(|f| {
                f.environment = noon(Dimension::End);
                f.inputs.dimension_always_fogs = true;
            }),
        ),
        (
            "fog-water",
            fog_scene(|f| f.inputs.medium = CameraMedium::Water),
        ),
        (
            "fog-water-respiration",
            fog_scene(|f| {
                f.inputs.medium = CameraMedium::Water;
                f.inputs.respiration_level = 2;
            }),
        ),
        (
            "fog-lava",
            fog_scene(|f| f.inputs.medium = CameraMedium::Lava),
        ),
        (
            "fog-cloud",
            fog_scene(|f| f.inputs.medium = CameraMedium::Cloud),
        ),
        (
            "fog-blind-full",
            fog_scene(|f| {
                f.inputs.blindness = 1.0;
                f.inputs.blindness_duration_ticks = Some(100);
            }),
        ),
        (
            "fog-blind-fading",
            fog_scene(|f| {
                f.inputs.blindness = 0.5;
                f.inputs.blindness_duration_ticks = Some(10);
            }),
        ),
        (
            "fog-blind-in-water",
            fog_scene(|f| {
                f.inputs.medium = CameraMedium::Water;
                f.inputs.blindness = 1.0;
                f.inputs.blindness_duration_ticks = Some(100);
            }),
        ),
    ]
}

// ---- clouds -------------------------------------------------------------------------------

/// Fancy clouds seen 20 degrees above the horizon from y=100 at render distance 8, fogged with
/// `fog` the client's way (radial distance). The cloud layer runs from ~34 blocks away at the
/// top of the frame to beyond the 128-block fog end near the horizon.
fn render_clouds(gpu: &mut Gpu, rm: &ResourceManager, fog: Option<FogState>) -> Vec<u8> {
    let mut clouds = CloudsPhase::new(gpu);
    clouds.load_texture(gpu, rm);
    clouds.mode = CloudMode::Fancy;
    clouds.set_fog(fog.map(|state| (state, FogDistance::Radial)));
    clouds.environment = noon(Dimension::Overworld);
    let mut cam = Camera {
        pos: DVec3::new(0.0, 100.0, 0.0),
        pitch: -20.0,
        ..Camera::default()
    };
    cam.set_viewport(SIZE, SIZE);
    cam.set_view_distance_chunks(8);
    cam.update();
    render(gpu, &cam, BLACK, vec![(PhaseOrder::SKY, boxed(clouds))])
}

/// The world-pass fog state the client hands the clouds for `inputs` at noon.
fn cloud_fog(inputs: FogInputs) -> FogState {
    FogState::compute(&noon(Dimension::Overworld), inputs, 0.0)
}

// ---- sky ----------------------------------------------------------------------------------

/// The sky dome at noon looking at the horizon from y=70 at render distance 8, with the sky
/// pass's fog inputs configured by `configure`. The upper half is the fogged sky plane fading
/// into the fog colour, the lower half the fogged horizon plane.
fn render_sky(gpu: &mut Gpu, configure: impl FnOnce(&mut FogInputs)) -> Vec<u8> {
    let mut sky = SkyPhase::new(gpu);
    sky.environment = noon(Dimension::Overworld);
    configure(&mut sky.fog_inputs);
    let mut cam = Camera {
        pos: DVec3::new(0.0, 70.0, 0.0),
        pitch: 0.0,
        ..Camera::default()
    };
    cam.set_viewport(SIZE, SIZE);
    cam.set_view_distance_chunks(8);
    cam.update();
    render(gpu, &cam, BLACK, vec![(PhaseOrder::SKY, boxed(sky))])
}

// ---- vignette -----------------------------------------------------------------------------

/// The world-border warning colour the client feeds at a few blocks from the border:
/// `(0, f, f, 1)`.
const BORDER_WARNING: [f32; 4] = [0.0, 0.6, 0.6, 1.0];

/// `OverlayPhase` with the real `vignette.png`, configured by `configure`, over grey. Returns
/// the frame and whether the phase was enabled.
fn render_overlay(
    gpu: &mut Gpu,
    rm: &ResourceManager,
    configure: impl FnOnce(&mut OverlayPhase),
) -> (Vec<u8>, bool) {
    let mut overlay = OverlayPhase::new(gpu);
    overlay.load_vignette_texture(gpu, rm);
    assert!(
        overlay.has_vignette_texture(),
        "vignette.png loads from the jar"
    );
    configure(&mut overlay);
    let overlay = Rc::new(RefCell::new(overlay));
    let enabled = overlay.borrow().enabled();
    let mut cam = Camera::default();
    cam.set_viewport(SIZE, SIZE);
    cam.update();
    let frame = render(
        gpu,
        &cam,
        GREY,
        vec![(PhaseOrder::POST, boxed(Handle(overlay)))],
    );
    (frame, enabled)
}

/// A dark place: player brightness 0 keeps `prevVignetteBrightness` at its fully dark start.
fn dark_vignette(phase: &mut OverlayPhase) {
    phase.set_vignette_brightness(Some(0.0));
}

/// A half-lit place after the brightness has eased for a while.
fn half_vignette(phase: &mut OverlayPhase) {
    for _ in 0..200 {
        phase.set_vignette_brightness(Some(0.5));
    }
}

fn vanilla_overlay_scenes() -> Vec<(&'static str, OverlayScene)> {
    vec![
        ("vignette-dark", overlay_scene(dark_vignette)),
        ("vignette-half", overlay_scene(half_vignette)),
        (
            "vignette-dark-border",
            overlay_scene(|p| {
                dark_vignette(p);
                p.set_border_warning(Some(BORDER_WARNING));
            }),
        ),
        (
            "vignette-fast-border",
            overlay_scene(|p| {
                p.set_vignette_brightness(None);
                p.set_border_warning(Some(BORDER_WARNING));
            }),
        ),
    ]
}

// ---- golden frames across builds ----------------------------------------------------------

/// Byte-identity with **today's** renderer, across builds. Opt-in:
///
/// * `W4_GOLDEN_DIR=<dir> W4_GOLDEN=write` renders every vanilla scene (fog, clouds, sky,
///   vignette) and stores the raw RGBA frames in `<dir>`. Run it on the build *before* a change.
/// * `W4_GOLDEN_DIR=<dir> W4_GOLDEN=check` renders the same scenes with the default (vanilla)
///   inputs and requires every frame to equal its stored one byte for byte.
///
/// Without both variables the test only renders (a smoke test). It uses nothing but the
/// renderer's pre-W4 API, so the same file runs on both builds.
#[test]
fn w4_golden_vanilla_frames() {
    let Some(mut gpu) = gpu() else { return };
    let dir = std::env::var_os("W4_GOLDEN_DIR").map(PathBuf::from);
    let mode = std::env::var("W4_GOLDEN").unwrap_or_default();
    let mut frames: Vec<(String, Vec<u8>)> = Vec::new();

    frames.push(("fog-none".into(), render_unfogged_ramp(&mut gpu)));
    for (name, configure) in vanilla_fog_scenes() {
        let (frame, enabled) = render_fog_ramp(&mut gpu, |f| configure(f));
        assert!(enabled, "{name}: vanilla fog phase is enabled");
        frames.push((name.into(), frame));
    }
    frames.push(("sky-noon-rd8".into(), render_sky(&mut gpu, |_| {})));
    match resources() {
        Some(rm) => {
            frames.push((
                "clouds-fogged-rd8".into(),
                render_clouds(&mut gpu, &rm, Some(cloud_fog(inputs(8)))),
            ));
            frames.push(("clouds-unfogged".into(), render_clouds(&mut gpu, &rm, None)));
            for (name, configure) in vanilla_overlay_scenes() {
                let (frame, enabled) = render_overlay(&mut gpu, &rm, |p| configure(p));
                assert!(enabled, "{name}: the vignette draws");
                frames.push((name.into(), frame));
            }
        }
        None => eprintln!("SKIP: 1.8.9.jar not found; golden frames without clouds/vignette"),
    }

    let Some(dir) = dir else {
        eprintln!(
            "golden: W4_GOLDEN_DIR unset, rendered {} frames only",
            frames.len()
        );
        return;
    };
    match mode.as_str() {
        "write" => {
            std::fs::create_dir_all(&dir).expect("golden dir");
            for (name, frame) in &frames {
                std::fs::write(dir.join(format!("{name}.rgba")), frame).expect("write golden");
            }
            eprintln!("golden: wrote {} frames to {}", frames.len(), dir.display());
        }
        "check" => {
            let mut failures = Vec::new();
            for (name, frame) in &frames {
                let stored = std::fs::read(dir.join(format!("{name}.rgba")))
                    .unwrap_or_else(|e| panic!("{name}: no golden frame ({e})"));
                let diff = differing(&stored, frame);
                eprintln!(
                    "golden: {name}: {diff} of {} pixels differ",
                    frame.len() / 4
                );
                if diff != 0 {
                    failures.push(format!("{name}: {diff} pixels"));
                }
            }
            assert!(failures.is_empty(), "frames changed: {failures:?}");
            eprintln!("golden: all {} frames byte-identical", frames.len());
        }
        other => eprintln!("golden: W4_GOLDEN={other:?} (want write|check), rendered only"),
    }
}

/// The fog scenes really fog: each vanilla scene differs from the unfogged ramp. A negative
/// control for every "off equals unfogged" assertion below.
#[test]
fn every_vanilla_fog_scene_visibly_fogs_the_ramp() {
    let Some(mut gpu) = gpu() else { return };
    let none = render_unfogged_ramp(&mut gpu);
    for (name, configure) in vanilla_fog_scenes() {
        let (frame, _) = render_fog_ramp(&mut gpu, |f| configure(f));
        let diff = differing(&none, &frame);
        assert!(diff > 0, "{name}: the vanilla fog changed nothing");
    }
    // The ramp's near columns are clear in air (fog starts at 96 blocks at 8 chunks)...
    let (air, _) = render_fog_ramp(&mut gpu, |_| {});
    assert_eq!(at(&air, 0, 32), at(&none, 0, 32));
    // ...and its far columns are fully fog-coloured.
    assert_ne!(at(&air, 63, 32), at(&none, 63, 32));
}

// ---- the mods off (W4) --------------------------------------------------------------------

/// With `W4_DUMP_DIR` set, store `frame` there as `<name>.rgba` (64x64 RGBA8) for a look.
fn dump(name: &str, frame: &[u8]) {
    if let Some(dir) = std::env::var_os("W4_DUMP_DIR").map(PathBuf::from) {
        std::fs::create_dir_all(&dir).expect("dump dir");
        std::fs::write(dir.join(format!("{name}.rgba")), frame).expect("dump");
    }
}

/// The vanilla scenes whose fog is `setupFog`'s air branch: what Fog off removes.
const CLEARED_BY_FOG_OFF: [&str; 6] = [
    "fog-air-rd8",
    "fog-air-rd8-planar",
    "fog-air-rd16",
    "fog-air-rd8-rain-dusk",
    "fog-nether-rd8",
    "fog-end-rd8",
];

/// Fog off: every air scene (overworld at two render distances and both distance metrics, rain
/// at dusk, the Nether, the End) is the unfogged frame byte for byte and the fog pass does not
/// run; every other scene (water, lava, in-cloud, blindness in air and in water) is its vanilla
/// frame byte for byte.
#[test]
fn fog_off_removes_exactly_the_render_distance_fog() {
    let Some(mut gpu) = gpu() else { return };
    let none = render_unfogged_ramp(&mut gpu);
    dump("fog-none", &none);
    for (name, configure) in vanilla_fog_scenes() {
        let (on, on_enabled) = render_fog_ramp(&mut gpu, |f| configure(f));
        let (off, off_enabled) = render_fog_ramp(&mut gpu, |f| {
            configure(f);
            f.inputs.distance_fog = false;
        });
        dump(&format!("{name}-on"), &on);
        dump(&format!("{name}-off"), &off);
        assert!(on_enabled, "{name}");
        if CLEARED_BY_FOG_OFF.contains(&name) {
            assert!(!off_enabled, "{name}: the fog pass is skipped");
            assert_eq!(
                differing(&none, &off),
                0,
                "{name}: Fog off is the unfogged frame"
            );
            assert_ne!(differing(&none, &on), 0, "{name}: negative control");
        } else {
            assert!(off_enabled, "{name}: this fog stays");
            assert_eq!(
                differing(&on, &off),
                0,
                "{name}: Fog off keeps it byte for byte"
            );
        }
    }
}

/// A phase forced to record even when it would skip itself.
struct ForceEnabled<T>(Rc<RefCell<T>>);

impl<T: RenderPhase> RenderPhase for ForceEnabled<T> {
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
}

/// Belt and braces: even if the screen-space pass ran with [`FogMode::CLEAR`] (it skips
/// itself), `fog.wgsl`'s own maths leaves every pixel untouched.
#[test]
fn the_clear_mode_is_an_identity_in_the_fog_shader_itself() {
    let Some(mut gpu) = gpu() else { return };
    let none = render_unfogged_ramp(&mut gpu);
    let mut fog = FogPhase::new(&gpu);
    fog.environment = noon(Dimension::Overworld);
    fog.inputs = FogInputs {
        distance_fog: false,
        ..inputs(8)
    };
    fog.set_distance(FogDistance::Radial);
    assert!(fog.state().mode.is_clear());
    let phases = vec![
        (PhaseOrder::TERRAIN_SOLID, boxed(DepthRamp::new(&gpu))),
        (
            FogPhase::ORDER,
            boxed(ForceEnabled(Rc::new(RefCell::new(fog)))),
        ),
    ];
    let forced = render(&mut gpu, &ramp_camera(), BLACK, phases);
    assert_eq!(differing(&none, &forced), 0);
}

/// "Clear water and lava": with Fog off there is no fog at all in either liquid; with Fog on
/// the liquid gets the air falloff in its own colour: the near ramp is clear (vanilla water and
/// lava fog it), the far ramp is the liquid's fog colour, as in vanilla.
#[test]
fn clear_water_and_lava_gives_the_liquids_the_air_fog_in_their_own_colour() {
    let Some(mut gpu) = gpu() else { return };
    let none = render_unfogged_ramp(&mut gpu);
    let (air, _) = render_fog_ramp(&mut gpu, |_| {});
    for (name, medium) in [("water", CameraMedium::Water), ("lava", CameraMedium::Lava)] {
        let (vanilla, _) = render_fog_ramp(&mut gpu, |f| f.inputs.medium = medium);
        let (off_clear, off_enabled) = render_fog_ramp(&mut gpu, |f| {
            f.inputs.medium = medium;
            f.inputs.distance_fog = false;
            f.inputs.clear_liquids = true;
        });
        let (on_clear, on_enabled) = render_fog_ramp(&mut gpu, |f| {
            f.inputs.medium = medium;
            f.inputs.clear_liquids = true;
        });
        dump(&format!("fog-{name}-clear-on"), &on_clear);
        dump(&format!("fog-{name}-clear-off"), &off_clear);
        assert!(!off_enabled, "{name}");
        assert_eq!(differing(&none, &off_clear), 0, "{name}: no fog at all");
        assert!(on_enabled, "{name}");
        // Near: under the 96-block fog start even by the radial metric.
        for x in 0..=20 {
            assert_eq!(at(&on_clear, x, 32), at(&none, x, 32), "{name} x={x}");
        }
        // Where vanilla's thick fog had already covered the ramp, this is clear.
        assert_ne!(
            at(&vanilla, 20, 32),
            at(&none, 20, 32),
            "{name}: negative control"
        );
        // Far: past the 128-block end, the liquid's own fog colour (vanilla's exponential fog
        // is complete there too; allow one unit of rounding between the two curves).
        for x in 40..SIZE {
            let (got, want) = (at(&on_clear, x, 32), at(&vanilla, x, 32));
            for c in 0..3 {
                assert!(
                    got[c].abs_diff(want[c]) <= 1,
                    "{name} x={x}: {got:?} vs {want:?}"
                );
            }
        }
        // The same falloff as air, in another colour.
        assert_ne!(
            at(&on_clear, 63, 32),
            at(&air, 63, 32),
            "{name}: the liquid colour stays"
        );
    }
}

/// Fog off unfogs the clouds exactly: the frame equals clouds drawn with no fog state at all,
/// where vanilla fades the far clouds. Blindness keeps fogging them.
#[test]
fn fog_off_unfogs_the_clouds_exactly_and_blindness_still_fogs_them() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let Some(mut gpu) = gpu() else { return };
    let vanilla = render_clouds(&mut gpu, &rm, Some(cloud_fog(inputs(8))));
    let unfogged = render_clouds(&mut gpu, &rm, None);
    let off_state = cloud_fog(FogInputs {
        distance_fog: false,
        ..inputs(8)
    });
    assert!(off_state.mode.is_clear());
    let off = render_clouds(&mut gpu, &rm, Some(off_state));
    dump("clouds-on", &vanilla);
    dump("clouds-off", &off);
    assert!(
        unfogged.chunks(4).filter(|p| p[..3] != [0, 0, 0]).count() > 400,
        "the scene shows clouds"
    );
    assert_ne!(
        differing(&unfogged, &vanilla),
        0,
        "negative control: vanilla fogs far clouds"
    );
    assert_eq!(differing(&unfogged, &off), 0, "Fog off: no cloud fog");

    let blind = FogInputs {
        blindness: 1.0,
        blindness_duration_ticks: Some(100),
        ..inputs(8)
    };
    let blind_on = render_clouds(&mut gpu, &rm, Some(cloud_fog(blind)));
    let blind_off = render_clouds(
        &mut gpu,
        &rm,
        Some(cloud_fog(FogInputs {
            distance_fog: false,
            ..blind
        })),
    );
    assert_ne!(
        differing(&unfogged, &blind_on),
        0,
        "blindness fogs the clouds"
    );
    assert_eq!(
        differing(&blind_on, &blind_off),
        0,
        "and keeps doing so with Fog off"
    );
}

/// The sky keeps its horizon fade with Fog off, byte for byte (see `sky_pass_fog_mode`), so the
/// sky and the horizon look exactly as they do today. From clear water it fades like air.
#[test]
fn the_sky_and_horizon_are_unchanged_with_fog_off() {
    let Some(mut gpu) = gpu() else { return };
    let vanilla = render_sky(&mut gpu, |_| {});
    let off = render_sky(&mut gpu, |i| i.distance_fog = false);
    dump("sky-on", &vanilla);
    dump("sky-off", &off);
    assert_eq!(differing(&vanilla, &off), 0);
    // The scene has a real gradient: the top and the horizon differ.
    assert_ne!(at(&vanilla, 32, 0), at(&vanilla, 32, 31));

    let water = render_sky(&mut gpu, |i| i.medium = CameraMedium::Water);
    let water_clear = render_sky(&mut gpu, |i| {
        i.medium = CameraMedium::Water;
        i.clear_liquids = true;
    });
    let water_clear_off = render_sky(&mut gpu, |i| {
        i.medium = CameraMedium::Water;
        i.clear_liquids = true;
        i.distance_fog = false;
    });
    assert_ne!(
        differing(&water, &water_clear),
        0,
        "clear water lifts the sky's water fog"
    );
    assert_eq!(
        differing(&water_clear, &water_clear_off),
        0,
        "the switch never reaches the sky"
    );
}

/// Vignette off: the ordinary darkening is gone (the frame is the untouched background and the
/// overlay pass does not even run), and the world-border warning is today's frame byte for byte,
/// with Fancy's darkening underneath or not.
#[test]
fn vignette_off_removes_only_the_darkening_and_keeps_the_border_warning() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let Some(mut gpu) = gpu() else { return };
    let (background, idle) = render_overlay(&mut gpu, &rm, |_| {});
    assert!(!idle, "nothing to draw");
    let first: &[u8] = &background[..4];
    assert!(background.chunks(4).all(|p| p == first));
    for (name, configure) in vanilla_overlay_scenes() {
        let (on, _) = render_overlay(&mut gpu, &rm, |p| configure(p));
        let (off, off_enabled) = render_overlay(&mut gpu, &rm, |p| {
            configure(p);
            p.ordinary_vignette = false;
        });
        dump(&format!("{name}-on"), &on);
        dump(&format!("{name}-off"), &off);
        if name.ends_with("-border") {
            assert!(off_enabled, "{name}");
            assert_eq!(differing(&on, &off), 0, "{name}: the warning is untouched");
            // The warning keeps red and suppresses green and blue at the edges.
            let corner = at(&off, 0, 0);
            assert!(
                corner[0] > corner[1] && corner[0] > corner[2],
                "{name}: {corner:?}"
            );
        } else {
            assert!(!off_enabled, "{name}: no pass at all");
            assert_eq!(differing(&background, &off), 0, "{name}: no darkening");
            assert_ne!(differing(&background, &on), 0, "{name}: negative control");
        }
    }
}

/// While the darkening is hidden its brightness keeps easing exactly as vanilla's does, so
/// switching Vignette back on shows vanilla's frame at once, with no fade from a stale value.
#[test]
fn the_hidden_vignette_keeps_easing_so_turning_it_back_on_is_exact() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: 1.8.9.jar not found");
        return;
    };
    let Some(mut gpu) = gpu() else { return };
    let (vanilla, _) = render_overlay(&mut gpu, &rm, half_vignette);
    let (resumed, _) = render_overlay(&mut gpu, &rm, |p| {
        p.ordinary_vignette = false;
        half_vignette(p);
        p.ordinary_vignette = true;
    });
    assert_eq!(differing(&vanilla, &resumed), 0);
    let mut shown = OverlayPhase::new(&gpu);
    let mut hidden = OverlayPhase::new(&gpu);
    hidden.ordinary_vignette = false;
    for brightness in [0.1, 0.9, 0.4, 0.0, 1.0] {
        for _ in 0..37 {
            shown.set_vignette_brightness(Some(brightness));
            hidden.set_vignette_brightness(Some(brightness));
            assert_eq!(
                shown.prev_vignette_brightness.to_bits(),
                hidden.prev_vignette_brightness.to_bits()
            );
        }
    }
}

// ---- a composite scene, to see that Fog off looks sensible -------------------------------

/// The composite scene's frame size: 16:9 like the user's display.
const SCENE_W: u32 = 256;
const SCENE_H: u32 = 144;

/// A flat checkered ground `drop` blocks below the eye that ends `radius` blocks out, like the
/// world ending at the render distance. Ray-cast per pixel; writes depth.
struct GroundDisc {
    pipeline: wgpu::RenderPipeline,
    buffer: wgpu::Buffer,
    group: wgpu::BindGroup,
    drop: f32,
    radius: f32,
}

const GROUND_WGSL: &str = r#"
struct G {
    inv_vp: mat4x4<f32>,
    vp: mat4x4<f32>,
    // x = plane y relative to the eye (negative), y = radius, zw = framebuffer size.
    params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> g: G;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var c = array<vec2<f32>, 3>(vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0));
    return vec4<f32>(c[i], 0.5, 1.0);
}

struct Out {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> Out {
    let ndc = vec2<f32>(pos.x / g.params.z * 2.0 - 1.0, 1.0 - pos.y / g.params.w * 2.0);
    let far = g.inv_vp * vec4<f32>(ndc, 1.0, 1.0);
    let dir = normalize(far.xyz / far.w);
    if (dir.y > -0.0001) {
        discard;
    }
    let hit = dir * (g.params.x / dir.y);
    if (length(hit.xz) > g.params.y) {
        discard;
    }
    let clip = g.vp * vec4<f32>(hit, 1.0);
    var out: Out;
    out.depth = clip.z / clip.w;
    let cell = (i32(floor(hit.x / 4.0)) + i32(floor(hit.z / 4.0))) & 1;
    out.color = select(vec4<f32>(0.33, 0.55, 0.22, 1.0), vec4<f32>(0.45, 0.68, 0.30, 1.0), cell == 1);
    return out;
}
"#;

impl GroundDisc {
    fn new(gpu: &Gpu, drop: f32, radius: f32) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("w4 ground"),
                source: wgpu::ShaderSource::Wgsl(GROUND_WGSL.into()),
            });
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("w4 ground"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: mc_render::DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
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
                    compilation_options: Default::default(),
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
            label: None,
            size: 144,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self {
            pipeline,
            buffer,
            group,
            drop,
            radius,
        }
    }
}

impl RenderPhase for GroundDisc {
    fn name(&self) -> &'static str {
        "test/w4-ground"
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        let vp = cam.view_proj();
        let mut uniform = [0.0_f32; 36];
        uniform[..16].copy_from_slice(&vp.inverse().to_cols_array());
        uniform[16..32].copy_from_slice(&vp.to_cols_array());
        uniform[32..].copy_from_slice(&[
            -self.drop,
            self.radius,
            gpu.config.width as f32,
            gpu.config.height as f32,
        ]);
        gpu.queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(&uniform));
        let mut pass = gpu.begin_pass(
            enc,
            view,
            "test/w4-ground",
            mc_render::DepthAttachment::Shared,
        );
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Sky, clouds, a ground that ends at the 8-chunk render distance, and the fog pass, the way
/// the client stacks them, at noon from 16 blocks up looking slightly down, with the given fog
/// inputs. Clouds need the jar; without it the scene has none.
fn render_composite(gpu: &mut Gpu, rm: Option<&ResourceManager>, fog_inputs: FogInputs) -> Vec<u8> {
    let env = noon(Dimension::Overworld);
    let mut cam = Camera {
        pos: DVec3::new(0.0, 80.0, 0.0),
        pitch: 6.0,
        yaw: 30.0,
        ..Camera::default()
    };
    cam.set_viewport(SCENE_W, SCENE_H);
    cam.set_view_distance_chunks(8);
    cam.update();

    let state = FogState::compute(&env, fog_inputs, 0.0);
    let mut sky = SkyPhase::new(gpu);
    sky.environment = env;
    sky.fog_inputs = fog_inputs;
    let mut fog = FogPhase::new(gpu);
    fog.environment = env;
    fog.inputs = fog_inputs;
    fog.set_distance(FogDistance::Radial);
    let mut phases = vec![(PhaseOrder::SKY, boxed(sky))];
    if let Some(rm) = rm {
        let mut clouds = CloudsPhase::new(gpu);
        clouds.load_texture(gpu, rm);
        clouds.mode = CloudMode::Fancy;
        clouds.environment = env;
        clouds.set_fog(Some((state, FogDistance::Radial)));
        phases.push((PhaseOrder::SKY, boxed(clouds)));
    }
    phases.push((
        PhaseOrder::TERRAIN_SOLID,
        boxed(GroundDisc::new(gpu, 16.0, 128.0)),
    ));
    phases.push((FogPhase::ORDER, boxed(fog)));
    // The client clears to the world fog colour.
    let clear = wgpu::Color {
        r: f64::from(state.color.x),
        g: f64::from(state.color.y),
        b: f64::from(state.color.z),
        a: 1.0,
    };
    render(gpu, &cam, clear, phases)
}

/// For a look (`W4_DUMP_DIR`): the client's stack with Fog on and off. Off, the ground stays
/// sharp to its edge against the unchanged sky; the near ground and the sky above the horizon
/// band are the same pixels either way.
#[test]
fn composite_scene_fog_off_keeps_the_sky_and_sharpens_the_world() {
    let Some(mut gpu) = gpu_sized(SCENE_W, SCENE_H) else {
        return;
    };
    let rm = resources();
    let on = render_composite(&mut gpu, rm.as_ref(), inputs(8));
    let off = render_composite(
        &mut gpu,
        rm.as_ref(),
        FogInputs {
            distance_fog: false,
            ..inputs(8)
        },
    );
    dump(&format!("composite__{SCENE_W}x{SCENE_H}-on"), &on);
    dump(&format!("composite__{SCENE_W}x{SCENE_H}-off"), &off);
    let px = |frame: &[u8], x: u32, y: u32| {
        let i = ((y * SCENE_W + x) * 4) as usize;
        [frame[i], frame[i + 1], frame[i + 2]]
    };
    // The bottom rows are near ground (well inside the 96-block fog start): identical.
    for y in SCENE_H - 8..SCENE_H {
        for x in 0..SCENE_W {
            assert_eq!(px(&on, x, y), px(&off, x, y), "near ground at {x},{y}");
        }
    }
    // The top row is sky (or near cloud): identical when no cloud covers it, and the sky pass
    // is unchanged, so count how much of it differs: only clouds may.
    let top_diff = (0..SCENE_W)
        .filter(|&x| px(&on, x, 0) != px(&off, x, 0))
        .count();
    eprintln!("top row: {top_diff} of {SCENE_W} pixels differ (clouds only)");
    // The far ground did change.
    assert!(differing(&on, &off) > 0);
}

/// GPU cost of the screen-space fog pass at the user's 1920x1080, over a ground that ends at
/// the 8-chunk render distance: what Fog off saves, since the pass is then skipped. Run with
/// `MC_GPU_TIMING=1 cargo test -p mc-render --test w4_fog_vignette -- --ignored --nocapture`.
/// Timings taken while other builds run are provisional.
#[test]
#[ignore = "benchmark; needs MC_GPU_TIMING=1"]
fn fog_pass_gpu_cost_at_1080p() {
    const FRAMES: u32 = 600;
    let (width, height) = (1920, 1080);
    let Some(mut gpu) = gpu_sized(width, height) else {
        return;
    };
    if !gpu.supports_gpu_timing() {
        eprintln!("SKIP: set MC_GPU_TIMING=1 (and the adapter needs timestamp queries)");
        return;
    }
    let mut cam = Camera {
        pitch: 6.0,
        yaw: 30.0,
        ..Camera::default()
    };
    cam.set_viewport(width, height);
    cam.set_view_distance_chunks(8);
    cam.update();
    eprintln!("{width}x{height}, {FRAMES} frames per row, ground to 128 blocks, mean per frame:");
    for distance_fog in [true, false] {
        let mut fog = FogPhase::new(&gpu);
        fog.environment = noon(Dimension::Overworld);
        fog.inputs = FogInputs {
            distance_fog,
            ..inputs(8)
        };
        fog.set_distance(FogDistance::Radial);
        let mut list = PhaseList::new();
        list.clear_color = BLACK;
        list.insert(
            PhaseOrder::TERRAIN_SOLID,
            boxed(GroundDisc::new(&gpu, 16.0, 128.0)),
        );
        list.insert(FogPhase::ORDER, boxed(fog));
        for i in 0..FRAMES {
            let frame = gpu.acquire().expect("acquire");
            list.render(&gpu, frame.view(), &cam, 0.0);
            drop(frame);
            if i % 50 == 49 {
                let _ = gpu.read_pixels();
            }
        }
        let timings = list.gpu_timings();
        let fog_ms = timings
            .scopes
            .iter()
            .find(|(name, _)| *name == "sky/fog")
            .map_or(0.0, |(_, stats)| stats.mean_ns() / 1e6);
        eprintln!(
            "Fog {}: sky/fog {fog_ms:.4} ms, gpu.frame {:.4} ms, {} timed frames",
            if distance_fog { "on " } else { "off" },
            timings.frame_ns.mean_ns() / 1e6,
            timings.frames,
        );
        assert!(timings.frames > 0, "no frame was timed");
    }
}
