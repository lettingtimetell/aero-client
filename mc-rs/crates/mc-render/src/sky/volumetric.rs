//! Volumetric clouds for the Aero title screen and, as the `volumetric_clouds` mod, in game.
//!
//! # What is here
//!
//! * [`VolumetricClouds`]: the renderer core. It raymarches a layer of clouds into a small target
//!   (a quarter or half of the frame in each direction) and accumulates it over frames. Anything
//!   that can describe a camera can use it: the in-game phase below, and the title scene
//!   (`crate::menu_scene`), which samples the result in its own sky shader.
//! * [`VolumetricCloudsPhase`]: the in-game phase pair. It traces, then lays the clouds over the
//!   frame at full resolution, depth-tested against the world.
//! * [`in_game_look`]: the clouds' light from vanilla's time of day and weather.
//!
//! # Technique (`volumetric.wgsl`, `volumetric_noise.wgsl`)
//!
//! A raymarched cloud layer: a 128³ Perlin-Worley + Worley-fBm shape texture eroded by a 32³
//! Worley-fBm detail texture, a 256² weather map (coverage, cloud type), a density profile per
//! cloud type through the layer, Beer's law with the powder term, a dual-lobe Henyey-Greenstein
//! phase summed over multiple-scattering octaves, a height-graded sky ambient, a short march
//! toward the light, and a jittered low-resolution trace accumulated over frames by
//! reprojection. All three noise textures are generated on the GPU by a compute pass the first
//! time they are needed.
//!
//! # Cost
//!
//! * **Nothing exists until something asks.** [`VolumetricClouds::new`] creates no GPU object and
//!   starts no thread; an installed but unused [`VolumetricCloudsPhase`] is two disabled phases
//!   whose `enabled` reads one `Cell<u8>`.
//! * [`VolumetricClouds::request`] builds the shaders and pipelines on a worker thread (wgpu's
//!   `Device` is `Send + Sync`), so neither the title screen nor the game ever waits for them.
//!   Until they are ready the caller draws what it drew before (vanilla's clouds, the title's old
//!   painted clouds). The noise is then generated on the GPU in the next frame (about a
//!   millisecond once) and the clouds appear.
//! * Per frame: one low-resolution trace pass, two small texture copies and, in game, one
//!   full-resolution composite pass that discards every pixel without cloud.
//! * [`VolumetricClouds::release`] gives everything back (in game: when the mod is turned off).
//!
//! # Plugging into a future shader pack
//!
//! The trace writes two named, stable textures ([`VolumetricClouds::out_views`]): premultiplied
//! in-scattered light with transmittance (`Rgba16Float`), and the cloud's distance along each ray
//! in blocks (`R16Float`), both at the trace resolution and valid after the trace for the rest of
//! the frame. A pack can composite them its own way (its own tone map, god rays that read the
//! transmittance, reflections), replace the composite, or call [`VolumetricClouds::trace`] with
//! another camera (a reflection probe) through another instance. Inputs are explicit:
//! [`CloudFrame`] (camera-relative view-projection, eye, optional scene depth),
//! [`CloudLayer`] (the layer's shape) and [`CloudLook`] (light and weather). In game the clouds
//! record at [`EARLY_ORDER`] (the eye is below the clouds: before translucent terrain, particles
//! and weather, as vanilla draws them before terrain) or [`LATE_ORDER`] (the eye is in or above
//! them: after translucent terrain and the screen-space fog, as vanilla draws them after
//! translucent terrain).

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::mpsc;
use std::time::Duration;

use glam::{DVec2, DVec3, Mat4, Vec3};

use crate::pipeline::{PhaseList, PhaseOrder, RenderPhase};
use crate::sky::environment::Environment;
use crate::sky::fog::FogPhase;
use crate::{Camera, DepthAttachment, Gpu};

const TRACE_WGSL: &str = include_str!("volumetric.wgsl");
const NOISE_WGSL: &str = include_str!("volumetric_noise.wgsl");

/// Edge of the shape noise texture, in texels (a 128³ volume).
pub const SHAPE_SIZE: u32 = 128;
/// Edge of the detail noise texture, in texels (a 32³ volume).
pub const DETAIL_SIZE: u32 = 32;
/// Edge of the weather map, in texels.
pub const WEATHER_SIZE: u32 = 256;

/// Premultiplied in-scattered light (rgb) and transmittance (a).
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The cloud's distance along the ray, in blocks.
const DISTANCE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
const NOISE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Where the in-game clouds record while the eye is below the cloud base: after the opaque world
/// (so they can be depth-tested against it) and before translucent terrain, particles and weather,
/// which then draw over them. java: `EntityRenderer.renderWorldPass` calls `renderCloudsCheck`
/// before terrain when the eye is below the clouds.
pub const EARLY_ORDER: PhaseOrder = PhaseOrder::VOLUMETRIC_CLOUDS;
/// Where they record while the eye is in or above the layer: after translucent terrain (which
/// would otherwise draw over clouds that are in front of it) and after the screen-space fog (which
/// fogs by the terrain's distance, not the cloud's), before waypoint labels and the hand. java:
/// `renderWorldPass` draws clouds after the translucent layer when the eye is above them.
pub const LATE_ORDER: PhaseOrder = PhaseOrder(FogPhase::ORDER.0 + 2);
/// [`RenderPhase::name`] of the phase at [`EARLY_ORDER`].
pub const EARLY_PHASE_NAME: &str = "sky/volumetric-clouds";
/// [`RenderPhase::name`] of the phase at [`LATE_ORDER`].
pub const LATE_PHASE_NAME: &str = "sky/volumetric-clouds-late";

/// How far the eye may jump between two frames before the accumulated clouds are thrown away
/// (a teleport, a respawn), in blocks.
const HISTORY_JUMP: f64 = 64.0;

/// How much of each new in-game trace goes into the accumulated clouds: about ten frames of
/// memory. Camera turns are reprojected exactly (clouds are far away), so this only trades a
/// little lag on wind-driven change (slow) for less shimmer.
const IN_GAME_HISTORY_WEIGHT: f32 = 0.1;

/// The quality presets the mod offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CloudQuality {
    /// A quarter of the frame in each direction, 32 steps, 3 light steps.
    Low,
    /// A quarter of the frame in each direction, 48 steps, 5 light steps.
    #[default]
    Medium,
    /// Half the frame in each direction, 64 steps, 6 light steps.
    High,
}

impl CloudQuality {
    /// The menu labels, in [`CloudQuality::from_index`] order.
    pub const LABELS: [&'static str; 3] = ["Low", "Medium", "High"];

    /// From the mod's `quality` choice index (anything unknown is [`CloudQuality::Medium`]).
    #[must_use]
    pub fn from_index(index: usize) -> Self {
        match index {
            0 => CloudQuality::Low,
            2 => CloudQuality::High,
            _ => CloudQuality::Medium,
        }
    }

    /// Frame pixels per trace pixel, in each direction.
    #[must_use]
    pub fn divisor(self) -> u32 {
        match self {
            CloudQuality::Low | CloudQuality::Medium => 4,
            CloudQuality::High => 2,
        }
    }

    /// Raymarch steps through the layer.
    #[must_use]
    pub fn view_steps(self) -> u32 {
        match self {
            CloudQuality::Low => 32,
            CloudQuality::Medium => 48,
            CloudQuality::High => 64,
        }
    }

    /// Steps toward the light from each lit sample.
    #[must_use]
    pub fn light_steps(self) -> u32 {
        match self {
            CloudQuality::Low => 3,
            CloudQuality::Medium => 5,
            CloudQuality::High => 6,
        }
    }
}

/// The shape of the cloud layer, in world units (blocks).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudLayer {
    /// World height of the cloud bases.
    pub bottom: f32,
    /// World height of the tallest tops.
    pub top: f32,
    /// How far along a ray clouds are traced at all.
    pub max_distance: f32,
    /// Where clouds start to dissolve into the sky behind them (they are gone at
    /// [`CloudLayer::max_distance`]).
    pub fade_start: f32,
    /// The weather map (coverage, cloud type) repeats every this many blocks.
    pub weather_tile: f64,
    /// The shape noise repeats every this many blocks.
    pub shape_tile: f64,
    /// The detail noise repeats every this many blocks.
    pub detail_tile: f64,
    /// Extinction per block at density 1.
    pub extinction: f32,
    /// How deeply the detail noise erodes the edges, `0..=1`.
    pub erosion: f32,
    /// Beyond this distance (blocks) the detail erosion is gone; it fades in from 40 % of it.
    pub detail_range: f32,
    /// How far the tops lean downwind of the bases, in blocks.
    pub lean: f32,
    /// The first step of the march toward the light, in blocks (later steps grow).
    pub light_step: f32,
    /// How far the layer sits above the height its noise belongs to, in blocks: the 3D noise is
    /// read this much lower, so a layer moved up or down carries the same clouds with it instead
    /// of rolling new shapes through them. `0` for a layer at its own height (the in-game "Above you" mode).
    pub noise_shift: f32,
}

impl CloudLayer {
    /// The in-game layer: bases at the vanilla cloud height (`WorldProvider.getCloudHeight`, 128
    /// in the overworld), tops 120 blocks higher.
    #[must_use]
    pub fn in_game(cloud_height: f32) -> Self {
        CloudLayer {
            bottom: cloud_height,
            top: cloud_height + 120.0,
            max_distance: 3600.0,
            fade_start: 900.0,
            weather_tile: 3072.0,
            shape_tile: 512.0,
            detail_tile: 56.0,
            extinction: 0.5,
            erosion: 0.45,
            detail_range: 1400.0,
            lean: 18.0,
            light_step: 4.0,
            noise_shift: 0.0,
        }
    }

    /// The in-game layer with its bases at `bottom` instead of the vanilla `cloud_height` (the
    /// "Above you" mode), showing the clouds the vanilla-height layer would show, moved with it.
    #[must_use]
    pub fn in_game_at(bottom: f32, cloud_height: f32) -> Self {
        CloudLayer {
            bottom,
            top: bottom + 120.0,
            noise_shift: bottom - cloud_height,
            ..CloudLayer::in_game(cloud_height)
        }
    }

    /// The title screen's layer: far above the greenhouse, seen through its roof.
    #[must_use]
    pub fn title() -> Self {
        CloudLayer {
            bottom: 380.0,
            top: 560.0,
            max_distance: 7000.0,
            fade_start: 2200.0,
            weather_tile: 6144.0,
            shape_tile: 1536.0,
            detail_tile: 192.0,
            extinction: 0.3,
            erosion: 0.4,
            detail_range: 3500.0,
            lean: 36.0,
            light_step: 8.0,
            noise_shift: 0.0,
        }
    }
}

/// Light and weather for one frame of clouds. Colours are linear radiance in the caller's own
/// scale (the title scene works in HDR, the game in display range).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudLook {
    /// How much of the sky is cloudy, `0..=1`: 0 is clear, 0.5 scattered, 1 overcast.
    pub coverage: f32,
    /// Rain, `0..=1` (informational; [`in_game_look`] has already folded it into the rest).
    pub rain: f32,
    /// Added to the weather map's cloud type: negative favours flat stratus, positive cumulus.
    pub type_bias: f32,
    /// Unit vector toward the light (the sun, or the moon at night).
    pub sun_dir: Vec3,
    /// The light's radiance.
    pub sun_color: Vec3,
    /// Sky light reaching the tops of the clouds.
    pub ambient_top: Vec3,
    /// Light reaching their bases.
    pub ambient_bottom: Vec3,
    /// Scales both ambient terms.
    pub ambient: f32,
    /// Strength of the powder term (dark edges with the light behind the viewer), `0..=1`.
    pub powder: f32,
    /// Multiplies the light before the in-game composite's tone curve.
    pub exposure: f32,
    /// How far the wind has carried the clouds, in blocks (x, z). Vanilla's clouds drift toward
    /// -X, which is a growing +X here.
    pub wind: [f64; 2],
    /// Seconds of slow vertical drift through the noise, which makes the clouds change shape as
    /// they move.
    pub drift: f64,
}

impl Default for CloudLook {
    /// A clear midday: the sun high in the south-east, white light, blue sky ambient.
    fn default() -> Self {
        CloudLook {
            coverage: 0.5,
            rain: 0.0,
            type_bias: 0.0,
            sun_dir: Vec3::new(0.35, 0.85, 0.25).normalize(),
            sun_color: Vec3::new(1.35, 1.31, 1.24),
            ambient_top: Vec3::new(0.30, 0.42, 0.62),
            ambient_bottom: Vec3::new(0.12, 0.15, 0.20),
            ambient: 1.0,
            powder: 0.8,
            exposure: 1.0,
            wind: [0.0; 2],
            drift: 0.0,
        }
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The in-game clouds' light from vanilla's sky state, so they follow the day, the night and the
/// weather the way vanilla's clouds and sky do. `coverage` is the mod's setting (`0..=1`); wind
/// and drift are left at zero for the caller to fill.
///
/// * The light comes from the sun by day and the moon by night, from where vanilla draws them
///   (`RenderGlobal.renderSky`: the sun rises in the east, +X, and sets in the west, -X), golden
///   when low; both fade out just below the horizon, so the hand-over never pops.
/// * The sky ambient is vanilla's own sky colour (`World.getSkyColor`: biome tint, night, rain
///   and thunder greying), linearised, plus the lightning flash.
/// * Rain and thunder dim the light, add coverage and flatten the clouds toward stratus.
#[must_use]
pub fn in_game_look(
    env: &Environment,
    biome_temperature: f32,
    partial_ticks: f32,
    coverage: f32,
) -> CloudLook {
    let angle = env.celestial_angle(partial_ticks) * std::f32::consts::TAU;
    let sun = Vec3::new(-angle.sin(), angle.cos(), 0.0);
    let rain = env.weather.rain_strength.clamp(0.0, 1.0);
    let thunder = env.weather.thunder_strength.clamp(0.0, 1.0);
    let gloom = (1.0 - 0.6 * rain) * (1.0 - 0.45 * thunder);
    let (dir, light) = if sun.y >= -0.06 {
        let day = smoothstep(-0.06, 0.14, sun.y);
        let warm = 1.0 - smoothstep(0.02, 0.5, sun.y);
        let tint = Vec3::new(1.0, 0.97, 0.92).lerp(Vec3::new(1.0, 0.56, 0.30), warm);
        (sun, tint * (2.0 * day))
    } else {
        let night = smoothstep(0.06, 0.3, -sun.y);
        (-sun, Vec3::new(0.55, 0.65, 0.90) * (0.16 * night))
    };
    let day = smoothstep(-0.1, 0.2, sun.y);
    let sky = env.sky_color(biome_temperature, partial_ticks);
    let sky = Vec3::new(sky.x.powf(2.2), sky.y.powf(2.2), sky.z.powf(2.2));
    let flash = env.lightning_flash(partial_ticks).max(0.0);
    CloudLook {
        coverage: (coverage + rain * 0.45).clamp(0.0, 1.0),
        rain,
        type_bias: -0.35 * rain,
        sun_dir: dir.normalize(),
        sun_color: light * gloom,
        ambient_top: sky * 0.5 + Vec3::splat(0.03 * day + flash * 1.2),
        ambient_bottom: sky * 0.22 + Vec3::splat(0.015 * day + flash * 0.8),
        ambient: 1.0,
        powder: 0.8,
        exposure: 1.0,
        wind: [0.0; 2],
        drift: 0.0,
    }
}

/// What the trace is drawn for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloudTarget {
    /// No scene depth; the caller samples [`VolumetricClouds::out_views`] itself (the title
    /// scene's sky shader).
    Offscreen,
    /// In game: traced against the shared depth buffer, composited onto a frame of this format
    /// and MSAA sample count.
    Frame {
        /// The frame's colour format (`Gpu::config.format`).
        format: wgpu::TextureFormat,
        /// `Gpu::sample_count`.
        samples: u32,
    },
}

/// One frame's camera and inputs for [`VolumetricClouds::trace`].
#[derive(Debug, Clone, Copy)]
pub struct CloudFrame<'a> {
    /// Projection times a view **without** the eye's translation (camera-relative, like
    /// [`Camera::view_proj`]).
    pub view_proj: Mat4,
    /// The eye in world space.
    pub eye: DVec3,
    /// The NDC depth of the near plane and of any farther point: `(0.0, 1.0)` for the game's
    /// 0..1 projection, `(1.0, 0.5)` for a reverse-Z infinite one.
    pub ray_z: (f32, f32),
    /// The frame's size in pixels; the trace is a fraction of it.
    pub frame_size: (u32, u32),
    /// The layer.
    pub layer: CloudLayer,
    /// Light and weather.
    pub look: CloudLook,
    /// Quality preset.
    pub quality: CloudQuality,
    /// The scene's depth buffer, to stop at the world (in game), or `None`.
    pub scene_depth: Option<&'a wgpu::Texture>,
    /// How much of each new frame goes into the accumulated result, `0..=1` (smaller is smoother
    /// and slower to follow changes).
    pub history_weight: f32,
}

/// Facts about the trace just recorded, for a caller binding [`VolumetricClouds::out_views`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CloudTraceInfo {
    /// The trace target's size in pixels.
    pub trace_size: (u32, u32),
    /// Frame pixels per trace pixel, in each direction.
    pub divisor: u32,
    /// Bumped whenever the output textures are replaced: rebuild bind groups when it changes.
    pub generation: u64,
}

/// Must match `struct Cloud` in `volumetric.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TraceUniform {
    inv_view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    cam_delta: [f32; 4],
    eye: [f32; 4],
    origin_weather: [f32; 4],
    origin_detail: [f32; 4],
    layer: [f32; 4],
    scale: [f32; 4],
    shape: [f32; 4],
    sun_dir: [f32; 4],
    sun_color: [f32; 4],
    amb_top: [f32; 4],
    amb_bottom: [f32; 4],
    quality: [f32; 4],
    size: [f32; 4],
    jitter: [f32; 4],
    ray_z: [f32; 4],
}

/// Must match `struct Composite` in `volumetric.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeUniform {
    inv_view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    grid: [f32; 4],
}

/// The pipelines and other objects that never depend on the frame: built on the worker thread.
struct Statics {
    target: CloudTarget,
    noise_layout: wgpu::BindGroupLayout,
    noise_shape: wgpu::ComputePipeline,
    noise_detail: wgpu::ComputePipeline,
    noise_weather: wgpu::ComputePipeline,
    trace_layout: wgpu::BindGroupLayout,
    trace_pipeline: wgpu::RenderPipeline,
    noise_sampler: wgpu::Sampler,
    clamp_sampler: wgpu::Sampler,
    trace_uniform: wgpu::Buffer,
    /// A 1x1 depth texture bound in place of the scene's for [`CloudTarget::Offscreen`].
    dummy_depth: Option<wgpu::TextureView>,
    composite: Option<CompositeStatics>,
}

struct CompositeStatics {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
}

fn texture_entry(
    binding: u32,
    dimension: wgpu::TextureViewDimension,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: dimension,
            multisampled: false,
        },
        count: None,
    }
}

fn depth_entry(binding: u32, multisampled: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

/// The trace/composite shader for a scene depth buffer with `samples` samples: the multisampled
/// spelling when there is MSAA (see `Gpu::depth_texture`).
fn trace_source(samples: u32) -> String {
    if samples > 1 {
        TRACE_WGSL.replace("texture_depth_2d", "texture_depth_multisampled_2d")
    } else {
        TRACE_WGSL.to_owned()
    }
}

impl Statics {
    #[allow(clippy::too_many_lines)] // One constructor: layouts, pipelines, samplers.
    fn new(device: &wgpu::Device, target: CloudTarget) -> Self {
        let samples = match target {
            CloudTarget::Offscreen => 1,
            CloudTarget::Frame { samples, .. } => samples,
        };
        let noise_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volumetric clouds noise"),
            source: wgpu::ShaderSource::Wgsl(NOISE_WGSL.into()),
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volumetric clouds"),
            source: wgpu::ShaderSource::Wgsl(trace_source(samples).into()),
        });

        let storage =
            |binding: u32, dimension: wgpu::TextureViewDimension| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: NOISE_FORMAT,
                    view_dimension: dimension,
                },
                count: None,
            };
        let noise_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volumetric clouds noise group0"),
            entries: &[
                storage(0, wgpu::TextureViewDimension::D3),
                storage(1, wgpu::TextureViewDimension::D3),
                storage(2, wgpu::TextureViewDimension::D2),
            ],
        });
        let noise_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("volumetric clouds noise layout"),
                bind_group_layouts: &[Some(&noise_layout)],
                immediate_size: 0,
            });
        let compute = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&noise_pipeline_layout),
                module: &noise_module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let noise_shape = compute("cs_shape");
        let noise_detail = compute("cs_detail");
        let noise_weather = compute("cs_weather");

        let fragment = wgpu::ShaderStages::FRAGMENT;
        let trace_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volumetric clouds trace group0"),
            entries: &[
                uniform_entry(0),
                texture_entry(1, wgpu::TextureViewDimension::D3, fragment),
                texture_entry(2, wgpu::TextureViewDimension::D3, fragment),
                texture_entry(3, wgpu::TextureViewDimension::D2, fragment),
                sampler_entry(4),
                texture_entry(5, wgpu::TextureViewDimension::D2, fragment),
                texture_entry(6, wgpu::TextureViewDimension::D2, fragment),
                sampler_entry(7),
                depth_entry(8, samples > 1),
            ],
        });
        let trace_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("volumetric clouds trace layout"),
                bind_group_layouts: &[Some(&trace_layout)],
                immediate_size: 0,
            });
        let trace_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volumetric clouds trace"),
            layout: Some(&trace_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_fullscreen"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_trace"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: COLOR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: DISTANCE_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });

        let sampler = |label: &str, address: wgpu::AddressMode| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                address_mode_u: address,
                address_mode_v: address,
                address_mode_w: address,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..wgpu::SamplerDescriptor::default()
            })
        };
        let noise_sampler = sampler("volumetric clouds noise", wgpu::AddressMode::Repeat);
        let clamp_sampler = sampler("volumetric clouds clamp", wgpu::AddressMode::ClampToEdge);
        let trace_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volumetric clouds trace uniform"),
            size: std::mem::size_of::<TraceUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dummy_depth = (target == CloudTarget::Offscreen).then(|| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("volumetric clouds no scene depth"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: crate::DEPTH_FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        });

        let composite = match target {
            CloudTarget::Offscreen => None,
            CloudTarget::Frame { format, samples } => {
                let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("volumetric clouds composite group0"),
                    entries: &[
                        uniform_entry(10),
                        texture_entry(11, wgpu::TextureViewDimension::D2, fragment),
                        texture_entry(12, wgpu::TextureViewDimension::D2, fragment),
                        sampler_entry(13),
                        depth_entry(14, samples > 1),
                    ],
                });
                let pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("volumetric clouds composite layout"),
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    });
                // Premultiplied over: frame = cloud light + frame * transmittance. The frame's
                // own alpha is left alone.
                let blend = wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::SrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::Zero,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                };
                let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("volumetric clouds composite"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs_fullscreen"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: samples,
                        ..wgpu::MultisampleState::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs_composite"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: Some(blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                });
                let uniform = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("volumetric clouds composite uniform"),
                    size: std::mem::size_of::<CompositeUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                Some(CompositeStatics {
                    layout,
                    pipeline,
                    uniform,
                })
            }
        };

        Statics {
            target,
            noise_layout,
            noise_shape,
            noise_detail,
            noise_weather,
            trace_layout,
            trace_pipeline,
            noise_sampler,
            clamp_sampler,
            trace_uniform,
            dummy_depth,
            composite,
        }
    }
}

/// The three noise textures; generated on the GPU the frame they are created.
struct Noise {
    shape: wgpu::TextureView,
    detail: wgpu::TextureView,
    weather: wgpu::TextureView,
    /// The storage bindings the generator writes through, until it has run.
    generate: Option<wgpu::BindGroup>,
}

impl Noise {
    fn new(device: &wgpu::Device, statics: &Statics) -> Self {
        let make = |label: &str, size: wgpu::Extent3d, dimension: wgpu::TextureDimension| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension,
                    format: NOISE_FORMAT,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let cube = |edge: u32| wgpu::Extent3d {
            width: edge,
            height: edge,
            depth_or_array_layers: edge,
        };
        let shape = make(
            "volumetric clouds shape noise",
            cube(SHAPE_SIZE),
            wgpu::TextureDimension::D3,
        );
        let detail = make(
            "volumetric clouds detail noise",
            cube(DETAIL_SIZE),
            wgpu::TextureDimension::D3,
        );
        let weather = make(
            "volumetric clouds weather map",
            wgpu::Extent3d {
                width: WEATHER_SIZE,
                height: WEATHER_SIZE,
                depth_or_array_layers: 1,
            },
            wgpu::TextureDimension::D2,
        );
        let generate = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volumetric clouds noise group0"),
            layout: &statics.noise_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shape),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&detail),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&weather),
                },
            ],
        });
        Noise {
            shape,
            detail,
            weather,
            generate: Some(generate),
        }
    }

    /// Record the generator if it has not run yet. The trace recorded after it in the same
    /// encoder sees its output (wgpu orders the two).
    fn generate(&mut self, statics: &Statics, enc: &mut wgpu::CommandEncoder) {
        let Some(group) = self.generate.take() else {
            return;
        };
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("volumetric clouds noise"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &group, &[]);
        pass.set_pipeline(&statics.noise_shape);
        pass.dispatch_workgroups(SHAPE_SIZE / 4, SHAPE_SIZE / 4, SHAPE_SIZE / 4);
        pass.set_pipeline(&statics.noise_detail);
        pass.dispatch_workgroups(DETAIL_SIZE / 4, DETAIL_SIZE / 4, DETAIL_SIZE / 4);
        pass.set_pipeline(&statics.noise_weather);
        pass.dispatch_workgroups(WEATHER_SIZE / 8, WEATHER_SIZE / 8, 1);
    }
}

/// The trace's render targets and the stable outputs they are copied into.
struct Targets {
    frame: (u32, u32),
    trace: (u32, u32),
    divisor: u32,
    /// The scene depth texture the bind group reads, to notice when the `Gpu` replaced it.
    scene_depth: Option<wgpu::Texture>,
    scratch_color: wgpu::Texture,
    scratch_color_view: wgpu::TextureView,
    scratch_distance: wgpu::Texture,
    scratch_distance_view: wgpu::TextureView,
    out_color: wgpu::Texture,
    out_color_view: wgpu::TextureView,
    out_distance: wgpu::Texture,
    out_distance_view: wgpu::TextureView,
    group: wgpu::BindGroup,
}

impl Targets {
    fn new(
        device: &wgpu::Device,
        statics: &Statics,
        noise: &Noise,
        frame: (u32, u32),
        divisor: u32,
        scene_depth: Option<&wgpu::Texture>,
    ) -> Self {
        let trace = (
            frame.0.max(1).div_ceil(divisor),
            frame.1.max(1).div_ceil(divisor),
        );
        let make = |label: &str, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: trace.0,
                    height: trace.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        let scratch = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
        let out = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let (scratch_color, scratch_color_view) =
            make("volumetric clouds trace", COLOR_FORMAT, scratch);
        let (scratch_distance, scratch_distance_view) =
            make("volumetric clouds trace distance", DISTANCE_FORMAT, scratch);
        let (out_color, out_color_view) = make("volumetric clouds", COLOR_FORMAT, out);
        let (out_distance, out_distance_view) =
            make("volumetric clouds distance", DISTANCE_FORMAT, out);
        let scene_view =
            scene_depth.map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));
        let depth_view = scene_view
            .as_ref()
            .or(statics.dummy_depth.as_ref())
            .expect("a scene depth, or the offscreen stand-in");
        let view = wgpu::BindingResource::TextureView;
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volumetric clouds trace group0"),
            layout: &statics.trace_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: statics.trace_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: view(&noise.shape),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: view(&noise.detail),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: view(&noise.weather),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&statics.noise_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: view(&out_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: view(&out_distance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&statics.clamp_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: view(depth_view),
                },
            ],
        });
        Targets {
            frame,
            trace,
            divisor,
            scene_depth: scene_depth.cloned(),
            scratch_color,
            scratch_color_view,
            scratch_distance,
            scratch_distance_view,
            out_color,
            out_color_view,
            out_distance,
            out_distance_view,
            group,
        }
    }

    fn matches(
        &self,
        frame: (u32, u32),
        divisor: u32,
        scene_depth: Option<&wgpu::Texture>,
    ) -> bool {
        self.frame == frame && self.divisor == divisor && self.scene_depth.as_ref() == scene_depth
    }
}

/// The previous trace's camera, for reprojection.
#[derive(Debug, Clone, Copy)]
struct History {
    view_proj: Mat4,
    eye: DVec3,
    trace: (u32, u32),
}

enum Build {
    /// Nothing requested: no GPU object, no thread.
    Idle,
    /// The worker is building the pipelines.
    Building(CloudTarget, mpsc::Receiver<Statics>),
    /// Ready to trace.
    Ready(Box<Statics>),
    /// This device cannot run the clouds (no compute shaders or storage textures, or the GL
    /// backend), or the worker failed: the caller keeps drawing what it drew before.
    Failed,
}

/// Whether this device can run the clouds: compute shaders writing 3D storage textures and two
/// colour targets, on a native backend. The GL backend is left out on purpose: its compute and
/// storage-texture support varies by driver, and a failure there would be a crash, not a fallback.
fn device_supports_clouds(gpu: &Gpu) -> bool {
    let limits = gpu.device.limits();
    let backend_ok = matches!(
        gpu.adapter_info().backend,
        wgpu::Backend::Vulkan | wgpu::Backend::Dx12 | wgpu::Backend::Metal
    );
    backend_ok
        && limits.max_storage_textures_per_shader_stage >= 3
        && limits.max_compute_invocations_per_workgroup >= 64
        && limits.max_compute_workgroup_size_x >= 8
        && limits.max_compute_workgroup_size_y >= 8
        && limits.max_compute_workgroup_size_z >= 4
        && limits.max_texture_dimension_3d >= SHAPE_SIZE
        && limits.max_color_attachments >= 2
}

fn halton(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut r = 0.0;
    while index > 0 {
        f /= base as f32;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

/// The volumetric cloud renderer core. See the [module docs](self).
///
/// Main-thread only, like every phase; it owns at most one worker thread while its pipelines are
/// being built.
pub struct VolumetricClouds {
    build: Build,
    noise: Option<Noise>,
    targets: Option<Targets>,
    history: Option<History>,
    frame: u32,
    generation: u64,
}

impl Default for VolumetricClouds {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for VolumetricClouds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let build = match &self.build {
            Build::Idle => "idle",
            Build::Building(..) => "building",
            Build::Ready(_) => "ready",
            Build::Failed => "failed",
        };
        f.debug_struct("VolumetricClouds")
            .field("build", &build)
            .field("noise", &self.noise.is_some())
            .field("targets", &self.targets.as_ref().map(|t| t.trace))
            .finish_non_exhaustive()
    }
}

impl VolumetricClouds {
    /// A renderer with nothing allocated: no GPU object, no thread.
    #[must_use]
    pub const fn new() -> Self {
        VolumetricClouds {
            build: Build::Idle,
            noise: None,
            targets: None,
            history: None,
            frame: 0,
            generation: 0,
        }
    }

    /// Start building the pipelines for `target` on a worker thread, if that has not started
    /// already. Cheap to call every frame. Asking for a different target than the one built
    /// (the frame's format or MSAA changed) starts over.
    pub fn request(&mut self, gpu: &Gpu, target: CloudTarget) {
        let stale = match &self.build {
            Build::Idle => false,
            Build::Building(built, _) => *built != target,
            Build::Ready(statics) => statics.target != target,
            Build::Failed => return,
        };
        if stale {
            self.release();
        } else if !matches!(self.build, Build::Idle) {
            return;
        }
        if !device_supports_clouds(gpu) {
            tracing::info!("volumetric clouds need compute shaders on Vulkan, DX12 or Metal");
            self.build = Build::Failed;
            return;
        }
        let device = gpu.device.clone();
        let (send, receive) = mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name("mc-volumetric-clouds".to_owned())
            .spawn(move || {
                let statics = Statics::new(&device, target);
                // The renderer may have been dropped meanwhile; then the pipelines just drop.
                let _ = send.send(statics);
            });
        self.build = match spawned {
            Ok(_) => Build::Building(target, receive),
            Err(error) => {
                tracing::warn!(%error, "volumetric clouds: no worker thread");
                Build::Failed
            }
        };
    }

    /// Pick up the worker's result if it has arrived. Returns whether the clouds can be traced.
    pub fn poll(&mut self) -> bool {
        if let Build::Building(_, receive) = &self.build {
            match receive.try_recv() {
                Ok(statics) => self.build = Build::Ready(Box::new(statics)),
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    tracing::warn!("volumetric clouds: building the pipelines failed");
                    self.build = Build::Failed;
                }
            }
        }
        self.is_ready()
    }

    /// Block until the worker has answered or `timeout` passed (tests and offline renders).
    /// Returns whether the clouds can be traced.
    pub fn wait_ready(&mut self, timeout: Duration) -> bool {
        if let Build::Building(_, receive) = &self.build {
            match receive.recv_timeout(timeout) {
                Ok(statics) => self.build = Build::Ready(Box::new(statics)),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => self.build = Build::Failed,
            }
        }
        self.is_ready()
    }

    /// Whether the pipelines are built.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        matches!(self.build, Build::Ready(_))
    }

    /// Whether this device cannot run the clouds, or building them failed.
    #[must_use]
    pub fn is_failed(&self) -> bool {
        matches!(self.build, Build::Failed)
    }

    /// Whether anything is allocated or being built: a GPU object, a worker thread.
    #[must_use]
    pub fn allocated(&self) -> bool {
        matches!(self.build, Build::Building(..) | Build::Ready(_))
            || self.noise.is_some()
            || self.targets.is_some()
    }

    /// Give everything back (pipelines, noise, targets) and forget any pending build. The next
    /// [`VolumetricClouds::request`] starts over.
    pub fn release(&mut self) {
        self.build = Build::Idle;
        self.noise = None;
        self.targets = None;
        self.history = None;
        self.generation += 1;
    }

    /// Give back the big memory (the noise textures, about 8.5 MB, and the targets) and keep the
    /// pipelines: the next trace regenerates the noise on the GPU, with no shader work.
    pub fn release_memory(&mut self) {
        self.noise = None;
        self.targets = None;
        self.history = None;
        self.generation += 1;
    }

    /// Forget the accumulated frames (the next trace starts from scratch).
    pub fn reset_history(&mut self) {
        self.history = None;
    }

    /// Drop the accumulated history *and*
    /// restart the jitter sequence, so what is drawn from here on depends on the frames that follow
    /// alone (the start of a video export: two exports of the same replay range then match byte
    /// for byte). Normal play never calls it.
    pub fn restart_sequence(&mut self) {
        self.history = None;
        self.frame = 0;
    }

    /// The outputs of the last trace: premultiplied light with transmittance, and the cloud's
    /// distance. Stable until [`CloudTraceInfo::generation`] changes.
    #[must_use]
    pub fn out_views(&self) -> Option<(&wgpu::TextureView, &wgpu::TextureView)> {
        self.targets
            .as_ref()
            .map(|t| (&t.out_color_view, &t.out_distance_view))
    }

    /// Record one frame of clouds: the noise generator the first time, the trace, and the copy
    /// into [`VolumetricClouds::out_views`]. `None` (and nothing recorded) until the pipelines are
    /// ready.
    pub fn trace(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        frame: &CloudFrame<'_>,
    ) -> Option<CloudTraceInfo> {
        if !self.poll() {
            return None;
        }
        let Build::Ready(statics) = &self.build else {
            return None;
        };
        let noise = self
            .noise
            .get_or_insert_with(|| Noise::new(&gpu.device, statics));
        noise.generate(statics, enc);
        let size = (frame.frame_size.0.max(1), frame.frame_size.1.max(1));
        let divisor = frame.quality.divisor();
        // The offscreen target never reads a scene depth; one built for the game always does.
        let scene_depth = match statics.target {
            CloudTarget::Offscreen => None,
            CloudTarget::Frame { .. } => Some(frame.scene_depth?),
        };
        if !self
            .targets
            .as_ref()
            .is_some_and(|t| t.matches(size, divisor, scene_depth))
        {
            self.targets = Some(Targets::new(
                &gpu.device,
                statics,
                noise,
                size,
                divisor,
                scene_depth,
            ));
            self.history = None;
            self.generation += 1;
        }
        let targets = self.targets.as_ref().expect("built above");

        let history = self
            .history
            .filter(|h| h.trace == targets.trace && (frame.eye - h.eye).length() < HISTORY_JUMP);
        let uniform = trace_uniform(frame, history, targets, self.frame);
        gpu.write_buffer_staged(&statics.trace_uniform, 0, bytemuck::bytes_of(&uniform));
        {
            let attachment = |view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })
            };
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("volumetric clouds trace"),
                color_attachments: &[
                    attachment(&targets.scratch_color_view),
                    attachment(&targets.scratch_distance_view),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&statics.trace_pipeline);
            pass.set_bind_group(0, &targets.group, &[]);
            pass.draw(0..3, 0..1);
        }
        let extent = wgpu::Extent3d {
            width: targets.trace.0,
            height: targets.trace.1,
            depth_or_array_layers: 1,
        };
        enc.copy_texture_to_texture(
            targets.scratch_color.as_image_copy(),
            targets.out_color.as_image_copy(),
            extent,
        );
        enc.copy_texture_to_texture(
            targets.scratch_distance.as_image_copy(),
            targets.out_distance.as_image_copy(),
            extent,
        );
        self.history = Some(History {
            view_proj: frame.view_proj,
            eye: frame.eye,
            trace: targets.trace,
        });
        self.frame = self.frame.wrapping_add(1);
        Some(CloudTraceInfo {
            trace_size: targets.trace,
            divisor,
            generation: self.generation,
        })
    }

    fn statics(&self) -> Option<&Statics> {
        match &self.build {
            Build::Ready(statics) => Some(statics),
            _ => None,
        }
    }
}

fn trace_uniform(
    frame: &CloudFrame<'_>,
    history: Option<History>,
    targets: &Targets,
    index: u32,
) -> TraceUniform {
    let layer = frame.layer;
    let look = frame.look;
    let (prev_view_proj, delta, valid) = match history {
        Some(h) => (h.view_proj, frame.eye - h.eye, 1.0),
        None => (frame.view_proj, DVec3::ZERO, 0.0),
    };
    let wrap = |v: f64, tile: f64| v.rem_euclid(tile) as f32;
    let eye = frame.eye;
    let (wind, drift) = (DVec2::from(look.wind), look.drift);
    let jitter = index % 16 + 1;
    let quality = frame.quality;
    TraceUniform {
        inv_view_proj: frame.view_proj.inverse().to_cols_array_2d(),
        prev_view_proj: prev_view_proj.to_cols_array_2d(),
        cam_delta: [delta.x as f32, delta.y as f32, delta.z as f32, valid],
        eye: [
            eye.y as f32,
            if frame.scene_depth.is_some() {
                1.0
            } else {
                0.0
            },
            layer.fade_start,
            layer.detail_range.max(1.0),
        ],
        origin_weather: [
            wrap(eye.x + wind.x, layer.weather_tile),
            wrap(eye.z + wind.y, layer.weather_tile),
            wrap(eye.x + wind.x, layer.shape_tile),
            wrap(eye.z + wind.y, layer.shape_tile),
        ],
        origin_detail: [
            wrap(eye.x + wind.x * 1.6, layer.detail_tile),
            wrap(eye.z + wind.y * 1.6, layer.detail_tile),
            wrap(drift * 0.6 - f64::from(layer.noise_shift), layer.shape_tile),
            wrap(
                drift * 2.2 - f64::from(layer.noise_shift),
                layer.detail_tile,
            ),
        ],
        layer: [
            layer.bottom,
            layer.top.max(layer.bottom + 1.0),
            layer.max_distance,
            layer.extinction,
        ],
        scale: [
            (1.0 / layer.weather_tile) as f32,
            (1.0 / layer.shape_tile) as f32,
            (1.0 / layer.detail_tile) as f32,
            layer.erosion,
        ],
        shape: [
            look.coverage.clamp(0.0, 1.0),
            look.rain.clamp(0.0, 1.0),
            look.type_bias,
            layer.lean,
        ],
        sun_dir: [
            look.sun_dir.x,
            look.sun_dir.y,
            look.sun_dir.z,
            layer.light_step,
        ],
        sun_color: [
            look.sun_color.x,
            look.sun_color.y,
            look.sun_color.z,
            look.ambient,
        ],
        amb_top: [
            look.ambient_top.x,
            look.ambient_top.y,
            look.ambient_top.z,
            0.0,
        ],
        amb_bottom: [
            look.ambient_bottom.x,
            look.ambient_bottom.y,
            look.ambient_bottom.z,
            0.0,
        ],
        quality: [
            quality.view_steps() as f32,
            quality.light_steps() as f32,
            if valid > 0.0 {
                frame.history_weight.clamp(0.01, 1.0)
            } else {
                1.0
            },
            index as f32,
        ],
        size: [
            targets.trace.0 as f32,
            targets.trace.1 as f32,
            targets.frame.0 as f32,
            targets.frame.1 as f32,
        ],
        jitter: [
            halton(jitter, 2),
            halton(jitter, 3),
            (index as f32 * 0.618_034).fract(),
            targets.divisor as f32,
        ],
        ray_z: [
            frame.ray_z.0,
            frame.ray_z.1,
            look.powder.clamp(0.0, 1.0),
            0.0,
        ],
    }
}

/// Which in-game slot records this frame.
const SLOT_NONE: u8 = 0;
const SLOT_EARLY: u8 = 1;
const SLOT_LATE: u8 = 2;

/// The in-game volumetric clouds: [`VolumetricClouds`] plus a full-resolution composite onto the
/// frame, depth-tested against the world, recorded by one of two slots ([`EARLY_ORDER`] or
/// [`LATE_ORDER`], chosen per frame from the eye's height). See the [module docs](self).
///
/// The client drives it every frame: [`VolumetricCloudsPhase::request`] once the mod is on,
/// then [`VolumetricCloudsPhase::set_frame`] with this frame's state, and fills
/// [`VolumetricCloudsPhase::layer`], [`VolumetricCloudsPhase::look`] and
/// [`VolumetricCloudsPhase::quality`]. With the mod off it calls
/// [`VolumetricCloudsPhase::release`] once and then never touches it.
pub struct VolumetricCloudsPhase {
    /// The cloud layer (bases at the vanilla cloud height).
    pub layer: CloudLayer,
    /// This frame's light and weather.
    pub look: CloudLook,
    /// The quality preset.
    pub quality: CloudQuality,
    core: VolumetricClouds,
    slot: Rc<Cell<u8>>,
    /// The composite's bind group and what it was built for: the core's output generation and
    /// the scene depth texture.
    composite_group: Option<(u64, wgpu::Texture, wgpu::BindGroup)>,
}

impl std::fmt::Debug for VolumetricCloudsPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VolumetricCloudsPhase")
            .field("core", &self.core)
            .field("slot", &self.slot.get())
            .finish_non_exhaustive()
    }
}

impl Default for VolumetricCloudsPhase {
    fn default() -> Self {
        Self::new()
    }
}

impl VolumetricCloudsPhase {
    /// Nothing allocated, nothing drawn.
    #[must_use]
    pub fn new() -> Self {
        VolumetricCloudsPhase {
            layer: CloudLayer::in_game(crate::sky::clouds::DEFAULT_CLOUD_HEIGHT),
            look: CloudLook::default(),
            quality: CloudQuality::default(),
            core: VolumetricClouds::new(),
            slot: Rc::new(Cell::new(SLOT_NONE)),
            composite_group: None,
        }
    }

    /// Register the two slots ([`EARLY_ORDER`], [`LATE_ORDER`]) and return the shared state to
    /// drive. Allocates nothing on the GPU.
    pub fn install(phases: &mut PhaseList) -> Rc<RefCell<VolumetricCloudsPhase>> {
        let phase = Rc::new(RefCell::new(VolumetricCloudsPhase::new()));
        let slot = Rc::clone(&phase.borrow().slot);
        phases.insert(
            EARLY_ORDER,
            Box::new(CloudSlot {
                shared: Rc::clone(&phase),
                active: Rc::clone(&slot),
                me: SLOT_EARLY,
            }),
        );
        phases.insert(
            LATE_ORDER,
            Box::new(CloudSlot {
                shared: Rc::clone(&phase),
                active: slot,
                me: SLOT_LATE,
            }),
        );
        phase
    }

    /// Start building the pipelines for this `Gpu`'s frame on a worker thread (idempotent).
    pub fn request(&mut self, gpu: &Gpu) {
        self.core.request(
            gpu,
            CloudTarget::Frame {
                format: gpu.config.format,
                samples: gpu.sample_count(),
            },
        );
    }

    /// Whether the clouds can draw (polls the worker).
    pub fn poll(&mut self) -> bool {
        self.core.poll()
    }

    /// Block until the pipelines are built or `timeout` passed (tests).
    pub fn wait_ready(&mut self, timeout: Duration) -> bool {
        self.core.wait_ready(timeout)
    }

    /// Whether this device cannot run the clouds, or building them failed.
    #[must_use]
    pub fn is_failed(&self) -> bool {
        self.core.is_failed()
    }

    /// This frame's decision: draw when `draw` is set and the pipelines are ready, from the early
    /// slot while the eye (world height `eye_y`) is below the cloud bases, else from the late one.
    /// Returns whether the clouds draw this frame (the caller then hides vanilla's).
    pub fn set_frame(&mut self, draw: bool, eye_y: f64) -> bool {
        let slot = if draw && self.core.poll() {
            if eye_y < f64::from(self.layer.bottom) {
                SLOT_EARLY
            } else {
                SLOT_LATE
            }
        } else {
            SLOT_NONE
        };
        self.slot.set(slot);
        slot != SLOT_NONE
    }

    /// Whether the clouds draw this frame (after [`VolumetricCloudsPhase::set_frame`]).
    #[must_use]
    pub fn drawing(&self) -> bool {
        self.slot.get() != SLOT_NONE
    }

    /// The order of the slot drawing this frame, if any.
    #[must_use]
    pub fn drawing_order(&self) -> Option<PhaseOrder> {
        match self.slot.get() {
            SLOT_EARLY => Some(EARLY_ORDER),
            SLOT_LATE => Some(LATE_ORDER),
            _ => None,
        }
    }

    /// Stop drawing and give every GPU object back; the next [`VolumetricCloudsPhase::request`]
    /// starts over.
    pub fn release(&mut self) {
        self.slot.set(SLOT_NONE);
        self.composite_group = None;
        self.core.release();
    }

    /// Whether anything is allocated or being built.
    #[must_use]
    pub fn allocated(&self) -> bool {
        self.core.allocated() || self.composite_group.is_some()
    }

    /// Forget the accumulated frames (a world change).
    pub fn reset_history(&mut self) {
        self.core.reset_history();
    }

    /// See the core's `restart_sequence`: history and jitter restart (a
    /// video export's start). Normal play never calls it.
    pub fn restart_sequence(&mut self) {
        self.core.restart_sequence();
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
        name: &'static str,
    ) {
        mc_core::profile_scope!("sky.volumetric_clouds.record");
        let view_proj = cam.view_proj();
        let frame = CloudFrame {
            view_proj,
            eye: cam.pos,
            ray_z: (0.0, 1.0),
            frame_size: (gpu.config.width, gpu.config.height),
            layer: self.layer,
            look: self.look,
            quality: self.quality,
            scene_depth: Some(gpu.depth_texture()),
            history_weight: IN_GAME_HISTORY_WEIGHT,
        };
        let Some(info) = self.core.trace(gpu, enc, &frame) else {
            return;
        };
        let (Some(statics), Some((color, distance))) = (self.core.statics(), self.core.out_views())
        else {
            return;
        };
        let Some(composite) = statics.composite.as_ref() else {
            return;
        };
        let depth = gpu.depth_texture();
        let current = self
            .composite_group
            .as_ref()
            .is_some_and(|(generation, texture, _)| {
                *generation == info.generation && texture == depth
            });
        if !current {
            let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("volumetric clouds composite group0"),
                layout: &composite.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: composite.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: wgpu::BindingResource::TextureView(color),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: wgpu::BindingResource::TextureView(distance),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: wgpu::BindingResource::Sampler(&statics.clamp_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 14,
                        resource: wgpu::BindingResource::TextureView(&depth_view),
                    },
                ],
            });
            self.composite_group = Some((info.generation, depth.clone(), group));
        }
        let Some((_, _, group)) = self.composite_group.as_ref() else {
            return;
        };
        let uniform = CompositeUniform {
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            eye: [
                cam.pos.y as f32,
                self.layer.bottom,
                self.layer.top.max(self.layer.bottom + 1.0),
                self.look.exposure,
            ],
            grid: [
                (info.trace_size.0 * info.divisor) as f32,
                (info.trace_size.1 * info.divisor) as f32,
                frame.ray_z.0,
                frame.ray_z.1,
            ],
        };
        gpu.write_buffer_staged(&composite.uniform, 0, bytemuck::bytes_of(&uniform));
        let mut pass = gpu.begin_pass(enc, view, name, DepthAttachment::None);
        pass.set_pipeline(&composite.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// One of the two in-game slots. Disabled unless [`VolumetricCloudsPhase::set_frame`] picked it.
struct CloudSlot {
    shared: Rc<RefCell<VolumetricCloudsPhase>>,
    active: Rc<Cell<u8>>,
    me: u8,
}

impl RenderPhase for CloudSlot {
    fn name(&self) -> &'static str {
        if self.me == SLOT_EARLY {
            EARLY_PHASE_NAME
        } else {
            LATE_PHASE_NAME
        }
    }

    fn enabled(&self) -> bool {
        self.active.get() == self.me
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        let name = self.name();
        self.shared.borrow_mut().record(gpu, enc, view, cam, name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sky::environment::{Dimension, Weather};

    fn validate(source: &str) -> wgpu::naga::Module {
        let module = wgpu::naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{error:?}"));
        module
    }

    /// `cargo check` never parses WGSL; this does, with the validator wgpu runs at pipeline
    /// creation, for both depth spellings and the noise generator.
    #[test]
    fn the_shaders_parse_and_validate() {
        validate(&trace_source(1));
        validate(&trace_source(4));
        validate(NOISE_WGSL);
    }

    /// The Rust uniforms and the WGSL structs must agree byte for byte.
    #[test]
    fn the_uniform_layouts_match_the_shader() {
        let module = validate(&trace_source(1));
        let mut layouter = wgpu::naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).expect("layouts compute");
        let span = |name: &str| {
            module
                .types
                .iter()
                .find(|(_, ty)| ty.name.as_deref() == Some(name))
                .map(|(handle, _)| layouter[handle].size)
                .unwrap_or_else(|| panic!("struct {name} in volumetric.wgsl"))
        };
        assert_eq!(span("Cloud") as usize, std::mem::size_of::<TraceUniform>());
        assert_eq!(
            span("Composite") as usize,
            std::mem::size_of::<CompositeUniform>()
        );
    }

    #[test]
    fn quality_presets_map_from_the_menu_index() {
        assert_eq!(CloudQuality::from_index(0), CloudQuality::Low);
        assert_eq!(CloudQuality::from_index(1), CloudQuality::Medium);
        assert_eq!(CloudQuality::from_index(2), CloudQuality::High);
        assert_eq!(CloudQuality::from_index(9), CloudQuality::Medium);
        assert_eq!(CloudQuality::default(), CloudQuality::Medium);
        for q in [CloudQuality::Low, CloudQuality::Medium, CloudQuality::High] {
            assert!(
                q.divisor() <= 4,
                "the trace footprint loop reads at most 4x4"
            );
            assert!(
                q.view_steps() <= 160 && q.light_steps() <= 8,
                "shader loop bounds"
            );
        }
    }

    fn env(world_time: i64, rain: f32) -> Environment {
        Environment {
            world_time,
            dimension: Dimension::Overworld,
            weather: Weather {
                rain_strength: rain,
                thunder_strength: 0.0,
                last_lightning_bolt: 0,
            },
        }
    }

    #[test]
    fn the_light_follows_vanillas_sun_and_moon() {
        // Noon (world time 6000): the sun overhead, bright and white.
        let noon = in_game_look(&env(6000, 0.0), 0.8, 0.0, 0.5);
        assert!(noon.sun_dir.y > 0.99, "{:?}", noon.sun_dir);
        assert!(noon.sun_color.x > 1.2 && noon.sun_color.z > 1.1);
        // Morning (world time 1000): the sun in the east (+X).
        let morning = in_game_look(&env(1000, 0.0), 0.8, 0.0, 0.5);
        assert!(morning.sun_dir.x > 0.3, "{:?}", morning.sun_dir);
        // Midnight: the moon overhead, far dimmer and bluer than the sun.
        let night = in_game_look(&env(18_000, 0.0), 0.8, 0.0, 0.5);
        assert!(night.sun_dir.y > 0.99, "the moon is overhead at midnight");
        assert!(night.sun_color.length() < noon.sun_color.length() * 0.2);
        assert!(night.sun_color.z > night.sun_color.x);
        assert!(night.ambient_top.length() < noon.ambient_top.length() * 0.2);
    }

    #[test]
    fn the_sun_to_moon_hand_over_never_pops() {
        // Sweep dusk tick by tick: the light's brightness must change smoothly.
        let mut last = in_game_look(&env(11_000, 0.0), 0.8, 0.0, 0.5)
            .sun_color
            .length();
        for t in 11_001..14_500 {
            let now = in_game_look(&env(t, 0.0), 0.8, 0.0, 0.5).sun_color.length();
            assert!(
                (now - last).abs() < 0.02,
                "a jump at world time {t}: {last} -> {now}"
            );
            last = now;
        }
    }

    #[test]
    fn rain_adds_cover_and_dims_the_light() {
        let dry = in_game_look(&env(6000, 0.0), 0.8, 0.0, 0.5);
        let wet = in_game_look(&env(6000, 1.0), 0.8, 0.0, 0.5);
        assert!(wet.coverage > dry.coverage);
        assert!(wet.sun_color.length() < dry.sun_color.length() * 0.5);
        assert!(
            wet.type_bias < 0.0,
            "rain flattens the clouds toward stratus"
        );
    }

    #[test]
    fn a_new_phase_allocates_nothing_and_draws_nothing() {
        let mut phase = VolumetricCloudsPhase::new();
        assert!(!phase.allocated());
        // Without a request the core is idle, so asking to draw does nothing.
        assert!(!phase.set_frame(true, 70.0));
        assert!(!phase.drawing());
        assert_eq!(phase.drawing_order(), None);
        assert!(!phase.allocated());
    }

    #[test]
    fn the_jitter_sequence_stays_inside_the_pixel() {
        for i in 1..=16 {
            let (x, y) = (halton(i, 2), halton(i, 3));
            assert!((0.0..1.0).contains(&x) && (0.0..1.0).contains(&y));
        }
    }
}
