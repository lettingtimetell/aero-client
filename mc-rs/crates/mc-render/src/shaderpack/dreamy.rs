//! The Dreamy shader pack: an atmospheric, moody look with deep
//! colour controls and an optional ground fog, tuned so players stay easy to see.
//!
//! # The frame
//!
//! Two phases, both gated on [`ShaderCore::is_running`]`(PackId::Dreamy)` (switched on, a world on
//! screen, no MSAA) plus their own switches:
//!
//! * [`DreamyAtmospherePhase`] at [`super::order::WORLD`] (641: after the translucent terrain and
//!   the screen-space fog, before the late volumetric clouds and TAA, so TAA cleans its noise and
//!   the hand, labels and HUD drawn later are never touched). It records the ground fog march
//!   ([`super::ground_fog`], half or quarter resolution) and the light rays
//!   ([`super::godrays`], quarter resolution) when they are on, then one full-resolution
//!   composite that reads the frame and the depth buffer: distance haze (with a glow toward the
//!   sun), the sky tint, the fog's bilateral upsample and the rays, in linear light.
//!   It steps aside while the eye is in water or lava (vanilla's fog there is gameplay).
//! * [`DreamyFinishPhase`] at [`super::order::FINAL`] (`POST`: after the hand and the overlays,
//!   before the HUD). The Orton glow and halation from a dual-filter pyramid (half resolution down
//!   to 1/32 or 1/64), the colour grade as one lookup in the baked LUT ([`super::grade`]), the
//!   vignette, film grain, chromatic aberration (off by default) and a one-level dither.
//!
//! The HUD, crosshair, chat and screens are drawn after both phases and are never graded,
//! glowed or vignetted.
//!
//! # PvP
//!
//! Nothing here blurs the sharp image: the glow is screen-blended over it, so edges, nametags and
//! hit flashes keep their detail. Haze and fog start past a distance ([`DreamySettings`]), and the
//! fog upsample is depth-aware, so a player in front of fog keeps a clean outline. The presets'
//! grades are checked against team red and blue in the client's tests.
//!
//! # Off costs nothing
//!
//! [`DreamyAtmospherePhase::new`] and [`DreamyFinishPhase::new`] allocate nothing. While the pack
//! does not run, both report `enabled() == false` and drop every GPU resource they built
//! ([`DreamyAtmospherePhase::release`] / [`DreamyFinishPhase::release`], called by the glue when
//! the pack goes off, and on their own the first frame they are disabled again).

use glam::{DVec3, Mat4, Vec3};

use super::frame::{EyeMedium, PackFrame, PackId};
use super::godrays::{light_on_screen, LightRays, RaysParams};
use super::grade::{GradeLut, GradeParams, LUT_APPLY_WGSL};
use super::ground_fog::{
    eye_relative, world_inverse_view_proj, FogLayer, FogLight, GroundFog, DIST_FORMAT, FOG_FORMAT,
    SKY_DISTANCE,
};
use super::shared::ShaderCore;
use crate::{Camera, Gpu, RenderPhase, SceneTextures};

/// The glow pyramid's format.
pub const GLOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The most glow levels (half resolution down to 1/64).
pub const MAX_GLOW_LEVELS: usize = 6;

const COMMON_WGSL: &str = include_str!("dreamy_common.wgsl");
const ATMOSPHERE_WGSL: &str = include_str!("dreamy_atmosphere.wgsl");
const GLOW_WGSL: &str = include_str!("dreamy_glow.wgsl");
const FINISH_WGSL: &str = include_str!("dreamy_finish.wgsl");

/// The atmosphere shader's full source.
#[must_use]
pub fn atmosphere_source() -> String {
    format!("{COMMON_WGSL}\n{ATMOSPHERE_WGSL}")
}

/// The glow shader's full source.
#[must_use]
pub fn glow_source() -> String {
    format!("{COMMON_WGSL}\n{GLOW_WGSL}")
}

/// The finish shader's full source (with the LUT helper).
#[must_use]
pub fn finish_source() -> String {
    format!("{COMMON_WGSL}\n{LUT_APPLY_WGSL}\n{FINISH_WGSL}")
}

/// Ground fog's settings (the layer's top is set per frame: [`DreamyAtmospherePhase::set_fog_top`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FogSettings {
    /// The layer's shape. `top` here is ignored; the glue sets it per frame.
    pub layer: FogLayer,
    /// The fog's colour (display RGB), or `None` to follow vanilla's fog colour (the sky's).
    pub color: Option<[f32; 3]>,
    /// How much the sun (or moon) lights the fog toward it, 0..1.
    pub light: f32,
}

impl Default for FogSettings {
    fn default() -> Self {
        FogSettings {
            layer: FogLayer::default(),
            color: None,
            light: 0.5,
        }
    }
}

/// The light rays' settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaySettings {
    /// Brightness, 0..1.
    pub strength: f32,
    /// How far the rays reach across the screen, 0..1.
    pub length: f32,
}

impl Default for RaySettings {
    fn default() -> Self {
        RaySettings {
            strength: 0.4,
            length: 0.7,
        }
    }
}

/// Everything the Dreamy pack's menu controls. The glue fills it whenever the settings change.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DreamySettings {
    /// The colour grade, baked into the LUT.
    pub grade: GradeParams,
    /// Distance haze, 0..1 (0 = off). 1 hides about half the contrast at 100 blocks past the
    /// start.
    pub haze: f32,
    /// No haze nearer than this, in blocks.
    pub haze_start: f32,
    /// The sky tint's colour (display RGB).
    pub sky_tint: [f32; 3],
    /// The sky tint's strength, 0..1 (0 = off).
    pub sky_tint_strength: f32,
    /// Ground fog, `None` = off.
    pub fog: Option<FogSettings>,
    /// Light rays, `None` = off.
    pub rays: Option<RaySettings>,
    /// The Orton glow, 0..1 (0 = off).
    pub glow: f32,
    /// How wide the glow spreads, 0..1.
    pub glow_radius: f32,
    /// Halation (a warm fringe around highlights), 0..1.
    pub halation: f32,
    /// Vignette, 0..1.
    pub vignette: f32,
    /// Film grain, 0..1.
    pub grain: f32,
    /// Chromatic aberration, 0..1 (0 = off).
    pub chromatic: f32,
}

impl Default for DreamySettings {
    fn default() -> Self {
        DreamySettings {
            grade: GradeParams::IDENTITY,
            haze: 0.0,
            haze_start: 32.0,
            sky_tint: [1.0; 3],
            sky_tint_strength: 0.0,
            fog: None,
            rays: None,
            glow: 0.0,
            glow_radius: 0.6,
            halation: 0.0,
            vignette: 0.0,
            grain: 0.0,
            chromatic: 0.0,
        }
    }
}

impl DreamySettings {
    /// Whether the atmosphere phase has anything to draw.
    #[must_use]
    pub fn atmosphere_on(&self) -> bool {
        self.haze > 0.0 || self.sky_tint_strength > 0.0 || self.fog.is_some() || self.rays.is_some()
    }

    /// Whether the finish phase has anything to draw. The dither alone is not a reason.
    #[must_use]
    pub fn finish_on(&self) -> bool {
        self.glow > 0.0
            || self.halation > 0.0
            || self.vignette > 0.0
            || self.grain > 0.0
            || self.chromatic > 0.0
            || !self.grade.is_identity()
    }

    /// The haze density per block for a 0..1 haze strength.
    #[must_use]
    pub fn haze_density(&self) -> f32 {
        self.haze.clamp(0.0, 1.0) * 0.007
    }

    /// How many glow levels the radius asks for.
    #[must_use]
    pub fn glow_levels(&self) -> usize {
        (4 + (self.glow_radius.clamp(0.0, 1.0) * 2.0).round() as usize).min(MAX_GLOW_LEVELS)
    }
}

/// How bright the light's glows (haze glow, fog scatter, rays) are against the scene. The core's
/// moon strength is tuned for lighting terrain under vanilla's darkened night lightmap; against a
/// night sky about 0.003 linear it lit the whole sky grey toward the moon (night pictures), so the
/// moon's glows are a quarter of the sun's. `light_strength` fades to zero at the horizon, so the
/// switch never pops.
fn glow_scale(frame: &PackFrame) -> f32 {
    if frame.light_is_sun {
        1.0
    } else {
        0.25
    }
}

/// The display-to-linear sRGB transfer, per channel.
fn to_linear(c: [f32; 3]) -> Vec3 {
    super::grade::srgb_to_linear(Vec3::from(c))
}

fn luma(c: Vec3) -> f32 {
    c.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}

/// A colour scaled to unit luminance (a tint that does not change brightness).
fn unit_luma(c: Vec3) -> Vec3 {
    let l = luma(c);
    if l > 1.0e-4 {
        c / l
    } else {
        Vec3::ONE
    }
}

fn fullscreen_pipeline(
    gpu: &Gpu,
    module: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    entry: &'static str,
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    gpu.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(entry),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_fullscreen"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            // Single-sampled: packs never run under MSAA.
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
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

fn texture_entry(binding: u32, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
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

fn linear_sampler(gpu: &Gpu, label: &str) -> wgpu::Sampler {
    gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    })
}

fn texture_2d(
    gpu: &Gpu,
    label: &str,
    size: (u32, u32),
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
    gpu.device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.0.max(1),
                height: size.1.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

fn overwrite(view: &wgpu::TextureView) -> Option<wgpu::RenderPassColorAttachment<'_>> {
    Some(wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        },
    })
}

fn fullscreen_pass(
    enc: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    group: &wgpu::BindGroup,
) {
    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[overwrite(target)],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.draw(0..3, 0..1);
}

// ------------------------------------------------------------------------------------ atmosphere

/// `AtmosphereUniform` in `dreamy_atmosphere.wgsl` (64 + 9 × 16 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct AtmosphereUniform {
    inv_view_proj: [[f32; 4]; 4],
    haze: [f32; 4],
    switches: [f32; 4],
    sky_tint: [f32; 4],
    light: [f32; 4],
    light_glow: [f32; 4],
    rays: [f32; 4],
    fog_size: [f32; 4],
    size: [f32; 4],
    eye_rel: [f32; 4],
}

struct AtmospherePipelines {
    format: wgpu::TextureFormat,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    /// Stand-ins bound while fog or rays are off (the shader skips them).
    dummy_fog: wgpu::TextureView,
    dummy_dist: wgpu::TextureView,
    dummy_rays: wgpu::TextureView,
}

impl AtmospherePipelines {
    fn new(gpu: &Gpu, format: wgpu::TextureFormat) -> AtmospherePipelines {
        let device = &gpu.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaderpack/dreamy atmosphere"),
            source: wgpu::ShaderSource::Wgsl(atmosphere_source().into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shaderpack/dreamy atmosphere"),
            entries: &[
                uniform_entry(0),
                texture_entry(1, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                texture_entry(3, false),
                texture_entry(4, false),
                texture_entry(5, true),
                sampler_entry(6),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shaderpack/dreamy atmosphere"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = fullscreen_pipeline(
            gpu,
            &module,
            &pipeline_layout,
            "fs_atmosphere",
            format,
            None,
        );
        AtmospherePipelines {
            format,
            layout,
            pipeline,
            sampler: linear_sampler(gpu, "shaderpack/dreamy atmosphere"),
            uniform: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shaderpack/dreamy atmosphere"),
                size: std::mem::size_of::<AtmosphereUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            dummy_fog: texture_2d(gpu, "shaderpack/dreamy no fog", (1, 1), FOG_FORMAT),
            dummy_dist: texture_2d(gpu, "shaderpack/dreamy no fog", (1, 1), DIST_FORMAT),
            dummy_rays: texture_2d(
                gpu,
                "shaderpack/dreamy no rays",
                (1, 1),
                super::godrays::RAYS_FORMAT,
            ),
        }
    }
}

/// The atmosphere composite's bind group and the views it was built from (compared by identity
/// every frame; rebuilt only when one changes: a resize, fog or rays switched on or off).
struct AtmosphereBinding {
    scene: wgpu::TextureView,
    depth: wgpu::Texture,
    fog: Option<wgpu::TextureView>,
    rays: Option<wgpu::TextureView>,
    group: wgpu::BindGroup,
}

/// Haze, sky tint, ground fog and light rays over the world. See the [module docs](self).
pub struct DreamyAtmospherePhase {
    core: ShaderCore,
    settings: DreamySettings,
    /// The fog layer's top this frame (world y), from the glue.
    fog_top: f32,
    /// How far the fog's noise has drifted, in blocks.
    wind: DVec3,
    /// How much of the sky a custom sky covers, 0..1: the fog keeps that much of it in view
    /// ([`GroundFog::set_keep_sky`]).
    keep_sky: f32,
    pipelines: Option<AtmospherePipelines>,
    binding: Option<AtmosphereBinding>,
    fog: Option<GroundFog>,
    rays: Option<LightRays>,
    uploaded: Option<AtmosphereUniform>,
    passes: u64,
}

impl std::fmt::Debug for DreamyAtmospherePhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DreamyAtmospherePhase")
            .field("settings", &self.settings)
            .field("fog_top", &self.fog_top)
            .field("allocated", &self.allocated())
            .field("passes", &self.passes)
            .finish_non_exhaustive()
    }
}

impl DreamyAtmospherePhase {
    /// The phase's name in `PhaseList`, GPU timings and captures.
    pub const NAME: &'static str = "shaderpack/dreamy-atmosphere";

    /// An idle phase. Allocates nothing.
    #[must_use]
    pub fn new(core: ShaderCore) -> DreamyAtmospherePhase {
        DreamyAtmospherePhase {
            core,
            settings: DreamySettings::default(),
            fog_top: 0.0,
            wind: DVec3::ZERO,
            keep_sky: 0.0,
            pipelines: None,
            binding: None,
            fog: None,
            rays: None,
            uploaded: None,
            passes: 0,
        }
    }

    /// Set the pack's settings. Drops the fog or rays resources when they are switched off.
    pub fn set_settings(&mut self, settings: DreamySettings) {
        self.settings = settings;
        if settings.fog.is_none() {
            self.fog = None;
        }
        if settings.rays.is_none() {
            self.rays = None;
        }
        if !settings.atmosphere_on() {
            self.release();
        }
    }

    /// The settings in effect.
    #[must_use]
    pub fn settings(&self) -> DreamySettings {
        self.settings
    }

    /// Set the fog layer's top for this frame (world y).
    pub fn set_fog_top(&mut self, top: f32) {
        self.fog_top = top;
    }

    /// The fog layer's top in effect.
    #[must_use]
    pub fn fog_top(&self) -> f32 {
        self.fog_top
    }

    /// Set how far the fog's noise has drifted with the wind, in blocks.
    pub fn set_wind(&mut self, wind: DVec3) {
        self.wind = wind;
    }

    /// Say how much of the sky a resource pack's custom sky layers cover this frame, 0..1. As far
    /// as they do, the ground fog leaves the sky below the horizon as they drew it, the way it
    /// looks with the pack off (the pack's tint and grade still apply); while they are faded out
    /// (0) the fog sea is drawn as under the vanilla sky. See [`GroundFog::set_keep_sky`].
    pub fn set_keep_sky(&mut self, keep: f32) {
        self.keep_sky = keep;
    }

    /// Whether the phase draws this frame.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.core.is_running(PackId::Dreamy)
            && self.settings.atmosphere_on()
            && self.core.frame().eye_medium == EyeMedium::Air
    }

    /// Drop every GPU resource.
    pub fn release(&mut self) {
        self.pipelines = None;
        self.binding = None;
        self.fog = None;
        self.rays = None;
        self.uploaded = None;
    }

    /// Whether any GPU resource exists (tests: off holds none).
    #[must_use]
    pub fn allocated(&self) -> bool {
        self.pipelines.is_some() || self.fog.is_some() || self.rays.is_some()
    }

    /// How many composites have been recorded.
    #[must_use]
    pub fn passes_recorded(&self) -> u64 {
        self.passes
    }

    /// The fog march and ray counters: `(marches, ray passes)`.
    #[must_use]
    pub fn sub_passes(&self) -> (u64, u64) {
        (
            self.fog.as_ref().map_or(0, GroundFog::marches),
            self.rays.as_ref().map_or(0, LightRays::records),
        )
    }

    fn fog_light(&self, frame: &PackFrame, fog: &FogSettings) -> FogLight {
        let direct = self
            .core
            .light_colors()
            .map_or(Vec3::ONE * frame.light_strength, |light| light.direct);
        let ambient = match fog.color {
            Some(color) => to_linear(color) * (0.25 + 0.75 * frame.day.max(0.12)),
            // Vanilla's night fog colour is nearly black (about 0.05 display), which would make
            // the fog vanish into the void: keep a faint moonlit blue under it at night.
            None => to_linear(frame.fog_color_display.to_array())
                .max(Vec3::new(0.018, 0.024, 0.045) * (1.0 - frame.day)),
        };
        FogLight {
            dir: frame.light_dir,
            color: direct * (0.3 * fog.light.clamp(0.0, 1.0) * glow_scale(frame)),
            g: 0.6,
            ambient,
            depth_darkening: 0.4,
        }
    }

    #[allow(clippy::too_many_lines)] // One frame's recording, in order.
    fn record_frame(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        let frame = self.core.frame();
        let settings = self.settings;
        let full = (scene.width, scene.height);
        let jitter_frame = if cam.jitter == glam::Vec2::ZERO {
            0
        } else {
            frame.frame_index % 64 + 1
        };

        // Ground fog: the low-resolution march.
        if let Some(fog_settings) = settings.fog {
            let light = self.fog_light(&frame, &fog_settings);
            let fog = self.fog.get_or_insert_with(|| GroundFog::new(gpu));
            fog.set_layer(FogLayer {
                top: self.fog_top,
                ..fog_settings.layer
            });
            fog.set_light(light);
            fog.set_wind(self.wind);
            fog.set_jitter_frame(jitter_frame);
            fog.set_keep_sky(self.keep_sky);
            fog.record(gpu, enc, cam, full);
        }

        // Light rays: only while the light is on screen and shining.
        let mut rays_colour = Vec3::ZERO;
        let mut rays_on = false;
        if let Some(rays) = settings.rays {
            let light = frame.light_dir;
            let strength = frame.light_strength * rays.strength.clamp(0.0, 1.0);
            if strength > 0.0 {
                if let Some(on_screen) = light_on_screen(cam, light) {
                    let direct = self
                        .core
                        .light_colors()
                        .map_or(Vec3::ONE * frame.light_strength, |l| l.direct);
                    let lights = self.rays.get_or_insert_with(|| LightRays::new(gpu));
                    lights.record(
                        gpu,
                        enc,
                        full,
                        // A tight glow around the light: the radial blur stretches it into
                        // streaks, broken where blocks stand in front of the light.
                        &RaysParams {
                            light_uv: on_screen.uv,
                            radius: 0.12 + 0.1 * rays.length.clamp(0.0, 1.0),
                            reach: 0.35 + 0.6 * rays.length.clamp(0.0, 1.0),
                            decay: 0.95,
                            disc: 1.0,
                            jitter_frame,
                        },
                    );
                    let unit = if frame.light_strength > 0.0 {
                        direct / frame.light_strength
                    } else {
                        direct
                    };
                    rays_colour = unit * strength * on_screen.visibility * 0.6 * glow_scale(&frame);
                    rays_on = true;
                }
            }
        }

        if self
            .pipelines
            .as_ref()
            .is_none_or(|pipelines| pipelines.format != scene.format)
        {
            self.pipelines = Some(AtmospherePipelines::new(gpu, scene.format));
            self.binding = None;
            self.uploaded = None;
        }
        let pipelines = self.pipelines.as_ref().expect("built above");

        let fog_views = settings
            .fog
            .and(self.fog.as_ref())
            .and_then(|fog| Some((fog.fog_view()?.clone(), fog.dist_view()?.clone())));
        let rays_view = if rays_on {
            self.rays.as_ref().and_then(|rays| rays.output().cloned())
        } else {
            None
        };
        let depth = gpu.depth_texture();
        let fits = self.binding.as_ref().is_some_and(|binding| {
            binding.scene == *scene.color
                && binding.depth == *depth
                && binding.fog.as_ref() == fog_views.as_ref().map(|(fog, _)| fog)
                && binding.rays == rays_view
        });
        if !fits {
            let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
            let (fog_view, dist_view) = match &fog_views {
                Some((fog, dist)) => (fog, dist),
                None => (&pipelines.dummy_fog, &pipelines.dummy_dist),
            };
            let rays = rays_view.as_ref().unwrap_or(&pipelines.dummy_rays);
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shaderpack/dreamy atmosphere"),
                layout: &pipelines.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pipelines.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(scene.color),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(fog_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(dist_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(rays),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&pipelines.sampler),
                    },
                ],
            });
            self.binding = Some(AtmosphereBinding {
                scene: scene.color.clone(),
                depth: depth.clone(),
                fog: fog_views.as_ref().map(|(fog, _)| fog.clone()),
                rays: rays_view.clone(),
                group,
            });
        }

        let eye_rel = eye_relative(cam);
        let fog_linear = to_linear(frame.fog_color_display.to_array());
        let tint = unit_luma(to_linear(settings.sky_tint));
        let tint_strength = settings.sky_tint_strength.clamp(0.0, 1.0);
        let haze_colour = fog_linear * Vec3::ONE.lerp(tint, 0.5 * tint_strength);
        let direct = self
            .core
            .light_colors()
            .map_or(Vec3::ONE * frame.light_strength, |light| light.direct);
        // The sun's glow in the haze. Kept low: at sunset it sits right behind the players you
        // look toward (round 1 of the pictures washed out an archer 33 blocks away).
        let glow = direct * (0.05 * glow_scale(&frame));
        let (fog_size, fog_on) = match self.fog.as_ref().and_then(GroundFog::low_size) {
            Some((size, divisor)) if fog_views.is_some() => (
                [size.0 as f32, size.1 as f32, divisor as f32, SKY_DISTANCE],
                1.0,
            ),
            _ => ([1.0, 1.0, 1.0, SKY_DISTANCE], 0.0),
        };
        let uniform = AtmosphereUniform {
            inv_view_proj: world_inverse_view_proj(cam).to_cols_array_2d(),
            haze: [
                haze_colour.x,
                haze_colour.y,
                haze_colour.z,
                settings.haze_density(),
            ],
            switches: [
                settings.haze_start.max(0.0),
                fog_on,
                f32::from(u8::from(rays_on)),
                0.6,
            ],
            sky_tint: [tint.x, tint.y, tint.z, tint_strength],
            light: [
                frame.light_dir.x,
                frame.light_dir.y,
                frame.light_dir.z,
                0.65,
            ],
            light_glow: [glow.x, glow.y, glow.z, 0.0],
            rays: [rays_colour.x, rays_colour.y, rays_colour.z, 0.0],
            fog_size,
            size: [
                full.0 as f32,
                full.1 as f32,
                1.0 / full.0.max(1) as f32,
                1.0 / full.1.max(1) as f32,
            ],
            eye_rel: [eye_rel.x, eye_rel.y, eye_rel.z, 0.0],
        };
        if self.uploaded != Some(uniform) {
            gpu.write_buffer_staged(&pipelines.uniform, 0, bytemuck::bytes_of(&uniform));
            self.uploaded = Some(uniform);
        }
        let group = &self.binding.as_ref().expect("built above").group;
        fullscreen_pass(enc, Self::NAME, view, &pipelines.pipeline, group);
        self.passes += 1;
    }
}

impl RenderPhase for DreamyAtmospherePhase {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn enabled(&self) -> bool {
        self.is_active()
    }

    fn needs_scene_color(&self) -> bool {
        self.is_active()
    }

    fn resize(&mut self, _gpu: &Gpu, _width: u32, _height: u32) {
        self.binding = None;
    }

    fn record(
        &mut self,
        _gpu: &Gpu,
        _enc: &mut wgpu::CommandEncoder,
        _view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        // Only reachable outside `PhaseList` or while disabled: without a snapshot there is
        // nothing correct to draw. A disabled phase frees what it built.
        if !self.core.is_running(PackId::Dreamy) && self.allocated() {
            self.release();
        }
    }

    fn record_scene(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        if !self.is_active() {
            return;
        }
        self.record_frame(gpu, enc, view, cam, scene);
    }
}

// ---------------------------------------------------------------------------------------- finish

/// `GlowUniform` in `dreamy_glow.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct GlowUniform {
    knee: [f32; 4],
}

/// `FinishUniform` in `dreamy_finish.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct FinishUniform {
    glow: [f32; 4],
    halation: [f32; 4],
    lens: [f32; 4],
    flags: [f32; 4],
    size: [f32; 4],
}

struct FinishPipelines {
    format: wgpu::TextureFormat,
    glow_layout: wgpu::BindGroupLayout,
    prefilter: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    finish_layout: wgpu::BindGroupLayout,
    finish: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    glow_uniform: wgpu::Buffer,
    finish_uniform: wgpu::Buffer,
    /// Bound in place of the glow while it is off.
    dummy_glow: wgpu::TextureView,
}

impl FinishPipelines {
    fn new(gpu: &Gpu, format: wgpu::TextureFormat) -> FinishPipelines {
        let device = &gpu.device;
        let glow_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaderpack/dreamy glow"),
            source: wgpu::ShaderSource::Wgsl(glow_source().into()),
        });
        let finish_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaderpack/dreamy finish"),
            source: wgpu::ShaderSource::Wgsl(finish_source().into()),
        });
        let glow_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shaderpack/dreamy glow"),
            entries: &[texture_entry(0, true), sampler_entry(1), uniform_entry(2)],
        });
        let glow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shaderpack/dreamy glow"),
            bind_group_layouts: &[Some(&glow_layout)],
            immediate_size: 0,
        });
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let glow = |entry, blend| {
            fullscreen_pipeline(
                gpu,
                &glow_module,
                &glow_pipeline_layout,
                entry,
                GLOW_FORMAT,
                blend,
            )
        };
        let prefilter = glow("fs_glow_prefilter", None);
        let down = glow("fs_glow_down", None);
        let up = glow("fs_glow_up", Some(additive));
        let finish_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shaderpack/dreamy finish"),
            entries: &[
                uniform_entry(0),
                texture_entry(1, true),
                texture_entry(2, true),
                sampler_entry(3),
                GradeLut::texture_entry(4),
                GradeLut::sampler_entry(5),
            ],
        });
        let finish_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shaderpack/dreamy finish"),
                bind_group_layouts: &[Some(&finish_layout)],
                immediate_size: 0,
            });
        let finish = fullscreen_pipeline(
            gpu,
            &finish_module,
            &finish_pipeline_layout,
            "fs_finish",
            format,
            None,
        );
        let buffer = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        FinishPipelines {
            format,
            glow_layout,
            prefilter,
            down,
            up,
            finish_layout,
            finish,
            sampler: linear_sampler(gpu, "shaderpack/dreamy finish"),
            glow_uniform: buffer(
                "shaderpack/dreamy glow",
                std::mem::size_of::<GlowUniform>() as u64,
            ),
            finish_uniform: buffer(
                "shaderpack/dreamy finish",
                std::mem::size_of::<FinishUniform>() as u64,
            ),
            dummy_glow: texture_2d(gpu, "shaderpack/dreamy no glow", (1, 1), GLOW_FORMAT),
        }
    }
}

/// The glow pyramid: `levels` targets at half, quarter, ... resolution, and the bind groups that
/// read each one (level `i`'s group samples level `i`).
struct GlowChain {
    full: (u32, u32),
    views: Vec<wgpu::TextureView>,
    groups: Vec<wgpu::BindGroup>,
    /// The prefilter's group, reading the scene snapshot it was built for.
    scene: wgpu::TextureView,
    scene_group: wgpu::BindGroup,
}

impl GlowChain {
    fn new(
        gpu: &Gpu,
        pipelines: &FinishPipelines,
        full: (u32, u32),
        levels: usize,
        scene: &wgpu::TextureView,
    ) -> GlowChain {
        let mut views = Vec::with_capacity(levels);
        let mut size = full;
        for _ in 0..levels {
            size = (size.0.div_ceil(2).max(1), size.1.div_ceil(2).max(1));
            views.push(texture_2d(gpu, "shaderpack/dreamy glow", size, GLOW_FORMAT));
        }
        let groups = views
            .iter()
            .map(|view| glow_group(gpu, pipelines, view))
            .collect();
        GlowChain {
            full,
            views,
            groups,
            scene: scene.clone(),
            scene_group: glow_group(gpu, pipelines, scene),
        }
    }
}

fn glow_group(gpu: &Gpu, pipelines: &FinishPipelines, view: &wgpu::TextureView) -> wgpu::BindGroup {
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shaderpack/dreamy glow"),
        layout: &pipelines.glow_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&pipelines.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: pipelines.glow_uniform.as_entire_binding(),
            },
        ],
    })
}

/// The finish composite's bind group and what it was built from.
struct FinishBinding {
    scene: wgpu::TextureView,
    glow: Option<wgpu::TextureView>,
    group: wgpu::BindGroup,
}

/// Glow, halation, the colour LUT, vignette, grain and chromatic aberration over the finished
/// world. See the [module docs](self).
pub struct DreamyFinishPhase {
    core: ShaderCore,
    settings: DreamySettings,
    /// `settings.finish_on()`, worked out when the settings change (it derives the grade's
    /// colour matrix), so a frame only reads a `bool`.
    finish_on: bool,
    /// The grade is not the identity (the frame takes the LUT lookup); likewise kept.
    grade_on: bool,
    pipelines: Option<FinishPipelines>,
    lut: Option<GradeLut>,
    chain: Option<GlowChain>,
    binding: Option<FinishBinding>,
    uploaded_glow: Option<GlowUniform>,
    uploaded_finish: Option<FinishUniform>,
    passes: u64,
}

impl std::fmt::Debug for DreamyFinishPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DreamyFinishPhase")
            .field("settings", &self.settings)
            .field("allocated", &self.allocated())
            .field("passes", &self.passes)
            .finish_non_exhaustive()
    }
}

impl DreamyFinishPhase {
    /// The phase's name in `PhaseList`, GPU timings and captures.
    pub const NAME: &'static str = "shaderpack/dreamy-finish";

    /// An idle phase. Allocates nothing.
    #[must_use]
    pub fn new(core: ShaderCore) -> DreamyFinishPhase {
        DreamyFinishPhase {
            core,
            settings: DreamySettings::default(),
            finish_on: false,
            grade_on: false,
            pipelines: None,
            lut: None,
            chain: None,
            binding: None,
            uploaded_glow: None,
            uploaded_finish: None,
            passes: 0,
        }
    }

    /// Set the pack's settings.
    pub fn set_settings(&mut self, settings: DreamySettings) {
        let glow_was = self.settings.glow_levels();
        self.settings = settings;
        self.finish_on = settings.finish_on();
        self.grade_on = !settings.grade.is_identity();
        if !self.glow_on() || settings.glow_levels() != glow_was {
            self.chain = None;
            self.binding = None;
        }
        if !self.finish_on {
            self.release();
        }
    }

    /// The settings in effect.
    #[must_use]
    pub fn settings(&self) -> DreamySettings {
        self.settings
    }

    fn glow_on(&self) -> bool {
        self.settings.glow > 0.0 || self.settings.halation > 0.0
    }

    /// Whether the phase draws this frame.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.core.is_running(PackId::Dreamy) && self.finish_on
    }

    /// Drop every GPU resource.
    pub fn release(&mut self) {
        self.pipelines = None;
        self.lut = None;
        self.chain = None;
        self.binding = None;
        self.uploaded_glow = None;
        self.uploaded_finish = None;
    }

    /// Whether any GPU resource exists (tests: off holds none).
    #[must_use]
    pub fn allocated(&self) -> bool {
        self.pipelines.is_some() || self.lut.is_some() || self.chain.is_some()
    }

    /// How many composites have been recorded.
    #[must_use]
    pub fn passes_recorded(&self) -> u64 {
        self.passes
    }

    /// How many LUT bakes have been recorded.
    #[must_use]
    pub fn lut_bakes(&self) -> u64 {
        self.lut.as_ref().map_or(0, GradeLut::bakes)
    }

    /// The LUT, once built (tests read it back).
    #[must_use]
    pub fn lut(&self) -> Option<&GradeLut> {
        self.lut.as_ref()
    }

    #[allow(clippy::too_many_lines)] // One frame's recording, in order.
    fn record_frame(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        scene: &SceneTextures<'_>,
    ) {
        let settings = self.settings;
        let frame = self.core.frame();
        let full = (scene.width, scene.height);
        if self
            .pipelines
            .as_ref()
            .is_none_or(|pipelines| pipelines.format != scene.format)
        {
            self.pipelines = Some(FinishPipelines::new(gpu, scene.format));
            self.chain = None;
            self.binding = None;
            self.uploaded_glow = None;
            self.uploaded_finish = None;
        }
        let pipelines = self.pipelines.as_ref().expect("built above");

        let lut = self.lut.get_or_insert_with(|| GradeLut::new(gpu));
        lut.set_params(settings.grade);
        lut.record_bake(gpu, enc);

        let glow_on = settings.glow > 0.0 || settings.halation > 0.0;
        let levels = settings.glow_levels();
        if glow_on {
            if self
                .chain
                .as_ref()
                .is_none_or(|chain| chain.full != full || chain.views.len() != levels)
            {
                self.chain = Some(GlowChain::new(gpu, pipelines, full, levels, scene.color));
                self.binding = None;
            }
            let chain = self.chain.as_mut().expect("built above");
            if chain.scene != *scene.color {
                chain.scene = scene.color.clone();
                chain.scene_group = glow_group(gpu, pipelines, scene.color);
            }
            // Dark pixels add 6 % of themselves, highlights all of themselves: the glow blooms
            // the bright sky and lit surfaces without lifting dark silhouettes (players) into
            // grey. (12 % and a lower knee lifted the midtones in round 1.)
            let glow_uniform = GlowUniform {
                knee: [0.06, 0.12, 0.75, 0.0],
            };
            if self.uploaded_glow != Some(glow_uniform) {
                gpu.write_buffer_staged(
                    &pipelines.glow_uniform,
                    0,
                    bytemuck::bytes_of(&glow_uniform),
                );
                self.uploaded_glow = Some(glow_uniform);
            }
            fullscreen_pass(
                enc,
                "shaderpack/dreamy glow prefilter",
                &chain.views[0],
                &pipelines.prefilter,
                &chain.scene_group,
            );
            for level in 1..levels {
                fullscreen_pass(
                    enc,
                    "shaderpack/dreamy glow down",
                    &chain.views[level],
                    &pipelines.down,
                    &chain.groups[level - 1],
                );
            }
            for level in (0..levels - 1).rev() {
                fullscreen_pass(
                    enc,
                    "shaderpack/dreamy glow up",
                    &chain.views[level],
                    &pipelines.up,
                    &chain.groups[level + 1],
                );
            }
        }

        let glow_view = if glow_on {
            self.chain.as_ref().map(|chain| chain.views[0].clone())
        } else {
            None
        };
        let fits = self
            .binding
            .as_ref()
            .is_some_and(|binding| binding.scene == *scene.color && binding.glow == glow_view);
        if !fits {
            let lut = self.lut.as_ref().expect("built above");
            let glow = glow_view.as_ref().unwrap_or(&pipelines.dummy_glow);
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shaderpack/dreamy finish"),
                layout: &pipelines.finish_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pipelines.finish_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(scene.color),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(glow),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&pipelines.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(lut.view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(lut.sampler()),
                    },
                ],
            });
            self.binding = Some(FinishBinding {
                scene: scene.color.clone(),
                glow: glow_view,
                group,
            });
        }

        let halation = to_linear([1.0, 0.42, 0.22]);
        // The pyramid's levels add up; the strength means the same whatever the radius.
        let grain_seed = if settings.grain > 0.0 {
            (frame.frame_index % 4096) as f32
        } else {
            0.0
        };
        let uniform = FinishUniform {
            glow: [
                settings.glow.clamp(0.0, 1.0) * 0.55,
                1.0 / levels as f32,
                settings.halation.clamp(0.0, 1.0) * 1.5,
                0.18,
            ],
            halation: [halation.x, halation.y, halation.z, 0.0],
            lens: [
                settings.vignette.clamp(0.0, 1.0) * 0.45,
                settings.grain.clamp(0.0, 1.0) * 0.06,
                settings.chromatic.clamp(0.0, 1.0) * 0.012,
                grain_seed,
            ],
            flags: [
                f32::from(u8::from(glow_on)),
                f32::from(u8::from(self.grade_on)),
                1.0,
                full.0 as f32 / full.1.max(1) as f32,
            ],
            size: [
                full.0 as f32,
                full.1 as f32,
                1.0 / full.0.max(1) as f32,
                1.0 / full.1.max(1) as f32,
            ],
        };
        if self.uploaded_finish != Some(uniform) {
            gpu.write_buffer_staged(&pipelines.finish_uniform, 0, bytemuck::bytes_of(&uniform));
            self.uploaded_finish = Some(uniform);
        }
        let group = &self.binding.as_ref().expect("built above").group;
        fullscreen_pass(enc, Self::NAME, view, &pipelines.finish, group);
        self.passes += 1;
    }
}

impl RenderPhase for DreamyFinishPhase {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn enabled(&self) -> bool {
        self.is_active()
    }

    fn needs_scene_color(&self) -> bool {
        self.is_active()
    }

    fn resize(&mut self, _gpu: &Gpu, _width: u32, _height: u32) {
        self.chain = None;
        self.binding = None;
    }

    fn record(
        &mut self,
        _gpu: &Gpu,
        _enc: &mut wgpu::CommandEncoder,
        _view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        if !self.core.is_running(PackId::Dreamy) && self.allocated() {
            self.release();
        }
    }

    fn record_scene(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        if !self.is_active() {
            return;
        }
        self.record_frame(gpu, enc, view, scene);
    }
}

/// The inverse view-projection the atmosphere uses, for tests and tools.
#[must_use]
pub fn inverse_view_proj(cam: &Camera) -> Mat4 {
    world_inverse_view_proj(cam)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(source: &str) {
        let module = wgpu::naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("validates");
    }

    #[test]
    fn every_dreamy_shader_parses_and_validates() {
        validate(&atmosphere_source());
        validate(&glow_source());
        validate(&finish_source());
    }

    #[test]
    fn the_uniforms_match_their_wgsl_structs() {
        assert_eq!(std::mem::size_of::<AtmosphereUniform>(), 64 + 9 * 16);
        assert_eq!(std::mem::size_of::<FinishUniform>(), 5 * 16);
        assert_eq!(std::mem::size_of::<GlowUniform>(), 16);
    }

    #[test]
    fn a_default_pack_has_nothing_to_draw_and_allocates_nothing() {
        let settings = DreamySettings::default();
        assert!(!settings.atmosphere_on());
        assert!(!settings.finish_on());
        let core = ShaderCore::new();
        let atmosphere = DreamyAtmospherePhase::new(core.clone());
        let finish = DreamyFinishPhase::new(core);
        assert!(!atmosphere.enabled() && !atmosphere.needs_scene_color());
        assert!(!finish.enabled() && !finish.needs_scene_color());
        assert!(!atmosphere.allocated() && !finish.allocated());
    }

    #[test]
    fn the_phases_wait_for_the_core_to_run_the_dreamy_pack() {
        let core = ShaderCore::new();
        let mut atmosphere = DreamyAtmospherePhase::new(core.clone());
        let mut finish = DreamyFinishPhase::new(core.clone());
        let settings = DreamySettings {
            haze: 0.5,
            glow: 0.3,
            ..DreamySettings::default()
        };
        atmosphere.set_settings(settings);
        finish.set_settings(settings);
        // Switched on in the settings, but no frame says a world is on screen.
        core.set_active(Some(PackId::Dreamy));
        assert!(!atmosphere.enabled() && !finish.enabled());
        // Another pack running never runs these.
        core.set_active(Some(PackId::Classic));
        assert!(!atmosphere.enabled() && !finish.enabled());
    }

    #[test]
    fn glow_levels_follow_the_radius() {
        let at = |radius| {
            DreamySettings {
                glow_radius: radius,
                ..DreamySettings::default()
            }
            .glow_levels()
        };
        assert_eq!(at(0.0), 4);
        assert_eq!(at(0.5), 5);
        assert_eq!(at(1.0), 6);
        assert_eq!(at(9.0), MAX_GLOW_LEVELS);
    }
}
