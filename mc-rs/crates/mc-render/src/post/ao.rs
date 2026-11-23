//! Screen-space ambient occlusion: the render half of the
//! `ambient_occlusion` mod.
//!
//! Records at [`crate::PhaseOrder::AMBIENT_OCCLUSION`]: after opaque terrain, entities and tile
//! entities, before the hitboxes (342) and the client's nameplates (345; both moved after AO, so it neither greys them nor reads label glyph depth), the selection outline,
//! particles, weather, translucent terrain, the screen-space fog and the first-person hand. So AO
//! darkens the opaque world only, the fog
//! recorded later fades it with distance like everything else, and the hand (which clears depth
//! and draws at [`crate::PhaseOrder::FIRST_PERSON`]) is never touched.
//!
//! # Passes
//!
//! 1. [`ViewGeometry`] prepass: planar distance + reconstructed view-space normals from
//!    [`Gpu::depth_texture`], at full or half resolution (reusable; see its module docs).
//! 2. `fs_gtao` (AO resolution): horizon-based visibility (GTAO):
//!    per slice, march both ways in screen space, keep each side's highest horizon (with a
//!    distance falloff so far-away geometry does not shade), clamp to the normal's hemisphere and
//!    integrate the cosine-weighted visible arc analytically. 4x4 ordered-dither rotation and
//!    step offsets.
//! 3. `fs_denoise` (AO resolution): a 4x4 box covering exactly one dither period, weighted by
//!    distance from the centre's tangent plane, so it never blurs across a silhouette.
//! 4. `fs_composite` (full resolution, into the frame): edge-aware upsample, strength curve,
//!    distance fade, multiply blend (`dst *= ao`). Pixels with no geometry (depth 1.0: sky,
//!    clouds) and pixels with nothing to darken `discard`, so they keep their exact bytes.
//!
//! The composite blends straight onto the frame target and the other passes read only depth, so
//! the phase never asks for [`crate::SceneTextures`]: turning AO on does not switch the frame to
//! the intermediate-target path.
//!
//! # Off costs nothing
//!
//! [`AoPhase::new`] allocates nothing. While the mod is off (or no world is on screen) the phase
//! reports `enabled() == false`, so [`crate::PhaseList`] never records it and the frame is byte
//! for byte the frame without it; the screen-sized targets are dropped the moment it turns off.
//! The pipelines (built on first use) and two small uniform buffers are kept.
//!
//! # MSAA
//!
//! Supported. The prepass and the composite read sample 0 of the multisampled depth buffer, and
//! the composite draws through [`Gpu::begin_pass`] with the frame's sample count, so the AO factor
//! multiplies every sample of a pixel before the resolve, like any other world phase.
//!
//! # Future: temporal accumulation
//!
//! Nothing here depends on the TAA lane. With a TAA resolve after this phase, rotating the dither
//! pattern each frame (`frame` index into `ao_bayer4`) would let TAA accumulate 16 rotations over
//! 16 frames, so `Low` would look like `High`; the denoise could then shrink to 2x2.

use crate::post::view_geometry::{self, ViewGeometry};
use crate::{Camera, DepthAttachment, Gpu, RenderPhase};

/// The search quality: slices (direction pairs) and steps per side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AoQuality {
    /// 1 slice x 3 steps per side.
    Low,
    /// 2 slices x 4 steps per side.
    #[default]
    Medium,
    /// 3 slices x 6 steps per side.
    High,
    /// 4 slices x 8 steps per side.
    Ultra,
}

impl AoQuality {
    /// Every quality, cheapest first: the menu's choice order.
    pub const ALL: [AoQuality; 4] = [
        AoQuality::Low,
        AoQuality::Medium,
        AoQuality::High,
        AoQuality::Ultra,
    ];

    /// The quality at a menu choice index (out of range clamps to `Ultra`).
    #[must_use]
    pub fn from_index(index: usize) -> AoQuality {
        Self::ALL[index.min(Self::ALL.len() - 1)]
    }

    /// Slice directions per pixel (each searched both ways).
    #[must_use]
    pub fn slices(self) -> u32 {
        match self {
            AoQuality::Low => 1,
            AoQuality::Medium => 2,
            AoQuality::High => 3,
            AoQuality::Ultra => 4,
        }
    }

    /// Depth samples per side of a slice.
    #[must_use]
    pub fn steps(self) -> u32 {
        match self {
            AoQuality::Low => 3,
            AoQuality::Medium => 4,
            AoQuality::High => 6,
            AoQuality::Ultra => 8,
        }
    }
}

/// What the mod's options say.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AoSettings {
    /// Search quality.
    pub quality: AoQuality,
    /// World-space radius in blocks.
    pub radius: f32,
    /// Darkness: the visibility is raised to this power (1 = 100 %). 0 turns the effect off.
    pub strength: f32,
    /// Compute at half resolution and upsample.
    pub half_resolution: bool,
}

impl Default for AoSettings {
    fn default() -> Self {
        AoSettings {
            quality: AoQuality::Medium,
            radius: 1.5,
            strength: 1.0,
            half_resolution: true,
        }
    }
}

impl AoSettings {
    /// The settings made safe to upload: a finite radius in `0.25..=8`, a finite strength in
    /// `0..=4`.
    #[must_use]
    pub fn sanitized(self) -> AoSettings {
        let finite_or = |value: f32, fallback: f32| {
            if value.is_finite() {
                value
            } else {
                fallback
            }
        };
        AoSettings {
            quality: self.quality,
            radius: finite_or(self.radius, 1.5).clamp(0.25, 8.0),
            strength: finite_or(self.strength, 1.0).clamp(0.0, 4.0),
            half_resolution: self.half_resolution,
        }
    }
}

/// Where the distance falloff starts, as a fraction of the radius: a sample this close shades
/// fully, one at the radius not at all.
pub const FALLOFF_START: f32 = 0.4;
/// The distance fade, as fractions of the render distance (`far / sqrt 2`): full AO up to the
/// start, none from the end on.
pub const FADE: (f32, f32) = (0.6, 1.0);
/// The largest screen-space search radius, as a fraction of the full-resolution frame height.
/// Standing against a wall, a 1.5-block radius would otherwise span the whole screen.
pub const MAX_RADIUS_OF_HEIGHT: f32 = 0.2;

/// The GTAO pass's output: `r` = visibility, `g` = the same estimator's unoccluded value, so the
/// denoise can normalise (see `ao.wgsl`). Half precision keeps estimates above 1.
pub const RAW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
/// The denoised, normalised visibility the composite upsamples.
pub const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

/// `ao.wgsl`'s `AoParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    grid: [f32; 4],
    gtao: [f32; 4],
    look: [f32; 4],
}

impl Params {
    fn new(settings: AoSettings, ao_size: (u32, u32), scale: u32, frame_height: u32) -> Params {
        Params {
            grid: [
                ao_size.0 as f32,
                ao_size.1 as f32,
                scale as f32,
                settings.quality.slices() as f32,
            ],
            gtao: [
                settings.radius,
                settings.quality.steps() as f32,
                FALLOFF_START,
                (MAX_RADIUS_OF_HEIGHT * frame_height as f32 / scale as f32).max(2.0),
            ],
            look: [settings.strength, FADE.0, FADE.1, 0.25 * settings.radius],
        }
    }
}

/// The AO module source as the GPU compiles it: the view-geometry helpers plus `ao.wgsl`.
#[must_use]
pub fn shader_source(sample_count: u32) -> String {
    view_geometry::depth_spelling(
        &format!(
            "{}\n{}",
            view_geometry::COMMON_WGSL,
            include_str!("ao.wgsl")
        ),
        sample_count,
    )
}

/// The three AO pipelines, their layouts and the uniform, for one frame format and sample count.
struct Pipelines {
    format: wgpu::TextureFormat,
    sample_count: u32,
    gtao_layout: wgpu::BindGroupLayout,
    denoise_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    gtao: wgpu::RenderPipeline,
    denoise: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    params: wgpu::Buffer,
}

impl Pipelines {
    fn new(gpu: &Gpu) -> Pipelines {
        let device = &gpu.device;
        let sample_count = gpu.sample_count();
        let format = gpu.config.format;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post/ao"),
            source: wgpu::ShaderSource::Wgsl(shader_source(sample_count).into()),
        });
        let texture = view_geometry::load_texture_entry;
        let uniform = view_geometry::uniform_entry;
        let layout = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            })
        };
        let gtao_layout = layout("post/ao gtao", &[texture(0), texture(1), uniform(2)]);
        let denoise_layout = layout(
            "post/ao denoise",
            &[texture(0), texture(1), uniform(2), texture(3)],
        );
        let composite_layout = layout(
            "post/ao composite",
            &[
                texture(0),
                uniform(2),
                texture(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: sample_count > 1,
                    },
                    count: None,
                },
            ],
        );
        let build = |entry: &str,
                     group: &wgpu::BindGroupLayout,
                     target: wgpu::ColorTargetState,
                     samples: u32| {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: &[Some(gpu.camera_layout()), Some(group)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: samples,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(target)],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let single = |format| wgpu::ColorTargetState {
            format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        };
        let gtao = build("fs_gtao", &gtao_layout, single(RAW_FORMAT), 1);
        let denoise = build("fs_denoise", &denoise_layout, single(AO_FORMAT), 1);
        let composite = build(
            "fs_composite",
            &composite_layout,
            wgpu::ColorTargetState {
                format,
                blend: Some(MULTIPLY),
                write_mask: wgpu::ColorWrites::ALL,
            },
            sample_count,
        );
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post/ao params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Pipelines {
            format,
            sample_count,
            gtao_layout,
            denoise_layout,
            composite_layout,
            gtao,
            denoise,
            composite,
            params,
        }
    }
}

/// `dst.rgb *= src.rgb`, alpha kept: the composite's blend.
pub const MULTIPLY: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Dst,
        dst_factor: wgpu::BlendFactor::Zero,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

/// The raw and denoised AO targets and the two bind groups that read the view geometry.
struct Targets {
    /// [`ViewGeometry::generation`] these were built against.
    geometry_generation: u64,
    size: (u32, u32),
    raw: wgpu::TextureView,
    clean: wgpu::TextureView,
    gtao_group: wgpu::BindGroup,
    denoise_group: wgpu::BindGroup,
}

/// The composite bind group, cached against the frame depth texture and the denoised target.
struct CompositeBinding {
    depth: wgpu::Texture,
    clean: wgpu::TextureView,
    group: wgpu::BindGroup,
}

/// Screen-space ambient occlusion over the opaque world. See the [module docs](self).
pub struct AoPhase {
    settings: AoSettings,
    on: bool,
    in_world: bool,
    geometry: ViewGeometry,
    pipelines: Option<Pipelines>,
    targets: Option<Targets>,
    composite: Option<CompositeBinding>,
    uploaded: Option<Params>,
    passes: u64,
}

impl std::fmt::Debug for AoPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AoPhase")
            .field("settings", &self.settings)
            .field("on", &self.on)
            .field("in_world", &self.in_world)
            .field("allocated", &self.allocated())
            .field("passes", &self.passes)
            .finish_non_exhaustive()
    }
}

impl Default for AoPhase {
    fn default() -> Self {
        Self::new()
    }
}

impl AoPhase {
    /// The phase's name in `PhaseList`, GPU timings and captures.
    pub const NAME: &'static str = "post/ao";

    /// An idle phase: off, not in a world, default settings. Allocates nothing on the GPU.
    #[must_use]
    pub fn new() -> AoPhase {
        AoPhase {
            settings: AoSettings::default(),
            on: false,
            in_world: false,
            geometry: ViewGeometry::new(2),
            pipelines: None,
            targets: None,
            composite: None,
            uploaded: None,
            passes: 0,
        }
    }

    /// Turn the effect on or off (the mod's switch). Off drops the screen-sized targets.
    pub fn set_enabled(&mut self, on: bool) {
        if on != self.on {
            self.on = on;
            self.release_unused();
        }
    }

    /// Whether the mod's switch is on.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.on
    }

    /// Apply the mod's options.
    pub fn set_settings(&mut self, settings: AoSettings) {
        let settings = settings.sanitized();
        if settings != self.settings {
            self.settings = settings;
            self.geometry
                .set_scale(if settings.half_resolution { 2 } else { 1 });
            self.release_unused();
        }
    }

    /// The options in effect.
    #[must_use]
    pub fn settings(&self) -> AoSettings {
        self.settings
    }

    /// Whether a world is on screen. Outside one (menus, loading screens) the phase is off.
    pub fn set_in_world(&mut self, in_world: bool) {
        if in_world != self.in_world {
            self.in_world = in_world;
            self.release_unused();
        }
    }

    /// Whether the phase records this frame.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.on && self.in_world && self.settings.strength > 0.0
    }

    /// Whether any screen-sized target exists (only while active).
    #[must_use]
    pub fn allocated(&self) -> bool {
        self.targets.is_some() || self.geometry.allocated()
    }

    /// How many frames this phase has recorded. Stays put while it is off.
    #[must_use]
    pub fn passes_recorded(&self) -> u64 {
        self.passes
    }

    /// The reusable distance/normal prepass this phase owns (for a later pass to share).
    #[must_use]
    pub fn view_geometry(&self) -> &ViewGeometry {
        &self.geometry
    }

    fn release_unused(&mut self) {
        if !self.is_active() {
            self.geometry.release();
            self.targets = None;
            self.composite = None;
        }
    }

    /// Build or rebuild what this frame needs. Returns `false` when there is nothing to draw with.
    fn prepare(&mut self, gpu: &Gpu) -> bool {
        if self
            .pipelines
            .as_ref()
            .is_none_or(|p| p.format != gpu.config.format || p.sample_count != gpu.sample_count())
        {
            self.pipelines = Some(Pipelines::new(gpu));
            self.targets = None;
            self.composite = None;
            self.uploaded = None;
        }
        self.geometry.prepare(gpu);
        let (Some(size), Some(distance), Some(normal)) = (
            self.geometry.size(),
            self.geometry.distance_view(),
            self.geometry.normal_view(),
        ) else {
            return false;
        };
        let pipelines = self.pipelines.as_ref().expect("built above");
        let generation = self.geometry.generation();
        if self
            .targets
            .as_ref()
            .is_none_or(|t| t.geometry_generation != generation || t.size != size)
        {
            let raw = view_geometry::target_view(gpu, "post/ao raw", RAW_FORMAT, size.0, size.1);
            let clean =
                view_geometry::target_view(gpu, "post/ao denoised", AO_FORMAT, size.0, size.1);
            let view = wgpu::BindingResource::TextureView;
            let gtao_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post/ao gtao"),
                layout: &pipelines.gtao_layout,
                entries: &[
                    entry(0, view(distance)),
                    entry(1, view(normal)),
                    entry(2, pipelines.params.as_entire_binding()),
                ],
            });
            let denoise_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post/ao denoise"),
                layout: &pipelines.denoise_layout,
                entries: &[
                    entry(0, view(distance)),
                    entry(1, view(normal)),
                    entry(2, pipelines.params.as_entire_binding()),
                    entry(3, view(&raw)),
                ],
            });
            self.targets = Some(Targets {
                geometry_generation: generation,
                size,
                raw,
                clean,
                gtao_group,
                denoise_group,
            });
            self.composite = None;
        }
        let targets = self.targets.as_ref().expect("built above");
        if self
            .composite
            .as_ref()
            .is_none_or(|c| c.depth != *gpu.depth_texture() || c.clean != targets.clean)
        {
            let depth_view = gpu
                .depth_texture()
                .create_view(&wgpu::TextureViewDescriptor::default());
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post/ao composite"),
                layout: &pipelines.composite_layout,
                entries: &[
                    entry(0, wgpu::BindingResource::TextureView(distance)),
                    entry(2, pipelines.params.as_entire_binding()),
                    entry(4, wgpu::BindingResource::TextureView(&targets.clean)),
                    entry(5, wgpu::BindingResource::TextureView(&depth_view)),
                ],
            });
            self.composite = Some(CompositeBinding {
                depth: gpu.depth_texture().clone(),
                clean: targets.clean.clone(),
                group,
            });
        }
        let params = Params::new(
            self.settings,
            size,
            self.geometry.scale(),
            gpu.config.height.max(1),
        );
        if self.uploaded != Some(params) {
            gpu.write_buffer_staged(&pipelines.params, 0, bytemuck::bytes_of(&params));
            self.uploaded = Some(params);
        }
        true
    }
}

fn entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

impl RenderPhase for AoPhase {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn enabled(&self) -> bool {
        self.is_active()
    }

    fn resize(&mut self, _gpu: &Gpu, _width: u32, _height: u32) {
        // `prepare` notices the new frame size and depth texture itself; drop the old targets now
        // so a minimised window does not keep them.
        self.geometry.release();
        self.targets = None;
        self.composite = None;
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        if !self.is_active() || !self.prepare(gpu) {
            return;
        }
        self.geometry.record(gpu, enc);
        let (Some(pipelines), Some(targets), Some(composite)) =
            (&self.pipelines, &self.targets, &self.composite)
        else {
            return;
        };
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post/ao gtao"),
                color_attachments: &[Some(view_geometry::overwrite(&targets.raw))],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipelines.gtao);
            pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
            pass.set_bind_group(1, &targets.gtao_group, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post/ao denoise"),
                color_attachments: &[Some(view_geometry::overwrite(&targets.clean))],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipelines.denoise);
            pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
            pass.set_bind_group(1, &targets.denoise_group, &[]);
            pass.draw(0..3, 0..1);
        }
        let mut pass = gpu.begin_pass(enc, view, Self::NAME, DepthAttachment::None);
        pass.set_pipeline(&pipelines.composite);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.set_bind_group(1, &composite.group, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        self.passes += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ao_wgsl_validates_single_and_multisampled() {
        for samples in [1, 4] {
            let source = shader_source(samples);
            let module = wgpu::naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
        }
    }

    #[test]
    fn the_phase_sits_between_the_opaque_world_and_everything_translucent() {
        use crate::PhaseOrder;
        let order = PhaseOrder::AMBIENT_OCCLUSION;
        assert!(order > PhaseOrder::ENTITIES);
        assert!(order < PhaseOrder::BLOCK_OVERLAY);
        assert!(order < PhaseOrder::PARTICLES);
        assert!(order < PhaseOrder::TERRAIN_TRANSLUCENT);
        assert!(order < crate::sky::fog::FogPhase::ORDER);
        assert!(order < PhaseOrder::FIRST_PERSON);
    }

    #[test]
    fn a_new_phase_is_off_and_owns_nothing() {
        let phase = AoPhase::new();
        assert!(!phase.enabled());
        assert!(!phase.needs_scene_color());
        assert!(!phase.allocated());
        assert_eq!(phase.passes_recorded(), 0);
    }

    #[test]
    fn it_runs_only_when_switched_on_in_a_world_with_a_strength() {
        let mut phase = AoPhase::new();
        phase.set_enabled(true);
        assert!(!phase.is_active(), "no world yet");
        phase.set_in_world(true);
        assert!(phase.is_active());
        assert!(
            !phase.needs_scene_color(),
            "AO never switches on scene mode"
        );
        phase.set_settings(AoSettings {
            strength: 0.0,
            ..AoSettings::default()
        });
        assert!(!phase.is_active(), "0 % strength draws nothing");
        phase.set_settings(AoSettings::default());
        phase.set_enabled(false);
        assert!(!phase.is_active());
    }

    #[test]
    fn settings_are_sanitized() {
        let s = AoSettings {
            quality: AoQuality::High,
            radius: f32::NAN,
            strength: 99.0,
            half_resolution: false,
        }
        .sanitized();
        assert_eq!(s.radius, 1.5);
        assert_eq!(s.strength, 4.0);
        assert_eq!(AoQuality::from_index(9), AoQuality::Ultra);
        assert_eq!(AoQuality::from_index(1), AoQuality::Medium);
    }

    #[test]
    fn the_params_carry_the_quality_and_scale() {
        let p = Params::new(
            AoSettings {
                quality: AoQuality::High,
                radius: 2.0,
                strength: 1.5,
                half_resolution: true,
            },
            (960, 540),
            2,
            1080,
        );
        assert_eq!(p.grid, [960.0, 540.0, 2.0, 3.0]);
        assert_eq!(p.gtao[0], 2.0);
        assert_eq!(p.gtao[1], 6.0);
        assert_eq!(p.gtao[3], 0.2 * 1080.0 / 2.0);
        assert_eq!(p.look[0], 1.5);
    }
}
