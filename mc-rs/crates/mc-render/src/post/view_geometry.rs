//! View-space geometry from the shared depth buffer: planar distance and normals.
//!
//! Every screen-space effect that reasons about surfaces (ambient occlusion today; contact
//! shadows, screen-space reflections, depth of field, a shader pack's lighting later) needs the
//! same two inputs, so they are built here once, as named targets, instead of inside one effect:
//!
//! | target | format | contents |
//! |---|---|---|
//! | [`ViewGeometry::distance_view`] | [`DISTANCE_FORMAT`] (`R32Float`) | planar eye distance `-z_view` in blocks; [`SKY_DISTANCE`] where nothing wrote depth |
//! | [`ViewGeometry::normal_view`] | [`NORMAL_FORMAT`] (`Rgba16Float`) | view-space unit normal facing the camera (`xyz`), `w` = 1 for geometry, 0 for sky |
//!
//! Both live at full resolution or at half ([`ViewGeometry::new`]'s `scale`); at half, texel
//! `(x, y)` is full-resolution pixel `(2x, 2y)`: one real depth sample, not an average, so the
//! distance is a surface that exists and the normal belongs to it. Normals are reconstructed
//! from full-resolution neighbours with a per-axis side choice (the side whose two-tap
//! extrapolation continues the centre's plane), so silhouettes and creases do not smear them.
//!
//! View space is the renderer's: right-handed, camera at the origin looking down -Z, rotation-only
//! (`CameraUniform::view`), so a normal here is `view * world_normal`.
//!
//! # Using it from another pass
//!
//! * Prepend [`COMMON_WGSL`] to your shader: it declares the camera uniform at `@group(0)`, a
//!   full-screen `vs`, `vg_linear_depth(d)` and `vg_view_position(full_px, dist)`.
//! * Own a `ViewGeometry`, call [`ViewGeometry::record`] in your phase before your own passes
//!   (it reads [`Gpu::depth_texture`], so record it after everything whose depth you want), and
//!   bind [`ViewGeometry::distance_view`] / [`ViewGeometry::normal_view`] with `textureLoad`
//!   (`R32Float` is not filterable).
//! * Rebuild your bind groups when [`ViewGeometry::generation`] moves (a resize, a scale change or
//!   a release reallocated the targets).
//!
//! Nothing here exists until the first [`ViewGeometry::record`]; [`ViewGeometry::release`] drops
//! the targets again (the pipeline is small and kept).

use crate::Gpu;

/// Format of [`ViewGeometry::distance_view`].
pub const DISTANCE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
/// Format of [`ViewGeometry::normal_view`].
pub const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The distance stored for a pixel where nothing drew (sky, clouds, void). Mirrors
/// `VG_SKY_DISTANCE` in [`COMMON_WGSL`].
pub const SKY_DISTANCE: f32 = 1.0e6;

/// Camera uniform, full-screen `vs` and the depth/position helpers. Prepend to a pass's WGSL.
pub const COMMON_WGSL: &str = include_str!("view_geometry_common.wgsl");
/// The prepass itself, appended to [`COMMON_WGSL`].
const PREPASS_WGSL: &str = include_str!("view_geometry.wgsl");

/// `source` with every `texture_depth_2d` spelled `texture_depth_multisampled_2d` when the frame
/// is multisampled: a multisampled depth buffer cannot be bound as the single-sampled type (see
/// [`Gpu::depth_texture`]). `textureLoad`'s third argument is then the sample index, and the
/// shaders here pass 0, which is right for both.
#[must_use]
pub fn depth_spelling(source: &str, sample_count: u32) -> String {
    if sample_count > 1 {
        source.replace("texture_depth_2d", "texture_depth_multisampled_2d")
    } else {
        source.to_owned()
    }
}

/// The prepass source (common helpers + prepass), as the GPU compiles it.
#[must_use]
pub fn prepass_source(sample_count: u32) -> String {
    depth_spelling(&format!("{COMMON_WGSL}\n{PREPASS_WGSL}"), sample_count)
}

/// The prepass pipeline and its uniform, built for one MSAA sample count.
struct Prepass {
    sample_count: u32,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    params: wgpu::Buffer,
    uploaded: Option<[f32; 4]>,
}

impl Prepass {
    fn new(gpu: &Gpu) -> Prepass {
        let sample_count = gpu.sample_count();
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post/view-geometry"),
            source: wgpu::ShaderSource::Wgsl(prepass_source(sample_count).into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post/view-geometry group1"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: sample_count > 1,
                    },
                    count: None,
                },
                uniform_entry(1),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post/view-geometry layout"),
            bind_group_layouts: &[Some(gpu.camera_layout()), Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("post/view-geometry"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_prepass"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: DISTANCE_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: NORMAL_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post/view-geometry params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Prepass {
            sample_count,
            layout,
            pipeline,
            params,
            uploaded: None,
        }
    }
}

/// A uniform-buffer layout entry visible to the fragment stage.
pub(crate) fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

/// An unfilterable float texture entry (`textureLoad` only) visible to the fragment stage.
pub(crate) fn load_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// A screen-sized target the passes render into and later passes `textureLoad`.
pub(crate) fn target_view(
    gpu: &Gpu,
    label: &str,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> wgpu::TextureView {
    gpu.device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
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

/// The output size for a frame of `width` x `height` at `scale` (1 or 2), never zero.
#[must_use]
pub fn scaled_size(width: u32, height: u32, scale: u32) -> (u32, u32) {
    let scale = scale.max(1);
    (
        width.max(1).div_ceil(scale).max(1),
        height.max(1).div_ceil(scale).max(1),
    )
}

/// The two targets at one size.
struct Targets {
    width: u32,
    height: u32,
    frame: (u32, u32),
    distance: wgpu::TextureView,
    normal: wgpu::TextureView,
}

/// The prepass bind group, cached against the depth texture it reads.
struct DepthBinding {
    /// Compared by identity every frame: [`Gpu::resize`] and [`Gpu::apply_options`] replace the
    /// depth texture. Holding the handle also keeps a stale one from being mistaken for a new one.
    depth: wgpu::Texture,
    group: wgpu::BindGroup,
}

/// Planar distance and view-space normals at full or half resolution. See the [module docs](self).
pub struct ViewGeometry {
    scale: u32,
    prepass: Option<Prepass>,
    targets: Option<Targets>,
    binding: Option<DepthBinding>,
    generation: u64,
}

impl std::fmt::Debug for ViewGeometry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ViewGeometry")
            .field("scale", &self.scale)
            .field("size", &self.size())
            .field("generation", &self.generation)
            .finish_non_exhaustive()
    }
}

impl ViewGeometry {
    /// An empty instance at `scale` full-resolution pixels per texel (1 = full, 2 = half; other
    /// values are clamped into that range). Allocates nothing.
    #[must_use]
    pub fn new(scale: u32) -> ViewGeometry {
        ViewGeometry {
            scale: scale.clamp(1, 2),
            prepass: None,
            targets: None,
            binding: None,
            generation: 0,
        }
    }

    /// Full-resolution pixels per texel: 1 or 2.
    #[must_use]
    pub fn scale(&self) -> u32 {
        self.scale
    }

    /// Change the scale. The targets are rebuilt on the next [`Self::record`].
    pub fn set_scale(&mut self, scale: u32) {
        let scale = scale.clamp(1, 2);
        if scale != self.scale {
            self.scale = scale;
            self.release();
        }
    }

    /// The targets' size, or `None` before the first [`Self::record`] (and after a release).
    #[must_use]
    pub fn size(&self) -> Option<(u32, u32)> {
        self.targets.as_ref().map(|t| (t.width, t.height))
    }

    /// Bumped every time the target views are replaced. Dependent bind groups compare it.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The planar-distance target. `None` before the first [`Self::record`].
    #[must_use]
    pub fn distance_view(&self) -> Option<&wgpu::TextureView> {
        self.targets.as_ref().map(|t| &t.distance)
    }

    /// The view-space normal target. `None` before the first [`Self::record`].
    #[must_use]
    pub fn normal_view(&self) -> Option<&wgpu::TextureView> {
        self.targets.as_ref().map(|t| &t.normal)
    }

    /// Whether the targets exist (for tests and the "off allocates nothing" checks).
    #[must_use]
    pub fn allocated(&self) -> bool {
        self.targets.is_some()
    }

    /// Drop the targets and the depth binding. The pipeline is kept.
    pub fn release(&mut self) {
        if self.targets.take().is_some() {
            self.generation += 1;
        }
        self.binding = None;
    }

    /// Build or rebuild whatever the current frame needs: the pipeline for this sample count,
    /// targets of this frame's size, the bind group for this depth texture.
    pub fn prepare(&mut self, gpu: &Gpu) {
        if self
            .prepass
            .as_ref()
            .is_none_or(|p| p.sample_count != gpu.sample_count())
        {
            self.prepass = Some(Prepass::new(gpu));
            self.binding = None;
        }
        let frame = (gpu.config.width.max(1), gpu.config.height.max(1));
        if self.targets.as_ref().is_none_or(|t| t.frame != frame) {
            let (width, height) = scaled_size(frame.0, frame.1, self.scale);
            self.targets = Some(Targets {
                width,
                height,
                frame,
                distance: target_view(
                    gpu,
                    "post/view-geometry distance",
                    DISTANCE_FORMAT,
                    width,
                    height,
                ),
                normal: target_view(
                    gpu,
                    "post/view-geometry normal",
                    NORMAL_FORMAT,
                    width,
                    height,
                ),
            });
            self.generation += 1;
        }
        let prepass = self.prepass.as_mut().expect("built above");
        let targets = self.targets.as_ref().expect("built above");
        let params = [
            targets.width as f32,
            targets.height as f32,
            self.scale as f32,
            0.0,
        ];
        if prepass.uploaded != Some(params) {
            gpu.write_buffer_staged(&prepass.params, 0, bytemuck::cast_slice(&params));
            prepass.uploaded = Some(params);
        }
        if self
            .binding
            .as_ref()
            .is_none_or(|b| b.depth != *gpu.depth_texture())
        {
            let depth_view = gpu
                .depth_texture()
                .create_view(&wgpu::TextureViewDescriptor::default());
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post/view-geometry group1"),
                layout: &prepass.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: prepass.params.as_entire_binding(),
                    },
                ],
            });
            self.binding = Some(DepthBinding {
                depth: gpu.depth_texture().clone(),
                group,
            });
        }
    }

    /// Run the prepass: read the shared depth buffer as it stands now and fill both targets.
    /// Calls [`Self::prepare`] first.
    pub fn record(&mut self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder) {
        self.prepare(gpu);
        let (Some(prepass), Some(targets), Some(binding)) =
            (&self.prepass, &self.targets, &self.binding)
        else {
            return;
        };
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("post/view-geometry"),
            color_attachments: &[
                Some(overwrite(&targets.distance)),
                Some(overwrite(&targets.normal)),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&prepass.pipeline);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.set_bind_group(1, &binding.group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// A colour attachment the pass overwrites completely (every texel is written by the full-screen
/// triangle), so its old contents are never read.
pub(crate) fn overwrite(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
            store: wgpu::StoreOp::Store,
        },
    }
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
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
    }

    /// `cargo check` never parses WGSL; this runs the naga front end and validator wgpu runs at
    /// pipeline creation, for both depth spellings.
    #[test]
    fn prepass_wgsl_validates_single_and_multisampled() {
        validate(&prepass_source(1));
        validate(&prepass_source(4));
        assert!(prepass_source(1).contains("var vg_depth: texture_depth_2d;"));
        assert!(prepass_source(4).contains("var vg_depth: texture_depth_multisampled_2d;"));
    }

    #[test]
    fn half_resolution_rounds_up_and_never_reaches_zero() {
        assert_eq!(scaled_size(1920, 1080, 2), (960, 540));
        assert_eq!(scaled_size(1921, 1081, 2), (961, 541));
        assert_eq!(scaled_size(1920, 1080, 1), (1920, 1080));
        assert_eq!(scaled_size(1, 1, 2), (1, 1));
        assert_eq!(scaled_size(0, 0, 2), (1, 1));
    }

    #[test]
    fn the_rust_sky_constant_matches_the_shader() {
        assert!(COMMON_WGSL.contains("const VG_SKY_DISTANCE: f32 = 1000000.0;"));
        assert_eq!(SKY_DISTANCE, 1_000_000.0);
    }
}
