//! Anisotropic filtering for the block atlas: the render half
//! of the `anisotropic_filtering` mod.
//!
//! A child module of `terrain::phase` (declared there with `#[path]`), so it shares that file's
//! private vertex-offset layout and layer helpers instead of widening their visibility.
//!
//! # What changes when it is on
//!
//! [`TerrainRenderer::set_anisotropy`] with 2, 4, 8 or 16 makes the `SOLID` and `CUTOUT_MIPPED`
//! layers draw with a second set of pipelines whose fragment stage takes explicit anisotropic
//! taps (`aniso.wgsl`, appended to `terrain.wgsl`):
//!
//! * a magnified or 1:1 texel (the larger screen derivative of the texel coordinate is at most one
//!   texel per pixel) samples today's sampler at level 0, which is exactly today's texel;
//! * a minified one takes `N = min(ceil(Pmax / Pmin), level)` taps along the footprint's major axis
//!   at `LOD = log2(Pmax / N)`, each clamped inside the 2^L-aligned atlas cell of its sprite (the
//!   stitcher reserves every sprite on multiples of 2^L, L = mip levels), so no tap reads a
//!   neighbouring sprite.
//!
//! `TRANSLUCENT` and the non-mipped `CUTOUT` layer keep today's pipelines: vanilla gives `CUTOUT`
//! no mipmaps on purpose, and translucent state is still being worked on by another lane in this
//! wave (a duplicated pipeline would drift from it).
//!
//! # Off costs nothing
//!
//! Off (the default, level 0 or 1) draws with the original pipelines, bind groups, shader module
//! and sampler, untouched; the per-layer cost is one `bool` test. The second module, its two
//! pipelines, one linear sampler and a 32-byte uniform are built the first time a frame draws
//! with it on, and dropped when it is switched off. An atlas without mip levels (Mipmap Levels 0)
//! leaves it inactive.
//!
//! # Limits
//!
//! The clamp cell is 2^L texels. For a sprite larger than that (a 32x or 64x pack with 4 mip
//! levels) taps also stop at the cell edges inside the sprite: no colour bleeds, but filtering
//! there is a little weaker. A per-sprite rectangle table from the stitcher would lift that.

use super::{layer_index, ChunkOffset, TerrainOptions, TerrainRenderer, LAYER_COUNT};
use crate::terrain::vertex::{PackedTerrainVertex, TerrainVertex};
use crate::Gpu;
use mc_data::RenderLayer;

/// The levels the mod offers. Anything else is clamped to the nearest one at or below it.
pub const ANISOTROPY_LEVELS: [u32; 4] = [2, 4, 8, 16];

/// The AF module source as the GPU compiles it.
#[must_use]
pub fn shader_source() -> String {
    format!(
        "{}\n{}",
        include_str!("terrain.wgsl"),
        include_str!("aniso.wgsl")
    )
}

/// Whether AF draws this layer.
pub(super) fn applies_to(layer: usize) -> bool {
    layer == layer_index(RenderLayer::Solid) || layer == layer_index(RenderLayer::CutoutMipped)
}

/// The fragment entry point for a layer AF draws.
fn fragment_entry(layer: usize) -> &'static str {
    if layer == layer_index(RenderLayer::Solid) {
        "fs_solid_aniso"
    } else {
        "fs_cutout_aniso"
    }
}

/// `aniso.wgsl`'s `AnisoUniform`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct AnisoUniform {
    atlas: [f32; 4],
    cell: [f32; 4],
}

/// The GPU half, built on first use.
struct AnisoGpu {
    sample_count: u32,
    format: wgpu::TextureFormat,
    pipelines: [Option<wgpu::RenderPipeline>; LAYER_COUNT],
    group: wgpu::BindGroup,
    params: wgpu::Buffer,
    uploaded: Option<AnisoUniform>,
}

/// The requested level and, while it is on, the pipelines that draw it.
#[derive(Default)]
pub(super) struct AnisoState {
    /// Maximum anisotropy; 0 or 1 = off.
    level: u32,
    gpu: Option<AnisoGpu>,
}

impl AnisoState {
    /// The AF pipeline for `layer`, when AF is built and draws that layer.
    pub(super) fn pipeline(&self, layer: usize) -> Option<&wgpu::RenderPipeline> {
        self.gpu.as_ref()?.pipelines[layer].as_ref()
    }

    /// The group-2 bind group the AF pipelines read.
    pub(super) fn group(&self) -> Option<&wgpu::BindGroup> {
        self.gpu.as_ref().map(|g| &g.group)
    }
}

/// The level actually used for a requested one: 0 (off) below 2, else the largest offered level
/// not above it.
#[must_use]
pub fn effective_level(requested: u32) -> u32 {
    ANISOTROPY_LEVELS
        .iter()
        .copied()
        .filter(|&level| level <= requested)
        .max()
        .unwrap_or(0)
}

impl TerrainRenderer {
    /// Set the Anisotropic Filtering mod's level: 2, 4, 8 or 16 (other values round down to one of
    /// those; below 2 is off). Off frees the AF pipelines; the next frame with it on builds them.
    pub fn set_anisotropy(&mut self, level: u32) {
        let level = effective_level(level);
        if level != self.aniso.level {
            self.aniso.level = level;
            if level == 0 {
                self.aniso.gpu = None;
            }
        }
    }

    /// The level in effect (0 = off).
    #[must_use]
    pub fn anisotropy(&self) -> u32 {
        self.aniso.level
    }

    /// Whether frames draw with AF: a level is set and the atlas has mip levels.
    #[must_use]
    pub fn anisotropy_active(&self) -> bool {
        self.aniso.level >= 2 && self.atlas_mip_level_count > 1
    }

    /// Whether the AF pipelines exist (only after a frame drew with AF on).
    #[must_use]
    pub fn anisotropy_allocated(&self) -> bool {
        self.aniso.gpu.is_some()
    }

    /// Build what an AF frame needs and upload the uniform if it changed. Returns whether this
    /// frame draws with AF. Off: one comparison.
    pub(super) fn prepare_aniso(&mut self, gpu: &Gpu) -> bool {
        if !self.anisotropy_active() {
            return false;
        }
        if self
            .aniso
            .gpu
            .as_ref()
            .is_none_or(|g| g.sample_count != gpu.sample_count() || g.format != gpu.config.format)
        {
            self.aniso.gpu = Some(build(gpu, &self.group1_layout, self.options));
        }
        let top = (self.atlas_mip_level_count - 1) as f32;
        let uniform = AnisoUniform {
            atlas: [
                self.atlas_width as f32,
                self.atlas_height as f32,
                top,
                self.aniso.level as f32,
            ],
            cell: [top.exp2(), 0.0, 0.0, 0.0],
        };
        let built = self.aniso.gpu.as_mut().expect("built above");
        if built.uploaded != Some(uniform) {
            gpu.write_buffer_staged(&built.params, 0, bytemuck::bytes_of(&uniform));
            built.uploaded = Some(uniform);
        }
        true
    }
}

/// The AF module, its group-2 layout, sampler and uniform, and one pipeline per layer it draws,
/// each with exactly the original layer pipeline's state except the fragment entry point.
fn build(gpu: &Gpu, group1_layout: &wgpu::BindGroupLayout, options: TerrainOptions) -> AnisoGpu {
    let device = &gpu.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("terrain aniso"),
        source: wgpu::ShaderSource::Wgsl(shader_source().into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("terrain aniso group2"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    // The explicit taps want plain trilinear: the shader picks the LOD and the tap positions.
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("terrain aniso linear sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain aniso params"),
        size: std::mem::size_of::<AnisoUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("terrain aniso group2"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params.as_entire_binding(),
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("terrain aniso layout"),
        bind_group_layouts: &[
            Some(gpu.camera_layout()),
            Some(group1_layout),
            Some(&layout),
        ],
        immediate_size: 0,
    });
    let pipelines = std::array::from_fn(|i| {
        applies_to(i).then(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("terrain aniso"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(if options.packed_vertices {
                        "vs_packed"
                    } else {
                        "vs"
                    }),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[
                        Some(if options.packed_vertices {
                            PackedTerrainVertex::layout()
                        } else {
                            TerrainVertex::layout()
                        }),
                        Some(ChunkOffset::LAYOUT),
                    ],
                },
                // The opaque layers' state in `TerrainRenderer::new_with_options_and_anaglyph`:
                // back-face culling, depth write, LessEqual, no blend.
                primitive: wgpu::PrimitiveState {
                    cull_mode: options.cull_backfaces.then_some(wgpu::Face::Back),
                    ..wgpu::PrimitiveState::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: crate::DEPTH_FORMAT,
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
                    entry_point: Some(fragment_entry(i)),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        })
    });
    AnisoGpu {
        sample_count: gpu.sample_count(),
        format: gpu.config.format,
        pipelines,
        group,
        params,
        uploaded: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aniso_wgsl_validates_with_the_terrain_shader() {
        let source = shader_source();
        let module = wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
    }

    #[test]
    fn the_off_path_module_is_terrain_wgsl_unchanged() {
        // The AF module only appends: the original module (what the off path compiles) is the
        // prefix, byte for byte, and `terrain.wgsl` itself names no AF entry point.
        let original = include_str!("terrain.wgsl");
        assert!(shader_source().starts_with(original));
        assert!(!original.contains("aniso"));
    }

    #[test]
    fn levels_round_down_to_an_offered_one() {
        assert_eq!(effective_level(0), 0);
        assert_eq!(effective_level(1), 0);
        assert_eq!(effective_level(2), 2);
        assert_eq!(effective_level(3), 2);
        assert_eq!(effective_level(8), 8);
        assert_eq!(effective_level(15), 8);
        assert_eq!(effective_level(16), 16);
        assert_eq!(effective_level(64), 16);
    }

    #[test]
    fn only_the_mipmapped_opaque_layers_are_filtered() {
        assert!(applies_to(layer_index(RenderLayer::Solid)));
        assert!(applies_to(layer_index(RenderLayer::CutoutMipped)));
        assert!(!applies_to(layer_index(RenderLayer::Cutout)));
        assert!(!applies_to(layer_index(RenderLayer::Translucent)));
    }
}
