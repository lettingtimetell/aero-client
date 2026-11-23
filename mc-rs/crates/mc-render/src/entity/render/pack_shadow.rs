//! Shader packs: entities casting sun shadows.
//!
//! A child module of `entity::render`, so it reads this frame's batches, vertex buffer, quad
//! indices and texture bind groups without widening their visibility.
//!
//! [`EntityPhase::record_pack_shadow`] draws the depth of every opaque or alpha-tested entity
//! batch (bodies, armour, items on the ground; never glints, flashes, contact shadows, the
//! invisible-teammate ghost or beams) from the light into a shadow map the core prepared (a copy
//! of the cached terrain map, `shaderpack::shadow`). Entity vertices are camera-relative, which
//! is exactly what the shadow uniform's `receiver` matrix takes. Entities only *cast*: they keep
//! vanilla's lighting, so a player standing in shade is never darkened.
//!
//! Nothing here runs or allocates unless a shader pack has entity shadows on: the pipeline is
//! built on the first call, and [`EntityPhase::record`] never looks at it.

use super::{EntityPhase, ENTITY_VERTEX_LAYOUT};
use crate::entity::layers::LayerBlend;
use crate::shaderpack::shadow::SHADOW_FORMAT;
use crate::shaderpack::COMMON_WGSL;
use crate::Gpu;

const SHADOW_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> shadow: ShadowUniform;
@group(1) @binding(0) var entity_tex: texture_2d<f32>;
@group(1) @binding(1) var entity_samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) alpha: f32,
};

@vertex
fn vs(
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    out.clip = shadow.receiver * vec4<f32>(pos, 1.0);
    out.uv = uv;
    out.alpha = color.a;
    return out;
}

// The entity pass's own cutout (`alphaFunc(GL_GREATER, 0.1)`), against texel x vertex alpha.
@fragment
fn fs(in: VsOut) {
    let a = textureSampleLevel(entity_tex, entity_samp, in.uv, 0.0).a * in.alpha;
    if (a <= 0.1) {
        discard;
    }
}
"#;

/// The entity shadow pipeline and its uniform binding.
pub(super) struct EntityShadowGpu {
    layout0: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    /// Group 0 and the shadow uniform buffer it was built for.
    group0: Option<(wgpu::Buffer, wgpu::BindGroup)>,
}

/// Whether a batch's geometry casts a shadow: everything opaque or alpha-tested with depth
/// writes; nothing additive, translucent, glinting, flashing or beam-like.
fn casts(blend: LayerBlend) -> bool {
    matches!(
        blend,
        LayerBlend::Opaque
            | LayerBlend::OpaqueUnlit
            | LayerBlend::OpaqueUnlitCulled
            | LayerBlend::TranslucentDepthWrite
            | LayerBlend::DroppedItem
            | LayerBlend::LivingHurt
    )
}

fn build(gpu: &Gpu, texture_layout: &wgpu::BindGroupLayout) -> EntityShadowGpu {
    let device = &gpu.device;
    let source = format!("{COMMON_WGSL}\n{SHADOW_WGSL}");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("entity pack shadow"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("entity pack shadow group0"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("entity pack shadow"),
        bind_group_layouts: &[Some(&layout0), Some(texture_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("entity pack shadow"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(ENTITY_VERTEX_LAYOUT)],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: SHADOW_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 1.75,
                clamp: 0.0,
            },
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[],
        }),
        multiview_mask: None,
        cache: None,
    });
    EntityShadowGpu {
        layout0,
        pipeline,
        group0: None,
    }
}

impl EntityPhase {
    /// Draw this frame's shadow-casting entity geometry into `target` (a depth map that already
    /// holds the terrain's shadow casters), from the light, with `shadow_uniform`'s receiver
    /// matrix. Returns how many batches were drawn.
    pub fn record_pack_shadow(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        shadow_uniform: &wgpu::Buffer,
    ) -> usize {
        let Some(buffer) = self.buffer.as_ref() else {
            return 0;
        };
        if self.uploaded == 0 || !self.batches.iter().any(|b| casts(b.blend)) {
            return 0;
        }
        let built = self
            .pack_shadow
            .get_or_insert_with(|| build(gpu, &self.texture_layout));
        if built
            .group0
            .as_ref()
            .is_none_or(|(b, _)| b != shadow_uniform)
        {
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("entity pack shadow group0"),
                layout: &built.layout0,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shadow_uniform.as_entire_binding(),
                }],
            });
            built.group0 = Some((shadow_uniform.clone(), group));
        }
        // Handles (reference-counted) so the batch walk below can borrow `self` again.
        let group0 = built.group0.as_ref().expect("built above").1.clone();
        let pipeline = built.pipeline.clone();
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shaderpack/shadow entities"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: target,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group0, &[]);
        pass.set_vertex_buffer(0, buffer.slice(..));
        if let Some(indices) = &self.quad_indices {
            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
        }
        let fallback = self.texture_entry(&self.fallback);
        let mut bound: Option<*const wgpu::BindGroup> = None;
        let mut drawn = 0;
        for batch in &self.batches {
            if !casts(batch.blend) {
                continue;
            }
            let Some(entry) = self
                .texture_slots
                .get(batch.texture.index())
                .and_then(Option::as_ref)
                .or(fallback)
            else {
                continue;
            };
            let group = &entry.bind_group as *const wgpu::BindGroup;
            if bound != Some(group) {
                pass.set_bind_group(1, &entry.bind_group, &[]);
                bound = Some(group);
            }
            if batch.indexed {
                if self.quad_indices.is_none() {
                    continue;
                }
                pass.draw_indexed(
                    0..(batch.end - batch.start) / 4 * 6,
                    batch.start as i32,
                    0..1,
                );
            } else {
                pass.draw(batch.start..batch.end, 0..1);
            }
            drawn += 1;
        }
        drawn
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_entity_shadow_shader_validates() {
        let source = format!("{COMMON_WGSL}\n{SHADOW_WGSL}");
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
    fn only_solid_geometry_casts() {
        assert!(casts(LayerBlend::Opaque));
        assert!(casts(LayerBlend::LivingHurt));
        assert!(
            !casts(LayerBlend::Translucent),
            "contact shadows do not cast"
        );
        assert!(
            !casts(LayerBlend::Ghost),
            "the invisible teammate does not cast"
        );
        assert!(!casts(LayerBlend::Glint));
        assert!(!casts(LayerBlend::Additive));
    }
}
