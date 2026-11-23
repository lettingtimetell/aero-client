//! Classic's sun glow: a soft halo around the sun (and a faint one
//! around the moon) and a warm horizon haze at sunrise and sunset, added over the sky only.
//!
//! One [`SkyGlowPhase`] at [`super::order::WORLD`]. It draws one full-screen triangle at depth 1
//! with an `Equal` depth test against the shared depth buffer (attached read-only), so only pixels
//! where nothing but the sky was drawn pass: no depth read in the shader, no scene copy. The glow
//! is added (one-one blend) in display units, anchored on the sun disc the sky draws
//! ([`super::PackFrame::sun_disc`]).
//!
//! Off (strength 0, the default, or Classic not running) it records nothing and allocates
//! nothing.

use super::frame::PackId;
use super::shared::ShaderCore;
use crate::{Camera, Gpu, RenderPhase};

const GLOW_WGSL: &str = r#"
struct Glow {
    // xyz: toward the sun disc, w: sun glow amount (0 = none).
    sun: vec4<f32>,
    // rgb: sun glow colour (display), w: moon glow amount.
    sun_color: vec4<f32>,
    // rgb: horizon haze colour (display), w: haze amount.
    haze: vec4<f32>,
};
@group(1) @binding(0) var<uniform> glow: Glow;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @invariant @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    // Depth exactly 1: with an Equal test only untouched (sky) pixels pass. `@invariant`, as
    // every other Equal-tested pass declares it (the constant depth is exact either way).
    return vec4<f32>(corners[index], 1.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = pos.xy / camera.screen.xy;
    let ray = pk_view_ray(uv, camera.proj, camera.view);
    let sun = glow.sun.xyz;
    let c = max(dot(ray, sun), 0.0);
    // A tight bright core and a wide soft halo.
    let sun_glow = (pow(c, 900.0) * 0.6 + pow(c, 60.0) * 0.30 + pow(c, 8.0) * 0.10) * glow.sun.w;
    let m = max(dot(ray, -sun), 0.0);
    let moon_glow = (pow(m, 400.0) * 0.4 + pow(m, 30.0) * 0.18) * glow.sun_color.w;
    // Warm haze hugging the horizon on the sun's side at sunrise and sunset.
    let horizon = 1.0 - smoothstep(0.0, 0.35, abs(ray.y));
    let side = pow(max(dot(normalize(vec3<f32>(ray.x, 0.0, ray.z) + vec3<f32>(1e-4)),
        normalize(vec3<f32>(sun.x, 0.0, sun.z) + vec3<f32>(1e-4))), 0.0), 3.0);
    let haze = horizon * side * glow.haze.w;
    let rgb = glow.sun_color.rgb * sun_glow + vec3<f32>(0.55, 0.65, 0.9) * moon_glow
        + glow.haze.rgb * haze;
    return vec4<f32>(rgb, 0.0);
}
"#;

/// The camera uniform's WGSL, as `post::view_geometry` declares it.
const CAMERA_WGSL: &str = r#"
struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen:     vec4<f32>,
    frame:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;
"#;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct Glow {
    sun: [f32; 4],
    sun_color: [f32; 4],
    haze: [f32; 4],
}

struct GlowGpu {
    format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    group: wgpu::BindGroup,
    uploaded: Option<Glow>,
}

/// The sky glow for one pack. See the [module docs](self).
pub struct SkyGlowPhase {
    core: ShaderCore,
    pack: PackId,
    strength: f32,
    gpu: Option<GlowGpu>,
}

/// The module source.
#[must_use]
pub fn shader_source() -> String {
    format!("{}\n{}\n{}", super::COMMON_WGSL, CAMERA_WGSL, GLOW_WGSL)
}

impl SkyGlowPhase {
    /// The phase's name.
    pub const NAME: &'static str = "shaderpack/sky-glow";

    /// An idle phase for `pack`. Allocates nothing.
    #[must_use]
    pub fn new(core: ShaderCore, pack: PackId) -> SkyGlowPhase {
        SkyGlowPhase {
            core,
            pack,
            strength: 0.0,
            gpu: None,
        }
    }

    /// The glow's strength, 0..=1 (0 = off: frees everything).
    pub fn set_strength(&mut self, strength: f32) {
        self.strength = if strength.is_finite() {
            strength.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if self.strength <= 0.0 {
            self.gpu = None;
        }
    }

    /// Whether anything is allocated.
    #[must_use]
    pub fn allocated(&self) -> bool {
        self.gpu.is_some()
    }

    /// Drop everything while the pack does not run.
    pub fn release_if_idle(&mut self) {
        if !self.enabled() {
            self.gpu = None;
        }
    }

    fn build(gpu: &Gpu) -> GlowGpu {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaderpack/sky-glow"),
            source: wgpu::ShaderSource::Wgsl(shader_source().into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shaderpack/sky-glow"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shaderpack/sky-glow"),
            size: std::mem::size_of::<Glow>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shaderpack/sky-glow"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shaderpack/sky-glow"),
            bind_group_layouts: &[Some(gpu.camera_layout()), Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shaderpack/sky-glow"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Equal),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        GlowGpu {
            format: gpu.config.format,
            pipeline,
            uniform,
            group,
            uploaded: None,
        }
    }
}

impl RenderPhase for SkyGlowPhase {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn enabled(&self) -> bool {
        self.strength > 0.0 && self.core.is_running(self.pack)
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        let frame = self.core.frame();
        if frame.dimension != crate::sky::environment::Dimension::Overworld {
            return;
        }
        let clear_sky = 1.0 - frame.rain.clamp(0.0, 1.0);
        // The sun's glow follows its height (none below the horizon) and the weather.
        let sun_up = smooth(-0.05, 0.10, frame.sun_disc.y);
        let moon_up = smooth(-0.05, 0.10, -frame.sun_disc.y);
        let colour = self
            .core
            .light_colors()
            .map_or(glam::Vec3::ONE, |c| {
                c.direct / frame.light_strength.max(1e-3)
            })
            .min(glam::Vec3::splat(1.6));
        let sun_colour = if frame.light_is_sun {
            colour
        } else {
            glam::Vec3::new(1.0, 0.85, 0.6)
        };
        let s = self.strength;
        let uniform = Glow {
            sun: [
                frame.sun_disc.x,
                frame.sun_disc.y,
                frame.sun_disc.z,
                s * sun_up * clear_sky,
            ],
            sun_color: [
                sun_colour.x * 0.55,
                sun_colour.y * 0.55,
                sun_colour.z * 0.55,
                s * moon_up * clear_sky * 0.6,
            ],
            haze: [0.9, 0.45, 0.2, s * frame.golden * sun_up * clear_sky * 0.22],
        };
        if self
            .gpu
            .as_ref()
            .is_none_or(|g| g.format != gpu.config.format)
        {
            self.gpu = Some(Self::build(gpu));
        }
        let built = self.gpu.as_mut().expect("built above");
        if built.uploaded != Some(uniform) {
            gpu.write_buffer_staged(&built.uniform, 0, bytemuck::bytes_of(&uniform));
            built.uploaded = Some(uniform);
        }
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(Self::NAME),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: gpu.depth_view(),
                depth_ops: None,
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&built.pipeline);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.set_bind_group(1, &built.group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn smooth(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_glow_shader_validates() {
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
    fn an_idle_glow_is_off_and_owns_nothing() {
        let mut phase = SkyGlowPhase::new(ShaderCore::new(), PackId::Classic);
        assert!(!phase.enabled());
        phase.set_strength(0.5);
        assert!(!phase.enabled(), "no pack runs");
        assert!(!phase.allocated());
    }
}
