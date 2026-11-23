//! Classic's light rays: The screen-space light rays
//! ([`super::godrays`]) composited additively over the world.
//!
//! One [`RaysPhase`] at [`super::order::WORLD`] (after the screen-space fog, before TAA, so TAA
//! smooths the rays' dither): it records SB's mask and blurs at quarter resolution and adds
//! `rays x sun colour x strength x on-screen visibility` onto the frame with an additive blend,
//! so it needs no scene copy. It records nothing while the sun is behind the camera, off screen,
//! below the horizon or hidden by rain.
//!
//! Off (the default, and whenever Classic does not run) it allocates nothing.

use glam::Vec3;

use super::frame::PackId;
use super::godrays::{light_on_screen, LightRays, RaysParams};
use super::shared::ShaderCore;
use crate::{Camera, Gpu, RenderPhase};

const COMPOSITE_WGSL: &str = r#"
struct Composite {
    // rgb: colour x strength x visibility (display units), w: unused.
    tint: vec4<f32>,
};
@group(0) @binding(0) var rays_tex: texture_2d<f32>;
@group(0) @binding(1) var rays_samp: sampler;
@group(0) @binding(2) var<uniform> composite: Composite;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VsOut {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    let c = corners[index];
    var out: VsOut;
    out.pos = vec4<f32>(c, 0.0, 1.0);
    out.uv = vec2<f32>(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let r = textureSampleLevel(rays_tex, rays_samp, in.uv, 0.0).r;
    // A soft response: faint rays stay faint, the core near the sun never clips to white.
    let glow = 1.0 - exp(-r * 1.6);
    return vec4<f32>(composite.tint.rgb * glow, 0.0);
}
"#;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct Composite {
    tint: [f32; 4],
}

struct Gpu2 {
    format: wgpu::TextureFormat,
    rays: LightRays,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    uploaded: Option<Composite>,
    group: Option<(wgpu::TextureView, wgpu::BindGroup)>,
}

/// Light rays from the sun for one pack. See the [module docs](self).
pub struct RaysPhase {
    core: ShaderCore,
    pack: PackId,
    /// 0 = off.
    strength: f32,
    gpu: Option<Gpu2>,
    records: u64,
}

impl RaysPhase {
    /// The phase's name.
    pub const NAME: &'static str = "shaderpack/rays";

    /// An idle phase for `pack`. Allocates nothing.
    #[must_use]
    pub fn new(core: ShaderCore, pack: PackId) -> RaysPhase {
        RaysPhase {
            core,
            pack,
            strength: 0.0,
            gpu: None,
            records: 0,
        }
    }

    /// The rays' strength, 0..=1 (0 = off: frees everything).
    pub fn set_strength(&mut self, strength: f32) {
        let strength = if strength.is_finite() {
            strength.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.strength = strength;
        if strength <= 0.0 {
            self.gpu = None;
        }
    }

    /// Frames the rays were drawn on.
    #[must_use]
    pub fn records(&self) -> u64 {
        self.records
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

    fn build(gpu: &Gpu) -> Gpu2 {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaderpack/rays composite"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shaderpack/rays composite"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shaderpack/rays composite"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shaderpack/rays composite"),
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
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shaderpack/rays composite"),
            size: std::mem::size_of::<Composite>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Gpu2 {
            format: gpu.config.format,
            rays: LightRays::new(gpu),
            layout,
            pipeline,
            uniform,
            uploaded: None,
            group: None,
        }
    }
}

impl RenderPhase for RaysPhase {
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
        cam: &Camera,
    ) {
        let frame = self.core.frame();
        if !frame.light_is_sun || frame.light_strength <= 0.01 {
            return;
        }
        // The unjittered camera: the rays are a smooth effect, TAA smooths their dither.
        let mut unjittered = *cam;
        unjittered.jitter = glam::Vec2::ZERO;
        let Some(on_screen) = light_on_screen(&unjittered, frame.sun_disc) else {
            return;
        };
        if self
            .gpu
            .as_ref()
            .is_none_or(|g| g.format != gpu.config.format)
        {
            self.gpu = Some(Self::build(gpu));
        }
        let built = self.gpu.as_mut().expect("built above");
        let full = (gpu.config.width.max(1), gpu.config.height.max(1));
        built.rays.record(
            gpu,
            enc,
            full,
            &RaysParams {
                light_uv: on_screen.uv,
                radius: 0.32,
                reach: 0.85,
                decay: 0.97,
                disc: 0.6,
                // A new dither every frame only while TAA averages it away (it never jitters by
                // exactly zero); without TAA a still one, or the rays shimmer on a still scene
                // (as the Dreamy pack's rays and fog already do).
                jitter_frame: if cam.jitter == glam::Vec2::ZERO && super::shadow::steady_shading() {
                    0
                } else {
                    frame.frame_index
                },
            },
        );
        let Some(output) = built.rays.output() else {
            return;
        };
        // The direct light's colour, warm at sunset, as a display-space glow.
        let colour = self
            .core
            .light_colors()
            .map_or(Vec3::ONE, |c| c.direct / frame.light_strength.max(1e-3));
        let amount = self.strength * on_screen.visibility * frame.light_strength * 0.55;
        let tint = (colour * amount).min(Vec3::splat(2.0));
        let uniform = Composite {
            tint: [tint.x, tint.y, tint.z, 0.0],
        };
        if built.uploaded != Some(uniform) {
            gpu.write_buffer_staged(&built.uniform, 0, bytemuck::bytes_of(&uniform));
            built.uploaded = Some(uniform);
        }
        if built.group.as_ref().is_none_or(|(v, _)| v != output) {
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shaderpack/rays composite"),
                layout: &built.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(output),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(built.rays.sampler()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: built.uniform.as_entire_binding(),
                    },
                ],
            });
            built.group = Some((output.clone(), group));
        }
        let group = &built.group.as_ref().expect("built above").1;
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
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&built.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..3, 0..1);
        self.records += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_composite_validates() {
        let module = wgpu::naga::front::wgsl::parse_str(COMPOSITE_WGSL)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(COMPOSITE_WGSL)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(COMPOSITE_WGSL)));
    }

    #[test]
    fn an_idle_phase_is_off_and_owns_nothing() {
        let mut phase = RaysPhase::new(ShaderCore::new(), PackId::Classic);
        assert!(!phase.enabled());
        phase.set_strength(0.5);
        assert!(!phase.enabled(), "no pack runs");
        assert!(!phase.allocated());
        phase.set_strength(f32::NAN);
        assert_eq!(phase.strength, 0.0);
    }
}
