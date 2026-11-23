//! Per-pixel motion blur: the `motion_blur` mod's Camera and Camera + objects modes.
//!
//! [`VelocityBlurPhase`] records at [`crate::PhaseOrder::VELOCITY_BLUR`] (644), after the
//! anti-aliasing and before the waypoint labels (645) and the first-person hand (650). It blurs
//! each pixel along the motion target's exposure motion (`ba`, [`crate::post::velocity`]): the
//! distance its surface moved over the last `exposure` seconds, which the client derives from
//! the mod's Strength so a strength means the same trail length in every mode. The hand, the
//! labels, the screen overlays and the HUD move with the camera and are drawn after it, so they
//! stay sharp; Frame blend (the mode that blends whole frames) still runs in
//! `post/motion-grade` at `PhaseOrder::POST`.
//!
//! The trail is clamped to [`MAX_TRAIL_FRACTION`] of the frame width and drawn with 3 to 16
//! tent-weighted samples (about one per two pixels of trail; the ends fade out like the frame
//! blend's exponential tail), offset per pixel by interleaved gradient noise.
//! Pixels whose trail is under half a pixel return at once, so a still view costs one fetch per
//! pixel.
//!
//! Off (Frame blend, motion blur off, outside a world, under MSAA): `enabled() == false`, so
//! `PhaseList` keeps its direct path; nothing of the phase exists on the GPU but its pipeline
//! after first use.

use crate::post::velocity::VelocityOutput;
use crate::{Camera, Gpu, RenderPhase, SceneTextures};

/// The longest trail, as a fraction of the frame width (96 pixels at 1920).
pub const MAX_TRAIL_FRACTION: f32 = 0.05;

/// `BlurParams` in `velocity_blur.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    params: [f32; 4],
    screen: [f32; 4],
}

struct Pipelines {
    format: wgpu::TextureFormat,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
}

impl Pipelines {
    fn new(gpu: &Gpu, format: wgpu::TextureFormat) -> Pipelines {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post/velocity-blur"),
            source: wgpu::ShaderSource::Wgsl(include_str!("velocity_blur.wgsl").into()),
        });
        let texture = |binding: u32, filterable: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post/velocity-blur"),
            entries: &[
                texture(0, true),
                texture(1, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
            label: Some("post/velocity-blur"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("post/velocity-blur"),
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
                entry_point: Some("fs_blur"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post/velocity-blur linear"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..wgpu::SamplerDescriptor::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post/velocity-blur params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Pipelines {
            format,
            layout,
            pipeline,
            sampler,
            uniform,
        }
    }
}

/// The bind group, keyed by the snapshot view and the motion target's generation.
struct Binding {
    scene: wgpu::TextureView,
    velocity: u64,
    group: wgpu::BindGroup,
}

/// Per-pixel motion blur. See the [module docs](self).
pub struct VelocityBlurPhase {
    on: bool,
    in_world: bool,
    msaa: bool,
    velocity: Option<VelocityOutput>,
    velocity_frame: u64,
    pipelines: Option<Pipelines>,
    binding: Option<Binding>,
    uploaded: Option<Params>,
    passes: u64,
}

impl std::fmt::Debug for VelocityBlurPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VelocityBlurPhase")
            .field("on", &self.on)
            .field("in_world", &self.in_world)
            .field("msaa", &self.msaa)
            .field("passes", &self.passes)
            .finish_non_exhaustive()
    }
}

impl VelocityBlurPhase {
    /// The phase's name in `PhaseList`, GPU timings and captures.
    pub const NAME: &'static str = "post/velocity-blur";

    /// An idle phase. Allocates nothing on the GPU.
    #[must_use]
    pub fn new(gpu: &Gpu) -> VelocityBlurPhase {
        VelocityBlurPhase {
            on: false,
            in_world: false,
            msaa: gpu.sample_count() > 1,
            velocity: None,
            velocity_frame: 0,
            pipelines: None,
            binding: None,
            uploaded: None,
            passes: 0,
        }
    }

    /// The motion target to blur along ([`crate::post::velocity::VelocityPhase::output`]).
    pub fn set_velocity(&mut self, velocity: VelocityOutput) {
        self.velocity = Some(velocity);
        self.binding = None;
    }

    /// Switch the blur on (Camera or Camera + objects mode with a strength above 0) or off.
    pub fn set_on(&mut self, on: bool) {
        if on != self.on {
            self.on = on;
            if !self.is_active() {
                self.binding = None;
            }
        }
    }

    /// Whether a world is on screen.
    pub fn set_in_world(&mut self, in_world: bool) {
        if in_world != self.in_world {
            self.in_world = in_world;
            if !self.is_active() {
                self.binding = None;
            }
        }
    }

    /// Whether the phase runs this frame.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.on && self.in_world && !self.msaa
    }

    /// How many passes the phase recorded.
    #[must_use]
    pub fn passes_recorded(&self) -> u64 {
        self.passes
    }
}

impl RenderPhase for VelocityBlurPhase {
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
        // Only reachable outside `PhaseList`: with no snapshot to read, draw nothing.
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
        if self
            .pipelines
            .as_ref()
            .is_none_or(|pipelines| pipelines.format != scene.format)
        {
            self.pipelines = Some(Pipelines::new(gpu, scene.format));
            self.binding = None;
            self.uploaded = None;
        }
        let pipelines = self.pipelines.as_ref().expect("built above");
        let Some(output) = self.velocity.as_ref() else {
            return;
        };
        let frame = output.get();
        let Some(motion) = frame.as_ref() else {
            return;
        };
        // A stale target (the velocity pass did not run this frame) would blur last frame's
        // motion: draw nothing instead.
        if motion.frame == self.velocity_frame {
            return;
        }
        self.velocity_frame = motion.frame;
        if self
            .binding
            .as_ref()
            .is_none_or(|b| b.scene != *scene.color || b.velocity != motion.generation)
        {
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post/velocity-blur"),
                layout: &pipelines.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(scene.color),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&motion.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&pipelines.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: pipelines.uniform.as_entire_binding(),
                    },
                ],
            });
            self.binding = Some(Binding {
                scene: scene.color.clone(),
                velocity: motion.generation,
                group,
            });
        }
        drop(frame);
        let (w, h) = (scene.width.max(1) as f32, scene.height.max(1) as f32);
        let params = Params {
            params: [w * MAX_TRAIL_FRACTION, 0.0, 0.0, 0.0],
            screen: [w, h, 1.0 / w, 1.0 / h],
        };
        if self.uploaded != Some(params) {
            gpu.write_buffer_staged(&pipelines.uniform, 0, bytemuck::bytes_of(&params));
            self.uploaded = Some(params);
        }
        let group = &self.binding.as_ref().expect("built above").group;
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
        pass.set_pipeline(&pipelines.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        self.passes += 1;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_wgsl_parses_and_validates() {
        let source = include_str!("velocity_blur.wgsl");
        let module = wgpu::naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("velocity_blur.wgsl validates");
    }
}
