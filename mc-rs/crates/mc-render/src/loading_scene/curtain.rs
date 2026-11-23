//! The loading screen's frame: owns the chosen [`Scene`], renders it into its own
//! full-window texture every frame, and lays that over the game's frame with the curtain's
//! opacity. One pass for the scene's own work, one full-screen composite.
//!
//! The composite blends, so the same phase covers the screen while the client loads (opacity 1:
//! the frame underneath does not matter) and fades out over the title screen afterwards.
//!
//! The title screen draws the Glass and Shore scenes through
//! this phase as its background, at opacity 1, with a [`super::BackdropStyle`] (a mirror and a
//! left scrim for the title's logo and buttons). [`Curtain::release`] gives the screen-sized
//! memory back while the title screen is away.

use std::cell::RefCell;
use std::rc::Rc;

use super::{BackdropStyle, Scene, SceneFrame, SCENE_FORMAT};
use crate::pipeline::RenderPhase;
use crate::{Camera, DepthAttachment, Gpu};

/// The scene's full-window target.
struct Target {
    size: (u32, u32),
    view: wgpu::TextureView,
    group: wgpu::BindGroup,
    _texture: wgpu::Texture,
}

/// A composite pipeline for one frame format and sample count.
struct Composite {
    key: (wgpu::TextureFormat, u32),
    pipeline: wgpu::RenderPipeline,
}

/// Everything the phase draws with; shared between [`super::LoadingCurtain`] and its phase.
pub(super) struct Curtain {
    scene: Box<dyn Scene>,
    pub(super) frame: SceneFrame,
    /// Multiplies the scene, `0..=1` (the client's opening settle).
    pub(super) brightness: f32,
    /// The mirror and the scrim; the default is neither.
    pub(super) style: BackdropStyle,
    module: wgpu::ShaderModule,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    uniform: wgpu::Buffer,
    target: Option<Target>,
    composite: Option<Composite>,
    /// Frames the scene has rendered (tests and the log).
    pub(super) frames: u64,
}

impl Curtain {
    pub(super) fn new(gpu: &Gpu, scene: Box<dyn Scene>) -> Curtain {
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("loading/curtain"),
                source: wgpu::ShaderSource::Wgsl(include_str!("curtain.wgsl").into()),
            });
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("loading/curtain"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("loading/curtain"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("loading/curtain"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Curtain {
            scene,
            frame: SceneFrame::default(),
            // Dark until the first update says otherwise.
            brightness: 0.0,
            style: BackdropStyle::default(),
            module,
            layout,
            pipeline_layout,
            uniform,
            target: None,
            composite: None,
            frames: 0,
        }
    }

    pub(super) fn scene_name(&self) -> &'static str {
        self.scene.name()
    }

    /// Give the screen-sized memory back (the scene target and the scene's own size-dependent
    /// resources) while keeping the pipelines and the scene. The next frame drawn builds them
    /// again.
    pub(super) fn release(&mut self) {
        self.target = None;
        self.scene.release_gpu_memory();
    }

    /// Whether [`Curtain::release`] left nothing screen-sized behind (nothing drawn since).
    pub(super) fn is_released(&self) -> bool {
        self.target.is_none()
    }

    fn ensure_target(&mut self, gpu: &Gpu, size: (u32, u32)) {
        if self.target.as_ref().is_some_and(|t| t.size == size) {
            return;
        }
        // Drop the old one first, so a resize never holds two.
        self.target = None;
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("loading/scene"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SCENE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("loading/curtain"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });
        self.target = Some(Target {
            size,
            view,
            group,
            _texture: texture,
        });
    }

    fn ensure_composite(&mut self, gpu: &Gpu) {
        let key = (gpu.config.format, gpu.sample_count());
        if self.composite.as_ref().is_some_and(|c| c.key == key) {
            return;
        }
        let blend = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("loading/curtain"),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &self.module,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: key.1,
                    ..wgpu::MultisampleState::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &self.module,
                    entry_point: Some("fs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: key.0,
                        blend: Some(wgpu::BlendState {
                            color: blend,
                            alpha: blend,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        self.composite = Some(Composite { key, pipeline });
    }

    pub(super) fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        let size = (gpu.config.width.max(1), gpu.config.height.max(1));
        self.frame.width = size.0;
        self.frame.height = size.1;
        self.ensure_target(gpu, size);
        self.ensure_composite(gpu);
        let target = self.target.as_ref().expect("ensured above");
        self.scene.render(gpu, enc, &target.view, &self.frame);
        self.frames += 1;

        let srgb = if gpu.config.format.is_srgb() {
            1.0
        } else {
            0.0
        };
        let params = [
            self.frame.opacity.clamp(0.0, 1.0),
            self.brightness.clamp(0.0, 1.0),
            srgb,
            if self.style.mirror { 1.0 } else { 0.0 },
            self.style.scrim.clamp(0.0, 0.5),
            0.0,
            0.0,
            0.0_f32,
        ];
        gpu.queue
            .write_buffer(&self.uniform, 0, bytemuck::cast_slice(&params));
        let composite = self.composite.as_ref().expect("ensured above");
        let mut pass = gpu.begin_pass(enc, view, "loading/curtain", DepthAttachment::None);
        pass.set_pipeline(&composite.pipeline);
        pass.set_bind_group(0, &target.group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// The curtain's [`RenderPhase`]. Disabled once fully transparent.
pub(super) struct CurtainPhase(pub(super) Rc<RefCell<Curtain>>);

impl RenderPhase for CurtainPhase {
    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        self.0.borrow_mut().record(gpu, enc, view);
    }

    fn name(&self) -> &'static str {
        super::CURTAIN_PHASE_NAME
    }

    fn enabled(&self) -> bool {
        self.0.borrow().frame.opacity > 0.0
    }
}
