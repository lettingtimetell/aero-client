//! The shader packs' final pass: bloom and tone mapping.
//!
//! One [`FinalPhase`] at [`super::order::FINAL`] (`PhaseOrder::POST`): after the world, the hand
//! and the overlays, before the HUD, so the HUD, the crosshair and chat stay exactly as vanilla
//! draws them. It runs while the running pack's [`CoreConfig`](super::CoreConfig) has bloom or a
//! tone map on.
//!
//! * **Bloom** (dual filter): a thresholded, Karis-averaged prefilter to half resolution, then
//!   5-tap downsamples to 1/4 .. 1/64 (three to six levels by radius) and 8-tap tent upsamples
//!   added back level by level, all in `Rgba16Float`, linear light.
//! * **Composite** (full resolution): linearise the frame, add the bloom, exposure, the tone
//!   curve, contrast, saturation, vibrance, re-encode, a fixed sub-LSB dither.
//!
//! # Off costs nothing
//!
//! [`FinalPhase::new`] allocates nothing. While no pack runs (or the running one wants neither
//! feature) `enabled()` is `false` and every target is dropped; the pipelines are built on the
//! first enabled frame.

use super::config::{BloomSettings, ToneCurve, TonemapSettings};
use super::shared::ShaderCore;
use crate::{Camera, Gpu, RenderPhase, SceneTextures};

/// The bloom chain's format.
pub const BLOOM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The most bloom levels (half resolution down to 1/64).
pub const MAX_BLOOM_LEVELS: usize = 6;

/// `final.wgsl`'s `FinalParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct FinalParams {
    bloom: [f32; 4],
    tone: [f32; 4],
    mode: [f32; 4],
}

impl FinalParams {
    fn new(bloom: Option<BloomSettings>, tone: Option<TonemapSettings>, levels: usize) -> Self {
        let (b, b_on) = match bloom {
            Some(b) => (b, 1.0),
            None => (BloomSettings::default(), 0.0),
        };
        let (t, t_on) = match tone {
            Some(t) => (t, 1.0),
            None => (TonemapSettings::default(), 0.0),
        };
        let curve = match t.curve {
            ToneCurve::Filmic => 0.0,
            ToneCurve::Aces => 1.0,
            ToneCurve::Linear => 2.0,
        };
        // The up chain adds `levels` copies of the glow; keep the strength meaning the same
        // whatever the radius.
        let strength = b.strength * 1.6 / levels.max(1) as f32;
        FinalParams {
            bloom: [b.threshold, (b.threshold * 0.5).max(0.05), strength, b_on],
            tone: [t.exposure.exp2(), t.contrast, t.saturation, t.vibrance],
            mode: [curve, t_on, levels as f32, 0.0],
        }
    }
}

/// How many bloom levels a radius asks for (3 ..= [`MAX_BLOOM_LEVELS`]).
#[must_use]
pub fn bloom_levels(radius: f32) -> usize {
    let extra = (radius.clamp(0.0, 1.0) * 3.0).round() as usize;
    (3 + extra).min(MAX_BLOOM_LEVELS)
}

struct Pipelines {
    format: wgpu::TextureFormat,
    layout: wgpu::BindGroupLayout,
    prefilter: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    params: wgpu::Buffer,
    /// A 1x1 black texture bound in place of the bloom while bloom is off.
    black: wgpu::TextureView,
}

impl Pipelines {
    fn new(gpu: &Gpu) -> Pipelines {
        let device = &gpu.device;
        let format = gpu.config.format;
        let source = format!("{}\n{}", super::COMMON_WGSL, include_str!("final.wgsl"));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaderpack/final"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shaderpack/final"),
            entries: &[
                texture(0),
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
                texture(3),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shaderpack/final"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |entry: &str, target: wgpu::TextureFormat, blend: Option<wgpu::BlendState>| {
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
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let additive = wgpu::BlendState {
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
        };
        let prefilter = build("fs_prefilter", BLOOM_FORMAT, None);
        let down = build("fs_down", BLOOM_FORMAT, None);
        let up = build("fs_up", BLOOM_FORMAT, Some(additive));
        let composite = build("fs_composite", format, None);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shaderpack/final linear clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shaderpack/final params"),
            size: std::mem::size_of::<FinalParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let black = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("shaderpack/final no bloom"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: BLOOM_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        Pipelines {
            format,
            layout,
            prefilter,
            down,
            up,
            composite,
            sampler,
            params,
            black,
        }
    }

    fn group(
        &self,
        gpu: &Gpu,
        src: &wgpu::TextureView,
        bloom: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shaderpack/final"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(bloom),
                },
            ],
        })
    }
}

/// The bloom chain for one frame size and level count.
struct BloomChain {
    size: (u32, u32),
    levels: Vec<(wgpu::TextureView, (u32, u32))>,
    /// `down[i]` reads level `i` (writes `i + 1`); `up[i]` reads level `i + 1` (adds into `i`).
    down: Vec<wgpu::BindGroup>,
    up: Vec<wgpu::BindGroup>,
}

impl BloomChain {
    fn new(gpu: &Gpu, pipelines: &Pipelines, size: (u32, u32), count: usize) -> BloomChain {
        let mut levels = Vec::with_capacity(count);
        let (mut w, mut h) = (size.0.div_ceil(2).max(1), size.1.div_ceil(2).max(1));
        for _ in 0..count {
            let view = gpu
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("shaderpack bloom level"),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: BLOOM_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default());
            levels.push((view, (w, h)));
            w = w.div_ceil(2).max(1);
            h = h.div_ceil(2).max(1);
        }
        let down = (0..count.saturating_sub(1))
            .map(|i| pipelines.group(gpu, &levels[i].0, &pipelines.black))
            .collect();
        let up = (0..count.saturating_sub(1))
            .map(|i| pipelines.group(gpu, &levels[i + 1].0, &pipelines.black))
            .collect();
        BloomChain {
            size,
            levels,
            down,
            up,
        }
    }
}

/// The scene-dependent bind groups, cached against the snapshot view they read.
struct SceneGroups {
    scene: wgpu::TextureView,
    bloom_level0: Option<wgpu::TextureView>,
    prefilter: wgpu::BindGroup,
    composite: wgpu::BindGroup,
}

/// Bloom and tone mapping over the finished frame. See the [module docs](self).
pub struct FinalPhase {
    core: ShaderCore,
    pipelines: Option<Pipelines>,
    chain: Option<BloomChain>,
    groups: Option<SceneGroups>,
    uploaded: Option<FinalParams>,
    passes: u64,
}

impl FinalPhase {
    /// The phase's name in `PhaseList` and GPU timings.
    pub const NAME: &'static str = "shaderpack/final";

    /// An idle phase reading `core`. Allocates nothing.
    #[must_use]
    pub fn new(core: ShaderCore) -> FinalPhase {
        FinalPhase {
            core,
            pipelines: None,
            chain: None,
            groups: None,
            uploaded: None,
            passes: 0,
        }
    }

    /// Frames recorded so far (stays put while off).
    #[must_use]
    pub fn passes_recorded(&self) -> u64 {
        self.passes
    }

    /// Whether any screen-sized target exists.
    #[must_use]
    pub fn allocated(&self) -> bool {
        self.chain.is_some() || self.groups.is_some()
    }

    /// Drop the screen-sized targets if the phase is not running (the glue calls this when a
    /// pack stops running). The pipelines are small and kept.
    pub fn release_if_idle(&mut self) {
        if !self.active() {
            self.chain = None;
            self.groups = None;
        }
    }

    fn active(&self) -> bool {
        self.core.any_running() && self.core.config().uses_final_pass()
    }
}

impl RenderPhase for FinalPhase {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn enabled(&self) -> bool {
        self.active()
    }

    fn needs_scene_color(&self) -> bool {
        true
    }

    fn resize(&mut self, _gpu: &Gpu, _width: u32, _height: u32) {
        self.chain = None;
        self.groups = None;
    }

    fn record(
        &mut self,
        _gpu: &Gpu,
        _enc: &mut wgpu::CommandEncoder,
        _view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        // Only reachable without scene colour, which this phase always asks for.
    }

    fn record_scene(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        let config = self.core.config();
        if self
            .pipelines
            .as_ref()
            .is_none_or(|p| p.format != scene.format)
        {
            self.pipelines = Some(Pipelines::new(gpu));
            self.chain = None;
            self.groups = None;
            self.uploaded = None;
        }
        let pipelines = self.pipelines.as_ref().expect("built above");
        let size = (scene.width, scene.height);
        let levels = config.bloom.map_or(0, |b| bloom_levels(b.radius));
        if levels == 0 {
            self.chain = None;
        } else if self
            .chain
            .as_ref()
            .is_none_or(|c| c.size != size || c.levels.len() != levels)
        {
            self.chain = Some(BloomChain::new(gpu, pipelines, size, levels));
            self.groups = None;
        }
        let level0 = self.chain.as_ref().map(|c| c.levels[0].0.clone());
        if self
            .groups
            .as_ref()
            .is_none_or(|g| g.scene != *scene.color || g.bloom_level0 != level0)
        {
            let prefilter = pipelines.group(gpu, scene.color, &pipelines.black);
            let composite = pipelines.group(
                gpu,
                scene.color,
                level0.as_ref().unwrap_or(&pipelines.black),
            );
            self.groups = Some(SceneGroups {
                scene: scene.color.clone(),
                bloom_level0: level0,
                prefilter,
                composite,
            });
        }
        let params = FinalParams::new(config.bloom, config.tonemap, levels.max(1));
        if self.uploaded != Some(params) {
            gpu.write_buffer_staged(&pipelines.params, 0, bytemuck::bytes_of(&params));
            self.uploaded = Some(params);
        }
        let groups = self.groups.as_ref().expect("built above");

        if let Some(chain) = &self.chain {
            let pass = |enc: &mut wgpu::CommandEncoder,
                        target: &wgpu::TextureView,
                        load: wgpu::LoadOp<wgpu::Color>,
                        pipeline: &wgpu::RenderPipeline,
                        group: &wgpu::BindGroup| {
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shaderpack/bloom"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rp.set_pipeline(pipeline);
                rp.set_bind_group(0, group, &[]);
                rp.draw(0..3, 0..1);
            };
            let clear = wgpu::LoadOp::Clear(wgpu::Color::BLACK);
            pass(
                enc,
                &chain.levels[0].0,
                clear,
                &pipelines.prefilter,
                &groups.prefilter,
            );
            for i in 0..chain.down.len() {
                pass(
                    enc,
                    &chain.levels[i + 1].0,
                    clear,
                    &pipelines.down,
                    &chain.down[i],
                );
            }
            for i in (0..chain.up.len()).rev() {
                pass(
                    enc,
                    &chain.levels[i].0,
                    wgpu::LoadOp::Load,
                    &pipelines.up,
                    &chain.up[i],
                );
            }
        }

        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
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
        rp.set_pipeline(&pipelines.composite);
        rp.set_bind_group(0, &groups.composite, &[]);
        rp.draw(0..3, 0..1);
        self.passes += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_wgsl_validates() {
        let source = format!(
            "{}\n{}",
            super::super::COMMON_WGSL,
            include_str!("final.wgsl")
        );
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
    fn bloom_levels_follow_the_radius() {
        assert_eq!(bloom_levels(0.0), 3);
        assert_eq!(bloom_levels(0.5), 5);
        assert_eq!(bloom_levels(1.0), 6);
        assert_eq!(bloom_levels(f32::NAN.max(0.0)), 3);
    }

    #[test]
    fn an_idle_phase_is_disabled_and_allocates_nothing() {
        let phase = FinalPhase::new(ShaderCore::new());
        assert!(!phase.enabled());
        assert!(!phase.allocated());
        assert_eq!(phase.passes_recorded(), 0);
    }
}
