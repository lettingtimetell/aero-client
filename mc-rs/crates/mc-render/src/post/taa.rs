//! Anti-aliasing of the world: TAA and FXAA.
//!
//! [`AntiAliasPhase`] is the render half of the `anti_aliasing` mod. It records at
//! [`crate::PhaseOrder::TEMPORAL_AA`] (643): after the whole world, the screen-space fog (640)
//! and the late volumetric clouds (642) included, and before the waypoint labels (645), the
//! first-person hand (650), the screen
//! overlays, vanilla's post chain and the HUD. So the HUD, the hand, labels and the mod menu are
//! never jittered, blurred or sharpened.
//!
//! # TAA
//!
//! * **Jitter.** The client asks [`AntiAliasPhase::next_jitter`] each frame and puts it on
//!   [`Camera::jitter`]: an 8-phase Halton(2, 3) sequence of sub-pixel offsets. Only the world
//!   phases see it (the camera uniform is restored at 643, see `Camera::jitter`).
//! * **Reprojection.** The motion target ([`crate::post::velocity`], `rg`) of the closest
//!   surface in the 3x3 neighbourhood gives the history position (closest-depth dilation keeps
//!   silhouettes on the moving object's motion). Moving entities carry their own motion there.
//! * **History.** `Rgba16Float`, ping-pong: rgb = the previous resolve, a = its view depth. Read
//!   with a 5-tap Catmull-Rom filter, which keeps a moving image sharper than bilinear.
//! * **Rejection.** Disocclusion: the reprojected surface lies clearly behind everything the
//!   history held around there (an occluder moved away), so the history is dropped. Otherwise
//!   the history is clipped (towards the centre, not per channel) into the current 3x3
//!   neighbourhood's YCoCg box: mean +- one standard deviation, inside min/max.
//! * **Blend.** 6 % current frame when still ([`STILL_CURRENT_WEIGHT`]), up to 50 % with
//!   motion. On a still edge the settled result repeats exactly every jitter cycle and moves at
//!   most 14 levels between frames (`tests/x9_aa.rs`); 8 % with inverse-luma weights rippled 29.
//!   **Sharpness** (0 to 100 %) runs a clamped unsharp mask on the way to the frame; the history
//!   itself stays unsharpened.
//! * **Reset.** A resize, a new history, a world change or teleport
//!   ([`AntiAliasPhase::reset_history`], and the motion target's own teleport test), or a frame
//!   with no fresh motion target shows the current frame alone and starts over.
//!
//! # FXAA
//!
//! One fullscreen pass, `fxaa.wgsl` (FXAA 3.11 quality, written from the published algorithm).
//! **Texture edges** off (the default) limits it to real geometric edges found in the depth
//! buffer, so the pixel-art textures stay crisp; on, it is classic FXAA.
//!
//! # Off costs nothing
//!
//! Mode off, outside a world or under MSAA: `enabled() == false` and `needs_scene_color() ==
//! false`, so `PhaseList` keeps its direct path byte for byte, and [`AntiAliasPhase::next_jitter`]
//! returns zero, so the camera uniform is today's too. No texture of the phase exists; the
//! pipelines are built on first use. Under MSAA the UI pass resolves over the frame and would
//! erase the result, so the phase stays off there, like `post/motion-grade`.

use glam::Vec2;

use crate::post::velocity::VelocityOutput;
use crate::{Camera, Gpu, RenderPhase, SceneTextures};

/// The history's format: fp16 colour plus the view depth in alpha.
pub const HISTORY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// How many jitter positions the Halton sequence cycles through.
pub const JITTER_PHASES: u64 = 8;

/// The current frame's weight in the blend when nothing moves.
pub const STILL_CURRENT_WEIGHT: f32 = 0.06;

/// Which anti-aliasing runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AaMode {
    /// Nothing: the phase is off.
    #[default]
    Off,
    /// One-pass FXAA.
    Fxaa,
    /// Temporal anti-aliasing (needs the motion target).
    Taa,
}

/// The radical inverse of `index` in `base`: element `index` of the Halton sequence, in [0, 1).
#[must_use]
pub fn halton(mut index: u32, base: u32) -> f32 {
    let mut fraction = 1.0;
    let mut result = 0.0;
    while index > 0 {
        fraction /= base as f32;
        result += fraction * (index % base) as f32;
        index /= base;
    }
    result
}

/// The sub-pixel offset for frame `frame`, in pixels within [-0.5, 0.5): Halton(2, 3), indices
/// 1 to [`JITTER_PHASES`] (index 0 is the pixel corner twice).
#[must_use]
pub fn jitter_pixels(frame: u64) -> Vec2 {
    let index = (frame % JITTER_PHASES) as u32 + 1;
    Vec2::new(halton(index, 2) - 0.5, halton(index, 3) - 0.5)
}

/// [`jitter_pixels`] in NDC for a `width` x `height` frame: what goes on [`Camera::jitter`].
/// NDC y points up, pixel y down.
#[must_use]
pub fn jitter_ndc(frame: u64, width: u32, height: u32) -> Vec2 {
    let px = jitter_pixels(frame);
    Vec2::new(
        2.0 * px.x / width.max(1) as f32,
        -2.0 * px.y / height.max(1) as f32,
    )
}

/// One `vec4` of parameters (`TaaParams` / `FxaaParams` in the WGSL).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    params: [f32; 4],
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

fn depth_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
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

fn color_target(format: wgpu::TextureFormat) -> Option<wgpu::ColorTargetState> {
    Some(wgpu::ColorTargetState {
        format,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    })
}

/// A colour attachment the pass overwrites completely (`Load`, as `PhaseList`'s own blit).
fn attachment(view: &wgpu::TextureView) -> Option<wgpu::RenderPassColorAttachment<'_>> {
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

fn fullscreen_pipeline(
    gpu: &Gpu,
    shader: &wgpu::ShaderModule,
    layouts: &[Option<&wgpu::BindGroupLayout>],
    entry: &'static str,
    targets: &[Option<wgpu::ColorTargetState>],
) -> wgpu::RenderPipeline {
    let layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(entry),
            bind_group_layouts: layouts,
            immediate_size: 0,
        });
    gpu.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(entry),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets,
            }),
            multiview_mask: None,
            cache: None,
        })
}

/// Every pipeline of the phase, for one frame format, built on first use.
struct Pipelines {
    format: wgpu::TextureFormat,
    sampler: wgpu::Sampler,
    fxaa_layout: wgpu::BindGroupLayout,
    fxaa: wgpu::RenderPipeline,
    fxaa_uniform: wgpu::Buffer,
    taa_layout: wgpu::BindGroupLayout,
    history_layout: wgpu::BindGroupLayout,
    taa_both: wgpu::RenderPipeline,
    taa_history: wgpu::RenderPipeline,
    taa_uniform: wgpu::Buffer,
    sharpen_layout: wgpu::BindGroupLayout,
    sharpen: wgpu::RenderPipeline,
}

impl Pipelines {
    fn new(gpu: &Gpu, format: wgpu::TextureFormat) -> Pipelines {
        let device = &gpu.device;
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post/aa linear"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..wgpu::SamplerDescriptor::default()
        });
        let uniform = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<Params>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let module = |label: &str, source: &'static str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
        };

        let fxaa_shader = module("post/fxaa", include_str!("fxaa.wgsl"));
        let fxaa_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post/fxaa"),
            entries: &[
                texture_entry(0, true),
                sampler_entry(1),
                depth_entry(2),
                uniform_entry(3),
            ],
        });
        let fxaa = fullscreen_pipeline(
            gpu,
            &fxaa_shader,
            &[Some(&fxaa_layout)],
            "fs_fxaa",
            &[color_target(format)],
        );

        let taa_shader = module("post/taa", include_str!("taa.wgsl"));
        let taa_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post/taa"),
            entries: &[
                texture_entry(0, false),
                depth_entry(1),
                texture_entry(2, false),
                uniform_entry(3),
                uniform_entry(4),
                sampler_entry(5),
            ],
        });
        let history_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post/taa history"),
            entries: &[texture_entry(0, true)],
        });
        let taa_layouts = [Some(&taa_layout), Some(&history_layout)];
        let taa_both = fullscreen_pipeline(
            gpu,
            &taa_shader,
            &taa_layouts,
            "fs_resolve",
            &[color_target(format), color_target(HISTORY_FORMAT)],
        );
        let taa_history = fullscreen_pipeline(
            gpu,
            &taa_shader,
            &taa_layouts,
            "fs_resolve_history",
            &[color_target(HISTORY_FORMAT)],
        );

        let sharpen_shader = module("post/taa-sharpen", include_str!("taa_sharpen.wgsl"));
        let sharpen_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post/taa sharpen"),
            entries: &[
                texture_entry(0, false),
                uniform_entry(1),
                texture_entry(2, false),
            ],
        });
        let sharpen = fullscreen_pipeline(
            gpu,
            &sharpen_shader,
            &[Some(&sharpen_layout)],
            "fs_sharpen",
            &[color_target(format)],
        );

        Pipelines {
            format,
            sampler,
            fxaa_layout,
            fxaa,
            fxaa_uniform: uniform("post/fxaa params"),
            taa_layout,
            history_layout,
            taa_both,
            taa_history,
            taa_uniform: uniform("post/taa params"),
            sharpen_layout,
            sharpen,
        }
    }
}

/// What the current bind groups were built from: rebuilt only when one of these changes (a
/// resize, a new motion target), never per frame.
#[derive(PartialEq)]
struct BindingKey {
    scene: wgpu::TextureView,
    depth: wgpu::Texture,
    velocity: u64,
}

struct Bindings {
    key: BindingKey,
    fxaa: wgpu::BindGroup,
    /// `None` without a motion target (FXAA only).
    taa: Option<wgpu::BindGroup>,
}

/// The TAA history: the frame reads `views[latest]` and writes the other one.
struct History {
    width: u32,
    height: u32,
    views: [wgpu::TextureView; 2],
    /// `reads[i]` samples `views[i]` (group 1 of the resolve).
    reads: [wgpu::BindGroup; 2],
    /// `sharpen[i]` reads `views[i]` and the scene snapshot; rebuilt with the scene view.
    sharpen: Option<(wgpu::TextureView, [wgpu::BindGroup; 2])>,
    latest: usize,
}

impl History {
    fn new(gpu: &Gpu, pipelines: &Pipelines, width: u32, height: u32) -> History {
        let views: [wgpu::TextureView; 2] = std::array::from_fn(|_| {
            gpu.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("post/taa history"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: HISTORY_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        });
        let reads = std::array::from_fn(|i| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post/taa history"),
                layout: &pipelines.history_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&views[i]),
                }],
            })
        });
        History {
            width,
            height,
            views,
            reads,
            sharpen: None,
            latest: 0,
        }
    }
}

/// TAA and FXAA over the world. See the [module docs](self).
pub struct AntiAliasPhase {
    mode: AaMode,
    /// 0..=1.
    sharpness: f32,
    /// FXAA also smooths edges inside textures (classic FXAA).
    texture_edges: bool,
    in_world: bool,
    /// The context is multisampled: the phase never runs (module docs).
    msaa: bool,
    velocity: Option<VelocityOutput>,
    /// The motion target's frame count last used: the same count twice means it is stale.
    velocity_frame: u64,
    history_valid: bool,
    jitter_frame: u64,
    pipelines: Option<Pipelines>,
    bindings: Option<Bindings>,
    history: Option<History>,
    /// What each uniform holds, so an unchanged frame uploads nothing.
    taa_uploaded: Option<Params>,
    fxaa_uploaded: Option<Params>,
    passes: u64,
}

impl std::fmt::Debug for AntiAliasPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AntiAliasPhase")
            .field("mode", &self.mode)
            .field("sharpness", &self.sharpness)
            .field("texture_edges", &self.texture_edges)
            .field("in_world", &self.in_world)
            .field("msaa", &self.msaa)
            .field("history", &self.history.is_some())
            .field("passes", &self.passes)
            .finish_non_exhaustive()
    }
}

impl AntiAliasPhase {
    /// The phase's name in `PhaseList`, GPU timings and captures.
    pub const NAME: &'static str = "post/aa";

    /// An idle phase (mode off, not in a world). Allocates nothing on the GPU.
    #[must_use]
    pub fn new(gpu: &Gpu) -> AntiAliasPhase {
        AntiAliasPhase {
            mode: AaMode::Off,
            sharpness: 0.0,
            texture_edges: false,
            in_world: false,
            msaa: gpu.sample_count() > 1,
            velocity: None,
            velocity_frame: 0,
            history_valid: false,
            jitter_frame: 0,
            pipelines: None,
            bindings: None,
            history: None,
            taa_uploaded: None,
            fxaa_uploaded: None,
            passes: 0,
        }
    }

    /// The motion target TAA reprojects with ([`crate::post::velocity::VelocityPhase::output`]).
    pub fn set_velocity(&mut self, velocity: VelocityOutput) {
        self.velocity = Some(velocity);
        self.bindings = None;
    }

    /// Choose the mode. Leaving TAA frees the history.
    pub fn set_mode(&mut self, mode: AaMode) {
        if mode != self.mode {
            self.mode = mode;
            self.release_unused();
        }
    }

    /// The mode.
    #[must_use]
    pub fn mode(&self) -> AaMode {
        self.mode
    }

    /// TAA sharpening, 0 (off) to 1, clamped.
    pub fn set_sharpness(&mut self, sharpness: f32) {
        self.sharpness = if sharpness.is_finite() {
            sharpness.clamp(0.0, 1.0)
        } else {
            0.0
        };
    }

    /// FXAA smooths edges inside textures too (classic FXAA) instead of geometry edges only.
    pub fn set_texture_edges(&mut self, on: bool) {
        self.texture_edges = on;
    }

    /// Whether a world is on screen. Outside one the phase is off; coming back starts over.
    pub fn set_in_world(&mut self, in_world: bool) {
        if in_world != self.in_world {
            self.in_world = in_world;
            self.release_unused();
        }
    }

    /// Drop the history: the next frame shows the current image alone (world change,
    /// teleport, respawn).
    pub fn reset_history(&mut self) {
        self.history_valid = false;
    }

    /// Whether the phase runs this frame.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.in_world && !self.msaa && self.mode != AaMode::Off
    }

    /// Whether the camera should be jittered this frame: TAA is running.
    #[must_use]
    pub fn wants_jitter(&self) -> bool {
        self.is_active() && self.mode == AaMode::Taa
    }

    /// The jitter for the next frame, in NDC, for [`Camera::jitter`]; zero unless TAA runs.
    /// Advances the sequence, so call it once per frame.
    pub fn next_jitter(&mut self, width: u32, height: u32) -> Vec2 {
        if !self.wants_jitter() {
            return Vec2::ZERO;
        }
        self.jitter_frame = self.jitter_frame.wrapping_add(1);
        jitter_ndc(self.jitter_frame, width, height)
    }

    /// Whether the TAA history textures exist (only while TAA runs).
    #[must_use]
    pub fn history_allocated(&self) -> bool {
        self.history.is_some()
    }

    /// How many passes the phase recorded (a sharpened TAA frame counts once).
    #[must_use]
    pub fn passes_recorded(&self) -> u64 {
        self.passes
    }

    fn release_unused(&mut self) {
        if !(self.is_active() && self.mode == AaMode::Taa) {
            self.history = None;
            self.history_valid = false;
        }
        if !self.is_active() {
            self.bindings = None;
        }
    }

    /// Build what this frame needs: pipelines for the frame format and bind groups for this
    /// snapshot, depth buffer and motion target.
    fn prepare(&mut self, gpu: &Gpu, scene: &SceneTextures<'_>) {
        if self
            .pipelines
            .as_ref()
            .is_none_or(|pipelines| pipelines.format != scene.format)
        {
            self.pipelines = Some(Pipelines::new(gpu, scene.format));
            self.bindings = None;
            self.history = None;
            self.history_valid = false;
            self.taa_uploaded = None;
            self.fxaa_uploaded = None;
        }
        let pipelines = self.pipelines.as_ref().expect("built above");
        let velocity_ref = self.velocity.as_ref().map(VelocityOutput::get);
        let velocity = velocity_ref.as_ref().and_then(|v| v.as_ref());
        let key = BindingKey {
            scene: scene.color.clone(),
            depth: gpu.depth_texture().clone(),
            velocity: velocity.map_or(0, |v| v.generation),
        };
        if self.bindings.as_ref().is_some_and(|b| b.key == key) {
            return;
        }
        let depth_view = key.depth.create_view(&wgpu::TextureViewDescriptor {
            label: Some("post/aa depth"),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..wgpu::TextureViewDescriptor::default()
        });
        let fxaa = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post/fxaa"),
            layout: &pipelines.fxaa_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(scene.color),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&pipelines.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pipelines.fxaa_uniform.as_entire_binding(),
                },
            ],
        });
        let taa = velocity.map(|velocity| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post/taa"),
                layout: &pipelines.taa_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(scene.color),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&velocity.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: velocity.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: pipelines.taa_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&pipelines.sampler),
                    },
                ],
            })
        });
        drop(velocity_ref);
        self.bindings = Some(Bindings { key, fxaa, taa });
    }

    fn record_fxaa(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        let pipelines = self.pipelines.as_ref().expect("built by prepare");
        let params = Params {
            params: [
                1.0 / scene.width.max(1) as f32,
                1.0 / scene.height.max(1) as f32,
                f32::from(u8::from(self.texture_edges)),
                cam.proj.z_axis.z,
            ],
        };
        if self.fxaa_uploaded != Some(params) {
            gpu.write_buffer_staged(&pipelines.fxaa_uniform, 0, bytemuck::bytes_of(&params));
            self.fxaa_uploaded = Some(params);
        }
        let group = &self.bindings.as_ref().expect("built by prepare").fxaa;
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("post/fxaa"),
            color_attachments: &[attachment(view)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipelines.fxaa);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        self.passes += 1;
    }

    fn record_taa(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        scene: &SceneTextures<'_>,
    ) {
        // The motion target this frame: fresh only if the velocity pass recorded since the
        // last resolve, and continuous only if it had a previous frame.
        let (fresh, continuous) = {
            let Some(output) = self.velocity.as_ref() else {
                return;
            };
            let frame = output.get();
            let Some(frame) = frame.as_ref() else {
                return;
            };
            let fresh = frame.frame != self.velocity_frame;
            self.velocity_frame = frame.frame;
            (fresh, frame.has_previous)
        };
        let pipelines = self.pipelines.as_ref().expect("built by prepare");
        let size = (scene.width, scene.height);
        if self
            .history
            .as_ref()
            .is_none_or(|history| (history.width, history.height) != size)
        {
            self.history = Some(History::new(gpu, pipelines, size.0, size.1));
            self.history_valid = false;
        }
        let Some(group) = self
            .bindings
            .as_ref()
            .and_then(|bindings| bindings.taa.as_ref())
        else {
            return;
        };
        let valid = self.history_valid && fresh && continuous;
        let params = Params {
            params: [
                STILL_CURRENT_WEIGHT,
                self.sharpness,
                f32::from(u8::from(valid)),
                0.0,
            ],
        };
        if self.taa_uploaded != Some(params) {
            gpu.write_buffer_staged(&pipelines.taa_uniform, 0, bytemuck::bytes_of(&params));
            self.taa_uploaded = Some(params);
        }
        let history = self.history.as_mut().expect("built above");
        let read = history.latest;
        let write = 1 - read;
        if self.sharpness <= 0.0 {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post/taa"),
                color_attachments: &[attachment(view), attachment(&history.views[write])],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipelines.taa_both);
            pass.set_bind_group(0, group, &[]);
            pass.set_bind_group(1, &history.reads[read], &[]);
            pass.draw(0..3, 0..1);
        } else {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post/taa"),
                color_attachments: &[attachment(&history.views[write])],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipelines.taa_history);
            pass.set_bind_group(0, group, &[]);
            pass.set_bind_group(1, &history.reads[read], &[]);
            pass.draw(0..3, 0..1);
            drop(pass);
            if history
                .sharpen
                .as_ref()
                .is_none_or(|(scene_view, _)| scene_view != scene.color)
            {
                let groups = std::array::from_fn(|i| {
                    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("post/taa sharpen"),
                        layout: &pipelines.sharpen_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&history.views[i]),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: pipelines.taa_uniform.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(scene.color),
                            },
                        ],
                    })
                });
                history.sharpen = Some((scene.color.clone(), groups));
            }
            let (_, groups) = history.sharpen.as_ref().expect("built above");
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post/taa sharpen"),
                color_attachments: &[attachment(view)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipelines.sharpen);
            pass.set_bind_group(0, &groups[write], &[]);
            pass.draw(0..3, 0..1);
        }
        history.latest = write;
        self.history_valid = true;
        self.passes += 1;
    }
}

impl RenderPhase for AntiAliasPhase {
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
        self.history = None;
        self.history_valid = false;
        self.bindings = None;
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
        cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        if !self.is_active() {
            return;
        }
        self.prepare(gpu, scene);
        match self.mode {
            AaMode::Off => {}
            AaMode::Fxaa => self.record_fxaa(gpu, enc, view, cam, scene),
            AaMode::Taa => self.record_taa(gpu, enc, view, scene),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Everything that draws into the world image from the (jittered) depth buffer records
    /// before the resolve; the labels and the hand after it (module docs).
    #[test]
    fn the_resolve_comes_after_everything_that_reads_jittered_depth() {
        use crate::PhaseOrder;
        assert!(crate::sky::fog::FogPhase::ORDER < PhaseOrder::TEMPORAL_AA);
        assert!(
            PhaseOrder(642) < PhaseOrder::TEMPORAL_AA,
            "the late volumetric clouds are world"
        );
        assert!(PhaseOrder::TEMPORAL_AA < PhaseOrder::VELOCITY_BLUR);
        assert!(
            PhaseOrder::VELOCITY_BLUR < PhaseOrder(645),
            "waypoint labels stay sharp"
        );
        assert!(PhaseOrder(645) < PhaseOrder::FIRST_PERSON);
    }

    #[test]
    fn halton_starts_as_the_textbook_sequence() {
        let base2: Vec<f32> = (1..=4).map(|i| halton(i, 2)).collect();
        assert_eq!(base2, [0.5, 0.25, 0.75, 0.125]);
        let base3: Vec<f32> = (1..=3).map(|i| halton(i, 3)).collect();
        assert!((base3[0] - 1.0 / 3.0).abs() < 1e-6 && (base3[2] - 1.0 / 9.0).abs() < 1e-6);
    }

    /// The 8 jitter positions are distinct, inside the pixel and centred on average, so the
    /// converged image is not shifted.
    #[test]
    fn the_jitter_cycle_covers_the_pixel_around_its_centre() {
        let offsets: Vec<Vec2> = (0..JITTER_PHASES).map(jitter_pixels).collect();
        for (i, a) in offsets.iter().enumerate() {
            assert!(a.x.abs() < 0.5 && a.y.abs() < 0.5, "{a}");
            for b in &offsets[i + 1..] {
                assert_ne!(a, b);
            }
        }
        let mean = offsets.iter().copied().sum::<Vec2>() / JITTER_PHASES as f32;
        assert!(mean.length() < 0.07, "mean offset {mean}");
        assert_eq!(jitter_pixels(3), jitter_pixels(3 + JITTER_PHASES));
    }

    #[test]
    fn one_pixel_of_jitter_is_two_over_the_size_in_ndc() {
        let px = jitter_pixels(5);
        let ndc = jitter_ndc(5, 1920, 1080);
        assert!((ndc.x - 2.0 * px.x / 1920.0).abs() < 1e-9);
        assert!((ndc.y + 2.0 * px.y / 1080.0).abs() < 1e-9, "NDC y is up");
    }

    #[test]
    fn the_wgsl_parses_and_validates() {
        for (name, source) in [
            ("taa.wgsl", include_str!("taa.wgsl")),
            ("taa_sharpen.wgsl", include_str!("taa_sharpen.wgsl")),
            ("fxaa.wgsl", include_str!("fxaa.wgsl")),
        ] {
            let module = wgpu::naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(source)));
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
        }
    }
}
