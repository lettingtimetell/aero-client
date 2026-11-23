//! Menu blur: blurs the game behind open screens.
//!
//! [`MenuBlurPhase`] is the render half of the `menu_blur` mod; the client glue is
//! `mc-client/src/app/mods/x10_blur.rs`. The client tells the phase between frames whether a
//! blurred screen is open ([`MenuBlurPhase::set_active`]), how far the blur reaches
//! ([`MenuBlurPhase::set_radius`]) and how far the fade-in has got ([`MenuBlurPhase::set_fade`]);
//! the phase does the rest.
//!
//! # Where it records: between the HUD and the open screen
//!
//! The HUD and the open screen are one `mc_ui` draw list in the UI phase ([`PhaseOrder::UI`]).
//! To blur the hotbar, chat and scoreboard with the world (the mod's default) the
//! client splits that list where the screen starts (`mc_ui::widget::UiPhase::set_split`): the UI
//! phase draws the HUD half, this phase records at [`PhaseOrder::MENU_BLUR`] and blurs everything
//! drawn so far, and `mc_ui::widget::UiScreenPhase` at [`PhaseOrder::UI_SCREEN`] draws the screen
//! half on top, sharp, with vanilla's dark gradient first. With the mod's "Blur HUD" off the
//! split sits at the start of the list and only the world is blurred. Without a split (the blur
//! off, or no blurred screen open) the UI phase draws its whole list exactly as before.
//!
//! # The blur: dual Kawase
//!
//! Level 0 is the frame so far (the [`SceneTextures`] snapshot); level `k` is half the size of
//! level `k - 1`. Each downsample takes five bilinear taps, each upsample eight, and the last
//! upsample is the composite into the frame, mixed with the sharp scene by the fade. A radius
//! picks how many levels to halve through and how far apart the taps sit ([`KawasePlan`]): the
//! blur's reach doubles with every level, so a 1080p frame needs two to five levels and the cost
//! stays a few cheap passes at half resolution and below (about 1.3 full-screen passes of
//! bandwidth in total). The levels are [`LEVEL_FORMAT`] (`Rgba16Float`) so the round trip adds no
//! banding to dark menu backgrounds.
//!
//! The radius is defined at [`REFERENCE_HEIGHT`] (1080p) and scaled with the frame height, so the
//! blur covers the same share of the screen at any resolution.
//!
//! # Off costs nothing
//!
//! While inactive the phase reports `enabled() == false` and `needs_scene_color() == false`, so
//! [`crate::PhaseList`] keeps its direct frame path byte for byte and never records it, and the
//! chain's textures do not exist: [`MenuBlurPhase::set_active`]`(false)` drops them. The three
//! pipelines and the uniform buffer are built on the first active frame and kept until
//! [`MenuBlurPhase::release`] (the client calls it when the mod is turned off).
//! `tests/x10_menu_blur.rs` proves both on the GPU.
//!
//! # MSAA
//!
//! The composite draws through [`Gpu::begin_pass`] with a pipeline of the context's sample count,
//! so under MSAA it writes the multisampled attachment and survives the next pass's resolve. The
//! chain itself is single-sampled; it reads the single-sampled snapshot.
//!
//! # For a future shader pack
//!
//! The phase is a plain [`RenderPhase`] that opts into [`SceneTextures`]; a shader pack that
//! wants its own menu backdrop registers a phase at the same slot (or replaces this one by name,
//! [`MenuBlurPhase::NAME`]) and gets the identical contract: the frame so far, world plus HUD,
//! as a readable snapshot, and the open screen drawn sharp on top of whatever it writes. The
//! chain's downsample pyramid is also the natural input for bloom; it is private for now.

use crate::{Camera, DepthAttachment, Gpu, RenderPhase, SceneTextures};

#[cfg(doc)]
use crate::PhaseOrder;

/// The format of the chain's levels. Sixteen-bit float keeps the ten-odd resampling passes from
/// banding the smooth, dark gradients a menu background mostly is.
pub const LEVEL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The most downsample levels a chain uses. At 1080p the sixth level is 30x17 pixels.
pub const MAX_LEVELS: u32 = 6;

/// The chain stops halving before a level's short side would drop below this many pixels.
const MIN_LEVEL_SIDE: u32 = 8;

/// The frame height the radius is defined at.
pub const REFERENCE_HEIGHT: f32 = 1080.0;

/// The largest radius the phase accepts, in physical pixels after scaling.
pub const MAX_RADIUS: f32 = 512.0;

/// The tap distance a level may stretch to before the chain adds a level instead. Dual Kawase
/// starts to show its sparse taps as a faint grid well beyond this.
const MAX_OFFSET: f32 = 2.5;

/// The tap distance the last level may stretch to when the frame is too small for another level.
const MAX_OFFSET_LAST_LEVEL: f32 = 4.0;

/// Bytes of one pass's uniform: `BlurPass` in `blur.wgsl`, two `vec4<f32>`.
const PASS_UNIFORM_SIZE: u64 = 32;

/// Passes in the longest chain: [`MAX_LEVELS`] downsamples and as many upsamples (the last of
/// which is the composite).
const MAX_PASSES: usize = 2 * MAX_LEVELS as usize;

/// How many downsample levels a `width` x `height` frame allows.
#[must_use]
pub fn max_levels(width: u32, height: u32) -> u32 {
    let mut side = width.min(height).max(1);
    let mut levels = 0;
    while levels < MAX_LEVELS && side.div_ceil(2) >= MIN_LEVEL_SIDE {
        side = side.div_ceil(2);
        levels += 1;
    }
    levels.max(1)
}

/// How one blur radius maps onto the chain: how many times to halve and how far apart the taps
/// sit.
///
/// The model behind it: a downsample adds a variance of `1/4 + offset²/2` source texels², an
/// upsample `1/6 + offset²/3`, and a level-`k` texel is `2^k` frame pixels wide. Summed over `L`
/// levels down and back up, the blur's standard deviation in frame pixels is
/// `sqrt((4^L - 1) / 3 * (11/12 + 11/6 * offset²))` ([`KawasePlan::sigma`]). The radius is read as
/// two standard deviations, roughly where a blurred edge has faded out.
/// `tests/x10_menu_blur.rs` measures the real spread on the GPU against this.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KawasePlan {
    /// Downsample levels, `1..=`[`MAX_LEVELS`].
    pub levels: u32,
    /// Tap distance in each pass's source texels.
    pub offset: f32,
}

impl KawasePlan {
    /// The plan for a blur radius of `radius_px` physical pixels on a `width` x `height` frame:
    /// the fewest levels that reach it with taps at most [`MAX_OFFSET`] apart, and the tap
    /// distance that hits the radius at that level. A radius below what one level gives at
    /// distance zero gets that (about two pixels).
    #[must_use]
    pub fn for_radius(radius_px: f32, width: u32, height: u32) -> KawasePlan {
        let radius = if radius_px.is_finite() {
            radius_px.clamp(0.0, MAX_RADIUS)
        } else {
            0.0
        };
        let sigma = radius * 0.5;
        let max = max_levels(width, height);
        let mut levels = 1;
        while levels < max && Self::sigma_of(levels, MAX_OFFSET) < sigma {
            levels += 1;
        }
        let cap = if levels == max {
            MAX_OFFSET_LAST_LEVEL
        } else {
            MAX_OFFSET
        };
        // sigma² = (4^L - 1) / 3 * (11/12 + 11/6 * offset²), solved for the offset.
        let scale = (4f32.powi(levels as i32) - 1.0) / 3.0;
        let offset_sq = (sigma * sigma / scale - 11.0 / 12.0) / (11.0 / 6.0);
        let offset = offset_sq.max(0.0).sqrt().min(cap);
        KawasePlan { levels, offset }
    }

    /// The blur's standard deviation in frame pixels (module docs on [`KawasePlan`]).
    #[must_use]
    pub fn sigma(self) -> f32 {
        Self::sigma_of(self.levels, self.offset)
    }

    fn sigma_of(levels: u32, offset: f32) -> f32 {
        let scale = (4f32.powi(levels as i32) - 1.0) / 3.0;
        (scale * (11.0 / 12.0 + 11.0 / 6.0 * offset * offset)).sqrt()
    }
}

/// The phase's pipelines, sampler and uniform buffer, built once for the frame's colour format
/// and sample count.
struct Pipelines {
    scene_format: wgpu::TextureFormat,
    samples: u32,
    chain_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    /// Bytes between two passes' uniforms (the device's uniform offset alignment).
    stride: u64,
}

impl Pipelines {
    fn new(gpu: &Gpu, scene_format: wgpu::TextureFormat) -> Pipelines {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post/menu-blur"),
            source: wgpu::ShaderSource::Wgsl(include_str!("blur.wgsl").into()),
        });
        let texture = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let sampler_entry = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(PASS_UNIFORM_SIZE),
            },
            count: None,
        };
        let chain_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post/menu-blur chain"),
            entries: &[texture(0), sampler_entry, uniform_entry],
        });
        let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post/menu-blur composite"),
            entries: &[texture(0), sampler_entry, uniform_entry, texture(3)],
        });
        let samples = gpu.sample_count();
        let down = pipeline(gpu, &shader, &chain_layout, "fs_down", LEVEL_FORMAT, 1);
        let up = pipeline(gpu, &shader, &chain_layout, "fs_up", LEVEL_FORMAT, 1);
        let composite = pipeline(
            gpu,
            &shader,
            &composite_layout,
            "fs_composite",
            scene_format,
            samples,
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post/menu-blur"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..wgpu::SamplerDescriptor::default()
        });
        let align = u64::from(device.limits().min_uniform_buffer_offset_alignment).max(1);
        let stride = PASS_UNIFORM_SIZE.div_ceil(align) * align;
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post/menu-blur passes"),
            size: stride * MAX_PASSES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Pipelines {
            scene_format,
            samples,
            chain_layout,
            composite_layout,
            down,
            up,
            composite,
            sampler,
            uniform,
            stride,
        }
    }

    /// Pass `slot`'s uniform, as a binding resource.
    fn slot(&self, slot: usize) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.uniform,
            offset: self.stride * slot as u64,
            size: wgpu::BufferSize::new(PASS_UNIFORM_SIZE),
        })
    }

    /// A bind group for a downsample or upsample sampling `source`.
    fn chain_group(&self, gpu: &Gpu, source: &wgpu::TextureView, slot: usize) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post/menu-blur pass"),
            layout: &self.chain_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.slot(slot),
                },
            ],
        })
    }
}

fn pipeline(
    gpu: &Gpu,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
    entry: &'static str,
    format: wgpu::TextureFormat,
    samples: u32,
) -> wgpu::RenderPipeline {
    let pipeline_layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(entry),
            bind_group_layouts: &[Some(layout)],
            immediate_size: 0,
        });
    gpu.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(entry),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: samples,
                ..wgpu::MultisampleState::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
}

/// One level of the pyramid (level `k` is `ceil(size / 2^k)`).
struct Level {
    width: u32,
    height: u32,
    /// Keeps its texture alive.
    view: wgpu::TextureView,
}

/// The pyramid for one frame size and level count, and the bind groups that never change with
/// it.
struct Chain {
    width: u32,
    height: u32,
    /// `levels[i]` is level `i + 1`.
    levels: Vec<Level>,
    /// `down_groups[i]` samples level `i + 1` to draw level `i + 2`.
    down_groups: Vec<wgpu::BindGroup>,
    /// `up_groups[j]` samples level `L - j` to draw level `L - j - 1` (`L` = level count); the
    /// composite, which samples level 1, is in [`SceneBinding`].
    up_groups: Vec<wgpu::BindGroup>,
}

impl Chain {
    fn new(gpu: &Gpu, pipelines: &Pipelines, width: u32, height: u32, count: u32) -> Chain {
        let mut levels = Vec::with_capacity(count as usize);
        let (mut w, mut h) = (width, height);
        for _ in 0..count {
            w = w.div_ceil(2).max(1);
            h = h.div_ceil(2).max(1);
            let view = gpu
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("post/menu-blur level"),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: LEVEL_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default());
            levels.push(Level {
                width: w,
                height: h,
                view,
            });
        }
        let n = count as usize;
        // Downsample pass `p` (0-based) draws level `p + 1` and uses uniform slot `p`; pass 0
        // samples the scene snapshot and lives in `SceneBinding`.
        let down_groups = (1..n)
            .map(|p| pipelines.chain_group(gpu, &levels[p - 1].view, p))
            .collect();
        // Upsample `j` draws level `n - 1 - j` from level `n - j`, uniform slot `n + j`.
        let up_groups = (0..n.saturating_sub(1))
            .map(|j| pipelines.chain_group(gpu, &levels[n - j - 1].view, n + j))
            .collect();
        Chain {
            width,
            height,
            levels,
            down_groups,
            up_groups,
        }
    }

    fn level_count(&self) -> u32 {
        self.levels.len() as u32
    }
}

/// The two bind groups that sample the scene snapshot: the first downsample and the composite.
/// `PhaseList` rebuilds its snapshot only on a resize or format change, so these are rebuilt then
/// (and with the chain), not per frame.
struct SceneBinding {
    /// A clone of the snapshot view, compared by identity every frame; holding it keeps the view
    /// alive, so a new view can never reuse its address while it is compared against.
    view: wgpu::TextureView,
    first_down: wgpu::BindGroup,
    composite: wgpu::BindGroup,
}

impl SceneBinding {
    fn new(
        gpu: &Gpu,
        pipelines: &Pipelines,
        chain: &Chain,
        scene: &wgpu::TextureView,
    ) -> SceneBinding {
        let n = chain.levels.len();
        let first_down = pipelines.chain_group(gpu, scene, 0);
        let composite = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post/menu-blur composite"),
            layout: &pipelines.composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&chain.levels[0].view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&pipelines.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pipelines.slot(2 * n - 1),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(scene),
                },
            ],
        });
        SceneBinding {
            view: scene.clone(),
            first_down,
            composite,
        }
    }
}

/// Everything the uniform buffer's contents follow from, so an unchanged frame uploads nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UniformKey {
    width: u32,
    height: u32,
    levels: u32,
    offset_bits: u32,
    fade_bits: u32,
}

/// A colour attachment the pass overwrites completely.
fn level_pass<'e>(
    enc: &'e mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
) -> wgpu::RenderPass<'e> {
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("post/menu-blur level"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

/// The menu blur. See the [module docs](self).
pub struct MenuBlurPhase {
    active: bool,
    /// The radius at [`REFERENCE_HEIGHT`], in pixels.
    radius: f32,
    /// How much of the blur shows, `0..=1` (the fade-in).
    fade: f32,
    pipelines: Option<Pipelines>,
    chain: Option<Chain>,
    scene_binding: Option<SceneBinding>,
    uploaded: Option<UniformKey>,
    /// Reused upload bytes, so a changed frame allocates nothing once warm.
    scratch: Vec<u8>,
    passes: u64,
}

impl std::fmt::Debug for MenuBlurPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MenuBlurPhase")
            .field("active", &self.active)
            .field("radius", &self.radius)
            .field("fade", &self.fade)
            .field("pipelines", &self.pipelines.is_some())
            .field("levels", &self.chain.as_ref().map(Chain::level_count))
            .field("passes", &self.passes)
            .finish_non_exhaustive()
    }
}

impl Default for MenuBlurPhase {
    fn default() -> Self {
        Self::new()
    }
}

impl MenuBlurPhase {
    /// The phase's name in `PhaseList`, GPU timings and captures.
    pub const NAME: &'static str = "post/menu-blur";

    /// The radius a new phase starts with, in pixels at 1080p (the mod's default: text behind
    /// the menu becomes unreadable, shapes and colours stay).
    pub const DEFAULT_RADIUS: f32 = 20.0;

    /// An idle phase. Allocates nothing on the GPU; that happens on its first active frame.
    #[must_use]
    pub fn new() -> MenuBlurPhase {
        MenuBlurPhase {
            active: false,
            radius: Self::DEFAULT_RADIUS,
            fade: 1.0,
            pipelines: None,
            chain: None,
            scene_binding: None,
            uploaded: None,
            scratch: Vec::new(),
            passes: 0,
        }
    }

    /// Whether a blurred screen is open. Turning it off drops the chain's textures at once (the
    /// pipelines stay until [`Self::release`]).
    pub fn set_active(&mut self, active: bool) {
        if active != self.active {
            self.active = active;
            if !active {
                self.drop_targets();
            }
        }
    }

    /// Whether the phase runs this frame.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Set the blur radius in pixels at [`REFERENCE_HEIGHT`]; it scales with the frame height.
    pub fn set_radius(&mut self, radius: f32) {
        self.radius = if radius.is_finite() {
            radius.clamp(0.0, MAX_RADIUS)
        } else {
            Self::DEFAULT_RADIUS
        };
    }

    /// The blur radius in pixels at [`REFERENCE_HEIGHT`].
    #[must_use]
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// Set how much of the blur shows: 0 is the sharp scene, 1 the full blur. The fade scales the
    /// taps' distance and mixes the result with the sharp scene, so the blur builds up instead of
    /// cross-fading.
    pub fn set_fade(&mut self, fade: f32) {
        self.fade = if fade.is_finite() {
            fade.clamp(0.0, 1.0)
        } else {
            1.0
        };
    }

    /// How much of the blur shows.
    #[must_use]
    pub fn fade(&self) -> f32 {
        self.fade
    }

    /// Drop every GPU resource, pipelines included, and go idle. The client calls this when the
    /// mod is turned off; the next active frame rebuilds what it needs.
    pub fn release(&mut self) {
        self.active = false;
        self.drop_targets();
        self.pipelines = None;
    }

    /// Whether the chain's textures exist (only while a blurred screen is open).
    #[must_use]
    pub fn targets_allocated(&self) -> bool {
        self.chain.is_some()
    }

    /// Whether the pipelines exist (from the first active frame until [`Self::release`]).
    #[must_use]
    pub fn pipelines_built(&self) -> bool {
        self.pipelines.is_some()
    }

    /// How many frames this phase has blurred. Stays put while it is inactive.
    #[must_use]
    pub fn passes_recorded(&self) -> u64 {
        self.passes
    }

    /// The plan the phase uses on a `width` x `height` frame at the current radius.
    #[must_use]
    pub fn plan(&self, width: u32, height: u32) -> KawasePlan {
        let scale = height.max(1) as f32 / REFERENCE_HEIGHT;
        KawasePlan::for_radius(self.radius * scale, width.max(1), height.max(1))
    }

    fn drop_targets(&mut self) {
        self.chain = None;
        self.scene_binding = None;
        self.uploaded = None;
    }

    /// Build or rebuild what this frame needs.
    fn prepare(&mut self, gpu: &Gpu, scene: &SceneTextures<'_>, levels: u32) {
        let samples = gpu.sample_count();
        if self
            .pipelines
            .as_ref()
            .is_none_or(|p| p.scene_format != scene.format || p.samples != samples)
        {
            self.pipelines = Some(Pipelines::new(gpu, scene.format));
            self.drop_targets();
        }
        let pipelines = self.pipelines.as_ref().expect("built above");
        let (width, height) = (scene.width.max(1), scene.height.max(1));
        if self.chain.as_ref().is_none_or(|chain| {
            (chain.width, chain.height, chain.level_count()) != (width, height, levels)
        }) {
            self.chain = Some(Chain::new(gpu, pipelines, width, height, levels));
            self.scene_binding = None;
            self.uploaded = None;
        }
        let chain = self.chain.as_ref().expect("built above");
        if self
            .scene_binding
            .as_ref()
            .is_none_or(|binding| binding.view != *scene.color)
        {
            self.scene_binding = Some(SceneBinding::new(gpu, pipelines, chain, scene.color));
        }
    }

    /// Write every pass's uniform when anything it follows from changed.
    fn upload(&mut self, gpu: &Gpu, plan: KawasePlan) {
        let (Some(pipelines), Some(chain)) = (self.pipelines.as_ref(), self.chain.as_ref()) else {
            return;
        };
        let offset = plan.offset * self.fade;
        let key = UniformKey {
            width: chain.width,
            height: chain.height,
            levels: chain.level_count(),
            offset_bits: offset.to_bits(),
            fade_bits: self.fade.to_bits(),
        };
        if self.uploaded == Some(key) {
            return;
        }
        let n = chain.levels.len();
        let size = |level: usize| -> (f32, f32) {
            if level == 0 {
                (chain.width as f32, chain.height as f32)
            } else {
                let l = &chain.levels[level - 1];
                (l.width as f32, l.height as f32)
            }
        };
        let passes = 2 * n;
        let stride = pipelines.stride as usize;
        self.scratch.clear();
        self.scratch.resize(stride * passes, 0);
        let mut write = |slot: usize, source: usize, target: usize, mix: f32| {
            let (sw, sh) = size(source);
            let (tw, th) = size(target);
            let values = [
                1.0 / sw,
                1.0 / sh,
                1.0 / tw,
                1.0 / th,
                offset,
                mix,
                0.0,
                0.0,
            ];
            let at = slot * stride;
            for (i, value) in values.iter().enumerate() {
                self.scratch[at + i * 4..at + i * 4 + 4].copy_from_slice(&value.to_le_bytes());
            }
        };
        for p in 0..n {
            write(p, p, p + 1, 0.0);
        }
        for j in 0..n - 1 {
            write(n + j, n - j, n - j - 1, 0.0);
        }
        write(2 * n - 1, 1, 0, self.fade);
        gpu.write_buffer_staged(&pipelines.uniform, 0, &self.scratch);
        self.uploaded = Some(key);
    }
}

impl RenderPhase for MenuBlurPhase {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn enabled(&self) -> bool {
        self.active
    }

    fn needs_scene_color(&self) -> bool {
        self.active
    }

    fn resize(&mut self, _gpu: &Gpu, _width: u32, _height: u32) {
        // A new size means a new snapshot and a pyramid of the wrong size.
        self.drop_targets();
    }

    fn record(
        &mut self,
        _gpu: &Gpu,
        _enc: &mut wgpu::CommandEncoder,
        _view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        // Only reachable when driven outside `PhaseList`: without a snapshot to read there is
        // nothing correct to draw, so draw nothing.
    }

    fn record_scene(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        if !self.active {
            return;
        }
        let plan = self.plan(scene.width, scene.height);
        self.prepare(gpu, scene, plan.levels);
        self.upload(gpu, plan);
        let (Some(pipelines), Some(chain), Some(binding)) = (
            self.pipelines.as_ref(),
            self.chain.as_ref(),
            self.scene_binding.as_ref(),
        ) else {
            return;
        };
        let n = chain.levels.len();
        for p in 0..n {
            let group = if p == 0 {
                &binding.first_down
            } else {
                &chain.down_groups[p - 1]
            };
            let mut pass = level_pass(enc, &chain.levels[p].view);
            pass.set_pipeline(&pipelines.down);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
        for (j, group) in chain.up_groups.iter().enumerate() {
            // Draws level `n - 1 - j`, which is `levels[n - 2 - j]`.
            let mut pass = level_pass(enc, &chain.levels[n - 2 - j].view);
            pass.set_pipeline(&pipelines.up);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
        let mut pass = gpu.begin_pass(enc, view, Self::NAME, DepthAttachment::None);
        pass.set_pipeline(&pipelines.composite);
        pass.set_bind_group(0, &binding.composite, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        self.passes += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_1080p_frame_allows_six_levels_and_a_tiny_one_at_least_one() {
        assert_eq!(max_levels(1920, 1080), 6);
        assert_eq!(max_levels(854, 480), 6);
        assert_eq!(max_levels(200, 100), 3);
        assert_eq!(max_levels(16, 16), 1);
        assert_eq!(max_levels(1, 1), 1);
    }

    #[test]
    fn the_plan_hits_the_radius_with_the_fewest_levels() {
        for radius in [4.0f32, 8.0, 14.0, 28.0, 48.0, 96.0] {
            let plan = KawasePlan::for_radius(radius, 1920, 1080);
            assert!(
                plan.levels >= 1 && plan.levels <= MAX_LEVELS,
                "{radius}: {plan:?}"
            );
            assert!(plan.offset <= MAX_OFFSET, "{radius}: {plan:?}");
            // Two standard deviations land on the radius.
            assert!(
                (plan.sigma() * 2.0 - radius).abs() < 0.01 * radius,
                "{radius}: {plan:?} reaches {}",
                plan.sigma() * 2.0
            );
            // One level fewer could not have reached it.
            if plan.levels > 1 {
                let fewer = KawasePlan::sigma_of(plan.levels - 1, MAX_OFFSET);
                assert!(fewer * 2.0 < radius, "{radius}: {plan:?}");
            }
        }
        // More radius never means fewer levels.
        let mut last = 0;
        for radius in 2..200 {
            let levels = KawasePlan::for_radius(radius as f32, 1920, 1080).levels;
            assert!(levels >= last, "{radius}");
            last = levels;
        }
    }

    #[test]
    fn a_radius_below_one_level_gets_the_smallest_blur_and_garbage_is_harmless() {
        let tiny = KawasePlan::for_radius(0.5, 1920, 1080);
        assert_eq!(
            tiny,
            KawasePlan {
                levels: 1,
                offset: 0.0
            }
        );
        assert_eq!(KawasePlan::for_radius(f32::NAN, 1920, 1080), tiny);
        assert_eq!(KawasePlan::for_radius(-5.0, 1920, 1080), tiny);
        // A small frame caps the levels and stretches the last one's taps instead.
        let capped = KawasePlan::for_radius(400.0, 64, 64);
        assert_eq!(capped.levels, max_levels(64, 64));
        assert!(capped.offset <= MAX_OFFSET_LAST_LEVEL);
    }

    #[test]
    fn the_radius_scales_with_the_frame_height() {
        let mut phase = MenuBlurPhase::new();
        phase.set_radius(14.0);
        let p1080 = phase.plan(1920, 1080);
        let p2160 = phase.plan(3840, 2160);
        assert!((p2160.sigma() - 2.0 * p1080.sigma()).abs() < 0.05 * p2160.sigma());
        phase.set_radius(f32::INFINITY);
        assert_eq!(phase.radius(), MenuBlurPhase::DEFAULT_RADIUS);
    }

    #[test]
    fn an_idle_phase_opts_out_and_owns_nothing() {
        let mut phase = MenuBlurPhase::new();
        assert!(!phase.enabled());
        assert!(!phase.needs_scene_color());
        assert!(!phase.targets_allocated() && !phase.pipelines_built());
        phase.set_active(true);
        assert!(phase.enabled() && phase.needs_scene_color());
        // Nothing is built until a frame records it.
        assert!(!phase.targets_allocated() && !phase.pipelines_built());
        phase.set_active(false);
        assert!(!phase.enabled() && !phase.needs_scene_color());
        phase.set_fade(f32::NAN);
        assert_eq!(phase.fade(), 1.0);
        phase.set_fade(-1.0);
        assert_eq!(phase.fade(), 0.0);
    }
}
