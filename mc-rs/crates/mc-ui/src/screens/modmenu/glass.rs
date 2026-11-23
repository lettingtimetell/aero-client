//! The menu's render phase: Aero liquid glass over the game's own frame, then the menu content.
//!
//! The glass is a direct WGSL port of the `FRAGMENT` shader in
//! the Aero web prototype's `glass.js` (rounded-rect SDF masks with anti-aliasing, the edge
//! factor, the lens pull and three time-animated waves, the RGB prism, mip-bias frost, the
//! `pow(causticWave, 14)` caustic, the edge tint and the `uDim` darkening). What glass.js needs
//! a screen capture, a pointer-free history texture and a cursor mask for, this phase gets for
//! free: it owns the frame. It opts into [`RenderPhase::needs_scene_color`], so the phase list
//! hands it the frame as drawn so far (world, hand, HUD); it copies that into a mipmapped
//! texture, builds the mip chain with a few box-filter passes (wgpu has no automatic mipmaps),
//! and samples it with `textureSampleBias` exactly where glass.js uses `texture2D(.., bias)`.
//!
//! Two deliberate additions, both needed by the open animation: each rectangle carries an
//! alpha that scales its mask, and the dim and strength are ramped by the caller.
//!
//! **Cost when closed: none.** [`RenderPhase::enabled`] is false, so the phase list skips it
//! and, with no phase asking for the scene, takes its normal direct-to-swapchain path.

use mc_assets::TextureMetadata;
use mc_render::{Camera, DepthAttachment, Gpu, RenderPhase, SceneTextures};

use crate::widget::{DrawList, ScaledResolution, UiRenderer};

/// glass.js `MAX_RECTS`.
pub const MAX_RECTS: usize = 24;
/// Mip levels built for the frost (bias reaches about 2.7).
const MIP_LEVELS: u32 = 6;

/// The `FRAGMENT` shader of glass.js, ported to WGSL line for line where WGSL allows.
const GLASS_WGSL: &str = r#"
const MAX_RECTS: u32 = 24u;

struct Glass {
    resolution: vec2<f32>,   // uResolution: canvas size, render px
    px_per_dip: f32,         // uPxPerDip
    time: f32,               // uTime
    dim: f32,                // uDim
    strength: f32,           // uStrength
    frost: f32,              // uFrost: mip bias inside glass, higher is frostier
    count: u32,              // uCount
    rects: array<vec4<f32>, 24>,   // uRects
    params: array<vec4<f32>, 24>,  // x = uRadii[i]; y = this port's open-animation alpha
};

@group(0) @binding(0) var<uniform> g: Glass;
@group(0) @binding(1) var u_scene: texture_2d<f32>;   // uScene: the frame behind the glass
@group(0) @binding(2) var u_scene_sampler: sampler;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

// vec2 sceneUv(vec2 px) { return clamp(uTexOrigin + px * uTexScale, 0.0, 1.0); }
// The scene is this frame, so uTexOrigin = 0 and uTexScale = 1 / uResolution.
fn scene_uv(px: vec2<f32>) -> vec2<f32> {
    return clamp(px / g.resolution, vec2<f32>(0.0), vec2<f32>(1.0));
}

fn rounded_distance(p: vec2<f32>, rect: vec4<f32>, radius: f32) -> f32 {
    let half_size = rect.zw * 0.5;
    let r = min(radius, min(half_size.x, half_size.y));
    let q = abs(p - (rect.xy + half_size)) - half_size + r;
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r;
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    // vec2 local = vec2(gl_FragCoord.x, uResolution.y - gl_FragCoord.y): WGSL's position is
    // already top-left based.
    let local = frag.xy;
    let base = textureSampleBias(u_scene, u_scene_sampler, scene_uv(local), 0.4).rgb;
    let shade = 1.0 - g.dim;

    var glass_mask = 0.0;
    var edge = 0.0;
    var lens = vec2<f32>(0.0);
    var lens_weight = 0.0;
    let aa = 1.2 * g.px_per_dip;
    for (var i = 0u; i < min(g.count, MAX_RECTS); i = i + 1u) {
        let rect = g.rects[i];
        let d = rounded_distance(local, rect, g.params[i].x);
        let mask = (1.0 - smoothstep(-aa, aa, d)) * g.params[i].y;
        // `if (mask <= 0.0) continue;` as a plain `if`, which keeps the samples below in
        // uniform control flow.
        if (mask > 0.0) {
            glass_mask = max(glass_mask, mask);
            edge = max(edge, mask * (1.0 - smoothstep(0.0, min(min(rect.z, rect.w) * 0.2, 36.0 * g.px_per_dip), max(0.0, -d))));
            lens = lens + (local - (rect.xy + rect.zw * 0.5)) / max(rect.zw, vec2<f32>(1.0)) * mask;
            lens_weight = lens_weight + mask;
        }
    }
    lens = lens / max(lens_weight, 0.001);

    let dip = local / g.px_per_dip;
    let wave_a = sin(dip.y * 0.011 + g.time * 0.34);
    let wave_b = cos(dip.x * 0.0085 - g.time * 0.28);
    let wave_c = sin((dip.x + dip.y) * 0.005 + g.time * 0.21);
    let bend = vec2<f32>(wave_a + wave_c * 0.5, wave_b + wave_c * 0.35) * (2.2 + edge * 4.4) + lens * (3.0 + edge * 9.0);
    let bent_px = local + bend * g.strength * g.px_per_dip;

    let pulse = 0.5 + 0.5 * sin((dip.x - dip.y) * 0.003 + g.time * 0.2);
    let prism = vec2<f32>(1.0, 0.3) * (2.0 + edge * 3.0 + pulse * 0.5) * g.strength * g.px_per_dip;
    let bias = g.frost + edge * 0.8;

    var refracted: vec3<f32>;
    refracted.r = textureSampleBias(u_scene, u_scene_sampler, scene_uv(bent_px + prism), bias).r;
    refracted.g = textureSampleBias(u_scene, u_scene_sampler, scene_uv(bent_px), bias).g;
    refracted.b = textureSampleBias(u_scene, u_scene_sampler, scene_uv(bent_px - prism), bias).b;

    let caustic_wave = 0.5 + 0.5 * sin(dip.x * 0.018 + wave_a * 2.0 + g.time * 0.44);
    refracted = refracted + pow(caustic_wave, 14.0) * (0.018 + edge * 0.04) * g.strength * vec3<f32>(1.0, 0.95, 0.82);
    refracted = refracted + edge * vec3<f32>(0.02, 0.03, 0.045);

    // glass.js returns `base * shade` early when glassMask < .001. WGSL wants the samples
    // above in uniform control flow, so both answers exist and the mask picks.
    let glass = mix(base, refracted, glass_mask) * shade;
    return vec4<f32>(select(glass, base * shade, glass_mask < 0.001), 1.0);
}
"#;

/// Scene copy and box-filter downsampling for the mip chain.
const MIP_WGSL: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var lin: sampler;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

// Level 0: an exact 1:1 copy of the scene snapshot.
@fragment
fn fs_copy(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(src, vec2<i32>(p.xy), 0);
}

// Level n from n-1: one bilinear tap at the centre of each 2x2 footprint is a box filter.
@fragment
fn fs_down(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let dst = max(vec2<u32>(1u), textureDimensions(src) / 2u);
    return textureSampleLevel(src, lin, p.xy / vec2<f32>(dst), 0.0);
}
"#;

/// One glass pane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlassRect {
    /// `x, y, w, h` in physical pixels.
    pub rect: [f32; 4],
    /// Corner radius in physical pixels.
    pub radius: f32,
    /// The pane's opacity (open animation), `0..=1`.
    pub alpha: f32,
}

/// The glass uniforms for one frame.
#[derive(Debug, Clone, PartialEq)]
pub struct GlassFrame {
    /// The panes (at most [`MAX_RECTS`]).
    pub rects: Vec<GlassRect>,
    /// `uDim`: darkening of the whole frame (glass.js default 0.35).
    pub dim: f32,
    /// `uStrength`: refraction strength (glass.js default 1).
    pub strength: f32,
    /// `uFrost`: mip bias inside the glass (glass.js default 1.5).
    pub frost: f32,
    /// `uPxPerDip`.
    pub px_per_dip: f32,
    /// `uTime` in seconds.
    pub time: f32,
}

impl Default for GlassFrame {
    fn default() -> Self {
        GlassFrame {
            rects: Vec::new(),
            dim: 0.0,
            strength: 1.0,
            frost: 1.5,
            px_per_dip: 1.0,
            time: 0.0,
        }
    }
}

impl GlassFrame {
    /// Whether the glass pass has anything to do (otherwise only the content is drawn).
    #[must_use]
    pub fn is_visible(&self) -> bool {
        self.dim > 0.0 || self.rects.iter().any(|r| r.alpha > 0.0)
    }

    /// The uniform block's bytes (800, std140-compatible).
    #[must_use]
    pub fn uniform_bytes(&self, width: u32, height: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(800);
        let mut f = |v: f32| out.extend_from_slice(&v.to_le_bytes());
        f(width as f32);
        f(height as f32);
        f(self.px_per_dip.max(0.01));
        f(self.time);
        f(self.dim);
        f(self.strength);
        f(self.frost);
        let count = self.rects.len().min(MAX_RECTS) as u32;
        out.extend_from_slice(&count.to_le_bytes());
        for i in 0..MAX_RECTS {
            let rect = self.rects.get(i).map_or([0.0; 4], |r| r.rect);
            for v in rect {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        for i in 0..MAX_RECTS {
            let (radius, alpha) = self
                .rects
                .get(i)
                .map_or((0.0, 0.0), |r| (r.radius, r.alpha.clamp(0.0, 1.0)));
            for v in [radius, alpha, 0.0, 0.0] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }
}

struct MipChain {
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    _texture: wgpu::Texture,
    /// The whole chain, sampled by the glass pass (through `glass_group`; kept with it).
    _full_view: wgpu::TextureView,
    /// One single-level view per mip, as render targets.
    level_views: Vec<wgpu::TextureView>,
    /// `down_groups[n]` samples level `n` (to draw level `n + 1`).
    down_groups: Vec<wgpu::BindGroup>,
    glass_group: wgpu::BindGroup,
}

/// The mod menu's render phase. Register at `PhaseOrder(PhaseOrder::UI.0 + 50)`, after the UI.
pub struct ModMenuPhase {
    active: bool,
    glass: GlassFrame,
    /// A private UI renderer for the menu content (its own projection: physical pixels).
    pub renderer: UiRenderer,
    list: DrawList,
    spare: DrawList,
    resolution: Option<(u32, u32)>,
    uniform: wgpu::Buffer,
    glass_layout: wgpu::BindGroupLayout,
    mip_layout: wgpu::BindGroupLayout,
    glass_pipeline: wgpu::RenderPipeline,
    copy_pipeline: Option<(
        wgpu::TextureFormat,
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
    )>,
    mip_shader: wgpu::ShaderModule,
    sampler: wgpu::Sampler,
    mips: Option<MipChain>,
}

impl std::fmt::Debug for ModMenuPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModMenuPhase")
            .field("active", &self.active)
            .field("glass", &self.glass)
            .finish_non_exhaustive()
    }
}

fn fullscreen_pipeline(
    gpu: &Gpu,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    fragment: &str,
    format: wgpu::TextureFormat,
    samples: u32,
) -> wgpu::RenderPipeline {
    let pipeline_layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(layout)],
            immediate_size: 0,
        });
    gpu.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
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
                entry_point: Some(fragment),
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

/// A render pass that clears and fills one mip level.
fn mip_pass<'e>(
    enc: &'e mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
) -> wgpu::RenderPass<'e> {
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("modmenu glass mip"),
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

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
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

impl ModMenuPhase {
    /// Build the pipelines. Cheap enough for start-up; nothing runs per frame until the menu
    /// opens.
    #[must_use]
    pub fn new(gpu: &Gpu) -> ModMenuPhase {
        let glass_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("modmenu glass"),
                source: wgpu::ShaderSource::Wgsl(GLASS_WGSL.into()),
            });
        let mip_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("modmenu glass mips"),
                source: wgpu::ShaderSource::Wgsl(MIP_WGSL.into()),
            });
        let glass_layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("modmenu glass group"),
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
                    texture_entry(1),
                    sampler_entry(2),
                ],
            });
        let mip_layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("modmenu mip group"),
                entries: &[texture_entry(0), sampler_entry(1)],
            });
        let glass_pipeline = fullscreen_pipeline(
            gpu,
            "modmenu glass",
            &glass_layout,
            &glass_shader,
            "fs",
            gpu.config.format,
            gpu.sample_count(),
        );
        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("modmenu glass uniform"),
            size: 800,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // glass.js: CLAMP_TO_EDGE, LINEAR, LINEAR_MIPMAP_LINEAR.
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("modmenu glass sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..wgpu::SamplerDescriptor::default()
        });
        let mut list = DrawList::new();
        list.enable_solid_merging();
        let mut spare = DrawList::new();
        spare.enable_solid_merging();
        ModMenuPhase {
            active: false,
            glass: GlassFrame::default(),
            renderer: UiRenderer::new(gpu),
            list,
            spare,
            resolution: None,
            uniform,
            glass_layout,
            mip_layout,
            glass_pipeline,
            copy_pipeline: None,
            mip_shader,
            sampler,
            mips: None,
        }
    }

    /// Whether the menu (or its closing animation) is on screen.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// An empty list (with solid merging on) to build the next frame into, reusing the
    /// allocations of the one before last.
    pub fn take_spare_list(&mut self) -> DrawList {
        let mut list = std::mem::take(&mut self.spare);
        list.clear();
        if !list.solid_merging() {
            list.enable_solid_merging();
        }
        list
    }

    /// Hand over this frame's content (physical pixels) and glass.
    pub fn submit(&mut self, list: DrawList, glass: GlassFrame) {
        let previous = std::mem::replace(&mut self.list, list);
        self.spare = previous;
        self.glass = glass;
        self.active = true;
    }

    /// The menu is gone: stop drawing (and stop asking for the scene).
    pub fn deactivate(&mut self) {
        if self.active {
            self.active = false;
            self.list.clear();
            self.glass.rects.clear();
            // The screen-sized blur chain goes with the menu (about
            // 12 MiB of GPU memory at 1080p); the next open makes it again in `ensure_mips`,
            // and every level is rewritten from the scene each frame before the glass samples
            // it, so the picture is the same. `MC_W21_MR_GLASS_RELEASE=0` keeps it.
            if mc_render::w21_mr::glass_release_enabled() {
                self.mips = None;
            }
        }
    }

    /// Whether the glass's blur chain is allocated (tests).
    #[must_use]
    pub fn glass_chain_allocated(&self) -> bool {
        self.mips.is_some()
    }

    /// The submitted glass uniforms (tests).
    #[must_use]
    pub fn glass(&self) -> &GlassFrame {
        &self.glass
    }

    /// Upload the Segoe UI glyph atlas (linear filtering, clamped).
    pub fn upload_atlas(&mut self, gpu: &Gpu, rgba: &[u8], size: u32) {
        let options = TextureMetadata {
            blur: true,
            clamp: true,
            ..TextureMetadata::default()
        };
        self.renderer.upload_with_options(
            gpu,
            super::text::atlas_key(),
            rgba,
            size,
            size,
            &options,
        );
    }

    fn ensure_mips(&mut self, gpu: &Gpu, scene: &SceneTextures<'_>) {
        let (width, height, format) = (scene.width.max(1), scene.height.max(1), scene.format);
        if self
            .mips
            .as_ref()
            .is_some_and(|m| m.width == width && m.height == height && m.format == format)
        {
            return;
        }
        if self
            .copy_pipeline
            .as_ref()
            .is_none_or(|(f, _, _)| *f != format)
        {
            let copy = fullscreen_pipeline(
                gpu,
                "modmenu glass copy",
                &self.mip_layout,
                &self.mip_shader,
                "fs_copy",
                format,
                1,
            );
            let down = fullscreen_pipeline(
                gpu,
                "modmenu glass downsample",
                &self.mip_layout,
                &self.mip_shader,
                "fs_down",
                format,
                1,
            );
            self.copy_pipeline = Some((format, copy, down));
        }
        let max_levels = 32 - width.max(height).leading_zeros();
        let levels = MIP_LEVELS.min(max_levels).max(1);
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("modmenu glass scene"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let full_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let level_views: Vec<_> = (0..levels)
            .map(|level| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("modmenu glass mip"),
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..wgpu::TextureViewDescriptor::default()
                })
            })
            .collect();
        let down_groups = level_views
            .iter()
            .take(levels.saturating_sub(1) as usize)
            .map(|view| {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("modmenu glass mip source"),
                    layout: &self.mip_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                })
            })
            .collect();
        let glass_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("modmenu glass"),
            layout: &self.glass_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&full_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.mips = Some(MipChain {
            width,
            height,
            format,
            _texture: texture,
            _full_view: full_view,
            level_views,
            down_groups,
            glass_group,
        });
    }

    fn sync_resolution(&mut self, gpu: &Gpu) {
        let size = (gpu.config.width.max(1), gpu.config.height.max(1));
        if self.resolution != Some(size) {
            self.resolution = Some(size);
            self.renderer
                .set_resolution(gpu, &ScaledResolution::new(size.0, size.1, 1, false));
        }
    }

    fn record_glass(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        scene: &SceneTextures<'_>,
    ) {
        self.ensure_mips(gpu, scene);
        let (Some(mips), Some((_, copy, down))) = (&self.mips, &self.copy_pipeline) else {
            return;
        };
        // Level 0: the scene as drawn so far.
        let scene_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("modmenu glass scene source"),
            layout: &self.mip_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(scene.color),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        {
            let mut pass = mip_pass(enc, &mips.level_views[0]);
            pass.set_pipeline(copy);
            pass.set_bind_group(0, &scene_group, &[]);
            pass.draw(0..3, 0..1);
        }
        for (level, group) in mips.down_groups.iter().enumerate() {
            let mut pass = mip_pass(enc, &mips.level_views[level + 1]);
            pass.set_pipeline(down);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
        let bytes = self.glass.uniform_bytes(mips.width, mips.height);
        gpu.write_buffer_staged(&self.uniform, 0, &bytes);
    }

    fn record_pass(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        with_glass: bool,
    ) {
        self.sync_resolution(gpu);
        self.renderer.upload_vertices(gpu, &self.list);
        let mut pass = gpu.begin_pass(enc, view, "modmenu", DepthAttachment::None);
        if with_glass {
            if let Some(mips) = &self.mips {
                pass.set_pipeline(&self.glass_pipeline);
                pass.set_bind_group(0, &mips.glass_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        if !self.list.is_empty() {
            self.renderer.draw(&mut pass, &self.list);
        }
    }
}

impl RenderPhase for ModMenuPhase {
    fn name(&self) -> &'static str {
        "modmenu"
    }

    fn enabled(&self) -> bool {
        self.active
    }

    fn needs_scene_color(&self) -> bool {
        self.active && self.glass.is_visible()
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        mc_core::profile_scope!("modmenu.record");
        self.record_pass(gpu, enc, view, false);
    }

    fn record_scene(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        mc_core::profile_scope!("modmenu.record_scene");
        self.record_glass(gpu, enc, scene);
        self.record_pass(gpu, enc, view, true);
    }

    fn resize(&mut self, _gpu: &Gpu, _width: u32, _height: u32) {
        // The scene targets are rebuilt lazily from the next frame's snapshot size.
        self.resolution = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_uniform_block_is_eight_hundred_bytes_with_rects_after_the_scalars() {
        let frame = GlassFrame {
            rects: vec![GlassRect {
                rect: [10.0, 20.0, 30.0, 40.0],
                radius: 25.0,
                alpha: 0.5,
            }],
            dim: 0.35,
            ..GlassFrame::default()
        };
        let bytes = frame.uniform_bytes(1920, 1080);
        assert_eq!(bytes.len(), 800);
        let f = |offset: usize| f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        assert_eq!((f(0), f(4)), (1920.0, 1080.0));
        assert_eq!(f(16), 0.35);
        assert_eq!(f(24), 1.5, "frost defaults to glass.js' 1.5");
        assert_eq!(u32::from_le_bytes(bytes[28..32].try_into().unwrap()), 1);
        assert_eq!((f(32), f(36), f(40), f(44)), (10.0, 20.0, 30.0, 40.0));
        assert_eq!((f(416), f(420)), (25.0, 0.5));
    }

    #[test]
    fn nothing_to_draw_means_no_scene_request() {
        let mut frame = GlassFrame::default();
        assert!(!frame.is_visible());
        frame.dim = 0.1;
        assert!(frame.is_visible());
    }

    // The WGSL itself is validated on the real GPU by `tests/modmenu_glass.rs` (pipeline
    // creation under a validation error scope, then a pixel readback), so mc-ui needs no
    // `naga` dev-dependency.
}
