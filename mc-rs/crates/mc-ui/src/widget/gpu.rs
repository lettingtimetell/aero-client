//! The `UI 800` render phase.
//!
//! Everything else in `mc-ui` is CPU-side; this is the one file that talks to wgpu. It takes a
//! finished [`DrawList`], uploads it, and draws it with an orthographic projection derived from
//! the [`ScaledResolution`].
//!
//! # Bind groups — a deliberate deviation
//!
//! S10's phase contract reserves `@group(0)` for the shared camera uniform. The UI is pure 2-D
//! screen space and reads nothing from the camera — no view matrix, no camera-relative
//! positions, no fog — so this pipeline's layout starts at its own `@group(0)` (the ortho
//! matrix) and puts the texture at `@group(1)`, and never calls
//! `set_bind_group(0, gpu.camera_bind_group(), …)`. That is legal (each pipeline validates
//! against its own layout) and it keeps the vertex shader down to one matrix multiply. Any
//! future UI effect that genuinely wants camera state should add the camera layout back at
//! index 0 and shift the other two up.
//!
//! # Blending
//!
//! Straight alpha, `src_alpha / one_minus_src_alpha`, matching vanilla's
//! `tryBlendFuncSeparate(770, 771, 1, 0)`. Sampling is **nearest** with repeat, because the
//! whole aesthetic depends on unfiltered texels and because the dirt background relies on wrap.
//! The item foil is the one additional colour blend: `GL_SRC_COLOR, GL_ONE`, matching
//! `RenderItem.renderEffect` (its two animated UV passes are baked into the CPU vertices). The
//! text-field selection is the other fixed-function exception: Java enables colour logic opcode
//! `5387` (`GL_OR_REVERSE`), so its opaque-blue `src | !dst` operation is emitted as two masked
//! pipelines (red/green complement, then blue/alpha replacement). The main UI phase attaches a
//! freshly-cleared shared depth buffer: ordinary widgets use an Always/no-write pipeline, item
//! models use LEQUAL/write, and foil batches use EQUAL/no-write. Item, foil and player-preview
//! batches additionally carry vanilla's `alphaFunc(GL_GREATER, 0.1F)` test (`fs_alpha_test`),
//! so a transparent model texel writes no depth; so does the vanilla crosshair, whose clear
//! texels must not invert the scene. The private panorama renderer remains
//! depthless because its offscreen seed target has no depth attachment.

use std::cell::RefCell;
use std::collections::HashMap;

use mc_assets::{AtlasAnimationUpload, ResourceLocation, ResourceManager, TextureMetadata};
use mc_render::pipeline::PanoramaBlur;
use mc_render::{DepthAttachment, Gpu, RenderPhase};

use crate::font::{FontData, FontTexture};
use crate::widget::draw::{DrawBatch, DrawList, UiBlend, UiCull, UiDepth, UiTexture, UiVertex};
use crate::widget::scaled::ScaledResolution;
use crate::UiError;

const SHADER: &str = r#"
struct Ui { proj: mat4x4<f32> };
@group(0) @binding(0) var<uniform> ui: Ui;
@group(1) @binding(0) var ui_tex: texture_2d<f32>;
@group(1) @binding(1) var ui_samp: sampler;
// <<NW-TEXT:sharp>> What the bound texture is: `flags.x` is 1 for a Minecraft-font sheet
// (sampled by `text_texel`), 0 for everything else. Part of the texture's own bind group, so
// font and non-font batches share every pipeline and no batch costs an extra bind.
struct TexKind { flags: vec4<u32> };
@group(1) @binding(2) var<uniform> tex_kind: TexKind;

struct VsOut {
    // Item foil replays the base model at exact depth using this shared vertex stage.
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs(
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) depth: f32,
) -> VsOut {
    var out: VsOut;
    out.clip = ui.proj * vec4<f32>(pos, depth, 1.0);
    out.uv = uv;
    out.color = color;
    return out;
}

// The bound texture's texel at `uv`: the font sampler for a font sheet, the nearest sampler
// otherwise. `tex_kind` is uniform, so both branches stay in uniform control flow.
fn ui_texel(uv: vec2<f32>) -> vec4<f32> {
    if (tex_kind.flags.x != 0u) {
        return text_texel(uv);
    }
    return textureSample(ui_tex, ui_samp, uv);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // Sample unconditionally (uniform control flow), then let merged Solid vertices, which
    // carry draw::SOLID_SENTINEL_UV, take exactly the 1.0 their 1x1 white texel would give.
    let texel = ui_texel(in.uv);
    return select(texel, vec4<f32>(1.0), in.uv.x < -32768.0) * in.color;
}

// java: RenderItem.renderItemIntoGUI / RendererLivingEntity.doRender -> enableAlpha() with
// alphaFunc(GL_GREATER, 0.1F). The discarded fragment writes neither colour nor depth.
@fragment
fn fs_alpha_test(in: VsOut) -> @location(0) vec4<f32> {
    let texel = ui_texel(in.uv);
    let color = select(texel, vec4<f32>(1.0), in.uv.x < -32768.0) * in.color;
    if (color.a <= 0.1) {
        discard;
    }
    return color;
}

// <<NW-TEXT:sharp>> Minecraft-font glyphs.
//
// Per texture axis: where one texel covers a whole number of pixels (every vanilla GUI scale,
// chat scale 100%, a HUD element at 1x or 2x) the tap is exactly the nearest texel, the one
// the nearest sampler picks, so those pixels are unchanged. Anywhere else (an HD pack's
// 256-pixel font at GUI 3 is 1.5 px per texel, as are unicode-page glyphs; a HUD element at
// 0.7x; chat scale below 100%; the splash) nearest sampling makes some font rows floor(s)
// pixels tall and others ceil(s): squashed and uneven letters. There the pixel takes the
// box-filtered average of the two texels on either side of the nearest texel seam ("sharp
// bilinear"):
// every row is exactly s pixels of ink, with at most one blended pixel per seam. Taps never
// leave the glyph's cell of the 16x16 sheet, so a neighbouring glyph can never bleed in.
struct TextTaps {
    i0: vec2<i32>,
    i1: vec2<i32>,
    t: vec2<f32>,
    exact: bool,
};

fn text_taps(uv: vec2<f32>) -> TextTaps {
    let size = vec2<f32>(textureDimensions(ui_tex));
    let p = uv * size;
    let dx = dpdx(p);
    let dy = dpdy(p);
    // Texels per pixel along each texture axis's own screen axis. The cross terms (italic's
    // shear, a rotation) are left out here so italic text at an integer scale stays nearest.
    let tpp = max(vec2<f32>(abs(dx.x), abs(dy.y)), vec2<f32>(1e-6));
    let ppt = 1.0 / tpp;
    // "A whole number of pixels per texel" (any vanilla-font GUI scale), or of texels per
    // pixel (the unicode font at GUI 1: vanilla point-samples every other texel), with room for
    // the glyph UV inset (font/mod.rs ASCII_UV_INSET): it shortens a quad's UV span by 0.01
    // texel against its position span, so a 2-wide glyph ('i', '!') maps at 1.0102x across and
    // every glyph at 1.00125x down (unicode up to ~2% across). Whatever is classed exact here is
    // sampled exactly as the nearest sampler did, so the tolerance only decides where blending
    // starts.
    let tol = vec2<f32>(0.025, 0.004);
    let up = round(ppt);
    let down = round(tpp);
    let exact = ((up >= vec2<f32>(1.0)) & (abs(ppt - up) <= up * tol))
        | ((down >= vec2<f32>(2.0)) & (abs(tpp - down) <= down * tol));
    // One pixel's footprint in texels: the width of the box filter.
    let foot = max(abs(dx) + abs(dy), vec2<f32>(1e-6));
    let seam = floor(p + 0.5);
    let blend = clamp((p - seam) / foot, vec2<f32>(-0.5), vec2<f32>(0.5)) + 0.5;
    let near = floor(p);
    let cell = size / 16.0;
    let lo = max(floor(p / cell) * cell, vec2<f32>(0.0));
    let hi = min(lo + cell - 1.0, size - 1.0);
    var taps: TextTaps;
    taps.i0 = vec2<i32>(clamp(select(seam - 1.0, near, exact), lo, hi));
    taps.i1 = vec2<i32>(clamp(select(seam, near, exact), lo, hi));
    taps.t = select(blend, vec2<f32>(0.0), exact);
    taps.exact = all(exact);
    return taps;
}

fn text_weights(t: vec2<f32>) -> vec4<f32> {
    return vec4<f32>((1.0 - t.x) * (1.0 - t.y), t.x * (1.0 - t.y), (1.0 - t.x) * t.y, t.x * t.y);
}

// A font sheet's texel at `uv` (see above).
fn text_texel(uv: vec2<f32>) -> vec4<f32> {
    let k = text_taps(uv);
    var texel = textureLoad(ui_tex, k.i0, 0);
    if (!k.exact) {
        // Premultiplied, so a transparent texel's colour cannot darken the edge.
        let b = textureLoad(ui_tex, vec2<i32>(k.i1.x, k.i0.y), 0);
        let c = textureLoad(ui_tex, vec2<i32>(k.i0.x, k.i1.y), 0);
        let d = textureLoad(ui_tex, k.i1, 0);
        let w = text_weights(k.t) * vec4<f32>(texel.a, b.a, c.a, d.a);
        let alpha = w.x + w.y + w.z + w.w;
        let rgb = w.x * texel.rgb + w.y * b.rgb + w.z * c.rgb + w.w * d.rgb;
        texel = vec4<f32>(select(texel.rgb, rgb / alpha, alpha > 0.0), alpha);
    }
    return texel;
}

// java: GuiIngame.renderScoreboard draws its text with GL_BLEND off under FontRenderer's
// alphaFunc(GL_GREATER, 0.1F): a texel lands opaque or not at all.
fn text_gate(texel: vec4<f32>, alpha: f32) -> f32 {
    return select(0.0, 1.0, texel.a * alpha > 0.1);
}

// The blend-off (`UiBlend::Disabled`) pipeline. The alpha test is applied per texel, then the
// 0/1 coverage is blended (this pipeline alpha-blends): every texel the old blend-off pipeline
// drew lands opaque in the same colour, while a font glyph at a fractional scale keeps smooth
// edges. Non-font and Solid vertices: coverage is the old alpha test's pass/fail, so they too
// replace the target's colour exactly as before.
@fragment
fn fs_unblended(in: VsOut) -> @location(0) vec4<f32> {
    let solid = in.uv.x < -32768.0;
    var a: vec4<f32>;
    var rgb: vec3<f32>;
    var cover: f32;
    if (tex_kind.flags.x != 0u) {
        let k = text_taps(in.uv);
        a = select(textureLoad(ui_tex, k.i0, 0), vec4<f32>(1.0), solid);
        rgb = a.rgb;
        cover = text_gate(a, in.color.a);
        if (!k.exact && !solid) {
            let b = textureLoad(ui_tex, vec2<i32>(k.i1.x, k.i0.y), 0);
            let c = textureLoad(ui_tex, vec2<i32>(k.i0.x, k.i1.y), 0);
            let d = textureLoad(ui_tex, k.i1, 0);
            let g = vec4<f32>(cover, text_gate(b, in.color.a), text_gate(c, in.color.a),
                text_gate(d, in.color.a));
            let w = text_weights(k.t) * g;
            cover = w.x + w.y + w.z + w.w;
            let sum = w.x * a.rgb + w.y * b.rgb + w.z * c.rgb + w.w * d.rgb;
            rgb = select(a.rgb, sum / cover, cover > 0.0);
        }
    } else {
        a = select(textureSample(ui_tex, ui_samp, in.uv), vec4<f32>(1.0), solid);
        rgb = a.rgb;
        cover = text_gate(a, in.color.a);
    }
    if (cover <= 0.0) {
        discard;
    }
    return vec4<f32>(rgb * in.color.rgb, cover);
}
"#;

/// The UI WGSL, for tests that keep CPU-side constants in lockstep with it.
#[cfg(test)]
pub(crate) fn ui_shader_source() -> &'static str {
    SHADER
}

/// The fixed-function blend state selected by one CPU draw-batch descriptor.
///
/// Kept as pure functions rather than hidden in [`UiRenderer::new`] so the two-pass
/// `GL_OR_REVERSE` decomposition can be checked without opening a device or running a GPU test.
#[must_use]
/// Whether the UI's vertices and depths are serialised straight into the
/// upload ring's staging memory. Read once;
/// `MC_W18_OB_UI_RING=0` serialises them into the renderer's scratch buffers and copies those, as
/// before. The uploaded bytes are the same either way.
fn ui_ring_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MC_W18_OB_UI_RING").is_none_or(|value| value != "0"))
}

fn blend_state_for(kind: UiBlend) -> wgpu::BlendState {
    match kind {
        UiBlend::Alpha => wgpu::BlendState::ALPHA_BLENDING,
        UiBlend::Invert => wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::OneMinusDst,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
        },
        // The source for this pass is opaque white.  On red/green, Java's blue source has a zero
        // bit in every position, so OR_REVERSE (`src | !dst`) is exactly 1 - dst.
        UiBlend::XorComplement => wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Subtract,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        },
        // The source for this pass is opaque blue.  On blue/alpha, OR_REVERSE is 1 | !dst = 1,
        // so replacement is exact; the write mask below keeps red/green untouched.
        UiBlend::Xor => wgpu::BlendState::REPLACE,
        // `GL_BLEND` off: the colour replaces the target. The target alpha composes as in
        // `ALPHA_BLENDING`, like every other UI write, so a screenshot keeps an opaque frame.
        UiBlend::Disabled => wgpu::BlendState {
            color: wgpu::BlendComponent::REPLACE,
            alpha: wgpu::BlendComponent::OVER,
        },
        // java: RenderItem.java:176 — `blendFunc(768, 1)` = `GL_SRC_COLOR, GL_ONE`.
        UiBlend::Glint => wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Src,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Src,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        },
    }
}

/// Colour channels written by one CPU draw-batch descriptor.
#[must_use]
fn write_mask_for(kind: UiBlend) -> wgpu::ColorWrites {
    match kind {
        UiBlend::Alpha | UiBlend::Invert | UiBlend::Glint | UiBlend::Disabled => {
            wgpu::ColorWrites::ALL
        }
        UiBlend::XorComplement => wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN,
        UiBlend::Xor => wgpu::ColorWrites::BLUE | wgpu::ColorWrites::ALPHA,
    }
}

/// Build one UI pipeline.  The vertex/fragment program is shared, but 1.8.9 switches the
/// fixed-function blend equation for the crosshair and text-field selection.
// Each argument maps directly to one fixed-function Java/UI pipeline choice.
#[allow(clippy::too_many_arguments)]
fn make_pipeline(
    gpu: &Gpu,
    shader: &wgpu::ShaderModule,
    uniform_layout: &wgpu::BindGroupLayout,
    texture_layout: &wgpu::BindGroupLayout,
    label: &'static str,
    kind: UiBlend,
    depth: UiDepth,
    cull: UiCull,
    attach_depth: bool,
) -> wgpu::RenderPipeline {
    make_pipeline_with_alpha_test(
        gpu,
        shader,
        uniform_layout,
        texture_layout,
        label,
        kind,
        depth,
        cull,
        attach_depth,
        false,
    )
}

/// [`make_pipeline`] with an explicit alpha-test choice: `true` selects the `fs_alpha_test`
/// fragment entry point (`alphaFunc(GL_GREATER, 0.1F)`).
#[allow(clippy::too_many_arguments)]
fn make_pipeline_with_alpha_test(
    gpu: &Gpu,
    shader: &wgpu::ShaderModule,
    uniform_layout: &wgpu::BindGroupLayout,
    texture_layout: &wgpu::BindGroupLayout,
    label: &'static str,
    kind: UiBlend,
    depth: UiDepth,
    cull: UiCull,
    attach_depth: bool,
    alpha_test: bool,
) -> wgpu::RenderPipeline {
    make_pipeline_with_entry(
        gpu,
        shader,
        uniform_layout,
        texture_layout,
        label,
        kind,
        depth,
        cull,
        attach_depth,
        if alpha_test { "fs_alpha_test" } else { "fs" },
    )
}

/// [`make_pipeline`] with an explicit fragment entry point (the blend-off
/// text pipeline uses `fs_unblended`).
#[allow(clippy::too_many_arguments)]
fn make_pipeline_with_entry(
    gpu: &Gpu,
    shader: &wgpu::ShaderModule,
    uniform_layout: &wgpu::BindGroupLayout,
    texture_layout: &wgpu::BindGroupLayout,
    label: &'static str,
    kind: UiBlend,
    depth: UiDepth,
    cull: UiCull,
    attach_depth: bool,
    fragment_entry: &'static str,
) -> wgpu::RenderPipeline {
    let pipeline_layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(uniform_layout), Some(texture_layout)],
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
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: UiVertex::SIZE as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 8,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 16,
                                shader_location: 2,
                            },
                        ],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: core::mem::size_of::<f32>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32,
                            offset: 0,
                            shader_location: 3,
                        }],
                    }),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(fragment_entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: Some(blend_state_for(kind)),
                    write_mask: write_mask_for(kind),
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: match cull {
                    UiCull::Back => Some(wgpu::Face::Back),
                    UiCull::None => None,
                },
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: attach_depth.then_some(wgpu::DepthStencilState {
                format: mc_render::DEPTH_FORMAT,
                depth_write_enabled: Some(matches!(depth, UiDepth::Item)),
                depth_compare: Some(match depth {
                    UiDepth::Overlay => wgpu::CompareFunction::Always,
                    UiDepth::Item => wgpu::CompareFunction::LessEqual,
                    UiDepth::Equal => wgpu::CompareFunction::Equal,
                }),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: gpu.sample_count(),
                ..wgpu::MultisampleState::default()
            },
            multiview_mask: None,
            cache: None,
        })
}

struct UiTextureEntry {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
    /// Raw decoded pixels retained for a live anaglyph toggle.  Keeping source bytes avoids
    /// applying Java's integer transform twice when a texture is rebuilt.
    source_rgba: RefCell<Vec<u8>>,
    /// Sampler flags parsed from the source sidecar, if any.
    options: TextureMetadata,
}

/// Opt-in per-frame UI work counters. Timing remains owned by the client/shell trace.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UiFrameStats {
    /// Inclusive UI preparation/submission CPU duration supplied by the client after phase selection.
    pub build_cpu_ns: Option<u64>,
    /// Submitted UI vertices, including an optional panorama seed.
    pub vertex_count: u64,
    /// Submitted contiguous UI batches, including an optional panorama seed.
    pub batch_count: u64,
    /// Builtin tile/entity descriptors referenced by submitted lists.
    pub builtin_tile_item_count: u64,
    /// Vertex payload bytes written to the UI vertex buffer.
    pub vertex_upload_bytes: u64,
    /// Parallel depth payload bytes written to the UI depth buffer.
    pub depth_upload_bytes: u64,
    /// Number of texture queue writes made by the UI renderer.
    pub texture_upload_calls: u64,
    /// RGBA payload bytes supplied to texture queue writes.
    pub texture_upload_bytes: u64,
    /// Actual non-empty `RenderPass::draw` calls.
    pub draw_calls: u64,
}

impl UiFrameStats {
    fn add_assign(&mut self, other: Self) {
        if other.build_cpu_ns.is_some() {
            self.build_cpu_ns = other.build_cpu_ns;
        }
        self.vertex_count = self.vertex_count.saturating_add(other.vertex_count);
        self.batch_count = self.batch_count.saturating_add(other.batch_count);
        self.builtin_tile_item_count = self
            .builtin_tile_item_count
            .saturating_add(other.builtin_tile_item_count);
        self.vertex_upload_bytes = self
            .vertex_upload_bytes
            .saturating_add(other.vertex_upload_bytes);
        self.depth_upload_bytes = self
            .depth_upload_bytes
            .saturating_add(other.depth_upload_bytes);
        self.texture_upload_calls = self
            .texture_upload_calls
            .saturating_add(other.texture_upload_calls);
        self.texture_upload_bytes = self
            .texture_upload_bytes
            .saturating_add(other.texture_upload_bytes);
        self.draw_calls = self.draw_calls.saturating_add(other.draw_calls);
    }
}

/// GPU resources for drawing a [`DrawList`]: one pipeline, one growable vertex buffer, and a
/// texture cache keyed by [`UiTexture`].
pub struct UiRenderer {
    pipeline: wgpu::RenderPipeline,
    overlay_no_cull_pipeline: wgpu::RenderPipeline,
    invert_pipeline: wgpu::RenderPipeline,
    xor_complement_pipeline: wgpu::RenderPipeline,
    xor_pipeline: wgpu::RenderPipeline,
    glint_pipeline: wgpu::RenderPipeline,
    /// The foil blend with no depth test, for the Shiny Pots mod's square glint. Vanilla's foil is always `GL_EQUAL`, so no vanilla batch selects this one.
    glint_overlay_pipeline: wgpu::RenderPipeline,
    item_pipeline: wgpu::RenderPipeline,
    player_pipeline: wgpu::RenderPipeline,
    player_glint_pipeline: wgpu::RenderPipeline,
    /// The four item/player pipelines and the crosshair's invert pipeline above under vanilla's
    /// `alphaFunc(GL_GREATER, 0.1F)` (`DrawBatch::alpha_test`), in the order item, player,
    /// glint, player glint, crosshair; then unblended text ([`UiBlend::Disabled`]), which only
    /// exists alpha-tested.
    alpha_test_pipelines: [wgpu::RenderPipeline; 6],
    /// The two `tex_kind` uniforms every texture bind group points at
    /// (binding 2): 0 for an ordinary texture, 1 for a Minecraft-font sheet (font sampler).
    kind_plain: wgpu::Buffer,
    kind_font: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    texture_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    textures: HashMap<UiTexture, UiTextureEntry>,
    /// Resource-backed keys that failed to load since the last [`Self::clear_textures`]. java:
    /// `TextureManager` keeps the missing-texture object bound under a failed location, so it is
    /// not looked up again every frame (GPURES-11 / HITCH-6).
    missing_textures: std::collections::HashSet<UiTexture>,
    /// Java's `GameSettings.anaglyph`, applied at every UI texture upload boundary.
    anaglyph: bool,
    vertex_buffer: wgpu::Buffer,
    depth_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    uploaded_vertices: usize,
    /// R1-18: retained CPU staging for [`Self::upload_vertices`]; reused every frame.
    vertex_scratch: Vec<u8>,
    depth_scratch: Vec<u8>,
    diagnostics_enabled: bool,
    diagnostic_stats: UiFrameStats,
}

impl std::fmt::Debug for UiRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UiRenderer")
            .field("textures", &self.textures.len())
            .field("vertex_capacity", &self.vertex_capacity)
            .finish_non_exhaustive()
    }
}

impl UiRenderer {
    pub(crate) fn set_diagnostics_enabled(&mut self, enabled: bool) {
        self.diagnostics_enabled = enabled;
        if !enabled {
            self.diagnostic_stats = UiFrameStats::default();
        }
    }

    pub(crate) fn begin_diagnostic_frame(&mut self) {
        if self.diagnostics_enabled {
            self.diagnostic_stats = UiFrameStats::default();
        }
    }

    fn diagnostic_stats(&self) -> UiFrameStats {
        self.diagnostic_stats
    }

    fn record_texture_upload(&mut self, bytes: usize) {
        if self.diagnostics_enabled {
            self.diagnostic_stats.texture_upload_calls =
                self.diagnostic_stats.texture_upload_calls.saturating_add(1);
            self.diagnostic_stats.texture_upload_bytes = self
                .diagnostic_stats
                .texture_upload_bytes
                .saturating_add(bytes as u64);
        }
    }

    fn record_vertex_upload(&mut self, count: usize) {
        if self.diagnostics_enabled {
            self.diagnostic_stats.vertex_upload_bytes = self
                .diagnostic_stats
                .vertex_upload_bytes
                .saturating_add((count * UiVertex::SIZE) as u64);
            self.diagnostic_stats.depth_upload_bytes = self
                .diagnostic_stats
                .depth_upload_bytes
                .saturating_add((count * core::mem::size_of::<f32>()) as u64);
        }
    }

    fn record_draw_calls(&mut self, count: u64) {
        if self.diagnostics_enabled {
            self.diagnostic_stats.draw_calls =
                self.diagnostic_stats.draw_calls.saturating_add(count);
        }
    }

    /// Build the pipeline. A 1×1 white texture is registered for [`UiTexture::Solid`]
    /// immediately, so untextured rects work before any asset is loaded.
    pub fn new(gpu: &Gpu) -> Self {
        Self::new_with_depth(gpu, false)
    }

    /// Build a renderer with Java's global texture-upload anaglyph switch selected.
    #[must_use]
    pub fn new_with_anaglyph(gpu: &Gpu, anaglyph: bool) -> Self {
        Self::new_with_depth_and_anaglyph(gpu, false, anaglyph)
    }

    /// Build a renderer whose main pass can replay vanilla's GUI depth windows. The panorama
    /// seed uses [`Self::new`] because its private colour target has no depth attachment.
    pub(crate) fn new_with_depth(gpu: &Gpu, attach_depth: bool) -> Self {
        Self::new_with_depth_and_anaglyph(gpu, attach_depth, false)
    }

    /// Internal constructor shared by the normal and panorama UI renderers.
    pub(crate) fn new_with_depth_and_anaglyph(
        gpu: &Gpu,
        attach_depth: bool,
        anaglyph: bool,
    ) -> Self {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mc-ui shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mc-ui projection layout"),
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

        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mc-ui texture layout"),
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
                // `tex_kind`: whether this texture is a font sheet.
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

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc-ui projection"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mc-ui projection"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline = make_pipeline(
            gpu,
            &shader,
            &uniform_layout,
            &texture_layout,
            "mc-ui",
            UiBlend::Alpha,
            UiDepth::Overlay,
            UiCull::Back,
            attach_depth,
        );
        let overlay_no_cull_pipeline = make_pipeline(
            gpu,
            &shader,
            &uniform_layout,
            &texture_layout,
            "mc-ui two-sided overlay",
            UiBlend::Alpha,
            UiDepth::Overlay,
            UiCull::None,
            attach_depth,
        );
        // java: GuiIngame.java:181 — `tryBlendFuncSeparate(775, 769, 1, 0)`.
        let invert_pipeline = make_pipeline(
            gpu,
            &shader,
            &uniform_layout,
            &texture_layout,
            "mc-ui crosshair invert",
            UiBlend::Invert,
            UiDepth::Overlay,
            UiCull::Back,
            attach_depth,
        );
        // java: GuiTextField.java:625-641 -- `color(0, 0, 255, 255)` plus opcode 5387
        // (`GL_OR_REVERSE`).  WGPU intentionally exposes no general logic-op state.  The
        // red/green half of the fixed source is zero, so `src | !dst` is exactly `1 - dst`; the
        // first masked pass uses an opaque-white source and this subtractive blend equation.
        let xor_complement_pipeline = make_pipeline(
            gpu,
            &shader,
            &uniform_layout,
            &texture_layout,
            "mc-ui text selection or-reverse complement",
            UiBlend::XorComplement,
            UiDepth::Overlay,
            UiCull::Back,
            attach_depth,
        );
        // The blue/alpha half is `1 | !dst = 1`; replace only those channels with the opaque
        // blue source emitted by `DrawList::rect_xor_blue`.
        let xor_pipeline = make_pipeline(
            gpu,
            &shader,
            &uniform_layout,
            &texture_layout,
            "mc-ui text selection or-reverse force-blue",
            UiBlend::Xor,
            UiDepth::Overlay,
            UiCull::Back,
            attach_depth,
        );
        // java: RenderItem.java:176 — the foil uses source-colour plus-one blending.  Keeping
        // it as a dedicated pipeline preserves the batch's blend contract all the way to WGPU.
        let glint_pipeline = make_pipeline(
            gpu,
            &shader,
            &uniform_layout,
            &texture_layout,
            "mc-ui item glint",
            UiBlend::Glint,
            UiDepth::Equal,
            UiCull::Back,
            attach_depth,
        );
        // 1.7's square potion glint has no item depth outside the
        // bottle to match, so it is drawn with the foil blend and no depth test.
        let glint_overlay_pipeline = make_pipeline(
            gpu,
            &shader,
            &uniform_layout,
            &texture_layout,
            "mc-ui square glint",
            UiBlend::Glint,
            UiDepth::Overlay,
            UiCull::Back,
            attach_depth,
        );

        let item_pipeline = make_pipeline(
            gpu,
            &shader,
            &uniform_layout,
            &texture_layout,
            "mc-ui item depth",
            UiBlend::Alpha,
            UiDepth::Item,
            UiCull::Back,
            attach_depth,
        );
        let player_pipeline = make_pipeline(
            gpu,
            &shader,
            &uniform_layout,
            &texture_layout,
            "mc-ui player preview depth",
            UiBlend::Alpha,
            UiDepth::Item,
            UiCull::None,
            attach_depth,
        );
        let player_glint_pipeline = make_pipeline(
            gpu,
            &shader,
            &uniform_layout,
            &texture_layout,
            "mc-ui player preview glint",
            UiBlend::Glint,
            UiDepth::Equal,
            UiCull::None,
            attach_depth,
        );
        // java: RenderItem.renderItemIntoGUI's `enableAlpha(); alphaFunc(516, 0.1F)` covers the
        // model and its foil; RendererLivingEntity.doRender enables it for the player preview.
        let alpha_test_pipelines = [
            (
                "mc-ui item depth alpha-test",
                UiBlend::Alpha,
                UiDepth::Item,
                UiCull::Back,
            ),
            (
                "mc-ui player depth alpha-test",
                UiBlend::Alpha,
                UiDepth::Item,
                UiCull::None,
            ),
            (
                "mc-ui item glint alpha-test",
                UiBlend::Glint,
                UiDepth::Equal,
                UiCull::Back,
            ),
            (
                "mc-ui player glint alpha-test",
                UiBlend::Glint,
                UiDepth::Equal,
                UiCull::None,
            ),
            // java: GuiIngame.java:178 — `enableAlpha()` around the inverting crosshair blit.
            (
                "mc-ui crosshair invert alpha-test",
                UiBlend::Invert,
                UiDepth::Overlay,
                UiCull::Back,
            ),
            // java: GuiIngame.renderScoreboard's text: blending off, FontRenderer's alpha test.
            // Built as `fs_unblended` with alpha blending of the 0/1 alpha-
            // test coverage (identical colour for every texel the blend-off draw lands, smooth
            // edges for a font at a fractional scale); the old blend-off state under
            // `MC_NW_TEXT_SHARP=0`.
            (
                "mc-ui unblended text alpha-test",
                UiBlend::Disabled,
                UiDepth::Overlay,
                UiCull::Back,
            ),
        ]
        .map(|(label, kind, depth, cull)| {
            let (kind, entry) = if kind == UiBlend::Disabled && super::scaled::sharp_text() {
                (UiBlend::Alpha, "fs_unblended")
            } else {
                (kind, "fs_alpha_test")
            };
            make_pipeline_with_entry(
                gpu,
                &shader,
                &uniform_layout,
                &texture_layout,
                label,
                kind,
                depth,
                cull,
                attach_depth,
                entry,
            )
        });
        // `tex_kind` for ordinary textures and for font sheets.
        let kind_buffer = |label: &'static str, flag: u32| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut bytes = [0u8; 16];
            bytes[..4].copy_from_slice(&flag.to_le_bytes());
            gpu.queue.write_buffer(&buffer, 0, &bytes);
            buffer
        };
        let kind_plain = kind_buffer("mc-ui texture kind: plain", 0);
        let kind_font = kind_buffer("mc-ui texture kind: font", 1);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mc-ui nearest"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..wgpu::SamplerDescriptor::default()
        });

        let vertex_capacity = 4096;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc-ui vertices"),
            size: (vertex_capacity * UiVertex::SIZE) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let depth_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc-ui depths"),
            size: (vertex_capacity * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut me = Self {
            pipeline,
            overlay_no_cull_pipeline,
            invert_pipeline,
            xor_complement_pipeline,
            xor_pipeline,
            glint_pipeline,
            glint_overlay_pipeline,
            item_pipeline,
            player_pipeline,
            player_glint_pipeline,
            alpha_test_pipelines,
            kind_plain,
            kind_font,
            uniform_buffer,
            uniform_bind_group,
            texture_layout,
            sampler,
            textures: HashMap::new(),
            missing_textures: std::collections::HashSet::new(),
            anaglyph,
            vertex_buffer,
            depth_buffer,
            vertex_capacity,
            uploaded_vertices: 0,
            vertex_scratch: Vec::new(),
            depth_scratch: Vec::new(),
            diagnostics_enabled: false,
            diagnostic_stats: UiFrameStats::default(),
        };
        me.upload(gpu, UiTexture::Solid, &[255, 255, 255, 255], 1, 1);
        me
    }

    /// Register RGBA8 pixels under `key`, replacing any previous entry.
    ///
    /// # Panics
    /// If `rgba` is shorter than `width * height * 4`.
    pub fn upload(&mut self, gpu: &Gpu, key: UiTexture, rgba: &[u8], width: u32, height: u32) {
        self.upload_with_options(gpu, key, rgba, width, height, &TextureMetadata::default());
    }

    /// Register a texture with the Java `SimpleTexture` sampler flags carried by its sidecar.
    pub fn upload_with_options(
        &mut self,
        gpu: &Gpu,
        key: UiTexture,
        rgba: &[u8],
        width: u32,
        height: u32,
        options: &TextureMetadata,
    ) {
        assert!(
            rgba.len() >= (width as usize) * (height as usize) * 4,
            "pixel buffer too small for {width}x{height}"
        );
        // A texture with a side over the device's limit is never
        // created (wgpu reports a validation error, which ends the process): it stays blank, as
        // vanilla's `glTexImage2D` leaves it. A pack's font sheet (`ascii.png`) reaches here
        // without the decode-time check: it is decoded with the assets, maybe before any device.
        let max_side = gpu.device.limits().max_texture_dimension_2d;
        if width > max_side || height > max_side {
            return self.upload_with_options(gpu, key, &[0; 4], 1, 1, options);
        }
        let byte_len = (width as usize) * (height as usize) * 4;
        let source_rgba = rgba[..byte_len].to_vec();
        // Java's `TextureUtil.copyToBufferPos` converts the temporary upload array, not the
        // resource-manager bytes.  This keeps reloads and other GPU copies raw and guarantees
        // that an enabled setting cannot double-convert a cached entry.
        let upload_rgba = mc_assets::sprite::anaglyph_rgba(&source_rgba, self.anaglyph);
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mc-ui texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Deliberately **not** sRGB: 1.8.9's GL pipeline does no colour-space conversion,
            // so a texel of 0x80 has to reach the framebuffer as 0x80. An sRGB view here would
            // darken every piece of GUI art by the linearisation curve.
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &upload_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.record_texture_upload(upload_rgba.len());
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let option_sampler = if *options == TextureMetadata::default() {
            None
        } else {
            Some(gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("mc-ui metadata sampler"),
                address_mode_u: if options.clamp {
                    wgpu::AddressMode::ClampToEdge
                } else {
                    wgpu::AddressMode::Repeat
                },
                address_mode_v: if options.clamp {
                    wgpu::AddressMode::ClampToEdge
                } else {
                    wgpu::AddressMode::Repeat
                },
                address_mode_w: if options.clamp {
                    wgpu::AddressMode::ClampToEdge
                } else {
                    wgpu::AddressMode::Repeat
                },
                mag_filter: if options.blur {
                    wgpu::FilterMode::Linear
                } else {
                    wgpu::FilterMode::Nearest
                },
                min_filter: if options.blur {
                    wgpu::FilterMode::Linear
                } else {
                    wgpu::FilterMode::Nearest
                },
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..wgpu::SamplerDescriptor::default()
            }))
        };
        let sampler = option_sampler.as_ref().unwrap_or(&self.sampler);
        // Font sheets take the font sampler (`text_texel`).
        let kind = if matches!(key, UiTexture::Font(_)) && super::scaled::sharp_text() {
            &self.kind_font
        } else {
            &self.kind_plain
        };
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mc-ui texture"),
            layout: &self.texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: kind.as_entire_binding(),
                },
            ],
        });
        self.textures.insert(
            key,
            UiTextureEntry {
                texture,
                bind_group,
                width,
                height,
                source_rgba: RefCell::new(source_rgba),
                options: options.clone(),
            },
        );
    }

    /// Whether this renderer applies Java's global upload-time anaglyph transform.
    #[must_use]
    pub const fn anaglyph_enabled(&self) -> bool {
        self.anaglyph
    }

    /// Toggle Java's global transform and rebuild every resident UI texture from raw bytes.
    ///
    /// Replaying the source cache is what makes a live options change exact and avoids applying
    /// the integer transform twice.  Font sheets are included exactly once at this boundary; the
    /// existing font renderer's anaglyph colour table remains a separate vertex-colour concern.
    pub fn set_anaglyph_with_gpu(&mut self, gpu: &Gpu, enabled: bool) {
        if self.anaglyph == enabled {
            return;
        }
        self.anaglyph = enabled;
        let uploads: Vec<_> = self
            .textures
            .iter()
            .map(|(key, entry)| {
                (
                    key.clone(),
                    entry.source_rgba.borrow().clone(),
                    entry.width,
                    entry.height,
                    entry.options.clone(),
                )
            })
            .collect();
        for (key, rgba, width, height, options) in uploads {
            self.upload_with_options(gpu, key, &rgba, width, height, &options);
        }
    }

    /// Set the transform for subsequent uploads without rebuilding resident entries.
    ///
    /// Use [`Self::set_anaglyph_with_gpu`] when a live setting change must affect textures that
    /// are already cached.
    pub fn set_anaglyph(&mut self, enabled: bool) {
        self.anaglyph = enabled;
    }

    /// Upload one keyed 128 by 128 filled-map image without reallocating its GPU texture when
    /// the map was already visible in an earlier frame.
    ///
    /// Map images are dynamic textures in vanilla: an S34 update changes the pixels behind the
    /// same `map_<id>` location. Keeping that identity and updating the existing level-zero
    /// texture also prevents a GUI map from briefly sampling the white fallback between frames.
    /// Returns `false` when `rgba` is not exactly one map image.
    pub fn upload_dynamic_map(&mut self, gpu: &Gpu, map_id: i32, rgba: &[u8]) -> bool {
        use crate::map::{dynamic_texture_key, MAP_TEXTURE_BYTES, MAP_TEXTURE_SIZE};

        if rgba.len() != MAP_TEXTURE_BYTES {
            return false;
        }
        let key = dynamic_texture_key(map_id);
        if let Some((width, height)) = self
            .textures
            .get(&key)
            .map(|entry| (entry.width, entry.height))
        {
            if width == MAP_TEXTURE_SIZE as u32 && height == MAP_TEXTURE_SIZE as u32 {
                // The resident texture always holds `source_rgba` through
                // the current transform (an anaglyph toggle replays every entry from it), so equal
                // bytes need neither the two copies nor the upload. An open container of maps
                // used to re-upload every map every frame; vanilla uploads one only on S34.
                let unchanged = self
                    .textures
                    .get(&key)
                    .is_some_and(|entry| entry.source_rgba.borrow().as_slice() == rgba);
                if unchanged {
                    return true;
                }
                let anaglyph = self.anaglyph;
                let upload_rgba = mc_assets::sprite::anaglyph_rgba(rgba, anaglyph);
                let entry = self
                    .textures
                    .get_mut(&key)
                    .expect("map texture entry was just observed");
                entry.source_rgba.borrow_mut().copy_from_slice(rgba);
                let texture = &entry.texture;
                gpu.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &upload_rgba,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(MAP_TEXTURE_SIZE as u32 * 4),
                        rows_per_image: Some(MAP_TEXTURE_SIZE as u32),
                    },
                    wgpu::Extent3d {
                        width: MAP_TEXTURE_SIZE as u32,
                        height: MAP_TEXTURE_SIZE as u32,
                        depth_or_array_layers: 1,
                    },
                );
                self.record_texture_upload(upload_rgba.len());
                return true;
            }
            // A key supplied by a different dynamic producer must not silently retain its
            // dimensions. Replace it with the map's fixed-size texture below.
            self.invalidate(&key);
        }
        self.upload(
            gpu,
            key,
            rgba,
            MAP_TEXTURE_SIZE as u32,
            MAP_TEXTURE_SIZE as u32,
        );
        true
    }

    /// Drop every UI-side dynamic texture created for a filled map.
    ///
    /// The CPU [`mc_ui::map::MapTextureCache`] is cleared on join, disconnect and changed
    /// dimension. This companion operation prevents a reused metadata id from retaining the
    /// previous world's image until its first new S34 packet arrives.
    pub fn clear_dynamic_map_textures(&mut self) -> usize {
        let keys: Vec<_> = self
            .textures
            .keys()
            .filter(|key| {
                matches!(
                    key,
                    UiTexture::Resource(location)
                        if location.namespace == "minecraft"
                            && location.path.starts_with("dynamic/map_")
                )
            })
            .cloned()
            .collect();
        let count = keys.len();
        for key in keys {
            self.textures.remove(&key);
        }
        count
    }

    /// Forget `key`'s cached bind group, so the next [`UiRenderer::ensure_texture`] re-reads it
    /// from the pack stack. Returns whether anything was cached.
    ///
    /// java: `TextureManager.deleteTexture` — which is what `refreshResources` reaches through
    /// `SimpleTexture.loadTexture` re-running on every reload listener.
    ///
    /// Without this a resource pack that replaces a GUI texture has no effect until restart:
    /// [`UiRenderer::ensure_texture`] returns early for any key already resident, and every
    /// texture a screen has drawn once is resident.
    pub fn invalidate(&mut self, key: &UiTexture) -> bool {
        // A remembered failure must not outlive an explicit re-read request either.
        self.missing_textures.remove(key);
        self.textures.remove(key).is_some()
    }

    /// Apply one S09 animated-atlas dirty rectangle to the UI's single-level atlas copy.
    ///
    /// GUI item icons sample the same stitched `textures/atlas/blocks.png` bytes as terrain,
    /// but the UI texture cache intentionally has no mip chain. Level-zero records are therefore
    /// the only ones this owner accepts.
    pub fn upload_atlas_animation(
        &mut self,
        gpu: &Gpu,
        key: &UiTexture,
        upload: &AtlasAnimationUpload,
    ) -> bool {
        let anaglyph = self.anaglyph;
        let Some(entry) = self.textures.get(key) else {
            return false;
        };
        if upload.mip_level != 0
            || upload.width == 0
            || upload.height == 0
            || upload.rgba.len()
                != (u64::from(upload.width) * u64::from(upload.height) * 4) as usize
            || upload
                .x
                .checked_add(upload.width)
                .is_none_or(|end| end > entry.width)
            || upload
                .y
                .checked_add(upload.height)
                .is_none_or(|end| end > entry.height)
        {
            return false;
        }
        // Keep the source cache raw and current so a live setting toggle replays the latest atlas
        // frame rather than reverting this UI copy to its initial sheet frame.
        for row in 0..upload.height {
            let src_start = (row * upload.width * 4) as usize;
            let dst_start = (((upload.y + row) * entry.width + upload.x) * 4) as usize;
            let count = (upload.width * 4) as usize;
            entry.source_rgba.borrow_mut()[dst_start..dst_start + count]
                .copy_from_slice(&upload.rgba[src_start..src_start + count]);
        }
        let upload_rgba = mc_assets::sprite::anaglyph_rgba(&upload.rgba, anaglyph);
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &entry.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: upload.x,
                    y: upload.y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &upload_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(upload.width * 4),
                rows_per_image: Some(upload.height),
            },
            wgpu::Extent3d {
                width: upload.width,
                height: upload.height,
                depth_or_array_layers: 1,
            },
        );
        self.record_texture_upload(upload_rgba.len());
        true
    }

    /// Drop every cached texture, then re-register the 1×1 white [`UiTexture::Solid`].
    ///
    /// The whole-stack version of [`UiRenderer::invalidate`], for a resource-pack swap where the
    /// set of changed textures is not known. `Solid` is re-registered rather than dropped because
    /// [`UiRenderer::draw`] uses it as the fallback for a batch whose texture is missing and bails
    /// out of the frame entirely without it — and no resource pack can change a 1×1 white pixel.
    ///
    /// Anything the caller uploaded by hand (`upload_ascii_font`, a screen's own atlas) must be
    /// re-uploaded afterwards; textures that came from the pack stack come back on their own
    /// through [`UiRenderer::ensure_list_textures`] on the next frame.
    pub fn clear_textures(&mut self, gpu: &Gpu) {
        self.textures.clear();
        self.missing_textures.clear();
        self.upload(gpu, UiTexture::Solid, &[255, 255, 255, 255], 1, 1);
    }

    /// Whether `key` has pixels uploaded.
    #[must_use]
    pub fn has_texture(&self, key: &UiTexture) -> bool {
        self.textures.contains_key(key)
    }

    /// How many textures are resident.
    #[must_use]
    pub fn texture_count(&self) -> usize {
        self.textures.len()
    }

    /// Upload the default font sheet from already-decoded [`FontData`].
    ///
    /// Saves re-reading and re-decoding `ascii.png`, which the font renderer already holds.
    pub fn upload_ascii_font(&mut self, gpu: &Gpu, data: &FontData) {
        let (w, h) = data.ascii_size();
        self.upload(
            gpu,
            UiTexture::Font(FontTexture::Ascii),
            data.ascii_rgba(),
            w,
            h,
        );
    }

    /// Load `key`'s pixels out of the resource packs if they are not resident already.
    ///
    /// # Errors
    /// [`UiError::FontAsset`] when the resource is missing, [`UiError::FontTexture`] when it
    /// will not decode as a PNG.
    pub fn ensure_texture(
        &mut self,
        gpu: &Gpu,
        assets: &ResourceManager,
        key: &UiTexture,
    ) -> Result<(), UiError> {
        if self.textures.contains_key(key) {
            return Ok(());
        }
        let location = match key {
            UiTexture::Solid => return Ok(()),
            UiTexture::Font(f) => f.location(),
            UiTexture::Resource(l) => l.clone(),
        };
        if self.missing_textures.contains(key) {
            // Already failed for this pack stack: answer without another lookup and decode.
            return Err(UiError::FontAsset(
                location.to_string(),
                mc_assets::AssetError::NotFound(location),
            ));
        }
        let loaded = self.load_texture_from_packs(gpu, assets, key, &location);
        if loaded.is_err() {
            self.missing_textures.insert(key.clone());
        }
        loaded
    }

    /// The uncached half of [`Self::ensure_texture`]: read, decode and upload `key` from `location`.
    fn load_texture_from_packs(
        &mut self,
        gpu: &Gpu,
        assets: &ResourceManager,
        key: &UiTexture,
        location: &ResourceLocation,
    ) -> Result<(), UiError> {
        let location = location.clone();
        let resource = assets
            .resource(&location)
            .map_err(|e| UiError::FontAsset(location.to_string(), e))?;
        // Refused (missing) when the device cannot hold it.
        let img = mc_assets::decode_texture_png(&resource.data)
            .map_err(|e| UiError::FontTexture {
                location: location.to_string(),
                reason: e.to_string(),
            })?
            .to_rgba8();
        let (w, h) = (img.width(), img.height());
        let options = mc_assets::parse_texture_metadata(&location, resource.metadata.as_deref())
            .map_err(|e| UiError::FontAsset(location.to_string(), e))?;
        self.upload_with_options(gpu, key.clone(), &img.into_raw(), w, h, &options);
        Ok(())
    }

    /// Ensure the entity sheets referenced by builtin/item geometry are resident.
    ///
    /// `TileEntityItemStackRenderer` does not sample the stitched item atlas: chests and skulls
    /// bind their fixed entity sheets, while banners bind a CPU-composited
    /// `LayeredColorMaskTexture`.  The draw list carries the small typed descriptor so this
    /// upload remains on the UI's existing GPU boundary and does not need a second render phase.
    fn ensure_builtin_tile_item_texture(
        &mut self,
        gpu: &Gpu,
        assets: &ResourceManager,
        item: &mc_render::entity::tilepass::BuiltinTileItem,
    ) {
        let Some(path) = item.texture_key(mc_render::entity::tilepass::christmas_today()) else {
            return;
        };
        let key = UiTexture::gui(&path);
        if self.has_texture(&key) {
            return;
        }

        match item {
            mc_render::entity::tilepass::BuiltinTileItem::Banner { state } => {
                let Some((returned_key, rgba, width, height)) =
                    mc_render::entity::tilepass::banner_texture_pixels(assets, state)
                else {
                    return;
                };
                // Keep the upload key tied to the descriptor's Java cache key.  The equality is
                // also a negative control for accidentally binding the fixed banner base sheet.
                debug_assert_eq!(returned_key, path);
                self.upload(gpu, key, &rgba, width, height);
            }
            mc_render::entity::tilepass::BuiltinTileItem::PlayerSkull { owner, .. }
                if path == owner.cache_key =>
            {
                // Player skulls are the one builtin item whose sheet is already decoded by the
                // async appearance cache rather than a resource-pack PNG.  If the profile has
                // not landed yet, `texture_key` selected Alex/Steve above, so this branch is only
                // reached for a resident normalized skin.
                if let Some(texture) = mc_render::entity::skin::shared_skin(&owner.cache_key) {
                    self.upload(
                        gpu,
                        key,
                        &texture.rgba,
                        mc_render::entity::SkinTexture::WIDTH,
                        mc_render::entity::SkinTexture::HEIGHT,
                    );
                }
            }
            _ => {
                // Fixed chest/skull sheets still pass through the normal pack-stack lookup, so
                // custom resource packs can replace them exactly as they can world TESR sheets.
                let _ = self.ensure_texture(gpu, assets, &key);
            }
        }
    }

    /// Load every texture a [`DrawList`] refers to, ignoring the ones that fail.
    ///
    /// Returns the locations that could not be loaded. A resource pack is allowed to be missing
    /// a texture; the affected batches fall back to the white 1×1 and simply draw untextured
    /// rather than aborting the frame.
    pub fn ensure_list_textures(
        &mut self,
        gpu: &Gpu,
        assets: &ResourceManager,
        list: &DrawList,
    ) -> Vec<ResourceLocation> {
        mc_core::profile_scope!("ui.ensure_textures");
        let mut missing = Vec::new();
        for item in list.builtin_tile_items() {
            self.ensure_builtin_tile_item_texture(gpu, assets, item);
        }
        for batch in list.batches() {
            if self.ensure_texture(gpu, assets, &batch.texture).is_err() {
                match &batch.texture {
                    UiTexture::Font(f) => missing.push(f.location()),
                    UiTexture::Resource(l) => missing.push(l.clone()),
                    UiTexture::Solid => {}
                }
            }
        }
        missing
    }

    /// Push the projection matrix for `res`.
    pub fn set_resolution(&mut self, gpu: &Gpu, res: &ScaledResolution) {
        let m = res.ortho();
        let mut bytes = Vec::with_capacity(64);
        for col in m {
            for f in col {
                bytes.extend_from_slice(&f.to_le_bytes());
            }
        }
        gpu.queue.write_buffer(&self.uniform_buffer, 0, &bytes);
    }

    /// Upload a frame's vertices, growing the buffer if needed.
    pub fn upload_vertices(&mut self, gpu: &Gpu, list: &DrawList) {
        mc_core::profile_scope!("ui.upload_vertices");
        let count = list.vertices().len();
        debug_assert_eq!(list.depths().len(), count);
        self.uploaded_vertices = count;
        if count == 0 {
            return;
        }
        self.record_vertex_upload(count);
        if count > self.vertex_capacity {
            self.vertex_capacity = count.next_power_of_two();
            self.vertex_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc-ui vertices"),
                size: (self.vertex_capacity * UiVertex::SIZE) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.depth_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc-ui depths"),
                size: (self.vertex_capacity * core::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        // The same bytes, serialised straight into the upload ring's staging
        // memory instead of into `vertex_scratch`/`depth_scratch` and copied from there.
        if ui_ring_enabled() && !mc_render::w18_ob_ab::off(mc_render::w18_ob_ab::UI_RING) {
            gpu.write_buffer_staged_with(
                &self.vertex_buffer,
                0,
                (count * UiVertex::SIZE) as u64,
                |out| list.fill_vertex_bytes(out),
            );
            gpu.write_buffer_staged_with(
                &self.depth_buffer,
                0,
                (count * core::mem::size_of::<f32>()) as u64,
                |out| list.fill_depth_bytes(out),
            );
            return;
        }
        list.write_vertex_bytes(&mut self.vertex_scratch);
        list.write_depth_bytes(&mut self.depth_scratch);
        // Per-frame upload ring: the frame encoder is these buffers' only reader.
        gpu.write_buffer_staged(&self.vertex_buffer, 0, &self.vertex_scratch);
        gpu.write_buffer_staged(&self.depth_buffer, 0, &self.depth_scratch);
    }

    /// The alpha-tested twin of an item/player or crosshair pipeline (`DrawBatch::alpha_test`),
    /// if the state names one. Only item, player-preview and foil windows and the vanilla
    /// crosshair ([`DrawList::blit_invert`]) ever open an alpha-test window.
    fn alpha_test_pipeline(
        &self,
        blend: UiBlend,
        depth: UiDepth,
        cull: UiCull,
    ) -> Option<&wgpu::RenderPipeline> {
        let index = match (blend, depth, cull) {
            (UiBlend::Alpha, UiDepth::Item, UiCull::Back) => 0,
            (UiBlend::Alpha, UiDepth::Item, UiCull::None) => 1,
            (UiBlend::Glint, UiDepth::Equal, UiCull::Back) => 2,
            (UiBlend::Glint, UiDepth::Equal, UiCull::None) => 3,
            // The untested invert pipeline is chosen for any depth/cull; so is its twin.
            (UiBlend::Invert, _, _) => 4,
            (UiBlend::Disabled, _, _) => 5,
            _ => return None,
        };
        Some(&self.alpha_test_pipelines[index])
    }

    /// Record the draw calls for an already-uploaded list.
    ///
    /// A batch whose texture is not resident falls back to the white 1×1 rather than being
    /// skipped, so a missing texture shows as flat colour instead of a hole.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, list: &DrawList) -> u64 {
        if self.uploaded_vertices == 0 {
            return 0;
        }
        pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, self.depth_buffer.slice(..));
        let Some(fallback) = self.textures.get(&UiTexture::Solid) else {
            return 0;
        };
        let mut draw_calls = 0;
        // #24: consecutive batches usually differ in only one of pipeline or texture (the
        // Solid/Font alternation keeps the pipeline; an item window keeps the atlas), so skip
        // re-binding whichever is unchanged. Bind groups persist across `set_pipeline` — group
        // 0 above is already bound once for every pipeline — so the recorded state at each
        // draw is identical.
        let mut bound_pipeline: Option<&wgpu::RenderPipeline> = None;
        let mut bound_texture: Option<&wgpu::BindGroup> = None;
        for batch in list.batches() {
            let pipeline = self.batch_pipeline(batch);
            if !bound_pipeline.is_some_and(|bound| std::ptr::eq(bound, pipeline)) {
                pass.set_pipeline(pipeline);
                bound_pipeline = Some(pipeline);
            }
            let bind = self.textures.get(&batch.texture).unwrap_or(fallback);
            if !bound_texture.is_some_and(|bound| std::ptr::eq(bound, &bind.bind_group)) {
                pass.set_bind_group(1, &bind.bind_group, &[]);
                bound_texture = Some(&bind.bind_group);
            }
            let end = batch.range.end.min(self.uploaded_vertices) as u32;
            let start = batch.range.start as u32;
            if end > start {
                if self.diagnostics_enabled {
                    draw_calls += 1;
                }
                pass.draw(start..end, 0..1);
            }
        }
        draw_calls
    }

    /// [`Self::draw`] restricted to the vertices in `vertices` (indices into
    /// [`DrawList::vertices`]), for a list drawn in two passes (the menu blur records between
    /// the HUD and the open screen, see [`UiPhase::set_split`]).
    ///
    /// A batch straddling either end is drawn in part with its own pipeline and texture, so the
    /// two halves of a split list produce exactly the pixels of one [`Self::draw`] of it. Batches
    /// wholly outside the range are skipped before any state is bound.
    pub fn draw_vertex_range(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        list: &DrawList,
        vertices: std::ops::Range<usize>,
    ) -> u64 {
        if self.uploaded_vertices == 0 {
            return 0;
        }
        pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, self.depth_buffer.slice(..));
        let Some(fallback) = self.textures.get(&UiTexture::Solid) else {
            return 0;
        };
        let limit = vertices.end.min(self.uploaded_vertices);
        let mut draw_calls = 0;
        let mut bound_pipeline: Option<&wgpu::RenderPipeline> = None;
        let mut bound_texture: Option<&wgpu::BindGroup> = None;
        for batch in list.batches() {
            let start = batch.range.start.max(vertices.start);
            let end = batch.range.end.min(limit);
            if end <= start {
                continue;
            }
            let pipeline = self.batch_pipeline(batch);
            if !bound_pipeline.is_some_and(|bound| std::ptr::eq(bound, pipeline)) {
                pass.set_pipeline(pipeline);
                bound_pipeline = Some(pipeline);
            }
            let bind = self.textures.get(&batch.texture).unwrap_or(fallback);
            if !bound_texture.is_some_and(|bound| std::ptr::eq(bound, &bind.bind_group)) {
                pass.set_bind_group(1, &bind.bind_group, &[]);
                bound_texture = Some(&bind.bind_group);
            }
            if self.diagnostics_enabled {
                draw_calls += 1;
            }
            pass.draw(start as u32..end as u32, 0..1);
        }
        draw_calls
    }

    /// How many `set_pipeline` calls [`Self::draw`] makes for `list` (it skips a bind when the
    /// next batch uses the pipeline already bound). A diagnostic count for the HUD/text cost work
    /// (a test proves the font sampler adds no bind).
    #[must_use]
    pub fn pipeline_binds(&self, list: &DrawList) -> u64 {
        let mut bound: Option<&wgpu::RenderPipeline> = None;
        let mut binds = 0;
        for batch in list.batches() {
            let pipeline = self.batch_pipeline(batch);
            if !bound.is_some_and(|b| std::ptr::eq(b, pipeline)) {
                binds += 1;
                bound = Some(pipeline);
            }
        }
        binds
    }

    /// The pipeline a batch draws with: its blend/depth/cull state, or that state's alpha-tested
    /// twin when the batch carries vanilla's item alpha test.
    fn batch_pipeline(&self, batch: &DrawBatch) -> &wgpu::RenderPipeline {
        let pipeline = match (batch.blend, batch.depth, batch.cull) {
            (UiBlend::Alpha, UiDepth::Overlay, UiCull::None) => &self.overlay_no_cull_pipeline,
            (UiBlend::Alpha, UiDepth::Item, UiCull::Back) => &self.item_pipeline,
            (UiBlend::Alpha, UiDepth::Item, UiCull::None) => &self.player_pipeline,
            (UiBlend::Glint, UiDepth::Equal, UiCull::None) => &self.player_glint_pipeline,
            (UiBlend::Glint, UiDepth::Overlay, _) => &self.glint_overlay_pipeline,
            (UiBlend::Glint, _, _) => &self.glint_pipeline,
            (UiBlend::Alpha, _, _) => &self.pipeline,
            (UiBlend::Invert, _, _) => &self.invert_pipeline,
            // Vanilla disables blending around text only, and FontRenderer.drawString always
            // enables the alpha test, so there is no untested unblended pipeline.
            (UiBlend::Disabled, _, _) => &self.alpha_test_pipelines[5],
            (UiBlend::XorComplement, _, _) => &self.xor_complement_pipeline,
            (UiBlend::Xor, _, _) => &self.xor_pipeline,
        };
        if batch.alpha_test {
            self.alpha_test_pipeline(batch.blend, batch.depth, batch.cull)
                .unwrap_or(pipeline)
        } else {
            pipeline
        }
    }
}

/// One main-menu frame's private panorama seed and final crop parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct PanoramaFrameSubmission {
    /// Vanilla's square-projected, 64-sample first stage in 256×256 GUI coordinates.
    pub seed: DrawList,
    /// Final `renderSkybox` crop UVs in top-left, bottom-left, top-right, bottom-right order.
    pub composite_uv: [[f32; 2]; 4],
}

impl PanoramaFrameSubmission {
    /// Bundle the menu's separately generated panorama seed and final crop contract.
    #[must_use]
    pub fn new(seed: DrawList, composite_uv: [[f32; 2]; 4]) -> Self {
        Self { seed, composite_uv }
    }
}

#[derive(Debug)]
struct PanoramaSupport {
    blur: PanoramaBlur,
    renderer: UiRenderer,
    submission: Option<PanoramaFrameSubmission>,
}

#[must_use]
fn panorama_seed_resolution() -> ScaledResolution {
    ScaledResolution::new(
        PanoramaBlur::VANILLA_SIZE,
        PanoramaBlur::VANILLA_SIZE,
        1,
        false,
    )
}

/// The `UI 800` phase: draws whatever [`DrawList`] was last handed to
/// [`UiPhase::submit`], optionally after an offscreen panorama submitted through
/// [`UiPhase::submit_with_panorama`].
///
/// Register with `phases.insert(PhaseOrder::UI, Box::new(phase))`.
pub struct UiPhase {
    /// The GPU resources. Public so screens can upload their own textures.
    pub renderer: UiRenderer,
    list: DrawList,
    /// The list submitted the frame before `list`, already cleared. [`Self::take_spare_list`]
    /// hands it back to the builder so the per-frame vertex/depth/batch `Vec`s keep their
    /// capacity (and the builtin `TileModels` cache) instead of regrowing from empty (C2-5).
    spare: DrawList,
    resolution: ScaledResolution,
    resolution_dirty: bool,
    gui_scale: i32,
    unicode: bool,
    panorama: Option<PanoramaSupport>,
    diagnostics_enabled: bool,
    diagnostic_list_stats: UiFrameStats,
    /// Where the open screen's half of `list` starts (a vertex index), while
    /// the menu blur records between the HUD and the screen ([`Self::set_split`]). `None`, the
    /// normal case, draws the whole list in [`RenderPhase::record`] exactly as before.
    split: Option<usize>,
}

impl std::fmt::Debug for UiPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UiPhase")
            .field("resolution", &self.resolution)
            .field("vertices", &self.list.vertices().len())
            .field(
                "panorama",
                &self
                    .panorama
                    .as_ref()
                    .and_then(|support| support.submission.as_ref())
                    .is_some(),
            )
            .finish_non_exhaustive()
    }
}

impl UiPhase {
    /// Build the phase.
    ///
    /// `gui_scale` and `unicode` are kept so that [`RenderPhase::resize`] can recompute the
    /// scaled resolution from the *setting*, not from the previous frame's derived scale factor
    /// — recomputing from the factor would silently pin an "auto" setting to whatever it
    /// happened to resolve to at startup.
    pub fn new(gpu: &Gpu, width: u32, height: u32, gui_scale: i32, unicode: bool) -> Self {
        Self::new_with_anaglyph(gpu, width, height, gui_scale, unicode, false)
    }

    /// Build the phase with Java's global texture-upload anaglyph switch selected.
    #[must_use]
    pub fn new_with_anaglyph(
        gpu: &Gpu,
        width: u32,
        height: u32,
        gui_scale: i32,
        unicode: bool,
        anaglyph: bool,
    ) -> Self {
        Self {
            renderer: UiRenderer::new_with_depth_and_anaglyph(gpu, true, anaglyph),
            list: DrawList::new(),
            spare: DrawList::new(),
            resolution: ScaledResolution::new(width, height, gui_scale, unicode),
            resolution_dirty: true,
            gui_scale,
            unicode,
            panorama: None,
            diagnostics_enabled: false,
            diagnostic_list_stats: UiFrameStats::default(),
            split: None,
        }
    }

    /// Enable or disable opt-in UI work counters; disabled is the constructor default.
    pub fn set_diagnostics_enabled(&mut self, enabled: bool) {
        self.diagnostics_enabled = enabled;
        self.renderer.set_diagnostics_enabled(enabled);
        if let Some(panorama) = &mut self.panorama {
            panorama.renderer.set_diagnostics_enabled(enabled);
        }
        if !enabled {
            self.diagnostic_list_stats = UiFrameStats::default();
        }
    }

    /// Reset the per-frame UI counters without reading a clock.
    pub fn begin_diagnostic_frame(&mut self) {
        if self.diagnostics_enabled {
            self.diagnostic_list_stats = UiFrameStats::default();
            self.renderer.begin_diagnostic_frame();
            if let Some(panorama) = &mut self.panorama {
                panorama.renderer.begin_diagnostic_frame();
            }
        }
    }

    /// Record the client-owned inclusive UI preparation/submission duration after phase selection.
    pub fn record_build_cpu(&mut self, duration: std::time::Duration) {
        if self.diagnostics_enabled {
            self.diagnostic_list_stats.build_cpu_ns =
                Some(duration.as_nanos().min(u64::MAX as u128) as u64);
        }
    }

    /// Return the foreground plus optional panorama UI work snapshot.
    #[must_use]
    pub fn diagnostic_stats(&self) -> Option<UiFrameStats> {
        if !self.diagnostics_enabled {
            return None;
        }
        let mut stats = self.diagnostic_list_stats;
        stats.add_assign(self.renderer.diagnostic_stats());
        if let Some(panorama) = &self.panorama {
            stats.add_assign(panorama.renderer.diagnostic_stats());
        }
        Some(stats)
    }

    fn record_list_stats(&mut self, list: &DrawList) {
        if self.diagnostics_enabled {
            self.diagnostic_list_stats.vertex_count = self
                .diagnostic_list_stats
                .vertex_count
                .saturating_add(list.vertices().len() as u64);
            self.diagnostic_list_stats.batch_count = self
                .diagnostic_list_stats
                .batch_count
                .saturating_add(list.batches().len() as u64);
            self.diagnostic_list_stats.builtin_tile_item_count = self
                .diagnostic_list_stats
                .builtin_tile_item_count
                .saturating_add(list.builtin_tile_items().len() as u64);
        }
    }

    /// Whether the main UI renderer currently applies the upload-time anaglyph transform.
    #[must_use]
    pub fn anaglyph_enabled(&self) -> bool {
        self.renderer.anaglyph_enabled()
    }

    /// Toggle Java's global texture-upload transform for the foreground and panorama renderers.
    ///
    /// Existing resident textures are rebuilt from raw bytes, so font sheets and resource-pack
    /// art change in the same frame without accumulating a second conversion.
    pub fn set_anaglyph_with_gpu(&mut self, gpu: &Gpu, enabled: bool) {
        self.renderer.set_anaglyph_with_gpu(gpu, enabled);
        if let Some(panorama) = &mut self.panorama {
            panorama.renderer.set_anaglyph_with_gpu(gpu, enabled);
        }
    }

    /// Set the transform for subsequent uploads in both UI renderers without rebuilding resident
    /// textures.  Use [`Self::set_anaglyph_with_gpu`] for a live setting change.
    pub fn set_anaglyph(&mut self, enabled: bool) {
        self.renderer.set_anaglyph(enabled);
        if let Some(panorama) = &mut self.panorama {
            panorama.renderer.set_anaglyph(enabled);
        }
    }

    /// Change the `guiScale` setting or the unicode flag, recomputing the resolution.
    pub fn set_scale_options(&mut self, gui_scale: i32, unicode: bool) {
        self.gui_scale = gui_scale;
        self.unicode = unicode;
        let (w, h) = self.resolution.display_size();
        self.set_resolution(ScaledResolution::new(
            w.max(0) as u32,
            h.max(0) as u32,
            gui_scale,
            unicode,
        ));
    }

    /// Hand over this frame's geometry. The previous frame's list is cleared and kept as the
    /// spare that [`Self::take_spare_list`] returns, so its allocations are reused.
    pub fn submit(&mut self, list: DrawList) {
        self.record_list_stats(&list);
        self.retire_list(list);
        if let Some(panorama) = &mut self.panorama {
            panorama.submission = None;
        }
    }

    /// An empty [`DrawList`] for the next frame, reusing the capacity of the list submitted two
    /// frames ago. [`DrawList::clear`] has already run on it, so it equals [`DrawList::new`];
    /// only its allocations, builtin tile-model cache and configuration (Solid merging) remain.
    pub fn take_spare_list(&mut self) -> DrawList {
        std::mem::take(&mut self.spare)
    }

    /// Install `list` as the list to draw, recycling the previous one as the spare. A new list
    /// starts unsplit: a split belongs to the list it was measured on.
    fn retire_list(&mut self, list: DrawList) {
        let mut previous = std::mem::replace(&mut self.list, list);
        previous.clear();
        self.spare = previous;
        self.split = None;
    }

    /// Split the submitted list in two at vertex `at`: [`RenderPhase::record`]
    /// then draws `[0, at)` (the HUD) and [`Self::record_tail`] (through [`UiScreenPhase`], at
    /// `PhaseOrder::UI_SCREEN`) draws the rest (the open screen), so a phase recorded between
    /// them (the menu blur, `PhaseOrder::MENU_BLUR`) sees the HUD but not the screen. `None`
    /// draws everything in `record`, exactly as before.
    ///
    /// Call it after [`Self::submit`] (a submit clears the split). The two halves are exactly the
    /// pixels of one unsplit draw when nothing records between them; `tests/x10_ui_split.rs`
    /// proves it byte for byte. The list is a triangle list, so `at` is rounded down to a whole
    /// triangle (a split inside one would drop it); the end of any widget already is one.
    pub fn set_split(&mut self, at: Option<usize>) {
        self.split = at.map(|at| {
            let at = at.min(self.list.vertices().len());
            at - at % 3
        });
    }

    /// The split set by [`Self::set_split`] for the current list.
    #[must_use]
    pub fn split(&self) -> Option<usize> {
        self.split
    }

    /// Draw the open screen's half of a split list (vertices from [`Self::split`] on), with the
    /// depth the HUD half left in place. Does nothing without a split. The vertex upload made by
    /// [`RenderPhase::record`] earlier in the frame is reused.
    pub fn record_tail(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        let Some(split) = self.split else {
            return;
        };
        let len = self.list.vertices().len();
        if split >= len {
            return;
        }
        let draw_calls = {
            let mut pass = gpu.begin_pass(enc, view, "ui/screen", DepthAttachment::Shared);
            self.renderer
                .draw_vertex_range(&mut pass, &self.list, split..len)
        };
        self.renderer.record_draw_calls(draw_calls);
    }

    /// Hand over a main-menu frame split into its private panorama seed and normal foreground.
    ///
    /// The panorama renderer is allocated lazily and retained across menu frames. Its independent
    /// [`UiRenderer`] is intentional: seed and foreground are recorded into the same command
    /// encoder, and reusing one projection uniform/vertex buffer would make both passes observe
    /// whichever upload happened last when the command buffer finally executes.
    ///
    /// Missing seed textures are returned with the same semantics as
    /// [`UiRenderer::ensure_list_textures`]. The caller should continue ensuring the foreground
    /// list through [`UiPhase::renderer`] before this call.
    pub fn submit_with_panorama(
        &mut self,
        gpu: &Gpu,
        assets: &ResourceManager,
        foreground: DrawList,
        submission: PanoramaFrameSubmission,
    ) -> Vec<ResourceLocation> {
        self.record_list_stats(&foreground);
        self.record_list_stats(&submission.seed);
        let panorama = self.panorama.get_or_insert_with(|| PanoramaSupport {
            blur: PanoramaBlur::new(
                gpu,
                PanoramaBlur::VANILLA_SIZE,
                PanoramaBlur::VANILLA_SIZE,
                gpu.config.format,
            ),
            renderer: UiRenderer::new_with_anaglyph(gpu, self.renderer.anaglyph_enabled()),
            submission: None,
        });
        if self.diagnostics_enabled {
            panorama.renderer.set_diagnostics_enabled(true);
        }
        let missing = panorama
            .renderer
            .ensure_list_textures(gpu, assets, &submission.seed);
        panorama.submission = Some(submission);
        self.retire_list(foreground);
        missing
    }

    /// Clear both normal and private-panorama texture caches after a resource-pack reload.
    ///
    /// Callers that only clear [`UiPhase::renderer`] would otherwise leave the six panorama faces
    /// pinned to the old pack in the dedicated seed renderer.
    pub fn clear_textures(&mut self, gpu: &Gpu) {
        self.renderer.clear_textures(gpu);
        if let Some(panorama) = &mut self.panorama {
            panorama.renderer.clear_textures(gpu);
        }
    }

    /// The list that will be drawn next.
    #[must_use]
    pub fn list(&self) -> &DrawList {
        &self.list
    }

    /// Whether the submitted frame contains the private-target main-menu panorama pipeline.
    #[must_use]
    pub fn has_panorama_submission(&self) -> bool {
        self.panorama
            .as_ref()
            .and_then(|support| support.submission.as_ref())
            .is_some()
    }

    /// Change the scaled resolution; the projection is re-uploaded on the next record.
    pub fn set_resolution(&mut self, resolution: ScaledResolution) {
        if self.resolution != resolution {
            self.resolution = resolution;
            self.resolution_dirty = true;
        }
    }

    /// The resolution currently in force.
    #[must_use]
    pub fn resolution(&self) -> ScaledResolution {
        self.resolution
    }
}

impl RenderPhase for UiPhase {
    fn name(&self) -> &'static str {
        "ui"
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &mc_render::Camera,
    ) {
        mc_core::profile_scope!("ui.record");
        if let Some(panorama) = &mut self.panorama {
            if let Some(submission) = &panorama.submission {
                panorama
                    .renderer
                    .set_resolution(gpu, &panorama_seed_resolution());
                panorama.renderer.upload_vertices(gpu, &submission.seed);
                if !submission.seed.is_empty() {
                    let draw_calls = {
                        let mut pass = panorama.blur.begin_seed_pass(enc);
                        panorama.renderer.draw(&mut pass, &submission.seed)
                    };
                    panorama.renderer.record_draw_calls(draw_calls);
                    panorama
                        .blur
                        .blur_and_composite(gpu, enc, view, submission.composite_uv);
                }
            }
        }
        if self.resolution_dirty {
            self.renderer.set_resolution(gpu, &self.resolution);
            self.resolution_dirty = false;
        }
        self.renderer.upload_vertices(gpu, &self.list);
        if self.list.is_empty() {
            return;
        }
        // `EntityRenderer.setupOverlayRendering` clears the GUI depth buffer before it enters
        // screen space. The UI phase mirrors that boundary, then item/player batches select
        // their own LEQUAL/EQUAL windows while ordinary widgets use Always/no-write pipelines.
        let draw_calls = {
            let mut pass = gpu.begin_pass(enc, view, "ui", DepthAttachment::Clear);
            match self.split {
                None => self.renderer.draw(&mut pass, &self.list),
                // Only the HUD half; `record_tail` draws the rest.
                Some(split) => self
                    .renderer
                    .draw_vertex_range(&mut pass, &self.list, 0..split),
            }
        };
        self.renderer.record_draw_calls(draw_calls);
    }

    fn resize(&mut self, _gpu: &Gpu, width: u32, height: u32) {
        self.set_resolution(ScaledResolution::new(
            width,
            height,
            self.gui_scale,
            self.unicode,
        ));
    }
}

/// The open screen's half of a split [`UiPhase`] list. Register it at `PhaseOrder::UI_SCREEN` with a second handle to the UI
/// phase; it is disabled (one `RefCell` borrow and an `Option` test per frame) unless
/// [`UiPhase::set_split`] split this frame's list.
pub struct UiScreenPhase(pub std::rc::Rc<RefCell<UiPhase>>);

impl UiScreenPhase {
    /// The phase's name in `PhaseList`, GPU timings and captures.
    pub const NAME: &'static str = "ui/screen";
}

impl std::fmt::Debug for UiScreenPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UiScreenPhase")
            .field("split", &self.0.borrow().split())
            .finish()
    }
}

impl RenderPhase for UiScreenPhase {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn enabled(&self) -> bool {
        self.0.borrow().split().is_some()
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &mc_render::Camera,
    ) {
        mc_core::profile_scope!("ui.record_screen");
        self.0.borrow_mut().record_tail(gpu, enc, view);
    }
}

#[cfg(test)]
mod panorama_contract_tests {
    use super::*;

    #[test]
    fn ui_diagnostics_are_opt_in_and_count_repeated_panorama_submissions() {
        use mc_render::{Gpu, GpuOptions, RenderError};

        let gpu = match Gpu::headless(
            64,
            64,
            GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(gpu) => gpu,
            Err(RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter available on this machine");
                return;
            }
            Err(error) => panic!("headless context creation failed: {error}"),
        };

        let mut phase = UiPhase::new(&gpu, 64, 64, 1, false);
        assert_eq!(phase.diagnostic_stats(), None);

        phase.set_diagnostics_enabled(true);
        phase.begin_diagnostic_frame();
        phase.record_build_cpu(std::time::Duration::from_nanos(17));

        let mut first = DrawList::new();
        first.rect(0, 0, 1, 1, 0xFFFF_FFFF);
        phase.submit(first);

        let mut foreground = DrawList::new();
        foreground.rect(2, 2, 3, 3, 0xFFFF_FFFF);
        let mut seed = DrawList::new();
        seed.rect(4, 4, 5, 5, 0xFFFF_FFFF);
        phase.submit_with_panorama(
            &gpu,
            &mc_assets::ResourceManager::new(),
            foreground,
            PanoramaFrameSubmission::new(seed, [[0.0; 2]; 4]),
        );

        // The foreground upload is a real queue write; the panorama seed is retained in its
        // private renderer and is still represented in the list totals above.
        let list = phase.list().clone();
        phase.renderer.upload_vertices(&gpu, &list);
        let stats = phase.diagnostic_stats().expect("diagnostics enabled");
        assert_eq!(stats.build_cpu_ns, Some(17));
        assert_eq!(stats.vertex_count, 18); // three six-vertex rects, including the panorama seed
        assert_eq!(stats.batch_count, 3);
        assert_eq!(stats.vertex_upload_bytes, 6 * UiVertex::SIZE as u64);
        assert_eq!(
            stats.depth_upload_bytes,
            6 * std::mem::size_of::<f32>() as u64
        );

        // Re-enabling is idempotent: only the explicit frame boundary resets counters.
        phase.set_diagnostics_enabled(true);
        assert_eq!(phase.diagnostic_stats(), Some(stats));
        phase.begin_diagnostic_frame();
        assert_eq!(
            phase.diagnostic_stats(),
            Some(UiFrameStats::default()),
            "begin_diagnostic_frame resets the opt-in snapshot"
        );
        phase.set_diagnostics_enabled(false);
        assert_eq!(phase.diagnostic_stats(), None);
    }

    #[test]
    fn crosshair_pipeline_descriptor_is_vanilla_inverting_blend_not_alpha() {
        let invert = blend_state_for(UiBlend::Invert);
        assert_eq!(
            invert.color,
            wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::OneMinusDst,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
            "GuiIngame's crosshair pass is GL_ONE_MINUS_DST_COLOR / GL_ONE_MINUS_SRC_COLOR"
        );
        assert_eq!(invert.alpha, wgpu::BlendComponent::REPLACE);
        assert_eq!(write_mask_for(UiBlend::Invert), wgpu::ColorWrites::ALL);
        assert_ne!(invert, wgpu::BlendState::ALPHA_BLENDING);

        // Numeric negative control: alpha blending and vanilla inversion only coincide for
        // special black/white inputs.  This pair catches a regression to the ordinary GUI
        // pipeline even when a screenshot happens to use a monochrome background.
        let src = 1.0_f32;
        let dst = 0.25_f32;
        let inverted = src * (1.0 - dst) + dst * (1.0 - src);
        let alpha_mutant = src * 0.5 + dst * 0.5;
        assert!((inverted - 0.75).abs() < f32::EPSILON);
        assert!((alpha_mutant - 0.625).abs() < f32::EPSILON);
        assert_ne!(inverted, alpha_mutant);
    }

    #[test]
    fn item_glint_pipeline_descriptor_is_src_color_plus_one() {
        let glint = blend_state_for(UiBlend::Glint);
        let expected = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Src,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        assert_eq!(glint.color, expected, "RenderItem.renderEffect color blend");
        assert_eq!(
            glint.alpha, expected,
            "RenderItem uses glBlendFunc for alpha too"
        );
        assert_eq!(write_mask_for(UiBlend::Glint), wgpu::ColorWrites::ALL);
        assert_ne!(glint, wgpu::BlendState::ALPHA_BLENDING);

        // Numeric negative control: SRC_COLOR/ONE is materially brighter than the ordinary
        // alpha pipeline for a nontrivial source/destination pair, so a regression to alpha
        // blending cannot satisfy this descriptor's intended equation.
        let src = 0.5_f32;
        let dst = 0.25_f32;
        let glint_value = src * src + dst;
        let alpha_mutant = src * src + dst * (1.0 - src);
        assert!((glint_value - 0.5).abs() < f32::EPSILON);
        assert!((alpha_mutant - 0.375).abs() < f32::EPSILON);
        assert_ne!(glint_value, alpha_mutant);
    }

    #[test]
    fn text_selection_pipeline_descriptors_are_or_reverse_not_alpha_or_xor() {
        let complement = blend_state_for(UiBlend::XorComplement);
        assert_eq!(
            complement.color,
            wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Subtract,
            },
            "opaque white subtracts the destination to produce 1 - dst"
        );
        assert_eq!(complement.alpha, wgpu::BlendComponent::REPLACE);
        assert_eq!(
            write_mask_for(UiBlend::XorComplement),
            wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN,
            "the complement pass cannot disturb blue or alpha"
        );

        let force_blue = blend_state_for(UiBlend::Xor);
        assert_eq!(force_blue, wgpu::BlendState::REPLACE);
        assert_eq!(
            write_mask_for(UiBlend::Xor),
            wgpu::ColorWrites::BLUE | wgpu::ColorWrites::ALPHA,
            "the opaque blue pass completes 1 | !dst = 1"
        );

        // Biting controls: the old single-pass XOR-shaped state and the ordinary alpha state
        // would both fail these descriptor assertions, even though a black/white screenshot can
        // make the resulting highlight look deceptively close.
        assert_ne!(complement, wgpu::BlendState::ALPHA_BLENDING);
        assert_ne!(force_blue, wgpu::BlendState::ALPHA_BLENDING);
        assert_ne!(
            write_mask_for(UiBlend::XorComplement),
            wgpu::ColorWrites::ALL
        );
    }

    #[test]
    fn panorama_submission_transports_seed_and_transposed_crop_losslessly() {
        let mut seed = DrawList::new();
        seed.rect(1, 2, 17, 19, 0xFF12_3456);
        let uv = [[0.71, 0.92], [0.21, 0.82], [0.61, 0.12], [0.31, 0.02]];
        let submission = PanoramaFrameSubmission::new(seed.clone(), uv);

        assert_eq!(submission.seed, seed);
        assert_eq!(submission.composite_uv, uv);
        assert_eq!(submission.seed.vertices().len(), 6);

        // Mutation control: the order is semantically meaningful and deliberately differs from
        // a conventional TL/TR/BL/BR quad. Sentinel corners make any constructor-side reorder
        // visible instead of letting symmetric crop values hide it.
        assert_ne!(submission.composite_uv[1], submission.composite_uv[2]);
    }

    #[test]
    fn panorama_seed_projection_is_the_fixed_two_fifty_six_square() {
        let resolution = panorama_seed_resolution();
        assert_eq!(resolution.display_size(), (256, 256));
        assert_eq!(
            (resolution.scaled_width(), resolution.scaled_height()),
            (256, 256)
        );
        assert_eq!(resolution.scale_factor(), 1);
    }
}
