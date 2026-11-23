//! [`PanoramaBlur`]: a `wgpu`-native rewrite of `GuiMainMenu.rotateAndBlurSkybox`.
//!
//! java: `client/gui/GuiMainMenu.java` — `drawPanorama` (`:374`), `rotateAndBlurSkybox` (`:464`),
//! `renderSkybox` (`:499`).
//!
//! # The cross-section question this answers
//!
//! Plan §13's S16→S10 row asked, of [`SceneTextures`](crate::SceneTextures): "a follow-up should
//! check rather than assume" whether it closes the wall the main menu's panorama blur hit — a
//! phase that must sample its own accumulated output, which
//! [`RenderPhase::record`](super::RenderPhase::record) cannot express (it hands out a bare
//! `TextureView`, never a `Texture`).
//!
//! **It does close the wall, structurally — but the concrete `SceneTargets` type (the private
//! struct backing [`SceneTextures`](crate::SceneTextures) inside [`super`]) is not itself
//! reusable here, so this module reimplements the same *pattern* at the right scale instead of
//! reaching for that type.** Three reasons the type doesn't fit directly:
//!
//! 1. `SceneTargets` is sized from `gpu.config.width`/`height` — the window's swap-chain size.
//!    Vanilla's blur runs over a fixed 256×256 buffer, `GlStateManager.viewport(0, 0, 256, 256)`,
//!    completely decoupled from the display's resolution.
//! 2. It is allocated once per [`PhaseList`](super::PhaseList) and shared by every phase that
//!    frame; the panorama's seven passes need their own private accumulator that nothing else
//!    touches, snapshotted and redrawn seven times *before* anything else in the frame runs.
//! 3. `PhaseList::render` copies the snapshot **once per phase**, immediately before that phase
//!    records. Vanilla's blur is seven applications of the *same* operation to its own prior
//!    output — closer to seven single-phase-list-cycle iterations than to seven different phases
//!    — so a bespoke loop expresses it more directly than seven distinct
//!    [`RenderPhase`](super::RenderPhase) impls would.
//!
//! What *does* generalise, and is exactly what makes this soluble at all, is the pattern
//! `SceneTargets` established: own a same-sized (accumulator, snapshot)
//! pair, `copy_texture_to_texture` the accumulator into the snapshot immediately before a pass
//! that wants to read what is already there, and let the pass write the accumulator while
//! sampling the snapshot — never the same texture bound as both input and render target in one
//! pass, which `wgpu` (rightly) forbids and which is exactly the GL feedback loop
//! `glCopyTexSubImage2D` was manually working around. [`PanoramaBlur`] is that pattern, run in a
//! loop instead of threaded through the phase list.
//!
//! # The maths, derived from the decompiled source
//!
//! `rotateAndBlurSkybox` (`GuiMainMenu.java:464-494`) does not literally rotate anything with a
//! matrix. Per call it:
//!
//! 1. `glCopyTexSubImage2D`s the current 256×256 framebuffer into `backgroundTexture` (`GL_LINEAR`
//!    filtered) — the snapshot step.
//! 2. Submits **one** draw call carrying three overlapping full-viewport quads (`for (j = 0; j < 3;
//!    ++j)`), each sampling `backgroundTexture` with a texel offset `f1 = (j - 1) / 256.0` added to
//!    one texture coordinate, blended with `tryBlendFuncSeparate(GL_SRC_ALPHA,
//!    GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ZERO)` at vertex alpha `1.0 / (j + 1)` — i.e. `1, 1/2,
//!    1/3` — with `colorMask(true, true, true, false)`, so only RGB blends and the framebuffer is
//!    never cleared first.
//!
//! Two things fall out of reading the vertex data literally (`worldrenderer.pos(x,
//! y).tex(u, v)`, four corners per quad):
//!
//! **Each pass mirrors the image across its diagonal.** Writing screen position as normalised
//! `(X, Y)` with `Y = 0` at the top (vanilla's GUI convention) and reading off the four
//! `.tex(...)` calls: `s = (1 - Y) + f1`, `t = X`. `glCopyTexSubImage2D` puts the framebuffer's
//! *bottom* row at `t = 0`, so in image terms (`y` down) a pass reads `(x, y) = (1 - Y, 1 - X)`:
//! a transpose, not a turn. The offset `f1` always perturbs `s`, but each pass's input is already
//! mirrored, so the blur direction alternates between the image's two axes pass over pass with no
//! per-pass state. Both shaders below keep Java's literal `(s, t)` and convert `t` to `1 - t`,
//! because the textures here store their top row first.
//!
//! **The screen shows six passes, turned upright.** Each `rotateAndBlurSkybox` copies the
//! framebuffer *before* it draws, and `renderSkybox`'s final quad samples that copy, so the screen
//! shows what the 7th call copied: six passes, an even number of mirrors. The final quad's swapped
//! texture coordinates then turn that image 90° counter-clockwise, which undoes the 90° clockwise
//! turn `drawPanorama`'s `rotate(90, 0, 0, 1)` gives the seed. [`PanoramaBlur::composite`]
//! therefore samples the snapshot the last pass took. This module used to read `t` top-down,
//! which made each pass a 90° turn, and composited the 7th pass's output: the main-menu backdrop
//! came out sideways and mirrored.
//!
//! **The three-draw alpha cascade collapses to a plain average, independent of what was there
//! before.** Sequential "source over" blending is `dst' = src·a + dst·(1 - a)`. With `a = 1` on
//! the *first* of the three draws, that draw alone completely replaces the destination —
//! `colorMask`'s "don't clear it first" is therefore visually irrelevant, not a hidden dependency
//! this port has to reproduce. Unrolling the three draws (`T` = the snapshot, sampled at the
//! rotated UV plus offset `oⱼ = (j - 1)/256`):
//!
//! ```text
//! after j=0 (a=1):   r0 = T(u+o0)
//! after j=1 (a=1/2): r1 = T(u+o1)·(1/2) + r0·(1/2)
//! after j=2 (a=1/3): r2 = T(u+o2)·(1/3) + r1·(2/3)
//!                       = [T(u+o0) + T(u+o1) + T(u+o2)] / 3
//! ```
//!
//! which is exactly a three-tap box average of `T` at `u - 1/256, u, u + 1/256` in the rotated
//! frame — the classic "weighted running average via alpha blending" trick, reproduced here as a
//! single fragment-shader evaluation instead of three blended draws. This is not an approximation
//! of vanilla's blend state machine; it is the closed form of it (see this module's
//! `the_three_tap_cascade_is_exactly_the_alpha_blend_math_it_replaces` test for the algebra
//! checked against literal sequential blending on the CPU).
//!
//! # What remains for the live caller
//!
//! `mc_ui::screens::menu::Panorama::draw_to_target` now emits vanilla's square-projected,
//! 64-sample first stage, and `PanoramaRenderPlan` exposes the literal final crop UVs. This module
//! owns the matching private target, seven self-sampling blur passes, and final full-window
//! composite pipeline. The only remaining cross-crate work is orchestration: draw that first-stage
//! list into [`PanoramaBlur::view`], then call [`PanoramaBlur::blur_and_composite`] before the rest
//! of the menu UI. Keeping that orchestration outside mc-render avoids a dependency cycle
//! (`mc-ui` already depends on this crate).
//!
//! # Usage
//!
//! ```no_run
//! use mc_render::pipeline::panorama_blur::PanoramaBlur;
//! # fn wire(gpu: &mc_render::Gpu, enc: &mut wgpu::CommandEncoder, frame: &wgpu::TextureView) {
//! let blur = PanoramaBlur::new(gpu, PanoramaBlur::VANILLA_SIZE, PanoramaBlur::VANILLA_SIZE, gpu.config.format);
//! // ... render `Panorama::draw_to_target` into `blur.view()` here ...
//! let crop_uv = [[0.75, 0.95], [0.25, 0.95], [0.75, 0.05], [0.25, 0.05]];
//! blur.blur_and_composite(gpu, enc, frame, crop_uv);
//! # }
//! ```

use crate::Gpu;

/// The vertex+fragment shader for one blur pass. Static: every one of the seven passes runs the
/// identical shader over its own freshly-copied snapshot, because the mirroring compounds through
/// the *texture contents*, not through any per-pass shader state. See the module docs for the
/// derivation.
const FS_SHADER: &str = r#"
struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Four screen corners (NDC) and the GL texture coordinates `(s, t) = (1 - Y, X)` they carry, read
// straight off `GuiMainMenu.rotateAndBlurSkybox`'s four `.tex(...)` calls (java:464-494). Vanilla
// fills that texture with `glCopyTexSubImage2D`, so its `t = 0` row is the framebuffer's *bottom*
// row, while this snapshot stores the target's top row first: `t` becomes `1 - t` below. The
// pass is therefore a transpose (a mirror across the diagonal), not a 90 degree turn.
@vertex
fn vs(@builtin(vertex_index) index: u32) -> VsOut {
    var corner_pos = array<vec2<f32>, 4>(
        vec2<f32>(-1.0,  1.0), // top-left     (X=0, Y=0)
        vec2<f32>( 1.0,  1.0), // top-right    (X=1, Y=0)
        vec2<f32>( 1.0, -1.0), // bottom-right (X=1, Y=1)
        vec2<f32>(-1.0, -1.0), // bottom-left  (X=0, Y=1)
    );
    var corner_uv = array<vec2<f32>, 4>(
        vec2<f32>(1.0, 0.0), // top-left:     u=1-Y=1, v=X=0
        vec2<f32>(1.0, 1.0), // top-right:    u=1-Y=1, v=X=1
        vec2<f32>(0.0, 1.0), // bottom-right: u=1-Y=0, v=X=1
        vec2<f32>(0.0, 0.0), // bottom-left:  u=1-Y=0, v=X=0
    );
    var order = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u);
    let corner = order[index];

    var out: VsOut;
    out.clip = vec4<f32>(corner_pos[corner], 0.0, 1.0);
    let gl_uv = corner_uv[corner];
    out.uv = vec2<f32>(gl_uv.x, 1.0 - gl_uv.y);
    return out;
}

@group(0) @binding(0) var snapshot: texture_2d<f32>;
@group(0) @binding(1) var snapshot_sampler: sampler;

// java: `f1 = (j - i / 2) / 256.0F` for `i = 3`, `j = 0..3` -> exactly -1/256, 0, +1/256.
const TAP_OFFSET: f32 = 1.0 / 256.0;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let a = textureSample(snapshot, snapshot_sampler, in.uv + vec2<f32>(-TAP_OFFSET, 0.0));
    let b = textureSample(snapshot, snapshot_sampler, in.uv);
    let c = textureSample(snapshot, snapshot_sampler, in.uv + vec2<f32>(TAP_OFFSET, 0.0));
    return vec4<f32>((a.rgb + b.rgb + c.rgb) / 3.0, 1.0);
}
"#;

/// Final `renderSkybox` composite: fill the destination while sampling the blurred accumulator
/// at the four crop UVs supplied by `mc_ui::screens::menu::PanoramaRenderPlan`.
const COMPOSITE_SHADER: &str = r#"
struct VsIn {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// `in.uv` is Java's literal `(s, t)`, whose `t = 0` is the bottom row of the texture
// `glCopyTexSubImage2D` filled. The snapshot stores the target's top row first, so `t` flips.
@vertex
fn vs(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = vec2<f32>(in.uv.x, 1.0 - in.uv.y);
    return out;
}

@group(0) @binding(0) var panorama: texture_2d<f32>;
@group(0) @binding(1) var panorama_sampler: sampler;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(textureSample(panorama, panorama_sampler, in.uv).rgb, 1.0);
}
"#;

/// One vertex of the full-window final panorama composite.
///
/// Public only so the CPU-side contract can be inspected without a GPU. Normal callers pass the
/// four crop UVs to [`PanoramaBlur::composite`] rather than constructing these directly.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PanoramaCompositeVertex {
    /// Full-window clip-space position.
    pub position: [f32; 2],
    /// Coordinate in the blurred 256×256 accumulator.
    pub uv: [f32; 2],
}

/// Expand `DrawList::quad` corner order (top-left, bottom-left, top-right, bottom-right) into the
/// two triangles consumed by the renderer.
///
/// `uv` is intentionally not recomputed here: `PanoramaRenderPlan` owns the Java window-aspect
/// crop and this renderer consumes that contract literally. This keeps mc-render independent of
/// mc-ui while preventing a second subtly different crop implementation.
#[must_use]
pub fn panorama_composite_vertices(uv: [[f32; 2]; 4]) -> [PanoramaCompositeVertex; 6] {
    const POSITIONS: [[f32; 2]; 4] = [[-1.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [1.0, -1.0]];
    [0, 1, 2, 1, 3, 2].map(|i| PanoramaCompositeVertex {
        position: POSITIONS[i],
        uv: uv[i],
    })
}

/// A self-referential accumulator: owns a same-sized (accumulator, snapshot) texture pair and
/// runs vanilla's `rotateAndBlurSkybox` maths — a diagonal-mirroring three-tap average — as
/// many times as asked. See the module docs for what this is, why the pattern (not the
/// `SceneTargets` type backing [`SceneTextures`](crate::SceneTextures)) is what actually
/// generalises here, and for the algebra behind the shader.
pub struct PanoramaBlur {
    width: u32,
    height: u32,
    accumulator: wgpu::Texture,
    accumulator_view: wgpu::TextureView,
    // UiRenderer's pipeline uses the window's MSAA count. When that is greater than one, the
    // sharp panorama must be drawn into a matching private attachment and resolved into the
    // single-sampled accumulator before the blur can sample it.
    seed_msaa_view: Option<wgpu::TextureView>,
    // Only `snapshot` (the `Texture`) is needed after construction — `blur_pass` copies into it
    // by reference. Its `TextureView` is only used to build `bind_group` and
    // `composite_bind_group`, and does not need to outlive that: `wgpu` retains its own
    // reference to the resources a bind group was created from.
    snapshot: wgpu::Texture,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    composite_pipeline: wgpu::RenderPipeline,
    composite_bind_group: wgpu::BindGroup,
    composite_vertex_buffer: wgpu::Buffer,
}

impl std::fmt::Debug for PanoramaBlur {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PanoramaBlur")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl PanoramaBlur {
    /// Vanilla's fixed offscreen buffer size (`GlStateManager.viewport(0, 0, 256, 256)`,
    /// `GuiMainMenu.java:502`). Not required — [`PanoramaBlur::new`] takes any size — but this is
    /// the value that reproduces vanilla's blur radius bit-for-bit, since the shader's tap offset
    /// (`TAP_OFFSET`) is exactly one 256th of the buffer's width regardless of what size is
    /// chosen.
    pub const VANILLA_SIZE: u32 = 256;

    /// How many times `renderSkybox` calls `rotateAndBlurSkybox` per frame
    /// (`GuiMainMenu.java:504-510`).
    pub const PASSES: u32 = 7;

    /// Build an accumulator of `width × height` texels in `format`. `format` must support
    /// filterable sampling (every uncompressed 8-bit-per-channel colour format does); pass
    /// `gpu.config.format` for a live integration or [`crate::HEADLESS_FORMAT`] in a test.
    #[must_use]
    pub fn new(gpu: &Gpu, width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        let (width, height) = (width.max(1), height.max(1));

        let make = |label: &str, extra: wgpu::TextureUsages| {
            let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | extra,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        // The accumulator is written by a render pass (blur output, or the caller's seed render)
        // and also accepts a direct CPU write (`load_rgba`), so it carries both usages up front.
        let (accumulator, accumulator_view) = make(
            "mc-render panorama blur accumulator",
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
        );
        let (snapshot, snapshot_view) = make(
            "mc-render panorama blur snapshot",
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        let seed_msaa_view = (gpu.sample_count() > 1).then(|| {
            gpu.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("mc-render panorama seed MSAA"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: gpu.sample_count(),
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        });

        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mc-render panorama blur"),
                source: wgpu::ShaderSource::Wgsl(FS_SHADER.into()),
            });
        let composite_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mc-render panorama composite"),
                source: wgpu::ShaderSource::Wgsl(COMPOSITE_SHADER.into()),
            });

        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("mc-render panorama blur group0"),
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
                    ],
                });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mc-render panorama blur layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mc-render panorama blur"),
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
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });

        let composite_pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("mc-render panorama composite"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &composite_shader,
                        entry_point: Some("vs"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[Some(wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<PanoramaCompositeVertex>() as u64,
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
                            ],
                        })],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        ..wgpu::PrimitiveState::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: gpu.sample_count(),
                        ..wgpu::MultisampleState::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &composite_shader,
                        entry_point: Some("fs"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            // The panorama is the opaque background of GuiMainMenu. Vanilla's
                            // final quad has color/alpha 1, so replacement is its exact blend.
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                });

        // java: `GL11.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN/MAG_FILTER, GL_LINEAR)`
        // (`GuiMainMenu.java:467-468`) — vanilla switches this texture to linear filtering only
        // for the blur; wrap mode is never touched there, so it keeps GL's own default (repeat).
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mc-render panorama blur linear"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..wgpu::SamplerDescriptor::default()
        });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mc-render panorama blur group0"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&snapshot_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        // java: `renderSkybox`'s final quad samples `backgroundTexture`, which still holds what
        // the 7th `rotateAndBlurSkybox` copied *before* drawing: six passes, not seven. The
        // snapshot is that copy; the 7th pass's own output never reaches the screen.
        let composite_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mc-render panorama composite group0"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&snapshot_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let composite_vertex_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc-render panorama composite vertices"),
            size: (std::mem::size_of::<PanoramaCompositeVertex>() * 6) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            width,
            height,
            accumulator,
            accumulator_view,
            seed_msaa_view,
            snapshot,
            pipeline,
            bind_group,
            composite_pipeline,
            composite_bind_group,
            composite_vertex_buffer,
        }
    }

    /// Width in texels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in texels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Begin the sharp-panorama seed pass, clearing the private target first.
    ///
    /// The attachment's sample count matches [`Gpu::sample_count`], so the existing mc-ui
    /// renderer can draw `Panorama::draw_to_target` without constructing a second UI pipeline.
    /// With MSAA enabled the pass resolves into the single-sampled accumulator automatically;
    /// otherwise it writes the accumulator directly. Drop the returned pass before calling
    /// [`Self::blur`] or [`Self::blur_and_composite`].
    pub fn begin_seed_pass<'a>(
        &'a self,
        enc: &'a mut wgpu::CommandEncoder,
    ) -> wgpu::RenderPass<'a> {
        let (view, resolve_target) = match &self.seed_msaa_view {
            Some(msaa) => (msaa, Some(&self.accumulator_view)),
            None => (&self.accumulator_view, None),
        };
        enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mc-render panorama seed"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    /// The accumulator's view. Before the first [`PanoramaBlur::blur_pass`] this is wherever the
    /// caller's seed render (or [`PanoramaBlur::load_rgba`]) left it; after `n` calls it holds the
    /// result of `n` blur passes. Usable as a render-pass colour attachment (to seed it) or as an
    /// ordinary sampled texture (to consume the result) — never both in the same pass.
    #[must_use]
    pub fn view(&self) -> &wgpu::TextureView {
        &self.accumulator_view
    }

    /// Seed the accumulator directly from CPU-side RGBA8 pixels, tightly packed
    /// (`width * height * 4` bytes). A convenience for callers and tests that already have the
    /// sharp panorama as pixels rather than as a render target to draw into.
    ///
    /// # Panics
    /// If `rgba.len()` does not match `width * height * 4`.
    pub fn load_rgba(&self, gpu: &Gpu, rgba: &[u8]) {
        let expected = (self.width as usize) * (self.height as usize) * 4;
        assert_eq!(
            rgba.len(),
            expected,
            "panorama blur seed must be exactly width*height*4 bytes"
        );
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.accumulator,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.width * 4),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// One `rotateAndBlurSkybox` call: snapshot the accumulator, then overwrite it with the
    /// rotated three-tap average of that snapshot. See the module docs for why this single
    /// full-screen draw is the exact closed form of vanilla's three-quad alpha cascade.
    ///
    /// Takes no `&Gpu` — matching `SceneTargets::snapshot`/`blit_to` in this same file — because
    /// recording draw commands needs only the encoder; the device was only ever needed to build
    /// the pipeline, back in [`PanoramaBlur::new`].
    pub fn blur_pass(&self, enc: &mut wgpu::CommandEncoder) {
        enc.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.accumulator,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.snapshot,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mc-render panorama blur pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.accumulator_view,
                depth_slice: None,
                resolve_target: None,
                // The fragment shader fully overwrites every covered texel (it is an unweighted
                // average, not a blend against the destination — see the module docs' derivation
                // for why that is faithful rather than a shortcut), so the load op is a formality.
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..6, 0..1);
    }

    /// Run [`PanoramaBlur::PASSES`] blur passes — the whole of `renderSkybox`'s
    /// `rotateAndBlurSkybox` × 7 (`GuiMainMenu.java:504-510`).
    pub fn blur(&self, enc: &mut wgpu::CommandEncoder) {
        for _ in 0..Self::PASSES {
            self.blur_pass(enc);
        }
    }

    /// Composite the blurred panorama over the full destination using the exact four crop UVs
    /// from `mc_ui::screens::menu::PanoramaRenderPlan::composite_uv` (Java's literal GL texture
    /// coordinates; the shader flips `t`).
    ///
    /// This samples the snapshot the last [`Self::blur_pass`] took, which is the texture
    /// vanilla's final quad samples, so it must follow at least one blur pass in the same
    /// encoder. See the module docs for why that is six passes, not seven.
    ///
    /// This is `GuiMainMenu.renderSkybox`'s last quad. The supplied UV order is top-left,
    /// bottom-left, top-right, bottom-right, matching `DrawList::quad`; the method expands that
    /// strip order to two triangles and writes an opaque full-screen image. The pipeline is built
    /// for the `format` and MSAA count supplied to [`Self::new`], so `target` must be the current
    /// frame view belonging to that same [`Gpu`].
    pub fn composite(
        &self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        composite_uv: [[f32; 2]; 4],
    ) {
        let vertices = panorama_composite_vertices(composite_uv);
        gpu.queue.write_buffer(
            &self.composite_vertex_buffer,
            0,
            bytemuck::cast_slice(&vertices),
        );

        let mut pass = gpu.begin_pass(
            enc,
            target,
            "mc-render panorama composite",
            crate::DepthAttachment::None,
        );
        pass.set_pipeline(&self.composite_pipeline);
        pass.set_bind_group(0, &self.composite_bind_group, &[]);
        pass.set_vertex_buffer(0, self.composite_vertex_buffer.slice(..));
        pass.draw(0..6, 0..1);
    }

    /// Run vanilla's seven self-sampling blur passes and immediately draw its final crop quad.
    ///
    /// The caller must seed [`Self::view`] first. Keeping blur and composite in the same command
    /// encoder guarantees the final draw observes this frame's 7th snapshot (the result of six
    /// passes, as in vanilla) rather than the previous frame's.
    pub fn blur_and_composite(
        &self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        composite_uv: [[f32; 2]; 4],
    ) {
        self.blur(enc);
        self.composite(gpu, enc, target, composite_uv);
    }

    /// Read the accumulator back to the CPU as tightly-packed RGBA8, blocking until the GPU is
    /// done. For tests and golden-pixel proofs; a live frame never calls this.
    #[must_use]
    pub fn read_rgba(&self, gpu: &Gpu) -> Vec<u8> {
        let unpadded = self.width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc-render panorama blur readback"),
            size: u64::from(padded) * u64::from(self.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mc-render panorama blur readback"),
            });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.accumulator,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(Some(enc.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll for readback");
        rx.recv().expect("map callback").expect("map succeeded");

        let view = staging.get_mapped_range(..).expect("mapped range");
        let mut out = Vec::with_capacity((unpadded * self.height) as usize);
        for row in 0..self.height as usize {
            let start = row * padded as usize;
            out.extend_from_slice(&view[start..start + unpadded as usize]);
        }
        drop(view);
        staging.unmap();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuOptions, HEADLESS_FORMAT};

    fn headless(w: u32, h: u32) -> Option<Gpu> {
        match Gpu::headless(
            w,
            h,
            GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(gpu) => Some(gpu),
            Err(crate::RenderError::NoAdapter) => {
                eprintln!("skip: no wgpu adapter available on this machine");
                None
            }
            Err(e) => panic!("headless context creation failed: {e}"),
        }
    }

    #[test]
    fn final_composite_expands_draw_list_corner_order_without_recomputing_the_crop() {
        let uv = [[0.11, 0.12], [0.21, 0.22], [0.31, 0.32], [0.41, 0.42]];
        let actual = panorama_composite_vertices(uv);
        let expected = [
            PanoramaCompositeVertex {
                position: [-1.0, 1.0],
                uv: uv[0],
            },
            PanoramaCompositeVertex {
                position: [-1.0, -1.0],
                uv: uv[1],
            },
            PanoramaCompositeVertex {
                position: [1.0, 1.0],
                uv: uv[2],
            },
            PanoramaCompositeVertex {
                position: [-1.0, -1.0],
                uv: uv[1],
            },
            PanoramaCompositeVertex {
                position: [1.0, -1.0],
                uv: uv[3],
            },
            PanoramaCompositeVertex {
                position: [1.0, 1.0],
                uv: uv[2],
            },
        ];
        assert_eq!(actual, expected);

        // Mutation controls: a conventional TL/TR/BR/BL order would silently transpose the
        // plan again; each distinct sentinel must occur exactly where DrawList's 0,1,2,1,3,2
        // expansion puts it.
        assert_eq!(
            actual.map(|v| v.uv),
            [uv[0], uv[1], uv[2], uv[1], uv[3], uv[2]]
        );
    }

    #[test]
    fn final_composite_two_triangles_cover_the_entire_clip_space_quad() {
        let vertices = panorama_composite_vertices([[0.0; 2]; 4]);
        let twice_area = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| {
            ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs()
        };
        let area0 = twice_area(
            vertices[0].position,
            vertices[1].position,
            vertices[2].position,
        ) * 0.5;
        let area1 = twice_area(
            vertices[3].position,
            vertices[4].position,
            vertices[5].position,
        ) * 0.5;
        assert_eq!(area0, 2.0);
        assert_eq!(area1, 2.0);
        assert_eq!(area0 + area1, 4.0, "full -1..1 clip-space area");
        assert_eq!(std::mem::size_of::<PanoramaCompositeVertex>(), 16);
    }

    /// A checkerboard: 50% grey background with a single fully-saturated red texel dropped in
    /// the middle. Sharp, deterministic, and trivially distinguishable from anything blurred.
    fn seeded_blur(gpu: &Gpu, size: u32) -> PanoramaBlur {
        let blur = PanoramaBlur::new(gpu, size, size, HEADLESS_FORMAT);
        let mut rgba = vec![128u8; (size * size * 4) as usize];
        for p in rgba.chunks_exact_mut(4) {
            p[3] = 255;
        }
        let mid = (size / 2) as usize;
        let idx = (mid * size as usize + mid) * 4;
        rgba[idx] = 255;
        rgba[idx + 1] = 0;
        rgba[idx + 2] = 0;
        rgba[idx + 3] = 255;
        blur.load_rgba(gpu, &rgba);
        blur
    }

    fn run(blur: &PanoramaBlur, gpu: &Gpu, passes: u32) -> Vec<u8> {
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for _ in 0..passes {
            blur.blur_pass(&mut enc);
        }
        gpu.queue.submit(Some(enc.finish()));
        blur.read_rgba(gpu)
    }

    #[test]
    fn a_single_blur_pass_measurably_differs_from_the_sharp_seed() {
        let Some(gpu) = headless(4, 4) else { return };
        let size = 64;
        let blur = seeded_blur(&gpu, size);
        let seed = blur.read_rgba(&gpu);
        let blurred = run(&blur, &gpu, 1);
        assert_ne!(
            seed, blurred,
            "one rotateAndBlurSkybox pass must change the image"
        );
    }

    #[test]
    fn a_sharp_red_texel_spreads_into_previously_solid_neighbours() {
        // Before any pass, everywhere except the seeded texel is flat grey (128,128,128). One
        // pass' three-tap average, applied along a screen axis after the 90-degree UV rotation,
        // must leave a trace of red on at least one texel adjacent to the seed that used to be
        // pure grey — this is what "the blur spreads energy" means at the pixel level, not just
        // "some byte somewhere changed".
        let Some(gpu) = headless(4, 4) else { return };
        let size = 64;
        let blur = seeded_blur(&gpu, size);
        let blurred = run(&blur, &gpu, 1);

        let mid = (size / 2) as usize;
        let px = |x: usize, y: usize| -> [u8; 4] {
            let i = (y * size as usize + x) * 4;
            [blurred[i], blurred[i + 1], blurred[i + 2], blurred[i + 3]]
        };
        // Scan a small neighbourhood: at least one texel besides the centre must have picked up
        // a red bias (r meaningfully above g/b), proving the tap actually reached out spatially
        // rather than the shader degenerating into an identity copy.
        let mut spread = false;
        for dy in -2i32..=2 {
            for dx in -2i32..=2 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let (x, y) = ((mid as i32 + dx) as usize, (mid as i32 + dy) as usize);
                let [r, g, b, _] = px(x, y);
                if r > g.saturating_add(15) && r > b.saturating_add(15) {
                    spread = true;
                }
            }
        }
        assert!(
            spread,
            "expected a red bias on a neighbour of the seeded texel after one pass"
        );
    }

    /// java: `rotateAndBlurSkybox` samples a texture `glCopyTexSubImage2D` filled from the
    /// bottom row up, at `(s, t) = (1 - Y, X)`. In image terms that reads `(1 - Y, 1 - X)`: a
    /// mirror across the bottom-left/top-right diagonal. A tall block in the top-left corner must
    /// come back as a wide block in the bottom-right one. Reading `t` top-down instead (this
    /// module's old shader) turns the image a quarter turn and puts the block bottom-left.
    #[test]
    fn one_pass_mirrors_the_image_across_its_diagonal_like_the_gl_copy() {
        let Some(gpu) = headless(4, 4) else { return };
        let size = 64_usize;
        let blur = PanoramaBlur::new(&gpu, size as u32, size as u32, HEADLESS_FORMAT);
        let mut rgba = [128u8, 128, 128, 255].repeat(size * size);
        for y in 0..32 {
            for x in 0..16 {
                rgba[(y * size + x) * 4..][..4].copy_from_slice(&[255, 0, 0, 255]);
            }
        }
        blur.load_rgba(&gpu, &rgba);
        let out = run(&blur, &gpu, 1);
        let red = |x: usize, y: usize| {
            let p = &out[(y * size + x) * 4..][..4];
            p[0] > 200 && p[1] < 60
        };
        assert!(red(48, 56), "the block's mirror image, bottom-right");
        assert!(
            red(34, 49) && red(62, 62),
            "the mirrored block spans x 32..64, y 48..64"
        );
        assert!(!red(8, 16), "the block must leave the top-left corner");
        assert!(
            !red(16, 56),
            "a quarter turn would have put the block bottom-left"
        );
        assert!(!red(20, 40), "nothing else turns red");
    }

    #[test]
    fn consecutive_passes_keep_changing_the_image_rather_than_reaching_a_fixed_point_at_once() {
        let Some(gpu) = headless(4, 4) else { return };
        let size = 64;
        let blur = seeded_blur(&gpu, size);
        let after_one = run(&blur, &gpu, 1);
        // `blur`'s accumulator now holds `after_one`'s content; one more pass must move it again.
        let after_two = {
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            blur.blur_pass(&mut enc);
            gpu.queue.submit(Some(enc.finish()));
            blur.read_rgba(&gpu)
        };
        assert_ne!(
            after_one, after_two,
            "a second pass reading the first pass's own output must accumulate further, not no-op"
        );
    }

    #[test]
    fn seven_passes_spread_the_seeded_texel_further_than_one_pass_does() {
        // A monotonic-spread proof that this is genuinely accumulating across `PASSES`, not just
        // producing *different* bytes each time (which a broken shader touching a PRNG could
        // also do). The seeded texel's peak must fall as more of its energy spreads to
        // neighbours, and it must fall further after seven passes than after one. The peak is
        // tracked wherever it is: each pass mirrors the image across its diagonal, so the texel
        // moves between (64, 64) and (63, 63) instead of staying on the pixel it was seeded on.
        let Some(gpu) = headless(4, 4) else { return };
        let size = 128;
        // How much redder than green the reddest pixel is; the grey background scores 0.
        let peak = |pixels: &[u8]| -> i32 {
            pixels
                .chunks_exact(4)
                .map(|p| i32::from(p[0]) - i32::from(p[1]))
                .max()
                .expect("a non-empty image")
        };

        let one_pass = seeded_blur(&gpu, size);
        let seed_peak = peak(&one_pass.read_rgba(&gpu));
        let after_one = run(&one_pass, &gpu, 1);

        let seven_pass = seeded_blur(&gpu, size);
        let after_seven = run(&seven_pass, &gpu, PanoramaBlur::PASSES);

        let p1 = peak(&after_one);
        let p7 = peak(&after_seven);
        assert!(
            p1 < seed_peak,
            "the seeded texel's peak must drop as it blurs, got {p1} from {seed_peak}"
        );
        assert!(
            p7 < p1,
            "seven passes must spread the seeded texel further than one pass: {p7} vs {p1}"
        );
    }

    #[test]
    fn a_flat_field_is_unchanged_by_blurring_which_only_a_correct_average_guarantees() {
        // Averaging three samples of a perfectly uniform field must reproduce that same uniform
        // value exactly (mod float rounding) — a shader that got the weights wrong (summed to
        // something other than 1, or read outside the intended taps) would drift a flat field
        // away from its starting colour.
        let Some(gpu) = headless(4, 4) else { return };
        let size = 32;
        let blur = PanoramaBlur::new(&gpu, size, size, HEADLESS_FORMAT);
        let rgba = [90u8, 60, 30, 255].repeat((size * size) as usize);
        blur.load_rgba(&gpu, &rgba);
        let blurred = run(&blur, &gpu, PanoramaBlur::PASSES);
        for p in blurred.chunks_exact(4) {
            assert!(p[0].abs_diff(90) <= 2, "{p:?}");
            assert!(p[1].abs_diff(60) <= 2, "{p:?}");
            assert!(p[2].abs_diff(30) <= 2, "{p:?}");
        }
    }

    /// The algebra from the module docs, checked independently on the CPU: sequentially applying
    /// "source over" blending with alpha `1, 1/2, 1/3` (vanilla's literal three draws) against
    /// three arbitrary samples must equal their plain average, for *any* starting destination
    /// colour — proving `colorMask`'s "don't clear first" truly is irrelevant, not just for the
    /// specific seed this module's GPU tests happen to use.
    #[test]
    fn the_three_tap_cascade_is_exactly_the_alpha_blend_math_it_replaces() {
        fn blend_over(src: f32, dst: f32, alpha: f32) -> f32 {
            src * alpha + dst * (1.0 - alpha)
        }
        let taps = [0.9_f32, 0.4, 0.65];
        for start in [0.0_f32, 0.5, 1.0, 0.123] {
            let mut acc = start;
            for (j, &t) in taps.iter().enumerate() {
                acc = blend_over(t, acc, 1.0 / (j as f32 + 1.0));
            }
            let average = taps.iter().sum::<f32>() / taps.len() as f32;
            assert!(
                (acc - average).abs() < 1e-5,
                "cascade({start}) = {acc}, expected average {average}"
            );
        }
    }

    #[test]
    fn vanilla_size_and_pass_count_match_the_decompiled_source() {
        assert_eq!(PanoramaBlur::VANILLA_SIZE, 256);
        assert_eq!(PanoramaBlur::PASSES, 7);
    }

    #[test]
    fn load_rgba_rejects_a_mis_sized_buffer() {
        let Some(gpu) = headless(4, 4) else { return };
        let blur = PanoramaBlur::new(&gpu, 8, 8, HEADLESS_FORMAT);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            blur.load_rgba(&gpu, &[0u8; 4]);
        }));
        assert!(
            result.is_err(),
            "a short seed buffer must panic, not silently misinterpret bytes"
        );
    }
}
