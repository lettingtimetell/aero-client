//! ASCII: the Super Secret Settings entry that redraws the world as Minecraft-font characters.
//!
//! The client glue is `mc-client/src/app/mods/nw_rfx.rs`; the option lives in the
//! `super_secret_settings` mod (its "Shader" choice "ASCII", with "Character size" and
//! "Character colors").
//!
//! # What it draws
//!
//! The frame is cut into square cells. Each cell becomes one character from a brightness ramp:
//! dark cells get sparse glyphs (`.`, `:`), bright ones dense glyphs (`#`, `@`). The glyphs are the
//! active resource pack's own font, `textures/font/ascii.png`, so the effect has the Minecraft font
//! look and follows a pack's font. [`glyph_ramp`] measures every printable ASCII glyph's ink and
//! picks up to [`MAX_GLYPHS`] with evenly spaced coverage. Without a font (a broken pack) a small
//! built-in ramp of dots and blocks is used ([`fallback_ramp`]).
//!
//! Cells are whole multiples of the font's 8-pixel cell, so every font pixel covers the same
//! number of screen pixels and the glyphs stay crisp ([`cell_size`]). The three sizes keep a
//! similar number of rows at any resolution (about 135, 68 and 45 rows at 1080p).
//!
//! Colours ([`AsciiColors`]): the cell's own colour (brightened so a sparse glyph keeps its hue,
//! over a faint wash of the same colour), white on black, or green phosphor on black.
//!
//! # Where it records
//!
//! [`AsciiPhase::ORDER`] is `POST + 60` (760): after every phase at [`PhaseOrder::POST`]
//! (overlays, vanilla's Super Secret chain, motion blur and grading, the shader packs' final
//! passes), before the title screen's background (790) and the UI (800). The world and the hand
//! become characters; the HUD, chat and menus stay sharp, as with vanilla's Super Secret shaders.
//!
//! # Cost
//!
//! Two fullscreen passes: a cells pass that writes one texel per character cell (at most 8x8
//! loads per cell, so about one load per frame pixel in total), and a composite that does one
//! cell load and one glyph load per frame pixel.
//!
//! **Off costs nothing:** while inactive the phase reports `enabled() == false` and
//! `needs_scene_color() == false`, so [`crate::PhaseList`] never records it and keeps its direct
//! frame path, and it owns no GPU object. Pipelines, the glyph ramp, the cells target and the
//! uniform buffer are built on the first active frame; [`AsciiPhase::set_active`]`(false)` drops
//! the per-size targets and [`AsciiPhase::release`] drops everything (the client calls it when the
//! effect is switched off). `tests/nw_rfx_ascii.rs` proves both on the GPU.
//!
//! # MSAA
//!
//! The composite draws through [`Gpu::begin_pass`] with a pipeline of the context's sample count,
//! so under MSAA it writes the multisampled attachment and survives the resolve, like the menu
//! blur. The cells pass reads the single-sampled scene snapshot.

use crate::{Camera, DepthAttachment, Gpu, PhaseOrder, RenderPhase, SceneTextures};

/// The cells target's format. Sixteen-bit float keeps the cell averages exact enough that the
/// glyph choice does not band.
pub const CELLS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The most glyphs a ramp holds.
pub const MAX_GLYPHS: usize = 16;

/// The printable ASCII glyphs the ramp may pick from, in the classic dark-to-light art ramp's
/// order (its 70 levels, reversed). The order only breaks ties between glyphs with the
/// same ink; the ink itself is measured from the font.
pub const CANDIDATES: &str =
    " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";

/// The edge glyphs that follow the brightness ramp, in this order: a vertical, a horizontal and
/// the two diagonal edges. Where brightness changes sharply between neighbouring cells the
/// composite draws one of these instead of a brightness glyph, so shapes keep their outlines.
pub const EDGE_GLYPHS: [char; 4] = ['|', '-', '/', '\\'];

/// Bytes of the uniform: `Ascii` in `ascii.wgsl`, three `vec4<f32>`.
const UNIFORM_SIZE: u64 = 48;

/// The most samples the cells pass takes along each side of a cell.
const MAX_CELL_SAMPLES: u32 = 8;

/// How big the characters are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AsciiSize {
    /// About 135 rows at 1080p (8-pixel cells).
    Small,
    /// About 68 rows at 1080p (16-pixel cells). The mod's default.
    #[default]
    Medium,
    /// About 45 rows at 1080p (24-pixel cells).
    Large,
}

impl AsciiSize {
    /// The mod's "Character size" choice index: 0 Small, 1 Medium, 2 Large (anything else is
    /// Medium).
    #[must_use]
    pub fn from_index(index: usize) -> AsciiSize {
        match index {
            0 => AsciiSize::Small,
            2 => AsciiSize::Large,
            _ => AsciiSize::Medium,
        }
    }
}

/// How the characters are coloured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AsciiColors {
    /// Each character in its cell's colour, over a faint wash of it. The mod's default.
    #[default]
    Color,
    /// White characters on black.
    White,
    /// Green phosphor characters on black.
    Green,
}

impl AsciiColors {
    /// The mod's "Character colors" choice index: 0 Color, 1 White, 2 Green (anything else is
    /// Color).
    #[must_use]
    pub fn from_index(index: usize) -> AsciiColors {
        match index {
            1 => AsciiColors::White,
            2 => AsciiColors::Green,
            _ => AsciiColors::Color,
        }
    }

    fn shader_mode(self) -> f32 {
        match self {
            AsciiColors::Color => 0.0,
            AsciiColors::White => 1.0,
            AsciiColors::Green => 2.0,
        }
    }
}

/// A character cell's side in pixels on a frame `height` pixels tall.
///
/// Always a whole multiple of the font's 8-pixel glyph cell, so the glyphs stay crisp. The
/// multiple follows the frame height (1, 2 and 3 at 1080p for Small, Medium and Large) and the
/// three sizes always differ: at 720p they are 8, 16 and 24 pixels, at 1440p 8, 24 and 32, at 4K
/// 16, 32 and 48.
#[must_use]
pub fn cell_size(size: AsciiSize, height: u32) -> u32 {
    let scale = height.max(1) as f32 / 1080.0;
    let multiple = |m: f32| (scale * m).round().max(1.0) as u32;
    let small = multiple(1.0);
    let medium = multiple(2.0).max(small + 1);
    let large = multiple(3.0).max(medium + 1);
    8 * match size {
        AsciiSize::Small => small,
        AsciiSize::Medium => medium,
        AsciiSize::Large => large,
    }
}

/// A brightness ramp of glyphs, ready to upload: `glyphs.len()` glyphs of `glyph_width` x
/// `glyph_height` texels side by side in `coverage` (one byte per texel, 0 = paper, 255 = ink),
/// followed by the four [`EDGE_GLYPHS`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlyphRamp {
    /// The brightness characters, darkest (always a blank one) first.
    pub glyphs: Vec<char>,
    /// Ink texels of each brightness glyph, in the same order.
    pub ink: Vec<u32>,
    /// One glyph's width in texels (the font's cell width).
    pub glyph_width: u32,
    /// One glyph's height in texels (the font's cell height).
    pub glyph_height: u32,
    /// Whether the four edge glyphs have ink (a font without them draws no edges).
    pub edges: bool,
    /// [`Self::width`] by `glyph_height` coverage bytes, row-major.
    pub coverage: Vec<u8>,
}

impl GlyphRamp {
    /// The ramp texture's width in texels: the brightness glyphs, then the edge glyphs.
    #[must_use]
    pub fn width(&self) -> u32 {
        (self.glyphs.len() + EDGE_GLYPHS.len()) as u32 * self.glyph_width
    }
}

/// Build the brightness ramp from a Minecraft font sheet (`textures/font/ascii.png`: 16 x 16
/// cells, white glyphs on transparent texels), or `None` when the image is not a font sheet.
///
/// Every candidate glyph in [`CANDIDATES`] is measured (texels with alpha above half count as
/// ink). Glyphs with the same ink keep the first in [`CANDIDATES`]; then up to [`MAX_GLYPHS`]
/// distinct ink levels are picked, nearest to evenly spaced targets from none to the densest
/// glyph, always strictly increasing. Each glyph is centred horizontally in its slot, so the
/// characters sit in the middle of their cells like a monospace font (Minecraft's own glyphs are
/// left-aligned and of varying width).
#[must_use]
pub fn glyph_ramp(rgba: &[u8], width: u32, height: u32) -> Option<GlyphRamp> {
    if width < 16 || height < 16 || !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return None;
    }
    if rgba.len() < (width as usize) * (height as usize) * 4 {
        return None;
    }
    let (cw, ch) = (width / 16, height / 16);
    let alpha = |x: u32, y: u32| rgba[((y * width + x) * 4 + 3) as usize];
    // (ink, first ink column, last ink column) of one glyph.
    let measure = |c: char| -> (u32, u32, u32) {
        let code = c as u32;
        let (ox, oy) = ((code % 16) * cw, (code / 16) * ch);
        let (mut ink, mut first, mut last) = (0u32, cw, 0u32);
        for y in 0..ch {
            for x in 0..cw {
                if alpha(ox + x, oy + y) > 127 {
                    ink += 1;
                    first = first.min(x);
                    last = last.max(x);
                }
            }
        }
        (ink, first, last)
    };
    // One glyph per distinct ink count: the first candidate that has it.
    let mut levels: Vec<(u32, char)> = Vec::new();
    for c in CANDIDATES.chars() {
        let (ink, _, _) = measure(c);
        if !levels.iter().any(|&(seen, _)| seen == ink) {
            levels.push((ink, c));
        }
    }
    levels.sort_by_key(|&(ink, _)| ink);
    if levels.first().is_none_or(|&(ink, _)| ink != 0) {
        // Some glyph (the space, normally) must be blank: a sheet without one is not a font.
        return None;
    }
    let densest = levels.last().map_or(0, |&(ink, _)| ink);
    if densest == 0 {
        return None;
    }
    let mut picked: Vec<(u32, char)> = vec![levels[0]];
    for step in 1..MAX_GLYPHS {
        let target = densest as f32 * step as f32 / (MAX_GLYPHS - 1) as f32;
        let floor = picked.last().map_or(0, |&(ink, _)| ink);
        let best = levels
            .iter()
            .filter(|&&(ink, _)| ink > floor)
            .min_by(|a, b| {
                let da = (a.0 as f32 - target).abs();
                let db = (b.0 as f32 - target).abs();
                da.total_cmp(&db)
            })
            .copied();
        match best {
            Some(level) => picked.push(level),
            None => break,
        }
    }
    let count = picked.len() as u32;
    let ramp_width = (count + EDGE_GLYPHS.len() as u32) * cw;
    let mut coverage = vec![0u8; (ramp_width * ch) as usize];
    // Copy glyph `c` into `slot`, its ink columns centred in the slot.
    let mut blit = |slot: u32, c: char| {
        let (ink, first, last) = measure(c);
        if ink == 0 {
            return false;
        }
        let code = c as u32;
        let (ox, oy) = ((code % 16) * cw, (code / 16) * ch);
        let span = last + 1 - first;
        let shift = (cw - span) / 2;
        for y in 0..ch {
            for x in first..=last {
                let a = alpha(ox + x, oy + y);
                if a > 0 {
                    let dx = slot * cw + (x - first) + shift;
                    coverage[(y * ramp_width + dx) as usize] = a;
                }
            }
        }
        true
    };
    for (slot, &(_, c)) in picked.iter().enumerate() {
        blit(slot as u32, c);
    }
    let mut edges = true;
    for (i, &c) in EDGE_GLYPHS.iter().enumerate() {
        edges &= blit(count + i as u32, c);
    }
    Some(GlyphRamp {
        glyphs: picked.iter().map(|&(_, c)| c).collect(),
        ink: picked.iter().map(|&(ink, _)| ink).collect(),
        glyph_width: cw,
        glyph_height: ch,
        edges,
        coverage,
    })
}

/// A built-in ramp for when the font cannot be read: eight 8x8 glyphs from a dot to a block,
/// then lines for the four edge glyphs.
#[must_use]
pub fn fallback_ramp() -> GlyphRamp {
    // Sides of a centred square, in texels (0 = empty).
    const SIZES: [u32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    let (gw, gh) = (8u32, 8u32);
    let ramp_width = (SIZES.len() + EDGE_GLYPHS.len()) as u32 * gw;
    let mut coverage = vec![0u8; (ramp_width * gh) as usize];
    let mut ink = Vec::with_capacity(SIZES.len());
    for (slot, &side) in SIZES.iter().enumerate() {
        let lo = (gw - side) / 2;
        let mut count = 0;
        for y in lo..lo + side {
            for x in lo..lo + side {
                coverage[(y * ramp_width + slot as u32 * gw + x) as usize] = 255;
                count += 1;
            }
        }
        ink.push(count);
    }
    // `|`, `-`, `/`, `\` as one-texel lines.
    let edge_slot = SIZES.len() as u32;
    for i in 0..8u32 {
        let lines = [(3, i), (i, 3), (7 - i, i), (i, i)];
        for (k, &(x, y)) in lines.iter().enumerate() {
            let slot = edge_slot + k as u32;
            coverage[(y * ramp_width + slot * gw + x) as usize] = 255;
        }
    }
    GlyphRamp {
        glyphs: vec![' ', '.', ':', '+', 'o', 'O', '#', '@'],
        ink,
        glyph_width: gw,
        glyph_height: gh,
        edges: true,
        coverage,
    }
}

/// The phase's pipelines and layouts, built once for the frame's colour format and sample count.
struct Pipelines {
    scene_format: wgpu::TextureFormat,
    samples: u32,
    cells_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    cells: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
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

impl Pipelines {
    fn new(gpu: &Gpu, scene_format: wgpu::TextureFormat) -> Pipelines {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post/ascii"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ascii.wgsl").into()),
        });
        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(UNIFORM_SIZE),
            },
            count: None,
        };
        let cells_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post/ascii cells"),
            entries: &[uniform_entry, texture_entry(1)],
        });
        let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post/ascii composite"),
            entries: &[uniform_entry, texture_entry(2), texture_entry(3)],
        });
        let samples = gpu.sample_count();
        let cells = pipeline(gpu, &shader, &cells_layout, "fs_cells", CELLS_FORMAT, 1);
        let composite = pipeline(
            gpu,
            &shader,
            &composite_layout,
            "fs_ascii",
            scene_format,
            samples,
        );
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post/ascii uniform"),
            size: UNIFORM_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Pipelines {
            scene_format,
            samples,
            cells_layout,
            composite_layout,
            cells,
            composite,
            uniform,
        }
    }
}

/// The glyph ramp on the GPU.
struct Ramp {
    view: wgpu::TextureView,
    glyphs: u32,
    glyph_width: u32,
    glyph_height: u32,
    edges: bool,
}

impl Ramp {
    fn upload(gpu: &Gpu, ramp: &GlyphRamp) -> Ramp {
        let size = wgpu::Extent3d {
            width: ramp.width().max(1),
            height: ramp.glyph_height.max(1),
            depth_or_array_layers: 1,
        };
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("post/ascii glyphs"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
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
            &ramp.coverage,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ramp.width()),
                rows_per_image: Some(ramp.glyph_height),
            },
            size,
        );
        Ramp {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            glyphs: ramp.glyphs.len() as u32,
            glyph_width: ramp.glyph_width,
            glyph_height: ramp.glyph_height,
            edges: ramp.edges,
        }
    }
}

/// The cells target for one frame size and cell size, and the composite's bind group.
struct CellTarget {
    width: u32,
    height: u32,
    cell: u32,
    columns: u32,
    rows: u32,
    view: wgpu::TextureView,
    composite_group: wgpu::BindGroup,
}

impl CellTarget {
    fn new(
        gpu: &Gpu,
        pipelines: &Pipelines,
        ramp: &Ramp,
        (width, height): (u32, u32),
        cell: u32,
    ) -> CellTarget {
        let columns = width.div_ceil(cell).max(1);
        let rows = height.div_ceil(cell).max(1);
        let view = gpu
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("post/ascii cells"),
                size: wgpu::Extent3d {
                    width: columns,
                    height: rows,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: CELLS_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        let composite_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post/ascii composite"),
            layout: &pipelines.composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pipelines.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&ramp.view),
                },
            ],
        });
        CellTarget {
            width,
            height,
            cell,
            columns,
            rows,
            view,
            composite_group,
        }
    }
}

/// The cells pass's bind group over the scene snapshot. `PhaseList` rebuilds its snapshot only on
/// a resize or a format change, so this is rebuilt then, not per frame.
struct SceneBinding {
    /// A clone of the snapshot view, compared by identity every frame; holding it keeps the view
    /// alive, so a new view can never reuse its address while it is compared against.
    view: wgpu::TextureView,
    group: wgpu::BindGroup,
}

/// Everything the uniform follows from, so an unchanged frame uploads nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UniformKey {
    width: u32,
    height: u32,
    srgb: bool,
    cell: u32,
    glyphs: u32,
    glyph_width: u32,
    glyph_height: u32,
    edges: bool,
    colors: AsciiColors,
}

/// The ASCII effect. See the [module docs](self).
pub struct AsciiPhase {
    active: bool,
    size: AsciiSize,
    colors: AsciiColors,
    /// The font the next ramp is built from: `Some(None)` means "use the fallback", `None` that
    /// the client has not supplied one since the ramp was last dropped.
    font: Option<Option<GlyphRamp>>,
    ramp: Option<Ramp>,
    pipelines: Option<Pipelines>,
    target: Option<CellTarget>,
    scene_binding: Option<SceneBinding>,
    uploaded: Option<UniformKey>,
    passes: u64,
}

impl std::fmt::Debug for AsciiPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsciiPhase")
            .field("active", &self.active)
            .field("size", &self.size)
            .field("colors", &self.colors)
            .field("pipelines", &self.pipelines.is_some())
            .field("ramp", &self.ramp.as_ref().map(|ramp| ramp.glyphs))
            .field("passes", &self.passes)
            .finish_non_exhaustive()
    }
}

impl Default for AsciiPhase {
    fn default() -> Self {
        Self::new()
    }
}

impl AsciiPhase {
    /// The phase's name in `PhaseList`, GPU timings and captures.
    pub const NAME: &'static str = "post/ascii";

    /// After every [`PhaseOrder::POST`] phase, before the title background (790) and the UI.
    pub const ORDER: PhaseOrder = PhaseOrder(PhaseOrder::POST.0 + 60);

    /// An idle phase. Allocates nothing on the GPU; that happens on its first active frame.
    #[must_use]
    pub fn new() -> AsciiPhase {
        AsciiPhase {
            active: false,
            size: AsciiSize::default(),
            colors: AsciiColors::default(),
            font: None,
            ramp: None,
            pipelines: None,
            target: None,
            scene_binding: None,
            uploaded: None,
            passes: 0,
        }
    }

    /// Whether the effect runs. Turning it off drops the cells target at once (the pipelines and
    /// the glyph ramp stay until [`Self::release`]).
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

    /// Set the character size and colours.
    pub fn set_style(&mut self, size: AsciiSize, colors: AsciiColors) {
        self.size = size;
        self.colors = colors;
    }

    /// The character size and colours.
    #[must_use]
    pub fn style(&self) -> (AsciiSize, AsciiColors) {
        (self.size, self.colors)
    }

    /// Whether the phase wants the font sheet: true from creation, and again after
    /// [`Self::release`], until [`Self::set_font`] supplies it.
    #[must_use]
    pub fn needs_font(&self) -> bool {
        self.font.is_none() && self.ramp.is_none()
    }

    /// Supply the font sheet (`textures/font/ascii.png` as RGBA8), or `None` when it could not be
    /// read: the built-in ramp is used then. The ramp is rebuilt on the next active frame.
    pub fn set_font(&mut self, rgba: Option<(&[u8], u32, u32)>) {
        let ramp = rgba.and_then(|(rgba, width, height)| glyph_ramp(rgba, width, height));
        self.font = Some(ramp);
        self.ramp = None;
        self.target = None;
        self.uploaded = None;
    }

    /// Drop every GPU resource, pipelines and the glyph ramp included, and go idle. The client
    /// calls this when the effect is switched off; the next active frame rebuilds what it needs
    /// (after the client supplies the font again, see [`Self::needs_font`]).
    pub fn release(&mut self) {
        self.active = false;
        self.drop_targets();
        self.pipelines = None;
        self.ramp = None;
        self.font = None;
    }

    /// Whether the cells target exists (only while the effect is on).
    #[must_use]
    pub fn targets_allocated(&self) -> bool {
        self.target.is_some()
    }

    /// Whether the pipelines exist (from the first active frame until [`Self::release`]).
    #[must_use]
    pub fn pipelines_built(&self) -> bool {
        self.pipelines.is_some()
    }

    /// Whether the glyph ramp is on the GPU.
    #[must_use]
    pub fn ramp_built(&self) -> bool {
        self.ramp.is_some()
    }

    /// How many frames this phase has drawn. Stays put while it is inactive.
    #[must_use]
    pub fn passes_recorded(&self) -> u64 {
        self.passes
    }

    /// The cells target's size in characters (columns, rows) and the cell side in pixels, once a
    /// frame has been drawn.
    #[must_use]
    pub fn grid(&self) -> Option<(u32, u32, u32)> {
        self.target
            .as_ref()
            .map(|target| (target.columns, target.rows, target.cell))
    }

    fn drop_targets(&mut self) {
        self.target = None;
        self.scene_binding = None;
        self.uploaded = None;
    }

    /// Build or rebuild what this frame needs.
    fn prepare(&mut self, gpu: &Gpu, scene: &SceneTextures<'_>) {
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
        if self.ramp.is_none() {
            let ramp = match self.font.take() {
                Some(Some(ramp)) => ramp,
                _ => fallback_ramp(),
            };
            self.ramp = Some(Ramp::upload(gpu, &ramp));
            self.target = None;
            self.uploaded = None;
        }
        let ramp = self.ramp.as_ref().expect("built above");
        let (width, height) = (scene.width.max(1), scene.height.max(1));
        let cell = cell_size(self.size, height);
        if self
            .target
            .as_ref()
            .is_none_or(|t| (t.width, t.height, t.cell) != (width, height, cell))
        {
            self.target = Some(CellTarget::new(gpu, pipelines, ramp, (width, height), cell));
            self.uploaded = None;
        }
        if self
            .scene_binding
            .as_ref()
            .is_none_or(|binding| binding.view != *scene.color)
        {
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post/ascii cells"),
                layout: &pipelines.cells_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pipelines.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(scene.color),
                    },
                ],
            });
            self.scene_binding = Some(SceneBinding {
                view: scene.color.clone(),
                group,
            });
        }
    }

    /// Write the uniform when anything it follows from changed.
    fn upload(&mut self, gpu: &Gpu, format: wgpu::TextureFormat) {
        let (Some(pipelines), Some(target), Some(ramp)) = (
            self.pipelines.as_ref(),
            self.target.as_ref(),
            self.ramp.as_ref(),
        ) else {
            return;
        };
        let key = UniformKey {
            width: target.width,
            height: target.height,
            srgb: format.is_srgb(),
            cell: target.cell,
            glyphs: ramp.glyphs,
            glyph_width: ramp.glyph_width,
            glyph_height: ramp.glyph_height,
            edges: ramp.edges,
            colors: self.colors,
        };
        if self.uploaded == Some(key) {
            return;
        }
        let step = target.cell.div_ceil(MAX_CELL_SAMPLES).max(1);
        let values: [f32; 12] = [
            target.width as f32,
            target.height as f32,
            if key.srgb { 1.0 } else { 0.0 },
            0.0,
            target.cell as f32,
            ramp.glyphs as f32,
            ramp.glyph_width as f32,
            ramp.glyph_height as f32,
            self.colors.shader_mode(),
            step as f32,
            if ramp.edges { 1.0 } else { 0.0 },
            0.0,
        ];
        let mut bytes = [0u8; UNIFORM_SIZE as usize];
        for (i, value) in values.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        gpu.write_buffer_staged(&pipelines.uniform, 0, &bytes);
        self.uploaded = Some(key);
    }
}

impl RenderPhase for AsciiPhase {
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
        // A new size means a new snapshot and a cells target of the wrong size.
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
        self.prepare(gpu, scene);
        self.upload(gpu, scene.format);
        let (Some(pipelines), Some(target), Some(binding)) = (
            self.pipelines.as_ref(),
            self.target.as_ref(),
            self.scene_binding.as_ref(),
        ) else {
            return;
        };
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post/ascii cells"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
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
            });
            pass.set_pipeline(&pipelines.cells);
            pass.set_bind_group(0, &binding.group, &[]);
            pass.draw(0..3, 0..1);
        }
        let mut pass = gpu.begin_pass(enc, view, Self::NAME, DepthAttachment::None);
        pass.set_pipeline(&pipelines.composite);
        pass.set_bind_group(0, &target.composite_group, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        self.passes += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 128x128 font sheet whose glyph `c` has `ink(c)` texels in its first rows.
    fn synthetic_font(ink: impl Fn(char) -> u32) -> Vec<u8> {
        let mut rgba = vec![0u8; 128 * 128 * 4];
        for code in 33u32..127 {
            let c = char::from_u32(code).unwrap();
            let count = ink(c).min(64);
            let (ox, oy) = ((code % 16) * 8, (code / 16) * 8);
            for i in 0..count {
                // Fill columns 0..5 row by row, then the rest, like a left-aligned glyph.
                let (x, y) = (i % 8, i / 8);
                let at = (((oy + y) * 128 + ox + x) * 4) as usize;
                rgba[at..at + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        rgba
    }

    #[test]
    fn the_three_sizes_are_whole_font_pixels_and_always_differ() {
        for height in [240, 480, 720, 768, 900, 1080, 1440, 2160, 4320] {
            let small = cell_size(AsciiSize::Small, height);
            let medium = cell_size(AsciiSize::Medium, height);
            let large = cell_size(AsciiSize::Large, height);
            assert!(
                small.is_multiple_of(8) && medium.is_multiple_of(8) && large.is_multiple_of(8),
                "{height}"
            );
            assert!(
                small < medium && medium < large,
                "{height}: {small} {medium} {large}"
            );
        }
        assert_eq!(
            [
                cell_size(AsciiSize::Small, 1080),
                cell_size(AsciiSize::Medium, 1080),
                cell_size(AsciiSize::Large, 1080),
            ],
            [8, 16, 24]
        );
        assert_eq!(cell_size(AsciiSize::Medium, 720), 16);
        assert_eq!(cell_size(AsciiSize::Large, 2160), 48);
        assert_eq!(cell_size(AsciiSize::Small, 0), 8);
    }

    #[test]
    fn the_ramp_starts_with_a_space_and_its_ink_strictly_increases() {
        // Ink = position in the candidate list, so every candidate has its own level.
        let rgba = synthetic_font(|c| CANDIDATES.find(c).unwrap_or(0) as u32);
        let ramp = glyph_ramp(&rgba, 128, 128).expect("a font sheet");
        assert_eq!(ramp.glyphs[0], ' ');
        assert_eq!(ramp.ink[0], 0);
        assert_eq!(ramp.glyphs.len(), MAX_GLYPHS);
        assert!(ramp.ink.windows(2).all(|w| w[0] < w[1]), "{:?}", ramp.ink);
        // The densest glyph (a full 8x8 cell) ends the brightness glyphs; the edge glyphs follow.
        assert_eq!(*ramp.ink.last().unwrap(), 64);
        assert_eq!(ramp.width(), (MAX_GLYPHS + EDGE_GLYPHS.len()) as u32 * 8);
        assert!(ramp.edges, "every edge glyph has ink in this sheet");
        // Every slot holds exactly its glyph's ink.
        for (slot, &ink) in ramp.ink.iter().enumerate() {
            let mut count = 0;
            for y in 0..8 {
                for x in 0..8 {
                    if ramp.coverage[y * ramp.width() as usize + slot * 8 + x] > 0 {
                        count += 1;
                    }
                }
            }
            assert_eq!(count, ink, "slot {slot}");
        }
    }

    #[test]
    fn narrow_glyphs_are_centred_in_their_slot() {
        // Every glyph is one column of `n` texels at x = 0: a 1-wide glyph moves to column 3.
        let mut font = vec![0u8; 128 * 128 * 4];
        for code in 33u32..127 {
            let c = char::from_u32(code).unwrap();
            let n = (CANDIDATES.find(c).unwrap_or(0) % 8) as u32;
            let (ox, oy) = ((code % 16) * 8, (code / 16) * 8);
            for y in 0..n {
                let at = (((oy + y) * 128 + ox) * 4) as usize;
                font[at..at + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        let ramp = glyph_ramp(&font, 128, 128).expect("a font sheet");
        for slot in 1..ramp.glyphs.len() {
            for y in 0..8usize {
                for x in 0..8usize {
                    let ink = ramp.coverage[y * ramp.width() as usize + slot * 8 + x] > 0;
                    if ink {
                        assert_eq!(x, 3, "slot {slot}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_sheet_that_is_not_a_font_is_refused() {
        assert!(glyph_ramp(&[], 0, 0).is_none());
        assert!(glyph_ramp(&vec![0; 100 * 100 * 4], 100, 100).is_none());
        // Blank: no glyph has ink.
        assert!(glyph_ramp(&vec![0; 128 * 128 * 4], 128, 128).is_none());
        // A space with ink is not a font.
        let mut rgba = synthetic_font(|_| 5);
        // ' ' is code 32: the first cell of the third row, whose top-left texel is (0, 16).
        let space = 16 * 128 * 4;
        rgba[space + 3] = 255;
        assert!(glyph_ramp(&rgba, 128, 128).is_none());
        // Too short a buffer.
        assert!(glyph_ramp(&[0; 16], 128, 128).is_none());
    }

    #[test]
    fn the_fallback_ramp_grows_from_empty_to_a_block() {
        let ramp = fallback_ramp();
        assert_eq!(ramp.glyphs[0], ' ');
        assert!(ramp.ink.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(ramp.coverage.len(), (ramp.width() * 8) as usize);
        // The `|` slot is a vertical line in column 3.
        let bar = ramp.glyphs.len() * 8;
        for y in 0..8usize {
            for x in 0..8usize {
                let ink = ramp.coverage[y * ramp.width() as usize + bar + x] > 0;
                assert_eq!(ink, x == 3, "({x},{y})");
            }
        }
    }

    #[test]
    fn a_font_without_edge_glyphs_draws_no_edges() {
        // Ink only on the brightness candidates that are letters or digits.
        let rgba = synthetic_font(|c| {
            if c.is_ascii_alphanumeric() {
                CANDIDATES.find(c).unwrap_or(0) as u32
            } else {
                0
            }
        });
        let ramp = glyph_ramp(&rgba, 128, 128).expect("a font sheet");
        assert!(!ramp.edges);
    }

    #[test]
    fn an_idle_phase_opts_out_and_owns_nothing() {
        let mut phase = AsciiPhase::new();
        assert!(!phase.enabled());
        assert!(!phase.needs_scene_color());
        assert!(!phase.targets_allocated() && !phase.pipelines_built() && !phase.ramp_built());
        assert!(phase.needs_font());
        phase.set_active(true);
        assert!(phase.enabled() && phase.needs_scene_color());
        // Nothing is built until a frame records it.
        assert!(!phase.targets_allocated() && !phase.pipelines_built());
        phase.set_font(None);
        assert!(!phase.needs_font());
        phase.set_active(false);
        assert!(!phase.enabled() && !phase.needs_scene_color());
        phase.release();
        assert!(phase.needs_font());
        assert_eq!(AsciiSize::from_index(0), AsciiSize::Small);
        assert_eq!(AsciiSize::from_index(9), AsciiSize::Medium);
        assert_eq!(AsciiColors::from_index(2), AsciiColors::Green);
        assert_eq!(AsciiColors::from_index(7), AsciiColors::Color);
        assert!(AsciiPhase::ORDER > PhaseOrder::POST && AsciiPhase::ORDER < PhaseOrder::UI);
    }
}
