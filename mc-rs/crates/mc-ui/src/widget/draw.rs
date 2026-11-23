//! The immediate-mode draw list: vanilla's `Gui` helpers, plus nine-slice.
//!
//! java: `client/gui/Gui.java`
//!
//! A [`DrawList`] is a flat triangle stream batched by texture and fixed-function state, in
//! submission order. Nothing here talks to wgpu — [`crate::widget::UiPhase`] does that — so a
//! screen's whole draw can be built, inspected and asserted on with no GPU at all. Ordinary
//! widgets retain their 2-D vertex bytes; item/entity windows additionally carry parallel depth
//! values and explicit cull/depth state.
//!
//! # Coordinates
//!
//! Everything is in **scaled GUI pixels**: `(0, 0)` is the top-left of the screen and
//! `(scaled_width, scaled_height)` the bottom-right (see [`crate::widget::ScaledResolution`]).
//! Vanilla's GUI textures are 256×256 and are addressed in texels, which is what
//! [`DrawList::blit`] takes.
//!
//! # Colours
//!
//! Packed `0xAARRGGBB`, vanilla's convention. Note vanilla's own quirk that alpha 0 in a
//! *text* colour means opaque (see [`crate::font::FontRenderer::draw_string`]) but alpha 0 in
//! a *rect* colour means transparent; nothing is fixed up here.

use std::ops::Range;

use mc_assets::{ResourceLocation, Sprite};
use mc_render::entity::tilepass::{BuiltinTileItem, TileModels};

use crate::font::{FontTexture, GlyphQuad, SolidRect, TextSink};
use crate::widget::transform::Transform;

/// Vanilla's GUI textures are all 256×256, and `drawTexturedModalRect` hard-codes `1/256`.
// java: Gui.java:140 (`float f = 0.00390625F`)
pub const GUI_TEXTURE_SIZE: f32 = 256.0;

/// `textures/gui/widgets.png` — buttons, the hotbar, the slider handle.
pub const WIDGETS_TEXTURE: &str = "textures/gui/widgets.png";
/// `textures/gui/options_background.png` — the tiled dirt behind menus.
pub const OPTIONS_BACKGROUND: &str = "textures/gui/options_background.png";
/// `textures/gui/icons.png` — hearts, armour, the crosshair, the ping bars.
pub const ICONS_TEXTURE: &str = "textures/gui/icons.png";

/// The UV every [`UiTexture::Solid`] vertex carries once [`DrawList::enable_solid_merging`] is
/// on. It is far outside any real GUI UV (atlas and font UVs are in `0..=1`, tiled backgrounds
/// and the glint's scrolling UVs stay within a few units), and the UI fragment shader replaces
/// the sample with exactly `1.0` below [`SOLID_UV_THRESHOLD`] — the value the 1×1 white Solid
/// texel samples to — so a Solid quad can share any textured batch with identical pixels.
pub const SOLID_SENTINEL_UV: [f32; 2] = [-65_536.0, -65_536.0];
/// See [`SOLID_SENTINEL_UV`]; mirrored by the WGSL in `widget::gpu`.
pub const SOLID_UV_THRESHOLD: f32 = -32_768.0;

/// The largest projected extent, in GUI pixels, on either axis, of a flat GUI item
/// quad that [`DrawList::enable_item_sliver_culling`] treats as edge-on and leaves out.
pub const GUI_ITEM_SLIVER_EXTENT: f32 = 1.0e-3;

/// Kill switch for [`DrawList::enable_item_sliver_culling`] on the live frame:
/// `MC_NW_F3_ITEM_SLIVERS=0` keeps every sliver quad, as before. Read once per process.
#[must_use]
pub fn nw_f3_item_slivers_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("MC_NW_F3_ITEM_SLIVERS").is_none_or(|value| value != "0")
    });
    *ON
}

/// An axis-aligned rectangle in GUI pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rect {
    /// Left edge.
    pub x: i32,
    /// Top edge.
    pub y: i32,
    /// Width. Negative widths are treated as empty by [`Rect::contains`].
    pub width: i32,
    /// Height.
    pub height: i32,
}

impl Rect {
    /// A rectangle from position and size.
    #[must_use]
    pub const fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Right edge, exclusive.
    #[must_use]
    pub const fn right(&self) -> i32 {
        self.x + self.width
    }

    /// Bottom edge, exclusive.
    #[must_use]
    pub const fn bottom(&self) -> i32 {
        self.y + self.height
    }

    /// Vanilla's hit test: **inclusive** on the top-left edges, **exclusive** on the
    /// bottom-right ones.
    ///
    /// Getting this wrong by one is the difference between a button that can be clicked on its
    /// last pixel row and one that cannot.
    // java: GuiButton.java:128 (mousePressed)
    #[must_use]
    pub const fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x && y >= self.y && x < self.right() && y < self.bottom()
    }
}

/// Which texture a batch samples.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UiTexture {
    /// A 1×1 opaque white texture: untextured rects, gradients, bars.
    Solid,
    /// One of the font sheets.
    Font(FontTexture),
    /// An arbitrary resource, e.g. `textures/gui/widgets.png`.
    Resource(ResourceLocation),
}

impl UiTexture {
    /// A `textures/gui/...` resource by path.
    #[must_use]
    pub fn gui(path: &str) -> Self {
        UiTexture::Resource(ResourceLocation::minecraft(path))
    }

    /// [`ICONS_TEXTURE`], built once. [`UiTexture::gui`] allocates two `String`s per call and
    /// the HUD asks for this sheet several times per frame (C2-12).
    #[must_use]
    pub fn icons() -> &'static UiTexture {
        static KEY: std::sync::LazyLock<UiTexture> =
            std::sync::LazyLock::new(|| UiTexture::gui(ICONS_TEXTURE));
        &KEY
    }

    /// [`WIDGETS_TEXTURE`], built once (see [`UiTexture::icons`]).
    #[must_use]
    pub fn widgets() -> &'static UiTexture {
        static KEY: std::sync::LazyLock<UiTexture> =
            std::sync::LazyLock::new(|| UiTexture::gui(WIDGETS_TEXTURE));
        &KEY
    }
}

/// One vertex. Position in GUI pixels, UV normalised, colour straight (non-premultiplied) RGBA.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UiVertex {
    /// Screen position in GUI pixels.
    pub pos: [f32; 2],
    /// Normalised texture coordinate.
    pub uv: [f32; 2],
    /// Straight-alpha RGBA in 0..=1.
    pub color: [f32; 4],
}

impl UiVertex {
    /// Bytes per vertex in the GPU buffer.
    pub const SIZE: usize = 8 * 4;

    /// Append this vertex's little-endian bytes to `out`.
    ///
    /// Done by hand rather than with `bytemuck::Pod`, because `mc-ui` keeps
    /// `#![forbid(unsafe_code)]` and the derive expands to an `unsafe impl`.
    pub fn write_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }

    /// This vertex's 32 little-endian bytes, in `pos`, `uv`, `color` order.
    ///
    /// R1-18 / GPURES-6: one fixed-size array per vertex, so a frame's upload is one capacity
    /// check and one 32-byte copy per vertex instead of eight 4-byte `extend_from_slice` calls.
    /// Byte-identical to the per-float path on every target, big-endian included.
    #[must_use]
    pub fn to_le_bytes(&self) -> [u8; Self::SIZE] {
        let floats = [
            self.pos[0],
            self.pos[1],
            self.uv[0],
            self.uv[1],
            self.color[0],
            self.color[1],
            self.color[2],
            self.color[3],
        ];
        let mut out = [0_u8; Self::SIZE];
        for (bytes, float) in out.chunks_exact_mut(4).zip(floats) {
            bytes.copy_from_slice(&float.to_le_bytes());
        }
        out
    }
}

/// The fixed-function blend equation a UI batch needs.
///
/// The usual GUI path is straight alpha.  The crosshair is the one 1.8.9 exception: it uses
/// `tryBlendFuncSeparate(GL_ONE_MINUS_DST_COLOR, GL_ONE_MINUS_SRC_COLOR, GL_ONE, GL_ZERO)`,
/// making the light pixels dark over a light scene and vice versa.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UiBlend {
    /// `GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ZERO`.
    #[default]
    Alpha,
    /// `GL_ONE_MINUS_DST_COLOR, GL_ONE_MINUS_SRC_COLOR, GL_ONE, GL_ZERO`.
    Invert,
    /// The red/green complement pass of `GuiTextField`'s colour-logic selection rectangle.
    ///
    /// Java asks OpenGL for opcode `5387` (`GL_OR_REVERSE`), not `GL_XOR`: `src | !dst`.
    /// For the fixed opaque-blue source this means `!dst` in red and green.  WGPU has no
    /// colour-logic-op state, so the exact operation is emitted as two masked passes: this pass
    /// uses an opaque white source and `1 - dst` (`Subtract`, source and destination both one),
    /// writing only red and green.
    XorComplement,
    /// The blue/alpha force pass completing Java's `GL_OR_REVERSE` selection rectangle.
    ///
    /// Its opaque blue source and blue/alpha write mask produce `1 | !dst = 1` in those two
    /// channels.  The historical `Xor` name is retained because the text-field seam and its
    /// public batch assertions already use it; the Java opcode is documented above and below.
    Xor,
    /// Vanilla's item-foil pass: `GlStateManager.blendFunc(768, 1)` =
    /// `GL_SRC_COLOR, GL_ONE`.
    ///
    /// The item glint is deliberately a separate batch kind rather than an alpha-tinted icon.
    /// Its source colour is added to the destination, so using the ordinary straight-alpha
    /// pipeline makes an enchanted stack visibly (and measurably) too faint.
    Glint,
    /// `GL_BLEND` disabled: the fragment's colour replaces the target's, whatever its alpha.
    ///
    /// Only [`UnblendedText`] emits it, always under the alpha test that
    /// `FontRenderer.drawString` enables, so a glyph texel lands opaque or not at all. The target
    /// alpha is composed like [`UiBlend::Alpha`]'s, since nothing visible reads it.
    Disabled,
}

/// The depth-test window selected for one UI batch.
///
/// Vanilla's ordinary widgets are drawn after `GuiContainer` has disabled depth. Item models
/// immediately re-enable `GL_LEQUAL` and depth writes, while `RenderItem.renderEffect` keeps the
/// test at `GL_EQUAL` and masks writes for its foil replay. Keeping that distinction beside the
/// batch lets the CPU draw list preserve the exact state window when a screen interleaves
/// ordinary widgets and inventory geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UiDepth {
    /// Screen-space overlay: depth is always accepted and does not update the shared buffer.
    #[default]
    Overlay,
    /// Opaque/baseline item or entity geometry: `GL_LEQUAL`, depth writes enabled.
    Item,
    /// Item foil/entity glint replay: `GL_EQUAL`, depth writes disabled.
    Equal,
}

impl UiDepth {
    /// A stable center depth for 2-D geometry emitted inside an item state window (for example a
    /// filled-map card). Fully 3-D callers use [`DrawList::triangle_with_blend_depth`] instead.
    #[must_use]
    pub const fn default_value(self) -> f32 {
        match self {
            Self::Overlay => 0.0,
            // Java's normal slot z-level is 200 in the `1000..3000` GUI projection: 0.4 in the
            // WGPU 0..1 depth range. Cursor stacks use a later submission state but share this
            // center value when their model has no per-vertex Z.
            Self::Item | Self::Equal => 0.4,
        }
    }
}

/// The back-face state selected for one UI batch.
///
/// The world pass leaves `GL_CULL_FACE/GL_BACK` active when GUI rendering starts. Player preview
/// rendering is the deliberate exception: `RendererLivingEntity` disables culling around the
/// model and all of its layers. Item batches retain the inherited back-face cull, including
/// generated front/back cards and builtin tile-item geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UiCull {
    /// Inherit vanilla's normal GUI item/world state: cull back-facing triangles.
    #[default]
    Back,
    /// RendererLivingEntity's local cull-disabled window used by the inventory preview.
    None,
}

/// A contiguous run of vertices sharing one texture and blend equation.
#[derive(Debug, Clone, PartialEq)]
pub struct DrawBatch {
    /// Texture for this run.
    pub texture: UiTexture,
    /// Fixed-function blend state selected for this run.
    pub blend: UiBlend,
    /// Depth-test/write state for this contiguous run.
    pub depth: UiDepth,
    /// Back-face state for this contiguous run.
    pub cull: UiCull,
    /// Whether this run is drawn under vanilla's item/entity alpha test
    /// (`GL_ALPHA_TEST` with `alphaFunc(GL_GREATER, 0.1F)`); see
    /// [`DrawList::with_alpha_test`].
    pub alpha_test: bool,
    /// Vertex indices into [`DrawList::vertices`].
    pub range: Range<usize>,
}

/// Unpack `0xAARRGGBB` into straight-alpha RGBA floats.
#[must_use]
pub fn unpack_argb(argb: u32) -> [f32; 4] {
    [
        ((argb >> 16) & 255) as f32 / 255.0,
        ((argb >> 8) & 255) as f32 / 255.0,
        (argb & 255) as f32 / 255.0,
        ((argb >> 24) & 255) as f32 / 255.0,
    ]
}

/// An ordered list of textured triangles: everything one frame of UI wants to draw.
///
/// # The transform stack
///
/// [`push_transform`](Self::push_transform)/[`pop_transform`](Self::pop_transform) are the port
/// of vanilla's `GlStateManager.pushMatrix()`/`popMatrix()` around a GUI draw — see
/// [`crate::widget::transform`] for why it is a stack rather than a matrix per [`DrawBatch`].
/// While the stack is **empty**, vertex positions are stored exactly as the caller computed
/// them, with no arithmetic applied at all: a draw that pushes nothing produces byte-identical
/// output to a `DrawList` that has no stack, which is pinned by
/// `an_untransformed_drawlist_is_byte_identical_to_the_pre_transform_tree`.
///
/// The stack is part of the list's state, so [`PartialEq`] compares it too; balanced callers
/// always compare with it empty.
#[derive(Debug, Clone, Default)]
pub struct DrawList {
    vertices: Vec<UiVertex>,
    /// One WGPU depth value per vertex. Kept in a parallel stream so the public 2-D [`UiVertex`]
    /// layout and its byte-identity tests remain stable for ordinary widgets.
    depths: Vec<f32>,
    batches: Vec<DrawBatch>,
    /// Each entry is the *fully composed* matrix at that depth, so emission only ever reads
    /// the last one.
    transforms: Vec<Transform>,
    /// Builtin/entity item descriptors whose geometry was emitted into this list.  Their fixed
    /// entity sheets (and a banner's layered composite) are uploaded by [`UiRenderer`] before
    /// the list reaches the GPU; keeping the descriptors beside the batches makes the dynamic
    /// banner key reachable without teaching every screen about GPU handles.
    builtin_tile_items: Vec<BuiltinTileItem>,
    /// CPU model cache for builtin/entity item geometry.  It is deliberately not part of list
    /// equality: it only holds reusable mutable model state, while `builtin_tile_items` is the
    /// observable submission contract.
    builtin_tile_models: Option<Box<TileModels>>,
    /// Current CPU state window. It is restored by [`Self::with_render_state`] after an item or
    /// entity helper returns, so depth/cull cannot leak into the next widget.
    depth_mode: UiDepth,
    cull_mode: UiCull,
    /// Current alpha-test window, restored by [`Self::with_alpha_test`] like the depth/cull pair.
    alpha_test: bool,
    /// Scoped default for flat geometry emitted inside an item/effect state window. Model callers
    /// normally provide an explicit per-vertex depth; map cards and projected quads use this so
    /// Java's per-call `RenderItem.zLevel` can be retained without changing [`UiVertex`].
    item_depth_override: Option<f32>,
    /// #24: let untextured (Solid) geometry join the neighbouring textured batch. Off by
    /// default so every pre-existing byte/batch golden keeps its meaning; the live UI frame turns
    /// it on. Configuration, not content: [`Self::clear`] keeps it and equality ignores it.
    merge_solid: bool,
    /// Skip the edge-on "sliver" quads of flat GUI item models (see
    /// [`Self::enable_item_sliver_culling`]). Off by default, like `merge_solid`; the live UI
    /// frame turns it on. Configuration, not content: [`Self::clear`] keeps it and equality
    /// ignores it.
    cull_item_slivers: bool,
}

impl PartialEq for DrawList {
    fn eq(&self, other: &Self) -> bool {
        self.vertices == other.vertices
            && self.depths == other.depths
            && self.batches == other.batches
            && self.transforms == other.transforms
            && self.builtin_tile_items == other.builtin_tile_items
    }
}

impl DrawList {
    /// An empty list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop everything, including any unbalanced transform stack. Call once per frame.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.depths.clear();
        self.batches.clear();
        self.transforms.clear();
        self.builtin_tile_items.clear();
        self.depth_mode = UiDepth::Overlay;
        self.cull_mode = UiCull::Back;
        self.alpha_test = false;
        self.item_depth_override = None;
    }

    /// Merge Solid rects into adjacent textured batches (#24, pixel-exact).
    ///
    /// From now on every [`UiTexture::Solid`] vertex is stored with [`SOLID_SENTINEL_UV`] and may
    /// extend the previous batch whatever its texture, as long as blend, depth and cull match;
    /// a batch holding only Solid vertices adopts the texture of the next textured vertex. Only
    /// adjacent geometry merges, so submission (painter's) order is unchanged, and the UI shader
    /// draws sentinel vertices with exactly the `1.0` texel the Solid texture provides. A
    /// sidebar row, chat line or F3 line then costs one batch instead of two.
    ///
    /// Call it on an empty list (the live frame does so right after taking its list); it stays
    /// on across [`Self::clear`].
    pub fn enable_solid_merging(&mut self) {
        debug_assert!(self.batches.is_empty(), "enable merging before drawing");
        self.merge_solid = true;
    }

    /// Whether [`Self::enable_solid_merging`] is on.
    #[must_use]
    pub fn solid_merging(&self) -> bool {
        self.merge_solid
    }

    /// Leave out the quads of flat (non-GUI-3D) item models that the GUI camera
    /// sees exactly edge-on along a pixel boundary. Pixel-identical; it only drops geometry that
    /// cannot cover a pixel centre.
    ///
    /// A generated item model is a front card, a back card and one thin side quad per run of edge
    /// texels (`ItemModelGenerator`'s spans: 40-60 per item, 3x that with the enchantment glint).
    /// Vanilla and the port submit them all, but the GUI transform looks straight down the
    /// model's Z axis, so every side quad projects to a line: its screen extent across the line
    /// is ~1e-7 GUI pixels (only `sin(180 deg)` in `f32` keeps it from being exactly 0), and the
    /// line is a texel boundary, i.e. a whole GUI pixel. A quad is skipped only when **all** of
    /// these hold, which together confine it to within ~0.006 physical pixels of a pixel
    /// boundary, while every sample point (a pixel centre) is 0.5 px from one:
    ///
    /// * its projected extent is at most [`GUI_ITEM_SLIVER_EXTENT`] GUI pixels on one axis, and
    ///   on that axis it lies within that distance of a whole GUI pixel (checked per model, at
    ///   slot origin; slot positions are whole GUI pixels);
    /// * the list's transform is the identity or a whole-pixel translation at scale 1
    ///   ([`Self::whole_pixel_placement`]): a scaled or animated icon (the hotbar's pick-up pop,
    ///   a HUD element at 70%, independent GUI scales) keeps every quad;
    /// * the GUI projection maps a GUI pixel to exactly `factor` physical pixels (the `exact_ortho`, on unless `MC_NW_TEXT_ORTHO=0`).
    ///
    /// Without the last two conditions a sliver can land within 1e-7 px of a pixel centre; a
    /// SkyWars frame with the hotbar's pop animation showed one blend 3 pixels by one level.
    /// Fronts, backs (left to the GPU's back-face cull, as before) and every GUI-3D face are
    /// untouched. `tests/nw_f3_item_slivers.rs` renders every item model both ways and compares
    /// the frames byte for byte.
    ///
    /// Call it on an empty list (the live frame does so right after taking its list); it stays on
    /// across [`Self::clear`].
    pub fn enable_item_sliver_culling(&mut self) {
        self.cull_item_slivers = true;
    }

    /// Whether [`Self::enable_item_sliver_culling`] is on.
    #[must_use]
    pub fn item_sliver_culling(&self) -> bool {
        self.cull_item_slivers
    }

    /// Whether geometry emitted now lands on the GUI pixel grid it is given in:
    /// the GUI projection is exact (`exact_ortho`) and the transform stack is empty or its top is
    /// a translation by whole GUI pixels at scale 1 with no rotation or shear.
    #[must_use]
    pub fn whole_pixel_placement(&self) -> bool {
        crate::widget::scaled::exact_ortho()
            && self.transforms.last().is_none_or(|t| {
                let [[xx, xy, dx], [yx, yy, dy]] = t.m;
                xx == 1.0
                    && yy == 1.0
                    && xy == 0.0
                    && yx == 0.0
                    && dx.fract() == 0.0
                    && dy.fract() == 0.0
            })
    }

    /// Whether a vertex for `texture`/`blend` may extend `batch` under the current state window,
    /// and if so whether the batch must adopt `texture` (a pure-Solid batch meeting a textured
    /// vertex while merging).
    fn batch_accepts(
        &self,
        batch: &DrawBatch,
        texture: &UiTexture,
        blend: UiBlend,
    ) -> Option<bool> {
        if batch.blend != blend
            || batch.depth != self.depth_mode
            || batch.cull != self.cull_mode
            || batch.alpha_test != self.alpha_test
        {
            return None;
        }
        if batch.texture == *texture {
            return Some(false);
        }
        if !self.merge_solid {
            return None;
        }
        if matches!(texture, UiTexture::Solid) {
            Some(false)
        } else if matches!(batch.texture, UiTexture::Solid) {
            Some(true)
        } else {
            None
        }
    }

    /// Push `local` onto the transform stack, composed inside whatever is already there.
    ///
    /// Positions handed to every drawing method from here until the matching
    /// [`pop_transform`](Self::pop_transform) are mapped through the result. Colours, UVs and
    /// batching are untouched.
    ///
    /// ```
    /// use mc_ui::widget::{DrawList, Transform};
    /// let mut d = DrawList::new();
    /// d.push_transform(Transform::translate(100.0, 50.0).then_scale(2.0));
    /// d.rect(0, 0, 4, 4, 0xFFFF_FFFF);
    /// d.pop_transform();
    /// assert_eq!(d.vertices()[0].pos, [100.0, 50.0]);
    /// ```
    // java: GlStateManager.pushMatrix() followed by translate/rotate/scale
    pub fn push_transform(&mut self, local: Transform) {
        let composed = match self.transforms.last() {
            Some(top) => top.concat(local),
            None => local,
        };
        self.transforms.push(composed);
    }

    /// Pop the innermost transform.
    ///
    /// Popping an empty stack does nothing, matching GL, which raises `GL_STACK_UNDERFLOW` and
    /// leaves the matrix alone rather than misbehaving.
    // java: GlStateManager.popMatrix()
    pub fn pop_transform(&mut self) {
        self.transforms.pop();
    }

    /// The transform currently in force, [`Transform::IDENTITY`] when the stack is empty.
    ///
    /// Its [`try_inverse`](Transform::try_inverse) is how a screen maps a cursor position back
    /// into the space it drew in.
    #[must_use]
    pub fn transform(&self) -> Transform {
        self.transforms
            .last()
            .copied()
            .unwrap_or(Transform::IDENTITY)
    }

    /// How deep the transform stack is. Zero means "no transform applied".
    #[must_use]
    pub fn transform_depth(&self) -> usize {
        self.transforms.len()
    }

    /// Run `f` with `local` pushed, popping it afterwards.
    ///
    /// The balanced form, for the common case where the push and the pop are in one function.
    pub fn with_transform<R>(&mut self, local: Transform, f: impl FnOnce(&mut Self) -> R) -> R {
        self.push_transform(local);
        let out = f(self);
        self.pop_transform();
        out
    }

    /// Multiply the opacity of geometry emitted by `f`, including cached/appended geometry.
    /// Existing vertices are untouched; nested scopes multiply their opacity factors.
    pub fn with_opacity<R>(&mut self, opacity: f32, f: impl FnOnce(&mut Self) -> R) -> R {
        let start = self.vertices.len();
        let result = f(self);
        let opacity = opacity.clamp(0.0, 1.0);
        if opacity < 1.0 {
            for vertex in &mut self.vertices[start..] {
                vertex.color[3] *= opacity;
            }
        }
        result
    }

    /// Whether nothing has been drawn.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Every vertex, in submission order.
    #[must_use]
    pub fn vertices(&self) -> &[UiVertex] {
        &self.vertices
    }

    /// The texture batches, in submission order.
    ///
    /// Consecutive draws sharing a texture are merged, so a screen that alternates text and
    /// widget art produces a batch per switch, not per glyph.
    #[must_use]
    pub fn batches(&self) -> &[DrawBatch] {
        &self.batches
    }

    /// One normalized WGPU depth value per vertex, parallel to [`Self::vertices`].
    ///
    /// Overlay vertices use zero because their pipeline accepts every fragment. Item/entity
    /// callers replace it with the GUI orthographic depth derived from their 3-D model Z.
    #[must_use]
    pub fn depths(&self) -> &[f32] {
        &self.depths
    }

    /// Builtin/entity item descriptors referenced by this frame's geometry.
    ///
    /// [`UiRenderer`](crate::widget::UiRenderer) consumes these before uploading the draw list:
    /// chest/skull sheets go through the ordinary resource-pack loader, while a banner descriptor
    /// asks the render-side tile-item API for its deterministic layered pixels.
    #[must_use]
    pub fn builtin_tile_items(&self) -> &[BuiltinTileItem] {
        &self.builtin_tile_items
    }

    /// Record one builtin/entity descriptor, coalescing repeated icons in the same frame.
    ///
    /// This is crate-visible because item screens should call the higher-level item renderer;
    /// keeping registration here makes the GPU submission contract explicit and testable.
    pub(crate) fn register_builtin_tile_item(&mut self, item: BuiltinTileItem) {
        if !self.builtin_tile_items.contains(&item) {
            self.builtin_tile_items.push(item);
        }
    }

    /// Borrow the frame's reusable tile-item model cache while emitting one descriptor.
    ///
    /// `TileModels` owns mutable lid/pose fields just like vanilla's cached
    /// `TileEntityItemStackRenderer` instances.  The cache lives on the list so independent
    /// headless lists never share mutable model state, and a new list pays construction only on
    /// the first builtin item it draws.
    pub(crate) fn with_builtin_tile_models<R>(
        &mut self,
        f: impl FnOnce(&mut TileModels) -> R,
    ) -> R {
        let models = self.builtin_tile_models.get_or_insert_with(|| {
            Box::new(TileModels::new(
                mc_render::entity::tilepass::christmas_today(),
            ))
        });
        f(models)
    }

    /// The whole vertex buffer as little-endian bytes, ready for `queue.write_buffer`.
    #[must_use]
    pub fn vertex_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_vertex_bytes(&mut out);
        out
    }

    /// [`Self::vertex_bytes`] into a caller-retained buffer, replacing its contents.
    ///
    /// R1-18: the renderer keeps `out` across frames, so a frame's UI upload no longer
    /// allocates (and, above the allocator's mmap threshold, page-faults) a fresh buffer.
    pub fn write_vertex_bytes(&self, out: &mut Vec<u8>) {
        out.clear();
        out.reserve(self.vertices.len() * UiVertex::SIZE);
        for v in &self.vertices {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    /// The per-vertex depth stream, ready for the UI pipeline's second vertex buffer.
    #[must_use]
    pub fn depth_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_depth_bytes(&mut out);
        out
    }

    /// [`Self::depth_bytes`] into a caller-retained buffer, replacing its contents.
    pub fn write_depth_bytes(&self, out: &mut Vec<u8>) {
        out.clear();
        out.reserve(self.depths.len() * core::mem::size_of::<f32>());
        for depth in &self.depths {
            out.extend_from_slice(&depth.to_le_bytes());
        }
    }

    /// The bytes of [`Self::write_vertex_bytes`], written straight into `out`
    /// (staging memory of exactly `vertices().len() * UiVertex::SIZE` bytes), one vertex at a
    /// time and in order, as the upload ring wants them.
    pub fn fill_vertex_bytes(&self, out: wgpu::WriteOnly<'_, [u8]>) {
        let (chunks, _) = out.into_chunks::<{ UiVertex::SIZE }>();
        for (chunk, v) in chunks.into_iter().zip(&self.vertices) {
            chunk.write(v.to_le_bytes());
        }
    }

    /// The bytes of [`Self::write_depth_bytes`], written straight into `out`
    /// (exactly `depths().len() * 4` bytes).
    pub fn fill_depth_bytes(&self, out: wgpu::WriteOnly<'_, [u8]>) {
        let (chunks, _) = out.into_chunks::<4>();
        for (chunk, depth) in chunks.into_iter().zip(&self.depths) {
            chunk.write(depth.to_le_bytes());
        }
    }

    /// Run `f` under one vanilla UI depth/cull state window, then restore the previous state.
    ///
    /// The state is recorded in each resulting [`DrawBatch`], so the GPU can change pipelines
    /// at the exact draw boundary without relying on a mutable global state machine.
    pub fn with_render_state<R>(
        &mut self,
        depth: UiDepth,
        cull: UiCull,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let previous = (self.depth_mode, self.cull_mode);
        self.depth_mode = depth;
        self.cull_mode = cull;
        let out = f(self);
        self.depth_mode = previous.0;
        self.cull_mode = previous.1;
        out
    }

    /// Run `f` under vanilla's item/entity alpha test, then restore the previous state.
    ///
    /// java: `RenderItem.renderItemIntoGUI` runs `enableAlpha(); alphaFunc(GL_GREATER, 0.1F)`
    /// around the item model *and* its `renderEffect` foil (and `disableAlpha()` after), and
    /// `RendererLivingEntity.doRender` enables the same test for the inventory player preview. A
    /// fragment whose final alpha is at most `0.1` is discarded, so it neither colours the target
    /// nor writes depth. That is what keeps the transparent texels of a generated item card out
    /// of the depth buffer, and therefore out of the `GL_EQUAL` foil replay, and what lets a
    /// beacon's core show through the clear texels of its glass shell. Ordinary widgets and the
    /// other depth-aware passes (map cards, the achievement map) stay untested.
    pub fn with_alpha_test<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        let previous = self.alpha_test;
        self.alpha_test = true;
        let out = f(self);
        self.alpha_test = previous;
        out
    }

    /// Run `f` with a scoped default depth for item/effect geometry, then restore the previous
    /// value. This is the UI equivalent of Java's mutable `RenderItem.zLevel`: the item state
    /// window remains selected by [`Self::with_render_state`], while flat/map quads inherit the
    /// exact normalized depth for that call. Explicit depths supplied to
    /// [`Self::triangle_with_blend_depth`] are unaffected.
    pub fn with_item_depth<R>(&mut self, depth: f32, f: impl FnOnce(&mut Self) -> R) -> R {
        let previous = self.item_depth_override;
        self.item_depth_override = Some(depth);
        let out = f(self);
        self.item_depth_override = previous;
        out
    }

    fn default_depth_value(&self) -> f32 {
        if matches!(self.depth_mode, UiDepth::Item | UiDepth::Equal) {
            self.item_depth_override
                .unwrap_or_else(|| self.depth_mode.default_value())
        } else {
            self.depth_mode.default_value()
        }
    }

    /// Account `count` new vertices for `texture`/`blend`: extend the last batch when it accepts
    /// them, otherwise open a new one.
    fn extend_or_open_batch(&mut self, texture: &UiTexture, blend: UiBlend, count: usize) {
        let accepted = self
            .batches
            .last()
            .and_then(|b| self.batch_accepts(b, texture, blend));
        match (accepted, self.batches.last_mut()) {
            (Some(adopt), Some(b)) => {
                if adopt {
                    b.texture = texture.clone();
                }
                b.range.end += count;
            }
            _ => {
                let start = self.vertices.len();
                self.batches.push(DrawBatch {
                    texture: texture.clone(),
                    blend,
                    depth: self.depth_mode,
                    cull: self.cull_mode,
                    alpha_test: self.alpha_test,
                    range: start..start + count,
                });
            }
        }
    }

    /// Push the six vertices of one quad into the batch for `texture`/`blend`, within the
    /// current [`UiDepth`]/[`UiCull`] state window.
    ///
    /// This replaced six per-vertex pushes with identical output: the batch key cannot change
    /// inside one quad, so the first vertex decided "extend or open" and the other five always
    /// extended. The key (a `String` compare for resource textures, C2-12) now runs once per
    /// quad. Blend state is part of the key: merging an inverted crosshair into an adjacent
    /// icons.png alpha batch would silently redraw it with the ordinary UI equation.
    ///
    /// The transform is applied here, to positions only, and only when something is actually on
    /// the stack — the empty case must not so much as multiply by 1.0, because `1.0 * -0.0`
    /// summed with `0.0 * y` is `+0.0`, a different bit pattern from the `-0.0` the caller
    /// passed in. [`Self::triangle_with_blend_depth`] keeps the same rule.
    fn push_quad_vertices(
        &mut self,
        texture: &UiTexture,
        blend: UiBlend,
        depth: f32,
        mut verts: [UiVertex; 6],
    ) {
        if self.merge_solid && matches!(texture, UiTexture::Solid) {
            for v in &mut verts {
                v.uv = SOLID_SENTINEL_UV;
            }
        }
        self.extend_or_open_batch(texture, blend, 6);
        match self.transforms.last() {
            Some(t) => {
                let t = *t;
                self.vertices.extend(verts.iter().map(|v| UiVertex {
                    pos: t.apply(v.pos),
                    ..*v
                }));
            }
            None => self.vertices.extend_from_slice(&verts),
        }
        self.depths.extend_from_slice(&[depth; 6]);
    }

    /// Two triangles covering a quad whose corners are given in vanilla's triangle-strip order:
    /// top-left, bottom-left, top-right, bottom-right.
    pub fn quad(
        &mut self,
        texture: &UiTexture,
        pos: [[f32; 2]; 4],
        uv: [[f32; 2]; 4],
        color: [[f32; 4]; 4],
    ) {
        self.quad_with_blend(texture, UiBlend::Alpha, pos, uv, color);
    }

    /// [`Self::quad`] with an explicit fixed-function blend state.
    fn quad_with_blend(
        &mut self,
        texture: &UiTexture,
        blend: UiBlend,
        pos: [[f32; 2]; 4],
        uv: [[f32; 2]; 4],
        color: [[f32; 4]; 4],
    ) {
        let v = |i: usize| UiVertex {
            pos: pos[i],
            uv: uv[i],
            color: color[i],
        };
        let depth = self.default_depth_value();
        self.push_quad_vertices(texture, blend, depth, [v(0), v(1), v(2), v(1), v(3), v(2)]);
    }

    /// One triangle. The batch's [`UiCull`] and [`UiDepth`] state come from the surrounding
    /// [`Self::with_render_state`] window; ordinary widgets use the default overlay state, while
    /// player/entity callers select the cull-disabled depth-aware window explicitly.
    ///
    /// For geometry that does not decompose into an axis-aligned or strip-ordered quad — the
    /// CPU-projected player model
    /// [`crate::screens::container::player_preview`] draws into the inventory screens is the one
    /// caller today — this is the lower-level primitive [`quad`](Self::quad) itself is built on.
    pub fn triangle(
        &mut self,
        texture: &UiTexture,
        pos: [[f32; 2]; 3],
        uv: [[f32; 2]; 3],
        color: [[f32; 4]; 3],
    ) {
        self.triangle_with_blend(texture, UiBlend::Alpha, pos, uv, color);
    }

    /// One triangle using an explicit fixed-function blend equation.
    ///
    /// Most UI geometry should use [`Self::triangle`].  This lower-level form is for the few
    /// vanilla passes whose blend state is part of their visible result, such as the animated
    /// item foil (`UiBlend::Glint`).
    pub fn triangle_with_blend(
        &mut self,
        texture: &UiTexture,
        blend: UiBlend,
        pos: [[f32; 2]; 3],
        uv: [[f32; 2]; 3],
        color: [[f32; 4]; 3],
    ) {
        self.triangle_with_blend_depth(
            texture,
            blend,
            [self.default_depth_value(); 3],
            pos,
            uv,
            color,
        );
    }

    /// One triangle with explicit per-vertex normalized depths.
    ///
    /// The current [`UiDepth`] and [`UiCull`] state still comes from [`Self::with_render_state`];
    /// this method only supplies the model-space Z projection that the GPU depth test compares.
    pub fn triangle_with_blend_depth(
        &mut self,
        texture: &UiTexture,
        blend: UiBlend,
        depth: [f32; 3],
        pos: [[f32; 2]; 3],
        uv: [[f32; 2]; 3],
        color: [[f32; 4]; 3],
    ) {
        // Same result as the former three per-vertex pushes (the key cannot change inside one
        // triangle); the batch key — a `String` compare for the item atlas — runs once, and the
        // transform rule of `push_quad_vertices` applies.
        let solid = self.merge_solid && matches!(texture, UiTexture::Solid);
        self.extend_or_open_batch(texture, blend, 3);
        let transform = self.transforms.last().copied();
        for i in 0..3 {
            let at = match transform {
                Some(t) => t.apply(pos[i]),
                None => pos[i],
            };
            self.vertices.push(UiVertex {
                pos: at,
                uv: if solid { SOLID_SENTINEL_UV } else { uv[i] },
                color: color[i],
            });
            self.depths.push(depth[i]);
        }
    }

    /// Append every vertex of `other` through this list's transform and state window, exactly
    /// as if each of its triangles had been emitted here (same vertices, UVs, depths and
    /// batching, the source batch's blend), with one batch check per batch of `other` instead
    /// of one per triangle.
    ///
    /// For UI that is built once into a private list (solid merging off, no transform) and
    /// replayed every frame: the Spotify widget's Aero style.
    pub fn append(&mut self, other: &DrawList) {
        let transform = self.transforms.last().copied();
        let depth = self.default_depth_value();
        for batch in &other.batches {
            let source = &other.vertices[batch.range.clone()];
            if source.is_empty() {
                continue;
            }
            let solid = self.merge_solid && matches!(batch.texture, UiTexture::Solid);
            self.extend_or_open_batch(&batch.texture, batch.blend, source.len());
            self.vertices.extend(source.iter().map(|v| UiVertex {
                pos: match transform {
                    Some(t) => t.apply(v.pos),
                    None => v.pos,
                },
                uv: if solid { SOLID_SENTINEL_UV } else { v.uv },
                color: v.color,
            }));
            self.depths.extend(std::iter::repeat_n(depth, source.len()));
        }
    }

    /// An axis-aligned textured quad.
    pub fn quad_rect(
        &mut self,
        texture: &UiTexture,
        (x0, y0, x1, y1): (f32, f32, f32, f32),
        (u0, v0, u1, v1): (f32, f32, f32, f32),
        argb: u32,
    ) {
        self.quad_rect_with_blend(
            texture,
            UiBlend::Alpha,
            (x0, y0, x1, y1),
            (u0, v0, u1, v1),
            argb,
        );
    }

    /// [`Self::quad_rect`] with an explicit fixed-function blend state.
    fn quad_rect_with_blend(
        &mut self,
        texture: &UiTexture,
        blend: UiBlend,
        (x0, y0, x1, y1): (f32, f32, f32, f32),
        (u0, v0, u1, v1): (f32, f32, f32, f32),
        argb: u32,
    ) {
        let c = unpack_argb(argb);
        let pos = [[x0, y0], [x0, y1], [x1, y0], [x1, y1]];
        let uv = [[u0, v0], [u0, v1], [u1, v0], [u1, v1]];
        let v = |i: usize| UiVertex {
            pos: pos[i],
            uv: uv[i],
            color: c,
        };
        let depth = self.default_depth_value();
        self.push_quad_vertices(texture, blend, depth, [v(0), v(1), v(2), v(1), v(3), v(2)]);
    }

    /// A solid rectangle. Corners may be given in either order.
    // java: Gui.java:50 (drawRect)
    pub fn rect(&mut self, left: i32, top: i32, right: i32, bottom: i32, argb: u32) {
        let (l, r) = (left.min(right), left.max(right));
        let (t, b) = (top.min(bottom), top.max(bottom));
        self.quad_rect(
            &UiTexture::Solid,
            (l as f32, t as f32, r as f32, b as f32),
            (0.0, 0.0, 1.0, 1.0),
            argb,
        );
    }

    /// Vanilla's `GuiTextField.drawCursorVertical` selection rectangle: opaque-blue
    /// `GL_OR_REVERSE` (opcode `5387`).
    ///
    /// WGPU does not expose OpenGL's colour logic operations.  The exact fixed palette is
    /// therefore decomposed into a red/green complement pass followed by a blue/alpha replace
    /// pass.  This is deliberately not a colour-taking general-purpose primitive: the only
    /// 1.8.9 caller sets `GlStateManager.color(0, 0, 255, 255)` before opcode `5387`.
    // java: GuiTextField.java:625-641 (`colorLogicOp(5387)` = `GL_OR_REVERSE`)
    pub fn rect_xor_blue(&mut self, left: i32, top: i32, right: i32, bottom: i32) {
        let (l, r) = (left.min(right), left.max(right));
        let (t, b) = (top.min(bottom), top.max(bottom));
        // For source red/green = 0, OR_REVERSE is `!dst`, i.e. `1 - dst` in an RGBA8 target.
        // The complement pipeline writes only those channels, leaving blue and alpha for the
        // force pass below.
        self.quad_rect_with_blend(
            &UiTexture::Solid,
            UiBlend::XorComplement,
            (l as f32, t as f32, r as f32, b as f32),
            (0.0, 0.0, 1.0, 1.0),
            0xFFFF_FFFF,
        );
        // For source blue/alpha = 1, OR_REVERSE is `1 | !dst = 1`.
        self.quad_rect_with_blend(
            &UiTexture::Solid,
            UiBlend::Xor,
            (l as f32, t as f32, r as f32, b as f32),
            (0.0, 0.0, 1.0, 1.0),
            0xFF00_00FF,
        );
    }

    /// A solid white rectangle in the crosshair's inverting blend ([`UiBlend::Invert`]): every
    /// pixel under it becomes `1 - dst`, exactly what a white texel of the vanilla crosshair
    /// sprite does. Used by the Crosshair mod. Corners may be given in either
    /// order.
    pub fn rect_invert(&mut self, left: i32, top: i32, right: i32, bottom: i32) {
        let (l, r) = (left.min(right), left.max(right));
        let (t, b) = (top.min(bottom), top.max(bottom));
        self.quad_rect_with_blend(
            &UiTexture::Solid,
            UiBlend::Invert,
            (l as f32, t as f32, r as f32, b as f32),
            (0.0, 0.0, 1.0, 1.0),
            0xFFFF_FFFF,
        );
    }

    /// A one-pixel horizontal line from `start_x` to `end_x` **inclusive**.
    // java: Gui.java:20 (drawHorizontalLine)
    pub fn horizontal_line(&mut self, start_x: i32, end_x: i32, y: i32, argb: u32) {
        let (a, b) = (start_x.min(end_x), start_x.max(end_x));
        self.rect(a, y, b + 1, y + 1, argb);
    }

    /// A one-pixel vertical line.
    ///
    /// Vanilla's version is asymmetric on purpose: it draws `startY + 1 .. endY`, so the line is
    /// one pixel *shorter* than the horizontal version for the same span. Reproduced.
    // java: Gui.java:35 (drawVerticalLine)
    pub fn vertical_line(&mut self, x: i32, start_y: i32, end_y: i32, argb: u32) {
        let (a, b) = (start_y.min(end_y), start_y.max(end_y));
        self.rect(x, a + 1, x + 1, b, argb);
    }

    /// A vertically interpolated rectangle.
    // java: Gui.java:90 (drawGradientRect)
    pub fn gradient_rect(
        &mut self,
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
        top_argb: u32,
        bottom_argb: u32,
    ) {
        let t = unpack_argb(top_argb);
        let b = unpack_argb(bottom_argb);
        self.quad(
            &UiTexture::Solid,
            [
                [left as f32, top as f32],
                [left as f32, bottom as f32],
                [right as f32, top as f32],
                [right as f32, bottom as f32],
            ],
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            [t, b, t, b],
        );
    }

    /// Vanilla's `drawTexturedModalRect`: blit `width`×`height` texels from `(u, v)` of a
    /// 256×256 texture.
    // java: Gui.java:138
    #[allow(clippy::too_many_arguments)]
    pub fn blit(
        &mut self,
        texture: &UiTexture,
        x: i32,
        y: i32,
        u: i32,
        v: i32,
        width: i32,
        height: i32,
    ) {
        self.blit_sized(
            texture,
            x,
            y,
            u as f32,
            v as f32,
            width,
            height,
            GUI_TEXTURE_SIZE,
            GUI_TEXTURE_SIZE,
        );
    }

    /// Draw the 1.8.9 inverting crosshair sprite.
    ///
    /// This intentionally is not a general toggle on [`Self::blit`]: vanilla changes the GL
    /// blend state only around `GuiIngame`'s crosshair call, and normal `icons.png` sprites such
    /// as hearts must remain ordinary alpha blended.
    ///
    /// The same call runs `enableAlpha()` under the `alphaFunc(GL_GREATER, 0.1F)` that
    /// `EntityRenderer` set before `renderGameOverlay`, so the batch is alpha-tested
    /// ([`Self::with_alpha_test`]). The inverting blend ignores source alpha, so without the test
    /// every transparent texel that stores white RGB turned the scene under it to `1 - dst`. The
    /// vanilla sprite's clear texels are black and hid that; a user's resource pack has three
    /// clear-white texels diagonal to the centre (B2, 2026-11-07).
    // java: GuiIngame.java:175-180, EntityRenderer.java:1168
    #[allow(clippy::too_many_arguments)]
    pub fn blit_invert(
        &mut self,
        texture: &UiTexture,
        x: i32,
        y: i32,
        u: i32,
        v: i32,
        width: i32,
        height: i32,
    ) {
        let fu = 1.0 / GUI_TEXTURE_SIZE;
        let fv = 1.0 / GUI_TEXTURE_SIZE;
        self.with_alpha_test(|d| {
            d.quad_rect_with_blend(
                texture,
                UiBlend::Invert,
                (x as f32, y as f32, (x + width) as f32, (y + height) as f32),
                (
                    u as f32 * fu,
                    v as f32 * fv,
                    (u + width) as f32 * fu,
                    (v + height) as f32 * fv,
                ),
                0xFFFF_FFFF,
            );
        });
    }

    /// Vanilla's `drawModalRectWithCustomSizedTexture`, tinted.
    // java: Gui.java:187
    #[allow(clippy::too_many_arguments)]
    pub fn blit_sized(
        &mut self,
        texture: &UiTexture,
        x: i32,
        y: i32,
        u: f32,
        v: f32,
        width: i32,
        height: i32,
        tex_width: f32,
        tex_height: f32,
    ) {
        self.blit_tinted(
            texture,
            x,
            y,
            u,
            v,
            width,
            height,
            tex_width,
            tex_height,
            0xFFFF_FFFF,
        );
    }

    /// Vanilla's `drawScaledCustomSizeModalRect`: scale an arbitrary signed source rectangle into
    /// an independently sized destination rectangle.
    ///
    /// Unlike [`Self::blit_sized`], `source_width`/`source_height` control only the UV extent and
    /// `width`/`height` control only screen geometry. A negative source extent is intentional and
    /// flips that texture axis; vanilla uses `source_height = -8` for Dinnerbone/Grumm heads.
    // java: Gui.java:204-218
    #[allow(clippy::too_many_arguments)]
    pub fn blit_scaled(
        &mut self,
        texture: &UiTexture,
        x: i32,
        y: i32,
        u: f32,
        v: f32,
        source_width: i32,
        source_height: i32,
        width: i32,
        height: i32,
        texture_width: f32,
        texture_height: f32,
    ) {
        let fu = 1.0 / texture_width;
        let fv = 1.0 / texture_height;
        self.quad_rect(
            texture,
            (x as f32, y as f32, (x + width) as f32, (y + height) as f32),
            (
                u * fu,
                v * fv,
                (u + source_width as f32) * fu,
                (v + source_height as f32) * fv,
            ),
            0xFFFF_FFFF,
        );
    }

    /// [`blit_sized`](Self::blit_sized) with a colour multiplier.
    #[allow(clippy::too_many_arguments)]
    pub fn blit_tinted(
        &mut self,
        texture: &UiTexture,
        x: i32,
        y: i32,
        u: f32,
        v: f32,
        width: i32,
        height: i32,
        tex_width: f32,
        tex_height: f32,
        argb: u32,
    ) {
        let fu = 1.0 / tex_width;
        let fv = 1.0 / tex_height;
        self.quad_rect(
            texture,
            (x as f32, y as f32, (x + width) as f32, (y + height) as f32),
            (
                u * fu,
                v * fv,
                (u + width as f32) * fu,
                (v + height as f32) * fv,
            ),
            argb,
        );
    }

    /// Blit a stitched atlas [`Sprite`] into a rectangle.
    ///
    /// The sprite already knows its own UV window in the atlas, so this is how block and item
    /// icons reach the GUI. `texture` names whichever atlas the sprite was stitched into.
    // java: Gui.java:172 (drawTexturedModalRect(int,int,TextureAtlasSprite,int,int))
    pub fn blit_sprite(
        &mut self,
        texture: &UiTexture,
        sprite: &Sprite,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
    ) {
        self.quad_rect(
            texture,
            (x as f32, y as f32, (x + width) as f32, (y + height) as f32),
            (
                sprite.uv.min_u,
                sprite.uv.min_v,
                sprite.uv.max_u,
                sprite.uv.max_v,
            ),
            0xFFFF_FFFF,
        );
    }

    /// Tile a texture across a rectangle at 1:1 texel scale.
    ///
    /// This is how vanilla's dirt background works: the UV range is the *pixel* extent divided
    /// by the tile size, and the sampler repeats.
    // java: GuiScreen.java:688 (drawBackground) — `f = 32.0F`, colour 0x404040
    pub fn tiled(&mut self, texture: &UiTexture, rect: Rect, tile: f32, scroll_v: f32, argb: u32) {
        self.quad_rect(
            texture,
            (
                rect.x as f32,
                rect.y as f32,
                rect.right() as f32,
                rect.bottom() as f32,
            ),
            (
                rect.x as f32 / tile,
                rect.y as f32 / tile + scroll_v,
                rect.right() as f32 / tile,
                rect.bottom() as f32 / tile + scroll_v,
            ),
            argb,
        );
    }

    /// Nine-slice a source rectangle into a destination rectangle.
    ///
    /// The four corners keep their texel size, the four edges stretch along one axis, and the
    /// centre stretches along both. 1.8.9's own widget art predates nine-slice — a button is
    /// drawn as two overlapping halves, see [`blit_button`](Self::blit_button) — but every
    /// panel-shaped piece of GUI art in the game is a nine-slice in disguise, and screens built
    /// on top of S15 want the general form.
    ///
    /// Degenerate inputs (borders wider than the destination) are clamped rather than rejected,
    /// so this never produces inside-out quads.
    #[allow(clippy::too_many_arguments)]
    pub fn nine_slice(
        &mut self,
        texture: &UiTexture,
        dest: Rect,
        src: Rect,
        border: Border,
        tex_width: f32,
        tex_height: f32,
    ) {
        let b = border.clamped(dest, src);
        // Column edges in destination and source space.
        let dx = [
            dest.x,
            dest.x + b.left,
            dest.right() - b.right,
            dest.right(),
        ];
        let dy = [
            dest.y,
            dest.y + b.top,
            dest.bottom() - b.bottom,
            dest.bottom(),
        ];
        let sx = [src.x, src.x + b.left, src.right() - b.right, src.right()];
        let sy = [src.y, src.y + b.top, src.bottom() - b.bottom, src.bottom()];

        for row in 0..3 {
            for col in 0..3 {
                let (dw, dh) = (dx[col + 1] - dx[col], dy[row + 1] - dy[row]);
                let (sw, sh) = (sx[col + 1] - sx[col], sy[row + 1] - sy[row]);
                if dw <= 0 || dh <= 0 || sw <= 0 || sh <= 0 {
                    continue;
                }
                self.quad_rect(
                    texture,
                    (
                        dx[col] as f32,
                        dy[row] as f32,
                        dx[col + 1] as f32,
                        dy[row + 1] as f32,
                    ),
                    (
                        sx[col] as f32 / tex_width,
                        sy[row] as f32 / tex_height,
                        sx[col + 1] as f32 / tex_width,
                        sy[row + 1] as f32 / tex_height,
                    ),
                    0xFFFF_FFFF,
                );
            }
        }
    }

    /// Vanilla's button art: the left half of the 200-wide sprite, then the right half, so a
    /// button of any width keeps both rounded ends and stretches nothing.
    ///
    /// `state` is 0 disabled, 1 idle, 2 hovered — the three 20-pixel rows starting at v=46.
    // java: GuiButton.java:90
    pub fn blit_button(&mut self, texture: &UiTexture, rect: Rect, state: u32) {
        let v = 46 + (state as i32) * 20;
        let half = rect.width / 2;
        self.blit(texture, rect.x, rect.y, 0, v, half, rect.height);
        self.blit(
            texture,
            rect.x + half,
            rect.y,
            200 - half,
            v,
            half,
            rect.height,
        );
    }
}

/// Nine-slice border widths in texels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Border {
    /// Left border width.
    pub left: i32,
    /// Right border width.
    pub right: i32,
    /// Top border height.
    pub top: i32,
    /// Bottom border height.
    pub bottom: i32,
}

impl Border {
    /// The same width on all four sides.
    #[must_use]
    pub const fn uniform(n: i32) -> Self {
        Self {
            left: n,
            right: n,
            top: n,
            bottom: n,
        }
    }

    /// Shrink the borders until they fit both rectangles, so the middle band is never negative.
    #[must_use]
    fn clamped(self, dest: Rect, src: Rect) -> Self {
        let fit = |a: i32, b: i32, limit: i32| -> (i32, i32) {
            let (a, b) = (a.max(0), b.max(0));
            let total = a + b;
            if total <= limit || total == 0 {
                (a, b)
            } else {
                // Keep the ratio; integer division biases the left/top side, as vanilla's own
                // halving does.
                let a2 = a * limit / total;
                (a2, limit - a2)
            }
        };
        let w = dest.width.min(src.width).max(0);
        let h = dest.height.min(src.height).max(0);
        let (left, right) = fit(self.left, self.right, w);
        let (top, bottom) = fit(self.top, self.bottom, h);
        Self {
            left,
            right,
            top,
            bottom,
        }
    }
}

impl TextSink for DrawList {
    fn glyph(&mut self, q: GlyphQuad) {
        let c = unpack_argb(q.color);
        self.quad(&UiTexture::Font(q.texture), q.pos, q.uv, [c; 4]);
    }

    fn bar(&mut self, r: SolidRect) {
        self.quad_rect(
            &UiTexture::Solid,
            (r.x0, r.y0, r.x1, r.y1),
            (0.0, 0.0, 1.0, 1.0),
            r.color,
        );
    }
}

/// A [`TextSink`] for text vanilla draws while `GL_BLEND` is disabled: [`UiBlend::Disabled`]
/// under the `alphaFunc(GL_GREATER, 0.1F)` test that `FontRenderer.drawString` always enables.
///
/// java: `GuiIngame.renderScoreboard` draws its `0x20FFFFFF` text right after `Gui.drawRect`,
/// which ends with `disableBlend()`, and `FontRenderer` never re-enables it. A glyph texel's
/// fragment alpha is `1.0 * 0.125 > 0.1`, so it passes and lands opaque; a clear texel (`0.0`)
/// is discarded. Through the ordinary [`DrawList`] sink the same colour blended at 12.5%.
pub struct UnblendedText<'a>(pub &'a mut DrawList);

impl TextSink for UnblendedText<'_> {
    fn glyph(&mut self, q: GlyphQuad) {
        let c = unpack_argb(q.color);
        self.0.with_alpha_test(|d| {
            d.quad_with_blend(
                &UiTexture::Font(q.texture),
                UiBlend::Disabled,
                q.pos,
                q.uv,
                [c; 4],
            );
        });
    }

    fn bar(&mut self, r: SolidRect) {
        self.0.with_alpha_test(|d| {
            d.quad_rect_with_blend(
                &UiTexture::Solid,
                UiBlend::Disabled,
                (r.x0, r.y0, r.x1, r.y1),
                (0.0, 0.0, 1.0, 1.0),
                r.color,
            );
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::test_support::{fnv1a, uniform_font};

    #[test]
    fn opacity_scope_preserves_previous_vertices_and_multiplies_nested_scopes() {
        let mut d = DrawList::new();
        d.rect(0, 0, 4, 4, 0xFFFF_FFFF);
        let old = d.vertices().to_vec();
        d.with_opacity(0.5, |d| {
            d.rect(4, 0, 8, 4, 0xFFFF_FFFF);
            d.with_opacity(0.5, |d| d.rect(8, 0, 12, 4, 0xFFFF_FFFF));
        });
        assert_eq!(&d.vertices()[..old.len()], old);
        assert!(d.vertices()[old.len()..2 * old.len()]
            .iter()
            .all(|v| v.color[3] == 0.5));
        assert!(d.vertices()[2 * old.len()..]
            .iter()
            .all(|v| v.color[3] == 0.25));
    }

    /// Every drawing entry point on [`DrawList`], exercised once, including the degenerate and
    /// negative-coordinate cases that a careless "just multiply by the identity" would perturb.
    ///
    /// Uses only API that existed before the transform stack was added, so this exact function
    /// can be — and was — run against the previous tree to capture the goldens below.
    fn canonical_scene() -> DrawList {
        let mut d = DrawList::new();
        draw_canonical_scene(&mut d);
        d
    }

    fn draw_canonical_scene(d: &mut DrawList) {
        let widgets = UiTexture::gui(WIDGETS_TEXTURE);
        let icons = UiTexture::gui(ICONS_TEXTURE);
        let dirt = UiTexture::gui(OPTIONS_BACKGROUND);

        d.rect(0, 0, 320, 240, 0xC010_1010);
        d.rect(17, -3, 5, 200, 0x8012_3456); // corners the wrong way round
        d.horizontal_line(2, 5, 1, 0xFFFF_FFFF);
        d.vertical_line(1, 2, 5, 0xFF01_0203);
        d.gradient_rect(10, 20, 110, 60, 0xFFFF_0000, 0x0000_00FF);
        d.blit(&widgets, 100, 50, 0, 46, 200, 20);
        d.blit_sized(&icons, 3, 7, 16.0, 32.0, 9, 9, 256.0, 256.0);
        d.blit_tinted(&icons, 4, 8, 0.5, 1.5, 9, 9, 128.0, 64.0, 0x8033_6699);
        d.tiled(&dirt, Rect::new(-4, 5, 300, 40), 32.0, 1.5, 0xFF40_4040);
        d.nine_slice(
            &widgets,
            Rect::new(0, 0, 100, 60),
            Rect::new(0, 0, 16, 16),
            Border::uniform(4),
            256.0,
            256.0,
        );
        d.nine_slice(
            &widgets,
            Rect::new(-3, 2, 5, 5),
            Rect::new(0, 0, 16, 16),
            Border {
                left: 7,
                right: 3,
                top: 2,
                bottom: 9,
            },
            256.0,
            256.0,
        );
        d.blit_button(&widgets, Rect::new(0, 0, 151, 20), 1);
        d.quad(
            &UiTexture::Solid,
            [[0.5, -1.5], [0.5, 2.5], [3.5, -1.5], [3.5, 2.5]],
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            [[0.1, 0.2, 0.3, 0.4]; 4],
        );
        d.quad_rect(
            &icons,
            (-0.5, 0.25, 12.75, 30.0),
            (0.0, 0.125, 0.5, 1.0),
            0x1234_5678,
        );
        // Signed zeros: `-0.0` and `0.0` have different bit patterns, and `1.0 * -0.0 + 0.0 * y`
        // would quietly turn the former into the latter. This is the canary for that.
        d.quad_rect(
            &UiTexture::Solid,
            (-0.0, -0.0, 4.0, 4.0),
            (0.0, 0.0, 1.0, 1.0),
            0xFF00_FF00,
        );

        let mut font = uniform_font();
        font.draw_string_with_shadow(d, "Hello \u{a7}lWorld\u{a7}r!", 12.0, 34.0, 0x00FF_FFFF);
        font.draw_string(
            d,
            "\u{a7}nunderline\u{a7}m/strike",
            5.0,
            60.0,
            0xFFAA_BBCCu32 as i32,
            false,
        );
    }

    /// One number per list, covering both the vertex bytes and the batch split.
    fn scene_fingerprint(d: &DrawList) -> (usize, usize, u64, u64) {
        let batches = d
            .batches()
            .iter()
            .map(|b| format!("{:?}:{}..{}", b.texture, b.range.start, b.range.end))
            .collect::<Vec<_>>()
            .join("|");
        (
            d.vertices().len(),
            d.batches().len(),
            fnv1a(&d.vertex_bytes()),
            fnv1a(batches.as_bytes()),
        )
    }

    /// **The load-bearing regression test of the transform amendment.**
    ///
    /// `(vertices, batches, FNV-1a of the vertex bytes, FNV-1a of the batch split)`, measured by
    /// running this exact function against the tree with [`DrawList::push`]'s transform lookup
    /// removed — i.e. against `mc-ui` as S16 and S17 built on it. Their screens and the ~500
    /// tests over them depend on this geometry, so a call with nothing pushed has to keep
    /// emitting the *identical* bytes, not merely equivalent ones.
    #[test]
    fn an_untransformed_drawlist_is_byte_identical_to_the_pre_transform_tree() {
        assert_eq!(
            scene_fingerprint(&canonical_scene()),
            (
                576,
                40,
                // Glyph UVs start `ASCII_UV_INSET`
                // into their cell, so only the vertex bytes moved; positions, counts and batches
                // are unchanged. With both insets set to 0 this is the old 8_529_985_026_101_681_830
                // again.
                499_248_697_313_974_749,
                5_808_749_226_640_554_780,
            ),
        );
    }

    /// The same scene under a pushed-then-popped transform: the transform must reach the
    /// vertices *while* it is pushed, and the list must be back to producing pristine bytes
    /// after the pop.
    /// The other half of byte-identity: a screen that *does* use the stack must leave the list
    /// producing pristine geometry for everything it draws afterwards.
    #[test]
    fn a_balanced_push_and_pop_leaves_later_draws_byte_identical() {
        let mut d = DrawList::new();
        d.push_transform(Transform::translate(11.0, 13.0).then_rotate_degrees(-20.0));
        d.push_transform(Transform::scale(2.5));
        d.rect(0, 0, 3, 3, 0xFFFF_FFFF);
        d.pop_transform();
        d.pop_transform();
        assert_eq!(d.transform_depth(), 0);
        assert_eq!(d.transform(), Transform::IDENTITY);

        let prefix = d.vertices().len();
        draw_canonical_scene(&mut d);
        let pristine = canonical_scene();
        assert_eq!(&d.vertices()[prefix..], pristine.vertices());
        assert_eq!(
            fnv1a(&d.vertex_bytes()[prefix * UiVertex::SIZE..]),
            fnv1a(&pristine.vertex_bytes()),
        );
    }

    #[test]
    fn a_transform_moves_positions_and_nothing_else() {
        let tex = UiTexture::gui(WIDGETS_TEXTURE);
        let mut plain = DrawList::new();
        plain.blit(&tex, 10, 20, 0, 46, 200, 20);

        let mut moved = DrawList::new();
        moved.push_transform(Transform::translate(5.0, -7.0));
        moved.blit(&tex, 10, 20, 0, 46, 200, 20);
        moved.pop_transform();

        assert_eq!(moved.batches(), plain.batches(), "batching is untouched");
        for (a, b) in moved.vertices().iter().zip(plain.vertices()) {
            assert_eq!(a.uv, b.uv, "UVs are untouched");
            assert_eq!(a.color, b.color, "colours are untouched");
            assert_eq!(a.pos, [b.pos[0] + 5.0, b.pos[1] - 7.0]);
        }
    }

    #[test]
    fn nested_pushes_compose_and_popping_unwinds_exactly_one_level() {
        let mut d = DrawList::new();
        d.push_transform(Transform::translate(200.0, 120.0));
        d.push_transform(Transform::scale(4.0));
        d.rect(0, 0, 1, 1, 0xFFFF_FFFF);
        assert_eq!(d.transform_depth(), 2);
        assert_eq!(d.vertices()[0].pos, [200.0, 120.0]);
        assert_eq!(
            d.vertices()
                .iter()
                .map(|v| v.pos[0])
                .fold(f32::MIN, f32::max),
            204.0,
            "one GUI pixel became four"
        );
        d.pop_transform();

        // Back at the outer translate, a second inner scale is independent of the first.
        d.push_transform(Transform::scale(2.0));
        d.rect(0, 0, 1, 1, 0xFF00_FF00);
        let second = &d.vertices()[6..12];
        assert_eq!(second[0].pos, [200.0, 120.0]);
        assert_eq!(
            second.iter().map(|v| v.pos[0]).fold(f32::MIN, f32::max),
            202.0
        );
        d.pop_transform();
        d.pop_transform();
        assert_eq!(d.transform_depth(), 0);
    }

    /// GL raises `GL_STACK_UNDERFLOW` and leaves the matrix alone; so do we.
    #[test]
    fn popping_an_empty_stack_is_a_silent_no_op() {
        let mut d = DrawList::new();
        d.pop_transform();
        d.pop_transform();
        assert_eq!(d.transform_depth(), 0);
        d.rect(0, 0, 1, 1, 0xFFFF_FFFF);
        assert_eq!(d.vertices()[0].pos, [0.0, 0.0]);
    }

    #[test]
    fn clear_drops_an_unbalanced_stack_so_the_next_frame_starts_clean() {
        let mut d = DrawList::new();
        d.push_transform(Transform::scale(9.0));
        d.rect(0, 0, 1, 1, 0xFFFF_FFFF);
        d.clear();
        assert_eq!(d.transform_depth(), 0);
        d.rect(0, 0, 1, 1, 0xFFFF_FFFF);
        assert_eq!(d.vertices()[0].pos, [0.0, 0.0]);
    }

    #[test]
    fn builtin_tile_descriptors_deduplicate_and_clear_with_the_frame() {
        let normal = BuiltinTileItem::Chest { trapped: false };
        let trapped = BuiltinTileItem::Chest { trapped: true };
        let mut d = DrawList::new();
        d.register_builtin_tile_item(normal.clone());
        d.register_builtin_tile_item(normal.clone());
        d.register_builtin_tile_item(trapped.clone());
        assert_eq!(d.builtin_tile_items(), &[normal, trapped]);

        d.clear();
        assert!(d.builtin_tile_items().is_empty());
        assert!(d.batches().is_empty());
        assert!(d.vertices().is_empty());
    }

    #[test]
    fn with_transform_pushes_and_pops_around_the_closure() {
        let mut d = DrawList::new();
        let n = d.with_transform(Transform::translate(50.0, 60.0), |d| {
            d.rect(0, 0, 1, 1, 0xFFFF_FFFF);
            d.vertices().len()
        });
        assert_eq!(n, 6);
        assert_eq!(d.transform_depth(), 0);
        assert_eq!(d.vertices()[0].pos, [50.0, 60.0]);
    }

    #[test]
    fn gui_render_state_is_scoped_and_depth_values_follow_the_state_window() {
        let texture = UiTexture::Solid;
        let triangle = [[0.0, 0.0], [0.0, 2.0], [2.0, 0.0]];
        let uv = [[0.0, 0.0]; 3];
        let color = [[1.0; 4]; 3];
        let mut d = DrawList::new();

        // Ordinary GUI work stays overlay/Back before and after the item bracket.
        d.rect(10, 10, 12, 12, 0xFFFF_FFFF);
        d.with_render_state(UiDepth::Item, UiCull::Back, |d| {
            d.triangle_with_blend_depth(&texture, UiBlend::Alpha, [0.6; 3], triangle, uv, color);
            d.with_render_state(UiDepth::Equal, UiCull::None, |d| {
                d.triangle_with_blend_depth(
                    &texture,
                    UiBlend::Glint,
                    [0.6; 3],
                    triangle,
                    uv,
                    color,
                );
            });
        });
        d.rect(20, 20, 22, 22, 0xFFFF_FFFF);

        assert_eq!(
            d.batches()
                .iter()
                .map(|batch| (batch.depth, batch.cull, batch.blend))
                .collect::<Vec<_>>(),
            vec![
                (UiDepth::Overlay, UiCull::Back, UiBlend::Alpha),
                (UiDepth::Item, UiCull::Back, UiBlend::Alpha),
                (UiDepth::Equal, UiCull::None, UiBlend::Glint),
                (UiDepth::Overlay, UiCull::Back, UiBlend::Alpha),
            ]
        );
        assert_eq!(d.depths()[..6], [0.0; 6]);
        assert_eq!(d.depths()[6..9], [0.6; 3]);
        assert_eq!(d.depths()[9..12], [0.6; 3]);
        assert_eq!(d.depths()[12..], [0.0; 6]);

        // The closure restored both state axes; this final draw must merge with neither item
        // batch nor accidentally inherit cull-disabled/player state.
        d.with_render_state(UiDepth::Item, UiCull::Back, |d| {
            d.triangle_with_blend_depth(&texture, UiBlend::Alpha, [0.4; 3], triangle, uv, color);
        });
        assert_eq!(d.batches().last().unwrap().depth, UiDepth::Item);
        assert_eq!(d.batches().last().unwrap().cull, UiCull::Back);
    }

    /// Hazard **H9**, restated for the transform stack: a scaled string has to be a scaled
    /// *picture* of the unscaled one. If the transform ever leaked back into
    /// [`crate::font::FontRenderer`]'s pen, glyph advances would be re-rounded at the scaled
    /// size and the text would drift a pixel per character — the exact failure H9 exists for.
    ///
    /// `4.0` and `0.5` are the two scales vanilla actually uses (`/title` and `chatScale`), and
    /// both are exact in binary, so this asserts equality rather than closeness.
    // java: GuiIngame.java:303 (scale(4.0F)) / GuiNewChat.java:48 (scale(chatScale))
    #[test]
    fn a_scaled_string_is_an_exactly_scaled_picture_of_the_unscaled_one() {
        for scale in [4.0f32, 2.0, 0.5] {
            let text = "\u{a7}lTitle\u{a7}r Wij 1.8.9";
            let mut font = uniform_font();

            let mut plain = DrawList::new();
            let plain_end = font.draw_string_with_shadow(&mut plain, text, 0.0, 0.0, 0x00FF_FFFF);

            let mut big = DrawList::new();
            big.push_transform(Transform::scale(scale));
            let big_end = font.draw_string_with_shadow(&mut big, text, 0.0, 0.0, 0x00FF_FFFF);
            big.pop_transform();

            assert_eq!(
                big_end, plain_end,
                "the returned pen position is measured in unscaled pixels, as vanilla's is"
            );
            assert_eq!(big.vertices().len(), plain.vertices().len());
            assert_eq!(big.batches(), plain.batches());
            for (i, (a, b)) in big.vertices().iter().zip(plain.vertices()).enumerate() {
                assert_eq!(
                    a.pos,
                    [b.pos[0] * scale, b.pos[1] * scale],
                    "vertex {i} at scale {scale}"
                );
                assert_eq!(a.uv, b.uv, "vertex {i}: the glyph is the same glyph");
            }
        }
    }

    /// The main-menu splash, transcribed from the Java call for call.
    // java: GuiMainMenu.java:560
    #[test]
    fn the_main_menu_splash_anchors_at_the_wordmark_and_tilts_up_to_the_right() {
        let width = 427;
        let mut font = uniform_font();
        let splash = "Also try Minecraft 2!";
        // f = 1.8 - abs(sin(...)*0.1) at the peak, then f * 100 / (stringWidth + 32).
        let f = 1.8f32 * 100.0 / (font.string_width(splash) + 32) as f32;

        let mut d = DrawList::new();
        d.push_transform(
            Transform::translate((width / 2 + 90) as f32, 70.0)
                .then_rotate_degrees(-20.0)
                .then_scale(f),
        );
        // -256 is vanilla's literal for 0xFFFFFF00.
        crate::widget::draw_centered_string(&mut d, &mut font, splash, 0, -8, -256);
        d.pop_transform();

        assert!(!d.is_empty());
        let xs: Vec<f32> = d.vertices().iter().map(|v| v.pos[0]).collect();
        let ys: Vec<f32> = d.vertices().iter().map(|v| v.pos[1]).collect();
        let (min_x, max_x) = (
            xs.iter().copied().fold(f32::MAX, f32::min),
            xs.iter().copied().fold(f32::MIN, f32::max),
        );
        // The anchor is (213 + 90, 70) = (303, 70), and the string is centred on it.
        let mid = (min_x + max_x) / 2.0;
        assert!((mid - 303.0).abs() < 6.0, "centred near x=303, got {mid}");
        // The tilt: the left end of the string sits lower on screen than the right end.
        let leftmost = ys[xs.iter().position(|x| *x == min_x).unwrap()];
        let rightmost = ys[xs.iter().position(|x| *x == max_x).unwrap()];
        assert!(
            rightmost < leftmost,
            "the splash should rise to the right: {leftmost} -> {rightmost}"
        );
        // At the anchor's own column the string sits just above y = 70, because
        // `drawCenteredString(..., 0, -8, ...)` puts its baseline eight local pixels up.
        let near_anchor: Vec<f32> = d
            .vertices()
            .iter()
            .filter(|v| (v.pos[0] - 303.0).abs() < 4.0)
            .map(|v| v.pos[1])
            .collect();
        assert!(
            !near_anchor.is_empty(),
            "nothing drawn at the anchor column"
        );
        assert!(
            near_anchor.iter().all(|y| (55.0..73.0).contains(y)),
            "{near_anchor:?}"
        );
    }

    /// The chat box's own transform, and the thing a screen needs it for: mapping a cursor
    /// position back into the space the chat lines were laid out in.
    // java: GuiNewChat.java:46 (drawChat) / :243 (getChatComponent)
    #[test]
    fn the_chat_transform_is_invertible_so_a_click_can_be_mapped_back() {
        let chat_scale = 0.5f32;
        let mut d = DrawList::new();
        d.push_transform(Transform::translate(2.0, 20.0).then_scale(chat_scale));
        let t = d.transform();
        d.rect(0, -9, 100, 0, 0x8000_0000);
        d.pop_transform();

        // The line's own rectangle, on screen.
        assert_eq!(d.vertices()[0].pos, [2.0, 15.5]);
        let inv = t.try_inverse().expect("a scale of 0.5 is invertible");
        // Vanilla's `(int)(mouseX / chatScale)` for a click at screen (52, 18).
        let local = inv.apply([52.0, 18.0]);
        assert_eq!(local, [100.0, -4.0]);
    }

    #[test]
    fn rect_hit_test_is_inclusive_top_left_exclusive_bottom_right() {
        let r = Rect::new(10, 20, 30, 40);
        assert_eq!(r.right(), 40);
        assert_eq!(r.bottom(), 60);
        assert!(r.contains(10, 20), "top-left corner is inside");
        assert!(r.contains(39, 59), "last pixel is inside");
        assert!(!r.contains(40, 59), "one past the right edge is outside");
        assert!(!r.contains(39, 60), "one past the bottom edge is outside");
        assert!(!r.contains(9, 20));
        assert!(!r.contains(10, 19));
        assert!(
            !Rect::new(0, 0, 0, 10).contains(0, 0),
            "a zero-width rect is empty"
        );
    }

    #[test]
    fn a_rect_is_two_triangles_in_one_batch() {
        let mut d = DrawList::new();
        d.rect(0, 0, 10, 5, 0xFF00_FF00);
        assert_eq!(d.vertices().len(), 6);
        assert_eq!(d.batches().len(), 1);
        assert_eq!(d.batches()[0].texture, UiTexture::Solid);
        assert_eq!(d.batches()[0].range, 0..6);
        assert_eq!(d.vertices()[0].pos, [0.0, 0.0]);
        assert_eq!(d.vertices()[0].color, [0.0, 1.0, 0.0, 1.0]);
        // The six vertices cover the quad: two corners appear twice.
        let corners: std::collections::BTreeSet<_> = d
            .vertices()
            .iter()
            .map(|v| (v.pos[0] as i32, v.pos[1] as i32))
            .collect();
        assert_eq!(corners.len(), 4);
    }

    #[test]
    fn text_selection_or_reverse_is_two_masked_opaque_batches() {
        let mut d = DrawList::new();
        d.rect(0, 0, 1, 1, 0xFFFF_FFFF);
        d.rect_xor_blue(2, 3, 7, 11);

        assert_eq!(d.batches().len(), 3);
        assert_eq!(d.batches()[0].blend, UiBlend::Alpha);
        assert_eq!(d.batches()[1].texture, UiTexture::Solid);
        assert_eq!(d.batches()[1].blend, UiBlend::XorComplement);
        assert_eq!(d.batches()[2].texture, UiTexture::Solid);
        assert_eq!(d.batches()[2].blend, UiBlend::Xor);
        assert_eq!(d.vertices()[6].pos, [2.0, 3.0]);
        assert_eq!(d.vertices()[6].color, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(d.vertices()[12].pos, [2.0, 3.0]);
        assert_eq!(d.vertices()[12].color, [0.0, 0.0, 1.0, 1.0]);
    }

    /// Apply the two CPU-visible batch descriptors to one RGBA8 destination sample.  The
    /// renderer's masked `Subtract` + `REPLACE` pipelines are the fixed-palette realization of
    /// Java's `src | !dst` opcode 5387 (`GL_OR_REVERSE`).
    fn apply_selection_or_reverse(mut dst: [u8; 4]) -> [u8; 4] {
        dst[0] = 255 - dst[0];
        dst[1] = 255 - dst[1];
        dst[2] = 255;
        dst[3] = 255;
        dst
    }

    /// The old implementation's 50%-alpha blue rectangle.  It is deliberately retained only as
    /// a negative control: this is the tempting mutant that used to be emitted by `TextField`.
    fn alpha_blue_mutant(dst: [u8; 4]) -> [u8; 4] {
        let blend = |src: u8, dst: u8| {
            u8::try_from((u32::from(src) * 128 + u32::from(dst) * 127 + 127) / 255).unwrap()
        };
        [
            blend(0, dst[0]),
            blend(0, dst[1]),
            blend(255, dst[2]),
            blend(255, dst[3]),
        ]
    }

    /// The previous single-pass XOR-shaped mutant preserved red/green and complemented blue/
    /// alpha.  It is also useful as a negative control because a black/white-only sample would
    /// otherwise make the two different logic operators look deceptively similar.
    fn xor_mutant(dst: [u8; 4]) -> [u8; 4] {
        [dst[0], dst[1], 255 - dst[2], 255 - dst[3]]
    }

    #[test]
    fn text_selection_or_reverse_bites_alpha_and_xor_mutants() {
        // These samples cover black, white, mixed mid-range channels, and transparent alpha.  In
        // particular, the mixed sample reaches every channel where OR_REVERSE differs from both
        // the former alpha veil and the prior XOR-shaped pipeline.
        for dst in [
            [0, 0, 0, 0],
            [255, 255, 255, 255],
            [0x12, 0x80, 0xFE, 0x40],
            [0x7F, 0x55, 0xAA, 0xC0],
        ] {
            let expected = [255 - dst[0], 255 - dst[1], 255, 255];
            assert_eq!(
                apply_selection_or_reverse(dst),
                expected,
                "destination {dst:?}"
            );
            assert_ne!(
                alpha_blue_mutant(dst),
                expected,
                "alpha mutant must bite {dst:?}"
            );
            assert_ne!(xor_mutant(dst), expected, "xor mutant must bite {dst:?}");
        }
    }

    #[test]
    fn inverted_rect_corners_are_normalised() {
        let mut a = DrawList::new();
        a.rect(10, 20, 0, 5, 0xFFFF_FFFF);
        let mut b = DrawList::new();
        b.rect(0, 5, 10, 20, 0xFFFF_FFFF);
        assert_eq!(a, b);
    }

    #[test]
    fn batches_merge_while_the_texture_is_unchanged() {
        let mut d = DrawList::new();
        d.rect(0, 0, 1, 1, 0xFF00_0000);
        d.rect(1, 1, 2, 2, 0xFF00_0000);
        assert_eq!(d.batches().len(), 1);
        assert_eq!(d.batches()[0].range, 0..12);

        d.blit(&UiTexture::gui(WIDGETS_TEXTURE), 0, 0, 0, 0, 4, 4);
        assert_eq!(d.batches().len(), 2);
        d.rect(0, 0, 1, 1, 0xFF00_0000);
        assert_eq!(d.batches().len(), 3, "switching back opens a third batch");
        assert_eq!(d.batches().iter().map(|b| b.range.len()).sum::<usize>(), 24);
    }

    #[test]
    fn blit_maps_texels_onto_a_two_fifty_six_texture() {
        let mut d = DrawList::new();
        d.blit(&UiTexture::gui(WIDGETS_TEXTURE), 100, 50, 0, 46, 200, 20);
        let v = d.vertices();
        assert_eq!(v[0].pos, [100.0, 50.0]);
        assert_eq!(v[0].uv, [0.0, 46.0 / 256.0]);
        // The bottom-right corner is the last vertex of the second triangle.
        let br = v
            .iter()
            .find(|v| v.pos == [300.0, 70.0])
            .expect("bottom-right");
        assert_eq!(br.uv, [200.0 / 256.0, 66.0 / 256.0]);
    }

    #[test]
    fn scaled_custom_blit_keeps_source_and_destination_extents_independent() {
        let texture = UiTexture::gui("textures/entity/example.png");
        let mut d = DrawList::new();
        d.blit_scaled(&texture, 10, 20, 32.0, 40.0, 8, 8, 12, 12, 64.0, 64.0);

        assert_eq!(d.vertices().len(), 6);
        assert_eq!(d.vertices()[0].pos, [10.0, 20.0]);
        assert_eq!(d.vertices()[0].uv, [32.0 / 64.0, 40.0 / 64.0]);
        let bottom_right = d
            .vertices()
            .iter()
            .find(|vertex| vertex.pos == [22.0, 32.0])
            .expect("scaled destination bottom-right");
        assert_eq!(bottom_right.uv, [40.0 / 64.0, 48.0 / 64.0]);

        // Mutation control: conflating the destination 12×12 with the source extent would end
        // at texel 44,52 rather than Java's u+8,v+8.
        assert_ne!(bottom_right.uv, [44.0 / 64.0, 52.0 / 64.0]);
    }

    #[test]
    fn scaled_custom_blit_preserves_negative_source_height_for_vertical_flip() {
        let texture = UiTexture::gui("textures/entity/head.png");
        let mut d = DrawList::new();
        d.blit_scaled(&texture, 3, 5, 8.0, 16.0, 8, -8, 12, 12, 64.0, 64.0);

        let top_left = d
            .vertices()
            .iter()
            .find(|vertex| vertex.pos == [3.0, 5.0])
            .expect("top-left");
        let bottom_left = d
            .vertices()
            .iter()
            .find(|vertex| vertex.pos == [3.0, 17.0])
            .expect("bottom-left");
        assert_eq!(top_left.uv, [8.0 / 64.0, 16.0 / 64.0]);
        assert_eq!(bottom_left.uv, [8.0 / 64.0, 8.0 / 64.0]);
        assert!(
            bottom_left.uv[1] < top_left.uv[1],
            "negative source height flips V without flipping destination geometry"
        );
    }

    #[test]
    fn gradient_rect_interpolates_top_to_bottom() {
        let mut d = DrawList::new();
        d.gradient_rect(0, 0, 10, 10, 0xFFFF_0000, 0xFF00_00FF);
        let top: Vec<_> = d
            .vertices()
            .iter()
            .filter(|v| v.pos[1] == 0.0)
            .map(|v| v.color)
            .collect();
        let bottom: Vec<_> = d
            .vertices()
            .iter()
            .filter(|v| v.pos[1] == 10.0)
            .map(|v| v.color)
            .collect();
        assert!(top.iter().all(|c| *c == [1.0, 0.0, 0.0, 1.0]));
        assert!(bottom.iter().all(|c| *c == [0.0, 0.0, 1.0, 1.0]));
    }

    #[test]
    fn lines_reproduce_vanillas_asymmetry() {
        let mut h = DrawList::new();
        h.horizontal_line(2, 5, 1, 0xFFFF_FFFF);
        let xs: Vec<f32> = h.vertices().iter().map(|v| v.pos[0]).collect();
        assert_eq!(xs.iter().cloned().fold(f32::MAX, f32::min), 2.0);
        assert_eq!(xs.iter().cloned().fold(f32::MIN, f32::max), 6.0, "endX + 1");

        let mut v = DrawList::new();
        v.vertical_line(1, 2, 5, 0xFFFF_FFFF);
        let ys: Vec<f32> = v.vertices().iter().map(|v| v.pos[1]).collect();
        assert_eq!(
            ys.iter().cloned().fold(f32::MAX, f32::min),
            3.0,
            "startY + 1"
        );
        assert_eq!(
            ys.iter().cloned().fold(f32::MIN, f32::max),
            5.0,
            "endY, not +1"
        );
    }

    #[test]
    fn a_button_is_two_halves_that_share_the_middle_texels() {
        let mut d = DrawList::new();
        d.blit_button(
            &UiTexture::gui(WIDGETS_TEXTURE),
            Rect::new(0, 0, 150, 20),
            2,
        );
        // 12 vertices: two blits of two triangles.
        assert_eq!(d.vertices().len(), 12);
        assert_eq!(
            d.batches().len(),
            1,
            "both halves share the widgets texture"
        );
        // Left half: texels u 0..75 of the idle+hover row at v = 86.
        assert_eq!(d.vertices()[0].uv, [0.0, 86.0 / 256.0]);
        // Right half starts at x = 75 and samples u = 200-75 = 125.
        let right_start = d.vertices()[6];
        assert_eq!(right_start.pos, [75.0, 0.0]);
        assert_eq!(right_start.uv, [125.0 / 256.0, 86.0 / 256.0]);
    }

    #[test]
    fn an_odd_width_button_loses_its_last_pixel_column_exactly_like_vanilla() {
        let mut d = DrawList::new();
        d.blit_button(
            &UiTexture::gui(WIDGETS_TEXTURE),
            Rect::new(0, 0, 151, 20),
            1,
        );
        // half = 75, so the two halves cover 0..150 and the 151st column is not drawn.
        let max_x = d
            .vertices()
            .iter()
            .map(|v| v.pos[0])
            .fold(f32::MIN, f32::max);
        assert_eq!(max_x, 150.0);
    }

    #[test]
    fn nine_slice_emits_nine_quads_with_fixed_corners() {
        let mut d = DrawList::new();
        d.nine_slice(
            &UiTexture::gui(WIDGETS_TEXTURE),
            Rect::new(0, 0, 100, 60),
            Rect::new(0, 0, 16, 16),
            Border::uniform(4),
            256.0,
            256.0,
        );
        assert_eq!(d.vertices().len(), 9 * 6);
        // Top-left corner keeps its 4x4 texel size.
        assert_eq!(d.vertices()[0].pos, [0.0, 0.0]);
        let tl_far = d.vertices()[..6]
            .iter()
            .map(|v| v.pos[0])
            .fold(f32::MIN, f32::max);
        assert_eq!(tl_far, 4.0);
        // The rightmost quad ends flush with the destination.
        let max_x = d
            .vertices()
            .iter()
            .map(|v| v.pos[0])
            .fold(f32::MIN, f32::max);
        assert_eq!(max_x, 100.0);
        let max_y = d
            .vertices()
            .iter()
            .map(|v| v.pos[1])
            .fold(f32::MIN, f32::max);
        assert_eq!(max_y, 60.0);
    }

    #[test]
    fn nine_slice_skips_degenerate_bands_instead_of_inverting_them() {
        let mut d = DrawList::new();
        // Destination narrower than the two 4-texel borders combined.
        d.nine_slice(
            &UiTexture::gui(WIDGETS_TEXTURE),
            Rect::new(0, 0, 5, 5),
            Rect::new(0, 0, 16, 16),
            Border::uniform(4),
            256.0,
            256.0,
        );
        for chunk in d.vertices().chunks(6) {
            let xs: Vec<f32> = chunk.iter().map(|v| v.pos[0]).collect();
            let ys: Vec<f32> = chunk.iter().map(|v| v.pos[1]).collect();
            assert!(xs.iter().cloned().fold(f32::MIN, f32::max) >= xs[0]);
            assert!(ys.iter().cloned().fold(f32::MIN, f32::max) >= ys[0]);
        }
        // No quad may leave the destination rectangle.
        assert!(d
            .vertices()
            .iter()
            .all(|v| v.pos[0] <= 5.0 && v.pos[1] <= 5.0));
    }

    #[test]
    fn vertex_bytes_are_thirty_two_per_vertex() {
        let mut d = DrawList::new();
        d.rect(0, 0, 1, 1, 0xFFFF_FFFF);
        let bytes = d.vertex_bytes();
        assert_eq!(bytes.len(), 6 * UiVertex::SIZE);
        assert_eq!(&bytes[..4], &0.0f32.to_le_bytes());
        assert_eq!(&bytes[16..20], &1.0f32.to_le_bytes(), "red channel");
    }

    /// R1-18: the array serializer and the retained-buffer upload path emit exactly the bytes
    /// of the original per-float path, including signed zeros, NaN payloads and subnormals,
    /// and fully replace whatever a reused buffer held before.
    #[test]
    fn retained_vertex_and_depth_bytes_match_the_per_float_serializer() {
        fn per_float(v: &UiVertex) -> Vec<u8> {
            v.pos
                .iter()
                .chain(&v.uv)
                .chain(&v.color)
                .flat_map(|f| f.to_le_bytes())
                .collect()
        }
        let specials = [
            0.0,
            -0.0,
            1.0,
            -1.5,
            f32::MIN_POSITIVE / 4.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7FC0_1234),
            f32::from_bits(0xFFA0_0001),
            1.0e-38,
            3.4e38,
        ];
        let mut d = DrawList::new();
        d.rect(0, 0, 3, 2, 0x80FF_4020);
        d.gradient_rect(1, 1, 7, 9, 0xFF00_0000, 0x2000_FF00);
        d.vertices
            .extend(specials.iter().enumerate().map(|(i, &f)| UiVertex {
                pos: [f, specials[(i + 1) % specials.len()]],
                uv: [specials[(i + 2) % specials.len()], -f],
                color: [f, 0.25, specials[(i + 5) % specials.len()], 1.0],
            }));
        d.depths.extend(
            specials
                .iter()
                .copied()
                .take(d.vertices.len() - d.depths.len()),
        );
        let expected: Vec<u8> = d.vertices().iter().flat_map(per_float).collect();
        let mut scratch = vec![0xAB; 3];
        d.write_vertex_bytes(&mut scratch);
        assert_eq!(scratch, expected);
        assert_eq!(d.vertex_bytes(), expected);
        let expected_depths: Vec<u8> = d.depths.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mut depth_scratch = vec![0xCD; 999];
        d.write_depth_bytes(&mut depth_scratch);
        assert_eq!(depth_scratch, expected_depths);
        assert_eq!(d.depth_bytes(), expected_depths);
    }

    #[test]
    fn unpack_argb_splits_the_channels() {
        assert_eq!(
            unpack_argb(0xFF80_4020),
            [128.0 / 255.0, 64.0 / 255.0, 32.0 / 255.0, 1.0]
        );
        assert_eq!(unpack_argb(0x0000_0000), [0.0; 4]);
    }

    /// The transform composes *inside* the GUI scale, checked on the real GPU rather than
    /// argued: at `guiScale = 2`, one GUI pixel under a pushed `scale(3)` has to reach the
    /// framebuffer as a 6×6 block, while an untransformed GUI pixel next to it stays 2×2.
    ///
    /// This is the [`crate::widget::ScaledResolution`] half of the amendment — the projection
    /// still maps GUI pixels to clip space and the pushed matrix sits under it, exactly as
    /// vanilla's modelview sits under `setupOverlayRendering`'s `glOrtho`.
    #[test]
    fn a_pushed_scale_composes_with_the_gui_scale_on_the_gpu() {
        use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

        const W: u32 = 640;
        const H: u32 = 480;
        let mut gpu = match Gpu::headless(
            W,
            H,
            GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(g) => g,
            Err(RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter available on this machine");
                return;
            }
            Err(e) => panic!("headless context creation failed: {e}"),
        };

        let mut phase = crate::widget::UiPhase::new(&gpu, W, H, 2, false);
        assert_eq!(phase.resolution().scale_factor(), 2);

        let mut list = DrawList::new();
        list.push_transform(Transform::translate(10.0, 20.0).then_scale(3.0));
        list.rect(0, 0, 1, 1, 0xFFFF_0000); // one GUI pixel, tripled -> 3 GUI px
        list.pop_transform();
        list.rect(40, 40, 41, 41, 0xFF00_FF00); // one GUI pixel, untransformed
        phase.submit(list);

        let mut phases = PhaseList::new();
        phases.clear_color = wgpu::Color::BLACK;
        phases.insert(PhaseOrder::UI, Box::new(phase));
        let frame = gpu.acquire().expect("acquire offscreen frame");
        phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
        let px = gpu.read_pixels().expect("readback");
        let at = |x: u32, y: u32| -> [u8; 4] {
            let i = ((y * W + x) * 4) as usize;
            [px[i], px[i + 1], px[i + 2], px[i + 3]]
        };

        // GUI (10, 20)..(13, 23) at guiScale 2 is device (20, 40)..(26, 46).
        for (x, y) in [(20, 40), (25, 45), (22, 43)] {
            assert_eq!(at(x, y), [255, 0, 0, 255], "({x}, {y}) is inside the block");
        }
        assert_eq!(
            at(26, 40),
            [0, 0, 0, 255],
            "and the block is exactly 6 wide"
        );
        assert_eq!(at(19, 40), [0, 0, 0, 255]);
        assert_eq!(at(20, 46), [0, 0, 0, 255], "and exactly 6 tall");

        // The untransformed pixel is still just the GUI scale: device (80, 80)..(82, 82).
        for (x, y) in [(80, 80), (81, 81)] {
            assert_eq!(at(x, y), [0, 255, 0, 255]);
        }
        assert_eq!(at(82, 80), [0, 0, 0, 255], "still 2x2, not 6x6");
    }

    /// The actual GUI slot state is depth-tested even though the surrounding widgets are not.
    /// The near triangle must win regardless of whether it was submitted before or after the
    /// farther triangle; the reverse-winding control also proves the inherited GL_BACK cull is
    /// attached to the item pipeline rather than merely documented on the CPU list.
    #[test]
    fn depth_buffer_replays_item_z_independent_of_submission_order() {
        use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

        const W: u32 = 64;
        const H: u32 = 64;
        let mut gpu = match Gpu::headless(
            W,
            H,
            GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(g) => g,
            Err(RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter available on this machine");
                return;
            }
            Err(e) => panic!("headless context creation failed: {e}"),
        };

        fn render_order(gpu: &mut mc_render::Gpu, near_first: bool) -> [u8; 4] {
            let mut phase = crate::widget::UiPhase::new(gpu, W, H, 1, false);
            let texture = UiTexture::Solid;
            let positions = [[16.0, 16.0], [16.0, 48.0], [48.0, 16.0]];
            let uv = [[0.0, 0.0]; 3];
            let red = [[1.0, 0.0, 0.0, 1.0]; 3];
            let green = [[0.0, 1.0, 0.0, 1.0]; 3];
            let mut list = DrawList::new();
            list.with_render_state(UiDepth::Item, UiCull::Back, |list| {
                let mut emit = |depth, color| {
                    list.triangle_with_blend_depth(
                        &texture,
                        UiBlend::Alpha,
                        [depth; 3],
                        positions,
                        uv,
                        color,
                    );
                };
                if near_first {
                    emit(0.4, green);
                    emit(0.6, red);
                } else {
                    emit(0.6, red);
                    emit(0.4, green);
                }
            });
            phase.submit(list);
            let mut phases = PhaseList::new();
            phases.clear_color = wgpu::Color::BLACK;
            phases.insert(PhaseOrder::UI, Box::new(phase));
            let frame = gpu.acquire().expect("acquire offscreen frame");
            phases.render(gpu, frame.view(), &Camera::default(), 0.0);
            let pixels = gpu.read_pixels().expect("readback");
            let index = (((H / 2) * W + (W * 3 / 8)) * 4) as usize;
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3],
            ]
        }

        assert_eq!(
            render_order(&mut gpu, false),
            [0, 255, 0, 255],
            "near green survives after far red"
        );
        assert_eq!(
            render_order(&mut gpu, true),
            [0, 255, 0, 255],
            "near green survives before far red"
        );

        // The second triangle below has the opposite winding. It must be rejected by the item
        // pipeline's back-face cull even though its depth is nearer.
        let mut phase = crate::widget::UiPhase::new(&gpu, W, H, 1, false);
        let mut list = DrawList::new();
        list.with_render_state(UiDepth::Item, UiCull::Back, |list| {
            list.triangle_with_blend_depth(
                &UiTexture::Solid,
                UiBlend::Alpha,
                [0.6; 3],
                [[16.0, 16.0], [16.0, 48.0], [48.0, 16.0]],
                [[0.0, 0.0]; 3],
                [[1.0, 0.0, 0.0, 1.0]; 3],
            );
            list.triangle_with_blend_depth(
                &UiTexture::Solid,
                UiBlend::Alpha,
                [0.4; 3],
                [[48.0, 16.0], [16.0, 48.0], [16.0, 16.0]],
                [[0.0, 0.0]; 3],
                [[0.0, 1.0, 0.0, 1.0]; 3],
            );
        });
        phase.submit(list);
        let mut phases = PhaseList::new();
        phases.clear_color = wgpu::Color::BLACK;
        phases.insert(PhaseOrder::UI, Box::new(phase));
        let frame = gpu.acquire().expect("acquire offscreen frame");
        phases.render(&gpu, frame.view(), &Camera::default(), 0.0);
        let pixels = gpu.read_pixels().expect("readback");
        let index = (((H / 2) * W + (W * 3 / 8)) * 4) as usize;
        assert_eq!(
            [
                pixels[index],
                pixels[index + 1],
                pixels[index + 2],
                pixels[index + 3]
            ],
            [255, 0, 0, 255],
            "reversed near triangle is culled"
        );
    }

    #[test]
    fn clearing_keeps_the_list_reusable() {
        let mut d = DrawList::new();
        d.rect(0, 0, 1, 1, 0xFFFF_FFFF);
        d.clear();
        assert!(d.is_empty());
        assert!(d.batches().is_empty());
        d.rect(0, 0, 1, 1, 0xFFFF_FFFF);
        assert_eq!(d.batches().len(), 1);
    }

    /// #24: Solid merging changes nothing a pixel depends on. Per vertex: position, colour,
    /// depth, blend, depth window and cull are identical; textured vertices keep their UV and
    /// texture; Solid vertices carry the sentinel that the shader turns into the white texel's
    /// exact `1.0`. Batches stay contiguous and ordered, and only their count drops.
    #[test]
    fn solid_merging_preserves_every_shading_input_and_only_drops_batches() {
        fn scene(d: &mut DrawList) {
            draw_canonical_scene(d);
            // A sidebar: plate, name, score per row — Solid/Font alternation.
            let mut font = uniform_font();
            for row in 0..15 {
                let y = 40 + row * 9;
                d.rect(200, y, 260, y + 9, 0x5000_0000);
                let y = y as f32;
                font.draw_string(d, "\u{a7}c[R] \u{a7}nPlayer", 202.0, y, 0x20FF_FFFF, false);
                font.draw_string(d, "\u{a7}c12", 250.0, y, 0x20FF_FFFF, false);
            }
            d.rect_xor_blue(1, 1, 9, 9);
            d.with_transform(Transform::translate(5.0, 7.0).then_scale(2.0), |d| {
                d.rect(0, 0, 3, 3, 0x8000_00FF);
                font.draw_string_with_shadow(d, "chat line", 1.0, 1.0, -1);
            });
            d.with_render_state(UiDepth::Item, UiCull::Back, |d| {
                d.rect(0, 0, 2, 2, 0xFF00_FF00);
                d.blit(&UiTexture::gui(ICONS_TEXTURE), 0, 0, 0, 0, 4, 4);
                d.rect(0, 0, 2, 2, 0xFF00_FF00);
            });
            d.rect(0, 0, 1, 1, 0xFFFF_FFFF);
        }
        let mut plain = DrawList::new();
        scene(&mut plain);
        let mut merged = DrawList::new();
        merged.enable_solid_merging();
        scene(&mut merged);

        let per_vertex = |d: &DrawList| {
            let mut out = Vec::with_capacity(d.vertices().len());
            let mut next = 0;
            for b in d.batches() {
                assert_eq!(b.range.start, next, "batches stay contiguous and ordered");
                next = b.range.end;
                for _ in b.range.clone() {
                    out.push((b.texture.clone(), b.blend, b.depth, b.cull));
                }
            }
            assert_eq!(next, d.vertices().len());
            out
        };
        let (p, m) = (per_vertex(&plain), per_vertex(&merged));
        assert_eq!(plain.vertices().len(), merged.vertices().len());
        assert_eq!(plain.depths(), merged.depths());
        for (i, ((pv, mv), (ps, ms))) in plain
            .vertices()
            .iter()
            .zip(merged.vertices())
            .zip(p.iter().zip(&m))
            .enumerate()
        {
            assert_eq!((ps.1, ps.2, ps.3), (ms.1, ms.2, ms.3), "vertex {i} state");
            assert_eq!(pv.pos, mv.pos, "vertex {i}");
            assert_eq!(pv.color, mv.color, "vertex {i}");
            if ps.0 == UiTexture::Solid {
                assert_eq!(mv.uv, SOLID_SENTINEL_UV, "vertex {i}");
            } else {
                assert_eq!(ps.0, ms.0, "vertex {i} texture");
                assert_eq!(pv.uv, mv.uv, "vertex {i}");
                assert!(
                    mv.uv[0] > SOLID_UV_THRESHOLD,
                    "vertex {i} must not look Solid"
                );
            }
        }
        assert!(
            merged.batches().len() * 2 < plain.batches().len(),
            "{} merged vs {} plain batches",
            merged.batches().len(),
            plain.batches().len()
        );
        assert!(!plain.solid_merging());
        let mut cleared = merged.clone();
        cleared.clear();
        assert!(cleared.solid_merging(), "clear keeps the configuration");
        assert_eq!(
            cleared,
            DrawList::new(),
            "equality ignores the configuration"
        );
    }

    /// The WGSL threshold must stay in lockstep with the CPU-side sentinel.
    #[test]
    fn the_ui_shader_tests_the_same_solid_threshold() {
        assert_eq!(SOLID_UV_THRESHOLD, -32_768.0);
        const { assert!(SOLID_SENTINEL_UV[0] < SOLID_UV_THRESHOLD) };
        assert!(crate::widget::gpu::ui_shader_source().contains("in.uv.x < -32768.0"));
    }
}
