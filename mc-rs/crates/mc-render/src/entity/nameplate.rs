//! Nameplates, and the depth behaviour that makes them read correctly.
//!
//! java: `client/renderer/entity/Render.renderLivingLabel`,
//! `RendererLivingEntity.renderName`/`canRenderName`.
//!
//! # Why this is not just "draw text in world space"
//!
//! Vanilla draws a name label **twice**:
//!
//! 1. depth test **off**, depth write **off**, colour `0x20FFFFFF` — white at alpha 32/255. This
//!    is the ghost you see through walls.
//! 2. depth test **on**, depth write **on**, colour `0xFFFFFFFF` — opaque white, so the name
//!    reads solid when there is line of sight.
//!
//! Both passes draw the same glyphs at the same place; the second simply overwrites the first
//! wherever it passes the depth test. Collapsing them to one pass is the usual mistake and it
//! produces either names that vanish behind terrain or names that are equally bright through
//! it. [`NameplatePass`] carries the two states so a caller cannot accidentally merge them.
//!
//! The black backdrop quad is drawn once, before both, with the depth *test* off and depth
//! writes off. The sneaking branch leaves the test on but still suppresses the write — see the
//! pipeline table below.
//!
//! Sneaking is a **third** behaviour, not a variant of the first: `RendererLivingEntity.
//! renderName` takes its own branch that never calls `disableDepth`, so a sneaking player's
//! name is dim *and* occluded, and its visibility range halves from 64 to 32 blocks.
//!
//! # What this module does and does not do
//!
//! It owns the billboard transform, the backdrop geometry, the depth/colour policy and the
//! visibility rules — everything that is decidable without a font. Glyph layout and the font
//! atlas are S15's (`mc-ui`), which `mc-render` does not depend on, so
//! [`Nameplate::layout`] takes the string width in pixels as an argument and returns positions
//! the caller feeds to the font renderer.
//!
//! # The `mc-ui` seam, resolved
//!
//! `mc-ui` depends on `mc-render` (`crates/mc-ui/Cargo.toml`), not the reverse, so this crate can
//! never call into S15's `FontRenderer` without a cycle. Two shapes were on the table: move S15's
//! glyph-atlas code down here, or draw nameplates from `mc-ui`/`mc-client` using geometry this
//! module exposes. This module takes the second path — the same shape S11 used when it stopped
//! depending on `FaceBakery` and classified corners geometrically instead: push the dependency
//! direction that already exists (`mc-ui` -> `mc-render`) rather than invert it.
//!
//! The seam is **three plain-data pieces and no trait**:
//!
//! | Piece | What crosses |
//! |---|---|
//! | [`NameplateGlyphVertex`] | a `#[repr(C)]` position + atlas UV + straight RGBA triple. No font type, no `mc-ui` type. |
//! | [`Nameplate::glyph_vertices`] | one label-local textured quad, in vanilla's `renderChar` corner order, billboarded into six camera-relative triangle vertices. |
//! | [`NameplatePhase::upload_font_texture`] / [`push_glyphs`](NameplatePhase::push_glyphs) | the atlas pixels and the geometry, keyed by an opaque `&str` the caller names. |
//!
//! `mc-ui`'s `nameplate` module (also S12-owned) is the other half: it runs S15's `FontRenderer`
//! at [`NameplateLayout::text_x`]/[`text_y`](NameplateLayout::text_y), converts each emitted
//! glyph quad through [`Nameplate::glyph_vertices`], and pushes the result here. Nothing in this
//! file knows a font exists, so the dependency graph is unchanged.
//!
//! # The five pipelines
//!
//! Two entity-label passes plus a sign-text pass, because vanilla's `depthMask` is *not* the same
//! for every draw:
//!
//! | Draw | depth test | depth write | java |
//! |---|---|---|---|
//! | backdrop, `ThroughWalls` | off | off | `disableDepth` + `depthMask(false)` |
//! | glyphs, `ThroughWalls` | off | off | same bracket, `drawString(…, 553648127)` |
//! | backdrop, `InFront` (sneaking only) | on | off | sneaking branch never calls `disableDepth`, but does call `depthMask(false)` around the quad |
//! | glyphs, `InFront` | on | on | `enableDepth` + `depthMask(true)`, `drawString(…, -1)` |
//! | sign glyphs | on | off | `TileEntitySignRenderer`: `depthMask(false)` |
//!
//! The backdrop never writes depth in *either* vanilla branch, which is load-bearing: the glyphs
//! sit at exactly the same label-local `z = 0` as the backdrop, so a depth-writing backdrop
//! would make the depth-tested text fail its own `Less` comparison and vanish.
//!
//! **What is proven**: the billboard matrix, the backdrop quad, the glyph quads with real
//! `ascii.png` texels, and the two-pass depth/blend policy are all drawn on a real GPU and read
//! back — see this module's tests and `crates/mc-ui/tests/nameplate_3d.rs`.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::context::{Gpu, DEPTH_FORMAT};
use crate::frame::Camera;
use crate::pipeline::{PassTarget, RenderPhase};

#[cfg(test)]
use super::EntityVertex;

/// Vanilla's label scale: `0.016666668 * 1.6`. java: `Render.renderLivingLabel`'s `f1`.
pub const NAMEPLATE_SCALE: f32 = 0.016_666_668 * 1.6;

/// Extra label-local Y a **sneaking** label is pushed down by, in label pixels.
///
/// java: `RendererLivingEntity.renderName`'s sneaking branch —
/// `GlStateManager.translate(0.0F, 9.374999F, 0.0F)` *after* the `scale(-f1, -f1, f1)`, so it is
/// measured in label pixels, and label `+y` is world *down*. `9.374999 * 0.02666667` is a
/// quarter of a block: a sneaking player's tag visibly drops.
pub const NAMEPLATE_SNEAK_DROP: f32 = 9.374_999;

/// Vertical gap between a scoreboard score line and the name above it, in blocks.
///
/// java: `RenderPlayer.renderOffsetLivingLabel` — `FONT_HEIGHT * 1.15F * p_177069_9_`, where
/// `p_177069_9_` is the `0.02666667` scale `RendererLivingEntity.renderName` passes in and
/// `FONT_HEIGHT` is 9.
pub const NAMEPLATE_SCORE_GAP: f32 = 9.0 * 1.15 * NAMEPLATE_SCALE;

/// Squared distance inside which a player's scoreboard score line is drawn under their name.
///
/// java: `RenderPlayer.renderOffsetLivingLabel` — `if (p_177069_10_ < 100.0D)`, and
/// `p_177069_10_` is `getDistanceSqToEntity`, so this is blocks **squared** (10 blocks).
pub const NAMEPLATE_SCORE_RANGE_SQ: f32 = 100.0;

/// Height above the entity's feet that the label sits at, before the entity's own height.
/// java: `translate(x, y + entityIn.height + 0.5F, z)`
pub const NAMEPLATE_LIFT: f32 = 0.5;

/// Maximum render distance in blocks. java: `renderOffsetLivingLabel` passes 64.
pub const NAMEPLATE_RANGE: f32 = 64.0;

/// Maximum render distance while sneaking. java: `RendererLivingEntity.renderName`'s `f`.
pub const NAMEPLATE_RANGE_SNEAKING: f32 = 32.0;

/// How entity labels look: the Nametags mod's knobs.
///
/// [`NameplateStyle::VANILLA`] is exactly `Render.renderLivingLabel`, and it is what every caller
/// gets unless the mod is on. The style only changes what is drawn, never which labels exist:
/// range, team visibility and the two depth passes are untouched.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NameplateStyle {
    /// Draw the glyphs with `FontRenderer`'s drop shadow (vanilla calls the three-int
    /// `drawString`, i.e. no shadow).
    pub text_shadow: bool,
    /// The backdrop's alpha. java: `color(0, 0, 0, 0.25)`. Zero leaves the backdrop out.
    pub background_alpha: f32,
    /// Multiplies [`NAMEPLATE_SCALE`] about the label's anchor (and the score line's gap).
    pub scale: f32,
}

impl NameplateStyle {
    /// `Render.renderLivingLabel` as it is.
    pub const VANILLA: Self = Self {
        text_shadow: false,
        background_alpha: 0.25,
        scale: 1.0,
    };

    /// Whether this is [`Self::VANILLA`].
    #[must_use]
    pub fn is_vanilla(&self) -> bool {
        *self == Self::VANILLA
    }
}

impl Default for NameplateStyle {
    fn default() -> Self {
        Self::VANILLA
    }
}

/// One of the two depth passes a nameplate is drawn in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NameplatePass {
    /// Depth test off — the faint name visible through terrain.
    ThroughWalls,
    /// Depth test on — the solid name when there is line of sight.
    InFront,
    /// Depth test on, depth writes off — vanilla sign text's `depthMask(false)` state.
    ///
    /// This is intentionally separate from [`NameplatePass::InFront`]: entity/nameplate text
    /// uses the generic in-front pipeline and writes depth, while `TileEntitySignRenderer`
    /// inherits depth testing but explicitly suppresses depth writes around its glyphs.
    SignText,
}

impl NameplatePass {
    /// Both passes, in the order vanilla draws them.
    pub const ORDER: [NameplatePass; 2] = [Self::ThroughWalls, Self::InFront];

    /// Whether this pass depth-tests. java: `disableDepth` then `enableDepth`.
    #[must_use]
    pub fn depth_test(self) -> bool {
        matches!(self, Self::InFront | Self::SignText)
    }

    /// Whether this pass writes depth. java: `depthMask(false)` then `depthMask(true)`.
    #[must_use]
    pub fn depth_write(self) -> bool {
        matches!(self, Self::InFront)
    }

    /// The ARGB the glyphs are tinted with.
    ///
    /// java: `553648127` (`0x20FFFFFF`) then `-1` (`0xFFFFFFFF`). The first is white at
    /// alpha 32/255, which is why a name behind a wall is a ghost rather than a solid label.
    #[must_use]
    pub fn argb(self) -> u32 {
        match self {
            Self::ThroughWalls => 0x20FF_FFFF,
            Self::InFront => 0xFFFF_FFFF,
            // Sign glyph colours come from FontRenderer's per-glyph vertex tint.  Keep this
            // method total for callers that inspect the pass state; the value is not used to
            // override those vertex colours.
            Self::SignText => 0xFFFF_FFFF,
        }
    }
}

/// Everything needed to draw one label.
#[derive(Clone, Debug, PartialEq)]
pub struct Nameplate {
    /// The already-formatted display name.
    pub text: String,
    /// Camera-relative position of the entity's feet.
    pub rel: Vec3,
    /// The entity's hitbox height in blocks. java: `Entity.height`
    pub height: f32,
    /// Babies draw their label half a hitbox lower. java: `renderName`'s `isChild` term.
    pub child: bool,
    /// Sneaking takes the occluded, dim, half-range branch.
    pub sneaking: bool,
}

/// The corners of the backdrop quad, in label-local pixels.
///
/// java: `renderLivingLabel` — `(-j - 1, -1 + i)` to `(j + 1, 8 + i)`, black at alpha 0.25,
/// where `j` is half the string width and `i` is the deadmau5 offset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NameplateBackdrop {
    /// Left edge in label pixels.
    pub x0: f32,
    /// Top edge in label pixels.
    pub y0: f32,
    /// Right edge in label pixels.
    pub x1: f32,
    /// Bottom edge in label pixels.
    pub y1: f32,
    /// Straight RGBA. java: `color(0, 0, 0, 0.25)`
    pub rgba: [f32; 4],
}

/// Where the glyphs and the backdrop go, in label-local pixel space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NameplateLayout {
    /// Backdrop rectangle.
    pub backdrop: NameplateBackdrop,
    /// X of the first glyph. java: `-fontrenderer.getStringWidth(str) / 2` — **integer**
    /// division, so an odd-width name is a half pixel left of centre. Vanilla does that; a
    /// float centre would be visibly different at close range.
    pub text_x: f32,
    /// Y of the glyph baseline row. java: `i`, the deadmau5 offset.
    pub text_y: f32,
}

impl Nameplate {
    /// The billboard matrix: face the camera, flip, and shrink to label pixels.
    ///
    /// java: `renderLivingLabel` — translate, `rotate(-playerViewY, Y)`, `rotate(playerViewX,
    /// X)`, `scale(-f1, -f1, f1)`. `view_yaw`/`view_pitch` are `RenderManager.playerViewY/X`,
    /// i.e. the camera's own yaw and pitch in degrees.
    ///
    /// A sneaking label gets one extra step vanilla's non-sneaking path does not have:
    /// [`NAMEPLATE_SNEAK_DROP`] label pixels of post-scale translation, which is a quarter of a
    /// block down.
    #[must_use]
    pub fn matrix(&self, view_yaw: f32, view_pitch: f32) -> Mat4 {
        let drop = if self.child { self.height / 2.0 } else { 0.0 };
        let y = self.rel.y - drop + self.height + NAMEPLATE_LIFT;
        let m = Mat4::from_translation(Vec3::new(self.rel.x, y, self.rel.z))
            * Mat4::from_rotation_y((-view_yaw).to_radians())
            * Mat4::from_rotation_x(view_pitch.to_radians())
            * Mat4::from_scale(Vec3::new(
                -NAMEPLATE_SCALE,
                -NAMEPLATE_SCALE,
                NAMEPLATE_SCALE,
            ));
        if self.sneaking {
            m * Mat4::from_translation(Vec3::new(0.0, NAMEPLATE_SNEAK_DROP, 0.0))
        } else {
            m
        }
    }

    /// [`Self::matrix`] with the label scale multiplied by `scale` about the label's anchor
    /// (`y + height + 0.5`), for [`NameplateStyle::scale`]. `scale == 1.0` is [`Self::matrix`]
    /// itself, bit for bit. The sneaking label's quarter-block drop is applied after the scale,
    /// so it stays a quarter block at any size.
    #[must_use]
    pub fn matrix_scaled(&self, view_yaw: f32, view_pitch: f32, scale: f32) -> Mat4 {
        // Exact comparison on purpose: only the untouched default may take the vanilla path.
        #[allow(clippy::float_cmp)]
        if scale == 1.0 {
            return self.matrix(view_yaw, view_pitch);
        }
        self.matrix(view_yaw, view_pitch) * Mat4::from_scale(Vec3::splat(scale))
    }

    /// Backdrop and glyph origin for a string of `width_px` pixels.
    ///
    /// The deadmau5 lift is a `Render.renderLivingLabel` quirk only:
    /// `RendererLivingEntity.renderName`'s sneaking branch is a *separate* transcription of the
    /// same drawing code that hard-codes `-1`/`8`/`0` and never looks at the string, so a
    /// sneaking deadmau5 gets no lift. Reproduced rather than unified.
    #[must_use]
    pub fn layout(&self, width_px: i32) -> NameplateLayout {
        // java: `if (str.equals("deadmau5")) i = -10;`
        let i = if self.text == "deadmau5" && !self.sneaking {
            -10.0
        } else {
            0.0
        };
        let j = (width_px / 2) as f32;
        NameplateLayout {
            backdrop: NameplateBackdrop {
                x0: -j - 1.0,
                y0: -1.0 + i,
                x1: j + 1.0,
                y1: 8.0 + i,
                rgba: [0.0, 0.0, 0.0, 0.25],
            },
            text_x: -((width_px / 2) as f32),
            text_y: i,
        }
    }

    /// The passes to draw, in order.
    ///
    /// Sneaking collapses to a single occluded, dim pass — java:
    /// `RendererLivingEntity.renderName`'s sneaking branch never calls `disableDepth`, and
    /// tints with `553648127` only.
    #[must_use]
    pub fn passes(&self) -> &'static [NameplatePass] {
        if self.sneaking {
            // Depth-tested, but with the dim colour. `ThroughWalls`'s colour with
            // `InFront`'s depth state is not one of the two enum values, so the sneaking case
            // is reported as the pass whose *depth* behaviour it shares and callers read
            // [`Nameplate::sneak_argb`] for the tint.
            &[NameplatePass::InFront]
        } else {
            &NameplatePass::ORDER
        }
    }

    /// The tint for the single sneaking pass.
    #[must_use]
    pub fn sneak_argb(&self) -> u32 {
        0x20FF_FFFF
    }

    /// The tint the glyphs are drawn with in `pass`, sneaking taken into account.
    ///
    /// java: `renderLivingLabel` passes `553648127` then `-1`; the sneaking branch passes
    /// `553648127` to its single `drawString`.
    #[must_use]
    pub fn argb_for(&self, pass: NameplatePass) -> u32 {
        if self.sneaking {
            self.sneak_argb()
        } else {
            pass.argb()
        }
    }

    /// Which pass the backdrop quad belongs to.
    ///
    /// java: `renderLivingLabel` draws it inside the `disableDepth` bracket
    /// ([`NameplatePass::ThroughWalls`]); the sneaking branch never disables depth, so its
    /// backdrop is depth-*tested* ([`NameplatePass::InFront`]) — but see this module's docs:
    /// neither branch lets the backdrop *write* depth, and [`NameplatePhase`]'s backdrop
    /// pipelines encode that.
    #[must_use]
    pub fn backdrop_pass(&self) -> NameplatePass {
        if self.sneaking {
            NameplatePass::InFront
        } else {
            NameplatePass::ThroughWalls
        }
    }

    /// The distance beyond which this label is not drawn at all.
    #[must_use]
    pub fn range(&self) -> f32 {
        if self.sneaking {
            NAMEPLATE_RANGE_SNEAKING
        } else {
            NAMEPLATE_RANGE
        }
    }

    /// Whether the label is close enough to draw.
    ///
    /// java: `RendererLivingEntity.renderName` tests `d0 < f * f` and then
    /// `renderLivingLabel` re-tests `d0 <= maxDistance * maxDistance`; the tighter of the two
    /// wins, which for a sneaking entity is the strict `<` at 32.
    #[must_use]
    pub fn in_range(&self, distance_sq: f32) -> bool {
        let r = self.range();
        distance_sq < r * r
    }

    /// Turns one label-local pixel coordinate into a camera-relative world position.
    ///
    /// This is the seam the module docs describe: `mc-ui`'s font renderer knows glyph widths
    /// and the atlas UVs a name's characters need, but has no reason to know the billboard math
    /// that decides *where in the world* a glyph sits. This function is that math, exposed once
    /// so a caller supplies only `(x, y)` label-pixel offsets — the same space
    /// [`Nameplate::layout`]'s [`NameplateLayout::text_x`] and [`text_y`](NameplateLayout::text_y)
    /// are in — and gets back the world position of whatever corner it asks for.
    #[must_use]
    pub fn local_to_world(&self, view_yaw: f32, view_pitch: f32, x: f32, y: f32) -> [f32; 3] {
        self.matrix(view_yaw, view_pitch)
            .transform_point3(Vec3::new(x, y, 0.0))
            .into()
    }

    /// The backdrop quad's six vertices (a 0-1-2/0-2-3 fan), already billboarded into
    /// camera-relative world space and coloured with [`NameplateBackdrop::rgba`].
    ///
    /// java: `renderLivingLabel`'s `drawRect` call between the two label passes — vanilla draws
    /// it with the fixed-function pipeline's immediate-mode quad; this is the wgpu-era
    /// equivalent, computed once so [`NameplatePhase::push`] has real geometry to upload instead
    /// of four corner floats a caller would have to triangulate itself.
    #[must_use]
    pub fn backdrop_vertices(
        &self,
        view_yaw: f32,
        view_pitch: f32,
        width_px: i32,
    ) -> [NameplateVertex; 6] {
        let l = self.layout(width_px);
        let b = l.backdrop;
        let m = self.matrix(view_yaw, view_pitch);
        let corners: [Vec3; 4] = [
            Vec3::new(b.x0, b.y0, 0.0),
            Vec3::new(b.x1, b.y0, 0.0),
            Vec3::new(b.x1, b.y1, 0.0),
            Vec3::new(b.x0, b.y1, 0.0),
        ]
        .map(|c| m.transform_point3(c));
        core::array::from_fn(|slot| {
            let i = [0usize, 1, 2, 0, 2, 3][slot];
            NameplateVertex {
                pos: corners[i].into(),
                color: b.rgba,
            }
        })
    }

    /// Billboards one **label-local textured quad** into six camera-relative triangle vertices.
    ///
    /// This is the `mc-render` half of the font seam (see the module docs). `corners` and `uv`
    /// are in vanilla's `FontRenderer.renderChar` order — `[top-left, bottom-left, top-right,
    /// bottom-right]`, the four vertices of the `GL_TRIANGLE_STRIP` it emits — so an `mc-ui`
    /// caller hands a glyph straight across with no reordering. Corners are in the same
    /// label-pixel space as [`NameplateLayout::text_x`]/[`text_y`](NameplateLayout::text_y);
    /// `uv` is already normalised to the atlas (S15 divides by 128 or 256 for you); `rgba` is
    /// straight, non-premultiplied `0..1` — [`argb_to_rgba`] converts vanilla's packed
    /// `0xAARRGGBB`.
    ///
    /// The strip becomes triangles `TL-BL-TR` and `TR-BL-BR`, which is exactly the winding
    /// `GL_TRIANGLE_STRIP` expands to. Italic glyphs shear the top and bottom corners in
    /// opposite directions, so the quad is a parallelogram and the four corners genuinely
    /// cannot be collapsed to a rectangle — hence four corners in, not two.
    #[must_use]
    pub fn glyph_vertices(
        &self,
        view_yaw: f32,
        view_pitch: f32,
        corners: [[f32; 2]; 4],
        uv: [[f32; 2]; 4],
        rgba: [f32; 4],
    ) -> [NameplateGlyphVertex; 6] {
        Self::glyph_vertices_with_matrix(self.matrix(view_yaw, view_pitch), corners, uv, rgba)
    }

    /// Billboards a glyph using a matrix already computed for this label.
    ///
    /// Callers emitting several glyphs can reuse [`Self::matrix`] for the label's current
    /// position and camera angles. Recompute it when moving to another label or frame.
    #[must_use]
    pub fn glyph_vertices_with_matrix(
        m: Mat4,
        corners: [[f32; 2]; 4],
        uv: [[f32; 2]; 4],
        rgba: [f32; 4],
    ) -> [NameplateGlyphVertex; 6] {
        let world: [[f32; 3]; 4] =
            corners.map(|[x, y]| m.transform_point3(Vec3::new(x, y, 0.0)).into());
        // GL_TRIANGLE_STRIP over [0,1,2,3] is triangles (0,1,2) and (2,1,3).
        const STRIP: [usize; 6] = [0, 1, 2, 2, 1, 3];
        core::array::from_fn(|slot| {
            let i = STRIP[slot];
            NameplateGlyphVertex {
                pos: world[i],
                uv: uv[i],
                color: rgba,
            }
        })
    }
}

/// Straight, non-premultiplied RGBA in `0..1` from vanilla's packed `0xAARRGGBB`.
///
/// java: every `GlStateManager.color` derived from a packed int in `FontRenderer.renderString`
/// — `(i >> 16 & 255) / 255F`, and so on, with alpha from the top byte.
#[must_use]
pub fn argb_to_rgba(argb: u32) -> [f32; 4] {
    [
        ((argb >> 16) & 0xFF) as f32 / 255.0,
        ((argb >> 8) & 0xFF) as f32 / 255.0,
        (argb & 0xFF) as f32 / 255.0,
        ((argb >> 24) & 0xFF) as f32 / 255.0,
    ]
}

/// One vertex of nameplate **glyph** geometry, and the type that crosses the `mc-render` /
/// `mc-ui` seam.
///
/// Deliberately not [`super::EntityVertex`]: that carries a normal (for
/// `RenderHelper.enableStandardItemLighting`) and no colour, and a nameplate is the exact
/// inverse — `renderLivingLabel` calls `disableLighting` and the whole point of the two passes is
/// a per-vertex tint. Deliberately not [`NameplateVertex`] either: the backdrop is untextured.
///
/// It is plain `#[repr(C)]` data with a `Pod` derive and no `mc-ui` types anywhere in it, which
/// is what lets `mc-ui` produce it without `mc-render` ever depending on `mc-ui`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct NameplateGlyphVertex {
    /// Camera-relative position, in blocks.
    pub pos: [f32; 3],
    /// Font-atlas texture coordinate, already normalised to `0..1`.
    pub uv: [f32; 2],
    /// Straight (non-premultiplied) RGBA, `0..1`. See [`argb_to_rgba`].
    pub color: [f32; 4],
}

/// One vertex of nameplate geometry: a camera-relative world position plus a straight RGBA
/// tint. Used for the backdrop today; a future glyph-quad path (S21, once `mc-ui` draws the
/// actual text — see the module docs' cross-section note) can reuse the same buffer/pipeline by
/// carrying a UV instead of relying on vertex colour, but that is not needed to prove the depth
/// policy, which is what this section owns.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct NameplateVertex {
    /// Camera-relative position, in blocks.
    pub pos: [f32; 3],
    /// Straight (non-premultiplied) RGBA, `0..1`.
    pub color: [f32; 4],
}

/// The vertex layout [`NameplateGlyphVertex`] is fed to [`NameplatePhase`]'s glyph pipelines
/// with.
pub const NAMEPLATE_GLYPH_VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> =
    wgpu::VertexBufferLayout {
        array_stride: core::mem::size_of::<NameplateGlyphVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 12,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 20,
                shader_location: 2,
            },
        ],
    };

/// The vertex layout [`NameplateVertex`] is fed to [`NameplatePhase`]'s pipelines with.
pub const NAMEPLATE_VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: core::mem::size_of::<NameplateVertex>() as wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &[
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 12,
            shader_location: 1,
        },
    ],
};

const NAMEPLATE_WGSL: &str = r#"
struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen:     vec4<f32>,
    frame:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs(@location(0) pos: vec3<f32>, @location(1) color: vec4<f32>) -> VsOut {
    var out: VsOut;
    // `pos` is already camera-relative, exactly like `crate::entity::render::EntityPhase`'s
    // vertices — the CPU built it under Camera::relative before it ever reached this buffer.
    out.clip = camera.view_proj * vec4<f32>(pos, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

const NAMEPLATE_GLYPH_WGSL: &str = r#"
struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen:     vec4<f32>,
    frame:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs(
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    // Camera-relative, exactly like the backdrop above: `Nameplate::glyph_vertices` already
    // ran the billboard matrix on the CPU.
    out.clip = camera.view_proj * vec4<f32>(pos, 1.0);
    out.uv = uv;
    out.color = color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // The font sheet is white ink on transparent, so modulating by the vertex tint is exactly
    // what `GlStateManager.color` + GL_MODULATE did.
    let c = textureSample(tex, samp, in.uv) * in.color;
    // java: RendererLivingEntity.renderName brackets both branches with
    // `GlStateManager.alphaFunc(516, 0.1F)` — GL_GREATER, ref 0.1. This is load-bearing rather
    // than an optimisation: without it the transparent corners of every glyph quad would write
    // depth in the InFront pass and punch holes in whatever draws after.
    if (c.a <= 0.1) {
        discard;
    }
    return c;
}
"#;

/// Draws nameplate backdrops and glyphs (plus the sign-text glyph stream) with
/// vanilla's two-pass depth policy: [`NameplatePass::ThroughWalls`] first (no depth test, no
/// depth write — the dim ghost seen through terrain), then [`NameplatePass::InFront`] (normal
/// depth test and write), so the second overwrites the first wherever there is line of sight.
///
/// java: `Render.renderLivingLabel`'s `disableDepth`/`enableDepth`/`depthMask` bracketing. The
/// two nameplate passes need separate [`wgpu::RenderPipeline`]s because depth compare/write are
/// baked into a pipeline at creation, not a per-draw call the way `GlStateManager` made it in
/// 1.8.9. Sign text gets a third glyph pipeline plus solid decoration geometry with `Less` and
/// depth writes disabled.
pub struct NameplatePhase {
    pipeline_through_walls: wgpu::RenderPipeline,
    pipeline_in_front: wgpu::RenderPipeline,
    glyph_pipeline_through_walls: wgpu::RenderPipeline,
    glyph_pipeline_in_front: wgpu::RenderPipeline,
    glyph_pipeline_sign_text: wgpu::RenderPipeline,
    texture_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    textures: HashMap<String, FontTextureEntry>,
    /// Java's `GameSettings.anaglyph`, applied at the font upload boundary.
    anaglyph: bool,
    through_walls: VertexStream<NameplateVertex>,
    in_front: VertexStream<NameplateVertex>,
    glyphs_through_walls: VertexStream<NameplateGlyphVertex>,
    glyphs_in_front: VertexStream<NameplateGlyphVertex>,
    glyphs_shadows: VertexStream<NameplateGlyphVertex>,
    glyphs_sign_text: VertexStream<NameplateGlyphVertex>,
    /// Sign strikethrough/underline bars, transformed by the sign matrix rather than billboarded.
    sign_bars_text: VertexStream<NameplateVertex>,
    glyph_batches_through_walls: Vec<GlyphBatch>,
    glyph_batches_in_front: Vec<GlyphBatch>,
    glyph_batches_shadows: Vec<GlyphBatch>,
    glyph_batches_sign_text: Vec<GlyphBatch>,
    enabled: bool,
    /// The Nametags mod's style; [`NameplateStyle::VANILLA`] unless the mod is on.
    style: NameplateStyle,
    /// Depth-tested, non-depth-writing drop shadows, built on first use (see
    /// [`build_shadowed_glyph_pipeline`]).
    glyph_pipeline_in_front_shadowed: Option<wgpu::RenderPipeline>,
}

/// A contiguous run of glyph vertices sharing one font atlas.
#[derive(Clone, Debug, PartialEq, Eq)]
struct GlyphBatch {
    /// The font keys are a fixed set
    /// (`mc_ui::nameplate::font_texture_key_str`), so a batch borrows its key instead of
    /// allocating a `String` per batch every frame.
    texture: &'static str,
    start: u32,
    end: u32,
}

/// A resident font sheet and the raw decoded pixels it came from.
///
/// Java converts the temporary upload buffer in `TextureUtil` rather than mutating the resource
/// bytes. Keeping the source here lets a live anaglyph toggle rebuild every resident sheet from
/// the same unconverted pixels instead of converting an already-converted atlas a second time.
struct FontTextureEntry {
    bind_group: wgpu::BindGroup,
    source_rgba: Vec<u8>,
    width: u32,
    height: u32,
}

/// Build the upload-owned copy used by Java's `TextureUtil.anaglyphColor` boundary.
fn font_upload_rgba(source_rgba: &[u8], anaglyph: bool) -> Vec<u8> {
    mc_assets::sprite::anaglyph_rgba(source_rgba, anaglyph)
}

/// CPU accumulation plus a lazily grown GPU vertex buffer, one per pass and material.
struct VertexStream<V> {
    verts: Vec<V>,
    buffer: Option<wgpu::Buffer>,
    capacity: u64,
    uploaded: u32,
    /// Set by [`Self::clear`]/[`Self::extend`], cleared by [`Self::finish`]. A clean stream's
    /// GPU buffer already holds exactly `verts`, so a second `finish` in the same frame (sign
    /// text, R2-20 / NAME-6) re-uploads only the streams that actually changed.
    dirty: bool,
}

impl<V: Pod> VertexStream<V> {
    fn new() -> Self {
        Self {
            verts: Vec::new(),
            buffer: None,
            capacity: 0,
            uploaded: 0,
            dirty: false,
        }
    }

    fn clear(&mut self) {
        self.verts.clear();
        self.dirty = true;
    }

    fn extend(&mut self, verts: &[V]) {
        self.verts.extend_from_slice(verts);
        self.dirty = true;
    }

    fn len(&self) -> usize {
        self.verts.len()
    }

    fn finish(&mut self, gpu: &Gpu, label: &str) {
        if !self.dirty {
            debug_assert_eq!(self.uploaded as usize, self.verts.len());
            return;
        }
        self.dirty = false;
        self.uploaded = self.verts.len() as u32;
        if self.verts.is_empty() {
            return;
        }
        let needed = std::mem::size_of_val(&self.verts[..]) as u64;
        if self.capacity < needed || self.buffer.is_none() {
            // Grow geometrically so a crowded frame does not reallocate every push.
            let cap = needed.next_power_of_two().max(4096);
            self.buffer = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: cap,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.capacity = cap;
        }
        if let Some(buf) = &self.buffer {
            gpu.write_buffer_staged(buf, 0, bytemuck::cast_slice(&self.verts));
        }
    }
}

/// One nameplate pipeline: straight-alpha blending, two-sided, the given depth state.
fn build_nameplate_pipeline(
    gpu: &Gpu,
    label: &str,
    module: &wgpu::ShaderModule,
    pipeline_layout: &wgpu::PipelineLayout,
    vertex_layout: wgpu::VertexBufferLayout<'static>,
    write_depth: bool,
    compare: wgpu::CompareFunction,
) -> wgpu::RenderPipeline {
    let blend = wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
    };
    gpu.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(vertex_layout)],
            },
            // java: nameplates draw two-sided, same as every entity (disableCull).
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(write_depth),
                depth_compare: Some(compare),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: gpu.sample_count(),
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
}

/// Shadows test world depth but never write it: mathematically coplanar glyph quads can
/// interpolate slightly different depths after camera rotation/scaling. Even `LessEqual`
/// then rejects foreground fragments against their own shadow. Foreground keeps its ordinary
/// depth-writing pipeline, so world and label occlusion remain intact.
fn build_shadowed_glyph_pipeline(
    gpu: &Gpu,
    texture_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let glyph_shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mc-render nameplate glyph (shadowed)"),
            source: wgpu::ShaderSource::Wgsl(NAMEPLATE_GLYPH_WGSL.into()),
        });
    let glyph_layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mc-render nameplate glyph layout (shadowed)"),
            bind_group_layouts: &[Some(gpu.camera_layout()), Some(texture_layout)],
            immediate_size: 0,
        });
    build_nameplate_pipeline(
        gpu,
        "mc-render nameplate/glyph/in-front/shadowed",
        &glyph_shader,
        &glyph_layout,
        NAMEPLATE_GLYPH_VERTEX_LAYOUT,
        false,
        wgpu::CompareFunction::LessEqual,
    )
}

impl NameplatePhase {
    /// Builds all four pipelines. Cheap enough to do at startup; holds no per-label state.
    #[must_use]
    pub fn new(gpu: &Gpu) -> Self {
        Self::new_with_anaglyph(gpu, false)
    }

    /// Builds all pipelines with Java's upload-time anaglyph switch selected.
    #[must_use]
    pub fn new_with_anaglyph(gpu: &Gpu, anaglyph: bool) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mc-render nameplate"),
                source: wgpu::ShaderSource::Wgsl(NAMEPLATE_WGSL.into()),
            });
        let glyph_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mc-render nameplate glyph"),
                source: wgpu::ShaderSource::Wgsl(NAMEPLATE_GLYPH_WGSL.into()),
            });
        let texture_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("mc-render nameplate font"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                            count: None,
                        },
                    ],
                });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mc-render nameplate layout"),
                bind_group_layouts: &[Some(gpu.camera_layout())],
                immediate_size: 0,
            });
        let glyph_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mc-render nameplate glyph layout"),
                bind_group_layouts: &[Some(gpu.camera_layout()), Some(&texture_layout)],
                immediate_size: 0,
            });
        let build = |label: &str,
                     module: &wgpu::ShaderModule,
                     pipeline_layout: &wgpu::PipelineLayout,
                     vertex_layout: wgpu::VertexBufferLayout<'static>,
                     write_depth: bool,
                     compare: wgpu::CompareFunction| {
            build_nameplate_pipeline(
                gpu,
                label,
                module,
                pipeline_layout,
                vertex_layout,
                write_depth,
                compare,
            )
        };
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mc-render nameplate font sampler"),
            // Vanilla point-samples the font sheet; bilinear turns 8x8 glyphs to mush.
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let mut phase = Self {
            pipeline_through_walls: build(
                "mc-render nameplate/backdrop/through-walls",
                &shader,
                &layout,
                NAMEPLATE_VERTEX_LAYOUT,
                false,
                wgpu::CompareFunction::Always,
            ),
            // The backdrop never writes depth in *either* vanilla branch — both bracket the
            // quad in `depthMask(false)`. See the module docs: a depth-writing backdrop would
            // make the glyphs, which sit at the same label-local z, fail their own `Less` test.
            pipeline_in_front: build(
                "mc-render nameplate/backdrop/in-front",
                &shader,
                &layout,
                NAMEPLATE_VERTEX_LAYOUT,
                false,
                wgpu::CompareFunction::Less,
            ),
            glyph_pipeline_through_walls: build(
                "mc-render nameplate/glyph/through-walls",
                &glyph_shader,
                &glyph_layout,
                NAMEPLATE_GLYPH_VERTEX_LAYOUT,
                false,
                wgpu::CompareFunction::Always,
            ),
            glyph_pipeline_in_front: build(
                "mc-render nameplate/glyph/in-front",
                &glyph_shader,
                &glyph_layout,
                NAMEPLATE_GLYPH_VERTEX_LAYOUT,
                true,
                wgpu::CompareFunction::Less,
            ),
            // java: TileEntitySignRenderer leaves depth testing enabled but brackets all sign
            // text in depthMask(false). This must not reuse the entity-label InFront pipeline,
            // whose depth writes are intentionally enabled for ordinary nameplates.
            glyph_pipeline_sign_text: build(
                "mc-render nameplate/glyph/sign-text",
                &glyph_shader,
                &glyph_layout,
                NAMEPLATE_GLYPH_VERTEX_LAYOUT,
                false,
                wgpu::CompareFunction::Less,
            ),
            texture_layout,
            sampler,
            textures: HashMap::new(),
            anaglyph,
            through_walls: VertexStream::new(),
            in_front: VertexStream::new(),
            glyphs_through_walls: VertexStream::new(),
            glyphs_in_front: VertexStream::new(),
            glyphs_shadows: VertexStream::new(),
            glyphs_sign_text: VertexStream::new(),
            sign_bars_text: VertexStream::new(),
            glyph_batches_through_walls: Vec::new(),
            glyph_batches_in_front: Vec::new(),
            glyph_batches_shadows: Vec::new(),
            glyph_batches_sign_text: Vec::new(),
            enabled: true,
            style: NameplateStyle::VANILLA,
            glyph_pipeline_in_front_shadowed: None,
        };
        phase.upload_font_texture(gpu, Self::FALLBACK_TEXTURE, &[255, 255, 255, 255], 1, 1);
        phase
    }

    /// The atlas key used when a label's font sheet is not resident yet.
    ///
    /// A 1x1 opaque white texel, so a label whose unicode page has not loaded draws as solid
    /// tinted boxes rather than disappearing or stalling the frame.
    pub const FALLBACK_TEXTURE: &'static str = "mc-render/nameplate/fallback";

    /// Uploads (or replaces) a font atlas. `rgba` must be `width * height * 4` bytes.
    ///
    /// The key is opaque to this crate — `mc-ui` picks it and passes the same string to
    /// [`push_glyphs`](Self::push_glyphs). That is deliberate: naming the sheets
    /// (`ascii.png` vs. `unicode_page_XX.png`) is a font concern and lives on the `mc-ui` side
    /// of the seam.
    ///
    /// The texture is `Rgba8Unorm`, **not** sRGB, for the reason S15 gives for the same asset in
    /// `mc-ui`'s widget phase: 1.8.9's GL does no colour conversion, and an sRGB view darkens
    /// every font texel.
    ///
    /// # Panics
    /// If `rgba` is not exactly `width * height * 4` bytes.
    pub fn upload_font_texture(
        &mut self,
        gpu: &Gpu,
        key: &str,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) {
        assert_eq!(
            rgba.len(),
            (width as usize) * (height as usize) * 4,
            "nameplate font atlas {key} is {} bytes, expected {width}x{height}x4",
            rgba.len()
        );
        // A font sheet with a side over the device's limit (a pack's
        // `ascii.png`) is never created (a validation error ends the process): it stays blank,
        // as vanilla's `glTexImage2D` leaves it.
        let max_side = gpu.device.limits().max_texture_dimension_2d;
        if width > max_side || height > max_side {
            tracing::warn!(
                key,
                width,
                height,
                max_side,
                "font sheet over the device limit: blank"
            );
            return self.upload_font_texture(gpu, key, &[0; 4], 1, 1);
        }
        let source_rgba = rgba.to_vec();
        let upload_rgba = font_upload_rgba(&source_rgba, self.anaglyph);
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(key),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            texture.as_image_copy(),
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
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(key),
            layout: &self.texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.textures.insert(
            key.to_string(),
            FontTextureEntry {
                bind_group,
                source_rgba,
                width,
                height,
            },
        );
    }

    /// Whether Java's upload-time anaglyph conversion is enabled for this phase.
    #[must_use]
    pub const fn anaglyph_enabled(&self) -> bool {
        self.anaglyph
    }

    /// Toggle Java's global transform and rebuild every resident font sheet from raw pixels.
    ///
    /// The source cache is intentionally replayed through [`Self::upload_font_texture`], which
    /// applies the conversion to an upload-owned copy. That keeps live off/on changes exact and
    /// prevents the integer transform from being applied twice to a cached sheet.
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
                    entry.source_rgba.clone(),
                    entry.width,
                    entry.height,
                )
            })
            .collect();
        for (key, source_rgba, width, height) in uploads {
            self.upload_font_texture(gpu, &key, &source_rgba, width, height);
        }
    }

    /// Set the transform for subsequent font uploads without rebuilding resident sheets.
    ///
    /// Use [`Self::set_anaglyph_with_gpu`] when an already-installed phase must change in place.
    pub fn set_anaglyph(&mut self, enabled: bool) {
        self.anaglyph = enabled;
    }

    /// True if `key` has been uploaded.
    #[must_use]
    pub fn has_font_texture(&self, key: &str) -> bool {
        self.textures.contains_key(key)
    }

    /// Drops all per-frame geometry. Call once at the top of the frame.
    pub fn begin(&mut self) {
        self.through_walls.clear();
        self.in_front.clear();
        self.glyphs_through_walls.clear();
        self.glyphs_in_front.clear();
        self.glyphs_shadows.clear();
        self.glyphs_sign_text.clear();
        self.sign_bars_text.clear();
        self.glyph_batches_through_walls.clear();
        self.glyph_batches_in_front.clear();
        self.glyph_batches_shadows.clear();
        self.glyph_batches_sign_text.clear();
    }

    /// Queues backdrop geometry for one of the two passes. A sneaking label
    /// ([`Nameplate::passes`] returning just [`NameplatePass::InFront`]) simply never calls this
    /// with [`NameplatePass::ThroughWalls`].
    pub fn push(&mut self, pass: NameplatePass, verts: &[NameplateVertex]) {
        match pass {
            NameplatePass::ThroughWalls => self.through_walls.extend(verts),
            NameplatePass::InFront => self.in_front.extend(verts),
            // SignText bars are untextured solid geometry, but use the same Less/depth-write-off
            // state as sign glyphs. The caller supplies a fixed sign matrix, never a billboard.
            NameplatePass::SignText => self.sign_bars_text.extend(verts),
        }
    }

    /// Queues **glyph** geometry — the other half of the font seam.
    ///
    /// `texture` is whatever key was handed to
    /// [`upload_font_texture`](Self::upload_font_texture); an unknown key falls back to
    /// [`FALLBACK_TEXTURE`](Self::FALLBACK_TEXTURE) rather than dropping the draw, so a
    /// still-loading unicode page cannot make a name silently vanish. Consecutive pushes sharing
    /// a key and a pass merge into one draw. The key is `'static` (the font keys are a fixed set,
    /// see `mc_ui::nameplate::font_texture_key_str`), so queuing a batch never allocates.
    pub fn push_glyphs(
        &mut self,
        pass: NameplatePass,
        texture: &'static str,
        verts: &[NameplateGlyphVertex],
    ) {
        if verts.is_empty() {
            return;
        }
        let (stream, batches) = match pass {
            NameplatePass::ThroughWalls => (
                &mut self.glyphs_through_walls,
                &mut self.glyph_batches_through_walls,
            ),
            NameplatePass::InFront => (&mut self.glyphs_in_front, &mut self.glyph_batches_in_front),
            NameplatePass::SignText => (
                &mut self.glyphs_sign_text,
                &mut self.glyph_batches_sign_text,
            ),
        };
        let start = stream.len() as u32;
        stream.extend(verts);
        let end = stream.len() as u32;
        match batches.last_mut() {
            Some(b) if b.texture == texture && b.end == start => b.end = end,
            _ => batches.push(GlyphBatch {
                texture,
                start,
                end,
            }),
        }
    }

    /// Queue the in-front drop-shadow glyphs separately from depth-writing foreground ink.
    /// Consecutive sheets still batch across labels; the shadow stream is drawn first.
    pub fn push_shadow_glyphs(&mut self, texture: &'static str, verts: &[NameplateGlyphVertex]) {
        if verts.is_empty() {
            return;
        }
        let start = self.glyphs_shadows.len() as u32;
        self.glyphs_shadows.extend(verts);
        let end = self.glyphs_shadows.len() as u32;
        match self.glyph_batches_shadows.last_mut() {
            Some(b) if b.texture == texture && b.end == start => b.end = end,
            _ => self.glyph_batches_shadows.push(GlyphBatch {
                texture,
                start,
                end,
            }),
        }
    }

    /// The Nametags mod's style. Only [`NameplateStyle::text_shadow`] matters to the
    /// phase (it selects the depth-tested, non-depth-writing shadow pipeline). The geometry
    /// side of the style is applied by whoever builds the labels (`mc_ui::nameplate::NameplateCache::set_style`).
    pub fn set_style(&mut self, style: NameplateStyle) {
        self.style = style;
    }

    /// The style last set with [`Self::set_style`].
    #[must_use]
    pub fn style(&self) -> NameplateStyle {
        self.style
    }

    /// Uploads this frame's geometry. Call once after the last push of either kind.
    ///
    /// Calling it again in the same frame (the sign-text path does, after entity labels) only
    /// re-uploads the streams pushed to since the previous call.
    pub fn finish(&mut self, gpu: &Gpu) {
        if self.style.text_shadow && self.glyph_pipeline_in_front_shadowed.is_none() {
            self.glyph_pipeline_in_front_shadowed =
                Some(build_shadowed_glyph_pipeline(gpu, &self.texture_layout));
        }
        self.through_walls
            .finish(gpu, "mc-render nameplate backdrop vertices");
        self.in_front
            .finish(gpu, "mc-render nameplate backdrop vertices");
        self.glyphs_through_walls
            .finish(gpu, "mc-render nameplate glyph vertices");
        self.glyphs_in_front
            .finish(gpu, "mc-render nameplate glyph vertices");
        self.glyphs_shadows
            .finish(gpu, "mc-render nameplate shadow glyph vertices");
        self.glyphs_sign_text
            .finish(gpu, "mc-render nameplate sign-text vertices");
        self.sign_bars_text
            .finish(gpu, "mc-render nameplate sign-text bars");
    }

    /// Backdrop vertices queued this frame, `(through_walls, in_front)`.
    #[must_use]
    pub fn vertex_counts(&self) -> (usize, usize) {
        (self.through_walls.len(), self.in_front.len())
    }

    /// Glyph vertices queued this frame, `(through_walls, in_front)`.
    #[must_use]
    pub fn glyph_vertex_counts(&self) -> (usize, usize) {
        (
            self.glyphs_through_walls.len(),
            self.glyphs_in_front.len() + self.glyphs_shadows.len(),
        )
    }

    /// Glyph draw calls this frame will issue, `(through_walls, in_front)`.
    #[must_use]
    pub fn glyph_batch_counts(&self) -> (usize, usize) {
        (
            self.glyph_batches_through_walls.len(),
            self.glyph_batches_in_front.len() + self.glyph_batches_shadows.len(),
        )
    }

    /// Sign-text glyph vertices queued this frame. Sign text has its own depth-tested,
    /// depth-write-disabled pipeline and is intentionally excluded from the entity-label pair
    /// returned by [`Self::glyph_vertex_counts`].
    #[must_use]
    pub fn sign_text_vertex_count(&self) -> usize {
        self.glyphs_sign_text.len()
    }

    /// Sign-text draw calls queued this frame.
    #[must_use]
    pub fn sign_text_batch_count(&self) -> usize {
        self.glyph_batches_sign_text.len()
    }

    /// Sign strikethrough/underline vertices queued this frame.
    #[must_use]
    pub fn sign_text_bar_vertex_count(&self) -> usize {
        self.sign_bars_text.len()
    }

    /// Turns the phase off without unregistering it.
    pub fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
    }

    fn draw_glyphs(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        stream: &VertexStream<NameplateGlyphVertex>,
        batches: &[GlyphBatch],
        pipeline: &wgpu::RenderPipeline,
    ) {
        let Some(buf) = &stream.buffer else {
            return;
        };
        if stream.uploaded == 0 {
            return;
        }
        pass.set_pipeline(pipeline);
        pass.set_vertex_buffer(0, buf.slice(..));
        for batch in batches {
            let Some(bind) = self
                .textures
                .get(batch.texture)
                .or_else(|| self.textures.get(Self::FALLBACK_TEXTURE))
            else {
                continue;
            };
            pass.set_bind_group(1, &bind.bind_group, &[]);
            pass.draw(batch.start..batch.end, 0..1);
        }
    }
}

impl RenderPhase for NameplatePhase {
    fn name(&self) -> &'static str {
        "nameplate"
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        self.record_to(gpu, PassTarget::Own { enc, view });
    }

    // One pass, nothing else on the encoder: it joins the world pass.
    fn shares_world_pass(&self) -> bool {
        true
    }

    fn record_into_pass(&mut self, gpu: &Gpu, pass: &mut wgpu::RenderPass<'static>, _cam: &Camera) {
        self.record_to(gpu, PassTarget::Shared(pass));
    }
}

impl NameplatePhase {
    /// The phase's draws, into a pass of its own or the shared world pass.
    fn record_to(&self, gpu: &Gpu, target: PassTarget<'_>) {
        if self.through_walls.uploaded == 0
            && self.in_front.uploaded == 0
            && self.glyphs_through_walls.uploaded == 0
            && self.glyphs_in_front.uploaded == 0
            && self.glyphs_shadows.uploaded == 0
            && self.glyphs_sign_text.uploaded == 0
            && self.sign_bars_text.uploaded == 0
        {
            return;
        }
        let mut own = None;
        let pass = target.begin(gpu, "nameplate", &mut own);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        // Draw order is vanilla's, and it matters: backdrop, then the through-walls ghost, then
        // the depth-tested solid text over the top of both.
        // java: `renderLivingLabel` — quad, `drawString(553648127)`, `enableDepth`,
        // `drawString(-1)`.
        if self.through_walls.uploaded > 0 {
            if let Some(buf) = &self.through_walls.buffer {
                pass.set_pipeline(&self.pipeline_through_walls);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..self.through_walls.uploaded, 0..1);
            }
        }
        self.draw_glyphs(
            pass,
            &self.glyphs_through_walls,
            &self.glyph_batches_through_walls,
            &self.glyph_pipeline_through_walls,
        );
        if self.in_front.uploaded > 0 {
            if let Some(buf) = &self.in_front.buffer {
                pass.set_pipeline(&self.pipeline_in_front);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..self.in_front.uploaded, 0..1);
            }
        }
        // One separate batched shadow stream avoids self-occlusion without changing the
        // foreground's world-depth writes or adding a draw call per label.
        let shadow_pipeline = self
            .glyph_pipeline_in_front_shadowed
            .as_ref()
            .filter(|_| self.style.text_shadow)
            .unwrap_or(&self.glyph_pipeline_in_front);
        self.draw_glyphs(
            pass,
            &self.glyphs_shadows,
            &self.glyph_batches_shadows,
            shadow_pipeline,
        );
        self.draw_glyphs(
            pass,
            &self.glyphs_in_front,
            &self.glyph_batches_in_front,
            &self.glyph_pipeline_in_front,
        );
        self.draw_glyphs(
            pass,
            &self.glyphs_sign_text,
            &self.glyph_batches_sign_text,
            &self.glyph_pipeline_sign_text,
        );
        if self.sign_bars_text.uploaded > 0 {
            if let Some(buf) = &self.sign_bars_text.buffer {
                // `pipeline_in_front` has Less plus depth writes disabled, which is exactly the
                // state left by TileEntitySignRenderer around both glyphs and decoration bars.
                // Draw bars after glyphs as FontRenderer does for each formatted character.
                pass.set_pipeline(&self.pipeline_in_front);
                pass.set_vertex_buffer(0, buf.slice(..));
                pass.draw(0..self.sign_bars_text.uploaded, 0..1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plate(text: &str) -> Nameplate {
        Nameplate {
            text: text.into(),
            rel: Vec3::new(0.0, 0.0, 5.0),
            height: 1.8,
            child: false,
            sneaking: false,
        }
    }

    #[test]
    fn the_two_passes_have_opposite_depth_states() {
        assert!(!NameplatePass::ThroughWalls.depth_test());
        assert!(!NameplatePass::ThroughWalls.depth_write());
        assert!(NameplatePass::InFront.depth_test());
        assert!(NameplatePass::InFront.depth_write());
        assert!(NameplatePass::SignText.depth_test());
        assert!(!NameplatePass::SignText.depth_write());
        assert_eq!(
            NameplatePass::ORDER,
            [NameplatePass::ThroughWalls, NameplatePass::InFront],
            "the through-walls ghost is drawn first and overwritten"
        );
    }

    #[test]
    fn glyph_alpha_cutout_matches_java_gl_greater_boundary() {
        // Java's alphaFunc(GL_GREATER, 0.1F) rejects alpha exactly at the reference value;
        // a strict `<` in WGSL would incorrectly let that boundary write colour/depth.
        assert!(
            NAMEPLATE_GLYPH_WGSL.contains("if (c.a <= 0.1)"),
            "glyph shader must discard alpha <= 0.1, matching GL_GREATER"
        );
    }

    #[test]
    fn the_ghost_pass_is_white_at_alpha_thirty_two() {
        // java: 553648127 == 0x20FFFFFF.
        assert_eq!(NameplatePass::ThroughWalls.argb(), 553_648_127);
        assert_eq!(NameplatePass::ThroughWalls.argb() >> 24, 32);
        assert_eq!(NameplatePass::ThroughWalls.argb() & 0xFF_FFFF, 0xFF_FFFF);
        // java: -1 == 0xFFFFFFFF.
        assert_eq!(NameplatePass::InFront.argb(), u32::MAX);
    }

    #[test]
    fn a_sneaking_name_is_occluded_and_dim() {
        let mut p = plate("Steve");
        p.sneaking = true;
        assert_eq!(p.passes(), &[NameplatePass::InFront]);
        assert!(
            p.passes()[0].depth_test(),
            "sneaking names do not show through walls"
        );
        assert_eq!(p.sneak_argb(), 553_648_127, "but they are still dim");
    }

    #[test]
    fn range_halves_while_sneaking() {
        let mut p = plate("Steve");
        assert!(p.in_range(63.0 * 63.0));
        assert!(!p.in_range(64.0 * 64.0));
        p.sneaking = true;
        assert!(p.in_range(31.0 * 31.0));
        assert!(!p.in_range(32.0 * 32.0));
    }

    #[test]
    fn the_label_sits_half_a_block_above_the_hitbox() {
        let p = plate("Steve");
        let m = p.matrix(0.0, 0.0);
        let origin = m.transform_point3(Vec3::ZERO);
        assert!((origin.y - (1.8 + 0.5)).abs() < 1e-6, "{}", origin.y);
        assert!((origin.z - 5.0).abs() < 1e-6);
    }

    #[test]
    fn a_baby_label_drops_by_half_its_own_hitbox() {
        // java: RendererLivingEntity.renderName subtracts `entity.height / 2.0F` from the
        // position it hands to Render.renderLivingLabel, which then adds `entity.height +
        // 0.5F`.  Use two non-default heights so this rejects all of the tempting mutations:
        // dropping no distance, a fixed player-sized 0.9 blocks, or halving the final height.
        for height in [0.875, 2.4] {
            let mut adult = plate("Pig");
            adult.height = height;
            adult.rel.y = -3.0;
            let mut baby = adult.clone();
            baby.child = true;
            let adult_y = adult.matrix(0.0, 0.0).transform_point3(Vec3::ZERO).y;
            let baby_y = baby.matrix(0.0, 0.0).transform_point3(Vec3::ZERO).y;

            assert!(
                (adult_y - (baby_y + height / 2.0)).abs() < 1e-6,
                "height {height}: child must be exactly half its own hitbox below the adult"
            );
            assert!(
                (baby_y - (baby.rel.y + height / 2.0 + NAMEPLATE_LIFT)).abs() < 1e-6,
                "height {height}: child final lift"
            );
        }
    }

    #[test]
    fn the_billboard_is_x_and_y_flipped_so_text_reads_the_right_way() {
        let p = plate("Steve");
        let m = p.matrix(0.0, 0.0);
        // A glyph pixel to the right and down in label space must land left-of-nothing and
        // *up* in world space — that is what scale(-f1, -f1, f1) buys.
        let o = m.transform_point3(Vec3::ZERO);
        let right = m.transform_point3(Vec3::new(10.0, 0.0, 0.0)) - o;
        let down = m.transform_point3(Vec3::new(0.0, 10.0, 0.0)) - o;
        assert!(right.x < 0.0, "label +x goes to world -x: {right}");
        assert!(down.y < 0.0, "label +y goes down in the world: {down}");
        assert!((right.length() - 10.0 * NAMEPLATE_SCALE).abs() < 1e-6);
    }

    #[test]
    fn the_billboard_turns_with_the_camera() {
        let p = plate("Steve");
        let m = p.matrix(90.0, 0.0);
        let o = m.transform_point3(Vec3::ZERO);
        let right = (m.transform_point3(Vec3::new(1.0, 0.0, 0.0)) - o).normalize();
        // Looking west (yaw 90), the label's horizontal axis runs along world Z.
        assert!(right.z.abs() > 0.99, "{right}");
    }

    #[test]
    fn the_backdrop_wraps_the_text_by_one_pixel() {
        let p = plate("Steve");
        // "Steve" is 27 px in vanilla's font; the exact number is the caller's problem.
        let l = p.layout(27);
        assert_eq!(l.backdrop.x0, -14.0, "-(27/2) - 1 with integer division");
        assert_eq!(l.backdrop.x1, 14.0);
        assert_eq!(l.backdrop.y0, -1.0);
        assert_eq!(l.backdrop.y1, 8.0);
        assert_eq!(l.backdrop.rgba, [0.0, 0.0, 0.0, 0.25]);
        assert_eq!(
            l.text_x, -13.0,
            "integer division, so odd widths sit a pixel left"
        );
    }

    #[test]
    fn deadmau5_gets_the_ten_pixel_lift() {
        let l = plate("deadmau5").layout(40);
        assert_eq!(l.text_y, -10.0);
        assert_eq!(l.backdrop.y0, -11.0);
        assert_eq!(l.backdrop.y1, -2.0);
        // Any other name does not.
        assert_eq!(plate("deadmau6").layout(40).text_y, 0.0);
    }

    #[test]
    fn the_scale_matches_the_sneaking_paths_literal() {
        // java: renderLivingLabel computes 0.016666668 * 1.6; renderName's sneaking branch
        // spells the same number out as 0.02666667. They must agree to within a float ulp of
        // the shorter literal.
        assert!((NAMEPLATE_SCALE - 0.026_666_67).abs() < 1e-7);
    }

    #[test]
    fn local_to_world_agrees_with_the_billboard_matrix() {
        let p = plate("Steve");
        let got = p.local_to_world(15.0, -5.0, 3.0, -2.0);
        let want = p
            .matrix(15.0, -5.0)
            .transform_point3(Vec3::new(3.0, -2.0, 0.0));
        assert_eq!(got, <[f32; 3]>::from(want));
    }

    #[test]
    fn local_to_world_at_the_origin_matches_the_labels_own_anchor() {
        // (0, 0) in label space is always the billboard's own translation: feet + height + lift.
        let p = plate("Steve");
        let got = p.local_to_world(0.0, 0.0, 0.0, 0.0);
        assert!((got[1] - (1.8 + 0.5)).abs() < 1e-6, "{got:?}");
        assert!((got[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn backdrop_vertices_are_the_layouts_corners_billboarded() {
        let p = plate("Steve");
        let verts = p.backdrop_vertices(0.0, 0.0, 27);
        assert_eq!(verts.len(), 6, "a 0-1-2/0-2-3 fan");
        for v in &verts {
            assert_eq!(v.color, [0.0, 0.0, 0.0, 0.25]);
        }
        // The four distinct corners (the fan repeats vertex 0 and 2) must match
        // `local_to_world` applied to the same layout's backdrop rectangle.
        let l = p.layout(27);
        let corners = [
            (l.backdrop.x0, l.backdrop.y0),
            (l.backdrop.x1, l.backdrop.y0),
            (l.backdrop.x1, l.backdrop.y1),
            (l.backdrop.x0, l.backdrop.y1),
        ];
        // The fan is `[0, 1, 2, 0, 2, 3]`: corners 0/1/2 each appear once at their own index,
        // and corner 3 only appears at slot 5.
        for (i, (x, y)) in corners.into_iter().enumerate() {
            let want = p.local_to_world(0.0, 0.0, x, y);
            let slot = if i < 3 { i } else { 5 };
            assert_eq!(verts[slot].pos, want, "corner {i}");
        }
    }

    // ------------------------------------------------------- the mc-ui glyph seam (CPU half)

    /// The four corners `FontRenderer.renderChar` emits, for a 6-pixel-wide `ascii.png` glyph
    /// drawn at the origin: `[top-left, bottom-left, top-right, bottom-right]`.
    fn glyph_corners() -> [[f32; 2]; 4] {
        [[0.0, 0.0], [0.0, 7.99], [4.99, 0.0], [4.99, 7.99]]
    }

    fn glyph_uv() -> [[f32; 2]; 4] {
        [
            [8.0 / 128.0, 32.0 / 128.0],
            [8.0 / 128.0, 39.99 / 128.0],
            [12.99 / 128.0, 32.0 / 128.0],
            [12.99 / 128.0, 39.99 / 128.0],
        ]
    }

    #[test]
    fn argb_unpacks_the_two_vanilla_tints() {
        // java: 553648127 == 0x20FFFFFF — white at alpha 32/255.
        let ghost = argb_to_rgba(NameplatePass::ThroughWalls.argb());
        assert_eq!(ghost[0], 1.0);
        assert_eq!(ghost[1], 1.0);
        assert_eq!(ghost[2], 1.0);
        assert!((ghost[3] - 32.0 / 255.0).abs() < 1e-7, "{}", ghost[3]);
        // The alpha cutout in the glyph shader is GL_GREATER 0.1; the ghost must clear it, and
        // it does so by a hair (0.1255 > 0.1). That margin is vanilla's, not ours.
        assert!(ghost[3] > 0.1);
        assert_eq!(argb_to_rgba(NameplatePass::InFront.argb()), [1.0; 4]);
        // A `§` colour code keeps the pass's alpha and replaces only the RGB. 0xFF5555 is
        // vanilla's `§c` red.
        let red = argb_to_rgba(0x20FF_5555);
        assert_eq!(red[0], 1.0);
        assert!((red[1] - 85.0 / 255.0).abs() < 1e-7);
        assert!((red[2] - 85.0 / 255.0).abs() < 1e-7);
        assert_eq!(red[3], ghost[3]);
    }

    #[test]
    fn a_glyph_quad_expands_to_the_strips_two_triangles() {
        let p = plate("Steve");
        let rgba = argb_to_rgba(0xFFFF_FFFF);
        let v = p.glyph_vertices(0.0, 0.0, glyph_corners(), glyph_uv(), rgba);
        assert_eq!(v.len(), 6);
        // GL_TRIANGLE_STRIP over TL, BL, TR, BR is (TL, BL, TR) then (TR, BL, BR).
        let uv = glyph_uv();
        assert_eq!(
            [v[0].uv, v[1].uv, v[2].uv, v[3].uv, v[4].uv, v[5].uv],
            [uv[0], uv[1], uv[2], uv[2], uv[1], uv[3]]
        );
        for x in &v {
            assert_eq!(x.color, rgba);
        }
    }

    #[test]
    fn glyph_positions_agree_with_local_to_world() {
        let p = plate("Steve");
        let corners = glyph_corners();
        let v = p.glyph_vertices(37.0, -11.0, corners, glyph_uv(), [1.0; 4]);
        for (slot, corner) in [0usize, 1, 2, 3].map(|i| (i, corners[i])) {
            let want = p.local_to_world(37.0, -11.0, corner[0], corner[1]);
            // Corner 3 (bottom-right) only appears at the last strip slot.
            let at = if slot < 3 { slot } else { 5 };
            assert_eq!(v[at].pos, want, "corner {slot}");
        }
    }

    #[test]
    fn a_glyph_is_the_same_size_in_the_world_as_the_backdrop_it_sits_on() {
        // One label pixel is one texel and one NAMEPLATE_SCALE of world, for glyphs and for the
        // backdrop alike — the whole reason `mc-ui` can hand label-pixel coordinates across.
        let p = plate("Steve");
        let v = p.glyph_vertices(0.0, 0.0, glyph_corners(), glyph_uv(), [1.0; 4]);
        let width = (Vec3::from(v[2].pos) - Vec3::from(v[0].pos)).length();
        assert!((width - 4.99 * NAMEPLATE_SCALE).abs() < 1e-6, "{width}");
        let height = (Vec3::from(v[1].pos) - Vec3::from(v[0].pos)).length();
        assert!((height - 7.99 * NAMEPLATE_SCALE).abs() < 1e-6, "{height}");
    }

    #[test]
    fn a_sneaking_label_hangs_a_quarter_of_a_block_lower() {
        // java: RendererLivingEntity.renderName's sneaking branch adds
        // `translate(0, 9.374999, 0)` after the scale, and label +y is world down.
        let up = plate("Steve");
        let mut down = up.clone();
        down.sneaking = true;
        let a = up.local_to_world(0.0, 0.0, 0.0, 0.0);
        let b = down.local_to_world(0.0, 0.0, 0.0, 0.0);
        let drop = a[1] - b[1];
        assert!((drop - 0.25).abs() < 1e-5, "dropped {drop} blocks");
        assert!((drop - NAMEPLATE_SNEAK_DROP * NAMEPLATE_SCALE).abs() < 1e-6);
        assert_eq!(a[0], b[0]);
        assert_eq!(a[2], b[2]);
    }

    #[test]
    fn sneaking_deadmau5_gets_no_lift_because_vanilla_forgot_to_check() {
        let mut p = plate("deadmau5");
        assert_eq!(p.layout(40).text_y, -10.0);
        p.sneaking = true;
        // The sneaking branch is a separate transcription that hard-codes -1/8/0.
        assert_eq!(p.layout(40).text_y, 0.0);
        assert_eq!(p.layout(40).backdrop.y0, -1.0);
        assert_eq!(p.layout(40).backdrop.y1, 8.0);
    }

    #[test]
    fn the_backdrop_follows_its_labels_depth_branch() {
        let mut p = plate("Steve");
        assert_eq!(p.backdrop_pass(), NameplatePass::ThroughWalls);
        assert_eq!(p.argb_for(NameplatePass::ThroughWalls), 0x20FF_FFFF);
        assert_eq!(p.argb_for(NameplatePass::InFront), 0xFFFF_FFFF);
        p.sneaking = true;
        assert_eq!(p.backdrop_pass(), NameplatePass::InFront);
        // Sneaking is dim in its one and only pass.
        assert_eq!(p.argb_for(NameplatePass::InFront), 0x20FF_FFFF);
    }

    #[test]
    fn the_score_line_constants_are_vanillas() {
        // java: RenderPlayer.renderOffsetLivingLabel — `FONT_HEIGHT * 1.15F * 0.02666667F`
        // and `if (p_177069_10_ < 100.0D)`.
        assert!((NAMEPLATE_SCORE_GAP - 9.0 * 1.15 * 0.026_666_67).abs() < 1e-6);
        assert_eq!(NAMEPLATE_SCORE_RANGE_SQ, 100.0);
        assert!(
            (NAMEPLATE_SCORE_GAP - 0.276).abs() < 1e-4,
            "{NAMEPLATE_SCORE_GAP}"
        );
    }

    #[test]
    fn the_glyph_vertex_layout_matches_the_struct() {
        assert_eq!(core::mem::size_of::<NameplateGlyphVertex>(), 36);
        assert_eq!(NAMEPLATE_GLYPH_VERTEX_LAYOUT.array_stride, 36);
        assert_eq!(NAMEPLATE_GLYPH_VERTEX_LAYOUT.attributes[0].offset, 0);
        assert_eq!(NAMEPLATE_GLYPH_VERTEX_LAYOUT.attributes[1].offset, 12);
        assert_eq!(NAMEPLATE_GLYPH_VERTEX_LAYOUT.attributes[2].offset, 20);
    }

    #[test]
    fn anaglyph_font_upload_mixes_colours_once_and_preserves_alpha() {
        // Java's TextureUtil.anaglyphColor is integer arithmetic on RGB only. A coloured glyph
        // catches an accidental grayscale/float implementation, while these alpha bytes catch
        // the common mistake of rewriting transparent corners during the conversion.
        let source = [
            255, 0, 0, 7, 0, 255, 0, 19, 0, 0, 255, 31, 128, 128, 128, 64,
        ];
        assert_eq!(font_upload_rgba(&source, false), source);
        assert_eq!(
            font_upload_rgba(&source, true),
            [76, 76, 76, 7, 178, 150, 0, 19, 0, 28, 178, 31, 128, 128, 128, 64]
        );
    }

    #[test]
    fn live_anaglyph_toggle_replays_nameplate_sheets_from_raw_source() {
        let Some(gpu) = headless(16, 16) else {
            return;
        };
        let source = [255, 0, 0, 255];
        let mut phase = NameplatePhase::new(&gpu);
        phase.upload_font_texture(&gpu, "coloured", &source, 1, 1);
        assert!(!phase.anaglyph_enabled());
        assert_eq!(
            phase
                .textures
                .get("coloured")
                .expect("uploaded sheet")
                .source_rgba,
            source
        );

        phase.set_anaglyph_with_gpu(&gpu, true);
        assert!(phase.anaglyph_enabled());
        assert_eq!(
            phase
                .textures
                .get("coloured")
                .expect("replayed sheet")
                .source_rgba,
            source,
            "the source cache must remain raw after enabling anaglyph"
        );
        phase.set_anaglyph_with_gpu(&gpu, false);
        assert!(!phase.anaglyph_enabled());
        assert_eq!(
            phase
                .textures
                .get("coloured")
                .expect("replayed sheet")
                .source_rgba,
            source,
            "turning anaglyph off must restore the original coloured glyph"
        );
    }

    #[test]
    fn backdrop_vertices_track_the_billboard_as_the_camera_turns() {
        let p = plate("Steve");
        let a = p.backdrop_vertices(0.0, 0.0, 20);
        let b = p.backdrop_vertices(90.0, 0.0, 20);
        assert_ne!(
            a[0].pos, b[0].pos,
            "turning the camera must move the backdrop"
        );
    }

    // --------------------------------------------------------------- real GPU

    fn headless(width: u32, height: u32) -> Option<crate::context::Gpu> {
        use crate::context::GpuOptions;
        use crate::RenderError;
        match crate::context::Gpu::headless(
            width,
            height,
            GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(g) => Some(g),
            Err(RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter on this machine");
                None
            }
            Err(e) => panic!("headless context creation failed: {e}"),
        }
    }

    #[test]
    fn sign_text_queues_separately_from_depth_writing_nameplate_glyphs() {
        let Some(gpu) = headless(16, 16) else {
            return;
        };
        let mut phase = NameplatePhase::new(&gpu);
        let verts = [NameplateGlyphVertex::default(); 6];
        phase.begin();
        phase.push_glyphs(
            NameplatePass::InFront,
            NameplatePhase::FALLBACK_TEXTURE,
            &verts,
        );
        phase.push_glyphs(
            NameplatePass::SignText,
            NameplatePhase::FALLBACK_TEXTURE,
            &verts,
        );
        let bars = [NameplateVertex::default(); 6];
        phase.push(NameplatePass::SignText, &bars);
        assert_eq!(phase.glyph_vertex_counts(), (0, 6));
        assert_eq!(phase.sign_text_vertex_count(), 6);
        assert_eq!(phase.glyph_batch_counts(), (0, 1));
        assert_eq!(phase.sign_text_batch_count(), 1);
        assert_eq!(phase.sign_text_bar_vertex_count(), 6);
        assert!(NameplatePass::SignText.depth_test());
        assert!(!NameplatePass::SignText.depth_write());
        phase.finish(&gpu);
    }

    fn pixel(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * width + x) * 4) as usize;
        [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
    }

    /// A backdrop drawn alone (no occluder) must actually reach the frame buffer, in both
    /// passes. This is the "nothing is drawn" gap closed: real vertices, a real pipeline, a real
    /// GPU readback.
    #[test]
    fn a_backdrop_alone_reaches_the_frame_buffer_in_both_passes() {
        let Some(mut gpu) = headless(64, 64) else {
            return;
        };
        for pass_kind in [NameplatePass::ThroughWalls, NameplatePass::InFront] {
            let mut phase = NameplatePhase::new(&gpu);
            let p = plate("Steve");
            let verts = p.backdrop_vertices(0.0, 0.0, 40);
            phase.begin();
            phase.push(pass_kind, &verts);
            phase.finish(&gpu);

            let mut phases = crate::pipeline::PhaseList::new();
            phases.clear_color = wgpu::Color::BLACK;
            phases.insert(crate::pipeline::PhaseOrder::UI, Box::new(phase));
            let mut cam = Camera::default();
            cam.set_viewport(64, 64);
            cam.update();
            let frame = gpu.acquire().expect("acquire offscreen frame");
            phases.render(&gpu, frame.view(), &cam, 0.0);
            frame.present(&gpu.queue);
            let pixels = gpu.read_pixels().expect("read back");
            let centre = pixel(&pixels, 64, 32, 32);
            assert!(
                centre[3] > 0 || centre[0] > 0 || centre[1] > 0 || centre[2] > 0,
                "{pass_kind:?} drew nothing at the centre: {centre:?}"
            );
        }
    }

    /// The load-bearing behaviour: `ThroughWalls` shows through an occluder written into the
    /// shared depth buffer first, `InFront` does not. Both passes alpha-blend a translucent
    /// black backdrop over a solid coloured "wall" that fills the whole frame, so rather than
    /// guess which exact pixel the (small) backdrop quad lands on, this scans the whole frame
    /// for the darkest green pixel each pass produces.
    ///
    /// The comparison is against a *measured* wall-only baseline, not the wall texture's raw
    /// `[0, 200, 0]` — `EntityPhase`'s shader applies directional lighting (java:
    /// `RenderHelper.enableStandardItemLighting`), which dims every wall pixel by a constant
    /// factor depending on the quad's normal, so the wall's *rendered* colour is not its texture
    /// colour. `InFront` is depth-tested against the nearer wall and must match that measured
    /// baseline everywhere; `ThroughWalls` is not, and must darken it somewhere.
    #[test]
    fn through_walls_darkens_some_pixel_that_in_front_leaves_untouched() {
        let Some(mut gpu) = headless(64, 64) else {
            return;
        };
        fn make_wall(gpu: &Gpu) -> crate::entity::render::EntityPhase {
            let mut ep = crate::entity::render::EntityPhase::new(gpu);
            ep.upload_texture(gpu, "wall", &[0u8, 200, 0, 255], 1, 1);
            let mut verts = Vec::new();
            let quad = [
                (-4.0f32, -4.0f32),
                (4.0, -4.0),
                (4.0, 4.0),
                (-4.0, -4.0),
                (4.0, 4.0),
                (-4.0, 4.0),
            ];
            for (x, y) in quad {
                verts.push(EntityVertex {
                    pos: [x, y, 3.0],
                    uv: [0.5, 0.5],
                    normal: [0.0, 0.0, -1.0],
                    ..EntityVertex::default()
                });
            }
            ep.begin();
            ep.push("wall", &verts);
            ep.finish(gpu);
            ep
        }
        // Centred on the camera's forward ray so it lands inside the wall's screen coverage.
        let mut plate = plate("Steve");
        plate.rel = Vec3::new(0.0, -2.3, 6.0);
        let verts = plate.backdrop_vertices(0.0, 0.0, 60);

        let mut min_green = |pass: Option<NameplatePass>| -> u8 {
            let mut np = NameplatePhase::new(&gpu);
            np.begin();
            if let Some(pass) = pass {
                np.push(pass, &verts);
            }
            np.finish(&gpu);

            let mut phases = crate::pipeline::PhaseList::new();
            phases.clear_color = wgpu::Color::BLACK;
            phases.insert(
                crate::pipeline::PhaseOrder::ENTITIES,
                Box::new(make_wall(&gpu)),
            );
            phases.insert(crate::pipeline::PhaseOrder::UI, Box::new(np));
            let mut cam = Camera::default();
            cam.set_viewport(64, 64);
            cam.update();
            let frame = gpu.acquire().expect("acquire offscreen frame");
            phases.render(&gpu, frame.view(), &cam, 0.0);
            frame.present(&gpu.queue);
            let pixels = gpu.read_pixels().expect("read back");
            pixels.chunks(4).map(|p| p[1]).min().unwrap_or(255)
        };

        let baseline = min_green(None);
        assert!(
            baseline > 20,
            "the wall itself must have drawn something: {baseline}"
        );
        let through = min_green(Some(NameplatePass::ThroughWalls));
        let in_front = min_green(Some(NameplatePass::InFront));
        assert!(
            (i32::from(in_front) - i32::from(baseline)).abs() <= 2,
            "InFront is depth-tested and must match the un-occluded wall baseline: in_front={in_front} baseline={baseline}"
        );
        assert!(
            through < baseline,
            "ThroughWalls ignores depth and must darken the wall somewhere: through={through} baseline={baseline}"
        );
    }

    /// The glyph half of the seam, exercised on the GPU without a font: a 1x1 opaque-red atlas
    /// stands in for `ascii.png`, so this proves the pipeline, the bind group, the vertex layout
    /// and the tint modulation independently of anything in `mc-ui`. (The real `ascii.png`,
    /// texel for texel, is `crates/mc-ui/tests/nameplate_3d.rs`.)
    #[test]
    fn a_glyph_quad_reaches_the_frame_buffer_through_the_font_pipeline() {
        let Some(mut gpu) = headless(64, 64) else {
            return;
        };
        // A big label-space quad so it certainly covers the middle of the frame.
        let mut p = plate("Steve");
        p.rel = Vec3::new(0.0, -2.3, 6.0);
        let corners = [[-40.0, -40.0], [-40.0, 40.0], [40.0, -40.0], [40.0, 40.0]];
        let uv = [[0.5, 0.5]; 4];

        for (pass_kind, want_red) in [
            (NameplatePass::ThroughWalls, 32u8),
            (NameplatePass::InFront, 255u8),
        ] {
            let mut phase = NameplatePhase::new(&gpu);
            phase.upload_font_texture(&gpu, "font", &[255u8, 0, 0, 255], 1, 1);
            assert!(phase.has_font_texture("font"));
            let verts = p.glyph_vertices(0.0, 0.0, corners, uv, argb_to_rgba(pass_kind.argb()));
            phase.begin();
            phase.push_glyphs(pass_kind, "font", &verts);
            let (tw, inf) = phase.glyph_batch_counts();
            assert_eq!(
                tw + inf,
                1,
                "one draw call, in exactly one of the two passes"
            );
            assert_eq!(
                phase.glyph_vertex_counts().0 + phase.glyph_vertex_counts().1,
                6
            );
            phase.finish(&gpu);

            let mut phases = crate::pipeline::PhaseList::new();
            phases.clear_color = wgpu::Color::BLACK;
            phases.insert(crate::pipeline::PhaseOrder::UI, Box::new(phase));
            let mut cam = Camera::default();
            cam.set_viewport(64, 64);
            cam.update();
            let frame = gpu.acquire().expect("acquire offscreen frame");
            phases.render(&gpu, frame.view(), &cam, 0.0);
            frame.present(&gpu.queue);
            let pixels = gpu.read_pixels().expect("read back");
            let centre = pixel(&pixels, 64, 32, 32);
            // HEADLESS_FORMAT is Rgba8Unorm, so the blend is exact: red at the pass's own alpha
            // over a black clear is `alpha * 255` with no transfer function in the way.
            assert!(
                (i32::from(centre[0]) - i32::from(want_red)).abs() <= 1,
                "{pass_kind:?} tinted red to {centre:?}, expected r≈{want_red}"
            );
            assert_eq!(centre[1], 0, "the atlas is pure red: {centre:?}");
        }
    }

    /// The load-bearing claim, now with *glyph* geometry rather than only a backdrop:
    /// `ThroughWalls` text survives an occluder written into the shared depth buffer,
    /// `InFront` text does not.
    #[test]
    fn glyph_text_keeps_the_two_pass_depth_policy_behind_an_occluder() {
        let Some(mut gpu) = headless(64, 64) else {
            return;
        };
        fn wall(gpu: &Gpu) -> crate::entity::render::EntityPhase {
            let mut ep = crate::entity::render::EntityPhase::new(gpu);
            ep.upload_texture(gpu, "wall", &[0u8, 0, 255, 255], 1, 1);
            let quad = [
                (-4.0f32, -4.0f32),
                (4.0, -4.0),
                (4.0, 4.0),
                (-4.0, -4.0),
                (4.0, 4.0),
                (-4.0, 4.0),
            ];
            let verts: Vec<EntityVertex> = quad
                .into_iter()
                .map(|(x, y)| EntityVertex {
                    pos: [x, y, 3.0],
                    uv: [0.5, 0.5],
                    normal: [0.0, 0.0, -1.0],
                    ..EntityVertex::default()
                })
                .collect();
            ep.begin();
            ep.push("wall", &verts);
            ep.finish(gpu);
            ep
        }
        let mut p = plate("Steve");
        p.rel = Vec3::new(0.0, -2.3, 6.0);
        let corners = [[-40.0, -40.0], [-40.0, 40.0], [40.0, -40.0], [40.0, 40.0]];
        let uv = [[0.5, 0.5]; 4];

        let mut max_red = |pass_kind: NameplatePass| -> u8 {
            let mut phase = NameplatePhase::new(&gpu);
            phase.upload_font_texture(&gpu, "font", &[255u8, 0, 0, 255], 1, 1);
            // Opaque white tint in both passes, so the only difference under test is depth.
            let verts = p.glyph_vertices(0.0, 0.0, corners, uv, [1.0; 4]);
            phase.begin();
            phase.push_glyphs(pass_kind, "font", &verts);
            phase.finish(&gpu);

            let mut phases = crate::pipeline::PhaseList::new();
            phases.clear_color = wgpu::Color::BLACK;
            phases.insert(crate::pipeline::PhaseOrder::ENTITIES, Box::new(wall(&gpu)));
            phases.insert(crate::pipeline::PhaseOrder::UI, Box::new(phase));
            let mut cam = Camera::default();
            cam.set_viewport(64, 64);
            cam.update();
            let frame = gpu.acquire().expect("acquire offscreen frame");
            phases.render(&gpu, frame.view(), &cam, 0.0);
            frame.present(&gpu.queue);
            let pixels = gpu.read_pixels().expect("read back");
            pixels.chunks(4).map(|px| px[0]).max().unwrap_or(0)
        };

        let through = max_red(NameplatePass::ThroughWalls);
        let in_front = max_red(NameplatePass::InFront);
        assert!(
            through > 200,
            "ThroughWalls glyphs must paint over the wall: max red {through}"
        );
        assert!(
            in_front < 10,
            "InFront glyphs are behind the wall and must be depth-rejected: max red {in_front}"
        );
    }
}
