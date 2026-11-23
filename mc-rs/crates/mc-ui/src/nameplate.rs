//! Entity nameplate **text**, drawn on the `mc-ui` side of the font seam.
//!
//! Owned by **S12** (last task), even though the file lives in `mc-ui`.
//!
//! java: `client/renderer/entity/Render.renderLivingLabel` /
//! `Render.renderOffsetLivingLabel`, `RendererLivingEntity.renderName` (the sneaking branch),
//! `RenderPlayer.renderOffsetLivingLabel` (the scoreboard-score line).
//!
//! # Why this module is here and not in `mc-render`
//!
//! `mc-render` cannot reach [`crate::font::FontRenderer`]: the crate dependency runs
//! `mc-ui → mc-render`, never the reverse, so a nameplate drawn from inside `mc-render`
//! would need a dependency cycle. S12 therefore finished everything *except* the glyphs on
//! the `mc-render` side — the billboard matrix, the backdrop quad, and the two-pass depth
//! policy are all GPU-proven in `mc_render::entity::nameplate` — and exposed
//! `Nameplate::local_to_world` as the typed handoff. This module is the other half of that
//! seam: it turns label-local glyph geometry from S15's font into world-space quads.
//!
//! The alternative — moving S15's glyph atlas down into `mc-render` — would have duplicated a
//! large, finished, texel-exact module. This direction adds nothing to either crate's
//! dependency list: `mc-ui` already depends on `mc-render`.
//!
//! # The shape of the seam
//!
//! Only plain data crosses. `mc-render` exposes
//! [`NameplateGlyphVertex`](mc_render::entity::NameplateGlyphVertex) (a `#[repr(C)]` position +
//! atlas UV + straight RGBA), `Nameplate::glyph_vertices` (one label-local textured quad in
//! vanilla's `renderChar` corner order → six billboarded triangle vertices), and
//! `NameplatePhase::upload_font_texture`/`push_glyphs`, keyed by an opaque `&str`. Nothing in
//! `mc-render` knows a font exists; nothing here re-derives the billboard maths.
//!
//! ```text
//!   Nameplate (mc-render)            FontRenderer (mc-ui, S15)
//!         │                                   │
//!         │  layout(width) ── text_x/text_y ──▶│  draw_string  ──▶ GlyphQuad (label pixels)
//!         │                                                              │
//!         ◀── glyph_vertices(corners, uv, rgba) ─────────────────────────┘
//!         │
//!         └─▶ NameplateGlyphVertex ──▶ NameplatePhase::push_glyphs ──▶ wgpu
//! ```
//!
//! # What vanilla does, and is reproduced here
//!
//! * The label is centred on `-getStringWidth(str) / 2` — **integer** division, so an
//!   odd-width name sits half a pixel left of true centre.
//! * A `0x20000000`-style black backdrop (`color(0, 0, 0, 0.25)`) one pixel wider than the text
//!   on each side, drawn once, never writing depth.
//! * `§` formatting codes run through S15's real state machine, so colours, bold, italic,
//!   obfuscation, strikethrough and underline all behave as they do in chat.
//! * Two glyph passes: `553648127` (`0x20FFFFFF`) with depth off, then `-1` with depth on.
//! * Sneaking is a third behaviour: one depth-tested pass at the dim tint, half the range, and
//!   a quarter-block drop (`translate(0, 9.374999, 0)` after the scale).
//! * A player inside 10 blocks gets their scoreboard sidebar score on a second line below the
//!   name, and the name moves up by `FONT_HEIGHT * 1.15 * 0.02666667`.
//! * Out of range (64 blocks, 32 while sneaking) nothing is built at all.
//!
//! # Deviations, stated rather than hidden
//!
//! * `§m`/`§n` bars are emitted as untextured [`SolidBatch`] triangles, i.e. through the same
//!   pipeline as the backdrop. Vanilla draws them inside the text pass, so an `InFront` bar
//!   there *does* write depth and here does not. Nothing samples nameplate depth, so this is
//!   invisible; it is recorded because it is a real difference.
//! * `Render.canRenderName`'s team-visibility rules (`Team.EnumVisible`, the
//!   `ALWAYS`/`NEVER`/`HIDE_FOR_OTHER_TEAMS`/`HIDE_FOR_OWN_TEAM` switch) are **not** here: they
//!   need a scoreboard and the local player, which is S14/S21 state. The caller decides whether
//!   a label exists; this module decides what it looks like.
//!
//! # Example
//!
//! ```no_run
//! use mc_render::entity::{Nameplate, NameplatePhase};
//! use mc_ui::font::FontRenderer;
//! use mc_ui::nameplate::{self, NameplateRequest};
//! # fn demo(gpu: &mc_render::Gpu, font: &mut FontRenderer, phase: &mut NameplatePhase,
//! #         plate: &Nameplate) {
//! nameplate::upload_ascii_font(phase, gpu, font.data());
//! phase.begin();
//! if let Some(geometry) = nameplate::build(font, &NameplateRequest {
//!     plate,
//!     view_yaw: 0.0,
//!     view_pitch: 0.0,
//!     distance_sq: 36.0,
//!     score: None,
//! }) {
//!     geometry.submit(phase);
//! }
//! phase.finish(gpu);
//! # }
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use mc_assets::ResourceManager;
use mc_render::entity::nameplate::{
    NameplateLayout, NameplateStyle, NAMEPLATE_SCORE_GAP, NAMEPLATE_SCORE_RANGE_SQ,
};
use mc_render::entity::{
    argb_to_rgba, Nameplate, NameplateGlyphVertex, NameplatePass, NameplatePhase, NameplateVertex,
};
use mc_render::Gpu;

use crate::font::{FontData, FontRenderer, FontTexture, GlyphQuad, SolidRect, TextSink};
use crate::UiError;

/// A run of glyph triangles sharing one depth pass and one font sheet.
#[derive(Clone, Debug, PartialEq)]
pub struct GlyphBatch {
    /// Font drop-shadow glyphs; their in-front pass depth-tests without writing depth.
    pub shadow: bool,
    /// Which of vanilla's two depth passes this run belongs to.
    pub pass: NameplatePass,
    /// The font sheet its UVs index. See [`font_texture_key`].
    pub texture: FontTexture,
    /// Camera-relative triangles, three vertices per triangle.
    pub verts: Vec<NameplateGlyphVertex>,
}

/// A run of untextured triangles: the label backdrop, and `§m`/`§n` bars.
#[derive(Clone, Debug, PartialEq)]
pub struct SolidBatch {
    /// Which of vanilla's two depth passes this run belongs to.
    pub pass: NameplatePass,
    /// Camera-relative triangles, three vertices per triangle.
    pub verts: Vec<NameplateVertex>,
}

/// Everything one call to [`build`] produced, in the shape
/// [`NameplatePhase`] consumes.
///
/// A single call can produce more than one label — a player within 10 blocks gets a scoreboard
/// score line as well as their name — which is why these are `Vec`s rather than one batch each.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NameplateGeometry {
    /// Backdrop quads and formatting bars.
    pub solid: Vec<SolidBatch>,
    /// Glyphs, batched by `(pass, sheet)` in emission order.
    pub glyphs: Vec<GlyphBatch>,
}

impl NameplateGeometry {
    /// Hand everything to the phase. This is the only call that crosses the crate boundary.
    ///
    /// Call [`NameplatePhase::begin`] before the first label of the frame and
    /// [`NameplatePhase::finish`] after the last, exactly as for entity geometry.
    pub fn submit(&self, phase: &mut NameplatePhase) {
        self.submit_to(phase);
    }

    /// [`Self::submit`] into any [`NameplateTarget`]; the phase is one such target.
    pub fn submit_to<T: NameplateTarget + ?Sized>(&self, target: &mut T) {
        for batch in &self.solid {
            target.push_solid(batch.pass, &batch.verts);
        }
        for batch in &self.glyphs {
            if batch.shadow && batch.pass == NameplatePass::InFront {
                target.push_glyph_shadow_run(batch.texture, &batch.verts);
            } else {
                target.push_glyph_run(batch.pass, batch.texture, &batch.verts);
            }
        }
    }

    /// Every distinct font sheet these glyphs sample, in first-use order.
    #[must_use]
    pub fn textures(&self) -> Vec<FontTexture> {
        let mut out: Vec<FontTexture> = Vec::new();
        for batch in &self.glyphs {
            if !out.contains(&batch.texture) {
                out.push(batch.texture);
            }
        }
        out
    }

    /// Total glyph vertices, across both passes and every sheet.
    #[must_use]
    pub fn glyph_vertex_count(&self) -> usize {
        self.glyphs.iter().map(|b| b.verts.len()).sum()
    }

    /// True when nothing at all would be drawn.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.solid.is_empty() && self.glyphs.is_empty()
    }
}

/// One label to build geometry for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NameplateRequest<'a> {
    /// Position, hitbox height, child/sneaking flags, and the already-formatted display name.
    ///
    /// java: `entity.getDisplayName().getFormattedText()` — the `§` codes are still in it, and
    /// S15's font renderer is what interprets them.
    pub plate: &'a Nameplate,
    /// Camera yaw in degrees. java: `RenderManager.playerViewY`
    pub view_yaw: f32,
    /// Camera pitch in degrees. java: `RenderManager.playerViewX`
    pub view_pitch: f32,
    /// Squared distance from the view entity, in blocks². java: `getDistanceSqToEntity`
    pub distance_sq: f32,
    /// The scoreboard sidebar line for this player, if the server has an objective in display
    /// slot 2. java: `RenderPlayer.renderOffsetLivingLabel` builds it as
    /// `score.getScorePoints() + " " + objective.getDisplayName()`.
    pub score: Option<&'a str>,
}

/// Build the geometry for one label, or `None` when it is out of range.
///
/// java: `RendererLivingEntity.renderName` — the range gate, the sneaking branch, and
/// `renderOffsetLivingLabel`/`renderLivingLabel` for everything else.
///
/// The font is taken by `&mut` because vanilla's renderer is a state machine with a cursor and
/// five style flags in fields; see [`FontRenderer`].
#[must_use]
pub fn build(font: &mut FontRenderer, req: &NameplateRequest<'_>) -> Option<NameplateGeometry> {
    // java: `if (d0 < (double)(f * f))` in renderName, re-checked as `<=` in renderLivingLabel.
    if !req.plate.in_range(req.distance_sq) {
        return None;
    }
    let mut out = NameplateGeometry::default();
    let mut name = req.plate.clone();

    // java: RenderPlayer.renderOffsetLivingLabel — the score goes *below*, and the name is
    // lifted out of its way. Only players get here, and only in the non-sneaking branch,
    // because the sneaking branch never calls renderOffsetLivingLabel at all.
    if let Some(score) = req.score {
        if !req.plate.sneaking && req.distance_sq < NAMEPLATE_SCORE_RANGE_SQ {
            let mut line = req.plate.clone();
            line.text = score.to_owned();
            push_label(
                &mut out,
                font,
                &line,
                req.view_yaw,
                req.view_pitch,
                NameplateStyle::VANILLA,
            );
            name.rel.y += NAMEPLATE_SCORE_GAP;
        }
    }

    push_label(
        &mut out,
        font,
        &name,
        req.view_yaw,
        req.view_pitch,
        NameplateStyle::VANILLA,
    );
    Some(out)
}

/// One label: its backdrop, then its glyphs once per depth pass, drawn in `style` (the Nametags
/// mod; [`NameplateStyle::VANILLA`] is exactly `Render.renderLivingLabel`).
fn push_label(
    out: &mut NameplateGeometry,
    font: &mut FontRenderer,
    plate: &Nameplate,
    view_yaw: f32,
    view_pitch: f32,
    style: NameplateStyle,
) {
    // java: `int j = fontrenderer.getStringWidth(str) / 2;` and the drawString x below it both
    // derive from this one measurement, so it is taken once.
    let width = font.string_width(&plate.text);
    let layout = plate.layout(width);

    if style.is_vanilla() {
        out.solid.push(SolidBatch {
            pass: plate.backdrop_pass(),
            verts: plate
                .backdrop_vertices(view_yaw, view_pitch, width)
                .to_vec(),
        });
    } else if style.background_alpha > 0.0 {
        let m = plate.matrix_scaled(view_yaw, view_pitch, style.scale);
        let b = layout.backdrop;
        let rgba = [b.rgba[0], b.rgba[1], b.rgba[2], style.background_alpha];
        let corners = [(b.x0, b.y0), (b.x1, b.y0), (b.x1, b.y1), (b.x0, b.y1)]
            .map(|(x, y)| m.transform_point3(glam::Vec3::new(x, y, 0.0)));
        out.solid.push(SolidBatch {
            pass: plate.backdrop_pass(),
            verts: [0usize, 1, 2, 0, 2, 3]
                .iter()
                .map(|&i| NameplateVertex {
                    pos: corners[i].into(),
                    color: rgba,
                })
                .collect(),
        });
    }

    let matrix = plate.matrix_scaled(view_yaw, view_pitch, style.scale);
    for &pass in plate.passes() {
        let tint = plate.argb_for(pass);
        let mut sink = BillboardSink {
            matrix,
            pass,
            shadow: false,
            glyphs: Vec::new(),
            bars: Vec::new(),
        };
        // java: `fontrenderer.drawString(str, -fontrenderer.getStringWidth(str) / 2, i, colour)`
        // — the three-int overload, i.e. no drop shadow (unless the Nametags mod asks for one).
        font.draw_string(
            &mut sink,
            &plate.text,
            layout.text_x,
            layout.text_y,
            tint as i32,
            style.text_shadow,
        );
        if !sink.bars.is_empty() {
            out.solid.push(SolidBatch {
                pass,
                verts: sink.bars,
            });
        }
        out.glyphs.append(&mut sink.glyphs);
    }
}

/// A [`TextSink`] that billboards each quad as the font emits it.
///
/// Batching is greedy over consecutive quads sharing a sheet, which for a real name means one
/// batch: mixing `ascii.png` and a unicode page inside one string only happens when the unicode
/// flag is off and the name contains a character `ascii.png` does not have.
struct BillboardSink {
    shadow: bool,
    matrix: glam::Mat4,
    pass: NameplatePass,
    glyphs: Vec<GlyphBatch>,
    bars: Vec<NameplateVertex>,
}

impl TextSink for BillboardSink {
    fn shadow_pass(&mut self, shadow: bool) {
        self.shadow = shadow;
    }
    fn glyph(&mut self, quad: GlyphQuad) {
        let verts = Nameplate::glyph_vertices_with_matrix(
            self.matrix,
            quad.pos,
            quad.uv,
            argb_to_rgba(quad.color),
        );
        match self.glyphs.last_mut() {
            Some(b) if b.texture == quad.texture && b.shadow == self.shadow => {
                b.verts.extend_from_slice(&verts)
            }
            _ => self.glyphs.push(GlyphBatch {
                shadow: self.shadow,
                pass: self.pass,
                texture: quad.texture,
                verts: verts.to_vec(),
            }),
        }
    }

    fn bar(&mut self, rect: SolidRect) {
        let rgba = argb_to_rgba(rect.color);
        let corners = [
            (rect.x0, rect.y0),
            (rect.x1, rect.y0),
            (rect.x1, rect.y1),
            (rect.x0, rect.y1),
        ];
        for i in [0usize, 1, 2, 0, 2, 3] {
            let (x, y) = corners[i];
            self.bars.push(NameplateVertex {
                pos: self
                    .matrix
                    .transform_point3(glam::Vec3::new(x, y, 0.0))
                    .into(),
                color: rgba,
            });
        }
    }
}

// ---------------------------------------------------------------------- label layout cache

/// Where finished label geometry goes. [`NameplatePhase`] is the production target; tests use
/// a recorder with the phase's exact stream/batch semantics, so the cached and uncached paths
/// can be compared without a GPU.
pub trait NameplateTarget {
    /// In-front drop shadows test world depth but must not occlude their coplanar text.
    fn push_glyph_shadow_run(&mut self, texture: FontTexture, verts: &[NameplateGlyphVertex]) {
        self.push_glyph_run(NameplatePass::InFront, texture, verts);
    }
    /// Untextured triangles (backdrop, `§m`/`§n` bars) for one depth pass.
    fn push_solid(&mut self, pass: NameplatePass, verts: &[NameplateVertex]);
    /// One run of glyph triangles that all sample `texture`.
    fn push_glyph_run(
        &mut self,
        pass: NameplatePass,
        texture: FontTexture,
        verts: &[NameplateGlyphVertex],
    );
}

impl NameplateTarget for NameplatePhase {
    fn push_glyph_shadow_run(&mut self, texture: FontTexture, verts: &[NameplateGlyphVertex]) {
        self.push_shadow_glyphs(font_texture_key_str(texture), verts);
    }
    fn push_solid(&mut self, pass: NameplatePass, verts: &[NameplateVertex]) {
        self.push(pass, verts);
    }

    fn push_glyph_run(
        &mut self,
        pass: NameplatePass,
        texture: FontTexture,
        verts: &[NameplateGlyphVertex],
    ) {
        self.push_glyphs(pass, font_texture_key_str(texture), verts);
    }
}

/// One glyph of a cached label, exactly as [`FontRenderer`] emitted it, minus the alpha byte.
#[derive(Clone, Copy, Debug, PartialEq)]
struct CachedGlyph {
    shadow: bool,
    texture: FontTexture,
    /// Label-pixel corners in `renderChar` order, as the font produced them.
    pos: [[f32; 2]; 4],
    uv: [[f32; 2]; 4],
    /// `0x00RRGGBB`. The pass supplies the alpha byte; see [`effective_alpha`].
    rgb: u32,
}

/// A cached `§m`/`§n` bar, alpha stripped like [`CachedGlyph::rgb`].
#[derive(Clone, Copy, Debug, PartialEq)]
struct CachedBar {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    rgb: u32,
}

/// One label's layout: the font's output for its text, drawn once at the label-local origin
/// vanilla uses, in label pixels. Only the billboard matrix is applied per frame.
#[derive(Clone, Debug, PartialEq)]
struct CachedLayout {
    /// `Nameplate::layout` for this text's measured width: backdrop and glyph origin.
    ///
    /// Recorded with the first pass's tint. Every nameplate tint is white (`0x20FFFFFF`,
    /// `0xFFFFFFFF`), which is what makes the passes differ only in alpha; a draw with any other
    /// tint RGB bypasses the cache (`push_label_cached`) rather than relying on that.
    layout: NameplateLayout,
    glyphs: Vec<CachedGlyph>,
    bars: Vec<CachedBar>,
    last_used: u64,
}

impl CachedLayout {
    /// This layout's share of [`NameplateCache::MAX_CACHED_PRIMITIVES`].
    fn primitives(&self) -> usize {
        self.glyphs.len() + self.bars.len()
    }
}

/// Records one draw as label-local geometry.
struct RecordSink<'a> {
    shadow: bool,
    glyphs: &'a mut Vec<CachedGlyph>,
    bars: &'a mut Vec<CachedBar>,
}

impl TextSink for RecordSink<'_> {
    fn shadow_pass(&mut self, shadow: bool) {
        self.shadow = shadow;
    }
    fn glyph(&mut self, quad: GlyphQuad) {
        self.glyphs.push(CachedGlyph {
            shadow: self.shadow,
            texture: quad.texture,
            pos: quad.pos,
            uv: quad.uv,
            rgb: quad.color & 0x00FF_FFFF,
        });
    }

    fn bar(&mut self, rect: SolidRect) {
        self.bars.push(CachedBar {
            x0: rect.x0,
            y0: rect.y0,
            x1: rect.x1,
            y1: rect.y1,
            rgb: rect.color & 0x00FF_FFFF,
        });
    }
}

/// The alpha byte every glyph of a draw tinted `tint` carries.
///
/// java: `FontRenderer.renderString` forces alpha opaque when `(color & 0xFC000000) == 0`, then
/// keeps that alpha for the whole string: a `§` colour is `(base & 0xFF000000) | rgb` and `§r`
/// restores the base. So alpha is a property of the draw, never of the glyph.
fn effective_alpha(tint: u32) -> u32 {
    if tint & 0xFC00_0000 == 0 {
        0xFF00_0000
    } else {
        tint & 0xFF00_0000
    }
}

/// Counters for [`NameplateCache`], for the profiler/telemetry. Monotonic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NameplateCacheStats {
    /// Labels drawn from a cached layout.
    pub hits: u64,
    /// Labels laid out and cached.
    pub misses: u64,
    /// Labels that can consume the `§k` stream, drawn the uncached way every frame.
    pub volatile: u64,
}

/// Per-text layout cache for world-space labels: entity names, holograms, score lines.
///
/// java has no such cache — `renderLivingLabel` re-runs `FontRenderer` twice per label per
/// frame — but the output of those draws is a pure function of `(text, sneaking, font state)`
/// for every string that does not use `§k`, so caching it is invisible:
///
/// * glyph corners, UVs and colours are recorded exactly as the font emits them, and the same
///   billboard matrix maths runs on them per frame, so every vertex is **bit-identical** to
///   the uncached path (pinned by `cached_labels_match_the_uncached_path_bit_for_bit`);
/// * the two depth passes share one layout and one set of billboarded corners; they differ
///   only in alpha, exactly as the font would have produced them;
/// * text that can reach `§k` ([`FontRenderer::may_obfuscate`]) is never cached: it is drawn
///   twice per frame as before, so the shared `fontRandom` stream is consumed in the same
///   order and amount and the two passes keep their independently rolled glyphs;
/// * any change to the font's layout inputs ([`FontRenderer::layout_generation`]) drops
///   everything.
///
/// Entries unused for [`Self::EVICT_AFTER_FRAMES`] frames are dropped by [`Self::begin_frame`].
/// Label text is server-chosen and can change every tick, so the cache is also bounded by size:
/// text over [`Self::MAX_TEXT_BYTES`] is never cached, and a cache holding more than
/// [`Self::MAX_CACHED_PRIMITIVES`] glyphs and bars starts over.
#[derive(Debug, Default)]
pub struct NameplateCache {
    /// Keyed by label text; slot 0 standing, slot 1 sneaking (the deadmau5 lift and the
    /// backdrop pass depend on it, so the two are different layouts of the same text).
    entries: HashMap<String, [Option<CachedLayout>; 2]>,
    /// Glyphs plus bars across every cached layout, for the size bound.
    cached_primitives: usize,
    /// Texts [`FontRenderer::may_obfuscate`] flagged, so the (allocating) check runs once per
    /// text rather than once per frame. Cleared with the entries.
    volatile_texts: HashSet<String>,
    generation: u64,
    frame: u64,
    /// Every sheet any cached label samples, for [`Self::ensure_textures`].
    sheets: Vec<FontTexture>,
    /// Sheets that would not load; not retried until the font changes.
    failed_sheets: Vec<FontTexture>,
    world: Vec<[[f32; 3]; 4]>,
    glyph_scratch: Vec<NameplateGlyphVertex>,
    solid_scratch: Vec<NameplateVertex>,
    stats: NameplateCacheStats,
    /// The Nametags mod's style; [`NameplateStyle::VANILLA`] unless the mod is on.
    /// The shadow is part of every cached layout; scale and backdrop alpha apply per draw.
    style: NameplateStyle,
}

impl NameplateCache {
    /// Frames an entry may go unused before [`Self::begin_frame`] drops it.
    pub const EVICT_AFTER_FRAMES: u64 = 120;
    /// Longest label text, in UTF-8 bytes, that is cached; longer labels are drawn the uncached
    /// way every frame. The same limit as the font's glyph-run cache.
    pub const MAX_TEXT_BYTES: usize = 256;
    /// Glyphs plus bars (each under 100 bytes) cached before the cache starts over. A busy
    /// lobby needs a few thousand; without a bound, hologram text rewritten every tick could
    /// grow the cache faster than the frame-age eviction drains it (phase 4b review S1).
    pub const MAX_CACHED_PRIMITIVES: usize = 65_536;
    /// Distinct `§k` texts remembered before that set starts over.
    pub const MAX_VOLATILE_TEXTS: usize = 1024;

    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance the frame clock and drop entries nobody has drawn for a while.
    ///
    /// Call once per frame before the first [`submit_cached`]. Forgetting to call it is safe;
    /// the cache then just never shrinks.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
        if self.frame.is_multiple_of(Self::EVICT_AFTER_FRAMES) {
            self.volatile_texts.clear();
            let horizon = self.frame.saturating_sub(Self::EVICT_AFTER_FRAMES);
            let mut freed = 0;
            self.entries.retain(|_, slots| {
                for slot in slots.iter_mut() {
                    if slot.as_ref().is_some_and(|l| l.last_used < horizon) {
                        freed += slot.take().map_or(0, |l| l.primitives());
                    }
                }
                slots.iter().any(Option::is_some)
            });
            self.cached_primitives -= freed;
        }
    }

    /// Cumulative hit/miss/volatile counts.
    #[must_use]
    pub fn stats(&self) -> NameplateCacheStats {
        self.stats
    }

    /// Draw every following label in `style` (the Nametags mod). A change of
    /// [`NameplateStyle::text_shadow`] drops the cached layouts, which record the shadow glyphs;
    /// scale and backdrop alpha are applied per draw and keep them. Pair it with
    /// `NameplatePhase::set_style`, which picks the depth state shadowed text needs.
    pub fn set_style(&mut self, style: NameplateStyle) {
        if style.text_shadow != self.style.text_shadow {
            self.entries.clear();
            self.cached_primitives = 0;
        }
        self.style = style;
    }

    /// The style last set with [`Self::set_style`].
    #[must_use]
    pub fn style(&self) -> NameplateStyle {
        self.style
    }

    /// Distinct label texts currently cached.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when nothing is cached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn sync_generation(&mut self, font: &FontRenderer) {
        let generation = font.layout_generation();
        if self.generation != generation {
            self.entries.clear();
            self.cached_primitives = 0;
            self.volatile_texts.clear();
            self.sheets.clear();
            self.failed_sheets.clear();
            self.generation = generation;
        }
    }

    fn note_sheet(&mut self, texture: FontTexture) {
        if !self.sheets.contains(&texture) {
            self.sheets.push(texture);
        }
    }

    /// Make every font sheet a label has used resident in `phase`, loading missing ones.
    ///
    /// NAME-5: before this, nothing but the sign path ever uploaded a unicode page for
    /// nameplates, so `❤`, `✪`, `⚔` (and *every* glyph under `forceUnicodeFont`) drew as the
    /// 1×1 white fallback — solid boxes. Vanilla loads the page on first use
    /// (`FontRenderer.loadGlyphTexture`), synchronously, which is what this does; a page that
    /// will not load is remembered and keeps the box fallback rather than being retried every
    /// frame. Cheap when nothing is missing: one key lookup per sheet ever used.
    pub fn ensure_textures(
        &mut self,
        phase: &mut NameplatePhase,
        gpu: &Gpu,
        assets: &ResourceManager,
        data: &FontData,
    ) {
        for &texture in &self.sheets {
            if self.failed_sheets.contains(&texture)
                || phase.has_font_texture(font_texture_key_str(texture))
            {
                continue;
            }
            let loaded = match texture {
                FontTexture::Ascii => {
                    upload_ascii_font(phase, gpu, data);
                    true
                }
                FontTexture::StandardGalactic => {
                    upload_standard_galactic(phase, gpu, assets).is_ok()
                }
                FontTexture::UnicodePage(page) => {
                    upload_unicode_page(phase, gpu, assets, page).is_ok()
                }
            };
            if !loaded {
                self.failed_sheets.push(texture);
            }
        }
    }
}

/// The cached equivalent of `build(font, req)` followed by
/// [`NameplateGeometry::submit_to`]`(target)`: same range gate, same score line, same vertices
/// in the same per-stream order, bit for bit. Returns `false` when the label is out of range.
pub fn submit_cached<T: NameplateTarget + ?Sized>(
    font: &mut FontRenderer,
    cache: &mut NameplateCache,
    req: &NameplateRequest<'_>,
    target: &mut T,
) -> bool {
    // java: `if (d0 < (double)(f * f))` in renderName, re-checked as `<=` in renderLivingLabel.
    if !req.plate.in_range(req.distance_sq) {
        return false;
    }
    cache.sync_generation(font);
    let mut name_rel = req.plate.rel;
    // java: RenderPlayer.renderOffsetLivingLabel — see `build`. The gap scales with the label
    // (`* 1.0` for the vanilla style is exact).
    if let Some(score) = req.score {
        if !req.plate.sneaking && req.distance_sq < NAMEPLATE_SCORE_RANGE_SQ {
            push_label_cached(font, cache, req.plate, score, name_rel, req, target);
            name_rel.y += NAMEPLATE_SCORE_GAP * cache.style.scale;
        }
    }
    push_label_cached(
        font,
        cache,
        req.plate,
        &req.plate.text,
        name_rel,
        req,
        target,
    );
    true
}

/// One label (`text` drawn at `rel` with `plate`'s height/child/sneaking): backdrop, then glyphs
/// once per depth pass, from the cache when the text allows it.
fn push_label_cached<T: NameplateTarget + ?Sized>(
    font: &mut FontRenderer,
    cache: &mut NameplateCache,
    plate: &Nameplate,
    text: &str,
    rel: glam::Vec3,
    req: &NameplateRequest<'_>,
    target: &mut T,
) {
    // `String::new()` does not allocate; nothing below reads `text` from this plate except
    // `layout`, which is only ever called on a plate carrying the real text.
    let placed = Nameplate {
        text: String::new(),
        rel,
        height: plate.height,
        child: plate.child,
        sneaking: plate.sneaking,
    };
    let first_tint = plate.argb_for(plate.passes()[0]);
    let frame = cache.frame;
    let slot = usize::from(plate.sneaking);
    let cacheable =
        first_tint & 0x00FF_FFFF == 0x00FF_FFFF && text.len() <= NameplateCache::MAX_TEXT_BYTES;
    if cacheable {
        let NameplateCache {
            entries,
            world,
            glyph_scratch,
            solid_scratch,
            stats,
            style,
            ..
        } = &mut *cache;
        if let Some(cached) = entries.get_mut(text).and_then(|slots| slots[slot].as_mut()) {
            // The steady state: one hash of the text, no font work, no allocation.
            stats.hits += 1;
            cached.last_used = frame;
            emit_cached(
                cached,
                &placed,
                req.view_yaw,
                req.view_pitch,
                *style,
                world,
                glyph_scratch,
                solid_scratch,
                target,
            );
            return;
        }
    }
    let volatile = !cacheable || cache.volatile_texts.contains(text) || font.may_obfuscate(text);
    if volatile {
        // `§k` (or a tint the cache does not model, or over-long text): exactly the uncached
        // path, every frame, so the shared obfuscation stream sees the same draws in the same
        // order as before.
        cache.stats.volatile += 1;
        if cacheable && !cache.volatile_texts.contains(text) {
            if cache.volatile_texts.len() >= NameplateCache::MAX_VOLATILE_TEXTS {
                cache.volatile_texts.clear();
            }
            cache.volatile_texts.insert(text.to_owned());
        }
        let label = Nameplate {
            text: text.to_owned(),
            ..placed
        };
        let mut out = NameplateGeometry::default();
        push_label(
            &mut out,
            font,
            &label,
            req.view_yaw,
            req.view_pitch,
            cache.style,
        );
        for batch in &out.glyphs {
            cache.note_sheet(batch.texture);
        }
        out.submit_to(target);
        return;
    }

    // Miss: lay the text out once, exactly as `push_label`'s first pass would.
    cache.stats.misses += 1;
    // java: `int j = fontrenderer.getStringWidth(str) / 2;` — see `push_label`.
    let width = font.string_width(text);
    let layout = Nameplate {
        text: text.to_owned(),
        ..placed
    }
    .layout(width);
    let mut glyphs = Vec::new();
    let mut bars = Vec::new();
    font.draw_string(
        &mut RecordSink {
            shadow: false,
            glyphs: &mut glyphs,
            bars: &mut bars,
        },
        text,
        layout.text_x,
        layout.text_y,
        first_tint as i32,
        // X6: the Nametags mod's shadow is recorded into the layout (vanilla: none).
        cache.style.text_shadow,
    );
    for g in &glyphs {
        cache.note_sheet(g.texture);
    }
    let cached = CachedLayout {
        layout,
        glyphs,
        bars,
        last_used: frame,
    };
    emit_cached(
        &cached,
        &placed,
        req.view_yaw,
        req.view_pitch,
        cache.style,
        &mut cache.world,
        &mut cache.glyph_scratch,
        &mut cache.solid_scratch,
        target,
    );
    let weight = cached.primitives();
    if cache.cached_primitives + weight > NameplateCache::MAX_CACHED_PRIMITIVES {
        cache.entries.clear();
        cache.cached_primitives = 0;
    }
    let replaced = cache.entries.entry(text.to_owned()).or_default()[slot].replace(cached);
    cache.cached_primitives =
        cache.cached_primitives + weight - replaced.map_or(0, |l| l.primitives());
}

/// Billboard a cached layout and push it, reproducing `push_label` + `submit_to` exactly.
#[allow(clippy::too_many_arguments)]
fn emit_cached<T: NameplateTarget + ?Sized>(
    cached: &CachedLayout,
    placed: &Nameplate,
    view_yaw: f32,
    view_pitch: f32,
    style: NameplateStyle,
    world: &mut Vec<[[f32; 3]; 4]>,
    glyph_scratch: &mut Vec<NameplateGlyphVertex>,
    solid_scratch: &mut Vec<NameplateVertex>,
    target: &mut T,
) {
    // `matrix_scaled(.., 1.0)` is `matrix` itself, so the vanilla style is bit-identical.
    let m = placed.matrix_scaled(view_yaw, view_pitch, style.scale);
    let place =
        |x: f32, y: f32| -> [f32; 3] { m.transform_point3(glam::Vec3::new(x, y, 0.0)).into() };

    // The backdrop: `Nameplate::backdrop_vertices` with the matrix and layout already in hand.
    // X6: the Nametags mod sets its alpha; zero leaves it out.
    let b = cached.layout.backdrop;
    let backdrop_rgba = if style.is_vanilla() {
        b.rgba
    } else {
        [b.rgba[0], b.rgba[1], b.rgba[2], style.background_alpha]
    };
    if backdrop_rgba[3] > 0.0 {
        let corners = [
            place(b.x0, b.y0),
            place(b.x1, b.y0),
            place(b.x1, b.y1),
            place(b.x0, b.y1),
        ];
        solid_scratch.clear();
        solid_scratch.extend([0usize, 1, 2, 0, 2, 3].map(|i| NameplateVertex {
            pos: corners[i],
            color: backdrop_rgba,
        }));
        target.push_solid(placed.backdrop_pass(), solid_scratch);
    }

    // Each corner is billboarded once per frame, not once per pass.
    world.clear();
    world.extend(
        cached
            .glyphs
            .iter()
            .map(|g| g.pos.map(|[x, y]| place(x, y))),
    );

    // GL_TRIANGLE_STRIP over [0,1,2,3] is triangles (0,1,2) and (2,1,3), as in
    // `Nameplate::glyph_vertices_with_matrix`.
    const STRIP: [usize; 6] = [0, 1, 2, 2, 1, 3];
    for &pass in placed.passes() {
        let alpha = effective_alpha(placed.argb_for(pass));
        if !cached.bars.is_empty() {
            solid_scratch.clear();
            for bar in &cached.bars {
                let rgba = argb_to_rgba(alpha | bar.rgb);
                let corners = [
                    place(bar.x0, bar.y0),
                    place(bar.x1, bar.y0),
                    place(bar.x1, bar.y1),
                    place(bar.x0, bar.y1),
                ];
                for i in [0usize, 1, 2, 0, 2, 3] {
                    solid_scratch.push(NameplateVertex {
                        pos: corners[i],
                        color: rgba,
                    });
                }
            }
            target.push_solid(pass, solid_scratch);
        }
        // One push per same-sheet run, the runs `BillboardSink` would have produced.
        let mut i = 0;
        while i < cached.glyphs.len() {
            let texture = cached.glyphs[i].texture;
            let shadow = cached.glyphs[i].shadow;
            glyph_scratch.clear();
            while i < cached.glyphs.len()
                && cached.glyphs[i].texture == texture
                && cached.glyphs[i].shadow == shadow
            {
                let glyph = &cached.glyphs[i];
                let corners = &world[i];
                let rgba = argb_to_rgba(alpha | glyph.rgb);
                glyph_scratch.extend(STRIP.map(|k| NameplateGlyphVertex {
                    pos: corners[k],
                    uv: glyph.uv[k],
                    color: rgba,
                }));
                i += 1;
            }
            if shadow && pass == NameplatePass::InFront {
                target.push_glyph_shadow_run(texture, glyph_scratch);
            } else {
                target.push_glyph_run(pass, texture, glyph_scratch);
            }
        }
    }
}

// ---------------------------------------------------------------------- font atlas plumbing

/// The opaque key a font sheet is uploaded to `mc-render` under, and drawn with.
///
/// `mc-render` never parses these; naming the sheets is a font concern and stays on this side
/// of the seam.
#[must_use]
pub fn font_texture_key(texture: FontTexture) -> String {
    font_texture_key_str(texture).to_owned()
}

/// [`font_texture_key`] without the allocation: the keys are fixed, so they are built once.
#[must_use]
pub fn font_texture_key_str(texture: FontTexture) -> &'static str {
    static PAGES: OnceLock<Vec<String>> = OnceLock::new();
    match texture {
        FontTexture::Ascii => "mc-ui/nameplate/ascii",
        FontTexture::StandardGalactic => "mc-ui/nameplate/ascii_sga",
        FontTexture::UnicodePage(page) => PAGES.get_or_init(|| {
            (0..=255u16)
                .map(|page| format!("mc-ui/nameplate/unicode_page_{page:02x}"))
                .collect()
        })[usize::from(page)]
        .as_str(),
    }
}

/// Upload `ascii.png` as a nameplate font atlas, reusing the copy [`FontData`] already decoded.
pub fn upload_ascii_font(phase: &mut NameplatePhase, gpu: &Gpu, data: &FontData) {
    let (width, height) = data.ascii_size();
    phase.upload_font_texture(
        gpu,
        &font_texture_key(FontTexture::Ascii),
        data.ascii_rgba(),
        width,
        height,
    );
}

/// Upload one `unicode_page_XX.png` out of the resource packs.
///
/// # Errors
/// [`UiError::FontAsset`] when the page is missing from every pack, [`UiError::FontTexture`]
/// when it will not decode as a PNG.
pub fn upload_unicode_page(
    phase: &mut NameplatePhase,
    gpu: &Gpu,
    assets: &ResourceManager,
    page: u8,
) -> Result<(), UiError> {
    let location = FontTexture::UnicodePage(page).location();
    let bytes = assets
        .load(&location)
        .map_err(|e| UiError::FontAsset(location.to_string(), e))?;
    let img = mc_assets::decode_texture_png(&bytes)
        .map_err(|e| UiError::FontTexture {
            location: location.to_string(),
            reason: e.to_string(),
        })?
        .to_rgba8();
    let (width, height) = (img.width(), img.height());
    phase.upload_font_texture(
        gpu,
        &font_texture_key(FontTexture::UnicodePage(page)),
        &img.into_raw(),
        width,
        height,
    );
    Ok(())
}

/// Upload the Standard Galactic alphabet sheet used by enchanting-table offer names.
pub fn upload_standard_galactic(
    phase: &mut NameplatePhase,
    gpu: &Gpu,
    assets: &ResourceManager,
) -> Result<(), UiError> {
    let texture = FontTexture::StandardGalactic;
    let location = texture.location();
    let bytes = assets
        .load(&location)
        .map_err(|e| UiError::FontAsset(location.to_string(), e))?;
    let img = mc_assets::decode_texture_png(&bytes)
        .map_err(|e| UiError::FontTexture {
            location: location.to_string(),
            reason: e.to_string(),
        })?
        .to_rgba8();
    let (width, height) = (img.width(), img.height());
    phase.upload_font_texture(
        gpu,
        &font_texture_key(texture),
        &img.into_raw(),
        width,
        height,
    );
    Ok(())
}

/// Make sure every sheet `geometry` samples is resident, loading the missing ones.
///
/// Returns the sheets that could not be loaded. A missing sheet is not fatal: `mc-render` falls
/// back to a 1×1 white texel, so the name draws as tinted boxes rather than vanishing or
/// stalling the frame — the same policy S12 uses for a skin that has not downloaded yet.
pub fn ensure_textures(
    phase: &mut NameplatePhase,
    gpu: &Gpu,
    assets: &ResourceManager,
    data: &FontData,
    geometry: &NameplateGeometry,
) -> Vec<FontTexture> {
    let mut missing = Vec::new();
    for texture in geometry.textures() {
        if phase.has_font_texture(&font_texture_key(texture)) {
            continue;
        }
        match texture {
            FontTexture::Ascii => upload_ascii_font(phase, gpu, data),
            FontTexture::StandardGalactic => {
                if upload_standard_galactic(phase, gpu, assets).is_err() {
                    missing.push(texture);
                }
            }
            FontTexture::UnicodePage(page) => {
                if upload_unicode_page(phase, gpu, assets, page).is_err() {
                    missing.push(texture);
                }
            }
        }
    }
    missing
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use mc_render::entity::nameplate::{NAMEPLATE_SCALE, NAMEPLATE_SNEAK_DROP};

    /// Every ascii cell is inked in columns 0..=3, so every default glyph measures 5 pixels.
    /// Same synthetic sheet S15's own unit tests use, and for the same reason: the assertions
    /// below are about the layout algorithm, not about the artwork. The real `ascii.png` is
    /// checked texel-for-texel in `tests/nameplate_3d.rs`.
    fn synthetic_ascii_png() -> Vec<u8> {
        let mut img = image::RgbaImage::new(128, 128);
        for cell in 0..256u32 {
            let cx = (cell % 16) * 8;
            let cy = (cell / 16) * 8;
            for col in 0..4 {
                img.put_pixel(cx + col, cy + 2, image::Rgba([255, 255, 255, 255]));
            }
        }
        let mut buf = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut buf, image::ImageFormat::Png)
            .expect("encode");
        buf.into_inner()
    }

    fn font() -> FontRenderer {
        let sizes = [0u8; 0x1_0000];
        let data = FontData::from_parts(&synthetic_ascii_png(), &sizes).expect("synthetic font");
        FontRenderer::new(data, false, false)
    }

    fn plate(text: &str) -> Nameplate {
        Nameplate {
            text: text.into(),
            rel: Vec3::new(0.0, 0.0, 5.0),
            height: 1.8,
            child: false,
            sneaking: false,
        }
    }

    fn req(plate: &Nameplate, distance_sq: f32) -> NameplateRequest<'_> {
        NameplateRequest {
            plate,
            view_yaw: 0.0,
            view_pitch: 0.0,
            distance_sq,
            score: None,
        }
    }

    fn scored<'a>(plate: &'a Nameplate, distance_sq: f32, score: &'a str) -> NameplateRequest<'a> {
        NameplateRequest {
            score: Some(score),
            ..req(plate, distance_sq)
        }
    }

    /// Glyph x-extent of the built geometry, in **label pixels**, recovered by undoing the
    /// billboard. At yaw/pitch 0 the label plane is the world XY plane and label `+x` is world
    /// `-x` (the `scale(-f1, …)` flip), so this is an exact inverse, not an approximation.
    fn label_x_range(g: &NameplateGeometry) -> (f32, f32) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for batch in &g.glyphs {
            for v in &batch.verts {
                let x = -v.pos[0] / NAMEPLATE_SCALE;
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
        (lo, hi)
    }

    // ------------------------------------------------------------------ centring

    #[test]
    fn the_label_is_centred_on_minus_half_the_string_width() {
        let mut f = font();
        let p = plate("AAAA"); // 4 glyphs x 5 px = 20 px wide in the synthetic font.
        assert_eq!(f.string_width(&p.text), 20);
        let g = build(&mut f, &req(&p, 25.0)).expect("in range");
        let (lo, hi) = label_x_range(&g);
        // drawString starts at -20/2 = -10 and each glyph quad spans `charWidth - 0.01 - 1`.
        assert!((lo - -10.0).abs() < 1e-4, "left edge {lo}");
        assert!((hi - (-10.0 + 15.0 + 3.99)).abs() < 1e-3, "right edge {hi}");
        // Ink is symmetric about the anchor to within the sub-pixel the quad's -0.01 costs.
        assert!((lo + hi).abs() < 1.5, "not centred: {lo}..{hi}");
    }

    #[test]
    fn an_odd_width_name_sits_a_pixel_left_of_true_centre() {
        // java: `-fontrenderer.getStringWidth(str) / 2` is *integer* division.
        let mut f = font();
        let p = plate("AAA"); // 15 px.
        assert_eq!(f.string_width(&p.text), 15);
        assert_eq!(p.layout(15).text_x, -7.0, "not -7.5");
        let g = build(&mut f, &req(&p, 25.0)).expect("in range");
        let (lo, _) = label_x_range(&g);
        assert!((lo - -7.0).abs() < 1e-4, "{lo}");
    }

    #[test]
    fn the_backdrop_wraps_the_measured_text_by_one_pixel_each_side() {
        let mut f = font();
        let p = plate("AAAA");
        let g = build(&mut f, &req(&p, 25.0)).expect("in range");
        assert_eq!(g.solid.len(), 1, "one backdrop, no bars");
        let backdrop = &g.solid[0];
        assert_eq!(backdrop.pass, NameplatePass::ThroughWalls);
        assert_eq!(backdrop.verts.len(), 6, "a two-triangle fan");
        for v in &backdrop.verts {
            assert_eq!(v.color, [0.0, 0.0, 0.0, 0.25], "java: color(0,0,0,0.25)");
        }
        let xs: Vec<f32> = backdrop
            .verts
            .iter()
            .map(|v| -v.pos[0] / NAMEPLATE_SCALE)
            .collect();
        let lo = xs.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!((lo - -11.0).abs() < 1e-3, "-(20/2) - 1: {lo}");
        assert!((hi - 11.0).abs() < 1e-3, "(20/2) + 1: {hi}");
    }

    // ------------------------------------------------------------------ scale

    #[test]
    fn one_label_pixel_is_one_vanilla_scale_of_world() {
        let mut f = font();
        let p = plate("A");
        let g = build(&mut f, &req(&p, 25.0)).expect("in range");
        let batch = &g.glyphs[0];
        // The strip is [TL, BL, TR, TR, BL, BR]; TL→BL is the glyph's full 7.99-pixel height.
        let top = Vec3::from(batch.verts[0].pos);
        let bottom = Vec3::from(batch.verts[1].pos);
        let height = (bottom - top).length();
        assert!(
            (height - 7.99 * NAMEPLATE_SCALE).abs() < 1e-6,
            "{height} blocks for 7.99 label pixels"
        );
        // java: renderLivingLabel's 0.016666668 * 1.6 and renderName's 0.02666667 agree.
        assert!((NAMEPLATE_SCALE - 0.026_666_67).abs() < 1e-7);
        // And it points *down* in the world, because label +y is world -y.
        assert!(bottom.y < top.y, "label +y must be world down");
    }

    // ------------------------------------------------------------------ sneaking

    #[test]
    fn a_sneaking_label_has_one_dim_depth_tested_pass_and_hangs_lower() {
        let mut f = font();
        let mut p = plate("A");
        p.sneaking = true;
        let g = build(&mut f, &req(&p, 25.0)).expect("in range");

        // java: the sneaking branch never calls disableDepth, and tints only with 553648127.
        assert!(
            g.glyphs.iter().all(|b| b.pass == NameplatePass::InFront),
            "a sneaking name must not show through walls"
        );
        assert_eq!(g.glyphs.len(), 1, "one pass, not two");
        let alpha = g.glyphs[0].verts[0].color[3];
        assert!(
            (alpha - 32.0 / 255.0).abs() < 1e-6,
            "dim, not opaque: {alpha}"
        );
        assert_eq!(g.solid[0].pass, NameplatePass::InFront, "backdrop too");

        // ... and a quarter of a block lower than the standing version.
        let mut standing = p.clone();
        standing.sneaking = false;
        let up = build(&mut f, &req(&standing, 25.0)).expect("in range");
        let drop = up.glyphs[0].verts[0].pos[1] - g.glyphs[0].verts[0].pos[1];
        assert!(
            (drop - NAMEPLATE_SNEAK_DROP * NAMEPLATE_SCALE).abs() < 1e-5,
            "{drop}"
        );
        assert!((drop - 0.25).abs() < 1e-4, "{drop} blocks");
    }

    #[test]
    fn a_standing_label_has_both_passes_and_only_the_second_is_opaque() {
        let mut f = font();
        let g = build(&mut f, &req(&plate("A"), 25.0)).expect("in range");
        assert_eq!(g.glyphs.len(), 2);
        assert_eq!(g.glyphs[0].pass, NameplatePass::ThroughWalls);
        assert_eq!(g.glyphs[1].pass, NameplatePass::InFront);
        assert!((g.glyphs[0].verts[0].color[3] - 32.0 / 255.0).abs() < 1e-6);
        assert_eq!(g.glyphs[1].verts[0].color[3], 1.0);
        // Both passes draw the *same* glyphs in the same place; only the tint differs.
        assert_eq!(g.glyphs[0].verts.len(), g.glyphs[1].verts.len());
        for (a, b) in g.glyphs[0].verts.iter().zip(&g.glyphs[1].verts) {
            assert_eq!(a.pos, b.pos);
            assert_eq!(a.uv, b.uv);
        }
    }

    // ------------------------------------------------------------------ § codes

    #[test]
    fn a_section_colour_code_recolours_the_glyphs_and_is_not_itself_drawn() {
        let mut f = font();
        // §c is "red" (0xFF5555); the code itself must not produce a quad.
        let plain = build(&mut f, &req(&plate("AA"), 25.0)).expect("in range");
        let g = build(&mut f, &req(&plate("§cAA"), 25.0)).expect("in range");
        assert_eq!(
            g.glyph_vertex_count(),
            plain.glyph_vertex_count(),
            "the two-character code adds no geometry"
        );
        let c = g.glyphs[1].verts[0].color; // the opaque InFront pass
        assert_eq!(c[3], 1.0);
        assert!(
            c[0] > 0.9 && c[1] < 0.5 && c[2] < 0.5,
            "expected red, got {c:?}"
        );
        // The ghost pass keeps the pass's alpha and takes only the code's RGB.
        let ghost = g.glyphs[0].verts[0].color;
        assert!((ghost[3] - 32.0 / 255.0).abs() < 1e-6, "{ghost:?}");
        assert_eq!([ghost[0], ghost[1], ghost[2]], [c[0], c[1], c[2]]);
    }

    #[test]
    fn a_section_code_does_not_widen_the_backdrop() {
        // java: getStringWidth skips the code, so `j` is the same either way.
        let mut f = font();
        assert_eq!(f.string_width("§cAA"), f.string_width("AA"));
        let plain = build(&mut f, &req(&plate("AA"), 25.0)).expect("in range");
        let coloured = build(&mut f, &req(&plate("§cAA"), 25.0)).expect("in range");
        assert_eq!(plain.solid[0].verts, coloured.solid[0].verts);
    }

    #[test]
    fn bold_widens_the_label_and_underline_emits_a_bar() {
        let mut f = font();
        assert_eq!(
            f.string_width("§lAA"),
            f.string_width("AA") + 2,
            "§l is +1/glyph"
        );
        let g = build(&mut f, &req(&plate("§nAA"), 25.0)).expect("in range");
        // One backdrop plus one bar batch per pass.
        assert_eq!(g.solid.len(), 3, "backdrop + two underline batches");
        assert_eq!(g.solid[1].pass, NameplatePass::ThroughWalls);
        assert_eq!(g.solid[2].pass, NameplatePass::InFront);
        assert!(!g.solid[1].verts.is_empty());
    }

    // ------------------------------------------------------------------ distance cull

    #[test]
    fn nothing_is_built_beyond_sixty_four_blocks() {
        let mut f = font();
        let p = plate("A");
        assert!(build(&mut f, &req(&p, 63.0 * 63.0)).is_some());
        assert!(
            build(&mut f, &req(&p, 64.0 * 64.0)).is_none(),
            "java: `d0 < f * f` with f = 64"
        );
    }

    #[test]
    fn the_cull_halves_while_sneaking() {
        let mut f = font();
        let mut p = plate("A");
        p.sneaking = true;
        assert!(build(&mut f, &req(&p, 31.0 * 31.0)).is_some());
        assert!(
            build(&mut f, &req(&p, 32.0 * 32.0)).is_none(),
            "java: f = 32 while sneaking"
        );
        // The same distance is fine standing up.
        p.sneaking = false;
        assert!(build(&mut f, &req(&p, 32.0 * 32.0)).is_some());
    }

    // ------------------------------------------------------------------ scoreboard line

    #[test]
    fn a_score_line_sits_below_the_name_and_only_within_ten_blocks() {
        let mut f = font();
        let p = plate("AA");
        // 99 < 100, so both lines exist: 2 backdrops and 2 passes x 2 labels.
        let near = build(&mut f, &scored(&p, 99.0, "7 Deaths")).expect("in range");
        assert_eq!(near.solid.len(), 2, "score backdrop + name backdrop");
        assert_eq!(near.glyphs.len(), 4, "two passes for each of two lines");
        // Emission order is vanilla's: the score first, then the name above it.
        let score_y = near.glyphs[0].verts[0].pos[1];
        let name_y = near.glyphs[2].verts[0].pos[1];
        assert!(name_y > score_y, "the name goes above the score");
        assert!((name_y - score_y - NAMEPLATE_SCORE_GAP).abs() < 1e-5);

        // java: `if (p_177069_10_ < 100.0D)` — at exactly 100 the score is gone.
        let far = build(&mut f, &scored(&p, 100.0, "7 Deaths")).expect("in range");
        assert_eq!(far.solid.len(), 1);
        assert_eq!(far.glyphs.len(), 2);
    }

    #[test]
    fn a_sneaking_player_never_gets_a_score_line() {
        // java: the sneaking branch of renderName never calls renderOffsetLivingLabel, which is
        // the only place RenderPlayer's score override lives.
        let mut f = font();
        let mut p = plate("AA");
        p.sneaking = true;
        let g = build(&mut f, &scored(&p, 9.0, "7 Deaths")).expect("in range");
        assert_eq!(g.solid.len(), 1);
        assert_eq!(g.glyphs.len(), 1);
    }

    // ------------------------------------------------------------------ seam plumbing

    #[test]
    fn texture_keys_are_stable_and_distinct() {
        assert_eq!(
            font_texture_key(FontTexture::Ascii),
            "mc-ui/nameplate/ascii"
        );
        assert_eq!(
            font_texture_key(FontTexture::UnicodePage(0x4e)),
            "mc-ui/nameplate/unicode_page_4e"
        );
        assert_ne!(
            font_texture_key(FontTexture::UnicodePage(0)),
            font_texture_key(FontTexture::UnicodePage(1))
        );
    }

    #[test]
    fn geometry_reports_the_sheets_it_needs() {
        let mut f = font();
        let g = build(&mut f, &req(&plate("AB"), 25.0)).expect("in range");
        assert_eq!(g.textures(), vec![FontTexture::Ascii]);
        assert!(!g.is_empty());
        assert_eq!(
            NameplateGeometry::default().textures(),
            Vec::<FontTexture>::new()
        );
        assert!(NameplateGeometry::default().is_empty());
    }

    #[test]
    fn an_empty_name_still_draws_its_backdrop_and_no_glyphs() {
        // java: nothing in renderLivingLabel guards against an empty string; `j` is 0 and the
        // quad degenerates to two pixels wide.
        let mut f = font();
        let g = build(&mut f, &req(&plate(""), 25.0)).expect("in range");
        assert_eq!(g.glyph_vertex_count(), 0);
        assert_eq!(g.solid.len(), 1);
    }

    fn reference_push_label(
        out: &mut NameplateGeometry,
        font: &mut FontRenderer,
        plate: &Nameplate,
        view_yaw: f32,
        view_pitch: f32,
    ) {
        // java: `int j = fontrenderer.getStringWidth(str) / 2;` and the drawString x below it both
        // derive from this one measurement, so it is taken once.
        let width = font.string_width(&plate.text);
        let layout = plate.layout(width);

        out.solid.push(SolidBatch {
            pass: plate.backdrop_pass(),
            verts: plate
                .backdrop_vertices(view_yaw, view_pitch, width)
                .to_vec(),
        });

        for &pass in plate.passes() {
            let tint = plate.argb_for(pass);
            let mut sink = ReferenceBillboardSink {
                plate,
                view_yaw,
                view_pitch,
                pass,
                glyphs: Vec::new(),
                bars: Vec::new(),
            };
            // java: `fontrenderer.drawString(str, -fontrenderer.getStringWidth(str) / 2, i, colour)`
            // — the three-int overload, i.e. no drop shadow.
            font.draw_string(
                &mut sink,
                &plate.text,
                layout.text_x,
                layout.text_y,
                tint as i32,
                false,
            );
            if !sink.bars.is_empty() {
                out.solid.push(SolidBatch {
                    pass,
                    verts: sink.bars,
                });
            }
            out.glyphs.append(&mut sink.glyphs);
        }
    }

    struct ReferenceBillboardSink<'a> {
        plate: &'a Nameplate,
        view_yaw: f32,
        view_pitch: f32,
        pass: NameplatePass,
        glyphs: Vec<GlyphBatch>,
        bars: Vec<NameplateVertex>,
    }

    impl TextSink for ReferenceBillboardSink<'_> {
        fn glyph(&mut self, quad: GlyphQuad) {
            let verts = self.plate.glyph_vertices(
                self.view_yaw,
                self.view_pitch,
                quad.pos,
                quad.uv,
                argb_to_rgba(quad.color),
            );
            match self.glyphs.last_mut() {
                Some(b) if b.texture == quad.texture => b.verts.extend_from_slice(&verts),
                _ => self.glyphs.push(GlyphBatch {
                    shadow: false,
                    pass: self.pass,
                    texture: quad.texture,
                    verts: verts.to_vec(),
                }),
            }
        }

        fn bar(&mut self, rect: SolidRect) {
            let rgba = argb_to_rgba(rect.color);
            let corners = [
                (rect.x0, rect.y0),
                (rect.x1, rect.y0),
                (rect.x1, rect.y1),
                (rect.x0, rect.y1),
            ];
            for i in [0usize, 1, 2, 0, 2, 3] {
                let (x, y) = corners[i];
                self.bars.push(NameplateVertex {
                    pos: self
                        .plate
                        .local_to_world(self.view_yaw, self.view_pitch, x, y),
                    color: rgba,
                });
            }
        }
    }

    #[test]
    fn cached_nameplate_transform_preserves_legacy_geometry_across_views() {
        let mut actual_font = font();
        let mut reference_font = font();
        for (yaw, pitch) in [(0.0, 0.0), (37.0, -11.0), (-153.25, 89.0), (270.0, -65.5)] {
            for (child, sneaking) in [(false, false), (true, false), (false, true), (true, true)] {
                for text in ["BenchName", "§lBold §oItalic", "§nLine §mStrike §rReset"] {
                    let mut label = plate(text);
                    label.rel = Vec3::new(13.25, -4.5, 39.75);
                    label.child = child;
                    label.sneaking = sneaking;
                    let mut actual = NameplateGeometry::default();
                    let mut expected = NameplateGeometry::default();
                    push_label(
                        &mut actual,
                        &mut actual_font,
                        &label,
                        yaw,
                        pitch,
                        NameplateStyle::VANILLA,
                    );
                    reference_push_label(&mut expected, &mut reference_font, &label, yaw, pitch);
                    assert_eq!(
                        actual, expected,
                        "{text}, yaw={yaw}, pitch={pitch}, child={child}, sneaking={sneaking}"
                    );
                    for (a, b) in actual
                        .glyphs
                        .iter()
                        .flat_map(|batch| &batch.verts)
                        .zip(expected.glyphs.iter().flat_map(|batch| &batch.verts))
                    {
                        assert_eq!(a.pos.map(f32::to_bits), b.pos.map(f32::to_bits));
                    }
                    for (a, b) in actual
                        .solid
                        .iter()
                        .flat_map(|batch| &batch.verts)
                        .zip(expected.solid.iter().flat_map(|batch| &batch.verts))
                    {
                        assert_eq!(a.pos.map(f32::to_bits), b.pos.map(f32::to_bits));
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------ layout cache (#18)

    /// A [`NameplateTarget`] with `NameplatePhase`'s exact stream semantics: one vertex stream
    /// per pass and kind, and glyph batches merged when consecutive pushes share a sheet.
    #[derive(Debug, Default)]
    struct PhaseRecorder {
        solid: [Vec<NameplateVertex>; 3],
        glyphs: [Vec<NameplateGlyphVertex>; 3],
        batches: [Vec<(FontTexture, u32, u32)>; 3],
    }

    fn pass_index(pass: NameplatePass) -> usize {
        match pass {
            NameplatePass::ThroughWalls => 0,
            NameplatePass::InFront => 1,
            NameplatePass::SignText => 2,
        }
    }

    impl PhaseRecorder {
        /// `NameplatePhase::begin`: drop the frame's geometry, keep the capacity.
        fn clear(&mut self) {
            for i in 0..3 {
                self.solid[i].clear();
                self.glyphs[i].clear();
                self.batches[i].clear();
            }
        }
    }

    impl NameplateTarget for PhaseRecorder {
        fn push_solid(&mut self, pass: NameplatePass, verts: &[NameplateVertex]) {
            self.solid[pass_index(pass)].extend_from_slice(verts);
        }

        fn push_glyph_run(
            &mut self,
            pass: NameplatePass,
            texture: FontTexture,
            verts: &[NameplateGlyphVertex],
        ) {
            // `NameplatePhase::push_glyphs` ignores an empty push.
            if verts.is_empty() {
                return;
            }
            let i = pass_index(pass);
            let start = self.glyphs[i].len() as u32;
            self.glyphs[i].extend_from_slice(verts);
            let end = self.glyphs[i].len() as u32;
            match self.batches[i].last_mut() {
                Some(b) if b.0 == texture && b.2 == start => b.2 = end,
                _ => self.batches[i].push((texture, start, end)),
            }
        }
    }

    /// Bit-level equality: `f32 ==` would call `-0.0` and `0.0` equal.
    fn assert_same_bits(actual: &PhaseRecorder, expected: &PhaseRecorder, what: &str) {
        for i in 0..3 {
            assert_eq!(
                actual.batches[i], expected.batches[i],
                "{what}: batches[{i}]"
            );
            assert_eq!(
                actual.solid[i].len(),
                expected.solid[i].len(),
                "{what}: solid[{i}]"
            );
            for (a, b) in actual.solid[i].iter().zip(&expected.solid[i]) {
                assert_eq!(a.pos.map(f32::to_bits), b.pos.map(f32::to_bits), "{what}");
                assert_eq!(
                    a.color.map(f32::to_bits),
                    b.color.map(f32::to_bits),
                    "{what}"
                );
            }
            assert_eq!(
                actual.glyphs[i].len(),
                expected.glyphs[i].len(),
                "{what}: glyphs[{i}]"
            );
            for (a, b) in actual.glyphs[i].iter().zip(&expected.glyphs[i]) {
                assert_eq!(a.pos.map(f32::to_bits), b.pos.map(f32::to_bits), "{what}");
                assert_eq!(a.uv.map(f32::to_bits), b.uv.map(f32::to_bits), "{what}");
                assert_eq!(
                    a.color.map(f32::to_bits),
                    b.color.map(f32::to_bits),
                    "{what}"
                );
            }
        }
    }

    /// The synthetic font plus two unicode-page glyphs, so mixed-sheet runs are exercised.
    fn font_with_unicode() -> FontRenderer {
        let mut sizes = [0u8; 0x1_0000];
        sizes[0x2764] = 0x0E; // ❤, page 0x27
        sizes[0x2694] = 0x1C; // ⚔, page 0x26
        let data = FontData::from_parts(&synthetic_ascii_png(), &sizes).expect("synthetic font");
        FontRenderer::new(data, false, false)
    }

    const CACHE_TEXTS: &[&str] = &[
        "BenchName",
        "deadmau5",
        "",
        "§c[R] §fName",
        "§lBold §oItalic",
        "§nLine §mStrike §rReset",
        "§eSpawns in §c23 §eseconds",
        "20 ❤",
        "A❤B⚔C",
        "§4❤ §lx§r ⚔",
    ];

    #[test]
    fn cached_labels_match_the_uncached_path_bit_for_bit() {
        let mut legacy_font = font_with_unicode();
        let mut cached_font = font_with_unicode();
        let mut cache = NameplateCache::new();
        // Two frames per view: the first builds every layout, the second is all hits.
        for (yaw, pitch) in [(0.0, 0.0), (37.0, -11.0), (-153.25, 89.0), (270.0, -65.5)] {
            for frame in 0..2 {
                cache.begin_frame();
                let mut expected = PhaseRecorder::default();
                let mut actual = PhaseRecorder::default();
                for (n, text) in CACHE_TEXTS.iter().enumerate() {
                    for (child, sneaking) in
                        [(false, false), (true, false), (false, true), (true, true)]
                    {
                        // Within 10 blocks with a score line, beyond it, and out of range.
                        for (distance_sq, score) in [
                            (25.0, Some("7 §cDeaths")),
                            (99.0, Some("§l12 ❤")),
                            (400.0, Some("3 Kills")),
                            (36.0, None),
                            (64.0 * 64.0, None),
                        ] {
                            let mut label = plate(text);
                            label.rel = Vec3::new(13.25 - n as f32, -4.5, 39.75 + n as f32);
                            label.child = child;
                            label.sneaking = sneaking;
                            let request = NameplateRequest {
                                plate: &label,
                                view_yaw: yaw,
                                view_pitch: pitch,
                                distance_sq,
                                score,
                            };
                            let built = build(&mut legacy_font, &request);
                            if let Some(geometry) = &built {
                                geometry.submit_to(&mut expected);
                            }
                            let drawn =
                                submit_cached(&mut cached_font, &mut cache, &request, &mut actual);
                            assert_eq!(drawn, built.is_some(), "{text} range gate");
                        }
                    }
                }
                assert_same_bits(
                    &actual,
                    &expected,
                    &format!("yaw={yaw} pitch={pitch} frame={frame}"),
                );
            }
        }
        let stats = cache.stats();
        assert!(stats.hits > stats.misses, "{stats:?}");
        assert_eq!(stats.volatile, 0, "no §k in these texts");
    }

    #[test]
    fn obfuscated_labels_bypass_the_cache_and_keep_the_random_stream_in_step() {
        // Both renderers start from the same seed. Non-§k labels never touch the stream, so
        // if the cache skipped or added a §k draw the two would desynchronise and the
        // obfuscated glyphs (and everything after them) would differ.
        let mut legacy_font = font_with_unicode();
        let mut cached_font = font_with_unicode();
        let mut cache = NameplateCache::new();
        let texts = [
            "§c[R] §fName",
            "§kSecret",
            "Plain",
            "§aA §KB",
            "§eSpawns in §c23",
        ];
        for frame in 0..4 {
            cache.begin_frame();
            let mut expected = PhaseRecorder::default();
            let mut actual = PhaseRecorder::default();
            for text in texts {
                let label = plate(text);
                let request = NameplateRequest {
                    score: Some("§k9 hearts"),
                    ..req(&label, 25.0)
                };
                if let Some(geometry) = build(&mut legacy_font, &request) {
                    geometry.submit_to(&mut expected);
                }
                assert!(submit_cached(
                    &mut cached_font,
                    &mut cache,
                    &request,
                    &mut actual
                ));
            }
            assert_same_bits(&actual, &expected, &format!("frame {frame}"));
        }
        let stats = cache.stats();
        // "§kSecret", "§aA §KB" and the score line on all five plates, every frame.
        assert_eq!(stats.volatile, 4 * 7, "{stats:?}");
        assert!(stats.hits > 0);
    }

    #[test]
    fn may_obfuscate_is_conservative_about_k() {
        let f = font();
        assert!(
            !f.may_obfuscate("Plain k text"),
            "no section sign, no styles"
        );
        assert!(!f.may_obfuscate("§c[R] §fkat"), "a k that is not a code");
        assert!(f.may_obfuscate("§kx"));
        assert!(
            f.may_obfuscate("§Kx"),
            "codes are read from the lowercased string"
        );
        assert!(f.may_obfuscate("§\u{212A}x"), "KELVIN SIGN lowercases to k");
        // `§§k` never obfuscates, but answering true there only costs a cache miss.
        assert!(f.may_obfuscate("§§k"));
    }

    #[test]
    fn a_font_change_drops_cached_layouts() {
        let mut legacy_font = font_with_unicode();
        let mut cached_font = font_with_unicode();
        let mut cache = NameplateCache::new();
        let label = plate("§cAB❤");
        for unicode in [false, true, false] {
            legacy_font.set_unicode_flag(unicode);
            cached_font.set_unicode_flag(unicode);
            cache.begin_frame();
            let mut expected = PhaseRecorder::default();
            let mut actual = PhaseRecorder::default();
            build(&mut legacy_font, &req(&label, 25.0))
                .expect("in range")
                .submit_to(&mut expected);
            assert!(submit_cached(
                &mut cached_font,
                &mut cache,
                &req(&label, 25.0),
                &mut actual
            ));
            assert_same_bits(&actual, &expected, &format!("unicode={unicode}"));
        }
        // Setting a flag to the value it already has is not a change.
        let generation = cached_font.layout_generation();
        cached_font.set_unicode_flag(false);
        cached_font.set_bidi_flag(false);
        cached_font.set_anaglyph(false);
        assert_eq!(cached_font.layout_generation(), generation);
        cached_font.set_anaglyph(true);
        assert_ne!(cached_font.layout_generation(), generation);
    }

    #[test]
    fn unused_layouts_are_evicted() {
        let mut f = font();
        let mut cache = NameplateCache::new();
        let mut sink = PhaseRecorder::default();
        cache.begin_frame();
        assert!(submit_cached(
            &mut f,
            &mut cache,
            &req(&plate("Gone"), 25.0),
            &mut sink
        ));
        assert_eq!(cache.len(), 1);
        for _ in 0..NameplateCache::EVICT_AFTER_FRAMES {
            cache.begin_frame();
            assert!(submit_cached(
                &mut f,
                &mut cache,
                &req(&plate("Kept"), 25.0),
                &mut sink
            ));
        }
        for _ in 0..NameplateCache::EVICT_AFTER_FRAMES {
            cache.begin_frame();
            assert!(submit_cached(
                &mut f,
                &mut cache,
                &req(&plate("Kept"), 25.0),
                &mut sink
            ));
        }
        assert_eq!(cache.len(), 1, "only the label still being drawn survives");
    }

    /// Phase 4b review S1: label text is server-chosen and can change every tick, so the cache
    /// is bounded by size as well as by frame age.
    #[test]
    fn the_layout_cache_is_bounded_by_size() {
        let mut f = font();
        let mut cache = NameplateCache::new();
        let mut sink = PhaseRecorder::default();
        cache.begin_frame();

        let long = "x".repeat(NameplateCache::MAX_TEXT_BYTES + 1);
        for _ in 0..2 {
            assert!(submit_cached(
                &mut f,
                &mut cache,
                &req(&plate(&long), 25.0),
                &mut sink
            ));
        }
        assert!(cache.is_empty(), "over-long text is drawn uncached");
        assert!(cache.volatile_texts.is_empty(), "and not remembered");

        // Distinct labels inside one eviction window: the budget holds, and the count of cached
        // primitives stays exact through every start-over.
        let label = |n: usize| format!("{n:0>32}");
        assert!(submit_cached(
            &mut f,
            &mut cache,
            &req(&plate(&label(0)), 25.0),
            &mut sink
        ));
        let per_label = cache.cached_primitives;
        assert!(per_label >= 32, "at least one primitive per glyph");
        let fits = NameplateCache::MAX_CACHED_PRIMITIVES / per_label;
        for n in 1..fits * 3 {
            assert!(submit_cached(
                &mut f,
                &mut cache,
                &req(&plate(&label(n)), 25.0),
                &mut sink
            ));
            assert!(cache.len() <= fits);
            assert_eq!(cache.cached_primitives, cache.len() * per_label);
        }

        // Age eviction gives the budget back.
        for _ in 0..NameplateCache::EVICT_AFTER_FRAMES * 2 {
            cache.begin_frame();
        }
        assert!(cache.is_empty());
        assert_eq!(cache.cached_primitives, 0);

        for n in 0..=NameplateCache::MAX_VOLATILE_TEXTS {
            let text = format!("§k{n}");
            assert!(submit_cached(
                &mut f,
                &mut cache,
                &req(&plate(&text), 25.0),
                &mut sink
            ));
            assert!(cache.volatile_texts.len() <= NameplateCache::MAX_VOLATILE_TEXTS);
        }
    }

    /// Measurement harness, not a correctness test: per-frame CPU cost of a BedWars-mid label
    /// set (20 players, 6 of them close enough for a `20 ❤` score line, plus 5 four-line
    /// generator holograms and 2 three-line NPC holograms) through the uncached path and through
    /// [`NameplateCache`], with the camera turning every frame. Run with
    /// `cargo test -p mc-ui --release --lib -- --ignored label_cache_cost --nocapture`.
    #[test]
    #[ignore = "measurement harness; run explicitly"]
    fn label_cache_cost() {
        let mut labels: Vec<(Nameplate, Option<&str>)> = Vec::new();
        for i in 0..20 {
            let mut p = plate(&format!("§c[R] §fPlayer{i:02}"));
            p.rel = Vec3::new(i as f32 - 10.0, 0.0, 6.0 + i as f32);
            labels.push((p, (i < 6).then_some("20 ❤")));
        }
        let generator = [
            "§b§lDIAMOND",
            "§eTier §cII",
            "§eSpawns in §c23 §eseconds",
            "§7[§a||||§7]",
        ];
        let npc = ["§a§lITEM SHOP", "§eRIGHT CLICK", "§7Solo"];
        for g in 0..5 {
            for (n, line) in generator.iter().enumerate() {
                let mut p = plate(line);
                p.rel = Vec3::new(g as f32 * 7.0 - 14.0, n as f32 * 0.25, 20.0);
                labels.push((p, None));
            }
        }
        for g in 0..2 {
            for (n, line) in npc.iter().enumerate() {
                let mut p = plate(line);
                p.rel = Vec3::new(g as f32 * 4.0, n as f32 * 0.25, 12.0);
                labels.push((p, None));
            }
        }
        const FRAMES: u32 = 2_000;
        let run = |cached: bool| -> (std::time::Duration, usize) {
            let mut font = font_with_unicode();
            let mut cache = NameplateCache::new();
            let mut sink = PhaseRecorder::default();
            let start = std::time::Instant::now();
            for frame in 0..FRAMES {
                sink.clear();
                cache.begin_frame();
                for (label, score) in &labels {
                    let request = NameplateRequest {
                        plate: label,
                        view_yaw: frame as f32 * 0.37,
                        view_pitch: 12.5,
                        distance_sq: if score.is_some() { 25.0 } else { 400.0 },
                        score: *score,
                    };
                    if cached {
                        submit_cached(&mut font, &mut cache, &request, &mut sink);
                    } else if let Some(geometry) = build(&mut font, &request) {
                        geometry.submit_to(&mut sink);
                    }
                }
            }
            (
                start.elapsed() / FRAMES,
                sink.glyphs[0].len() + sink.glyphs[1].len(),
            )
        };
        let (legacy, legacy_verts) = run(false);
        let (cached, cached_verts) = run(true);
        assert_eq!(legacy_verts, cached_verts);
        eprintln!(
            "{} labels, {} glyph vertices/frame: uncached {legacy:?}/frame, cached {cached:?}/frame",
            labels.len(),
            legacy_verts
        );
    }

    #[test]
    fn static_texture_keys_match_the_allocating_ones() {
        for texture in [
            FontTexture::Ascii,
            FontTexture::StandardGalactic,
            FontTexture::UnicodePage(0),
            FontTexture::UnicodePage(0x27),
            FontTexture::UnicodePage(0xff),
        ] {
            assert_eq!(font_texture_key_str(texture), font_texture_key(texture));
        }
    }
}
