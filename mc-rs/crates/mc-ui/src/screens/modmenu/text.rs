//! The menu's text: Segoe UI from `C:\Windows\Fonts`, rasterised with `ab_glyph` into a glyph
//! atlas at the exact pixel size each style needs, or the Minecraft font when the TTFs are
//! missing.
//!
//! Glyphs are rasterised on first use and cached by `(weight, size, glyph)`; the atlas is
//! re-uploaded only when it gained glyphs ([`MenuText::take_upload`]). The font files are read
//! once per process ([`prewarm`] starts that on a background thread at client start-up, so the
//! first menu open does not pay for it).

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use ab_glyph::{Font, FontArc, GlyphId, PxScale, ScaleFont};

use super::paint::{Painter, Rect, Rgba};
use crate::font::{FontRenderer, GlyphQuad, SolidRect, TextSink};
use crate::widget::UiTexture;

/// Atlas side length (RGBA8).
pub const ATLAS_SIZE: u32 = 1024;

/// The UI texture key the atlas is uploaded under.
#[must_use]
pub fn atlas_key() -> UiTexture {
    UiTexture::Resource(mc_assets::ResourceLocation::new(
        "mcrs",
        "modmenu/segoe_atlas",
    ))
}

/// Font weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Weight {
    /// `segoeui.ttf` (400).
    Regular,
    /// `seguisb.ttf` (600).
    Semibold,
    /// `segoeuib.ttf` (700).
    Bold,
}

impl Weight {
    fn index(self) -> usize {
        match self {
            Weight::Regular => 0,
            Weight::Semibold => 1,
            Weight::Bold => 2,
        }
    }
}

/// How one run of text looks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextStyle {
    /// CSS `font-size` in physical pixels (the em size).
    pub size: f32,
    /// Weight.
    pub weight: Weight,
    /// CSS `letter-spacing` in em.
    pub tracking: f32,
    /// CSS `text-transform: uppercase` (ASCII).
    pub upper: bool,
}

impl TextStyle {
    /// Regular text at `size` px.
    #[must_use]
    pub const fn new(size: f32) -> TextStyle {
        TextStyle {
            size,
            weight: Weight::Regular,
            tracking: 0.0,
            upper: false,
        }
    }

    /// The same style at another weight.
    #[must_use]
    pub const fn weight(self, weight: Weight) -> TextStyle {
        TextStyle { weight, ..self }
    }

    /// Small caps label style: uppercase with letter spacing.
    #[must_use]
    pub const fn caps(self, tracking: f32) -> TextStyle {
        TextStyle {
            tracking,
            upper: true,
            ..self
        }
    }
}

type Faces = Option<Arc<[FontArc; 3]>>;

static FACES: OnceLock<Faces> = OnceLock::new();

fn fonts_dir() -> std::path::PathBuf {
    let windir = std::env::var_os("WINDIR").unwrap_or_else(|| "C:\\Windows".into());
    std::path::PathBuf::from(windir).join("Fonts")
}

fn load_faces() -> Faces {
    let dir = fonts_dir();
    let read = |name: &str| {
        let path = dir.join(name);
        // Mapped from the OS's file cache (shared, no private copy) when
        // possible; read into memory as before otherwise. The same bytes either way.
        if let Some(font) = mc_render::w21_mb::map_file_for_process(&path)
            .and_then(|bytes| FontArc::try_from_slice(bytes).ok())
        {
            return Some(font);
        }
        std::fs::read(path)
            .ok()
            .and_then(|bytes| FontArc::try_from_vec(bytes).ok())
    };
    let regular = read("segoeui.ttf")?;
    let semibold = read("seguisb.ttf").unwrap_or_else(|| regular.clone());
    let bold = read("segoeuib.ttf").unwrap_or_else(|| semibold.clone());
    Some(Arc::new([regular, semibold, bold]))
}

fn faces() -> Faces {
    FACES.get_or_init(load_faces).clone()
}

/// Read the Segoe UI files on a background thread so the first menu open is instant.
pub fn prewarm() {
    if FACES.get().is_none() {
        let _ = std::thread::Builder::new()
            .name("mc-rs modmenu fonts".into())
            .spawn(|| {
                let _ = faces();
            });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GlyphKey {
    weight: u8,
    /// Size in quarter pixels.
    size: u16,
    id: u16,
}

#[derive(Debug, Clone, Copy)]
struct GlyphSlot {
    /// UV rectangle (u0, v0, u1, v1).
    uv: [f32; 4],
    /// Offset of the bitmap's top-left from (pen x, baseline).
    offset: [f32; 2],
    /// Bitmap size in pixels (0 for blank glyphs such as space).
    size: [f32; 2],
}

/// Segoe UI or Minecraft-font text for the menu.
pub struct MenuText {
    faces: Faces,
    pixels: Vec<u8>,
    glyphs: HashMap<GlyphKey, GlyphSlot>,
    shelf_x: u32,
    shelf_y: u32,
    shelf_h: u32,
    dirty: bool,
}

impl std::fmt::Debug for MenuText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MenuText")
            .field("segoe", &self.faces.is_some())
            .field("glyphs", &self.glyphs.len())
            .finish_non_exhaustive()
    }
}

impl Default for MenuText {
    fn default() -> Self {
        Self::load()
    }
}

impl MenuText {
    /// Segoe UI if the system has it, else the Minecraft font.
    #[must_use]
    pub fn load() -> MenuText {
        MenuText::with_faces(faces())
    }

    /// Always the Minecraft font (tests, and machines without Segoe UI).
    #[must_use]
    pub fn minecraft_only() -> MenuText {
        MenuText::with_faces(None)
    }

    fn with_faces(faces: Faces) -> MenuText {
        MenuText {
            faces,
            pixels: Vec::new(),
            glyphs: HashMap::new(),
            shelf_x: 0,
            shelf_y: 0,
            shelf_h: 0,
            dirty: false,
        }
    }

    /// Whether Segoe UI is in use.
    #[must_use]
    pub fn is_segoe(&self) -> bool {
        self.faces.is_some()
    }

    /// Whether this renderer has a glyph for every character of `text`: always with the
    /// Minecraft font; with Segoe UI, unless the text needs a script it lacks (CJK, for
    /// example), where the caller can fall back to the Minecraft font's unicode pages.
    /// (W9: song titles.)
    #[must_use]
    pub fn covers(&self, text: &str) -> bool {
        let Some(faces) = &self.faces else {
            return true;
        };
        let face = &faces[Weight::Regular.index()];
        text.chars()
            .all(|c| c.is_whitespace() || c.is_control() || face.glyph_id(c).0 != 0)
    }

    /// Make the next [`MenuText::take_upload`] hand out the whole atlas again (the renderer
    /// that held it was rebuilt). Nothing happens before the first glyph exists.
    pub fn mark_dirty(&mut self) {
        if !self.pixels.is_empty() {
            self.dirty = true;
        }
    }

    /// The atlas pixels if glyphs were added since the last call (upload them).
    pub fn take_upload(&mut self) -> Option<&[u8]> {
        if std::mem::take(&mut self.dirty) {
            Some(&self.pixels)
        } else {
            None
        }
    }

    fn scale_for(font: &FontArc, size: f32) -> PxScale {
        // ab_glyph's PxScale is the ascent-to-descent height; CSS sizes are the em.
        let em = font.units_per_em().unwrap_or(1000.0);
        PxScale::from(size * font.height_unscaled() / em)
    }

    /// The ascent above the baseline and the full line height, in pixels.
    #[must_use]
    pub fn metrics(&self, style: TextStyle) -> (f32, f32) {
        match &self.faces {
            Some(faces) => {
                let font = &faces[style.weight.index()];
                let scaled = font.as_scaled(Self::scale_for(font, style.size));
                (scaled.ascent(), scaled.ascent() - scaled.descent())
            }
            None => {
                let k = style.size / 8.0;
                (7.0 * k, 9.0 * k)
            }
        }
    }

    /// The advance width of `text` in pixels.
    #[must_use]
    pub fn measure(&self, text: &str, style: TextStyle, mc: &FontRenderer) -> f32 {
        let Some(faces) = &self.faces else {
            let k = style.size / 8.0;
            let tracking = style.tracking * style.size * text.chars().count() as f32;
            return if style.upper {
                let width: i32 = text
                    .chars()
                    .map(|c| mc.char_width(c.to_ascii_uppercase()))
                    .sum();
                width as f32 * k + tracking
            } else {
                mc.string_width(text) as f32 * k + tracking
            };
        };
        let font = &faces[style.weight.index()];
        let scaled = font.as_scaled(Self::scale_for(font, style.size));
        let mut width = 0.0;
        let mut previous: Option<GlyphId> = None;
        for c in text.chars() {
            let c = if style.upper {
                c.to_ascii_uppercase()
            } else {
                c
            };
            let id = scaled.glyph_id(c);
            if let Some(p) = previous {
                width += scaled.kern(p, id);
            }
            width += scaled.h_advance(id) + style.tracking * style.size;
            previous = Some(id);
        }
        width
    }

    /// The longest prefix of `text` (in bytes) that fits in `max_width`, cut at a word boundary
    /// when possible. Used for wrapping.
    #[must_use]
    pub fn fit_words(
        &self,
        text: &str,
        max_width: f32,
        style: TextStyle,
        mc: &FontRenderer,
    ) -> usize {
        if self.measure(text, style, mc) <= max_width {
            return text.len();
        }
        let mut best = 0;
        for (index, c) in text.char_indices() {
            if c == ' ' {
                if self.measure(&text[..index], style, mc) <= max_width {
                    best = index;
                } else {
                    break;
                }
            }
        }
        if best == 0 {
            // One long word: cut it by characters.
            let mut end = 0;
            for (index, c) in text.char_indices() {
                let next = index + c.len_utf8();
                if self.measure(&text[..next], style, mc) > max_width {
                    break;
                }
                end = next;
            }
            return end.max(text.chars().next().map_or(0, char::len_utf8));
        }
        best
    }

    fn glyph(
        &mut self,
        font: &FontArc,
        weight: Weight,
        size: f32,
        id: GlyphId,
    ) -> Option<GlyphSlot> {
        let key = GlyphKey {
            weight: weight.index() as u8,
            size: (size * 4.0).round().clamp(1.0, 65_535.0) as u16,
            id: id.0,
        };
        if let Some(slot) = self.glyphs.get(&key) {
            return Some(*slot);
        }
        let scale = Self::scale_for(font, size);
        let glyph = id.with_scale_and_position(scale, ab_glyph::point(0.0, 0.0));
        let Some(outlined) = font.outline_glyph(glyph) else {
            let slot = GlyphSlot {
                uv: [0.0; 4],
                offset: [0.0; 2],
                size: [0.0; 2],
            };
            self.glyphs.insert(key, slot);
            return Some(slot);
        };
        let bounds = outlined.px_bounds();
        let (w, h) = (
            (bounds.max.x - bounds.min.x).ceil().max(0.0) as u32,
            (bounds.max.y - bounds.min.y).ceil().max(0.0) as u32,
        );
        if w == 0 || h == 0 || w + 2 > ATLAS_SIZE || h + 2 > ATLAS_SIZE {
            return None;
        }
        if self.pixels.is_empty() {
            self.pixels = vec![0; (ATLAS_SIZE * ATLAS_SIZE * 4) as usize];
        }
        if self.shelf_x + w + 1 > ATLAS_SIZE {
            self.shelf_x = 0;
            self.shelf_y += self.shelf_h + 1;
            self.shelf_h = 0;
        }
        if self.shelf_y + h + 1 > ATLAS_SIZE {
            // Full: start over. Rare (it needs many sizes); glyphs re-rasterise on demand.
            self.glyphs.clear();
            self.pixels.fill(0);
            self.shelf_x = 0;
            self.shelf_y = 0;
            self.shelf_h = 0;
        }
        let (ox, oy) = (self.shelf_x + 1, self.shelf_y + 1);
        let pixels = &mut self.pixels;
        outlined.draw(|x, y, coverage| {
            let (px, py) = (ox + x, oy + y);
            if px < ATLAS_SIZE && py < ATLAS_SIZE {
                let i = ((py * ATLAS_SIZE + px) * 4) as usize;
                // A slight contrast lift: thin strokes stay legible over busy glass.
                let a = coverage.clamp(0.0, 1.0).powf(0.85);
                pixels[i] = 255;
                pixels[i + 1] = 255;
                pixels[i + 2] = 255;
                pixels[i + 3] = (a * 255.0).round() as u8;
            }
        });
        self.shelf_x += w + 1;
        self.shelf_h = self.shelf_h.max(h + 1);
        self.dirty = true;
        let atlas = ATLAS_SIZE as f32;
        let slot = GlyphSlot {
            uv: [
                ox as f32 / atlas,
                oy as f32 / atlas,
                (ox + w) as f32 / atlas,
                (oy + h) as f32 / atlas,
            ],
            offset: [bounds.min.x, bounds.min.y],
            size: [w as f32, h as f32],
        };
        self.glyphs.insert(key, slot);
        Some(slot)
    }

    /// Draw `text` with its line box's top-left at `(x, top)`. Returns the advance width.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        painter: &mut Painter<'_>,
        mc: &mut FontRenderer,
        text: &str,
        x: f32,
        top: f32,
        style: TextStyle,
        color: Rgba,
    ) -> f32 {
        let Some(faces) = self.faces.clone() else {
            return draw_minecraft(painter, mc, text, x, top, style, color);
        };
        let font = &faces[style.weight.index()];
        let scaled = font.as_scaled(Self::scale_for(font, style.size));
        let baseline = (top + scaled.ascent()).round();
        let texture = atlas_key();
        let mut pen = x;
        let mut previous: Option<GlyphId> = None;
        for c in text.chars() {
            let c = if style.upper {
                c.to_ascii_uppercase()
            } else {
                c
            };
            let id = scaled.glyph_id(c);
            if let Some(p) = previous {
                pen += scaled.kern(p, id);
            }
            if let Some(slot) = self.glyph(font, style.weight, style.size, id) {
                if slot.size[0] > 0.0 {
                    let gx = pen.round() + slot.offset[0];
                    let gy = baseline + slot.offset[1];
                    painter.textured(
                        &texture,
                        Rect::new(gx, gy, slot.size[0], slot.size[1]),
                        slot.uv,
                        color,
                    );
                }
            }
            pen += scaled.h_advance(id) + style.tracking * style.size;
            previous = Some(id);
        }
        pen - x
    }
}

/// Minecraft-font fallback: the vanilla glyphs scaled so the cell matches the style's size.
fn draw_minecraft(
    painter: &mut Painter<'_>,
    mc: &mut FontRenderer,
    text: &str,
    x: f32,
    top: f32,
    style: TextStyle,
    color: Rgba,
) -> f32 {
    let k = style.size / 8.0;
    let packed = ((color[3].clamp(0.0, 1.0) * 255.0) as u32) << 24
        | ((color[0].clamp(0.0, 1.0) * 255.0) as u32) << 16
        | ((color[1].clamp(0.0, 1.0) * 255.0) as u32) << 8
        | (color[2].clamp(0.0, 1.0) * 255.0) as u32;
    // Vanilla treats an alpha below 4 as opaque; keep fully faded text invisible instead.
    if (packed >> 24) < 4 {
        return mc.string_width(text) as f32 * k;
    }
    let mut sink = ScaledSink {
        painter,
        x,
        y: top,
        k,
    };
    let advance = if style.upper {
        let upper: String = text.chars().map(|c| c.to_ascii_uppercase()).collect();
        mc.draw_string(&mut sink, &upper, 0.0, 0.0, packed as i32, false)
    } else {
        mc.draw_string(&mut sink, text, 0.0, 0.0, packed as i32, false)
    };
    advance as f32 * k
}

struct ScaledSink<'p, 'a> {
    painter: &'p mut Painter<'a>,
    x: f32,
    y: f32,
    k: f32,
}

impl TextSink for ScaledSink<'_, '_> {
    fn glyph(&mut self, quad: GlyphQuad) {
        let p = |i: usize| {
            [
                self.x + quad.pos[i][0] * self.k,
                self.y + quad.pos[i][1] * self.k,
            ]
        };
        let (a, d) = (p(0), p(3));
        let color = crate::widget::unpack_argb(quad.color);
        self.painter.textured(
            &UiTexture::Font(quad.texture),
            Rect::new(a[0], a[1], d[0] - a[0], d[1] - a[1]),
            [quad.uv[0][0], quad.uv[0][1], quad.uv[3][0], quad.uv[3][1]],
            color,
        );
    }

    fn bar(&mut self, rect: SolidRect) {
        self.painter.rect(
            Rect::new(
                self.x + rect.x0 * self.k,
                self.y + rect.y0 * self.k,
                (rect.x1 - rect.x0) * self.k,
                (rect.y1 - rect.y0) * self.k,
            ),
            crate::widget::unpack_argb(rect.color),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::test_support::uniform_font;
    use crate::widget::DrawList;

    #[test]
    fn the_minecraft_fallback_measures_and_draws_scaled_glyphs() {
        let mut text = MenuText::minecraft_only();
        let mut mc = uniform_font();
        let style = TextStyle::new(16.0);
        // Uniform font: 5 px per glyph at 8 px, so 10 px per glyph at 16 px.
        assert_eq!(text.measure("abc", style, &mc), 30.0);
        let mut list = DrawList::new();
        let mut painter = Painter::new(&mut list);
        let advance = text.draw(&mut painter, &mut mc, "ab", 100.0, 50.0, style, [1.0; 4]);
        assert_eq!(advance, 20.0);
        assert!(!list.is_empty());
        assert!(list
            .vertices()
            .iter()
            .all(|v| v.pos[0] >= 100.0 && v.pos[1] >= 50.0));
    }

    #[test]
    fn wrapping_breaks_at_spaces() {
        let text = MenuText::minecraft_only();
        let mc = uniform_font();
        let style = TextStyle::new(8.0);
        // 5 px per glyph: "aaa bbb ccc" is 55 px; 40 px fits "aaa bbb" (35 px).
        let cut = text.fit_words("aaa bbb ccc", 40.0, style, &mc);
        assert_eq!(&"aaa bbb ccc"[..cut], "aaa bbb");
        assert_eq!(text.fit_words("short", 100.0, style, &mc), 5);
    }

    #[test]
    fn segoe_glyphs_land_in_the_atlas_when_the_font_exists() {
        let mut text = MenuText::load();
        if !text.is_segoe() {
            eprintln!("SKIP: Segoe UI is not installed");
            return;
        }
        let mut mc = uniform_font();
        let mut list = DrawList::new();
        let mut painter = Painter::new(&mut list);
        let style = TextStyle::new(14.0);
        let width = text.draw(&mut painter, &mut mc, "Glass", 0.0, 0.0, style, [1.0; 4]);
        assert!(width > 20.0 && width < 60.0, "{width}");
        assert!(text.take_upload().is_some(), "new glyphs need an upload");
        assert!(text.take_upload().is_none(), "and only once");
        // Cached: drawing again adds nothing.
        let mut list = DrawList::new();
        let mut painter = Painter::new(&mut list);
        text.draw(&mut painter, &mut mc, "Glass", 0.0, 0.0, style, [1.0; 4]);
        assert!(text.take_upload().is_none());
        assert!((text.measure("Glass", style, &mc) - width).abs() < 0.01);
        let (ascent, line) = text.metrics(style);
        assert!(ascent > 10.0 && line > ascent);
    }
}
