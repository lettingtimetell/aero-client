//! The "Aero Client" logo: the title's fonts and looks, compiled into the client.
//!
//! **The logo has styles** ([`LogoStyle`], the Aero Title Screen mod's "Logo style" option). The
//! default is Inter Display Italic with no drop shadow. The others are a light italic with wider
//! letter spacing, a soft gradient, a thin underline accent, and Classic, the original upright
//! Inter Regular with its shadow. The title screen and the loading screen both draw it with
//! [`TitleFont::draw_logo`].
//!
//! The fonts are Inter (SIL Open Font License 1.1): the italics are the official Inter 4.1 release's
//! `InterDisplay-Italic.ttf` and `InterDisplay-LightItalic.ttf` (licence
//! `crates/mc-ui/assets/fonts/inter/LICENSE-Inter-4.1.txt`), Classic is Inter's 28pt Regular
//! (licence `OFL.txt` beside it). They are embedded rather than read at run time because the
//! installer ships only `mc-client.exe`. To try another font, point a style's `include_bytes!`
//! below at another `.ttf` or `.otf` (keep its licence beside it).
//!
//! The title is one short, fixed string, so it is rasterised as a whole: one bitmap of the line
//! (with kerning, the style's letter spacing and its fill colours) plus two Gaussian-blurred copies
//! of its coverage, a tight one for the drop shadow and a wide one for the glow. The three images
//! are stacked in one texture that is re-rasterised only when the size or the style changes, so a
//! frame costs two or three quads (and one more for the underline accent).

use ab_glyph::{Font, FontArc, GlyphId, PxScale, ScaleFont};

use crate::screens::modmenu::paint::{Fill, Painter, Rect, Rgba};
use crate::widget::UiTexture;

/// The default logo font: Inter Display Italic, from the official Inter 4.1 release.
pub const TITLE_FONT: &[u8] = include_bytes!("../../../assets/fonts/inter/InterDisplay-Italic.ttf");
/// The light italic style's font: Inter Display Light Italic (Inter 4.1).
pub const LIGHT_ITALIC_FONT: &[u8] =
    include_bytes!("../../../assets/fonts/inter/InterDisplay-LightItalic.ttf");
/// The Classic style's font: Inter Regular, Inter's 28pt display cut (the title before this change).
pub const CLASSIC_FONT: &[u8] =
    include_bytes!("../../../assets/fonts/inter/Inter_28pt-Regular.ttf");

/// The Aero Title Screen mod's id and its "Logo style" option's id (`mc_mods::mods::aero_title`).
pub const LOGO_MOD: &str = "aero_title";
/// See [`LOGO_MOD`].
pub const LOGO_OPTION: &str = "logo_style";

/// How the "Aero Client" logo looks: the "Logo style" option, in option order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LogoStyle {
    /// Inter Display Italic, white, no shadow, a faint glow. The default.
    #[default]
    Italic,
    /// Inter Display Light Italic with wider letter spacing: airier and quieter.
    LightItalic,
    /// Inter Display Italic filled with a soft white-to-sky-blue gradient.
    Gradient,
    /// Inter Display Italic with a thin accent line under its first letters.
    Underline,
    /// The original logo: upright Inter Regular with a soft drop shadow.
    Classic,
}

impl LogoStyle {
    /// Every style, in the option's order (index = the stored choice).
    pub const ALL: [LogoStyle; 5] = [
        LogoStyle::Italic,
        LogoStyle::LightItalic,
        LogoStyle::Gradient,
        LogoStyle::Underline,
        LogoStyle::Classic,
    ];

    /// The style stored as choice `index` (out of range: the default).
    #[must_use]
    pub fn from_choice(index: usize) -> LogoStyle {
        LogoStyle::ALL.get(index).copied().unwrap_or_default()
    }

    /// The style the settings pick (the Aero Title Screen mod's "Logo style"). A lookup in the
    /// settings store: cache it by the settings' generation in per-frame code.
    #[must_use]
    pub fn from_settings(settings: &mc_mods::ModSettings) -> LogoStyle {
        LogoStyle::from_choice(settings.choice(LOGO_MOD, LOGO_OPTION))
    }

    /// The option's label for this style (the descriptor lists the same, in the same order).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            LogoStyle::Italic => "Italic",
            LogoStyle::LightItalic => "Light italic",
            LogoStyle::Gradient => "Gradient",
            LogoStyle::Underline => "Underline",
            LogoStyle::Classic => "Classic",
        }
    }

    /// The style's font file.
    #[must_use]
    pub fn font(self) -> &'static [u8] {
        match self {
            LogoStyle::LightItalic => LIGHT_ITALIC_FONT,
            LogoStyle::Classic => CLASSIC_FONT,
            LogoStyle::Italic | LogoStyle::Gradient | LogoStyle::Underline => TITLE_FONT,
        }
    }

    /// Extra space after every letter, as a fraction of the em size.
    #[must_use]
    pub fn tracking_em(self) -> f32 {
        match self {
            LogoStyle::LightItalic => 0.05,
            _ => 0.0,
        }
    }

    /// The fill: one colour (`None`, white), or a gradient from the top-left to the bottom-right
    /// of the line, baked into the bitmap.
    #[must_use]
    pub fn gradient(self) -> Option<(Rgba, Rgba)> {
        match self {
            LogoStyle::Gradient => Some(([1.0, 1.0, 1.0, 1.0], [0.52, 0.78, 1.0, 1.0])),
            _ => None,
        }
    }

    /// How the style is lit: no drop shadow except Classic's, and the same faint glow.
    #[must_use]
    pub fn look(self) -> TitleLook {
        match self {
            LogoStyle::Classic => TitleLook::default(),
            LogoStyle::LightItalic => TitleLook {
                shadow: [0.0; 4],
                glow: [1.0, 1.0, 1.0, 0.18],
                ..TitleLook::default()
            },
            _ => TitleLook {
                shadow: [0.0; 4],
                ..TitleLook::default()
            },
        }
    }

    /// Whether the style draws the thin accent line under the logo.
    #[must_use]
    pub fn underline(self) -> bool {
        self == LogoStyle::Underline
    }
}

/// The tight blur under the title (the drop shadow), as a fraction of the em size.
const SHADOW_SIGMA_EM: f32 = 0.045;
/// The wide blur around the title (the glow), as a fraction of the em size.
const GLOW_SIGMA_EM: f32 = 0.11;
/// Empty rows between the stacked images, so linear filtering never bleeds between them.
const GAP: u32 = 2;

/// The UI texture key the title bitmap is uploaded under.
#[must_use]
pub fn atlas_key() -> UiTexture {
    UiTexture::Resource(mc_assets::ResourceLocation::new("mcrs", "aero_title/title"))
}

/// How the title is lit: a soft drop shadow for contrast over the bright greenhouse, and a faint
/// white glow for the dreamy look.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TitleLook {
    /// The text colour.
    pub color: Rgba,
    /// The drop shadow's colour (its alpha is the strength).
    pub shadow: Rgba,
    /// The shadow's downward offset, as a fraction of the em size.
    pub shadow_offset_em: f32,
    /// The glow's colour (its alpha is the strength).
    pub glow: Rgba,
}

impl Default for TitleLook {
    /// Tuned over the greenhouse: its roof and leaves are bright and busy right behind the
    /// title, so the dark shadow carries the legibility and the white glow stays faint (the
    /// scene's own bloom adds the rest of the glow).
    fn default() -> Self {
        TitleLook {
            color: [1.0, 1.0, 1.0, 1.0],
            shadow: [0.0, 0.03, 0.05, 0.58],
            shadow_offset_em: 0.02,
            glow: [1.0, 1.0, 1.0, 0.14],
        }
    }
}

/// One rasterised line: three stacked RGBA images (sharp, shadow blur, glow blur), white with the
/// coverage in alpha.
#[derive(Debug, Clone)]
struct TitleBitmap {
    text: String,
    /// The em size in quarter pixels.
    size_q: u32,
    /// One image's size; the texture is `width` by `3 * height + 2 * GAP`.
    width: u32,
    height: u32,
    /// RGBA8, the whole texture.
    pixels: Vec<u8>,
    /// Where the image's top-left sits relative to the pen origin on the baseline.
    origin: [f32; 2],
    /// The left edge of the first glyph's ink, relative to the pen origin.
    ink_left: f32,
    /// The right edge of the line's ink, relative to the pen origin.
    ink_right: f32,
}

impl TitleBitmap {
    fn texture_height(&self) -> u32 {
        self.height * 3 + GAP * 2
    }

    /// UVs of image `index` (0 sharp, 1 shadow, 2 glow).
    fn uv(&self, index: u32) -> [f32; 4] {
        let total = self.texture_height() as f32;
        let top = (index * (self.height + GAP)) as f32;
        [0.0, top / total, 1.0, (top + self.height as f32) / total]
    }
}

/// The title font and its rasterised line.
pub struct TitleFont {
    font: Option<FontArc>,
    style: LogoStyle,
    bitmap: Option<TitleBitmap>,
    dirty: bool,
}

impl std::fmt::Debug for TitleFont {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TitleFont")
            .field("loaded", &self.font.is_some())
            .field("style", &self.style)
            .field("bitmap", &self.bitmap.as_ref().map(|b| (b.width, b.height)))
            .finish_non_exhaustive()
    }
}

impl Default for TitleFont {
    fn default() -> Self {
        TitleFont::embedded()
    }
}

impl TitleFont {
    /// The default style's compiled-in font ([`TITLE_FONT`]).
    #[must_use]
    pub fn embedded() -> TitleFont {
        TitleFont::for_style(LogoStyle::default())
    }

    /// The compiled-in font of `style`, drawn in that style.
    #[must_use]
    pub fn for_style(style: LogoStyle) -> TitleFont {
        TitleFont {
            style,
            ..TitleFont::from_bytes(style.font())
        }
    }

    /// A title font from TTF/OTF bytes (the review renders try other weights this way), drawn in
    /// the default style's look. Bytes that do not parse leave the font unavailable, and the
    /// caller draws the title another way.
    #[must_use]
    pub fn from_bytes(bytes: &'static [u8]) -> TitleFont {
        TitleFont {
            font: FontArc::try_from_slice(bytes).ok(),
            style: LogoStyle::default(),
            bitmap: None,
            dirty: false,
        }
    }

    /// The style in use.
    #[must_use]
    pub fn style(&self) -> LogoStyle {
        self.style
    }

    /// Switch to `style` (its font, spacing, fill and look). A no-op when it is already in use,
    /// so it can be called every frame; a change re-rasterises the line on the next
    /// [`TitleFont::prepare`].
    pub fn set_style(&mut self, style: LogoStyle) {
        if style == self.style {
            return;
        }
        self.style = style;
        self.font = FontArc::try_from_slice(style.font()).ok();
        self.bitmap = None;
    }

    /// How the current style is lit ([`LogoStyle::look`]).
    #[must_use]
    pub fn look(&self) -> TitleLook {
        self.style.look()
    }

    /// Whether the font parsed.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.font.is_some()
    }

    fn scale_for(font: &FontArc, size: f32) -> PxScale {
        // ab_glyph's PxScale is the ascent-to-descent height; sizes here are the em.
        let em = font.units_per_em().unwrap_or(1000.0);
        PxScale::from(size * font.height_unscaled() / em)
    }

    /// The height of a capital letter at em size `size`, in pixels (from the font's own `H`).
    #[must_use]
    pub fn cap_height(&self, size: f32) -> f32 {
        let Some(font) = &self.font else {
            return size * 0.7;
        };
        let scale = Self::scale_for(font, size);
        let id = font.glyph_id('H');
        font.outline_glyph(id.with_scale_and_position(scale, ab_glyph::point(0.0, 0.0)))
            .map_or(size * 0.7, |g| {
                let b = g.px_bounds();
                b.max.y - b.min.y
            })
    }

    /// The advance width of `text` at em size `size`, in pixels.
    #[must_use]
    pub fn measure(&self, text: &str, size: f32) -> f32 {
        let Some(font) = &self.font else {
            return 0.0;
        };
        let scaled = font.as_scaled(Self::scale_for(font, size));
        let tracking = self.style.tracking_em() * size;
        let mut width = 0.0;
        let mut previous: Option<GlyphId> = None;
        for c in text.chars() {
            let id = scaled.glyph_id(c);
            if let Some(p) = previous {
                width += scaled.kern(p, id) + tracking;
            }
            width += scaled.h_advance(id);
            previous = Some(id);
        }
        width
    }

    /// Rasterise `text` at em size `size` unless that exact line is already rasterised. Marks the
    /// texture for upload when it changed. Returns whether the font is available.
    pub fn prepare(&mut self, text: &str, size: f32) -> bool {
        let Some(font) = self.font.clone() else {
            return false;
        };
        let size_q = (size * 4.0).round().clamp(4.0, 65_535.0) as u32;
        if self
            .bitmap
            .as_ref()
            .is_some_and(|b| b.size_q == size_q && b.text == text)
        {
            return true;
        }
        let size = size_q as f32 / 4.0;
        self.bitmap = Some(rasterise(&font, text, (size, size_q), self.style));
        self.dirty = true;
        true
    }

    /// Make the next [`TitleFont::take_upload`] hand out the texture again (its renderer was
    /// rebuilt).
    pub fn mark_dirty(&mut self) {
        if self.bitmap.is_some() {
            self.dirty = true;
        }
    }

    /// The texture's pixels and size, if it changed since the last call (upload them under
    /// [`atlas_key`]).
    pub fn take_upload(&mut self) -> Option<(&[u8], u32, u32)> {
        if !std::mem::take(&mut self.dirty) {
            return None;
        }
        let bitmap = self.bitmap.as_ref()?;
        Some((&bitmap.pixels, bitmap.width, bitmap.texture_height()))
    }

    /// The left edge of the prepared line's first glyph ink, relative to its pen origin (so a
    /// caller can align the ink, not the pen, to a column).
    #[must_use]
    pub fn ink_left(&self) -> f32 {
        self.bitmap.as_ref().map_or(0.0, |b| b.ink_left)
    }

    /// The right edge of the prepared line's ink, relative to its pen origin.
    #[must_use]
    pub fn ink_right(&self) -> f32 {
        self.bitmap.as_ref().map_or(0.0, |b| b.ink_right)
    }

    /// Draw the prepared line as the logo, in the current style: [`TitleFont::draw`] with the
    /// style's look, plus the underline accent for [`LogoStyle::Underline`]. The pen origin is at
    /// `(x, baseline)`. Does nothing before [`TitleFont::prepare`].
    pub fn draw_logo(&self, painter: &mut Painter<'_>, x: f32, baseline: f32) {
        let Some(bitmap) = &self.bitmap else {
            return;
        };
        self.draw(painter, x, baseline, self.look());
        if self.style.underline() {
            let size = bitmap.size_q as f32 / 4.0;
            let left = (x + bitmap.ink_left).round();
            let ink = (bitmap.ink_right - bitmap.ink_left).max(1.0);
            let thick = (0.032 * size).max(2.0).round();
            let rect = Rect::new(
                left,
                (baseline + 0.17 * size).round(),
                (ink * 0.56).round(),
                thick,
            );
            let fill = Fill::css_angle(rect, 90.0, [1.0, 1.0, 1.0, 0.95], [0.62, 0.84, 1.0, 0.0]);
            painter.round_rect(rect, thick * 0.5, fill);
        }
    }

    /// Draw the prepared line with its pen origin at `(x, baseline)`: the shadow, the glow, then
    /// the text. Does nothing before [`TitleFont::prepare`].
    pub fn draw(&self, painter: &mut Painter<'_>, x: f32, baseline: f32, look: TitleLook) {
        let Some(bitmap) = &self.bitmap else {
            return;
        };
        let size = bitmap.size_q as f32 / 4.0;
        // Whole pixels, so the sharp image maps texel to pixel.
        let left = (x + bitmap.origin[0]).round();
        let top = (baseline + bitmap.origin[1]).round();
        let (w, h) = (bitmap.width as f32, bitmap.height as f32);
        let texture = atlas_key();
        let shadow_dy = (look.shadow_offset_em * size).round();
        if look.shadow[3] > 0.0 {
            painter.textured(
                &texture,
                Rect::new(left, top + shadow_dy, w, h),
                bitmap.uv(1),
                look.shadow,
            );
        }
        if look.glow[3] > 0.0 {
            painter.textured(
                &texture,
                Rect::new(left, top, w, h),
                bitmap.uv(2),
                look.glow,
            );
        }
        painter.textured(
            &texture,
            Rect::new(left, top, w, h),
            bitmap.uv(0),
            look.color,
        );
    }
}

/// Rasterise one line into the three stacked images, in `style`'s spacing and fill.
fn rasterise(
    font: &FontArc,
    text: &str,
    (size, size_q): (f32, u32),
    style: LogoStyle,
) -> TitleBitmap {
    let scale = TitleFont::scale_for(font, size);
    let scaled = font.as_scaled(scale);
    let shadow_sigma = (SHADOW_SIGMA_EM * size).max(0.5);
    let glow_sigma = (GLOW_SIGMA_EM * size).max(1.0);
    let tracking = style.tracking_em() * size;
    // Italic strokes lean past their advance: a little more room on the right.
    let pad = (glow_sigma * 3.0).ceil() as u32 + 2 + (0.12 * size).ceil() as u32;
    let ascent = scaled.ascent();
    let descent = scaled.descent();
    // Lay the glyphs out first, to size the image.
    let mut glyphs = Vec::with_capacity(text.len());
    let mut pen = 0.0_f32;
    let mut previous: Option<GlyphId> = None;
    for c in text.chars() {
        let id = scaled.glyph_id(c);
        if let Some(p) = previous {
            pen += scaled.kern(p, id) + tracking;
        }
        glyphs.push((id, pen));
        pen += scaled.h_advance(id);
        previous = Some(id);
    }
    let width = pen.ceil().max(1.0) as u32 + pad * 2 + 2;
    let height = (ascent - descent).ceil().max(1.0) as u32 + pad * 2;
    let (w, h) = (width as usize, height as usize);
    let mut coverage = vec![0.0_f32; w * h];
    let baseline = pad as f32 + ascent;
    let mut ink_left = f32::INFINITY;
    let mut ink_right = f32::NEG_INFINITY;
    for &(id, x) in &glyphs {
        let glyph =
            id.with_scale_and_position(scale, ab_glyph::point(pad as f32 + x, baseline.round()));
        let Some(outlined) = font.outline_glyph(glyph) else {
            continue;
        };
        let bounds = outlined.px_bounds();
        if ink_left.is_infinite() {
            ink_left = bounds.min.x - pad as f32;
        }
        ink_right = ink_right.max(bounds.max.x - pad as f32);
        let (ox, oy) = (bounds.min.x as i64, bounds.min.y as i64);
        outlined.draw(|gx, gy, c| {
            let px = ox + i64::from(gx);
            let py = oy + i64::from(gy);
            if px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < h {
                let cell = &mut coverage[py as usize * w + px as usize];
                *cell = (*cell + c).min(1.0);
            }
        });
    }
    let shadow = gaussian_blur(&coverage, w, h, shadow_sigma);
    let glow = gaussian_blur(&coverage, w, h, glow_sigma);
    let texture_height = height * 3 + GAP * 2;
    let mut pixels = vec![0_u8; w * texture_height as usize * 4];
    let ink_left = if ink_left.is_finite() { ink_left } else { 0.0 };
    let ink_right = if ink_right.is_finite() {
        ink_right
    } else {
        pen
    };
    // The sharp image carries the fill: white, or the style's gradient from the ink's top-left
    // to its bottom-right. The blurred ones stay white (the look tints them).
    let (ink_top, ink_bottom) = (baseline - ascent * 0.75, baseline);
    let fill = |x: usize, y: usize| -> [u8; 3] {
        let Some((a, b)) = style.gradient() else {
            return [255; 3];
        };
        let tx =
            ((x as f32 - pad as f32 - ink_left) / (ink_right - ink_left).max(1.0)).clamp(0.0, 1.0);
        let ty = ((y as f32 - ink_top) / (ink_bottom - ink_top).max(1.0)).clamp(0.0, 1.0);
        let t = (tx * 0.7 + ty * 0.3).clamp(0.0, 1.0);
        let t = t * t * (3.0 - 2.0 * t);
        [0, 1, 2].map(|k| ((a[k] + (b[k] - a[k]) * t).clamp(0.0, 1.0) * 255.0).round() as u8)
    };
    for (index, image) in [&coverage, &shadow, &glow].into_iter().enumerate() {
        let top = index * (h + GAP as usize);
        for y in 0..h {
            for x in 0..w {
                let a = image[y * w + x].clamp(0.0, 1.0);
                let i = ((top + y) * w + x) * 4;
                let rgb = if index == 0 { fill(x, y) } else { [255; 3] };
                pixels[i..i + 3].copy_from_slice(&rgb);
                pixels[i + 3] = (a * 255.0).round() as u8;
            }
        }
    }
    TitleBitmap {
        text: text.to_owned(),
        size_q,
        width,
        height,
        pixels,
        origin: [-(pad as f32), -(pad as f32 + ascent)],
        ink_left,
        ink_right,
    }
}

/// A Gaussian blur of a single-channel image, approximated by three box blurs per axis (O(1)
/// per pixel whatever the radius, so the wide glow costs the same as the shadow).
fn gaussian_blur(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    // Three passes of width `2r + 1` have a variance of `r(r + 1)`: pick r so that matches.
    let r = ((-1.0 + (1.0 + 4.0 * sigma * sigma).sqrt()) * 0.5)
        .round()
        .max(1.0) as usize;
    let mut a = src.to_vec();
    let mut b = vec![0.0_f32; src.len()];
    for _ in 0..3 {
        box_rows(&a, &mut b, w, h, r);
        box_cols(&b, &mut a, w, h, r);
    }
    a
}

fn box_rows(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: usize) {
    let norm = 1.0 / (2 * r + 1) as f32;
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let out = &mut dst[y * w..(y + 1) * w];
        let mut sum: f32 = row.iter().take(r.min(w)).sum();
        for x in 0..w {
            if x + r < w {
                sum += row[x + r];
            }
            out[x] = sum * norm;
            if x >= r {
                sum -= row[x - r];
            }
        }
    }
}

fn box_cols(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: usize) {
    let norm = 1.0 / (2 * r + 1) as f32;
    for x in 0..w {
        let mut sum = 0.0_f32;
        for y in 0..r.min(h) {
            sum += src[y * w + x];
        }
        for y in 0..h {
            if y + r < h {
                sum += src[(y + r) * w + x];
            }
            dst[y * w + x] = sum * norm;
            if y >= r {
                sum -= src[(y - r) * w + x];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::DrawList;

    #[test]
    fn the_embedded_inter_parses_and_measures_the_title() {
        let font = TitleFont::embedded();
        assert!(font.is_available(), "the bundled Inter must parse");
        let width = font.measure("Aero Client", 100.0);
        // Inter is about 0.5 em per glyph here: well over 4 em, well under 7.
        assert!(width > 400.0 && width < 700.0, "{width}");
        let cap = font.cap_height(100.0);
        assert!(
            cap > 65.0 && cap < 80.0,
            "Inter's cap height is ~0.73 em: {cap}"
        );
    }

    #[test]
    fn the_line_is_rasterised_once_per_size_and_uploaded_once() {
        let mut font = TitleFont::embedded();
        assert!(font.prepare("Aero Client", 64.0));
        let (pixels, w, h) = font
            .take_upload()
            .map(|(p, w, h)| (p.to_vec(), w, h))
            .unwrap();
        assert_eq!(pixels.len(), (w * h * 4) as usize);
        assert!(font.take_upload().is_none(), "uploaded once");
        assert!(font.prepare("Aero Client", 64.0));
        assert!(font.take_upload().is_none(), "the same line is cached");
        assert!(font.prepare("Aero Client", 80.0));
        assert!(font.take_upload().is_some(), "a new size re-rasterises");
        // The sharp image has solid ink, the blurred ones only soft alpha.
        let alpha = |img: u32| {
            let bitmap = font.bitmap.as_ref().unwrap();
            let top = (img * (bitmap.height + GAP)) as usize;
            let mut max = 0_u8;
            for y in 0..bitmap.height as usize {
                for x in 0..bitmap.width as usize {
                    let i = ((top + y) * bitmap.width as usize + x) * 4 + 3;
                    max = max.max(bitmap.pixels[i]);
                }
            }
            max
        };
        assert!(alpha(0) >= 250, "sharp ink");
        assert!(
            alpha(1) > 60 && alpha(1) < 255,
            "shadow is soft: {}",
            alpha(1)
        );
        assert!(
            alpha(2) > 10 && alpha(2) < alpha(1),
            "glow is wider and fainter"
        );
    }

    #[test]
    fn drawing_emits_shadow_glow_and_text_quads_on_whole_pixels() {
        let mut font = TitleFont::embedded();
        assert!(font.prepare("Aero Client", 50.0));
        let mut list = DrawList::new();
        let mut painter = Painter::new(&mut list);
        font.draw(&mut painter, 100.3, 200.7, TitleLook::default());
        // Three quads of two triangles.
        assert_eq!(list.vertices().len(), 18);
        for v in list.vertices() {
            assert_eq!(v.pos[0], v.pos[0].round(), "whole pixels: {:?}", v.pos);
        }
    }

    /// The user's ask: the logo without its shadow, in italics, as the new default.
    #[test]
    fn the_default_logo_is_inter_italic_without_a_shadow() {
        assert_eq!(LogoStyle::default(), LogoStyle::Italic);
        let font = TitleFont::embedded();
        assert_eq!(font.style(), LogoStyle::Italic);
        assert_eq!(font.look().shadow[3], 0.0, "no drop shadow");
        let parsed = FontArc::try_from_slice(TITLE_FONT).expect("Inter Display Italic parses");
        // The italic leans: the top of the "l" sits right of its foot.
        let scaled = parsed.as_scaled(PxScale::from(100.0));
        let l = parsed
            .outline_glyph(scaled.scaled_glyph('l'))
            .expect("an outline");
        let upright = FontArc::try_from_slice(CLASSIC_FONT).expect("Inter Regular parses");
        let l_up = upright
            .outline_glyph(upright.as_scaled(PxScale::from(100.0)).scaled_glyph('l'))
            .expect("an outline");
        let width = |g: &ab_glyph::OutlinedGlyph| g.px_bounds().max.x - g.px_bounds().min.x;
        assert!(
            width(&l) > width(&l_up) * 1.8,
            "a slanted stroke is much wider than an upright one: {} vs {}",
            width(&l),
            width(&l_up)
        );
        // Drawn: the glow and the text, no shadow quad (Classic keeps all three).
        let mut font = TitleFont::embedded();
        assert!(font.prepare("Aero Client", 50.0));
        let mut list = DrawList::new();
        font.draw_logo(&mut Painter::new(&mut list), 10.0, 80.0);
        assert_eq!(list.vertices().len(), 12);
        let mut classic = TitleFont::for_style(LogoStyle::Classic);
        assert!(classic.prepare("Aero Client", 50.0));
        let mut list = DrawList::new();
        classic.draw_logo(&mut Painter::new(&mut list), 10.0, 80.0);
        assert_eq!(list.vertices().len(), 18, "Classic keeps its shadow");
    }

    #[test]
    fn every_style_draws_its_own_way() {
        let mut widths = Vec::new();
        for style in LogoStyle::ALL {
            let mut font = TitleFont::for_style(style);
            assert!(font.is_available(), "{style:?}'s font parses");
            assert!(font.prepare("Aero Client", 64.0));
            assert!(font.ink_right() > font.ink_left() + 200.0, "{style:?}");
            widths.push(font.measure("Aero Client", 64.0));
            let bitmap = font.bitmap.as_ref().expect("rasterised");
            // The sharp image's colours: white, except the gradient's.
            let tinted = bitmap.pixels[..(bitmap.width * bitmap.height * 4) as usize]
                .chunks(4)
                .any(|p| p[3] > 128 && p[0] < 230);
            assert_eq!(tinted, style == LogoStyle::Gradient, "{style:?}");
            let mut list = DrawList::new();
            font.draw_logo(&mut Painter::new(&mut list), 0.0, 100.0);
            let quads = list.vertices().len();
            match style {
                LogoStyle::Underline => assert!(quads > 12, "the accent line: {quads}"),
                LogoStyle::Classic => assert_eq!(quads, 18),
                _ => assert_eq!(quads, 12, "{style:?}"),
            }
        }
        // The light italic is spaced wider than the italic, by exactly its tracking: 0.05 em
        // between each of the ten pairs of letters, over the same font untracked.
        assert!(widths[1] > widths[0], "{widths:?}");
        let untracked = TitleFont::from_bytes(LIGHT_ITALIC_FONT).measure("Aero Client", 64.0);
        let extra = widths[1] - untracked;
        assert!((extra - 10.0 * 0.05 * 64.0).abs() < 0.01, "{extra}");
    }

    #[test]
    fn switching_style_re_rasterises_once() {
        let mut font = TitleFont::embedded();
        assert!(font.prepare("Aero Client", 64.0));
        assert!(font.take_upload().is_some());
        font.set_style(LogoStyle::Italic);
        assert!(font.prepare("Aero Client", 64.0));
        assert!(
            font.take_upload().is_none(),
            "the same style changes nothing"
        );
        font.set_style(LogoStyle::Gradient);
        assert!(font.prepare("Aero Client", 64.0));
        assert!(
            font.take_upload().is_some(),
            "a new style uploads a new line"
        );
        assert!(font.take_upload().is_none());
    }

    /// The option's choices are the styles, in order, with matching labels (the settings file
    /// stores the label).
    #[test]
    fn the_logo_style_option_lists_the_styles_in_order() {
        let descriptor = mc_mods::catalog::find(LOGO_MOD).expect("the Aero Title Screen mod");
        let option = descriptor
            .option(LOGO_OPTION)
            .expect("the Logo style option");
        let mc_mods::OptionKind::Choice { choices } = option.kind else {
            panic!("a choice: {:?}", option.kind);
        };
        let labels: Vec<&str> = LogoStyle::ALL.iter().map(|s| s.label()).collect();
        assert_eq!(choices, labels.as_slice());
        assert_eq!(option.default, mc_mods::OptionValue::Choice(0));
        assert_eq!(LogoStyle::from_choice(0), LogoStyle::default());
        assert_eq!(LogoStyle::from_choice(99), LogoStyle::default());
        // Read through the settings store, the default and a change.
        let mut settings = mc_mods::ModSettings::default();
        assert_eq!(LogoStyle::from_settings(&settings), LogoStyle::Italic);
        assert!(settings.set_value(LOGO_MOD, LOGO_OPTION, mc_mods::OptionValue::Choice(4)));
        assert_eq!(LogoStyle::from_settings(&settings), LogoStyle::Classic);
    }

    #[test]
    fn the_blur_keeps_the_total_and_spreads_it() {
        let (w, h) = (41, 41);
        let mut src = vec![0.0; w * h];
        src[20 * w + 20] = 1.0;
        let out = gaussian_blur(&src, w, h, 3.0);
        let total: f32 = out.iter().sum();
        assert!((total - 1.0).abs() < 1e-3, "{total}");
        assert!(out[20 * w + 20] < 0.1 && out[20 * w + 23] > 0.0);
    }
}
