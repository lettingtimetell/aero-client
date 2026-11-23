//! Vanilla's `FontRenderer`, ported.
//!
//! java: `client/gui/FontRenderer.java`
//!
//! # UTF-16, deliberately
//!
//! Every measurement and wrap routine here walks **UTF-16 code units**, not Rust `char`s,
//! because vanilla walks Java `char`s and the observable behaviour differs: a supplementary
//! character is two zero-width units to vanilla, `sizeStringToWidth` returns a UTF-16 index,
//! and `trimStringToWidth` can split a surrogate pair. Reproducing that is the point — a chat
//! line has to break in exactly the place the server assumed it would. Public entry points take
//! and return `&str`/`String`; the UTF-16 detail is internal, except that a split surrogate pair
//! comes back as `U+FFFD` where Java would have produced an unpaired surrogate.
//!
//! # Bidirectional text
//!
//! [`FontRenderer::set_bidi_flag`] is live: when enabled, drawn strings go through
//! [`bidi_reorder`] before width alignment and glyph emission. The implementation keeps the
//! UAX #9 run order and Arabic presentation-form shaping needed by the mixed Hebrew/Latin,
//! number and punctuation strings used by 1.8.9's UI, without making the renderer depend on a
//! platform ICU installation.
//!
//! # Rendering model
//!
//! Nothing here touches the GPU. Layout emits [`GlyphQuad`]s and [`SolidRect`]s into a
//! [`TextSink`]; [`crate::widget::DrawList`] is the sink that turns them into triangles, and
//! [`crate::widget::UiPhase`] is what eventually draws them. Tests use [`TextMesh`], a sink
//! that just collects, which is how the exact vertex coordinates below are checked without a
//! display.

pub mod bidi;
pub mod component;
pub mod format;
pub mod glyphs;

use mc_assets::{LanguageInfo, ResourceManager};
use mc_math::random::JavaRandom;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::UiError;
pub use bidi::reorder as bidi_reorder;
pub use component::{ChatComponent, ChatStyle, ClickAction, ClickEvent, HoverAction, HoverEvent};
pub use format::{
    color_table, format_from_string, is_format_color, is_format_special, strip_formatting,
    TextFormat, SECTION_SIGN,
};
pub use glyphs::{
    font_char_index, measure_ascii_widths, unicode_page_location, FontData, ASCII_FONT, FONT_CHARS,
    FONT_HEIGHT, GLYPH_SIZES, STANDARD_GALACTIC_FONT,
};

/// Which font texture a [`GlyphQuad`] samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FontTexture {
    /// `textures/font/ascii.png`, a 16×16 grid of 8×8 cells.
    Ascii,
    /// `textures/font/ascii_sga.png`, used only by enchanting-table offer runes.
    StandardGalactic,
    /// `textures/font/unicode_page_XX.png`, a 16×16 grid of 16×16 cells.
    UnicodePage(u8),
}

impl FontTexture {
    /// The resource this texture loads from.
    #[must_use]
    pub fn location(self) -> mc_assets::ResourceLocation {
        match self {
            FontTexture::Ascii => mc_assets::ResourceLocation::minecraft(ASCII_FONT),
            FontTexture::StandardGalactic => {
                mc_assets::ResourceLocation::minecraft(STANDARD_GALACTIC_FONT)
            }
            FontTexture::UnicodePage(p) => unicode_page_location(p),
        }
    }
}

/// One glyph, as the four corners of vanilla's `GL_TRIANGLE_STRIP` quad.
///
/// Corner order is vanilla's: `[top-left, bottom-left, top-right, bottom-right]`. Italic text
/// shears the top corners `+1` and the bottom corners `−1` in x, so the quad is a
/// parallelogram and the four positions cannot be collapsed to a rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphQuad {
    /// Texture to sample.
    pub texture: FontTexture,
    /// Corner positions in GUI pixels.
    pub pos: [[f32; 2]; 4],
    /// Corner texture coordinates, normalised to the texture size.
    pub uv: [[f32; 2]; 4],
    /// Packed `0xAARRGGBB`.
    pub color: u32,
}

/// An untextured rectangle: strikethrough and underline bars.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolidRect {
    /// Left edge.
    pub x0: f32,
    /// Top edge.
    pub y0: f32,
    /// Right edge.
    pub x1: f32,
    /// Bottom edge.
    pub y1: f32,
    /// Packed `0xAARRGGBB`.
    pub color: u32,
}

/// Where laid-out text goes.
pub trait TextSink {
    /// Marks the shadow or foreground pass. Most 2D sinks need no special handling; world
    /// labels use this to keep a coplanar shadow from writing depth over its own foreground.
    fn shadow_pass(&mut self, _shadow: bool) {}
    /// Emit one textured glyph quad.
    fn glyph(&mut self, quad: GlyphQuad);
    /// Emit one solid bar (strikethrough or underline).
    fn bar(&mut self, rect: SolidRect);
}

/// A [`TextSink`] that just collects, for tests and for callers that want the geometry.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct TextMesh {
    /// Glyph quads in emission order.
    pub glyphs: Vec<GlyphQuad>,
    /// Strikethrough/underline bars in emission order.
    pub bars: Vec<SolidRect>,
}

impl TextMesh {
    /// An empty mesh.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop everything, keeping the allocations.
    pub fn clear(&mut self) {
        self.glyphs.clear();
        self.bars.clear();
    }
}

impl TextSink for TextMesh {
    fn glyph(&mut self, quad: GlyphQuad) {
        self.glyphs.push(quad);
    }
    fn bar(&mut self, rect: SolidRect) {
        self.bars.push(rect);
    }
}

/// A [`TextSink`] that discards everything — for measuring a draw without drawing it.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullSink;

impl TextSink for NullSink {
    fn glyph(&mut self, _: GlyphQuad) {}
    fn bar(&mut self, _: SolidRect) {}
}

/// The index of the last colour code seen, `-1` for none. Used only by the `§r` reset path.
const NO_COLOR: i32 = -1;

/// How far inside its cell a default (`ascii.png`) glyph's **top and left** texture edges start,
/// in the sheet's 128-unit texel space. Item 10 ("lines over
/// O's").
///
/// Vanilla already keeps the bottom and right edges `0.01` inside the cell (`j + 7.99`,
/// `i + width - 1.01`) but puts the top and left exactly on its edge. When a glyph's top edge
/// lands on a pixel centre, which fractional scales do (chat scale, HUD scale, the Spotify
/// widget; the user's chat scale 95/142 at GUI scale 3 puts the 8th chat line at 853.5 px), the
/// rasteriser covers that pixel but the interpolated `v` can come out a hair above the cell, and
/// nearest filtering samples the bottom row of the glyph above: `_` sits over `o`, `g` over `w`,
/// `j` over `z`. The same `0.01` on the top and left makes the cell symmetric. The vertex
/// positions stay vanilla's, so coverage is unchanged, and at integer scales every pixel still
/// samples the same texel (`the_uv_inset_keeps_every_integer_scale_pixel_on_the_same_texel`).
const ASCII_UV_INSET: f32 = 0.01;

/// [`ASCII_UV_INSET`] for the unicode pages: `0.01` font pixels is `0.02` of their texels, the
/// margin vanilla keeps on their right (`f1 - f - 0.02`) and bottom (`f3 + 15.98`) edges.
const UNICODE_UV_INSET: f32 = 0.02;

// java.util.Random's no-argument constructor combines a process-wide uniquifier with
// System.nanoTime().  FontRenderer declares `fontRandom = new Random()` as a field
// initializer, so this seed is chosen once per renderer construction, not per draw.
const JAVA_RANDOM_SEED_UNIQUIFIER: u64 = 8_682_522_807_148_012;
const JAVA_RANDOM_SEED_MULTIPLIER: u64 = 181_783_497_276_258_689;
static FONT_RANDOM_SEED_UNIQUIFIER: AtomicU64 = AtomicU64::new(JAVA_RANDOM_SEED_UNIQUIFIER);

/// Source of [`FontRenderer::layout_generation`] values. Process-wide rather than per renderer so
/// a renderer that *replaces* another (`Client::font = FontRenderer::load(..)`) can never reuse
/// an identity a layout cache already holds entries for.
static FONT_LAYOUT_GENERATION: AtomicU64 = AtomicU64::new(1);

fn next_layout_generation() -> u64 {
    FONT_LAYOUT_GENERATION.fetch_add(1, Ordering::Relaxed)
}

fn font_random_seed() -> i64 {
    let mut old = FONT_RANDOM_SEED_UNIQUIFIER.load(Ordering::Relaxed);
    let uniquifier = loop {
        let next = old.wrapping_mul(JAVA_RANDOM_SEED_MULTIPLIER);
        match FONT_RANDOM_SEED_UNIQUIFIER.compare_exchange_weak(
            old,
            next,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break next,
            Err(actual) => old = actual,
        }
    };
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos() as u64);
    (uniquifier ^ nanos) as i64
}

/// Vanilla's font renderer: measurement, the `§` state machine, and glyph-quad emission.
///
/// Measurement ([`string_width`](Self::string_width), [`trim_to_width`](Self::trim_to_width),
/// [`wrap_to_width`](Self::wrap_to_width)) takes `&self`. Drawing takes `&mut self` because
/// vanilla's renderer is a state machine with a cursor, a colour and five style flags living in
/// fields — including the quirk that `drawString`'s two shadow passes *share* that state.
pub struct FontRenderer {
    data: FontData,
    default_texture: FontTexture,
    anaglyph: bool,
    colors: [i32; 32],
    unicode: bool,
    bidi: bool,
    font_height: i32,
    rng: JavaRandom,
    /// Changes whenever anything that decides glyph *layout* changes: the sheet's widths
    /// (resource reload), the unicode/bidi flags, or the anaglyph palette. Not a vanilla field;
    /// see [`Self::layout_generation`].
    layout_generation: u64,
    /// Whether symbols cast a visible shadow ([`Self::set_symbol_shadows`]). Off, the default,
    /// is vanilla. Not a vanilla field.
    symbol_shadows: bool,

    // --- per-string render state, one field per vanilla field ---
    pos_x: f32,
    pos_y: f32,
    /// Base colour of the string, restored by `§r`. Vanilla stores this as four floats.
    base_color: u32,
    /// The colour actually in effect for the next glyph.
    color: u32,
    text_color: i32,
    random_style: bool,
    bold_style: bool,
    italic_style: bool,
    underline_style: bool,
    strikethrough_style: bool,

    // --- not vanilla state: allocation reuse and the glyph-run cache (#8, C2-1, C2-8) ---
    /// Reused UTF-16 buffer for [`Self::draw_string`], so a draw allocates nothing.
    units_scratch: Vec<u16>,
    /// Whether the draw in progress wrote `text_color`. Only a colour code writes it, so a
    /// cached run must replay that write (and only that write) on a hit.
    text_color_written: bool,
    /// Whether the draw in progress consumed `fontRandom` (`§k`). Such a draw is never cached:
    /// vanilla re-rolls obfuscated glyphs on every draw and the stream must keep advancing.
    rng_used: bool,
    /// Finished [`Self::draw_string`] output, keyed by everything the draw reads.
    runs: RunCache,
}

impl std::fmt::Debug for FontRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FontRenderer")
            .field("unicode", &self.unicode)
            .field("bidi", &self.bidi)
            .field("font_height", &self.font_height)
            .finish_non_exhaustive()
    }
}

impl FontRenderer {
    /// Build from already-loaded width tables.
    #[must_use]
    pub fn new(data: FontData, anaglyph: bool, unicode: bool) -> Self {
        // Keep the historical unit-test fixture reproducible while preserving Java's
        // constructor-time entropy in production builds. Tests that need a particular stream
        // should use `new_with_seed`, which is available in all build modes.
        #[cfg(test)]
        {
            Self::new_with_seed(data, anaglyph, unicode, 0)
        }
        #[cfg(not(test))]
        {
            Self::new_with_texture(data, FontTexture::Ascii, anaglyph, unicode)
        }
    }

    /// Build the ordinary renderer with an explicitly seeded obfuscation stream.
    ///
    /// This is the deterministic construction seam for tests and replay tooling. Vanilla's
    /// production path uses [`Self::new`] / [`Self::new_with_texture`], whose stream is seeded
    /// from constructor-time entropy just like `new Random()` in Java.
    #[must_use]
    pub fn new_with_seed(data: FontData, anaglyph: bool, unicode: bool, seed: i64) -> Self {
        Self::new_with_texture_and_seed(data, FontTexture::Ascii, anaglyph, unicode, seed)
    }

    /// Build a renderer whose default 256-glyph sheet has a distinct GPU identity.
    #[must_use]
    pub fn new_with_texture(
        data: FontData,
        default_texture: FontTexture,
        anaglyph: bool,
        unicode: bool,
    ) -> Self {
        Self::new_with_texture_and_seed(
            data,
            default_texture,
            anaglyph,
            unicode,
            font_random_seed(),
        )
    }

    /// Build a renderer with an explicitly seeded obfuscation stream and texture identity.
    ///
    /// This differs from Java only in exposing the seed as a test/replay seam; normal
    /// construction should use [`Self::new_with_texture`].
    #[must_use]
    pub fn new_with_texture_and_seed(
        data: FontData,
        default_texture: FontTexture,
        anaglyph: bool,
        unicode: bool,
        seed: i64,
    ) -> Self {
        Self {
            data,
            default_texture,
            anaglyph,
            colors: color_table(anaglyph),
            unicode,
            bidi: false,
            font_height: FONT_HEIGHT,
            rng: JavaRandom::new(seed),
            layout_generation: next_layout_generation(),
            symbol_shadows: false,
            pos_x: 0.0,
            pos_y: 0.0,
            base_color: 0xFFFF_FFFF,
            color: 0xFFFF_FFFF,
            text_color: NO_COLOR,
            random_style: false,
            bold_style: false,
            italic_style: false,
            underline_style: false,
            strikethrough_style: false,
            units_scratch: Vec::new(),
            text_color_written: false,
            rng_used: false,
            runs: RunCache::new(),
        }
    }

    /// Reset the shared obfuscation stream to a known Java `Random` seed.
    ///
    /// This is intended for deterministic tests/replays. Ordinary rendering never reseeds the
    /// stream: consecutive draws continue from the state left by the preceding draw, matching
    /// vanilla's `fontRandom` field.
    pub fn reseed_obfuscation(&mut self, seed: i64) {
        self.rng.set_seed(seed);
    }

    /// Load `ascii.png` and `glyph_sizes.bin` from the resource packs.
    ///
    /// # Errors
    /// Whatever [`FontData::load`] returns.
    pub fn load(assets: &ResourceManager, anaglyph: bool, unicode: bool) -> Result<Self, UiError> {
        Ok(Self::new(FontData::load(assets)?, anaglyph, unicode))
    }

    /// Load the dedicated enchanting-table Standard Galactic renderer.
    pub fn load_standard_galactic(
        assets: &ResourceManager,
        anaglyph: bool,
    ) -> Result<Self, UiError> {
        Ok(Self::new_with_texture(
            FontData::load_with_sheet(assets, STANDARD_GALACTIC_FONT)?,
            FontTexture::StandardGalactic,
            anaglyph,
            false,
        ))
    }

    /// Refresh the default-font sheet without replacing constructor state.
    ///
    /// Vanilla's resource listener re-runs only `readFontTexture`; `glyph_sizes.bin` and the
    /// 32-entry colour palette remain those built when this renderer was constructed.
    ///
    /// # Errors
    /// Whatever [`FontData::reload_ascii`] returns.
    // java: FontRenderer.java:145-148 (onResourceManagerReload)
    pub fn reload_ascii(&mut self, assets: &ResourceManager) -> Result<(), UiError> {
        // New character widths change every cached advance and quad.
        self.runs.invalidate();
        self.layout_generation = next_layout_generation();
        self.data.reload_ascii(assets)
    }

    /// Refresh `ascii_sga.png` without changing this renderer's constructor state.
    pub fn reload_standard_galactic(&mut self, assets: &ResourceManager) -> Result<(), UiError> {
        self.runs.invalidate();
        self.layout_generation = next_layout_generation();
        self.data.reload_sheet(assets, STANDARD_GALACTIC_FONT)
    }

    /// An identity for everything that decides where glyphs go and what colour they are: the
    /// width tables, the unicode and bidi flags, the anaglyph palette and the symbol shadows.
    ///
    /// Not a vanilla concept. Callers that cache laid-out text (world-space labels, sign text)
    /// compare it to drop their cache when the font changes underneath them. It is drawn from a
    /// process-wide counter, so a *replaced* renderer never repeats an earlier renderer's value.
    /// Setters that are handed the value already in effect do not change it.
    #[must_use]
    pub fn layout_generation(&self) -> u64 {
        self.layout_generation
    }

    /// Whether drawing `text` could consume the shared `§k` obfuscation stream.
    ///
    /// Conservative: it may answer `true` for text that would not actually obfuscate (`§§k`,
    /// say), never `false` for text that would. Text for which this is `false` draws the same
    /// quads every time and leaves the stream untouched, which is what lets a layout cache skip
    /// the draw without changing any later `§k` draw's random glyphs.
    ///
    /// Mirrors `render_string_at_pos` exactly: every draw starts with `reset_styles`, only a
    /// `§` marker can set `random_style`, and the code is read from the **whole-string
    /// lowercase** at `i + 1` (so `§K` and `§\u{212A}` count). Bidi reordering can move units
    /// around, so with the bidi flag on every formatted string is treated as volatile.
    #[must_use]
    pub fn may_obfuscate(&self, text: &str) -> bool {
        if !text.contains('\u{a7}') {
            return false;
        }
        if self.bidi {
            return true;
        }
        let units = utf16(text);
        let lowercase: Vec<u16> = String::from_utf16_lossy(&units)
            .to_lowercase()
            .encode_utf16()
            .collect();
        units.iter().enumerate().any(|(i, &unit)| {
            unit == 167 && i + 1 < units.len() && lowercase.get(i + 1) == Some(&107)
        })
    }

    /// The texture identity emitted for ordinary 256-cell glyphs.
    #[must_use]
    pub fn default_texture(&self) -> FontTexture {
        self.default_texture
    }

    /// Whether Java's red/cyan anaglyph palette is selected for formatted glyph colours.
    #[must_use]
    pub const fn anaglyph_enabled(&self) -> bool {
        self.anaglyph
    }

    /// Change the live anaglyph palette without resetting any renderer state.
    ///
    /// The 32 entries are rebuilt from the same Java constructor algorithm used by
    /// [`Self::new_with_texture_and_seed`].  In particular, this does not reload font data,
    /// change the selected sheet, alter unicode/bidi flags, or reseed the obfuscation stream.
    pub fn set_anaglyph(&mut self, enabled: bool) {
        // `sync_anaglyph_with_gpu` calls this every frame, so only an actual change may drop
        // the glyph-run cache (the palette is a pure function of the flag).
        if self.anaglyph != enabled {
            self.runs.invalidate();
            self.layout_generation = next_layout_generation();
        }
        self.anaglyph = enabled;
        self.colors = color_table(enabled);
    }

    /// The baked width tables.
    #[must_use]
    pub fn data(&self) -> &FontData {
        &self.data
    }

    /// Line height, 9 GUI pixels.
    #[must_use]
    pub fn font_height(&self) -> i32 {
        self.font_height
    }

    /// Whether every glyph is taken from the unicode pages instead of `ascii.png`.
    #[must_use]
    pub fn unicode_flag(&self) -> bool {
        self.unicode
    }

    /// Switch the unicode pages on or off.
    ///
    /// Vanilla drives this from `forceUnicodeFont` in `options.txt` **or** from the active
    /// language's own "more than 10% non-Latin-1" heuristic
    /// ([`mc_assets::Language::is_unicode`]).
    // java: FontRenderer.java:822 (setUnicodeFlag)
    pub fn set_unicode_flag(&mut self, unicode: bool) {
        if self.unicode != unicode {
            self.runs.invalidate();
            self.layout_generation = next_layout_generation();
        }
        self.unicode = unicode;
    }

    /// Whether the bidirectional algorithm runs for drawn strings.
    #[must_use]
    pub fn bidi_flag(&self) -> bool {
        self.bidi
    }

    /// Set the bidi flag used by the live render path.
    pub fn set_bidi_flag(&mut self, bidi: bool) {
        if self.bidi != bidi {
            self.runs.invalidate();
            self.layout_generation = next_layout_generation();
        }
        self.bidi = bidi;
    }

    /// Whether symbols cast a visible drop shadow ([`Self::set_symbol_shadows`]).
    #[must_use]
    pub fn symbol_shadows(&self) -> bool {
        self.symbol_shadows
    }

    /// Give *symbols* the drop shadow letters have. Off, the default, is vanilla. The client sets
    /// it from the Chat mod's "Symbol shadows".
    ///
    /// A symbol is any glyph `ascii.png` does not have, drawn from a unicode page: hearts,
    /// stars, swords, check marks, arrows, CJK. java: `renderStringAtPos` moves the shadow pass's
    /// glyph back up and left by `f1` whenever `c0 == 0 || j == -1 || unicodeFlag`
    /// (FontRenderer.java:479-494), with `f1 = unicodeFlag ? 0.5 : 1.0`. With the unicode font
    /// on, every glyph comes from a unicode page and its shadow lands half a pixel down and
    /// right. With it off, a symbol's shadow pass starts one pixel down and right and is moved
    /// back the whole pixel, onto the glyph. The unicode pages are 0/255 alpha, so the glyph
    /// covers its shadow exactly: vanilla's symbols show no shadow. On, a symbol drawn while the
    /// unicode font is off keeps its shadow one pixel down and right, like a letter's (with the
    /// unicode font on nothing changes).
    ///
    /// Only a real change drops the run cache and moves [`Self::layout_generation`] (the client
    /// calls this every frame), so cached labels and HUD text pick the new shadow up.
    pub fn set_symbol_shadows(&mut self, on: bool) {
        if self.symbol_shadows != on {
            self.runs.invalidate();
            self.layout_generation = next_layout_generation();
        }
        self.symbol_shadows = on;
    }

    /// Turn the glyph-run cache behind [`Self::draw_string`] on or off (on by default).
    ///
    /// The cache is exact — a hit replays the very [`GlyphQuad`]s/[`SolidRect`]s the miss
    /// emitted and restores every renderer field the draw left behind — so this switch exists
    /// for A/B measurement and for tests that compare the two paths, not for parity.
    pub fn set_run_cache_enabled(&mut self, enabled: bool) {
        if !enabled {
            self.runs.invalidate();
        }
        self.runs.enabled = enabled;
    }

    /// Hit/miss counters and the current size of the glyph-run cache.
    #[must_use]
    pub fn run_cache_stats(&self) -> RunCacheStats {
        RunCacheStats {
            hits: self.runs.hits,
            misses: self.runs.misses,
            entries: self.runs.runs.len(),
        }
    }

    /// Adopt the direction declared by a resource-pack language descriptor.
    ///
    /// The loaded [`mc_assets::Language`] intentionally stores translation data only; the
    /// bidirectional bit belongs to the descriptor in `pack.mcmeta`, just as it does in
    /// vanilla's `LanguageManager`. Keeping this tiny bridge here gives the client one named
    /// seam to call after startup and after a language/resource-pack reload.
    pub fn set_language_direction(&mut self, language: &LanguageInfo) {
        self.set_bidi_flag(language.bidirectional);
    }

    /// Reorder a string with this renderer's current bidi setting.
    ///
    /// This is public so language/UI code can place a caret or hit-test against the same visual
    /// order that [`Self::draw_string`] emits. With bidi disabled the input is returned exactly,
    /// including formatting markers.
    #[must_use]
    pub fn reorder_text(&self, text: &str) -> String {
        if self.bidi {
            bidi::reorder(text)
        } else {
            text.to_owned()
        }
    }

    /// The colour the last `§`-code selected, or `-1` if none has been seen.
    ///
    /// Vanilla keeps this as a field and `drawSplitString` writes the caller's colour into it;
    /// S16's chat code reads it back to decide the colour of the "…" continuation marker.
    // java: FontRenderer.java:82 (textColor)
    #[must_use]
    pub fn text_color(&self) -> i32 {
        self.text_color
    }

    /// The rendered RGB for one of `0`–`9`, `a`–`f`.
    ///
    /// # Panics
    /// If `c` is not a colour code character.
    // java: FontRenderer.java:992 (getColorCode)
    #[must_use]
    pub fn color_code(&self, c: char) -> i32 {
        let idx = "0123456789abcdef".find(c).expect("not a colour code");
        self.colors[idx]
    }

    /// The 32-entry palette (16 colours + 16 shadow colours).
    #[must_use]
    pub fn colors(&self) -> &[i32; 32] {
        &self.colors
    }

    // ---------------------------------------------------------------- measurement

    /// Width of one UTF-16 code unit, in GUI pixels.
    ///
    /// `§` measures **−1**, which is how the string walkers detect a formatting code; `' '`
    /// measures 4 whatever the tables say.
    // java: FontRenderer.java:658 (getCharWidth)
    #[must_use]
    pub fn unit_width(&self, unit: u16) -> i32 {
        if unit == 167 {
            return -1;
        }
        if unit == 32 {
            return 4;
        }
        let i = font_char_index(unit);
        if unit > 0 && i != -1 && !self.unicode {
            self.data.char_widths()[i as usize]
        } else if self.data.glyph_widths()[unit as usize] != 0 {
            let (mut start, mut end) = self.data.unicode_nibbles(unit);
            if end > 7 {
                end = 15;
                start = 0;
            }
            end += 1;
            (end - start) / 2 + 1
        } else {
            0
        }
    }

    /// Width of one `char`.
    ///
    /// A supplementary (non-BMP) character is measured the way vanilla would see it: as its two
    /// surrogate code units, whose widths are summed.
    #[must_use]
    pub fn char_width(&self, c: char) -> i32 {
        let mut buf = [0u16; 2];
        c.encode_utf16(&mut buf)
            .iter()
            .map(|&u| self.unit_width(u))
            .sum()
    }

    /// Width of a whole string, honouring `§l` (bold adds one pixel per glyph).
    ///
    /// Note the asymmetry vanilla has here and nowhere else: after `§`, only `l`/`L` and
    /// `r`/`R` change the bold flag. A **colour** code does *not* clear bold in this routine,
    /// though it does in [`size_string_to_width`](Self::size_string_to_width).
    // java: FontRenderer.java:607 (getStringWidth)
    #[must_use]
    pub fn string_width(&self, text: &str) -> i32 {
        self.width_of_units(text.encode_utf16())
    }

    /// [`string_width`](Self::string_width) of the concatenation of `parts`, without building it.
    ///
    /// The walk only sees UTF-16 units, and the encoding of a concatenation is the
    /// concatenation of the encodings, so this equals `string_width(&parts.concat())` exactly —
    /// including a `§` that ends one part and consumes the first unit of the next.
    #[must_use]
    pub fn string_width_of_parts(&self, parts: &[&str]) -> i32 {
        self.width_of_units(parts.iter().flat_map(|part| part.encode_utf16()))
    }

    /// Same walk as `string_width_units`, streamed so measuring allocates nothing (C2-8: the
    /// sidebar alone measured 3–4 strings per row per frame). Pinned against the slice walk by
    /// `streamed_string_width_matches_the_unit_walk`.
    fn width_of_units(&self, mut units: impl Iterator<Item = u16>) -> i32 {
        let mut i = 0i32;
        let mut bold = false;
        while let Some(c0) = units.next() {
            let mut k = self.unit_width(c0);
            // java: `if (k < 0 && j < text.length() - 1)` — a trailing `§` keeps its -1.
            if k < 0 {
                if let Some(c1) = units.next() {
                    if c1 != 108 && c1 != 76 {
                        if c1 == 114 || c1 == 82 {
                            bold = false;
                        }
                    } else {
                        bold = true;
                    }
                    k = 0;
                }
            }

            i += k;

            if bold && k > 0 {
                i += 1;
            }
        }
        i
    }

    fn string_width_units(&self, t: &[u16]) -> i32 {
        let mut i = 0i32;
        let mut bold = false;
        let mut j = 0usize;
        while j < t.len() {
            let mut c0 = t[j];
            let mut k = self.unit_width(c0);

            if k < 0 && j < t.len() - 1 {
                j += 1;
                c0 = t[j];
                if c0 != 108 && c0 != 76 {
                    if c0 == 114 || c0 == 82 {
                        bold = false;
                    }
                } else {
                    bold = true;
                }
                k = 0;
            }

            i += k;

            if bold && k > 0 {
                i += 1;
            }
            j += 1;
        }
        i
    }

    /// Longest prefix (or suffix, when `reverse`) of `text` that fits in `width` pixels.
    // java: FontRenderer.java:708 (trimStringToWidth)
    #[must_use]
    pub fn trim_to_width(&self, text: &str, width: i32) -> String {
        self.trim_to_width_dir(text, width, false)
    }

    /// [`trim_to_width`](Self::trim_to_width), optionally from the right-hand end.
    #[must_use]
    pub fn trim_to_width_dir(&self, text: &str, width: i32, reverse: bool) -> String {
        let t = utf16(text);
        let mut out: Vec<u16> = Vec::new();
        let mut i = 0i32;
        let start = if reverse { t.len() as i32 - 1 } else { 0 };
        let step = if reverse { -1 } else { 1 };
        let mut after_section = false;
        let mut bold = false;
        let mut l = start;

        while l >= 0 && (l as usize) < t.len() && i < width {
            let c0 = t[l as usize];
            let i1 = self.unit_width(c0);

            if after_section {
                after_section = false;
                if c0 != 108 && c0 != 76 {
                    if c0 == 114 || c0 == 82 {
                        bold = false;
                    }
                } else {
                    bold = true;
                }
            } else if i1 < 0 {
                after_section = true;
            } else {
                i += i1;
                if bold {
                    i += 1;
                }
            }

            if i > width {
                break;
            }

            if reverse {
                out.insert(0, c0);
            } else {
                out.push(c0);
            }
            l += step;
        }

        String::from_utf16_lossy(&out)
    }

    /// How many UTF-16 code units of `text` fit on one line of `wrap_width` pixels.
    ///
    /// Returns the index to cut at: the last space if the line has one and the cut would land
    /// mid-word, otherwise the index of the first character that did not fit. A `\n` terminates
    /// the line and its own index is returned.
    // java: FontRenderer.java:873 (sizeStringToWidth)
    #[must_use]
    pub fn size_string_to_width(&self, text: &str, wrap_width: i32) -> usize {
        self.size_units_to_width(&utf16(text), wrap_width)
    }

    fn size_units_to_width(&self, t: &[u16], wrap_width: i32) -> usize {
        let i = t.len();
        let mut j = 0i32; // accumulated pixels
        let mut k = 0usize; // cursor
        let mut l: i64 = -1; // index of the last space
        let mut bold = false;

        while k < i {
            let c0 = t[k];
            match c0 {
                10 => {
                    // java: `case '\n': --k; break;` — undone by the `if (c0 == 10)` below.
                    k = k.wrapping_sub(1);
                }
                0xA7 => {
                    if k < i - 1 {
                        k += 1;
                        let c1 = t[k];
                        if c1 != 108 && c1 != 76 {
                            let ch = char::from_u32(u32::from(c1)).unwrap_or('\u{fffd}');
                            if c1 == 114 || c1 == 82 || is_format_color(ch) {
                                bold = false;
                            }
                        } else {
                            bold = true;
                        }
                    }
                }
                _ => {
                    // java falls through from `case ' '` into `default`.
                    if c0 == 32 {
                        l = k as i64;
                    }
                    j += self.unit_width(c0);
                    if bold {
                        j += 1;
                    }
                }
            }

            if c0 == 10 {
                k = k.wrapping_add(1);
                l = k as i64;
                break;
            }

            if j > wrap_width {
                break;
            }

            k += 1;
        }

        if k != i && l != -1 && l < k as i64 {
            l as usize
        } else {
            k
        }
    }

    /// Insert `\n` (and repeat the active formatting prefix) so `text` fits `wrap_width`.
    // java: FontRenderer.java:852 (wrapFormattedStringToWidth)
    #[must_use]
    pub fn wrap_to_width(&self, text: &str, wrap_width: i32) -> String {
        let mut rest = utf16(text);
        let mut out: Vec<u16> = Vec::new();
        loop {
            let i = self.size_units_to_width(&rest, wrap_width);
            if rest.len() <= i {
                out.extend_from_slice(&rest);
                return String::from_utf16_lossy(&out);
            }
            let c0 = rest[i];
            let skip = usize::from(c0 == 32 || c0 == 10);

            // Cutting at 0 without consuming a separator leaves the remainder unchanged, so
            // vanilla recurses forever (a `StackOverflowError`) whenever one glyph is wider
            // than `wrapWidth`. Server-sent chat can trigger that, so emit one unit as its own
            // line and move on. Every non-degenerate width takes the normal path below and
            // behaves exactly as vanilla does.
            if i == 0 && skip == 0 {
                out.push(rest[0]);
                out.push(u16::from(b'\n'));
                rest = rest[1..].to_vec();
                continue;
            }

            let head = &rest[..i];
            let prefix = utf16(&format_from_string(&String::from_utf16_lossy(head)));
            let mut next = prefix;
            next.extend_from_slice(&rest[i + skip..]);
            out.extend_from_slice(head);
            out.push(u16::from(b'\n'));
            rest = next;
        }
    }

    /// [`wrap_to_width`](Self::wrap_to_width), split into lines.
    ///
    /// Splitting follows Java's `String.split("\n")`: **trailing** empty lines are dropped, so
    /// `"hello\n"` yields one line and `"\n"` yields none at all.
    // java: FontRenderer.java:844 (listFormattedStringToWidth)
    #[must_use]
    pub fn wrap_to_lines(&self, text: &str, wrap_width: i32) -> Vec<String> {
        java_split_newline(&self.wrap_to_width(text, wrap_width))
    }

    /// Pixel height of `text` once wrapped: `FONT_HEIGHT * line_count`.
    // java: FontRenderer.java:813 (splitStringWidth)
    #[must_use]
    pub fn split_string_height(&self, text: &str, wrap_width: i32) -> i32 {
        self.font_height * self.wrap_to_lines(text, wrap_width).len() as i32
    }

    // ---------------------------------------------------------------- drawing

    /// Draw `text` at `(x, y)` with a one-pixel drop shadow. Returns the x the cursor ended at.
    // java: FontRenderer.java:325 (drawStringWithShadow)
    pub fn draw_string_with_shadow(
        &mut self,
        sink: &mut dyn TextSink,
        text: &str,
        x: f32,
        y: f32,
        color: i32,
    ) -> i32 {
        self.draw_string(sink, text, x, y, color, true)
    }

    /// Draw `text` at `(x, y)`. Returns the x the cursor ended at.
    ///
    /// With `drop_shadow`, vanilla draws the string **twice** — offset by `(1, 1)` in the
    /// darkened colour first, then in place — and calls `resetStyles` only *once*, before both
    /// passes. The second pass therefore inherits whatever style flags the first pass left set,
    /// so `"Text§l"` renders its shadow un-bold and its foreground bold. That is reproduced.
    ///
    /// Text whose alpha vanilla's alpha test discards (an alpha byte of 4..=25) reaches the sink
    /// as nothing at all; see [`alpha_test_discards`].
    // java: FontRenderer.java:341 (drawString)
    pub fn draw_string(
        &mut self,
        sink: &mut dyn TextSink,
        text: &str,
        x: f32,
        y: f32,
        color: i32,
        drop_shadow: bool,
    ) -> i32 {
        // java: `drawString` begins with `GlStateManager.enableAlpha()`.
        // The layout below still runs in full (the advance, the end state and the `§k` stream
        // are vanilla's); only the sink is skipped, as the GPU would discard every fragment.
        let mut discarded = NullSink;
        let sink: &mut dyn TextSink = if alpha_test_discards(color) {
            &mut discarded
        } else {
            sink
        };
        self.reset_styles();
        if !self.runs.enabled || text.len() > RUN_CACHE_MAX_TEXT_BYTES {
            return self.draw_string_passes(sink, text, x, y, color, drop_shadow);
        }

        // Glyph-run cache (#8 / C2-1). The output of a draw is a pure function of the key below
        // plus the renderer configuration (sheet widths, unicode/bidi flags, palette), and every
        // configuration change invalidates the cache. The key uses the *absolute* x/y bits, so
        // a hit replays bit-identical quads; `§k` draws are never stored (see `rng_used`).
        let hash = run_key_hash(text, x, y, color, drop_shadow);
        self.runs.clock += 1;
        let clock = self.runs.clock;
        if let Some(run) = self.runs.runs.get_mut(&hash) {
            if run.matches(text, x, y, color, drop_shadow) {
                run.last_used = clock;
                for op in run.ops.iter() {
                    match *op {
                        TextOp::Glyph(quad) => sink.glyph(quad),
                        TextOp::Bar(rect) => sink.bar(rect),
                        TextOp::Shadow(shadow) => sink.shadow_pass(shadow),
                    }
                }
                let (advance, end) = (run.advance, run.end);
                self.runs.hits += 1;
                self.restore_run_end(end);
                return advance;
            }
        }
        self.runs.misses += 1;

        // Admit a run on its second sighting only: strings that change every frame (F3
        // coordinates, timers) then cost one hash-map insert instead of an entry allocation
        // that would be evicted unused.
        if !self.runs.note_seen(hash, clock) {
            return self.draw_string_passes(sink, text, x, y, color, drop_shadow);
        }
        let mut ops = std::mem::take(&mut self.runs.scratch);
        ops.clear();
        self.text_color_written = false;
        self.rng_used = false;
        let advance = {
            let mut recorder = RecordingSink {
                inner: sink,
                ops: &mut ops,
            };
            self.draw_string_passes(&mut recorder, text, x, y, color, drop_shadow)
        };
        if !self.rng_used {
            let end = self.capture_run_end();
            self.runs.store(
                hash,
                CachedRun {
                    text: text.into(),
                    x: x.to_bits(),
                    y: y.to_bits(),
                    color,
                    shadow: drop_shadow,
                    ops: ops.as_slice().into(),
                    advance,
                    end,
                    last_used: clock,
                },
            );
        }
        self.runs.scratch = ops;
        advance
    }

    /// The uncached body of [`Self::draw_string`]: both vanilla passes over one reused UTF-16
    /// buffer. Styles must already have been reset.
    fn draw_string_passes(
        &mut self,
        sink: &mut dyn TextSink,
        text: &str,
        x: f32,
        y: f32,
        color: i32,
        drop_shadow: bool,
    ) -> i32 {
        let mut units = std::mem::take(&mut self.units_scratch);
        units.clear();
        units.extend(text.encode_utf16());
        let advance = if drop_shadow {
            let i = self.render_string(sink, &units, x + 1.0, y + 1.0, color, true);
            i.max(self.render_string(sink, &units, x, y, color, false))
        } else {
            self.render_string(sink, &units, x, y, color, false)
        };
        self.units_scratch = units;
        advance
    }

    /// Every renderer field a draw leaves behind, for replay on a cache hit.
    fn capture_run_end(&self) -> RunEndState {
        RunEndState {
            pos_x: self.pos_x,
            pos_y: self.pos_y,
            base_color: self.base_color,
            color: self.color,
            text_color: self.text_color_written.then_some(self.text_color),
            random_style: self.random_style,
            bold_style: self.bold_style,
            italic_style: self.italic_style,
            underline_style: self.underline_style,
            strikethrough_style: self.strikethrough_style,
        }
    }

    fn restore_run_end(&mut self, end: RunEndState) {
        self.pos_x = end.pos_x;
        self.pos_y = end.pos_y;
        self.base_color = end.base_color;
        self.color = end.color;
        if let Some(text_color) = end.text_color {
            self.text_color = text_color;
        }
        self.random_style = end.random_style;
        self.bold_style = end.bold_style;
        self.italic_style = end.italic_style;
        self.underline_style = end.underline_style;
        self.strikethrough_style = end.strikethrough_style;
    }

    /// Draw `text` word-wrapped to `wrap_width`, one [`FONT_HEIGHT`] per line, no shadow.
    // java: FontRenderer.java:786 (drawSplitString)
    pub fn draw_split_string(
        &mut self,
        sink: &mut dyn TextSink,
        text: &str,
        x: i32,
        mut y: i32,
        wrap_width: i32,
        color: i32,
    ) {
        self.reset_styles();
        self.text_color = color;
        let trimmed = trim_trailing_newlines(text);
        for line in self.wrap_to_lines(trimmed, wrap_width) {
            let units = utf16(&line);
            // `renderStringAtPos` writes `textColor` when it consumes a colour code, and
            // vanilla feeds that field back into the next wrapped line. This matters when a
            // continuation begins with the repeated colour prefix and later consumes `§r`:
            // reset restores the colour carried from the preceding line, not the original
            // caller colour.
            let line_color = self.text_color;
            self.render_string_aligned(sink, &units, x, y, wrap_width, line_color, false);
            y += self.font_height;
        }
    }

    /// Clear the five style flags. Called at the start of a draw, never between shadow passes.
    // java: FontRenderer.java:380 (resetStyles)
    fn reset_styles(&mut self) {
        self.random_style = false;
        self.bold_style = false;
        self.italic_style = false;
        self.underline_style = false;
        self.strikethrough_style = false;
    }

    // java: FontRenderer.java:555 (renderStringAligned) — one argument per Java parameter.
    #[allow(clippy::too_many_arguments)]
    fn render_string_aligned(
        &mut self,
        sink: &mut dyn TextSink,
        text: &[u16],
        mut x: i32,
        y: i32,
        width: i32,
        color: i32,
        drop_shadow: bool,
    ) -> i32 {
        if self.bidi {
            // Java measures the visual string after bidiReorder before applying right alignment.
            let reordered = bidi::reorder_units(text);
            x = x + width - self.string_width_units(&reordered);
        }
        self.render_string(sink, text, x as f32, y as f32, color, drop_shadow)
    }

    // java: FontRenderer.java:569 (renderString)
    fn render_string(
        &mut self,
        sink: &mut dyn TextSink,
        text: &[u16],
        x: f32,
        y: f32,
        color: i32,
        drop_shadow: bool,
    ) -> i32 {
        sink.shadow_pass(drop_shadow);
        let mut color = color;
        // "If the top six bits of alpha are clear, force opaque." Vanilla's mask is
        // 0xFC000000, not 0xFF000000, so alpha 1..3 also counts as "unset".
        if (color & -67_108_864) == 0 {
            color |= -16_777_216;
        }
        if drop_shadow {
            color = (color & 16_579_836) >> 2 | color & -16_777_216;
        }
        self.base_color = color as u32;
        self.color = color as u32;
        self.pos_x = x;
        self.pos_y = y;
        if self.bidi {
            let reordered = bidi::reorder_units(text);
            self.render_string_at_pos(sink, &reordered, drop_shadow);
        } else {
            self.render_string_at_pos(sink, text, drop_shadow);
        }
        self.pos_x as i32
    }

    // java: FontRenderer.java:392 (renderStringAtPos)
    fn render_string_at_pos(&mut self, sink: &mut dyn TextSink, text: &[u16], shadow: bool) {
        // java: FontRenderer.renderStringAtPos calls `text.toLowerCase().charAt(i + 1)` for
        // every formatting marker. Lower-casing the whole string is observably different from
        // lower-casing only the code unit after `§`: Unicode case conversion can expand an
        // earlier character and thereby shift the lookup index.
        // Most nameplate strings contain no formatting marker. Delay the exact same whole-string
        // conversion until the first marker, then reuse it for every marker in this draw.
        //
        // C2-8: when the string has no U+0130 and no surrogate, whole-string lower-casing maps
        // every UTF-16 unit to exactly one unit (U+0130 is the only BMP character whose
        // lowercase expands), so no index can shift and `lowercase[i + 1]` is decided by the
        // unit itself. That covers practically every server string and skips the three
        // allocations; anything else takes the whole-string path below, unchanged.
        let mut per_unit_lowercase: Option<bool> = None;
        let mut lowercase: Option<Vec<u16>> = None;
        let mut i = 0usize;
        while i < text.len() {
            let mut c0 = text[i];

            if c0 == 167 && i + 1 < text.len() {
                let per_unit =
                    *per_unit_lowercase.get_or_insert_with(|| lowercase_is_per_unit(text));
                let mut i1 = if per_unit {
                    marker_code_index(text[i + 1])
                } else {
                    let lowercase = lowercase.get_or_insert_with(|| {
                        String::from_utf16_lossy(text)
                            .to_lowercase()
                            .encode_utf16()
                            .collect::<Vec<_>>()
                    });
                    format_code_index(lowercase[i + 1])
                };

                if i1 < 16 {
                    self.random_style = false;
                    self.bold_style = false;
                    self.strikethrough_style = false;
                    self.underline_style = false;
                    self.italic_style = false;

                    if !(0..=15).contains(&i1) {
                        i1 = 15;
                    }
                    if shadow {
                        i1 += 16;
                    }
                    let j1 = self.colors[i1 as usize];
                    self.text_color = j1;
                    self.text_color_written = true;
                    self.color = (self.base_color & 0xFF00_0000) | (j1 as u32 & 0x00FF_FFFF);
                } else if i1 == 16 {
                    self.random_style = true;
                } else if i1 == 17 {
                    self.bold_style = true;
                } else if i1 == 18 {
                    self.strikethrough_style = true;
                } else if i1 == 19 {
                    self.underline_style = true;
                } else if i1 == 20 {
                    self.italic_style = true;
                } else if i1 == 21 {
                    self.random_style = false;
                    self.bold_style = false;
                    self.strikethrough_style = false;
                    self.underline_style = false;
                    self.italic_style = false;
                    self.color = self.base_color;
                }

                i += 1;
            } else {
                let mut j = font_char_index(c0);

                if self.random_style && j != -1 {
                    self.rng_used = true;
                    let k = self.unit_width(c0);
                    loop {
                        j = self.rng.next_i32_bound(256);
                        let c1 = font_char_at(j as usize);
                        if k == self.unit_width(c1) {
                            c0 = c1;
                            break;
                        }
                    }
                }

                let f1 = if self.unicode { 0.5f32 } else { 1.0f32 };
                // The shadow pass of a unicode-page glyph (or of a glyph the sheet does not
                // have) is moved back up and left by `f1`: with the unicode font on, its shadow
                // ends half a pixel off the glyph; with it off, exactly under it (hidden).
                let mut nudged = (c0 == 0 || j == -1 || self.unicode) && shadow;
                // With symbol shadows on, a symbol drawn while the
                // unicode font is off stays where its shadow pass put it, one pixel down and
                // right, like a letter's shadow. Off (vanilla) leaves `nudged` alone.
                if self.symbol_shadows && j == -1 && !self.unicode {
                    nudged = false;
                }

                if nudged {
                    self.pos_x -= f1;
                    self.pos_y -= f1;
                }

                let mut f = self.render_char(sink, c0, self.italic_style);

                if nudged {
                    self.pos_x += f1;
                    self.pos_y += f1;
                }

                if self.bold_style {
                    self.pos_x += f1;
                    if nudged {
                        self.pos_x -= f1;
                        self.pos_y -= f1;
                    }
                    self.render_char(sink, c0, self.italic_style);
                    self.pos_x -= f1;
                    if nudged {
                        self.pos_x += f1;
                        self.pos_y += f1;
                    }
                    f += 1.0;
                }

                if self.strikethrough_style {
                    let mid = self.pos_y + (self.font_height / 2) as f32;
                    sink.bar(SolidRect {
                        x0: self.pos_x,
                        y0: mid - 1.0,
                        x1: self.pos_x + f,
                        y1: mid,
                        color: self.color,
                    });
                }

                if self.underline_style {
                    let base = self.pos_y + self.font_height as f32;
                    sink.bar(SolidRect {
                        x0: self.pos_x - 1.0,
                        y0: base - 1.0,
                        x1: self.pos_x + f,
                        y1: base,
                        color: self.color,
                    });
                }

                // Truncation, not rounding: a unicode glyph that advances 8.5 moves the cursor
                // 8, even though getStringWidth counted 9 for it.
                self.pos_x += (f as i32) as f32;
            }

            i += 1;
        }
    }

    // java: FontRenderer.java:232 (renderChar)
    fn render_char(&mut self, sink: &mut dyn TextSink, ch: u16, italic: bool) -> f32 {
        if ch == 32 {
            return 4.0;
        }
        let i = font_char_index(ch);
        if i != -1 && !self.unicode {
            self.render_default_char(sink, i, italic)
        } else {
            self.render_unicode_char(sink, ch, italic)
        }
    }

    // java: FontRenderer.java:248 (renderDefaultChar)
    fn render_default_char(&mut self, sink: &mut dyn TextSink, ch: i32, italic: bool) -> f32 {
        let i = (ch % 16 * 8) as f32;
        let j = (ch / 16 * 8) as f32;
        let k = if italic { 1.0f32 } else { 0.0 };
        let l = self.data.char_widths()[ch as usize];
        let f = l as f32 - 0.01;
        let (x, y) = (self.pos_x, self.pos_y);
        // The top and left texture edges start ASCII_UV_INSET into the cell
        // (vanilla: exactly on it); positions are vanilla's. See `ASCII_UV_INSET`.
        let (u0, v0) = ((i + ASCII_UV_INSET) / 128.0, (j + ASCII_UV_INSET) / 128.0);
        sink.glyph(GlyphQuad {
            texture: self.default_texture,
            pos: [
                [x + k, y],
                [x - k, y + 7.99],
                [x + f - 1.0 + k, y],
                [x + f - 1.0 - k, y + 7.99],
            ],
            uv: [
                [u0, v0],
                [u0, (j + 7.99) / 128.0],
                [(i + f - 1.0) / 128.0, v0],
                [(i + f - 1.0) / 128.0, (j + 7.99) / 128.0],
            ],
            color: self.color,
        });
        l as f32
    }

    // java: FontRenderer.java:290 (renderUnicodeChar)
    fn render_unicode_char(&mut self, sink: &mut dyn TextSink, ch: u16, italic: bool) -> f32 {
        if self.data.glyph_widths()[ch as usize] == 0 {
            return 0.0;
        }
        let page = (ch / 256) as u8;
        let (start, end) = self.data.unicode_nibbles(ch);
        let f = start as f32;
        let f1 = (end + 1) as f32;
        let f2 = (u32::from(ch % 16) * 16) as f32 + f;
        let f3 = ((u32::from(ch & 255) / 16) * 16) as f32;
        let f4 = f1 - f - 0.02;
        let f5 = if italic { 1.0f32 } else { 0.0 };
        let (x, y) = (self.pos_x, self.pos_y);
        // As in `render_default_char`, in this sheet's texels.
        let (u0, v0) = (
            (f2 + UNICODE_UV_INSET) / 256.0,
            (f3 + UNICODE_UV_INSET) / 256.0,
        );
        sink.glyph(GlyphQuad {
            texture: FontTexture::UnicodePage(page),
            pos: [
                [x + f5, y],
                [x - f5, y + 7.99],
                [x + f4 / 2.0 + f5, y],
                [x + f4 / 2.0 - f5, y + 7.99],
            ],
            uv: [
                [u0, v0],
                [u0, (f3 + 15.98) / 256.0],
                [(f2 + f4) / 256.0, v0],
                [(f2 + f4) / 256.0, (f3 + 15.98) / 256.0],
            ],
            color: self.color,
        });
        (f1 - f) / 2.0 + 1.0
    }
}

/// Whether vanilla's alpha test hides every fragment of a [`FontRenderer::draw_string`] in
/// `color` (a `FontRenderer.java` parity check).
///
/// java: `drawString` enables `GL_ALPHA_TEST` (FontRenderer.java:343) and the GUI draws under
/// `alphaFunc(GL_GREATER, 0.1F)` (EntityRenderer.java:1168). A glyph fragment's alpha is the
/// texel's (0 or 1 in vanilla's sheets) times the colour's, a bar's is the colour's, and both
/// passes share the alpha, so text whose alpha byte is 25 or less (25/255 < 0.1 < 26/255) shows
/// nothing at all. An alpha byte of 0..=3 is never hidden: `renderString` makes it opaque first.
/// Vanilla's callers reach it at the end of every fade: a chat line's last ~0.3 s, a title, the
/// action bar, the held item's name, and a chat at 0% text opacity (`0.1 * 255 = 25`).
#[must_use]
pub fn alpha_test_discards(color: i32) -> bool {
    (4..=25).contains(&((color as u32) >> 24))
}

/// UTF-16 code units of a string.
fn utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

/// `FONT_CHARS.charAt(i)` as a code unit. Every character in the sheet is BMP.
///
/// A memoized table: the `§k` re-roll loop calls this until a width matches, and walking the
/// string with `chars().nth(i)` made each retry O(i) (C2-15).
fn font_char_at(i: usize) -> u16 {
    static TABLE: std::sync::OnceLock<Vec<u16>> = std::sync::OnceLock::new();
    TABLE
        .get_or_init(|| FONT_CHARS.chars().map(|c| c as u16).collect())
        .get(i)
        .copied()
        .unwrap_or(0)
}

/// Whether lower-casing `text` as a whole maps each UTF-16 unit to exactly one unit at the same
/// index, so `toLowerCase().charAt(i)` can be decided from `text[i]` alone.
///
/// Among BMP characters only U+0130 (`İ` → `i̇`) has a multi-character lowercase mapping;
/// surrogates are excluded outright rather than reasoned about. Pinned exhaustively by
/// `per_unit_marker_lookup_matches_whole_string_lowercase_for_every_bmp_unit`.
fn lowercase_is_per_unit(text: &[u16]) -> bool {
    text.iter()
        .all(|&unit| unit != 0x0130 && !(0xD800..=0xDFFF).contains(&unit))
}

/// `format_code_index` of the lowercase of one unit, valid when [`lowercase_is_per_unit`]
/// holds for the whole string.
///
/// Only the 22 ASCII code letters matter, so the context-dependent final-sigma rule (Σ → ς or
/// σ, neither a code) cannot change the answer. The one non-ASCII unit that lowercases into
/// the code set is U+212A KELVIN SIGN → `k`, which `char::to_lowercase` handles.
fn marker_code_index(unit: u16) -> i32 {
    if unit < 0x80 {
        return format_code_index(u16::from((unit as u8).to_ascii_lowercase()));
    }
    let Some(c) = char::from_u32(u32::from(unit)) else {
        return -1;
    };
    let mut lower = c.to_lowercase();
    match (lower.next(), lower.next()) {
        (Some(l), None) if (l as u32) < 0x80 => format_code_index(l as u16),
        _ => -1,
    }
}

// ---------------------------------------------------------------- glyph-run cache

/// Longest string (in UTF-8 bytes) the glyph-run cache stores. Longer text (books, long wrapped
/// lines) is drawn uncached so one entry can never pin a large allocation.
const RUN_CACHE_MAX_TEXT_BYTES: usize = 256;
/// Entry count at which the cache sweeps out runs not drawn in the last [`RUN_CACHE_WINDOW`]
/// draws. A sweep leaves at most `RUN_CACHE_WINDOW` entries, because every draw touches one.
const RUN_CACHE_CAPACITY: usize = 1024;
/// Recency window, in `draw_string` calls, that survives a sweep: a few frames of a busy HUD
/// (sidebar + chat + Tab + nameplates is ~150–350 draws per frame).
const RUN_CACHE_WINDOW: u64 = 768;

/// Glyph-run cache counters, see [`FontRenderer::run_cache_stats`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunCacheStats {
    /// Draws served from the cache.
    pub hits: u64,
    /// Draws laid out from scratch (including uncacheable `§k` draws).
    pub misses: u64,
    /// Runs currently stored.
    pub entries: usize,
}

/// One primitive a draw emitted, in emission order (glyphs and bars interleave).
#[derive(Debug, Clone, Copy, PartialEq)]
enum TextOp {
    Shadow(bool),
    Glyph(GlyphQuad),
    Bar(SolidRect),
}

/// The renderer fields a [`FontRenderer::draw_string`] leaves behind.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RunEndState {
    pos_x: f32,
    pos_y: f32,
    base_color: u32,
    color: u32,
    /// `Some` when the draw consumed a colour code, i.e. wrote `textColor`.
    text_color: Option<i32>,
    random_style: bool,
    bold_style: bool,
    italic_style: bool,
    underline_style: bool,
    strikethrough_style: bool,
}

/// One cached `draw_string`: its full key (hash collisions are verified, never trusted), the
/// emitted primitives, the returned advance and the final renderer state.
#[derive(Debug)]
struct CachedRun {
    text: Box<str>,
    x: u32,
    y: u32,
    color: i32,
    shadow: bool,
    ops: Box<[TextOp]>,
    advance: i32,
    end: RunEndState,
    last_used: u64,
}

impl CachedRun {
    fn matches(&self, text: &str, x: f32, y: f32, color: i32, shadow: bool) -> bool {
        self.x == x.to_bits()
            && self.y == y.to_bits()
            && self.color == color
            && self.shadow == shadow
            && *self.text == *text
    }
}

/// Keyed by [`run_key_hash`], which is a fixed function of server-chosen text and its draw
/// position. The table still hashes that key with the per-process seeded [`mc_core::FastHasher`]:
/// passed through unchanged, its bucket and control bits would be predictable, and a server
/// could pre-search chat, scoreboard or Tab text whose keys all probe the same group.
type RunMap<V> = mc_core::FastMap<u64, V>;

#[derive(Debug)]
struct RunCache {
    enabled: bool,
    /// Monotonic `draw_string` counter; the recency stamp for eviction.
    clock: u64,
    runs: RunMap<CachedRun>,
    /// Key hashes seen once and not stored yet, with the clock of that sighting.
    seen: RunMap<u64>,
    /// Reused recording buffer for the miss path.
    scratch: Vec<TextOp>,
    hits: u64,
    misses: u64,
}

impl RunCache {
    fn new() -> Self {
        Self {
            enabled: true,
            clock: 0,
            runs: RunMap::default(),
            seen: RunMap::default(),
            scratch: Vec::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Drop every stored run (configuration changed). Counters survive.
    fn invalidate(&mut self) {
        self.runs.clear();
        self.seen.clear();
    }

    /// Record a sighting of `hash`; true when it had already been seen (admit it now).
    fn note_seen(&mut self, hash: u64, clock: u64) -> bool {
        if self.seen.remove(&hash).is_some() {
            return true;
        }
        if self.seen.len() >= RUN_CACHE_CAPACITY {
            let floor = clock.saturating_sub(RUN_CACHE_WINDOW);
            self.seen.retain(|_, last| *last > floor);
        }
        self.seen.insert(hash, clock);
        false
    }

    fn store(&mut self, hash: u64, run: CachedRun) {
        if self.runs.len() >= RUN_CACHE_CAPACITY {
            let floor = run.last_used.saturating_sub(RUN_CACHE_WINDOW);
            self.runs.retain(|_, stored| stored.last_used > floor);
        }
        self.runs.insert(hash, run);
    }
}

/// Mix a draw's key into 64 bits: Fx-style word folding plus a murmur3 finalizer. Equality is
/// re-checked on every hit ([`CachedRun::matches`]), so quality only affects the hit rate.
fn run_key_hash(text: &str, x: f32, y: f32, color: i32, shadow: bool) -> u64 {
    const K: u64 = 0x517c_c1b7_2722_0a95;
    fn add(h: u64, word: u64) -> u64 {
        (h.rotate_left(5) ^ word).wrapping_mul(K)
    }
    let bytes = text.as_bytes();
    let mut h = 0u64;
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let mut word = [0u8; 8];
        word.copy_from_slice(chunk);
        h = add(h, u64::from_le_bytes(word));
    }
    let rest = chunks.remainder();
    if !rest.is_empty() {
        let mut word = [0u8; 8];
        word[..rest.len()].copy_from_slice(rest);
        h = add(h, u64::from_le_bytes(word));
    }
    h = add(h, bytes.len() as u64);
    h = add(h, u64::from(x.to_bits()) | (u64::from(y.to_bits()) << 32));
    h = add(h, u64::from(color as u32) | (u64::from(shadow) << 32));
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h
}

/// Forwards every primitive to the real sink while keeping a copy for the cache.
struct RecordingSink<'a, S: TextSink + ?Sized> {
    inner: &'a mut S,
    ops: &'a mut Vec<TextOp>,
}

impl<S: TextSink + ?Sized> TextSink for RecordingSink<'_, S> {
    fn shadow_pass(&mut self, shadow: bool) {
        self.ops.push(TextOp::Shadow(shadow));
        self.inner.shadow_pass(shadow);
    }
    fn glyph(&mut self, quad: GlyphQuad) {
        self.ops.push(TextOp::Glyph(quad));
        self.inner.glyph(quad);
    }
    fn bar(&mut self, rect: SolidRect) {
        self.ops.push(TextOp::Bar(rect));
        self.inner.bar(rect);
    }
}

/// `"0123456789abcdefklmnor".indexOf(c)`.
fn format_code_index(unit: u16) -> i32 {
    const CODES: &[u8; 22] = b"0123456789abcdefklmnor";
    if unit > 0x7F {
        return -1;
    }
    CODES
        .iter()
        .position(|&c| u16::from(c) == unit)
        .map_or(-1, |i| i as i32)
}

/// Strip trailing `\n`s, the way `trimStringNewline` does.
// java: FontRenderer.java:773
fn trim_trailing_newlines(text: &str) -> &str {
    text.trim_end_matches('\n')
}

/// Java's `String.split("\n")` with the default limit.
///
/// Trailing empty strings are removed; if the string contains no `\n` at all the whole string
/// comes back as the single element (even when it is empty).
#[must_use]
pub fn java_split_newline(s: &str) -> Vec<String> {
    if !s.contains('\n') {
        return vec![s.to_owned()];
    }
    let mut parts: Vec<String> = s.split('\n').map(str::to_owned).collect();
    while parts.last().is_some_and(String::is_empty) {
        parts.pop();
    }
    parts
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod rng_oracles {
    use super::*;

    fn seeded_font(seed: i64) -> FontRenderer {
        let mut image = image::RgbaImage::new(128, 128);
        for cell in 0..256u32 {
            let ink_columns = if cell % 2 == 0 { 3 } else { 4 };
            let x0 = (cell % 16) * 8;
            let y0 = (cell / 16) * 8;
            for x in 0..ink_columns {
                image.put_pixel(x0 + x, y0 + 2, image::Rgba([255, 255, 255, 255]));
            }
        }
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut png, image::ImageFormat::Png)
            .expect("encode synthetic font");
        let data = FontData::from_parts(&png.into_inner(), &[0; 0x1_0000]).expect("synthetic font");
        FontRenderer::new_with_seed(data, false, false, seed)
    }

    #[test]
    fn seeded_obfuscation_matches_consecutive_java_random_draws() {
        let seed = 0x5EED_i64;
        let mut font = seeded_font(seed);
        let target_width = font.char_width('A');
        let mut oracle = JavaRandom::new(seed);

        let mut mesh = TextMesh::new();
        font.draw_string(&mut mesh, "\u{a7}kAAA", 0.0, 0.0, -1, false);
        assert_eq!(mesh.glyphs.len(), 3);
        assert_mesh_matches_oracle(&font, &mesh, &mut oracle, target_width);

        let mut next_mesh = TextMesh::new();
        font.draw_string(&mut next_mesh, "\u{a7}kAAA", 0.0, 0.0, -1, false);
        assert_mesh_matches_oracle(&font, &next_mesh, &mut oracle, target_width);
    }

    fn assert_mesh_matches_oracle(
        font: &FontRenderer,
        mesh: &TextMesh,
        oracle: &mut JavaRandom,
        target_width: i32,
    ) {
        for glyph in &mesh.glyphs {
            let selected = loop {
                let index = oracle.next_i32_bound(256) as usize;
                let candidate = font_char_at(index);
                if font.char_width(char::from_u32(u32::from(candidate)).unwrap()) == target_width {
                    break candidate;
                }
            };
            let selected_index = font_char_index(selected);
            // The cell's top-left, plus the bleed inset.
            let expected_u = ((selected_index % 16 * 8) as f32 + ASCII_UV_INSET) / 128.0;
            let expected_v = ((selected_index / 16 * 8) as f32 + ASCII_UV_INSET) / 128.0;
            assert_eq!(glyph.uv[0], [expected_u, expected_v]);
            assert_eq!(
                font.char_width(char::from_u32(u32::from(selected)).unwrap()),
                target_width
            );
        }
    }

    #[test]
    fn constructor_seed_and_reseed_are_deterministic_without_fixing_production() {
        let mut a = seeded_font(17);
        let mut b = seeded_font(17);
        let mut first_a = TextMesh::new();
        let mut first_b = TextMesh::new();
        a.draw_string(&mut first_a, "\u{a7}kAA", 0.0, 0.0, -1, false);
        b.draw_string(&mut first_b, "\u{a7}kAA", 0.0, 0.0, -1, false);
        assert_eq!(first_a, first_b);

        a.reseed_obfuscation(17);
        let mut replay = TextMesh::new();
        a.draw_string(&mut replay, "\u{a7}kAA", 0.0, 0.0, -1, false);
        assert_eq!(first_a, replay);
    }
}

#[cfg(test)]
mod anaglyph_tests {
    use super::*;

    fn synthetic_font(texture: FontTexture, anaglyph: bool, seed: i64) -> FontRenderer {
        let mut image = image::RgbaImage::new(128, 128);
        for cell in 0..256u32 {
            let x0 = (cell % 16) * 8;
            let y0 = (cell / 16) * 8;
            for x in 0..4 {
                image.put_pixel(x0 + x, y0 + 2, image::Rgba([255, 255, 255, 255]));
            }
        }
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut png, image::ImageFormat::Png)
            .expect("encode synthetic font");
        let data = FontData::from_parts(&png.into_inner(), &[0; 0x1_0000]).expect("synthetic font");
        FontRenderer::new_with_texture_and_seed(data, texture, anaglyph, false, seed)
    }

    #[test]
    fn live_anaglyph_palette_has_decisive_normal_and_dark_entries() {
        let mut ordinary = synthetic_font(FontTexture::Ascii, false, 0xA11CE);
        let mut galactic = synthetic_font(FontTexture::StandardGalactic, false, 0xA11CE);

        for font in [&ordinary, &galactic] {
            assert!(!font.anaglyph_enabled());
            assert_eq!(font.colors()[0], 0x000000);
            assert_eq!(font.colors()[6], 0xFFAA00, "normal gold");
            assert_eq!(font.colors()[15], 0xFFFFFF, "normal white");
            assert_eq!(font.colors()[16], 0x000000, "dark black");
            assert_eq!(font.colors()[22], 0x2A2A00, "dark gold");
            assert_eq!(font.colors()[31], 0x3F3F3F, "dark white");
        }

        ordinary.set_anaglyph(true);
        galactic.set_anaglyph(true);
        for font in [&ordinary, &galactic] {
            assert!(font.anaglyph_enabled());
            assert_eq!(font.colors()[1], 0x120077, "anaglyph dark blue");
            assert_eq!(font.colors()[6], 0xB0C34C, "anaglyph gold");
            assert_eq!(font.colors()[22], 0x252A0C, "anaglyph dark gold");
            assert_eq!(font.colors()[31], 0x3F3F3F, "anaglyph dark white");
        }
        assert_eq!(ordinary.colors(), galactic.colors());
    }

    #[test]
    fn false_true_false_restores_the_exact_palette_for_both_font_sheets() {
        for texture in [FontTexture::Ascii, FontTexture::StandardGalactic] {
            let mut font = synthetic_font(texture, false, 0xF00D);
            let normal = *font.colors();

            font.set_anaglyph(true);
            assert_eq!(font.colors(), &color_table(true));
            assert!(font.anaglyph_enabled());

            font.set_anaglyph(false);
            assert_eq!(*font.colors(), normal);
            assert_eq!(font.colors(), &color_table(false));
            assert!(!font.anaglyph_enabled());
        }
    }

    #[test]
    fn live_toggle_preserves_renderer_state_and_formatting_inputs() {
        let mut font = synthetic_font(FontTexture::StandardGalactic, false, 0x5EED);
        font.set_unicode_flag(true);
        font.set_bidi_flag(true);

        let mut mesh = TextMesh::new();
        font.draw_string(&mut mesh, "\u{a7}c\u{a7}kA", 3.0, 4.0, -1, false);
        assert!(font.random_style);

        let data = std::ptr::from_ref(font.data());
        let texture = font.default_texture();
        let unicode = font.unicode_flag();
        let bidi = font.bidi_flag();
        let height = font.font_height();
        let rng_seed = font.rng.raw_seed();
        let pos = (font.pos_x, font.pos_y);
        let base_color = font.base_color;
        let color = font.color;
        let text_color = font.text_color;
        let styles = (
            font.random_style,
            font.bold_style,
            font.italic_style,
            font.underline_style,
            font.strikethrough_style,
        );
        let width = font.string_width("\u{a7}lA");
        let trimmed = font.trim_to_width("\u{a7}cHello", 13);
        let wrapped = font.wrap_to_width("\u{a7}cHello world", 20);

        font.set_anaglyph(true);

        assert!(font.anaglyph_enabled());
        assert_eq!(std::ptr::from_ref(font.data()), data);
        assert_eq!(font.default_texture(), texture);
        assert_eq!(font.unicode_flag(), unicode);
        assert_eq!(font.bidi_flag(), bidi);
        assert_eq!(font.font_height(), height);
        assert_eq!(font.rng.raw_seed(), rng_seed);
        assert_eq!((font.pos_x, font.pos_y), pos);
        assert_eq!(font.base_color, base_color);
        assert_eq!(font.color, color);
        assert_eq!(font.text_color, text_color);
        assert_eq!(
            (
                font.random_style,
                font.bold_style,
                font.italic_style,
                font.underline_style,
                font.strikethrough_style,
            ),
            styles
        );
        assert_eq!(font.string_width("\u{a7}lA"), width);
        assert_eq!(font.trim_to_width("\u{a7}cHello", 13), trimmed);
        assert_eq!(font.wrap_to_width("\u{a7}cHello world", 20), wrapped);
    }
}
