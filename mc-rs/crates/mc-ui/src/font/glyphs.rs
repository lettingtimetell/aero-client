//! Where every glyph width comes from — hazard **H9**.
//!
//! java: `client/gui/FontRenderer.java:150-227` (`readFontTexture`, `readGlyphSizes`)
//!
//! Vanilla has two glyph sources and two width tables, and they do not agree with each other:
//!
//! | Source | Texture | Widths | Used when |
//! |---|---|---|---|
//! | default | `textures/font/ascii.png`, 16×16 cells of 8×8 | [`FontData::char_widths`] | the char is one of the 256 in [`FONT_CHARS`] **and** the unicode flag is off |
//! | unicode | `textures/font/unicode_page_XX.png`, 16×16 cells of 16×16 | [`FontData::glyph_widths`] from `font/glyph_sizes.bin` | otherwise |
//!
//! **H9 restated precisely.** The unicode widths are read verbatim from `glyph_sizes.bin` —
//! one byte per code point, start column in the high nibble and end column in the low nibble.
//! The ascii widths *are* derived from the texture, but by vanilla's exact algorithm, run
//! **once at load**: scan each 8-pixel cell right-to-left for the last column containing any
//! non-zero alpha, add one, scale by `8 / cell_width`, round with `(int)(0.5 + x)`, then add a
//! one-pixel gap. Measuring ink bounds any other way — or re-measuring per draw call — is what
//! the hazard warns about: it drifts a pixel or two per character.
//!
//! Two vanilla quirks are reproduced deliberately and asserted in the tests:
//!
//! * `char_widths[32]` (space) ends up **1**, not 4: the `if (j1 == 32)` special case in
//!   `readFontTexture` is immediately overwritten by the scan, which finds no ink. Nothing
//!   reads it, because both `getCharWidth` and `renderChar` special-case `' '` earlier.
//! * The unicode nibbles are extracted from a Java `byte`, so `glyphWidth[c] >>> 4`
//!   sign-extends first. For the 50 bytes in the real `glyph_sizes.bin` with the high bit set
//!   this yields a nonsensical start column — harmless in `getCharWidth`, which clamps first,
//!   but not in the quad builder. See [`FontData::unicode_nibbles`].

use std::sync::OnceLock;

use mc_assets::{ResourceLocation, ResourceManager};

use crate::UiError;

/// The 256 characters `ascii.png` holds, in cell order.
///
/// Vanilla stores this as one string literal and uses `indexOf` on it, so the *first*
/// occurrence wins: `'\0'` appears at index 25 and again at 26..=31, 127 and 255, but only
/// index 25 is ever selected.
// java: FontRenderer.java:240 (the literal repeated in renderChar/getCharWidth/renderStringAtPos)
pub const FONT_CHARS: &str = "\u{c0}\u{c1}\u{c2}\u{c8}\u{ca}\u{cb}\u{cd}\u{d3}\u{d4}\u{d5}\u{da}\u{df}\u{e3}\u{f5}\u{11f}\u{130}\u{131}\u{152}\u{153}\u{15e}\u{15f}\u{174}\u{175}\u{17e}\u{207}\u{0}\u{0}\u{0}\u{0}\u{0}\u{0}\u{0} !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\u{0}\u{c7}\u{fc}\u{e9}\u{e2}\u{e4}\u{e0}\u{e5}\u{e7}\u{ea}\u{eb}\u{e8}\u{ef}\u{ee}\u{ec}\u{c4}\u{c5}\u{c9}\u{e6}\u{c6}\u{f4}\u{f6}\u{f2}\u{fb}\u{f9}\u{ff}\u{d6}\u{dc}\u{f8}\u{a3}\u{d8}\u{d7}\u{192}\u{e1}\u{ed}\u{f3}\u{fa}\u{f1}\u{d1}\u{aa}\u{ba}\u{bf}\u{ae}\u{ac}\u{bd}\u{bc}\u{a1}\u{ab}\u{bb}\u{2591}\u{2592}\u{2593}\u{2502}\u{2524}\u{2561}\u{2562}\u{2556}\u{2555}\u{2563}\u{2551}\u{2557}\u{255d}\u{255c}\u{255b}\u{2510}\u{2514}\u{2534}\u{252c}\u{251c}\u{2500}\u{253c}\u{255e}\u{255f}\u{255a}\u{2554}\u{2569}\u{2566}\u{2560}\u{2550}\u{256c}\u{2567}\u{2568}\u{2564}\u{2565}\u{2559}\u{2558}\u{2552}\u{2553}\u{256b}\u{256a}\u{2518}\u{250c}\u{2588}\u{2584}\u{258c}\u{2590}\u{2580}\u{3b1}\u{3b2}\u{393}\u{3c0}\u{3a3}\u{3c3}\u{3bc}\u{3c4}\u{3a6}\u{398}\u{3a9}\u{3b4}\u{221e}\u{2205}\u{2208}\u{2229}\u{2261}\u{b1}\u{2265}\u{2264}\u{2320}\u{2321}\u{f7}\u{2248}\u{b0}\u{2219}\u{b7}\u{221a}\u{207f}\u{b2}\u{25a0}\u{0}";

/// `textures/font/ascii.png` — the default 256-glyph font sheet.
pub const ASCII_FONT: &str = "textures/font/ascii.png";
/// `textures/font/ascii_sga.png` — the enchanting table's Standard Galactic sheet.
pub const STANDARD_GALACTIC_FONT: &str = "textures/font/ascii_sga.png";
/// `font/glyph_sizes.bin` — the baked unicode width table (hazard H9).
pub const GLYPH_SIZES: &str = "font/glyph_sizes.bin";
/// Vanilla's line height in GUI pixels.
// java: FontRenderer.java:35 (FONT_HEIGHT)
pub const FONT_HEIGHT: i32 = 9;

/// The resource location of one unicode page, e.g. page `0x04` →
/// `textures/font/unicode_page_04.png`.
// java: FontRenderer.java:269 (getUnicodePageLocation)
#[must_use]
pub fn unicode_page_location(page: u8) -> ResourceLocation {
    ResourceLocation::minecraft(format!("textures/font/unicode_page_{page:02x}.png"))
}

/// `FONT_CHARS.indexOf(c)`, memoised over the whole BMP.
///
/// Returns `-1` for a code unit that is not in the sheet, matching Java's `indexOf` contract
/// (the callers branch on `!= -1`, so a signed sentinel is the honest shape here).
#[must_use]
pub fn font_char_index(unit: u16) -> i32 {
    static TABLE: OnceLock<Box<[i16]>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = vec![-1i16; 0x1_0000].into_boxed_slice();
        // Reverse iteration makes the *first* occurrence win, matching `indexOf`.
        let cells: Vec<char> = FONT_CHARS.chars().collect();
        for (i, c) in cells.into_iter().enumerate().rev() {
            t[c as usize] = i as i16;
        }
        t
    });
    i32::from(table[unit as usize])
}

/// Vanilla's one-shot alpha-column scan over `ascii.png`. Hazard **H9**.
///
/// `rgba` is tightly packed RGBA8, `width` × `height`. The sheet is 16×16 cells; the returned
/// width for cell `n` is `round(last_inked_column + 1) * 8 / cell_width + 1`.
///
/// # Panics
/// Never — a too-small buffer or a non-multiple-of-16 size yields zero widths rather than an
/// out-of-bounds read. Call [`FontData::from_parts`] for the checked path.
// java: FontRenderer.java:150 (readFontTexture)
#[must_use]
pub fn measure_ascii_widths(rgba: &[u8], width: u32, height: u32) -> [i32; 256] {
    let mut char_width = [0i32; 256];
    let i = width as i32;
    let j = height as i32;
    let k = j / 16; // cell height
    let l = i / 16; // cell width
    if k <= 0 || l <= 0 {
        return char_width;
    }
    let i1 = 1; // the one-pixel inter-glyph gap vanilla adds to every width
    let f = 8.0f32 / l as f32;

    for (j1, slot) in char_width.iter_mut().enumerate() {
        let k1 = (j1 % 16) as i32;
        let l1 = (j1 / 16) as i32;

        // Vanilla writes 4 here and then unconditionally overwrites it below; the space cell
        // has no ink, so the surviving value is 1. Kept for fidelity — see the module docs.
        if j1 == 32 {
            *slot = 3 + i1;
        }

        let mut i2 = l - 1;
        while i2 >= 0 {
            let j2 = k1 * l + i2;
            let mut inked = false;
            let mut k2 = 0;
            while k2 < k && !inked {
                // java uses `l1 * l + k2`, the *cell width*, as the row stride multiplier.
                // For a square sheet (every vanilla font is) l == k, so it is only a latent
                // bug; transcribed as-is so a resource pack with a non-square ascii.png
                // misbehaves the same way it does in vanilla.
                let row = l1 * l + k2;
                // In i64: `row * i` overflows an i32 for a sheet
                // about 48,000 pixels wide (a panic in debug builds); real sheets give the same
                // index either way. Such a sheet decodes (the upload leaves an over-limit one
                // blank), so it is measured.
                let idx = (i64::from(row) * i64::from(i) + i64::from(j2)) * 4 + 3;
                let alpha = usize::try_from(idx).ok().and_then(|idx| rgba.get(idx));
                if alpha.is_some_and(|&a| a != 0) {
                    inked = true;
                }
                k2 += 1;
            }
            if inked {
                break;
            }
            i2 -= 1;
        }

        i2 += 1;
        *slot = (0.5f64 + f64::from(i2 as f32 * f)) as i32 + i1;
    }
    char_width
}

/// The two baked width tables, plus the decoded `ascii.png` pixels.
///
/// Construct with [`FontData::load`] (from a [`ResourceManager`]) or [`FontData::from_parts`]
/// (from raw bytes, for tests).
pub struct FontData {
    char_widths: [i32; 256],
    glyph_widths: Box<[u8; 0x1_0000]>,
    ascii_rgba: Vec<u8>,
    ascii_size: (u32, u32),
}

impl std::fmt::Debug for FontData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FontData")
            .field("ascii_size", &self.ascii_size)
            .field("char_widths[65]", &self.char_widths[65])
            .finish_non_exhaustive()
    }
}

impl FontData {
    /// Read `textures/font/ascii.png` and `font/glyph_sizes.bin` out of the resource packs.
    ///
    /// # Errors
    /// [`UiError::FontAsset`] if either resource is missing, [`UiError::FontTexture`] if the
    /// PNG will not decode, [`UiError::GlyphSizes`] if the width table is the wrong length.
    pub fn load(assets: &ResourceManager) -> Result<Self, UiError> {
        Self::load_with_sheet(assets, ASCII_FONT)
    }

    /// Read an arbitrary 16x16 default-font sheet plus vanilla's shared unicode widths.
    ///
    /// Java constructs `standardGalacticFontRenderer` with `ascii_sga.png` through the same
    /// constructor as the ordinary renderer, so its measured widths must come from that sheet.
    pub fn load_with_sheet(assets: &ResourceManager, sheet: &str) -> Result<Self, UiError> {
        let ascii_loc = ResourceLocation::minecraft(sheet);
        let sizes_loc = ResourceLocation::minecraft(GLYPH_SIZES);
        let png = assets
            .load(&ascii_loc)
            .map_err(|e| UiError::FontAsset(ascii_loc.to_string(), e))?;
        let sizes = assets
            .load(&sizes_loc)
            .map_err(|e| UiError::FontAsset(sizes_loc.to_string(), e))?;
        Self::from_parts_at(&png, &sizes, sheet)
    }

    /// Re-read only the default-font sheet after a resource reload.
    ///
    /// `FontRenderer.onResourceManagerReload` calls `readFontTexture` but not
    /// `readGlyphSizes`: the unicode table is constructor state in 1.8.9. Keeping the latter
    /// here also means a pack reload cannot quietly replace the renderer's long-lived glyph
    /// metrics.
    ///
    /// # Errors
    /// [`UiError::FontAsset`] if `ascii.png` is missing, or [`UiError::FontTexture`] if it
    /// cannot be decoded as a 16x16 cell grid.
    // java: FontRenderer.java:145-148 (onResourceManagerReload -> readFontTexture)
    pub fn reload_ascii(&mut self, assets: &ResourceManager) -> Result<(), UiError> {
        self.reload_sheet(assets, ASCII_FONT)
    }

    /// Re-read the selected default-font sheet while retaining constructor-owned unicode data.
    pub fn reload_sheet(&mut self, assets: &ResourceManager, sheet: &str) -> Result<(), UiError> {
        let ascii_loc = ResourceLocation::minecraft(sheet);
        let png = assets
            .load(&ascii_loc)
            .map_err(|e| UiError::FontAsset(ascii_loc.to_string(), e))?;
        self.reload_sheet_from_png(&png, sheet)
    }

    /// Build from raw `ascii.png` bytes and raw `glyph_sizes.bin` bytes.
    ///
    /// # Errors
    /// [`UiError::FontTexture`] if the PNG will not decode, [`UiError::GlyphSizes`] if
    /// `glyph_sizes` is not exactly 65536 bytes.
    pub fn from_parts(ascii_png: &[u8], glyph_sizes: &[u8]) -> Result<Self, UiError> {
        Self::from_parts_at(ascii_png, glyph_sizes, ASCII_FONT)
    }

    fn from_parts_at(ascii_png: &[u8], glyph_sizes: &[u8], sheet: &str) -> Result<Self, UiError> {
        if glyph_sizes.len() != 0x1_0000 {
            return Err(UiError::GlyphSizes(glyph_sizes.len()));
        }
        let (char_widths, ascii_rgba, ascii_size) = decode_ascii(ascii_png, sheet)?;
        let mut glyph_widths = vec![0u8; 0x1_0000].into_boxed_slice();
        glyph_widths.copy_from_slice(glyph_sizes);
        let glyph_widths: Box<[u8; 0x1_0000]> = glyph_widths
            .try_into()
            .map_err(|_| UiError::GlyphSizes(glyph_sizes.len()))?;
        Ok(Self {
            char_widths,
            glyph_widths,
            ascii_rgba,
            ascii_size,
        })
    }

    fn reload_sheet_from_png(&mut self, ascii_png: &[u8], sheet: &str) -> Result<(), UiError> {
        let (char_widths, ascii_rgba, ascii_size) = decode_ascii(ascii_png, sheet)?;
        self.char_widths = char_widths;
        self.ascii_rgba = ascii_rgba;
        self.ascii_size = ascii_size;
        Ok(())
    }

    /// The 256 default-font widths, indexed by position in [`FONT_CHARS`].
    #[must_use]
    pub fn char_widths(&self) -> &[i32; 256] {
        &self.char_widths
    }

    /// The raw `glyph_sizes.bin` bytes, indexed by BMP code point.
    #[must_use]
    pub fn glyph_widths(&self) -> &[u8; 0x1_0000] {
        &self.glyph_widths
    }

    /// Decoded `ascii.png`, tightly packed RGBA8.
    #[must_use]
    pub fn ascii_rgba(&self) -> &[u8] {
        &self.ascii_rgba
    }

    /// `ascii.png`'s pixel dimensions.
    #[must_use]
    pub fn ascii_size(&self) -> (u32, u32) {
        self.ascii_size
    }

    /// The `(start_column, end_column)` nibbles for a code unit, **with Java's sign extension**.
    ///
    /// `glyph_sizes.bin` is a `byte[]` in Java, so `glyphWidth[c] >>> 4` widens to a signed
    /// `int` before the *unsigned* shift. For a byte ≥ `0x80` that produces a start column of
    /// roughly 2.7×10⁸ rather than 8..15. `getCharWidth` never notices because it clamps
    /// (`end > 7 → end = 15, start = 0`) before using the start column, but `renderUnicodeChar`
    /// does not clamp, so those glyphs sample far outside their page — a real vanilla bug that
    /// this port reproduces rather than silently fixes.
    ///
    /// In the shipped 1.8.9 `glyph_sizes.bin` exactly 50 bytes have the high bit set and every
    /// one of them also has `end > 7`, so the clamp always rescues the *width* path.
    // java: FontRenderer.java:300 / :678
    #[must_use]
    pub fn unicode_nibbles(&self, unit: u16) -> (i32, i32) {
        let b = i32::from(self.glyph_widths[unit as usize] as i8);
        let start = ((b as u32) >> 4) as i32;
        let end = b & 15;
        (start, end)
    }
}

type DecodedAscii = ([i32; 256], Vec<u8>, (u32, u32));

fn decode_ascii(ascii_png: &[u8], location: &str) -> Result<DecodedAscii, UiError> {
    let img = image::load_from_memory_with_format(ascii_png, image::ImageFormat::Png)
        .map_err(|e| UiError::FontTexture {
            location: location.to_owned(),
            reason: e.to_string(),
        })?
        .to_rgba8();
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 || w % 16 != 0 || h % 16 != 0 {
        return Err(UiError::FontTexture {
            location: location.to_owned(),
            reason: format!("{w}x{h} is not a 16x16 grid of cells"),
        });
    }
    let rgba = img.into_raw();
    Ok((measure_ascii_widths(&rgba, w, h), rgba, (w, h)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn font_chars_is_exactly_256_cells() {
        assert_eq!(FONT_CHARS.chars().count(), 256);
    }

    #[test]
    fn font_char_index_matches_java_indexof() {
        // First occurrence wins, exactly like String.indexOf.
        assert_eq!(font_char_index(0), 25, "NUL first appears at cell 25");
        assert_eq!(font_char_index(b' '.into()), 32);
        assert_eq!(font_char_index(b'A'.into()), 65);
        assert_eq!(font_char_index(b'a'.into()), 97);
        assert_eq!(font_char_index(0x00c0), 0, "A-grave is cell 0");
        assert_eq!(
            font_char_index(0x25a0),
            254,
            "the filled square is cell 254"
        );
        assert_eq!(
            font_char_index(0x4e00),
            -1,
            "a CJK ideograph is not in the sheet"
        );
        // Every listed char resolves back to its own index unless an earlier duplicate exists.
        for (i, c) in FONT_CHARS.chars().enumerate() {
            let got = font_char_index(c as u16);
            assert!(got >= 0);
            assert!(got as usize <= i);
            assert_eq!(FONT_CHARS.chars().nth(got as usize), Some(c));
        }
    }

    #[test]
    fn unicode_page_locations_are_lower_case_hex() {
        assert_eq!(
            unicode_page_location(0).to_string(),
            "minecraft:textures/font/unicode_page_00.png"
        );
        assert_eq!(
            unicode_page_location(0xab).to_string(),
            "minecraft:textures/font/unicode_page_ab.png"
        );
        assert_eq!(
            unicode_page_location(0xff).to_string(),
            "minecraft:textures/font/unicode_page_ff.png"
        );
    }

    /// A synthetic 128×128 sheet where cell `n` is inked in columns `0..=(n % 8)`.
    fn synthetic_sheet() -> Vec<u8> {
        let mut rgba = vec![0u8; 128 * 128 * 4];
        for cell in 0..256usize {
            let cx = (cell % 16) * 8;
            let cy = (cell / 16) * 8;
            let last = cell % 8;
            for col in 0..=last {
                let idx = ((cy + 3) * 128 + cx + col) * 4;
                rgba[idx + 3] = 255;
            }
        }
        rgba
    }

    #[test]
    fn measure_ascii_widths_is_last_inked_column_plus_two() {
        let widths = measure_ascii_widths(&synthetic_sheet(), 128, 128);
        for (cell, width) in widths.iter().enumerate() {
            // last inked column = cell % 8, so i2 lands on (cell % 8) + 1 and the +1 gap
            // brings the width to (cell % 8) + 2.
            assert_eq!(*width, (cell % 8) as i32 + 2, "cell {cell}");
        }
    }

    #[test]
    fn a_blank_cell_measures_one_pixel_wide() {
        let rgba = vec![0u8; 128 * 128 * 4];
        let widths = measure_ascii_widths(&rgba, 128, 128);
        assert!(widths.iter().all(|&w| w == 1));
    }

    #[test]
    fn measure_ascii_widths_looks_only_at_alpha() {
        let mut rgba = vec![0u8; 128 * 128 * 4];
        // Java's readFontTexture tests `(argb >> 24) & 255`, so opaque RGB in a
        // fully transparent texel must not make the cell wider.
        let rgb_only = ((2 * 128 + 7) * 4) as usize;
        rgba[rgb_only..rgb_only + 3].copy_from_slice(&[255, 1, 127]);
        assert_eq!(measure_ascii_widths(&rgba, 128, 128)[0], 1);

        // Conversely, alpha is sufficient even when every colour channel is zero.
        let alpha_only = ((2 * 128 + 3) * 4) as usize;
        rgba[alpha_only + 3] = 1;
        assert_eq!(measure_ascii_widths(&rgba, 128, 128)[0], 5);
    }

    #[test]
    fn measure_ascii_widths_preserves_vanillas_non_square_row_stride_quirk() {
        // Java uses the cell width `l`, rather than the cell height `k`, in
        // `(l1 * l + k2) * image_width`.  A 32x256 sheet makes that observable
        // while keeping every read inside the backing image.  The ink is at
        // physical row 2: Java therefore sees it in cell 16 (row 1), even
        // though that cell's actual origin is physical row 16.
        let width = 32usize;
        let height = 256usize;
        let mut rgba = vec![0u8; width * height * 4];
        let idx = (2 * width + 1) * 4;
        rgba[idx + 3] = 255;

        let widths = measure_ascii_widths(&rgba, width as u32, height as u32);
        assert_eq!(widths[16], 9, "the Java l1*l row stride must be retained");
    }

    #[test]
    fn measure_scales_a_double_resolution_sheet_back_to_eight_pixel_cells() {
        // A 256x256 pack sheet: cell width 16, f = 0.5, so an ink column at 15 -> i2 = 16
        // -> (int)(0.5 + 16*0.5) + 1 = 9.
        let mut rgba = vec![0u8; 256 * 256 * 4];
        for cell in 0..256usize {
            let cx = (cell % 16) * 16;
            let cy = (cell / 16) * 16;
            let idx = ((cy + 4) * 256 + cx + 15) * 4;
            rgba[idx + 3] = 255;
        }
        let widths = measure_ascii_widths(&rgba, 256, 256);
        assert!(widths.iter().all(|&w| w == 9), "got {:?}", &widths[..4]);
    }

    #[test]
    fn from_parts_rejects_a_short_glyph_table() {
        // A 1x1 PNG is enough to get past the decode and hit the size check.
        let png = tiny_png();
        let err = FontData::from_parts(&png, &[0u8; 16]).unwrap_err();
        assert!(matches!(err, UiError::GlyphSizes(16)), "{err}");
    }

    #[test]
    fn from_parts_rejects_a_non_grid_texture() {
        let png = tiny_png();
        let err = FontData::from_parts(&png, &[0u8; 0x1_0000]).unwrap_err();
        assert!(matches!(err, UiError::FontTexture { .. }), "{err}");
    }

    #[test]
    fn from_parts_rejects_garbage() {
        let err = FontData::from_parts(b"not a png", &[0u8; 0x1_0000]).unwrap_err();
        assert!(matches!(err, UiError::FontTexture { .. }), "{err}");
    }

    #[test]
    fn reloading_ascii_keeps_the_constructor_unicode_table() {
        let initial = sheet_png(0);
        let replacement = sheet_png(6);
        let mut glyphs = vec![0u8; 0x1_0000];
        glyphs[b'a' as usize] = 0x16;
        let mut data = FontData::from_parts(&initial, &glyphs).expect("initial font data");

        data.reload_sheet_from_png(&replacement, ASCII_FONT)
            .expect("replacement ascii sheet");

        assert_eq!(data.char_widths()[b'a' as usize], 8);
        assert_eq!(data.glyph_widths()[b'a' as usize], 0x16);
    }

    #[test]
    fn unicode_nibbles_preserve_java_byte_sign_extension() {
        let mut glyphs = vec![0u8; 0x1_0000];
        glyphs[0x1234] = 0x80;
        glyphs[0x1235] = 0xf1;
        let data = FontData::from_parts(&sheet_png(0), &glyphs).expect("font data");

        // Java promotes byte values to signed int before applying >>> 4:
        // (byte)0x80 >>> 4 == 0x0ffffff8, not 8.
        assert_eq!(data.unicode_nibbles(0x1234), (268_435_448, 0));
        assert_eq!(data.unicode_nibbles(0x1235), (268_435_455, 1));
    }

    fn sheet_png(last_inked_column: u32) -> Vec<u8> {
        let mut img = image::RgbaImage::new(128, 128);
        for cell in 0..256u32 {
            let x = (cell % 16) * 8 + last_inked_column;
            let y = (cell / 16) * 8;
            img.put_pixel(x, y, image::Rgba([255, 255, 255, 255]));
        }
        let mut buf = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut buf, image::ImageFormat::Png)
            .expect("encode");
        buf.into_inner()
    }

    fn tiny_png() -> Vec<u8> {
        let img = image::RgbaImage::new(1, 1);
        let mut buf = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut buf, image::ImageFormat::Png)
            .expect("encode");
        buf.into_inner()
    }
}
