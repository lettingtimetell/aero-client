//! Drawing primitives, GUI scaling, and the vanilla widget set.
//!
//! java: `client/gui/Gui.java`, `GuiButton.java`, `GuiSlider.java`, `GuiTextField.java`,
//! `GuiSlot.java`, `ScaledResolution.java`
//!
//! | Module | What it holds |
//! |---|---|
//! | [`draw`] | [`DrawList`], the triangle soup every widget writes into, plus vanilla's `Gui` helpers and nine-slice |
//! | [`transform`] | [`Transform`], the 2-D affine [`DrawList`]'s matrix stack applies |
//! | [`scaled`] | [`ScaledResolution`] — GUI scaling and window↔GUI coordinate mapping |
//! | [`keys`] | LWJGL key codes, [`Modifiers`], the [`Clipboard`] seam |
//! | [`button`] | [`Button`], [`Slider`] |
//! | [`text_field`] | [`TextField`] |
//! | [`list`] | [`ScrollList`] and its [`Scrollbar`] |
//! | [`checkbox`] | [`Checkbox`] (a 1.9 forward-port) and [`ToggleButton`] (the real 1.8.9 shape) |
//! | [`gpu`] | [`UiPhase`], the one file that touches wgpu |
//!
//! Widgets are **plain state plus arithmetic**. None of them own a font, a resource manager or
//! a GPU handle: measurement takes `&FontRenderer`, drawing takes `&mut DrawList`. That is what
//! makes hit-testing, scrolling and text editing testable with no display attached.

pub mod button;
pub mod checkbox;
pub mod draw;
pub mod gpu;
pub mod keys;
pub mod list;
pub mod scaled;
pub mod text_field;
pub mod transform;

pub use button::{
    capture_button_press, draw_centered_string, draw_string, set_precise_cursor_x, snap_position,
    Button, Slider,
};
pub use checkbox::{Checkbox, ToggleButton};
pub use draw::{
    unpack_argb, Border, DrawBatch, DrawList, Rect, UiCull, UiDepth, UiTexture, UiVertex,
    UnblendedText, GUI_TEXTURE_SIZE, ICONS_TEXTURE, OPTIONS_BACKGROUND, WIDGETS_TEXTURE,
};
pub use gpu::{UiPhase, UiRenderer, UiScreenPhase};
pub use keys::{Clipboard, MemoryClipboard, Modifiers};
pub use list::{ScrollList, Scrollbar, ScrollbarX};
pub use scaled::ScaledResolution;
pub use text_field::TextField;
pub use transform::Transform;

#[cfg(test)]
pub(crate) mod test_support {
    //! A synthetic font whose every default glyph is exactly 5 pixels wide, so widget tests can
    //! state expected pixel positions as small integers instead of quoting the real artwork.

    use crate::font::{FontData, FontRenderer, FontTexture};

    /// 128×128 sheet, every cell inked in columns 0..=3 → measured width 5.
    fn uniform_ascii_png() -> Vec<u8> {
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

    /// A renderer over [`uniform_ascii_png`] with an empty unicode table.
    pub(crate) fn uniform_font() -> FontRenderer {
        uniform_font_with_texture(FontTexture::Ascii)
    }

    /// The same deterministic fixture with a caller-selected default-sheet identity.
    pub(crate) fn uniform_font_with_texture(texture: FontTexture) -> FontRenderer {
        let data = FontData::from_parts(&uniform_ascii_png(), &[0u8; 0x1_0000])
            .expect("synthetic font data");
        FontRenderer::new_with_texture(data, texture, false, false)
    }

    /// 128×128 sheet whose cells are inked to varying depths, so glyph advances are
    /// **proportional** (2 to 8 pixels) rather than all the same.
    ///
    /// A uniform font hides every off-by-one in code that maps a pixel column back to a
    /// character index, because every column boundary is a multiple of one number.
    fn ragged_ascii_png() -> Vec<u8> {
        let mut img = image::RgbaImage::new(128, 128);
        for cell in 0..256u32 {
            let cx = (cell % 16) * 8;
            let cy = (cell / 16) * 8;
            // Ink through column `c` measures `c + 2` wide.
            for col in 0..=(cell % 7) {
                img.put_pixel(cx + col, cy + 2, image::Rgba([255, 255, 255, 255]));
            }
        }
        let mut buf = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut buf, image::ImageFormat::Png)
            .expect("encode");
        buf.into_inner()
    }

    /// A renderer over [`ragged_ascii_png`].
    pub(crate) fn ragged_font() -> FontRenderer {
        let data =
            FontData::from_parts(&ragged_ascii_png(), &[0u8; 0x1_0000]).expect("synthetic font");
        FontRenderer::new(data, false, false)
    }

    /// FNV-1a 64, so a whole vertex buffer can be pinned as one literal.
    ///
    /// Used by the byte-identity regression tests in [`super::draw`] and [`super::list`]: their
    /// golden values were captured from the tree **before** the transform stack and the
    /// configurable scrollbar landed, so a change in either fingerprint means an existing call
    /// stopped producing the geometry S16 and S17 were built against.
    pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in bytes {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    #[test]
    fn the_synthetic_font_really_is_five_wide() {
        let f = uniform_font();
        assert_eq!(f.char_width('A'), 5);
        assert_eq!(f.string_width("abcd"), 20);
        assert_eq!(f.char_width(' '), 4);
    }

    #[test]
    fn the_ragged_font_really_is_proportional() {
        let f = ragged_font();
        // Cell `n` is inked through column `n % 7`, so widths cycle over 2..=8. Which cell a
        // character lands in is `FONT_CHARS`' own order, not its code point, so this asserts the
        // *shape* of the table rather than restating that ordering.
        let widths: Vec<i32> = ('!'..='~').map(|c| f.char_width(c)).collect();
        assert!(widths.iter().all(|w| (2..=8).contains(w)), "{widths:?}");
        let distinct: std::collections::BTreeSet<_> = widths.iter().collect();
        assert_eq!(distinct.len(), 7, "all seven widths occur: {distinct:?}");
        // And a string is the sum of its characters, as the caret cache assumes.
        let s = "!\"#$%&'";
        let sum: i32 = s.chars().map(|c| f.char_width(c)).sum();
        assert_eq!(f.string_width(s), sum);
    }

    #[test]
    fn the_fingerprint_helper_is_the_real_fnv1a() {
        // The canonical FNV-1a 64 test vectors.
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(b"foobar"), 0x8594_4171_f739_67e8);
    }
}
