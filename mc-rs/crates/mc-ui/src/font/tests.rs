//! Unit tests for the layout state machine.
//!
//! These use a *synthetic* font whose metrics are chosen so every expected value can be worked
//! out by hand, which keeps the assertions about the algorithm rather than about the artwork.
//! The golden values from the real `ascii.png` / `glyph_sizes.bin` live in
//! `tests/vanilla_font.rs`.

use super::*;

/// Every ascii cell is inked in columns 0..=3, so every default glyph measures 5 pixels
/// (`last_inked + 1 + 1`). Space is special-cased to 4 by the renderer regardless.
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

/// `'a'` → start 1, end 6 (a 4-wide unicode glyph). `'b'` → 0x0F, the "full cell" byte, which
/// the `end > 7` clamp turns into 9. Everything else is zero, i.e. absent.
fn synthetic_glyph_sizes() -> Vec<u8> {
    let mut v = vec![0u8; 0x1_0000];
    v[b'a' as usize] = 0x16;
    v[b'b' as usize] = 0x0F;
    v[b'c' as usize] = 0x0E; // start 0, end 14: clamped to 9 for width, 8.5 when drawn
    v[0x4e00] = 0x0F; // a CJK ideograph, not in ascii.png at all
    v
}

fn font(unicode: bool) -> FontRenderer {
    let data = FontData::from_parts(&synthetic_ascii_png(), &synthetic_glyph_sizes())
        .expect("synthetic font");
    FontRenderer::new(data, false, unicode)
}

#[test]
fn standard_galactic_glyphs_keep_a_distinct_sheet_identity() {
    let data = FontData::from_parts(&synthetic_ascii_png(), &synthetic_glyph_sizes())
        .expect("synthetic font");
    let mut font =
        FontRenderer::new_with_texture(data, FontTexture::StandardGalactic, false, false);
    assert_eq!(font.default_texture(), FontTexture::StandardGalactic);
    assert!(!font.unicode_flag());
    assert_eq!(
        FontTexture::StandardGalactic.location().to_string(),
        "minecraft:textures/font/ascii_sga.png"
    );

    let mut mesh = TextMesh::new();
    font.draw_string(&mut mesh, "runes", 0.0, 0.0, -1, false);
    assert!(!mesh.glyphs.is_empty());
    assert!(mesh
        .glyphs
        .iter()
        .all(|glyph| glyph.texture == FontTexture::StandardGalactic));
}

// ------------------------------------------------------------------ widths

#[test]
fn every_synthetic_default_glyph_is_five_wide() {
    let f = font(false);
    for c in "abcXYZ019".chars() {
        assert_eq!(f.char_width(c), 5, "{c:?}");
    }
    assert_eq!(f.char_width(' '), 4, "space is hard-coded to 4");
    assert_eq!(f.char_width('§'), -1, "the section sign is the sentinel");
}

#[test]
fn unicode_widths_come_from_the_byte_table_not_the_texture() {
    let f = font(true);
    // 0x16: start 1, end 6 -> (6+1-1)/2 + 1 = 4
    assert_eq!(f.char_width('a'), 4);
    // 0x0F: end 15 > 7 -> clamped to (15,0) -> (15+1-0)/2 + 1 = 9
    assert_eq!(f.char_width('b'), 9);
    // 0x0E: end 14 > 7 -> same clamp -> 9, even though the ink stops one column earlier
    assert_eq!(f.char_width('c'), 9);
    // No entry at all -> zero width, and the glyph is not drawn.
    assert_eq!(f.char_width('d'), 0);
    assert_eq!(f.char_width(' '), 4, "space still bypasses both tables");
    assert_eq!(f.char_width('\u{4e00}'), 9);
}

#[test]
fn the_unicode_flag_switches_which_table_answers() {
    let mut f = font(false);
    assert_eq!(f.char_width('a'), 5, "ascii sheet");
    f.set_unicode_flag(true);
    assert_eq!(f.char_width('a'), 4, "unicode page");
    f.set_unicode_flag(false);
    assert_eq!(f.char_width('a'), 5);
    // A char the ascii sheet does not have always falls through to the unicode table.
    assert_eq!(f.char_width('\u{4e00}'), 9);
}

#[test]
fn a_supplementary_char_measures_as_its_two_surrogates() {
    let f = font(false);
    // Neither surrogate is in ascii.png and neither has a glyph_sizes entry, so both are 0.
    assert_eq!(f.char_width('\u{1f600}'), 0);
    assert_eq!(f.string_width("\u{1f600}"), 0);
    assert_eq!(
        f.string_width("a\u{1f600}a"),
        10,
        "the surrogates add nothing"
    );
}

// ------------------------------------------------------------------ string_width

#[test]
fn string_width_sums_glyphs() {
    let f = font(false);
    assert_eq!(f.string_width(""), 0);
    assert_eq!(f.string_width("abc"), 15);
    assert_eq!(f.string_width("a b"), 14, "5 + 4 + 5");
}

#[test]
fn bold_adds_one_pixel_per_glyph_until_reset() {
    let f = font(false);
    assert_eq!(f.string_width("§labc"), 18, "3 glyphs at 6");
    assert_eq!(f.string_width("§lab§rc"), 17, "6 + 6 + 5");
    assert_eq!(f.string_width("§Lab"), 12, "the code is case-insensitive");
    assert_eq!(f.string_width("§lab§Rc"), 17);
}

#[test]
fn bold_does_not_add_a_pixel_to_a_zero_width_glyph() {
    let f = font(false);
    assert_eq!(
        f.unit_width(0x200b),
        0,
        "the synthetic font has no zero-width-space glyph"
    );
    assert_eq!(
        f.string_width("§l\u{200b}"),
        0,
        "getStringWidth gates bold's extra pixel on a positive glyph width"
    );
}

/// The one place vanilla's two width walkers disagree: `getStringWidth` only clears bold for
/// `r`/`R`, while `sizeStringToWidth` also clears it for any colour code.
#[test]
fn a_colour_code_does_not_clear_bold_in_string_width() {
    let f = font(false);
    assert_eq!(f.string_width("§lab§cc"), 18, "still bold after §c: 6+6+6");
    // The wrapping walker *does* clear bold on §c, so it measures 6+6+5 = 17 for the same
    // string — one pixel narrower than `string_width` believes it to be.
    assert_eq!(
        f.size_string_to_width("§lab§cc", 17),
        7,
        "all 7 units fit at 17"
    );
    assert_eq!(
        f.size_string_to_width("§lab§cc", 16),
        6,
        "the last glyph tips it over"
    );
}

#[test]
fn formatting_codes_are_free_and_a_dangling_section_sign_costs_minus_one() {
    let f = font(false);
    assert_eq!(f.string_width("§ca"), 5, "the code itself is zero-width");
    assert_eq!(
        f.string_width("§"),
        -1,
        "a lone § is the sentinel, unconsumed"
    );
    assert_eq!(f.string_width("a§"), 4, "5 + (-1)");
    assert_eq!(
        f.string_width("§§c"),
        5,
        "the first § consumes the second, leaving a real 'c' to measure"
    );
}

#[test]
fn an_unknown_code_after_section_is_still_consumed() {
    let f = font(false);
    assert_eq!(f.string_width("§za"), 5, "the z is eaten, only 'a' counts");
}

// ------------------------------------------------------------------ trimming

#[test]
fn trim_to_width_takes_the_longest_prefix_that_fits() {
    let f = font(false);
    assert_eq!(f.trim_to_width("abcdef", 15), "abc");
    assert_eq!(f.trim_to_width("abcdef", 14), "ab", "c would reach 15 > 14");
    assert_eq!(f.trim_to_width("abcdef", 0), "");
    assert_eq!(f.trim_to_width("abcdef", 1000), "abcdef");
    assert_eq!(f.trim_to_width("", 100), "");
}

#[test]
fn trim_reversed_takes_the_suffix() {
    let f = font(false);
    assert_eq!(f.trim_to_width_dir("abcdef", 15, true), "def");
    assert_eq!(f.trim_to_width_dir("abcdef", 14, true), "ef");
}

#[test]
fn trim_carries_the_bold_flag_but_never_charges_for_the_code() {
    let f = font(false);
    // §l is free; then each glyph costs 6.
    assert_eq!(f.trim_to_width("§labcd", 12), "§lab");
    assert_eq!(f.trim_to_width("§labcd", 11), "§la");
}

// ------------------------------------------------------------------ wrapping

#[test]
fn size_string_to_width_prefers_the_last_space() {
    let f = font(false);
    // "aa bb cc" at 5 px per glyph, 4 per space: cut lands mid-"cc", so back up to the space.
    assert_eq!(f.size_string_to_width("aa bb cc", 30), 5);
    assert_eq!(f.wrap_to_lines("aa bb cc", 30), vec!["aa bb", "cc"]);
}

#[test]
fn size_string_to_width_breaks_mid_word_when_there_is_no_space() {
    let f = font(false);
    assert_eq!(
        f.size_string_to_width("aaaaaa", 22),
        4,
        "4 glyphs = 20, the 5th hits 25"
    );
    assert_eq!(f.wrap_to_lines("aaaaaa", 22), vec!["aaaa", "aa"]);
}

#[test]
fn zero_width_wrap_makes_progress_instead_of_recursing_forever() {
    let f = font(false);
    assert_eq!(f.wrap_to_lines("aa", 0), vec!["a", "a"]);
}

#[test]
fn an_explicit_newline_ends_the_line_at_its_own_index() {
    let f = font(false);
    assert_eq!(f.size_string_to_width("ab\ncd", 1000), 2);
    assert_eq!(f.wrap_to_lines("ab\ncd", 1000), vec!["ab", "cd"]);
}

#[test]
fn wrapping_repeats_the_active_formatting_prefix() {
    let f = font(false);
    // §l is free, glyphs cost 6 each: 3 fit in 18.
    assert_eq!(f.wrap_to_lines("§laaaaaa", 18), vec!["§laaa", "§laaa"]);
    // A colour replaces the prefix rather than appending to it.
    assert_eq!(f.wrap_to_lines("§l§caaaa", 12), vec!["§l§caa", "§caa"]);
}

#[test]
fn java_split_drops_trailing_empty_lines() {
    assert_eq!(java_split_newline("a\nb"), vec!["a", "b"]);
    assert_eq!(java_split_newline("a\n"), vec!["a"]);
    assert_eq!(
        java_split_newline(""),
        vec![""],
        "no separator: the whole string"
    );
    assert_eq!(java_split_newline("\n"), Vec::<String>::new());
    assert_eq!(
        java_split_newline("\na"),
        vec!["", "a"],
        "leading empties survive"
    );
}

#[test]
fn split_string_height_is_nine_per_line() {
    let f = font(false);
    assert_eq!(f.split_string_height("aaaaaa", 22), 18);
    assert_eq!(
        f.split_string_height("\n", 100),
        0,
        "java's split yields no lines"
    );
}

/// Vanilla recurses forever when one glyph is wider than the wrap width.  The first half of this
/// test is a bounded oracle for that failure: its next recursive input is the same UTF-16 slice,
/// so evaluating another frame can never make progress.  Do not call the real recursive Java
/// method here: that would only turn a regression test into a process-level stack overflow.
///
/// The Rust guard instead takes one unit per line, so hostile chat cannot hang or overflow the
/// stack.
#[test]
fn wrapping_terminates_when_a_glyph_is_wider_than_the_wrap_width() {
    let f = font(false);
    let rest = utf16("abc");
    let cut = f.size_units_to_width(&rest, 1);
    assert_eq!(cut, 0, "the first five-pixel glyph does not fit");
    let separator_len = usize::from(matches!(rest[cut], 32 | 10));
    assert_eq!(separator_len, 0);
    assert_eq!(
        &rest[cut + separator_len..],
        rest.as_slice(),
        "vanilla's recursive substring is unchanged, so its next frame cannot progress"
    );

    assert_eq!(f.wrap_to_lines("abc", 1), vec!["a", "b", "c"]);
    assert_eq!(f.wrap_to_lines("abc", 0), vec!["a", "b", "c"]);
}

// ------------------------------------------------------------------ geometry

fn mesh_of(f: &mut FontRenderer, text: &str, shadow: bool) -> TextMesh {
    let mut m = TextMesh::new();
    f.draw_string(&mut m, text, 0.0, 0.0, 0x00FF_FFFF, shadow);
    m
}

#[test]
fn a_default_glyph_quad_matches_vanillas_vertex_maths() {
    let mut f = font(false);
    let m = mesh_of(&mut f, "A", false);
    assert_eq!(m.glyphs.len(), 1);
    let q = m.glyphs[0];
    assert_eq!(q.texture, FontTexture::Ascii);
    // 'A' is cell 65: column 1, row 4 -> texel (8, 32). Width 5, so f = 4.99.
    let (i, j, w) = (8.0f32, 32.0f32, 5.0f32 - 0.01);
    assert_eq!(q.pos[0], [0.0, 0.0]);
    assert_eq!(q.pos[1], [0.0, 7.99]);
    assert_eq!(q.pos[2], [w - 1.0, 0.0]);
    assert_eq!(q.pos[3], [w - 1.0, 7.99]);
    // Vanilla's UVs, except that the top-left starts `ASCII_UV_INSET` inside the cell (X2: no bleed from the glyph above at fractional scales).
    assert_eq!(
        q.uv[0],
        [(i + ASCII_UV_INSET) / 128.0, (j + ASCII_UV_INSET) / 128.0]
    );
    assert_eq!(q.uv[3], [(i + w - 1.0) / 128.0, (j + 7.99) / 128.0]);
    assert_eq!(q.color, 0xFFFF_FFFF, "alpha 0 is forced opaque");
}

/// The top/left UV inset never moves a pixel to another
/// texel where vanilla's pixels are well defined. At every integer pixels-per-font-pixel scale
/// with the quad on a pixel boundary, each pixel centre samples the same texel with vanilla's
/// UVs and with the inset ones, along both axes, for every glyph width, on both sheets. (The
/// only pixels the inset can move are those whose centre lies within 0.01 font pixels of a texel
/// boundary, which is exactly the edge-on-centre case that bled.)
#[test]
fn the_uv_inset_keeps_every_integer_scale_pixel_on_the_same_texel() {
    // One axis of a quad: `len_units` font pixels on screen mapped to `len_texels` texels,
    // starting `inset` texels into the cell (vanilla: 0).
    let same_texels = |len_units: f32, len_texels: f32, inset: f32, scale: u32| {
        let pixels = (len_units * scale as f32).ceil() as u32;
        for k in 0..pixels {
            let along = (k as f32 + 0.5) / scale as f32;
            if along >= len_units {
                continue;
            }
            let t = along / len_units;
            let vanilla = t * len_texels;
            let inset_texel = inset + t * (len_texels - inset);
            assert_eq!(
                vanilla.floor(),
                inset_texel.floor(),
                "scale {scale}, pixel {k} of a {len_units}-unit edge: {vanilla} vs {inset_texel}"
            );
        }
    };
    for scale in 1..=8 {
        // ascii.png: 7.99 units tall over 7.99 texels; `width - 1.01` wide over as many texels.
        same_texels(7.99, 7.99, ASCII_UV_INSET, scale);
        for width in 2..=9 {
            let span = width as f32 - 1.01;
            same_texels(span, span, ASCII_UV_INSET, scale);
        }
        // Unicode pages: 7.99 units tall over 15.98 texels; `f4 / 2` wide over `f4` texels.
        // Only even scales: at an odd one every pixel centre sits on a texel boundary even in
        // vanilla, which is why `ScaledResolution` drops odd factors with the unicode font.
        if scale % 2 == 0 {
            same_texels(7.99, 15.98, UNICODE_UV_INSET, scale);
            for (start, end) in [(0u32, 15u32), (1, 6), (2, 13), (0, 3)] {
                let f4 = (end + 1 - start) as f32 - 0.02;
                same_texels(f4 / 2.0, f4, UNICODE_UV_INSET, scale);
            }
        }
    }
}

#[test]
fn italic_shears_the_quad_into_a_parallelogram() {
    let mut f = font(false);
    let m = mesh_of(&mut f, "§oA", false);
    let q = m.glyphs[0];
    assert_eq!(q.pos[0][0], 1.0, "top-left moves right");
    assert_eq!(q.pos[1][0], -1.0, "bottom-left moves left");
    assert!(
        (q.pos[2][0] - q.pos[3][0] - 2.0).abs() < 1e-4,
        "and the top edge stays 2 ahead of the bottom"
    );
    // The shear is purely positional: an italic quad samples exactly the texels an upright one
    // does, so the left and right edges keep a single u each.
    let upright = mesh_of(&mut f, "A", false).glyphs[0];
    assert_eq!(q.uv, upright.uv);
    assert_eq!(q.uv[0][0], q.uv[1][0]);
    assert_eq!(q.uv[2][0], q.uv[3][0]);
}

#[test]
fn a_unicode_glyph_quad_uses_the_page_texture_and_half_scale_advance() {
    let mut f = font(true);
    let m = mesh_of(&mut f, "a", false);
    assert_eq!(m.glyphs.len(), 1);
    let q = m.glyphs[0];
    assert_eq!(q.texture, FontTexture::UnicodePage(0));
    // 'a' = 0x61: column 1, row 6 -> texel origin (16, 96); start nibble 1, end nibble 6.
    let (f2, f3) = (16.0f32 + 1.0, 96.0f32);
    let f4 = 7.0f32 - 1.0 - 0.02;
    // Vanilla's corner plus the bleed inset (0.02 unicode texels).
    assert_eq!(
        q.uv[0],
        [
            (f2 + UNICODE_UV_INSET) / 256.0,
            (f3 + UNICODE_UV_INSET) / 256.0
        ]
    );
    assert_eq!(
        q.pos[2],
        [f4 / 2.0, 0.0],
        "the glyph is drawn at half width"
    );
    assert_eq!(q.pos[1], [0.0, 7.99]);
}

#[test]
fn a_zero_width_unicode_glyph_emits_nothing() {
    let mut f = font(true);
    let m = mesh_of(&mut f, "d", false);
    assert!(m.glyphs.is_empty(), "glyph_sizes[d] == 0");
}

#[test]
fn bold_draws_the_glyph_twice_one_pixel_apart() {
    let mut f = font(false);
    let m = mesh_of(&mut f, "§lA", false);
    assert_eq!(m.glyphs.len(), 2);
    assert_eq!(m.glyphs[1].pos[0][0] - m.glyphs[0].pos[0][0], 1.0);
    assert_eq!(m.glyphs[0].uv, m.glyphs[1].uv, "same glyph, same texels");
}

#[test]
fn bold_advances_one_extra_pixel() {
    let mut f = font(false);
    let mut m = TextMesh::new();
    let end = f.draw_string(&mut m, "§lAA", 0.0, 0.0, -1, false);
    assert_eq!(end, 12, "two glyphs at 5+1");
    assert_eq!(m.glyphs[2].pos[0][0], 6.0, "the second glyph starts at 6");
}

#[test]
fn strikethrough_and_underline_bars_sit_where_vanilla_puts_them() {
    let mut f = font(false);
    let m = mesh_of(&mut f, "§mA", false);
    assert_eq!(m.bars.len(), 1);
    // FONT_HEIGHT / 2 == 4, and the bar is the pixel row above that line.
    assert_eq!(
        m.bars[0],
        SolidRect {
            x0: 0.0,
            y0: 3.0,
            x1: 5.0,
            y1: 4.0,
            color: 0xFFFF_FFFF
        }
    );

    let m = mesh_of(&mut f, "§nA", false);
    assert_eq!(m.bars.len(), 1);
    assert_eq!(
        m.bars[0],
        SolidRect {
            x0: -1.0,
            y0: 8.0,
            x1: 5.0,
            y1: 9.0,
            color: 0xFFFF_FFFF
        }
    );
}

#[test]
fn the_drop_shadow_is_one_pixel_down_right_and_a_quarter_as_bright() {
    let mut f = font(false);
    let mut m = TextMesh::new();
    let end = f.draw_string(&mut m, "A", 10.0, 20.0, 0x00FF_FFFF, true);
    assert_eq!(m.glyphs.len(), 2, "shadow pass then main pass");
    assert_eq!(m.glyphs[0].pos[0], [11.0, 21.0], "shadow offset (+1, +1)");
    assert_eq!(m.glyphs[1].pos[0], [10.0, 20.0]);
    // 0xFF / 4 = 0x3F, alpha preserved.
    assert_eq!(m.glyphs[0].color, 0xFF3F_3F3F);
    assert_eq!(m.glyphs[1].color, 0xFFFF_FFFF);
    // `drawString` returns max(shadow pass, main pass), and the shadow pass started a pixel
    // right — so a shadowed string reports an end x one past where its ink stops. Vanilla's
    // text field relies on this: it is why the caret sits a pixel clear of the last glyph.
    assert_eq!(end, 16);
    let mut plain = TextMesh::new();
    assert_eq!(
        f.draw_string(&mut plain, "A", 10.0, 20.0, 0x00FF_FFFF, false),
        15
    );
}

#[test]
fn a_colour_code_recolours_from_the_shadow_palette_in_the_shadow_pass() {
    let mut f = font(false);
    let mut m = TextMesh::new();
    f.draw_string(&mut m, "§cA", 0.0, 0.0, -1, true);
    assert_eq!(m.glyphs.len(), 2);
    // §c is red 0xFF5555; its shadow-palette twin is index 12+16.
    assert_eq!(m.glyphs[0].color & 0x00FF_FFFF, 0x3F1515);
    assert_eq!(m.glyphs[1].color & 0x00FF_FFFF, 0xFF5555);
}

/// `resetStyles` runs once, *before* both shadow passes, so a style opened at the end of the
/// string leaks into the start of the second pass. Vanilla does this; so do we.
#[test]
fn styles_leak_from_the_shadow_pass_into_the_main_pass() {
    let mut f = font(false);
    let mut m = TextMesh::new();
    f.draw_string(&mut m, "A§l", 0.0, 0.0, -1, true);
    // Shadow pass: one plain 'A'. Main pass: 'A' drawn twice, because bold is still set.
    assert_eq!(m.glyphs.len(), 3, "1 + 2");
}

#[test]
fn an_unknown_code_resets_every_style_and_turns_the_text_white() {
    let mut f = font(false);
    let mut m = TextMesh::new();
    f.draw_string(&mut m, "§l§zA", 0.0, 0.0, 0x00FF_0000, false);
    assert_eq!(m.glyphs.len(), 1, "§z cleared the bold flag");
    assert_eq!(
        m.glyphs[0].color, 0xFFFF_FFFF,
        "§z falls back to white, not to red"
    );
}

#[test]
fn reset_restores_the_strings_own_colour_not_white() {
    let mut f = font(false);
    let mut m = TextMesh::new();
    f.draw_string(&mut m, "§cA§rA", 0.0, 0.0, 0x00FF_0000, false);
    assert_eq!(m.glyphs[0].color, 0xFFFF_5555);
    assert_eq!(m.glyphs[1].color, 0xFFFF_0000, "back to the caller's red");
}

#[test]
fn alpha_below_four_is_promoted_to_opaque() {
    let mut f = font(false);
    let mut m = TextMesh::new();
    f.draw_string(&mut m, "A", 0.0, 0.0, 0x0300_0000u32 as i32, false);
    assert_eq!(
        m.glyphs[0].color, 0xFF00_0000,
        "the mask is 0xFC000000, not 0xFF000000"
    );
    m.clear();
    // Alpha 4 is left alone, so `drawString`'s alpha test (GREATER 0.1) discards it: nothing is
    // drawn. The first alpha that shows keeps its own value.
    f.draw_string(&mut m, "A", 0.0, 0.0, 0x0400_0000, false);
    assert!(
        m.glyphs.is_empty(),
        "alpha 4 is left alone, then alpha-tested away"
    );
    f.draw_string(&mut m, "A", 0.0, 0.0, 0x1A00_0000, false);
    assert_eq!(m.glyphs[0].color, 0x1A00_0000, "alpha 26 is drawn as it is");
}

/// java: `drawString` enables the alpha test, under the GUI's `alphaFunc(GL_GREATER, 0.1F)`:
/// an alpha byte of 4..=25 shows nothing, in both passes, glyphs and bars alike, while the
/// layout (advance, end state) is the same as a visible draw's.
#[test]
fn drawstrings_alpha_test_hides_text_at_a_tenth_or_less() {
    for alpha in [0u32, 3, 4, 25, 26, 255] {
        let mut f = font(false);
        let mut m = TextMesh::new();
        let color = ((alpha << 24) | 0x00FF_FFFF) as i32;
        let end = f.draw_string(&mut m, "§nA\u{4e00}§mB", 2.0, 3.0, color, true);
        let mut visible = font(false);
        let want = visible.draw_string(&mut TextMesh::new(), "§nA\u{4e00}§mB", 2.0, 3.0, -1, true);
        assert_eq!(end, want, "alpha {alpha}: the advance never changes");
        assert_eq!(
            end_state(&f).0,
            end_state(&visible).0,
            "alpha {alpha}: end x"
        );
        let hidden = (4..=25).contains(&alpha);
        assert_eq!(alpha_test_discards(color), hidden, "alpha {alpha}");
        assert_eq!(m.glyphs.is_empty(), hidden, "alpha {alpha}: glyphs");
        assert_eq!(m.bars.is_empty(), hidden, "alpha {alpha}: bars");
    }
    // `drawSplitString` does not enable the alpha test itself: its text is left to the caller's
    // state, so the renderer draws it.
    let mut f = font(false);
    let mut m = TextMesh::new();
    f.draw_split_string(&mut m, "A", 0, 0, 100, 0x1000_0000);
    assert_eq!(m.glyphs.len(), 1);
}

#[test]
fn obfuscated_text_substitutes_a_same_width_glyph() {
    let mut f = font(false);
    // The synthetic sheet is uniform, so any substitute is legal; what matters is that one
    // glyph still comes out, at the same advance, and that the initial seed is deterministic.
    let mut a = TextMesh::new();
    f.draw_string(&mut a, "§kAAA", 0.0, 0.0, -1, false);

    let mut same_initial_seed = font(false);
    let mut b = TextMesh::new();
    same_initial_seed.draw_string(&mut b, "§kAAA", 0.0, 0.0, -1, false);
    assert_eq!(a, b, "same initial seed, same first scramble");
    assert_eq!(a.glyphs.len(), 3);
    for (i, q) in a.glyphs.iter().enumerate() {
        assert_eq!(q.pos[0][0], (i * 5) as f32, "advance is unchanged");
    }

    let mut c = TextMesh::new();
    f.draw_string(&mut c, "§kAAA", 0.0, 0.0, -1, false);
    assert_ne!(
        a.glyphs[0].uv, c.glyphs[0].uv,
        "the shared RNG advances between draws without any per-frame reseed"
    );
}

#[test]
fn obfuscation_leaves_chars_the_sheet_does_not_have_alone() {
    let mut f = font(false);
    let mut m = TextMesh::new();
    f.draw_string(&mut m, "§k\u{4e00}", 0.0, 0.0, -1, false);
    assert_eq!(m.glyphs.len(), 1);
    assert_eq!(m.glyphs[0].texture, FontTexture::UnicodePage(0x4e));
}

#[test]
fn a_unicode_shadow_is_nudged_half_a_pixel_not_a_whole_one() {
    let mut f = font(true);
    let mut m = TextMesh::new();
    f.draw_string(&mut m, "a", 10.0, 20.0, -1, true);
    assert_eq!(m.glyphs.len(), 2);
    // The shadow pass starts at (11, 21) and is nudged back by 0.5 in both axes, so a unicode
    // glyph's shadow ends up only *half* a pixel off its body instead of a whole one.
    assert_eq!(m.glyphs[0].pos[0], [10.5, 20.5]);
    // The nudge is gated on `&& shadow`, so the main pass is not moved at all.
    assert_eq!(m.glyphs[1].pos[0], [10.0, 20.0]);
}

#[test]
fn the_cursor_advance_truncates_where_the_width_table_rounds() {
    let mut f = font(true);
    let mut m = TextMesh::new();
    // 'c' is byte 0x0E: measured 9 (after the end>7 clamp) but drawn advancing (15-0)/2+1 = 8.5,
    // truncated to 8. This *is* vanilla, and it is why unicode text drifts left of its measure.
    let end = f.draw_string(&mut m, "c", 0.0, 0.0, -1, false);
    assert_eq!(f.string_width("c"), 9);
    assert_eq!(end, 8);
}

#[test]
fn draw_split_string_steps_nine_pixels_per_line() {
    let mut f = font(false);
    let mut m = TextMesh::new();
    f.draw_split_string(&mut m, "aaaa", 0, 100, 12, -1);
    assert_eq!(m.glyphs.len(), 4);
    assert_eq!(m.glyphs[0].pos[0][1], 100.0);
    assert_eq!(m.glyphs[2].pos[0][1], 109.0, "second line is 9 lower");
}

#[test]
fn split_lines_repeat_the_active_color_prefix() {
    let mut f = font(false);
    let mut m = TextMesh::new();
    f.draw_split_string(&mut m, "§caaaa", 0, 100, 12, -1);
    assert_eq!(m.glyphs.len(), 4);
    assert_eq!(m.glyphs[0].color, 0xFFFF_5555);
    assert_eq!(
        m.glyphs[2].pos[0][1], 109.0,
        "the third glyph starts line two"
    );
    assert_eq!(
        m.glyphs[2].color, 0xFFFF_5555,
        "wrapFormattedStringToWidth must prefix the next line with the active color"
    );
}

#[test]
fn split_lines_carry_the_last_color_as_the_next_lines_reset_base() {
    let mut f = font(false);
    let mut m = TextMesh::new();

    // At 40 pixels the spaces produce three lines: `§4aaaa`, `§4bbbb`, and
    // `§4§rcccc`. Vanilla passes the renderer's current `textColor` into each new line,
    // so the reset on line three restores dark red rather than the caller's white.
    f.draw_split_string(&mut m, "§4aaaa bbbb §rcccc", 0, 100, 40, -1);

    assert_eq!(m.glyphs.len(), 12);
    assert_eq!(m.glyphs[0].pos[0][1], 100.0);
    assert_eq!(m.glyphs[4].pos[0][1], 109.0);
    assert_eq!(m.glyphs[8].pos[0][1], 118.0);
    assert_eq!(
        m.glyphs[8].color, 0xFFAA_0000,
        "§r on a continuation line restores the color carried from the prior line"
    );
}

#[test]
fn language_descriptor_drives_bidi_and_visual_order() {
    let mut f = font(false);
    let english = mc_assets::LanguageInfo {
        code: "en_US".into(),
        region: "US".into(),
        name: "English".into(),
        bidirectional: false,
    };
    let hebrew = mc_assets::LanguageInfo {
        code: "he_IL".into(),
        region: "Israel".into(),
        name: "Hebrew".into(),
        bidirectional: true,
    };
    f.set_language_direction(&english);
    assert!(!f.bidi_flag());
    assert_eq!(f.reorder_text("abc אבג!"), "abc אבג!");
    f.set_language_direction(&hebrew);
    assert!(f.bidi_flag());
    assert_eq!(f.reorder_text("abc אבג!"), "abc גבא!");
}

#[test]
fn formatting_lookup_indexes_the_whole_lowercased_string() {
    let mut f = font(false);
    let mut m = TextMesh::new();
    // Java lower-cases the entire string first. U+0130 expands to `i` + U+0307, shifting the
    // index used for the later §A lookup onto the section sign itself, so `A` is not accepted
    // as color `a`. A per-code-unit lower-case mutant incorrectly makes X green here.
    f.draw_string(&mut m, "İ§AX", 0.0, 0.0, -1, false);
    let x = m.glyphs.last().expect("X must render");
    assert_eq!(x.color, 0xFFFF_FFFF);
}

#[test]
fn null_sink_measures_without_allocating_geometry() {
    let mut f = font(false);
    let end = f.draw_string(&mut NullSink, "abc", 0.0, 0.0, -1, false);
    assert_eq!(end, 15);
}

#[test]
fn color_code_lookup_covers_the_palette() {
    let f = font(false);
    assert_eq!(f.color_code('0'), 0x000000);
    assert_eq!(f.color_code('f'), 0xFFFFFF);
    assert_eq!(f.color_code('6'), 0xFFAA00);
}

/// java: `FontRenderer.getColorCode` indexes the exact lowercase lookup string, then indexes the
/// palette with the result. An upper-case code therefore produces `-1` and throws rather than
/// being normalised the way formatting-code parsing is elsewhere.
#[test]
#[should_panic(expected = "not a colour code")]
fn color_code_lookup_rejects_uppercase_like_java() {
    let _ = font(false).color_code('F');
}

// ------------------------------------------------------------------ C2-8 / #8 fast paths

/// A sink that keeps glyphs and bars in one stream, so interleaving order is compared too.
#[derive(Debug, Default, PartialEq)]
struct OrderSink(Vec<TextOp>);

impl TextSink for OrderSink {
    fn glyph(&mut self, quad: GlyphQuad) {
        self.0.push(TextOp::Glyph(quad));
    }
    fn bar(&mut self, rect: SolidRect) {
        self.0.push(TextOp::Bar(rect));
    }
}

/// Strings covering every branch the cached and fast paths could disagree on: plain text,
/// colour and reset codes, every style (bars interleave with glyphs), a trailing `§`, a style
/// set at the very end (the shadow pass inherits it), unicode-page and absent glyphs, the
/// U+0130 index shift, Kelvin `§K`, and a surrogate pair.
const CORPUS: &[&str] = &[
    "",
    "abc",
    "Hello world",
    "\u{a7}cRed \u{a7}lBold\u{a7}r plain",
    "\u{a7}nUnder\u{a7}mStrike\u{a7}oItalic",
    "\u{a7}6\u{a7}lGold bold \u{a7}9blue",
    "Text\u{a7}l",
    "trailing\u{a7}",
    "\u{a7}",
    "\u{a7}\u{a7}x",
    "\u{a7}C\u{a7}LUPPER",
    "\u{4e00}\u{a7}ab\u{4e00}",
    "\u{2714} \u{a7}a\u{2714}\u{a7}c\u{2718}",
    "\u{130}\u{a7}AX",
    "I\u{a7}AX",
    "\u{3a3}\u{a7}e\u{3a3}",
    "\u{a7}\u{212a}x",
    "x\u{1F600}\u{a7}by",
    "\u{a7}7[\u{a7}c\u{272b}\u{a7}7] \u{a7}cPlayer\u{a7}f: hi",
];

fn end_state(f: &FontRenderer) -> (u32, u32, u32, u32, i32, [bool; 5]) {
    (
        f.pos_x.to_bits(),
        f.pos_y.to_bits(),
        f.base_color,
        f.color,
        f.text_color,
        [
            f.random_style,
            f.bold_style,
            f.italic_style,
            f.underline_style,
            f.strikethrough_style,
        ],
    )
}

#[test]
fn streamed_string_width_matches_the_unit_walk() {
    for unicode in [false, true] {
        let f = font(unicode);
        for text in CORPUS {
            assert_eq!(
                f.string_width(text),
                f.string_width_units(&utf16(text)),
                "{text:?} unicode={unicode}"
            );
        }
    }
}

/// Measuring the parts of a string equals measuring the joined string, including a `§` that
/// ends one part and formats the first unit of the next (the sidebar's `name + ": §c" + points`).
#[test]
fn width_of_parts_equals_width_of_the_concatenation() {
    for unicode in [false, true] {
        let f = font(unicode);
        for a in CORPUS {
            for b in ["", ": ", "\u{a7}c12", "l bold", "\u{a7}", "r", "\u{4e00}"] {
                let joined = format!("{a}{b}");
                assert_eq!(
                    f.string_width_of_parts(&[*a, b, ": \u{a7}c-7"]),
                    f.string_width(&format!("{joined}: \u{a7}c-7")),
                    "{a:?} + {b:?} unicode={unicode}"
                );
            }
        }
    }
}

/// The C2-8 fast path decides `toLowerCase().charAt(i + 1)` from the unit itself. Check it
/// against the whole-string conversion for **every** BMP unit, both right after the marker and
/// with the unit repeated before it (which exercises the context-sensitive final-sigma rule).
#[test]
fn per_unit_marker_lookup_matches_whole_string_lowercase_for_every_bmp_unit() {
    for unit in 0..=u16::MAX {
        if (0xD800..=0xDFFF).contains(&unit) {
            continue;
        }
        for text in [vec![0xA7, unit], vec![unit, 0xA7, unit, 0x20]] {
            let marker = text.iter().position(|&u| u == 0xA7).expect("marker");
            let whole: Vec<u16> = String::from_utf16_lossy(&text)
                .to_lowercase()
                .encode_utf16()
                .collect();
            let expected = format_code_index(whole[marker + 1]);
            if unit == 0x0130 {
                assert!(
                    !lowercase_is_per_unit(&text),
                    "U+0130 must take the slow path"
                );
                continue;
            }
            assert!(lowercase_is_per_unit(&text), "unit {unit:#06x}");
            assert_eq!(
                marker_code_index(text[marker + 1]),
                expected,
                "unit {unit:#06x} in {text:x?}"
            );
        }
    }
    assert!(
        !lowercase_is_per_unit(&utf16("x\u{1F600}")),
        "surrogates take the slow path"
    );
}

/// Draw every corpus string uncached and through the cache (first sighting, admission, hit) on
/// two identically seeded renderers: the primitives, their order, the advance and every
/// renderer field left behind must be identical on all three cached draws. Strings that roll
/// the obfuscation stream (Kelvin-sign `§K`) must stay uncached and in lockstep.
#[test]
fn cached_draws_replay_bit_identical_primitives_and_state() {
    for unicode in [false, true] {
        for shadow in [false, true] {
            let mut reference = font(unicode);
            reference.set_run_cache_enabled(false);
            let mut cached = font(unicode);
            let mut cacheable = 0u64;
            for (n, text) in CORPUS.iter().enumerate() {
                let (x, y) = (3.0 + n as f32, 7.5);
                let color = if n % 2 == 0 { -1 } else { 0x20FF_FFFF };
                let mut rolled = false;
                for pass in 0..3 {
                    let seed = reference.rng.raw_seed();
                    let mut expected = OrderSink::default();
                    let advance = reference.draw_string(&mut expected, text, x, y, color, shadow);
                    rolled |= reference.rng.raw_seed() != seed;
                    let mut actual = OrderSink::default();
                    let got = cached.draw_string(&mut actual, text, x, y, color, shadow);
                    assert_eq!(actual, expected, "{text:?} pass {pass} shadow={shadow}");
                    assert_eq!(got, advance, "{text:?} pass {pass}");
                    assert_eq!(end_state(&cached), end_state(&reference), "{text:?} {pass}");
                    assert_eq!(cached.rng.raw_seed(), reference.rng.raw_seed());
                }
                cacheable += u64::from(!rolled);
            }
            let stats = cached.run_cache_stats();
            assert!(
                cacheable + 1 >= CORPUS.len() as u64,
                "only the §K string rolls"
            );
            assert_eq!(
                stats.hits, cacheable,
                "third draw of each cacheable string hits"
            );
            assert_eq!(stats.entries as u64, cacheable);
        }
    }
}

/// `§k` draws consume `fontRandom`, so they must never be served from the cache: with the cache
/// on, a sequence of obfuscated draws stays in lockstep with the uncached renderer's stream.
#[test]
fn obfuscated_draws_are_never_cached_and_keep_advancing_the_stream() {
    let data =
        || FontData::from_parts(&synthetic_ascii_png(), &synthetic_glyph_sizes()).expect("font");
    let mut reference = FontRenderer::new_with_seed(data(), false, false, 0x5EED);
    reference.set_run_cache_enabled(false);
    let mut cached = FontRenderer::new_with_seed(data(), false, false, 0x5EED);
    for _ in 0..6 {
        for text in ["\u{a7}kAAA", "a\u{a7}kb c", "\u{a7}cx"] {
            let mut expected = OrderSink::default();
            let mut actual = OrderSink::default();
            reference.draw_string(&mut expected, text, 1.0, 2.0, -1, true);
            cached.draw_string(&mut actual, text, 1.0, 2.0, -1, true);
            assert_eq!(actual, expected, "{text:?}");
            assert_eq!(cached.rng.raw_seed(), reference.rng.raw_seed());
        }
    }
    assert_eq!(
        cached.run_cache_stats().entries,
        1,
        "only the §c string is stored"
    );
}

/// A hit must replay the draw's `textColor` write — and only when the draw made one — because
/// `drawSplitString` and the chat "…" marker read it back.
#[test]
fn a_hit_replays_the_text_color_write_only_when_the_draw_made_one() {
    let mut f = font(false);
    for _ in 0..3 {
        f.draw_string(&mut NullSink, "\u{a7}cA", 0.0, 0.0, -1, false);
        f.draw_string(&mut NullSink, "plain", 0.0, 9.0, -1, false);
    }
    assert!(f.run_cache_stats().hits >= 2);
    f.draw_string(&mut NullSink, "\u{a7}9B", 0.0, 18.0, -1, false);
    assert_eq!(f.text_color(), f.colors()[9]);
    f.draw_string(&mut NullSink, "\u{a7}cA", 0.0, 0.0, -1, false);
    assert_eq!(
        f.text_color(),
        f.colors()[12],
        "cached colour code rewrites textColor"
    );
    f.draw_string(&mut NullSink, "plain", 0.0, 9.0, -1, false);
    assert_eq!(
        f.text_color(),
        f.colors()[12],
        "a run without codes leaves it alone"
    );
}

/// `sync_anaglyph_with_gpu` calls `set_anaglyph` every frame; an unchanged flag must keep the
/// cache, a real change must drop it, and the output after the change must use the new palette.
#[test]
fn only_real_configuration_changes_invalidate_the_run_cache() {
    let mut f = font(false);
    let mut reference = font(false);
    reference.set_run_cache_enabled(false);
    for _ in 0..2 {
        f.draw_string(&mut NullSink, "\u{a7}6gold", 0.0, 0.0, -1, true);
    }
    assert_eq!(f.run_cache_stats().entries, 1);
    f.set_anaglyph(false);
    f.set_unicode_flag(false);
    f.set_bidi_flag(false);
    assert_eq!(
        f.run_cache_stats().entries,
        1,
        "unchanged flags keep the cache"
    );

    f.set_anaglyph(true);
    reference.set_anaglyph(true);
    assert_eq!(
        f.run_cache_stats().entries,
        0,
        "palette change drops the cache"
    );
    let mut expected = OrderSink::default();
    let mut actual = OrderSink::default();
    reference.draw_string(&mut expected, "\u{a7}6gold", 0.0, 0.0, -1, true);
    f.draw_string(&mut actual, "\u{a7}6gold", 0.0, 0.0, -1, true);
    assert_eq!(actual, expected);

    for _ in 0..2 {
        f.draw_string(&mut NullSink, "\u{a7}6gold", 0.0, 0.0, -1, true);
    }
    f.set_unicode_flag(true);
    assert_eq!(
        f.run_cache_stats().entries,
        0,
        "unicode change drops the cache"
    );
}

/// The production sink: a `DrawList` built through the cache (including under a transform) is
/// equal — vertices, depths and batches — to one built without it.
#[test]
fn draw_list_output_is_identical_with_and_without_the_run_cache() {
    use crate::widget::{DrawList, Transform};
    let draw = |f: &mut FontRenderer, d: &mut DrawList| {
        for (n, text) in CORPUS.iter().enumerate() {
            f.draw_string(d, text, 2.0, n as f32 * 9.0, 0x20FF_FFFF, false);
            d.rect(0, 0, 4, 4, 0x5000_0000);
            d.with_transform(Transform::translate(10.0, 5.0).then_scale(0.5), |d| {
                f.draw_string_with_shadow(d, text, 1.0, n as f32 * 9.0, -1);
            });
        }
    };
    // Both renderers draw every frame, so strings that roll `fontRandom` (the Kelvin-sign
    // `§K` corpus entry) stay in lockstep too.
    let mut reference = font(false);
    reference.set_run_cache_enabled(false);
    let mut cached = font(false);
    for frame in 0..3 {
        let mut expected = DrawList::new();
        draw(&mut reference, &mut expected);
        let mut actual = DrawList::new();
        draw(&mut cached, &mut actual);
        assert_eq!(actual, expected, "frame {frame}");
        assert_eq!(
            actual.vertex_bytes(),
            expected.vertex_bytes(),
            "frame {frame}"
        );
    }
    assert!(cached.run_cache_stats().hits > 0);
}

/// Run keys are a fixed function of server-chosen text and position, so the tables must re-hash
/// them with a seed: used as their own hash, their bucket and control bits would be predictable
/// and a server could pre-search text whose keys all probe one group (phase 4b review S3).
#[test]
fn run_cache_tables_rehash_their_keys_with_a_per_table_seed() {
    use std::hash::BuildHasher;
    let key = run_key_hash("§cRed§r name", -20.0, 0.0, 0x20FF_FFFF, false);
    let first = RunMap::<u64>::default();
    let second = RunMap::<u64>::default();
    assert_ne!(first.hasher().hash_one(key), key);
    assert_ne!(first.hasher().hash_one(key), second.hasher().hash_one(key));
}

/// Not a correctness test: a timing report for the glyph-run cache on a BedWars-like HUD text
/// load (15 sidebar rows, 10 shadowed chat lines). Run on demand:
/// `cargo test -p mc-ui --lib run_cache_timing_report -- --ignored --nocapture`.
#[test]
#[ignore = "timing report; run on demand"]
fn run_cache_timing_report() {
    use crate::widget::DrawList;
    let rows: Vec<String> = (0..15)
        .map(|i| format!("\u{a7}c[R] \u{a7}fPlayer{i}\u{a7}7: \u{a7}c{}", i * 3))
        .collect();
    let chat: Vec<String> = (0..10)
        .map(|i| {
            format!(
                "\u{a7}r\u{a7}7[\u{a7}c\u{272b}\u{a7}7] \u{a7}bSomeone{i}\u{a7}f: gg that was close {i}"
            )
        })
        .collect();
    fn frame(f: &mut FontRenderer, sink: &mut dyn TextSink, rows: &[String], chat: &[String]) {
        for (i, row) in rows.iter().enumerate() {
            f.draw_string(sink, row, 700.0, 100.0 + i as f32 * 9.0, 0x20FF_FFFF, false);
        }
        for (i, line) in chat.iter().enumerate() {
            f.draw_string_with_shadow(sink, line, 0.0, 300.0 - i as f32 * 9.0, -1);
        }
    }
    let frames = 3_000u32;
    for enabled in [false, true] {
        let mut f = font(false);
        f.set_run_cache_enabled(enabled);
        let mut d = DrawList::new();
        for _ in 0..50 {
            d.clear();
            frame(&mut f, &mut d, &rows, &chat);
        }
        let start = std::time::Instant::now();
        for _ in 0..frames {
            d.clear();
            frame(&mut f, &mut d, &rows, &chat);
        }
        let into_list = start.elapsed() / frames;
        let start = std::time::Instant::now();
        for _ in 0..frames {
            frame(&mut f, &mut NullSink, &rows, &chat);
        }
        let font_only = start.elapsed() / frames;
        eprintln!(
            "run cache {}: {into_list:?}/frame into a DrawList ({} vertices), {font_only:?}/frame \
             font-only; stats {:?}",
            if enabled { "ON " } else { "OFF" },
            d.vertices().len(),
            f.run_cache_stats()
        );
    }
}

// ------------------------------------------------------------------ alpha test and symbols
// "symbols in the chat dont render with a background shadow like text". `U+4E00` stands in for
// every symbol: `ascii.png` does not have it, so it is drawn from a unicode page.

/// A sink that tags every primitive with the pass it was drawn in.
#[derive(Debug, Default)]
struct PassSink {
    shadow: bool,
    glyphs: Vec<(bool, GlyphQuad)>,
    bars: Vec<(bool, SolidRect)>,
}

impl TextSink for PassSink {
    fn shadow_pass(&mut self, shadow: bool) {
        self.shadow = shadow;
    }
    fn glyph(&mut self, quad: GlyphQuad) {
        self.glyphs.push((self.shadow, quad));
    }
    fn bar(&mut self, rect: SolidRect) {
        self.bars.push((self.shadow, rect));
    }
}

/// How far a symbol's shadow sits from the symbol, in GUI pixels: vanilla's 0 (under it), or a
/// letter's 1 with symbol shadows on.
fn symbol_shadow_offset(on: bool) -> f32 {
    if on {
        1.0
    } else {
        0.0
    }
}

/// java: FontRenderer.java:479-494. The shadow pass starts at (+1, +1) and a glyph `ascii.png`
/// does not have is moved back by `f1 = 1.0` (unicode font off): its shadow is drawn exactly
/// under it, in the shadow colour, where the glyph covers it.
#[test]
fn vanilla_draws_a_symbols_shadow_exactly_under_the_symbol() {
    let mut f = font(false);
    assert!(!f.symbol_shadows(), "vanilla is the default");
    let mut m = TextMesh::new();
    let end = f.draw_string(&mut m, "A\u{4e00}", 10.0, 20.0, -1, true);
    assert_eq!(
        m.glyphs.len(),
        4,
        "shadow pass A + symbol, then main pass A + symbol"
    );
    assert_eq!(
        m.glyphs[0].pos[0],
        [11.0, 21.0],
        "a letter's shadow is (+1, +1)"
    );
    assert_eq!(m.glyphs[3].pos[0], [15.0, 20.0]);
    assert_eq!(
        m.glyphs[1].pos, m.glyphs[3].pos,
        "the symbol's shadow is under it"
    );
    assert_eq!(m.glyphs[1].uv, m.glyphs[3].uv);
    assert_eq!(m.glyphs[1].texture, FontTexture::UnicodePage(0x4e));
    assert_eq!(
        m.glyphs[1].color, 0xFF3F_3F3F,
        "drawn in the shadow colour all the same"
    );
    // 'A' advances 5, the full-cell symbol (16 - 0) / 2 + 1 = 9; the shadow pass ends one
    // pixel further right.
    assert_eq!(end, 25);
}

#[test]
fn symbol_shadows_move_only_the_symbols_shadow() {
    let mut vanilla = font(false);
    let mut want = TextMesh::new();
    let want_end = vanilla.draw_string(&mut want, "A\u{4e00}", 10.0, 20.0, -1, true);
    let mut f = font(false);
    f.set_symbol_shadows(true);
    let mut m = TextMesh::new();
    let end = f.draw_string(&mut m, "A\u{4e00}", 10.0, 20.0, -1, true);
    assert_eq!(end, want_end, "the advance is unchanged");
    assert_eq!(m.glyphs.len(), 4);
    assert_eq!(
        m.glyphs[0], want.glyphs[0],
        "the letter's shadow is unchanged"
    );
    assert_eq!(
        m.glyphs[2..],
        want.glyphs[2..],
        "the main pass is unchanged"
    );
    assert_eq!(
        m.glyphs[1].pos[0],
        [16.0, 21.0],
        "one pixel down and right of the symbol"
    );
    assert_eq!(m.glyphs[1].uv, want.glyphs[1].uv, "same texels");
    assert_eq!(m.glyphs[1].color, 0xFF3F_3F3F, "same colour");
}

/// A bold symbol is drawn twice, one pixel apart (`f1` with the unicode font off); both shadow
/// copies move, so bold symbols shadow like bold letters.
#[test]
fn both_copies_of_a_bold_symbol_cast_the_shadow() {
    for on in [false, true] {
        let mut f = font(false);
        f.set_symbol_shadows(on);
        let mut m = TextMesh::new();
        f.draw_string(&mut m, "\u{a7}l\u{4e00}", 10.0, 20.0, -1, true);
        assert_eq!(
            m.glyphs.len(),
            4,
            "on={on}: two shadow copies, two main copies"
        );
        let o = symbol_shadow_offset(on);
        assert_eq!(m.glyphs[2].pos[0], [10.0, 20.0]);
        assert_eq!(m.glyphs[3].pos[0], [11.0, 20.0]);
        assert_eq!(m.glyphs[0].pos[0], [10.0 + o, 20.0 + o], "on={on}");
        assert_eq!(m.glyphs[1].pos[0], [11.0 + o, 20.0 + o], "on={on}");
    }
}

/// With the unicode font on, every glyph's shadow is vanilla's half pixel either way.
#[test]
fn the_unicode_font_keeps_vanillas_half_pixel_shadow_either_way() {
    let mut want = TextMesh::new();
    font(true).draw_string(&mut want, "a\u{4e00}", 10.0, 20.0, -1, true);
    assert_eq!(want.glyphs[0].pos[0], [10.5, 20.5]);
    let mut f = font(true);
    f.set_symbol_shadows(true);
    let mut m = TextMesh::new();
    f.draw_string(&mut m, "a\u{4e00}", 10.0, 20.0, -1, true);
    assert_eq!(m, want);
}

/// Over the whole corpus (styles, colours, bars, absent glyphs, surrogates) and both unicode
/// flags: symbol shadows change only where the shadow pass draws a symbol, by exactly one pixel;
/// the main pass, the bars, the letters' shadows and the advance are identical.
#[test]
fn symbol_shadows_differ_from_vanilla_only_in_the_symbols_shadow() {
    let mut texts: Vec<&str> = CORPUS.to_vec();
    texts.extend([
        "\u{a7}c\u{4e00}\u{a7}lA\u{4e00}\u{a7}r \u{a7}n\u{4e00}\u{a7}m\u{4e00}",
        "\u{a7}o\u{4e00}a\u{4e00}",
        "\u{4e00}\u{a7}l",
    ]);
    let mut moved = 0;
    for unicode in [false, true] {
        for text in &texts {
            let mut vanilla = font(unicode);
            let mut want = PassSink::default();
            let want_end = vanilla.draw_string(&mut want, text, 3.0, 7.0, 0x00FF_AA00, true);
            let mut f = font(unicode);
            f.set_symbol_shadows(true);
            let mut got = PassSink::default();
            let end = f.draw_string(&mut got, text, 3.0, 7.0, 0x00FF_AA00, true);
            let ctx = format!("{text:?} unicode={unicode}");
            assert_eq!(end, want_end, "{ctx}: advance");
            assert_eq!(got.bars, want.bars, "{ctx}: bars");
            assert_eq!(got.glyphs.len(), want.glyphs.len(), "{ctx}: glyph count");
            for ((shadow, g), (_, w)) in got.glyphs.iter().zip(&want.glyphs) {
                let symbol = matches!(g.texture, FontTexture::UnicodePage(_)) && !unicode;
                let moves = *shadow && symbol;
                if !moves {
                    assert_eq!(g, w, "{ctx}: only a symbol's shadow may move");
                    continue;
                }
                moved += 1;
                assert_eq!(
                    (g.uv, g.color, g.texture),
                    (w.uv, w.color, w.texture),
                    "{ctx}"
                );
                for (p, q) in g.pos.iter().zip(&w.pos) {
                    assert!(
                        (p[0] - q[0] - 1.0).abs() < 1e-4 && (p[1] - q[1] - 1.0).abs() < 1e-4,
                        "{ctx}: {p:?} vs vanilla {q:?}"
                    );
                }
            }
        }
    }
    assert!(
        moved >= 10,
        "the corpus has symbols in the shadow pass ({moved})"
    );
}

/// The client sets the flag every frame: an unchanged flag keeps the run cache and the layout
/// generation, a real change drops both, and nothing stale is replayed afterwards.
#[test]
fn a_symbol_shadows_change_drops_the_run_cache_and_moves_the_layout_generation() {
    let mut f = font(false);
    for _ in 0..2 {
        f.draw_string(&mut NullSink, "\u{4e00}", 0.0, 0.0, -1, true);
    }
    assert_eq!(f.run_cache_stats().entries, 1);
    let generation = f.layout_generation();
    f.set_symbol_shadows(false);
    assert_eq!(
        f.run_cache_stats().entries,
        1,
        "an unchanged flag keeps the cache"
    );
    assert_eq!(f.layout_generation(), generation);

    f.set_symbol_shadows(true);
    assert_eq!(f.run_cache_stats().entries, 0, "a change drops the cache");
    assert_ne!(f.layout_generation(), generation);
    for pass in 0..3 {
        let mut m = TextMesh::new();
        f.draw_string(&mut m, "\u{4e00}", 0.0, 0.0, -1, true);
        assert_eq!(
            m.glyphs[0].pos[0],
            [1.0, 1.0],
            "pass {pass}: the new shadow"
        );
    }
    assert!(
        f.run_cache_stats().hits > 0,
        "and the new runs are cached again"
    );
}
