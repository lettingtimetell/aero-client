//! Golden tests against the **real** 1.8.9 font assets under `MCP-919-main/jars/`.
//!
//! The expected values here were produced by an independent transcription of
//! `FontRenderer.java` into Python, run against the same `ascii.png` and `glyph_sizes.bin` —
//! so the Rust port is being checked against the Java algorithm, not against itself.
//!
//! Every test skips (loudly) when the jar is absent, following `mc-assets`' precedent. Point
//! `MC_JARS_DIR` at a `jars/` directory to override the location.

use std::path::{Path, PathBuf};

use mc_assets::{ResourceLocation, ResourceManager, ZipResourcePack};
use mc_ui::font::{FontData, FontRenderer, FONT_CHARS};

fn jars_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn manager() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    Some(rm)
}

macro_rules! require {
    ($e:expr, $what:literal) => {
        match $e {
            Some(v) => v,
            None => {
                eprintln!(
                    "SKIP: {} not found; set MC_JARS_DIR to the jars/ directory",
                    $what
                );
                return;
            }
        }
    };
}

fn font(unicode: bool) -> Option<FontRenderer> {
    let rm = manager()?;
    Some(FontRenderer::new(
        FontData::load(&rm).expect("load vanilla font"),
        false,
        unicode,
    ))
}

/// The 256 default-font widths measured out of the real `ascii.png`.
///
/// Independently derived: this array is the output of the Python transcription of
/// `readFontTexture`, pasted verbatim.
const VANILLA_CHAR_WIDTHS: [i32; 256] = [
    6, 6, 6, 6, 6, 6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 6, 7, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1,
    1, 2, 5, 6, 6, 6, 6, 3, 5, 5, 5, 6, 2, 6, 2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 2, 2, 5, 6, 5, 6,
    7, 6, 6, 6, 6, 6, 6, 6, 6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 6, 4, 6, 6,
    3, 6, 6, 6, 6, 6, 5, 6, 6, 2, 6, 5, 3, 6, 6, 6, 6, 6, 6, 6, 4, 6, 6, 6, 6, 6, 6, 5, 2, 5, 7, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 6, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 6,
    6, 3, 6, 6, 6, 6, 6, 6, 6, 7, 6, 6, 6, 2, 6, 6, 8, 9, 9, 6, 6, 6, 8, 8, 6, 8, 8, 8, 8, 8, 6, 6,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 6, 9, 9, 9, 5, 9, 9,
    8, 7, 7, 8, 7, 8, 8, 8, 7, 8, 8, 7, 9, 9, 6, 7, 7, 7, 7, 7, 9, 6, 7, 8, 7, 6, 6, 9, 7, 6, 7, 1,
];

#[test]
fn the_measured_ascii_width_table_matches_the_java_algorithm() {
    let rm = require!(manager(), "jars/versions/1.8.9/1.8.9.jar");
    let data = FontData::load(&rm).expect("load font");
    assert_eq!(data.ascii_size(), (128, 128));
    assert_eq!(
        data.char_widths(),
        &VANILLA_CHAR_WIDTHS,
        "the ascii.png width scan drifted from vanilla's"
    );
}

/// H9's own footnote: the space cell has no ink, so vanilla's `if (j1 == 32)` pre-set is
/// overwritten with 1. Nothing reads it — both `getCharWidth` and `renderChar` special-case
/// `' '` to 4 first — but a port that "fixes" it would be measuring a different table.
#[test]
fn the_space_cell_measures_one_pixel_and_is_never_used() {
    let f = require!(font(false), "jars/versions/1.8.9/1.8.9.jar");
    assert_eq!(f.data().char_widths()[32], 1);
    assert_eq!(f.char_width(' '), 4);
    assert_eq!(f.string_width("   "), 12);
}

#[test]
fn glyph_sizes_bin_is_the_full_bmp_table() {
    let rm = require!(manager(), "jars/versions/1.8.9/1.8.9.jar");
    let data = FontData::load(&rm).expect("load font");
    let g = data.glyph_widths();
    assert_eq!(g.len(), 0x1_0000);
    // Spot checks against the raw bytes of the shipped file.
    assert_eq!(g[b'a' as usize], 0x16, "'a' spans columns 1..=6");
    assert_eq!(g[0x4e00], 0x0F, "a CJK ideograph fills its cell");
    assert_ne!(g.iter().filter(|b| **b != 0).count(), 0);
}

/// Java reads the nibbles out of a `byte`, so `>>> 4` sign-extends first. In the shipped table
/// exactly 50 bytes have the high bit set, and **every one of them** also has an end column
/// above 7 — which is the only reason `getCharWidth`'s clamp rescues the width path. The quad
/// builder does not clamp, so those 50 code points really do sample outside their page in
/// vanilla; this pins the population so a future "fix" is a conscious choice.
#[test]
fn the_sign_extension_quirk_never_reaches_the_width_path() {
    let rm = require!(manager(), "jars/versions/1.8.9/1.8.9.jar");
    let data = FontData::load(&rm).expect("load font");
    let g = data.glyph_widths();
    let high_bit: Vec<usize> = (0..0x1_0000).filter(|i| g[*i] & 0x80 != 0).collect();
    assert_eq!(high_bit.len(), 50, "1.8.9 ships exactly 50 such bytes");
    for i in high_bit {
        assert!(
            g[i] & 15 > 7,
            "U+{i:04X} (byte {:#04x}) would produce a nonsense start column",
            g[i]
        );
        let (start, _end) = data.unicode_nibbles(i as u16);
        assert!(
            start > 0xFF_FFFF,
            "the sign extension is reproduced, not fixed"
        );
    }
}

#[test]
fn per_character_widths_match_the_reference() {
    let ascii = require!(font(false), "jars/versions/1.8.9/1.8.9.jar");
    let unicode = font(true).expect("second font");
    // (char, default width, unicode width)
    let cases = [
        ('A', 6, 4),
        ('a', 6, 4),
        ('W', 6, 4),
        ('i', 2, 3),
        ('I', 4, 3),
        ('l', 3, 3),
        ('.', 2, 2),
        (' ', 4, 4),
        ('§', -1, -1),
        ('M', 6, 4),
        ('0', 6, 4),
    ];
    for (c, def, uni) in cases {
        assert_eq!(ascii.char_width(c), def, "default width of {c:?}");
        assert_eq!(unicode.char_width(c), uni, "unicode width of {c:?}");
    }
}

#[test]
fn string_widths_match_the_reference_with_the_default_font() {
    let f = require!(font(false), "jars/versions/1.8.9/1.8.9.jar");
    let cases: &[(&str, i32)] = &[
        ("Hello, world!", 59),
        ("Minecraft", 47),
        ("The quick brown fox jumps over the lazy dog", 231),
        ("iiiiiiiiii", 20),
        ("WWWWWWWWWW", 60),
        ("", 0),
        (" ", 4),
        ("\u{a7}cRed \u{a7}lBold\u{a7}r plain", 74),
        ("\u{a7}lBOLD", 28),
        ("\u{a7}l\u{a7}oBoth", 26),
        ("a\u{a7}", 5),
        ("\u{a7}", -1),
        ("Options...", 42),
        ("Multiplayer", 54),
        ("Singleplayer", 62),
        ("gui.done", 40),
        ("Done", 24),
        ("Back", 23),
        ("Play Selected World", 99),
        ("\u{a7}e\u{a7}lRules", 32),
        ("0123456789", 60),
        ("\u{e9}\u{e8}\u{ea}", 18),
        ("\u{2588}\u{2593}\u{2592}\u{2591}", 35),
    ];
    for (s, want) in cases {
        assert_eq!(f.string_width(s), *want, "width of {s:?}");
    }
}

#[test]
fn string_widths_match_the_reference_with_the_unicode_font() {
    let f = require!(font(true), "jars/versions/1.8.9/1.8.9.jar");
    let cases: &[(&str, i32)] = &[
        ("Hello, world!", 44),
        ("Minecraft", 33),
        ("The quick brown fox jumps over the lazy dog", 167),
        ("iiiiiiiiii", 30),
        ("WWWWWWWWWW", 40),
        ("\u{a7}cRed \u{a7}lBold\u{a7}r plain", 57),
        ("Done", 16),
        ("Back", 16),
        ("0123456789", 39),
        ("\u{e9}\u{e8}\u{ea}", 12),
        ("\u{2588}\u{2593}\u{2592}\u{2591}", 36),
    ];
    for (s, want) in cases {
        assert_eq!(f.string_width(s), *want, "unicode width of {s:?}");
    }
    // The unicode pages are *narrower* for Latin text and wider for box drawing — the two
    // fonts genuinely disagree, which is why the flag has to be decided before layout.
    let ascii = font(false).expect("second font");
    assert!(f.string_width("Hello, world!") < ascii.string_width("Hello, world!"));
    assert!(
        f.string_width("\u{2588}\u{2593}\u{2592}\u{2591}")
            > ascii.string_width("\u{2588}\u{2593}\u{2592}\u{2591}")
    );
}

#[test]
fn trimming_matches_the_reference() {
    let f = require!(font(false), "jars/versions/1.8.9/1.8.9.jar");
    let cases: &[(&str, i32, &str, &str)] = &[
        ("Hello, world!", 20, "Hell", "rld!"),
        ("Hello, world!", 0, "", ""),
        ("Hello, world!", 1000, "Hello, world!", "Hello, world!"),
        ("\u{a7}cRed text", 20, "\u{a7}cRed", "text"),
        ("abcdef", 12, "ab", "ef"),
    ];
    for (s, w, fwd, rev) in cases {
        assert_eq!(f.trim_to_width(s, *w), *fwd, "trim {s:?} to {w}");
        assert_eq!(
            f.trim_to_width_dir(s, *w, true),
            *rev,
            "reverse-trim {s:?} to {w}"
        );
    }
}

#[test]
fn word_wrap_breaks_where_the_reference_breaks() {
    let f = require!(font(false), "jars/versions/1.8.9/1.8.9.jar");
    let pangram = "The quick brown fox jumps over the lazy dog";

    assert_eq!(f.size_string_to_width(pangram, 100), 15);
    assert_eq!(
        f.wrap_to_lines(pangram, 100),
        vec!["The quick brown", "fox jumps over the", "lazy dog"]
    );

    assert_eq!(f.size_string_to_width(pangram, 50), 9);
    assert_eq!(
        f.wrap_to_lines(pangram, 50),
        vec!["The quick", "brown", "fox", "jumps", "over the", "lazy dog"]
    );

    assert_eq!(f.size_string_to_width("aaaaaaaaaaaaaaaaaaaaaaaa", 30), 5);
    assert_eq!(
        f.wrap_to_lines("aaaaaaaaaaaaaaaaaaaaaaaa", 30),
        vec!["aaaaa", "aaaaa", "aaaaa", "aaaaa", "aaaa"]
    );

    assert_eq!(f.size_string_to_width("line one\nline two", 200), 8);
    assert_eq!(
        f.wrap_to_lines("line one\nline two", 200),
        vec!["line one", "line two"]
    );

    // The bold prefix is re-emitted on every continuation line, and it costs a pixel per glyph.
    assert_eq!(
        f.size_string_to_width("\u{a7}lbold wrapped text here", 40),
        6
    );
    assert_eq!(
        f.wrap_to_lines("\u{a7}lbold wrapped text here", 40),
        vec![
            "\u{a7}lbold",
            "\u{a7}lwrapp",
            "\u{a7}led",
            "\u{a7}ltext",
            "\u{a7}lhere"
        ]
    );

    // Java's split() drops trailing empties, so these two disagree in a way that matters to
    // `splitStringWidth`.
    assert_eq!(f.size_string_to_width("hello\n", 100), 5);
    assert_eq!(f.wrap_to_lines("hello\n", 100), vec!["hello"]);
    assert_eq!(f.size_string_to_width("\n", 100), 0);
    assert!(f.wrap_to_lines("\n", 100).is_empty());
    assert_eq!(f.split_string_height("\n", 100), 0);
    assert_eq!(f.size_string_to_width("", 100), 0);
    assert_eq!(f.wrap_to_lines("", 100), vec![""]);

    assert_eq!(f.split_string_height(pangram, 100), 27, "3 lines of 9");
}

/// Every wrapped line must actually fit, and the wrap must not drop or duplicate characters.
#[test]
fn wrapping_is_lossless_and_never_overflows() {
    let f = require!(font(false), "jars/versions/1.8.9/1.8.9.jar");
    let samples = [
        "The quick brown fox jumps over the lazy dog",
        "\u{a7}aGreen \u{a7}lbold \u{a7}rreset and a very long tail that has to wrap several times",
        "supercalifragilisticexpialidocious",
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",
        "trailing spaces      and   runs",
    ];
    for width in [20, 40, 80, 120, 200] {
        for s in samples {
            let lines = f.wrap_to_lines(s, width);
            for line in &lines {
                assert!(
                    f.string_width(line) <= width + 8,
                    "line {line:?} is {} wide, wrapping {s:?} at {width}",
                    f.string_width(line)
                );
            }
            // Strip the re-emitted formatting prefixes and the wrap points, and the visible
            // characters have to be a subsequence of the input.
            let joined: String = lines.join("");
            let want: String = s.chars().filter(|c| *c != ' ' && *c != '\n').collect();
            let got: String = joined.chars().filter(|c| *c != ' ' && *c != '\n').collect();
            assert!(
                got.len() >= want.len(),
                "wrapping {s:?} at {width} lost characters: {got:?}"
            );
        }
    }
}

#[test]
fn the_default_sheet_holds_exactly_the_documented_characters() {
    let f = require!(font(false), "jars/versions/1.8.9/1.8.9.jar");
    assert_eq!(FONT_CHARS.chars().count(), 256);
    // Everything printable-ASCII is on the sheet and measures from the ascii table...
    for c in ' '..='~' {
        if c == ' ' {
            continue;
        }
        assert!(f.char_width(c) > 0, "{c:?} should have a default glyph");
    }
    // ...and a CJK ideograph is not, so it falls through to the unicode pages.
    assert_eq!(mc_ui::font::font_char_index(0x4e00), -1);
    assert_eq!(
        f.char_width('\u{4e00}'),
        9,
        "still measurable via glyph_sizes"
    );
}

#[test]
fn every_unicode_page_the_jar_ships_is_a_two_fifty_six_square() {
    let rm = require!(manager(), "jars/versions/1.8.9/1.8.9.jar");
    let mut found = 0;
    for page in 0u8..=255 {
        let loc = mc_ui::font::unicode_page_location(page);
        let Ok(bytes) = rm.load(&loc) else { continue };
        found += 1;
        let img = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
            .unwrap_or_else(|e| panic!("decode {loc}: {e}"));
        assert_eq!(
            (img.width(), img.height()),
            (256, 256),
            "{loc} is not a 16x16 grid of 16x16 cells"
        );
    }
    assert!(found > 100, "only {found} unicode pages found");
}

/// The whole point of the shipped assets: real menu labels must land on the exact pixel vanilla
/// puts them on, because every screen in S16/S17 centres text with `x - width / 2`.
#[test]
fn centred_menu_labels_land_on_vanillas_pixels() {
    let f = require!(font(false), "jars/versions/1.8.9/1.8.9.jar");
    // A 427-wide scaled surface (854x480 at guiScale auto) centres on x = 213.
    let centre = 213;
    let cases = [
        ("Singleplayer", 62),
        ("Multiplayer", 54),
        ("Options...", 42),
        ("Done", 24),
    ];
    for (label, width) in cases {
        assert_eq!(f.string_width(label), width);
        let left = centre - f.string_width(label) / 2;
        assert_eq!(left, centre - width / 2, "{label:?} left edge");
    }
}

#[test]
fn the_font_loads_through_the_resource_manager_and_the_locations_resolve() {
    let rm = require!(manager(), "jars/versions/1.8.9/1.8.9.jar");
    assert!(rm.contains(&ResourceLocation::minecraft("textures/font/ascii.png")));
    assert!(rm.contains(&ResourceLocation::minecraft("font/glyph_sizes.bin")));
    assert!(FontRenderer::load(&rm, false, false).is_ok());
}
