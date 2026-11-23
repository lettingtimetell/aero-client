//! Vanilla's `§` formatting codes and the 32-entry colour table.
//!
//! java: `util/EnumChatFormatting.java`, `client/gui/FontRenderer.java:110-140`
//!
//! Two things live here that look similar but are not:
//!
//! * [`TextFormat`] — the *enum* of codes a chat component can name (`"color": "red"`), and
//!   the `§x` control strings that encode them in a flat string.
//! * [`color_table`] — the *rendered* RGB for each of the 16 colours, plus the 16 darkened
//!   variants used for drop shadow. Vanilla derives these arithmetically rather than listing
//!   them, and the derivation has a special case for gold (`§6`).

/// The section sign that introduces a formatting code. `U+00A7`.
pub const SECTION_SIGN: char = '\u{a7}';

/// A vanilla text formatting code.
///
/// The first sixteen are colours (`§0`–`§f`), then five style flags (`§k`–`§o`), then
/// [`TextFormat::Reset`] (`§r`).
// java: EnumChatFormatting.java:12
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TextFormat {
    /// `§0`
    Black,
    /// `§1`
    DarkBlue,
    /// `§2`
    DarkGreen,
    /// `§3`
    DarkAqua,
    /// `§4`
    DarkRed,
    /// `§5`
    DarkPurple,
    /// `§6`
    Gold,
    /// `§7`
    Gray,
    /// `§8`
    DarkGray,
    /// `§9`
    Blue,
    /// `§a`
    Green,
    /// `§b`
    Aqua,
    /// `§c`
    Red,
    /// `§d`
    LightPurple,
    /// `§e`
    Yellow,
    /// `§f`
    White,
    /// `§k` — glyphs are replaced every frame with a random same-width glyph.
    Obfuscated,
    /// `§l`
    Bold,
    /// `§m`
    Strikethrough,
    /// `§n`
    Underline,
    /// `§o`
    Italic,
    /// `§r` — clears colour *and* every style flag.
    Reset,
}

impl TextFormat {
    /// Every code, in vanilla's declaration order.
    pub const ALL: [TextFormat; 22] = [
        TextFormat::Black,
        TextFormat::DarkBlue,
        TextFormat::DarkGreen,
        TextFormat::DarkAqua,
        TextFormat::DarkRed,
        TextFormat::DarkPurple,
        TextFormat::Gold,
        TextFormat::Gray,
        TextFormat::DarkGray,
        TextFormat::Blue,
        TextFormat::Green,
        TextFormat::Aqua,
        TextFormat::Red,
        TextFormat::LightPurple,
        TextFormat::Yellow,
        TextFormat::White,
        TextFormat::Obfuscated,
        TextFormat::Bold,
        TextFormat::Strikethrough,
        TextFormat::Underline,
        TextFormat::Italic,
        TextFormat::Reset,
    ];

    /// The lower-case character that follows `§`.
    #[must_use]
    pub const fn code(self) -> char {
        match self {
            TextFormat::Black => '0',
            TextFormat::DarkBlue => '1',
            TextFormat::DarkGreen => '2',
            TextFormat::DarkAqua => '3',
            TextFormat::DarkRed => '4',
            TextFormat::DarkPurple => '5',
            TextFormat::Gold => '6',
            TextFormat::Gray => '7',
            TextFormat::DarkGray => '8',
            TextFormat::Blue => '9',
            TextFormat::Green => 'a',
            TextFormat::Aqua => 'b',
            TextFormat::Red => 'c',
            TextFormat::LightPurple => 'd',
            TextFormat::Yellow => 'e',
            TextFormat::White => 'f',
            TextFormat::Obfuscated => 'k',
            TextFormat::Bold => 'l',
            TextFormat::Strikethrough => 'm',
            TextFormat::Underline => 'n',
            TextFormat::Italic => 'o',
            TextFormat::Reset => 'r',
        }
    }

    /// Look a code up by character. Upper case is accepted, exactly as vanilla's
    /// `"0123456789abcdefklmnor".indexOf(lowerCase(c))` does.
    #[must_use]
    pub fn from_code(c: char) -> Option<Self> {
        let lower = c.to_ascii_lowercase();
        Self::ALL.into_iter().find(|f| f.code() == lower)
    }

    /// Index into the 16-entry colour palette, or `-1` for a style flag and for
    /// [`TextFormat::Reset`].
    // java: EnumChatFormatting.java:86 (getColorIndex)
    #[must_use]
    pub const fn color_index(self) -> i32 {
        match self {
            TextFormat::Black => 0,
            TextFormat::DarkBlue => 1,
            TextFormat::DarkGreen => 2,
            TextFormat::DarkAqua => 3,
            TextFormat::DarkRed => 4,
            TextFormat::DarkPurple => 5,
            TextFormat::Gold => 6,
            TextFormat::Gray => 7,
            TextFormat::DarkGray => 8,
            TextFormat::Blue => 9,
            TextFormat::Green => 10,
            TextFormat::Aqua => 11,
            TextFormat::Red => 12,
            TextFormat::LightPurple => 13,
            TextFormat::Yellow => 14,
            TextFormat::White => 15,
            _ => -1,
        }
    }

    /// `true` for `§k`–`§o`, the style flags. Colours and `§r` are not "fancy".
    // java: EnumChatFormatting.java:94 (isFancyStyling)
    #[must_use]
    pub const fn is_fancy(self) -> bool {
        matches!(
            self,
            TextFormat::Obfuscated
                | TextFormat::Bold
                | TextFormat::Strikethrough
                | TextFormat::Underline
                | TextFormat::Italic
        )
    }

    /// `true` for `§0`–`§f`.
    // java: EnumChatFormatting.java:102 (isColor)
    #[must_use]
    pub const fn is_color(self) -> bool {
        !self.is_fancy() && !matches!(self, TextFormat::Reset)
    }

    /// The name a chat component's `"color"` field uses, e.g. `"dark_blue"`.
    #[must_use]
    pub const fn friendly_name(self) -> &'static str {
        match self {
            TextFormat::Black => "black",
            TextFormat::DarkBlue => "dark_blue",
            TextFormat::DarkGreen => "dark_green",
            TextFormat::DarkAqua => "dark_aqua",
            TextFormat::DarkRed => "dark_red",
            TextFormat::DarkPurple => "dark_purple",
            TextFormat::Gold => "gold",
            TextFormat::Gray => "gray",
            TextFormat::DarkGray => "dark_gray",
            TextFormat::Blue => "blue",
            TextFormat::Green => "green",
            TextFormat::Aqua => "aqua",
            TextFormat::Red => "red",
            TextFormat::LightPurple => "light_purple",
            TextFormat::Yellow => "yellow",
            TextFormat::White => "white",
            TextFormat::Obfuscated => "obfuscated",
            TextFormat::Bold => "bold",
            TextFormat::Strikethrough => "strikethrough",
            TextFormat::Underline => "underline",
            TextFormat::Italic => "italic",
            TextFormat::Reset => "reset",
        }
    }

    /// Look a code up by friendly name.
    ///
    /// Vanilla lower-cases and strips every non-`a`–`z` character before matching, so
    /// `"DARK_BLUE"`, `"darkblue"` and `"dark blue"` all resolve.
    // java: EnumChatFormatting.java:59 (func_175745_c) + :131 (getValueByName)
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        let key = normalise_name(name);
        Self::ALL
            .into_iter()
            .find(|f| normalise_name(f.friendly_name()) == key)
    }

    /// The two-character control string, e.g. `"§c"`.
    // java: EnumChatFormatting.java:115 (toString)
    #[must_use]
    pub fn control_string(self) -> String {
        let mut s = String::with_capacity(3);
        s.push(SECTION_SIGN);
        s.push(self.code());
        s
    }
}

fn normalise_name(name: &str) -> String {
    name.chars()
        .flat_map(char::to_lowercase)
        .filter(|c| c.is_ascii_lowercase())
        .collect()
}

/// `true` for `0`–`9`, `a`–`f`, `A`–`F`.
///
/// This is exactly `is_ascii_hexdigit`, but spelling out the three ranges keeps the line
/// readable next to the Java it is ported from — and `is_ascii_hexdigit` is not `const`.
// java: FontRenderer.java:942 (isFormatColor)
#[must_use]
#[allow(clippy::manual_is_ascii_check)]
pub const fn is_format_color(c: char) -> bool {
    matches!(c, '0'..='9' | 'a'..='f' | 'A'..='F')
}

/// `true` for `k`–`o`, `K`–`O`, `r`, `R`.
// java: FontRenderer.java:950 (isFormatSpecial)
#[must_use]
pub const fn is_format_special(c: char) -> bool {
    matches!(c, 'k'..='o' | 'K'..='O' | 'r' | 'R')
}

/// Strip every `§`-code from a string.
///
/// Matches vanilla's `(?i)§[0-9A-FK-OR]` regex: a `§` followed by anything else (including a
/// trailing `§` at the very end) is left alone.
// java: EnumChatFormatting.java:41 + :123 (getTextWithoutFormattingCodes)
#[must_use]
pub fn strip_formatting(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == SECTION_SIGN {
            if let Some(&next) = chars.peek() {
                if is_format_color(next) || is_format_special(next) {
                    chars.next();
                    continue;
                }
            }
        }
        out.push(c);
    }
    out
}

/// The formatting prefix a wrapped continuation line has to repeat.
///
/// Walks every `§` in the string; a **colour** code *replaces* the accumulated prefix while a
/// **style** code *appends* to it. That asymmetry is vanilla's, and it is why a line that turns
/// bold and then changes colour loses its bold on the next wrapped line.
// java: FontRenderer.java:958 (getFormatFromString)
#[must_use]
pub fn format_from_string(text: &str) -> String {
    let units: Vec<u16> = text.encode_utf16().collect();
    let mut s: Vec<u16> = Vec::new();
    let j = units.len();
    let mut i: isize = -1;
    loop {
        let from = (i + 1) as usize;
        let Some(found) = units[from.min(j)..]
            .iter()
            .position(|&u| u == SECTION_SIGN as u16)
        else {
            break;
        };
        i = (from + found) as isize;
        if (i as usize) < j.saturating_sub(1) {
            let c0 = units[i as usize + 1];
            let ch = char::from_u32(u32::from(c0)).unwrap_or('\u{fffd}');
            if is_format_color(ch) {
                s.clear();
                s.push(SECTION_SIGN as u16);
                s.push(c0);
            } else if is_format_special(ch) {
                s.push(SECTION_SIGN as u16);
                s.push(c0);
            }
        }
    }
    String::from_utf16_lossy(&s)
}

/// Vanilla's 32-entry colour table: 16 chat colours followed by 16 darkened drop-shadow copies.
///
/// Each entry is `0x00RRGGBB`. The values are derived, not tabulated — bit 3 of the index is a
/// +85 "bright" bias, bits 2/1/0 pick red/green/blue, and index 6 (gold) gets an extra +85 on
/// red so that it comes out `#FFAA00` instead of `#AAAA00`.
///
/// `anaglyph` reproduces the red/cyan 3-D mode's channel mixing, which vanilla applies to the
/// palette rather than to the frame.
// java: FontRenderer.java:110-140
#[must_use]
pub fn color_table(anaglyph: bool) -> [i32; 32] {
    let mut out = [0i32; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        let i = i as i32;
        let j = (i >> 3 & 1) * 85;
        let mut k = (i >> 2 & 1) * 170 + j;
        let mut l = (i >> 1 & 1) * 170 + j;
        let mut i1 = (i & 1) * 170 + j;

        if i == 6 {
            k += 85;
        }

        if anaglyph {
            let j1 = (k * 30 + l * 59 + i1 * 11) / 100;
            let k1 = (k * 30 + l * 70) / 100;
            let l1 = (k * 30 + i1 * 70) / 100;
            k = j1;
            l = k1;
            i1 = l1;
        }

        if i >= 16 {
            k /= 4;
            l /= 4;
            i1 /= 4;
        }

        *slot = (k & 255) << 16 | (l & 255) << 8 | (i1 & 255);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_round_trip() {
        for f in TextFormat::ALL {
            assert_eq!(TextFormat::from_code(f.code()), Some(f));
            assert_eq!(
                TextFormat::from_code(f.code().to_ascii_uppercase()),
                Some(f),
                "{f:?} should accept upper case"
            );
            assert_eq!(TextFormat::from_name(f.friendly_name()), Some(f));
            assert_eq!(f.control_string().chars().next(), Some(SECTION_SIGN));
        }
        assert_eq!(TextFormat::from_code('z'), None);
        assert_eq!(TextFormat::from_code('§'), None);
    }

    #[test]
    fn friendly_names_normalise_like_vanilla() {
        // java: func_175745_c lower-cases and drops everything outside a-z.
        assert_eq!(
            TextFormat::from_name("DARK_BLUE"),
            Some(TextFormat::DarkBlue)
        );
        assert_eq!(
            TextFormat::from_name("darkblue"),
            Some(TextFormat::DarkBlue)
        );
        assert_eq!(
            TextFormat::from_name("dark blue"),
            Some(TextFormat::DarkBlue)
        );
        assert_eq!(TextFormat::from_name("nonsense"), None);
    }

    #[test]
    fn color_and_fancy_partition_the_codes() {
        let colors: Vec<_> = TextFormat::ALL
            .into_iter()
            .filter(|f| f.is_color())
            .collect();
        let fancy: Vec<_> = TextFormat::ALL
            .into_iter()
            .filter(|f| f.is_fancy())
            .collect();
        assert_eq!(colors.len(), 16);
        assert_eq!(fancy.len(), 5);
        // Reset is in neither bucket.
        assert!(!TextFormat::Reset.is_color());
        assert!(!TextFormat::Reset.is_fancy());
        for (i, f) in colors.iter().enumerate() {
            assert_eq!(f.color_index(), i as i32);
        }
        for f in &fancy {
            assert_eq!(f.color_index(), -1);
        }
        assert_eq!(TextFormat::Reset.color_index(), -1);
    }

    /// The palette is vanilla's 16 formatting colours; gold's `+85` special case is the
    /// only entry that is not a plain 0/85/170/255 combination.
    #[test]
    fn color_table_matches_the_documented_palette() {
        let t = color_table(false);
        let expected = [
            0x000000, 0x0000AA, 0x00AA00, 0x00AAAA, 0xAA0000, 0xAA00AA, 0xFFAA00, 0xAAAAAA,
            0x555555, 0x5555FF, 0x55FF55, 0x55FFFF, 0xFF5555, 0xFF55FF, 0xFFFF55, 0xFFFFFF,
        ];
        for (i, want) in expected.iter().enumerate() {
            assert_eq!(t[i], *want, "colour {i} (§{:x})", i);
        }
    }

    #[test]
    fn shadow_colours_are_the_palette_over_four_except_gold() {
        let t = color_table(false);
        for i in 0..16 {
            if i == 6 {
                continue;
            }
            let (r, g, b) = (t[i] >> 16 & 255, t[i] >> 8 & 255, t[i] & 255);
            let want = (r / 4) << 16 | (g / 4) << 8 | (b / 4);
            assert_eq!(t[i + 16], want, "shadow colour {i}");
        }
        assert_eq!(t[16 + 15], 0x3F3F3F, "white's shadow is 63,63,63");
        // The `if (i == 6)` +85 is keyed on the *index*, so it fires for §6 and not for its
        // shadow twin at 22. Gold's shadow is therefore (170,170,0)/4, not (255,170,0)/4 —
        // a genuine vanilla asymmetry, not a rounding artefact.
        assert_eq!(t[16 + 6], 0x2A2A00);
        assert_ne!(t[16 + 6], (0xFF / 4) << 16 | (0xAA / 4) << 8);
    }

    #[test]
    fn anaglyph_mixes_channels() {
        let plain = color_table(false);
        let ana = color_table(true);
        assert_ne!(plain[12], ana[12]);
        assert_eq!(ana[0], 0, "black stays black under anaglyph");
    }

    #[test]
    fn strip_formatting_removes_only_real_codes() {
        assert_eq!(strip_formatting("§cRed §lBold§r plain"), "Red Bold plain");
        assert_eq!(strip_formatting("no codes"), "no codes");
        assert_eq!(strip_formatting("§"), "§", "a lone trailing § survives");
        assert_eq!(strip_formatting("§z kept"), "§z kept", "§z is not a code");
        assert_eq!(
            strip_formatting("§C upper"),
            " upper",
            "codes are case-insensitive"
        );
        assert_eq!(
            strip_formatting("§§c"),
            "§",
            "the second § is consumed with its c"
        );
    }

    /// Golden values from the Python transcription of `getFormatFromString`.
    #[test]
    fn format_from_string_matches_vanilla() {
        assert_eq!(format_from_string("§cabc"), "§c");
        assert_eq!(format_from_string("§c§ldef"), "§c§l");
        // A colour *replaces* the accumulated prefix, so the earlier §l is lost.
        assert_eq!(format_from_string("§l§cghi"), "§c");
        assert_eq!(format_from_string("plain"), "");
        assert_eq!(
            format_from_string("§"),
            "",
            "a trailing § has no code after it"
        );
        assert_eq!(format_from_string("§z"), "", "§z is not a formatting code");
        assert_eq!(format_from_string("§l§m§n"), "§l§m§n", "styles accumulate");
    }
}
