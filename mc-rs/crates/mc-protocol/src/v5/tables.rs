//! Small protocol-5 lookup tables that decoding needs.
//!
//! java17: `world/WorldType.java`, `java.lang.String.equalsIgnoreCase` (JDK 8)

/// 1.7.10's world types as `S01PacketJoinGame` and `S07PacketRespawn` name them, in
/// `WorldType.worldTypes` order (ids 0, 1, 2, 3 and 8). 1.8.9 adds `customized` and
/// `debug_all_block_states`, neither of which exists here.
///
/// java17: `WorldType.java:6-10`
pub const WORLD_TYPES: [&str; 5] = ["default", "flat", "largeBiomes", "amplified", "default_1_1"];

/// The canonical name of the world type a packet's string selects.
///
/// java17: `WorldType.parseWorldType` compares with `String.equalsIgnoreCase` and the packet
/// falls back to `WorldType.DEFAULT` when nothing matches (`S01PacketJoinGame.readPacketData`,
/// `S07PacketRespawn.readPacketData`); `writePacketData` writes `getWorldTypeName()`.
#[must_use]
pub fn canonical_world_type(name: &str) -> &'static str {
    WORLD_TYPES
        .into_iter()
        .find(|candidate| java_equals_ignore_case_ascii(name, candidate))
        .unwrap_or(WORLD_TYPES[0])
}

/// `String.equalsIgnoreCase` (JDK 8) against an ASCII string.
///
/// Java compares UTF-16 units: equal, or equal after `Character.toUpperCase`, or equal after
/// `toLowerCase(toUpperCase(..))`. Besides ASCII's own two cases, exactly four characters reach
/// an ASCII letter that way: `U+0131` and `U+0130` (`i`), `U+017F` (`s`) and `U+212A` (`k`). A
/// byte-wise ASCII comparison would miss them.
#[must_use]
pub fn java_equals_ignore_case_ascii(input: &str, ascii: &str) -> bool {
    debug_assert!(ascii.is_ascii());
    input.encode_utf16().count() == ascii.len()
        && input
            .encode_utf16()
            .zip(ascii.bytes())
            .all(|(unit, target)| java_unit_eq_ignore_case(unit, target))
}

fn java_unit_eq_ignore_case(unit: u16, target: u8) -> bool {
    if let Ok(ascii) = u8::try_from(unit) {
        if ascii.is_ascii() {
            return ascii.eq_ignore_ascii_case(&target);
        }
    }
    match target.to_ascii_lowercase() {
        b'i' => unit == 0x0131 || unit == 0x0130,
        b's' => unit == 0x017F,
        b'k' => unit == 0x212A,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_type_lookup_is_case_insensitive_with_a_default_fallback() {
        assert_eq!(canonical_world_type("FLAT"), "flat");
        assert_eq!(canonical_world_type("largebiomes"), "largeBiomes");
        assert_eq!(canonical_world_type("DEFAULT_1_1"), "default_1_1");
        // 1.8.9-only names and junk fall back to "default".
        assert_eq!(canonical_world_type("customized"), "default");
        assert_eq!(canonical_world_type("debug_all_block_states"), "default");
        assert_eq!(canonical_world_type(""), "default");
    }

    #[test]
    fn javas_unicode_case_folding_reaches_ascii() {
        assert_eq!(canonical_world_type("largeB\u{131}omes"), "largeBiomes");
        assert_eq!(canonical_world_type("amplif\u{130}ed"), "amplified");
        assert_eq!(canonical_world_type("largeBiome\u{17F}"), "largeBiomes");
        assert!(java_equals_ignore_case_ascii("\u{212A}", "k"));
        // Negative controls: length counts UTF-16 units, and other letters do not fold.
        assert!(!java_equals_ignore_case_ascii("fla\u{1D42D}", "flat"));
        assert!(!java_equals_ignore_case_ascii("fl\u{E0}t", "flat"));
        assert!(!"largeB\u{131}omes".eq_ignore_ascii_case("largeBiomes"));
    }
}
