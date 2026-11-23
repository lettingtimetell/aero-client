//! Java's *Modified UTF-8* — the string encoding used by `DataInput::readUTF` and
//! `DataOutput::writeUTF`, and therefore by every NBT string and every NBT tag name.
//!
//! Hazard **H2**. This is **not** standard UTF-8 and `String::from_utf8` will get it wrong
//! on real data. Two differences matter:
//!
//! 1. `U+0000` is written *overlong*, as the two bytes `0xC0 0x80`, so that a NUL never
//!    appears in the encoded byte stream.
//! 2. Characters above `U+FFFF` are split into a UTF-16 surrogate pair and each surrogate
//!    is then encoded as its own three-byte sequence (CESU-8), for six bytes total —
//!    never the four-byte form standard UTF-8 uses.
//!
//! The length prefix that precedes the data on the wire counts **bytes, not characters**,
//! and is a `u16`, so a single string can never exceed 65 535 encoded bytes.
//!
//! java: `java.io.DataOutputStream.writeUTF` / `java.io.DataInputStream.readUTF`

use thiserror::Error;

/// The maximum number of encoded bytes a single modified-UTF-8 string may occupy.
///
/// The on-the-wire length prefix is an unsigned 16-bit big-endian integer.
pub const MAX_ENCODED_LEN: usize = u16::MAX as usize;

/// Something went wrong encoding or decoding modified UTF-8.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ModifiedUtf8Error {
    /// A multi-byte sequence ran off the end of the input.
    #[error("truncated modified-UTF-8 sequence starting at byte {index}")]
    Truncated {
        /// Byte offset of the lead byte whose sequence is incomplete.
        index: usize,
    },
    /// A lead byte in the `0x80..=0xBF` or `0xF0..=0xFF` range, which Java rejects.
    #[error("invalid modified-UTF-8 lead byte 0x{byte:02x} at byte {index}")]
    InvalidLeadByte {
        /// Byte offset of the offending byte.
        index: usize,
        /// The offending byte.
        byte: u8,
    },
    /// A continuation byte that did not have the `10xxxxxx` bit pattern.
    #[error("invalid modified-UTF-8 continuation byte 0x{byte:02x} at byte {index}")]
    InvalidContinuationByte {
        /// Byte offset of the offending byte.
        index: usize,
        /// The offending byte.
        byte: u8,
    },
    /// A UTF-16 surrogate that had no partner.
    ///
    /// No longer returned by [`decode`]: a lone surrogate now decodes to its
    /// [`LONE_SURROGATE_ESCAPE`] code point, the way Java's `readUTF` keeps it. The variant
    /// stays so the public enum does not change shape.
    #[error("unpaired UTF-16 surrogate U+{unit:04X} in modified-UTF-8 data")]
    UnpairedSurrogate {
        /// The lone surrogate code unit.
        unit: u16,
    },
    /// The string encodes to more bytes than the `u16` length prefix can describe.
    #[error("string encodes to {len} modified-UTF-8 bytes; the u16 length prefix caps this at {MAX_ENCODED_LEN}")]
    TooLong {
        /// The number of bytes the string would have needed.
        len: usize,
    },
}

/// Where a lone UTF-16 surrogate lives in a Rust `String`: surrogate `unit` is the code point
/// `unit + LONE_SURROGATE_ESCAPE`, i.e. `U+FD800..=U+FDFFF` (Supplementary Private Use Area-A).
///
/// A Java `String` may hold an unpaired surrogate, and `readUTF` returns one for a three-byte
/// sequence such as `ED A0 BD`: half an emoji, say, cut off by the 30-character anvil name
/// limit and then sent to every nearby player in the item's NBT. A Rust `String` cannot hold
/// one, so [`decode`] maps it here and [`encode_into`] maps it back. Java's bytes therefore
/// round-trip exactly (a click-window packet carries the item back to the server unchanged),
/// and the text draws exactly as in 1.8.9: the font walks UTF-16 units, and both halves of this
/// supplementary code point are zero-width surrogate glyphs, like the lone surrogate itself.
/// The price is that a genuine `U+FD800..=U+FDFFF` character, which 1.8.9 cannot draw anyway,
/// re-encodes as a lone surrogate.
pub const LONE_SURROGATE_ESCAPE: u32 = 0xF_0000;

/// The lone surrogate `c` stands for, if `c` is a [`LONE_SURROGATE_ESCAPE`] code point.
#[must_use]
pub fn escaped_lone_surrogate(c: char) -> Option<u16> {
    let unit = u32::from(c).checked_sub(LONE_SURROGATE_ESCAPE)?;
    u16::try_from(unit)
        .ok()
        .filter(|unit| (0xD800..=0xDFFF).contains(unit))
}

/// Number of bytes `s` occupies once encoded as modified UTF-8.
///
/// ```
/// # use mc_nbt::modified_utf8::encoded_len;
/// assert_eq!(encoded_len("abc"), 3);
/// assert_eq!(encoded_len("\0"), 2);       // NUL is overlong-encoded
/// assert_eq!(encoded_len("\u{1F600}"), 6); // surrogate pair, 3 bytes each
/// ```
#[must_use]
pub fn encoded_len(s: &str) -> usize {
    s.chars().map(char_encoded_len).sum()
}

/// Bytes one `char` occupies once encoded as modified UTF-8.
#[must_use]
fn char_encoded_len(c: char) -> usize {
    if escaped_lone_surrogate(c).is_some() {
        return 3;
    }
    match c as u32 {
        0x0001..=0x007F => 1,
        0x0000 | 0x0080..=0x07FF => 2,
        0x0800..=0xFFFF => 3,
        // Supplementary: one three-byte sequence per surrogate half.
        _ => 6,
    }
}

/// Encode `s` as modified UTF-8, appending to `out`.
///
/// Returns the number of bytes appended. Fails only if the result would overflow the
/// `u16` length prefix, matching Java's `UTFDataFormatException`.
pub fn encode_into(s: &str, out: &mut Vec<u8>) -> Result<usize, ModifiedUtf8Error> {
    let len = encoded_len(s);
    if len > MAX_ENCODED_LEN {
        return Err(ModifiedUtf8Error::TooLong { len });
    }
    out.reserve(len);
    for c in s.chars() {
        if let Some(unit) = escaped_lone_surrogate(c) {
            push_three(out, unit);
            continue;
        }
        let cp = c as u32;
        match cp {
            0x0001..=0x007F => out.push(cp as u8),
            0x0000 | 0x0080..=0x07FF => {
                out.push(0xC0 | (cp >> 6) as u8);
                out.push(0x80 | (cp & 0x3F) as u8);
            }
            0x0800..=0xFFFF => push_three(out, cp as u16),
            _ => {
                // java: writeUTF iterates UTF-16 code units, so an astral char arrives as
                // its surrogate pair and each half is encoded separately (CESU-8).
                let v = cp - 0x1_0000;
                push_three(out, (0xD800 + (v >> 10)) as u16);
                push_three(out, (0xDC00 + (v & 0x3FF)) as u16);
            }
        }
    }
    Ok(len)
}

/// Encode `s` as modified UTF-8 into a fresh `Vec`.
pub fn encode(s: &str) -> Result<Vec<u8>, ModifiedUtf8Error> {
    let mut out = Vec::new();
    encode_into(s, &mut out)?;
    Ok(out)
}

fn push_three(out: &mut Vec<u8>, unit: u16) {
    let unit = u32::from(unit);
    out.push(0xE0 | (unit >> 12) as u8);
    out.push(0x80 | ((unit >> 6) & 0x3F) as u8);
    out.push(0x80 | (unit & 0x3F) as u8);
}

/// Decode `bytes` from modified UTF-8.
///
/// Overlong forms are accepted, because Java produces them deliberately for `U+0000` and
/// its own decoder never checks for them. Surrogate halves are re-paired into astral
/// characters; a lone surrogate becomes its [`LONE_SURROGATE_ESCAPE`] code point, where
/// Java's `String` keeps it as is (failing here used to fail the whole packet).
///
/// ```
/// # use mc_nbt::modified_utf8::decode;
/// assert_eq!(decode(&[0xC0, 0x80]).unwrap(), "\0");
/// assert_eq!(decode(&[0xED, 0xA0, 0xBD, 0xED, 0xB8, 0x80]).unwrap(), "\u{1F600}");
/// assert_eq!(decode(&[0xED, 0xA0, 0xBD]).unwrap(), "\u{FD83D}");
/// ```
pub fn decode(bytes: &[u8]) -> Result<String, ModifiedUtf8Error> {
    // One code unit per input byte is the worst case (all-ASCII), so this never over-reserves
    // by more than 3x and the input is already length-capped by the caller's u16 prefix.
    let mut units: Vec<u16> = Vec::with_capacity(bytes.len());
    let mut i = 0usize;

    while i < bytes.len() {
        let b = bytes[i];
        // java: readUTF switches on `c >> 4`; 0..=7 is one byte, 12|13 is two, 14 is three.
        match b >> 4 {
            0x0..=0x7 => {
                units.push(u16::from(b));
                i += 1;
            }
            0xC | 0xD => {
                let b2 = *bytes
                    .get(i + 1)
                    .ok_or(ModifiedUtf8Error::Truncated { index: i })?;
                check_continuation(b2, i + 1)?;
                units.push((u16::from(b & 0x1F) << 6) | u16::from(b2 & 0x3F));
                i += 2;
            }
            0xE => {
                let b2 = *bytes
                    .get(i + 1)
                    .ok_or(ModifiedUtf8Error::Truncated { index: i })?;
                let b3 = *bytes
                    .get(i + 2)
                    .ok_or(ModifiedUtf8Error::Truncated { index: i })?;
                check_continuation(b2, i + 1)?;
                check_continuation(b3, i + 2)?;
                units.push(
                    (u16::from(b & 0x0F) << 12)
                        | (u16::from(b2 & 0x3F) << 6)
                        | u16::from(b3 & 0x3F),
                );
                i += 3;
            }
            _ => return Err(ModifiedUtf8Error::InvalidLeadByte { index: i, byte: b }),
        }
    }

    Ok(char::decode_utf16(units)
        .map(|unit| {
            unit.unwrap_or_else(|lone| {
                char::from_u32(u32::from(lone.unpaired_surrogate()) + LONE_SURROGATE_ESCAPE)
                    .unwrap_or(char::REPLACEMENT_CHARACTER)
            })
        })
        .collect())
}

fn check_continuation(byte: u8, index: usize) -> Result<(), ModifiedUtf8Error> {
    if byte & 0xC0 == 0x80 {
        Ok(())
    } else {
        Err(ModifiedUtf8Error::InvalidContinuationByte { index, byte })
    }
}

/// Length of `s` in UTF-16 code units — Java's `String.length()`.
///
/// [`crate::limits::SizeTracker`] charges per code unit, so it needs this rather than
/// `str::len` (bytes) or `chars().count()` (scalar values). An escaped lone surrogate is the
/// one unit it was in Java.
#[must_use]
pub fn utf16_len(s: &str) -> usize {
    s.chars()
        .map(|c| {
            if escaped_lone_surrogate(c).is_some() {
                1
            } else {
                c.len_utf16()
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_matches_standard_utf8() {
        assert_eq!(encode("Hello, world!").unwrap(), b"Hello, world!");
        assert_eq!(decode(b"Hello, world!").unwrap(), "Hello, world!");
    }

    #[test]
    fn nul_is_two_bytes() {
        // The whole point of "modified": a NUL must not appear in the byte stream.
        assert_eq!(encode("\0").unwrap(), vec![0xC0, 0x80]);
        assert_eq!(encode("a\0b").unwrap(), vec![b'a', 0xC0, 0x80, b'b']);
        assert_eq!(decode(&[0xC0, 0x80]).unwrap(), "\0");
        assert!(!encode("a\0b").unwrap().contains(&0));
    }

    #[test]
    fn bare_nul_byte_decodes_even_though_we_never_write_it() {
        // Java's readUTF treats 0x00 as a one-byte character.
        assert_eq!(decode(&[0x00]).unwrap(), "\0");
    }

    #[test]
    fn two_and_three_byte_boundaries() {
        assert_eq!(encode("\u{7F}").unwrap(), vec![0x7F]);
        assert_eq!(encode("\u{80}").unwrap(), vec![0xC2, 0x80]);
        assert_eq!(encode("\u{7FF}").unwrap(), vec![0xDF, 0xBF]);
        assert_eq!(encode("\u{800}").unwrap(), vec![0xE0, 0xA0, 0x80]);
        assert_eq!(encode("\u{FFFF}").unwrap(), vec![0xEF, 0xBF, 0xBF]);
    }

    #[test]
    fn astral_is_cesu8_not_utf8() {
        // U+1F600 GRINNING FACE. Standard UTF-8 is F0 9F 98 80; Java writes six bytes.
        let bytes = encode("\u{1F600}").unwrap();
        assert_eq!(bytes, vec![0xED, 0xA0, 0xBD, 0xED, 0xB8, 0x80]);
        assert_ne!(bytes, "\u{1F600}".as_bytes());
        assert_eq!(decode(&bytes).unwrap(), "\u{1F600}");
    }

    #[test]
    fn astral_round_trip_byte_identical() {
        let s = "emoji \u{1F600}\u{1F680} and \u{10FFFF} plus \0 and \u{4E2D}\u{6587}";
        let bytes = encode(s).unwrap();
        assert_eq!(bytes.len(), encoded_len(s));
        assert_eq!(decode(&bytes).unwrap(), s);
        assert_eq!(encode(&decode(&bytes).unwrap()).unwrap(), bytes);
    }

    /// java: `readUTF` builds a `String` from the units as they come, so a lone surrogate is
    /// one ordinary `char` there. It must decode (failing here fails the whole packet, which
    /// disconnects), keep Java's one-unit length, and re-encode to the same three bytes.
    #[test]
    fn a_lone_surrogate_decodes_to_its_escape_and_round_trips() {
        let high = [0xED, 0xA0, 0xBD];
        assert_eq!(decode(&high).unwrap(), "\u{FD83D}");
        assert_eq!(escaped_lone_surrogate('\u{FD83D}'), Some(0xD83D));
        assert_eq!(encode("\u{FD83D}").unwrap(), high);
        assert_eq!(encoded_len("\u{FD83D}"), 3);
        assert_eq!(
            utf16_len("\u{FD83D}"),
            1,
            "java: String.length() of one lone unit"
        );

        // A low half first, a high half before ASCII, and a high half followed by a
        // complete pair: only the units without a partner are escaped.
        let bytes = [
            0xED, 0xB8, 0x80, // lone DE00
            0xED, 0xA0, 0xBD, b'A', // lone D83D, then 'A'
            0xED, 0xA0, 0xBD, 0xED, 0xA0, 0xBD, 0xED, 0xB8, 0x80, // lone D83D, then U+1F600
        ];
        let text = decode(&bytes).unwrap();
        assert_eq!(text, "\u{FDE00}\u{FD83D}A\u{FD83D}\u{1F600}");
        assert_eq!(encode(&text).unwrap(), bytes, "byte-identical round trip");
        assert_eq!(encoded_len(&text), bytes.len());
        assert_eq!(utf16_len(&text), 6);

        // The range edges, and their neighbours that are ordinary characters.
        assert_eq!(decode(&[0xED, 0xA0, 0x80]).unwrap(), "\u{FD800}");
        assert_eq!(decode(&[0xED, 0xBF, 0xBF]).unwrap(), "\u{FDFFF}");
        assert_eq!(escaped_lone_surrogate('\u{FD7FF}'), None);
        assert_eq!(escaped_lone_surrogate('\u{FE000}'), None);
        assert_eq!(
            encode("\u{FE000}").unwrap().len(),
            6,
            "an ordinary supplementary char"
        );
    }

    #[test]
    fn malformed_input_is_rejected_not_panicked() {
        assert!(matches!(
            decode(&[0xC2]),
            Err(ModifiedUtf8Error::Truncated { index: 0 })
        ));
        assert!(matches!(
            decode(&[0xE0, 0xA0]),
            Err(ModifiedUtf8Error::Truncated { index: 0 })
        ));
        assert!(matches!(
            decode(&[0xF0, 0x9F, 0x98, 0x80]),
            Err(ModifiedUtf8Error::InvalidLeadByte { index: 0, .. })
        ));
        assert!(matches!(
            decode(&[0x80]),
            Err(ModifiedUtf8Error::InvalidLeadByte { index: 0, .. })
        ));
        assert!(matches!(
            decode(&[0xC2, 0x41]),
            Err(ModifiedUtf8Error::InvalidContinuationByte { index: 1, .. })
        ));
    }

    #[test]
    fn over_long_strings_are_rejected() {
        let s: String = core::iter::repeat_n('\u{800}', 30_000).collect();
        assert!(matches!(
            encode(&s),
            Err(ModifiedUtf8Error::TooLong { len: 90_000 })
        ));
    }

    #[test]
    fn utf16_len_counts_code_units() {
        assert_eq!(utf16_len("abc"), 3);
        assert_eq!(utf16_len("\u{4E2D}"), 1);
        assert_eq!(utf16_len("\u{1F600}"), 2);
    }
}
