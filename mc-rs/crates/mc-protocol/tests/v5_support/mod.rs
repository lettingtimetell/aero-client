//! Shared by the protocol-5 integration tests: byte comparison that allows only the deflate body
//! of a gzip NBT blob to differ.

#![allow(dead_code)]

use std::io::Read;

use flate2::read::DeflateDecoder;

/// Lower-case hex.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Parse lower- or upper-case hex.
pub fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
        .collect()
}

/// A gzip NBT blob starting at `at`: an `i16` length, then `1f 8b 08`.
pub fn gzip_blob_at(bytes: &[u8], at: usize) -> Option<&[u8]> {
    let head = bytes.get(at..at + 5)?;
    let len = i16::from_be_bytes([head[0], head[1]]);
    if len <= 0 || head[2..5] != [0x1F, 0x8B, 0x08] {
        return None;
    }
    bytes.get(at + 2..at + 2 + len as usize)
}

/// The inflated content of a single-member gzip blob with a ten-byte header.
pub fn inflate_member(blob: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    DeflateDecoder::new(&blob[10..blob.len() - 8])
        .read_to_end(&mut out)
        .expect("deflate body");
    out
}

/// Byte equality, except that the deflate body of each gzip NBT blob may differ: its header,
/// trailer and inflated bytes must match. Returns how many blobs differed only in the body.
pub fn assert_wire_eq(name: &str, ours: &[u8], java: &[u8]) -> usize {
    let (mut i, mut j) = (0, 0);
    let mut relaxed = 0;
    while i < ours.len() && j < java.len() {
        if let (Some(a), Some(b)) = (gzip_blob_at(ours, i), gzip_blob_at(java, j)) {
            assert_eq!(&a[..10], &b[..10], "{name}: gzip header at byte {j}");
            assert_eq!(
                &a[a.len() - 8..],
                &b[b.len() - 8..],
                "{name}: gzip trailer at byte {j}"
            );
            assert_eq!(
                inflate_member(a),
                inflate_member(b),
                "{name}: inflated NBT at byte {j}"
            );
            if a != b {
                relaxed += 1;
            }
            i += 2 + a.len();
            j += 2 + b.len();
            continue;
        }
        assert_eq!(
            ours[i],
            java[j],
            "{name}: first difference at byte {j}\n  vanilla: {}\n  ours:    {}",
            hex(java),
            hex(ours)
        );
        i += 1;
        j += 1;
    }
    assert!(
        i == ours.len() && j == java.len(),
        "{name}: length differs\n  vanilla: {}\n  ours:    {}",
        hex(java),
        hex(ours)
    );
    relaxed
}
