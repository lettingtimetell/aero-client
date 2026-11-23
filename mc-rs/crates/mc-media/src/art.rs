//! Album covers: JPEG or PNG bytes to straight RGBA8, shrunk to fit [`MAX_SIDE`] on each side.
//!
//! Runs on the media worker only, and only when the cover's bytes change.

use std::io::Cursor;

use image::imageops::FilterType;
use image::{ImageFormat, ImageReader, Limits, RgbaImage};
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

use crate::AlbumArt;

/// Covers are shrunk to fit this many pixels on their longer side (never enlarged).
pub(crate) const MAX_SIDE: u32 = 128;

/// Larger thumbnails are not decoded. Spotify desktop's is a 300x300 PNG of 60-120 KB (measured
/// on 2026-11-07); browsers send JPEGs.
pub(crate) const MAX_ENCODED_BYTES: usize = 16 << 20;

/// Larger source images are refused before decoding, which bounds the decode's memory.
/// 4096 (was 8192): with the "any player" source, a web page's
/// artwork reaches this decoder, and an 8192-pixel JPEG cost about 450 MB (RGB, then RGBA) on
/// each cover change, for a picture drawn at 128 pixels. Web artwork is 3000 pixels at most.
const MAX_SOURCE_SIDE: u32 = 4096;

const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
const JPEG_SIGNATURE: &[u8] = &[0xFF, 0xD8, 0xFF];

/// A cheap fingerprint of encoded cover bytes (64-bit FNV-1a over the length and the bytes), so
/// an unchanged cover is neither decoded again nor reported as new.
#[must_use]
pub(crate) fn fingerprint(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let length = (bytes.len() as u64).to_le_bytes();
    length.iter().chain(bytes).fold(OFFSET, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(PRIME)
    })
}

/// Decodes a JPEG or PNG cover (recognised by its signature, not by a declared content type) and
/// shrinks it to fit `MAX_SIDE`. `None` for anything else, for corrupt data, and for oversized
/// images. Never panics on bad input.
#[must_use]
pub(crate) fn decode_cover(bytes: &[u8]) -> Option<AlbumArt> {
    if bytes.len() > MAX_ENCODED_BYTES {
        return None;
    }
    let (width, height, rgba) = if bytes.starts_with(JPEG_SIGNATURE) {
        decode_jpeg(bytes)?
    } else if bytes.starts_with(PNG_SIGNATURE) {
        decode_png(bytes)?
    } else {
        return None;
    };
    fit(width, height, rgba)
}

fn decode_jpeg(bytes: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let side = MAX_SOURCE_SIDE as usize;
    // RGB output, then alpha added here: zune-jpeg converts every input colour space (YCbCr,
    // greyscale, CMYK, YCCK, RGB) to RGB, but has no RGB -> RGBA path.
    let options = DecoderOptions::default()
        .jpeg_set_out_colorspace(ColorSpace::RGB)
        .set_max_width(side)
        .set_max_height(side);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    let rgb = decoder.decode().ok()?;
    let (width, height) = decoder.dimensions()?;
    if width == 0 || height == 0 || rgb.len() != width.checked_mul(height)?.checked_mul(3)? {
        return None;
    }
    let mut rgba = Vec::with_capacity(rgb.len() / 3 * 4);
    for pixel in rgb.chunks_exact(3) {
        rgba.extend_from_slice(&[pixel[0], pixel[1], pixel[2], u8::MAX]);
    }
    Some((
        u32::try_from(width).ok()?,
        u32::try_from(height).ok()?,
        rgba,
    ))
}

fn decode_png(bytes: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let mut reader = ImageReader::with_format(Cursor::new(bytes), ImageFormat::Png);
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_SOURCE_SIDE);
    limits.max_image_height = Some(MAX_SOURCE_SIDE);
    limits.max_alloc = Some(512 << 20);
    reader.limits(limits);
    let image = reader.decode().ok()?.into_rgba8();
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return None;
    }
    Some((width, height, image.into_raw()))
}

/// Shrinks to fit `MAX_SIDE` on both sides, keeping the aspect ratio. Smaller covers are kept.
fn fit(width: u32, height: u32, rgba: Vec<u8>) -> Option<AlbumArt> {
    if width <= MAX_SIDE && height <= MAX_SIDE {
        return Some(AlbumArt {
            width,
            height,
            rgba,
        });
    }
    let (new_width, new_height) = fitted_size(width, height);
    let source = RgbaImage::from_raw(width, height, rgba)?;
    // Triangle widens its footprint with the reduction, so the shrink is properly averaged.
    let shrunk = image::imageops::resize(&source, new_width, new_height, FilterType::Triangle);
    Some(AlbumArt {
        width: new_width,
        height: new_height,
        rgba: shrunk.into_raw(),
    })
}

/// The size `width` x `height` shrinks to so that both sides fit `MAX_SIDE`.
#[must_use]
fn fitted_size(width: u32, height: u32) -> (u32, u32) {
    let max = f64::from(MAX_SIDE);
    let scale = (max / f64::from(width)).min(max / f64::from(height));
    let side = |length: u32| {
        // The rounded value lies in 0..=MAX_SIDE, so the cast cannot truncate.
        ((f64::from(length) * scale).round() as u32).clamp(1, MAX_SIDE)
    };
    (side(width), side(height))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 40x30 baseline JPEG (System.Drawing, quality 90): left half pure red (255,0,0), right half
    /// pure blue (0,0,255).
    const RED_BLUE_JPEG: &[u8] = include_bytes!("../testdata/red-blue-40x30.jpg");

    fn png(width: u32, height: u32, pixel: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
        let image = RgbaImage::from_fn(width, height, |x, y| image::Rgba(pixel(x, y)));
        let mut bytes = Vec::new();
        image
            .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
            .expect("encode test PNG");
        bytes
    }

    fn pixel(art: &AlbumArt, x: u32, y: u32) -> [u8; 4] {
        let at = ((y * art.width + x) * 4) as usize;
        [
            art.rgba[at],
            art.rgba[at + 1],
            art.rgba[at + 2],
            art.rgba[at + 3],
        ]
    }

    #[test]
    fn a_spotify_sized_png_shrinks_to_128_and_keeps_its_colours() {
        // Spotify's thumbnails are 300x300. Left half white, right half black.
        let bytes = png(
            300,
            300,
            |x, _| if x < 150 { [255; 4] } else { [0, 0, 0, 255] },
        );
        let art = decode_cover(&bytes).expect("decodes");
        assert_eq!((art.width, art.height), (128, 128));
        assert_eq!(art.rgba.len(), 128 * 128 * 4);
        assert_eq!(pixel(&art, 10, 64), [255, 255, 255, 255]);
        assert_eq!(pixel(&art, 117, 64), [0, 0, 0, 255]);
    }

    #[test]
    fn shrinking_keeps_the_aspect_ratio_and_small_covers_are_kept_as_they_are() {
        assert_eq!(fitted_size(300, 300), (128, 128));
        assert_eq!(fitted_size(320, 180), (128, 72));
        assert_eq!(fitted_size(180, 320), (72, 128));
        assert_eq!(fitted_size(4000, 10), (128, 1), "never 0 pixels high");
        let small = decode_cover(&png(64, 48, |_, _| [1, 2, 3, 4])).expect("decodes");
        assert_eq!((small.width, small.height), (64, 48));
        assert_eq!(pixel(&small, 63, 47), [1, 2, 3, 4], "straight alpha kept");
    }

    #[test]
    fn a_jpeg_cover_decodes_to_rgba() {
        let art = decode_cover(RED_BLUE_JPEG).expect("decodes");
        assert_eq!((art.width, art.height), (40, 30));
        assert_eq!(art.rgba.len(), 40 * 30 * 4);
        let close = |got: [u8; 4], want: [u8; 3]| {
            got[3] == 255 && got.iter().zip(want).all(|(&g, w)| g.abs_diff(w) <= 12)
        };
        let left = pixel(&art, 5, 15);
        let right = pixel(&art, 34, 15);
        assert!(close(left, [255, 0, 0]), "left half is red, got {left:?}");
        assert!(
            close(right, [0, 0, 255]),
            "right half is blue, got {right:?}"
        );
    }

    #[test]
    fn garbage_truncated_and_unknown_data_give_no_cover_without_panicking() {
        assert!(decode_cover(&[]).is_none());
        assert!(decode_cover(b"GIF89a....").is_none(), "only JPEG and PNG");
        assert!(decode_cover(&RED_BLUE_JPEG[..RED_BLUE_JPEG.len() / 3]).is_none());
        let bytes = png(300, 300, |_, _| [9; 4]);
        assert!(decode_cover(&bytes[..bytes.len() / 2]).is_none());
        let mut junk = PNG_SIGNATURE.to_vec();
        junk.extend(std::iter::repeat_n(0xAB, 500));
        assert!(decode_cover(&junk).is_none());
        let mut junk = JPEG_SIGNATURE.to_vec();
        junk.extend(std::iter::repeat_n(0x00, 500));
        assert!(decode_cover(&junk).is_none());
    }

    /// The side limit holds for both formats' headers: one pixel
    /// over it is refused before decoding, at it a cover still decodes.
    #[test]
    fn a_cover_over_the_source_side_is_refused() {
        let wide = png(MAX_SOURCE_SIDE + 1, 1, |_, _| [9, 9, 9, 255]);
        assert!(decode_cover(&wide).is_none());
        let at_limit = png(MAX_SOURCE_SIDE, 1, |_, _| [9, 9, 9, 255]);
        let art = decode_cover(&at_limit).expect("decodes");
        assert_eq!(art.width, MAX_SIDE);
        // A JPEG header claiming one pixel more (the start-of-frame width field).
        let mut jpeg = RED_BLUE_JPEG.to_vec();
        let sof = jpeg
            .windows(2)
            .position(|marker| marker == [0xFF, 0xC0])
            .expect("a baseline frame header");
        let too_wide = u16::try_from(MAX_SOURCE_SIDE + 1)
            .expect("fits")
            .to_be_bytes();
        jpeg[sof + 7..sof + 9].copy_from_slice(&too_wide);
        assert!(decode_cover(&jpeg).is_none());
    }

    #[test]
    fn fingerprints_tell_covers_apart() {
        let a = png(8, 8, |_, _| [1, 2, 3, 255]);
        let b = png(8, 8, |_, _| [1, 2, 4, 255]);
        assert_eq!(fingerprint(&a), fingerprint(&a.clone()));
        assert_ne!(fingerprint(&a), fingerprint(&b));
        assert_ne!(fingerprint(&[]), fingerprint(&[0]));
    }
}
