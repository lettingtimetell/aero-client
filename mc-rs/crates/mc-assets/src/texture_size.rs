//! PNGs that become a texture of their own (GUI sheets, entity, sky, font and particle
//! textures), refused at decode when a side is larger than the graphics device can hold.
//!
//! A texture larger than the device's `max_texture_dimension_2d` cannot be created: the device
//! reports a validation error, which ends the process unless a caller installed an error scope.
//! Vanilla 1.8.9's `glTexImage2D` only records a GL error there and leaves the texture blank.
//! A resource pack (the player's own, or one a server offered) is a few hundred bytes away from
//! such a texture (a 40000x1 PNG decodes to 160 KB), and it stays selected across restarts, so
//! the client would fail on every start. Here the PNG's header is refused before any pixel is
//! allocated, with the error type [`image::load_from_memory_with_format`] returns, and every
//! caller already treats a PNG that does not decode as a missing texture.
//!
//! Atlas sprites do not come through here: an animation strip is taller than the frames the
//! atlas holds, and the stitcher refuses a frame that does not fit.

use std::io::Cursor;
use std::sync::atomic::{AtomicU32, Ordering};

/// No graphics device has reported its limit yet.
const UNLIMITED: u32 = u32::MAX;

/// The smallest `max_texture_dimension_2d` of the devices created in this process.
static MAX_TEXTURE_SIDE: AtomicU32 = AtomicU32::new(UNLIMITED);

/// Record a graphics device's largest 2D texture side (`max_texture_dimension_2d`). With several
/// devices (an export that renders on another adapter) the smallest wins, so a texture decoded
/// once fits on each of them.
pub fn limit_texture_side(max_side: u32) {
    MAX_TEXTURE_SIDE.fetch_min(max_side.max(1), Ordering::Relaxed);
}

/// The largest side [`decode_texture_png`] accepts: the smallest limit given to
/// [`limit_texture_side`], or `u32::MAX` (no limit) before any device exists.
#[must_use]
pub fn texture_side_limit() -> u32 {
    MAX_TEXTURE_SIDE.load(Ordering::Relaxed)
}

/// Decode a PNG that becomes a texture of its own, refusing one with a side over
/// [`texture_side_limit`] from its header.
///
/// # Errors
///
/// As [`image::load_from_memory_with_format`], plus [`image::ImageError::Limits`] for a side
/// over the limit. The image crate's own allocation cap (512 MiB) applies as before.
pub fn decode_texture_png(bytes: &[u8]) -> image::ImageResult<image::DynamicImage> {
    decode_png_within(bytes, texture_side_limit())
}

/// [`decode_texture_png`] with an explicit side limit (`u32::MAX`: none).
///
/// # Errors
///
/// As [`decode_texture_png`].
pub fn decode_png_within(bytes: &[u8], max_side: u32) -> image::ImageResult<image::DynamicImage> {
    let mut reader = image::ImageReader::with_format(Cursor::new(bytes), image::ImageFormat::Png);
    // `Limits::default()` is what `load_from_memory_with_format` decodes with (its 512 MiB
    // allocation cap); only the two sides are added.
    let mut limits = image::Limits::default();
    if max_side != UNLIMITED {
        limits.max_image_width = Some(max_side);
        limits.max_image_height = Some(max_side);
    }
    reader.limits(limits);
    reader.decode()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(width: u32, height: u32) -> Vec<u8> {
        let image = image::RgbaImage::from_pixel(width, height, image::Rgba([1, 2, 3, 4]));
        let mut out = Cursor::new(Vec::new());
        image
            .write_to(&mut out, image::ImageFormat::Png)
            .expect("encode test PNG");
        out.into_inner()
    }

    #[test]
    fn a_side_over_the_limit_is_refused_and_one_at_it_decodes() {
        let at_limit = decode_png_within(&png(64, 3), 64).expect("64 wide fits a 64 limit");
        assert_eq!((at_limit.width(), at_limit.height()), (64, 3));
        for (width, height) in [(65, 3), (3, 65)] {
            let error = decode_png_within(&png(width, height), 64).expect_err("over the limit");
            assert!(
                matches!(error, image::ImageError::Limits(_)),
                "{width}x{height}: {error}"
            );
        }
    }

    #[test]
    fn without_a_limit_it_decodes_exactly_like_the_plain_decoder() {
        let bytes = png(300, 7);
        let ours = decode_png_within(&bytes, UNLIMITED).expect("decode");
        let plain =
            image::load_from_memory_with_format(&bytes, image::ImageFormat::Png).expect("decode");
        assert_eq!(ours.to_rgba8().into_raw(), plain.to_rgba8().into_raw());
        assert!(decode_png_within(b"not a png", UNLIMITED).is_err());
    }
}
