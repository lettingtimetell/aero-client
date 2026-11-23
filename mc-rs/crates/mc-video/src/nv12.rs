//! NV12: the frame layout the encoders take, and the CPU reference conversion.
//!
//! An NV12 frame of `w x h` (both even) is `w * h` bytes of luma (Y), one byte per pixel, rows
//! packed with a stride of `w`, followed by `w * h / 2` bytes of chroma: `h / 2` rows of `w / 2`
//! interleaved (U, V) pairs, one pair per 2x2 block of pixels.
//!
//! The colour conversion is BT.709 with limited ("TV") range, which is what the encoder is told
//! and what every player assumes for HD video: Y in 16..=235, U and V in 16..=240. The game
//! frame's bytes are already display-encoded (gamma), so they go straight into the matrix with no
//! transfer function. Chroma is the mean of the four pixels' U and V.
//!
//! The export's GPU compute shader (`mc-render`, `export/nv12.wgsl`) implements exactly this; a
//! test there compares the two byte for byte (within one code value of rounding).

/// Bytes in one NV12 frame.
#[must_use]
pub fn frame_bytes(width: u32, height: u32) -> usize {
    let (w, h) = (width as usize, height as usize);
    w * h + w * h / 2
}

/// Byte offset of the chroma plane.
#[must_use]
pub fn chroma_offset(width: u32, height: u32) -> usize {
    width as usize * height as usize
}

/// BT.709 luma weights.
const KR: f32 = 0.2126;
const KB: f32 = 0.0722;
const KG: f32 = 1.0 - KR - KB;

/// Limited-range Y, U and V (as floats, before rounding) of one display-encoded pixel, each
/// channel in 0..=1.
#[must_use]
pub fn yuv709_limited(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let y = KR * r + KG * g + KB * b;
    let u = (b - y) / (2.0 * (1.0 - KB));
    let v = (r - y) / (2.0 * (1.0 - KR));
    (16.0 + 219.0 * y, 128.0 + 224.0 * u, 128.0 + 224.0 * v)
}

/// Round to the nearest code value and clamp to a byte.
#[inline]
fn quantize(value: f32) -> u8 {
    (value + 0.5).floor().clamp(0.0, 255.0) as u8
}

/// Channel order of a 4-byte-per-pixel source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelOrder {
    /// R, G, B, A.
    Rgba,
    /// B, G, R, A (the usual Windows swap-chain order).
    Bgra,
}

/// Convert a 4-byte-per-pixel image to NV12.
///
/// `stride` is the source's bytes per row (at least `width * 4`). `out` must be exactly
/// [`frame_bytes`] long.
///
/// # Panics
/// If the buffers are too small or the size is odd.
pub fn from_rgba(
    src: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    order: PixelOrder,
    out: &mut [u8],
) {
    assert!(
        width.is_multiple_of(2) && height.is_multiple_of(2),
        "NV12 needs an even size"
    );
    let (w, h) = (width as usize, height as usize);
    assert!(stride >= w * 4, "stride too small");
    assert!(src.len() >= stride * (h - 1) + w * 4, "source too small");
    assert_eq!(
        out.len(),
        frame_bytes(width, height),
        "wrong NV12 buffer size"
    );
    let (luma, chroma) = out.split_at_mut(w * h);
    let (ri, bi) = match order {
        PixelOrder::Rgba => (0, 2),
        PixelOrder::Bgra => (2, 0),
    };
    let pixel = |x: usize, y: usize| {
        let at = y * stride + x * 4;
        (
            f32::from(src[at + ri]) / 255.0,
            f32::from(src[at + 1]) / 255.0,
            f32::from(src[at + bi]) / 255.0,
        )
    };
    for by in 0..h / 2 {
        for bx in 0..w / 2 {
            let mut u_sum = 0.0;
            let mut v_sum = 0.0;
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let (x, y) = (bx * 2 + dx, by * 2 + dy);
                let (r, g, b) = pixel(x, y);
                let (yy, u, v) = yuv709_limited(r, g, b);
                luma[y * w + x] = quantize(yy);
                u_sum += u;
                v_sum += v;
            }
            let at = by * w + bx * 2;
            chroma[at] = quantize(u_sum * 0.25);
            chroma[at + 1] = quantize(v_sum * 0.25);
        }
    }
}

/// Convert NV12 back to RGBA (BT.709 limited range), for tests and previews.
///
/// # Panics
/// If the buffers have the wrong size.
#[must_use]
pub fn to_rgba(nv12: &[u8], width: u32, height: u32) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    assert_eq!(
        nv12.len(),
        frame_bytes(width, height),
        "wrong NV12 buffer size"
    );
    let (luma, chroma) = nv12.split_at(w * h);
    let mut out = vec![0_u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let yy = (f32::from(luma[y * w + x]) - 16.0) / 219.0;
            let at = (y / 2) * w + (x / 2) * 2;
            let u = (f32::from(chroma[at]) - 128.0) / 224.0;
            let v = (f32::from(chroma[at + 1]) - 128.0) / 224.0;
            let r = yy + 2.0 * (1.0 - KR) * v;
            let b = yy + 2.0 * (1.0 - KB) * u;
            let g = (yy - KR * r - KB * b) / KG;
            let o = (y * w + x) * 4;
            out[o] = quantize(r * 255.0);
            out[o + 1] = quantize(g * 255.0);
            out[o + 2] = quantize(b * 255.0);
            out[o + 3] = 255;
        }
    }
    out
}

/// Peak signal-to-noise ratio between two equally sized byte images, in dB (`f64::INFINITY` when
/// identical). Used by the encode tests to check a decoded file against its source.
#[must_use]
pub fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "images differ in size");
    let mse = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum::<f64>()
        / a.len().max(1) as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_sizes() {
        assert_eq!(frame_bytes(1920, 1080), 3_110_400);
        assert_eq!(chroma_offset(1920, 1080), 2_073_600);
        assert_eq!(frame_bytes(2, 2), 6);
    }

    #[test]
    fn reference_colours_land_on_the_bt709_limited_codes() {
        // White, black and the primaries at full intensity: the textbook BT.709 8-bit values.
        let cases = [
            ([255, 255, 255], (235, 128, 128)),
            ([0, 0, 0], (16, 128, 128)),
            ([255, 0, 0], (63, 102, 240)),
            ([0, 255, 0], (173, 42, 26)),
            ([0, 0, 255], (32, 240, 118)),
        ];
        for (rgb, (y, u, v)) in cases {
            let src: Vec<u8> = (0..4).flat_map(|_| [rgb[0], rgb[1], rgb[2], 255]).collect();
            let mut out = vec![0_u8; frame_bytes(2, 2)];
            from_rgba(&src, 2, 2, 8, PixelOrder::Rgba, &mut out);
            assert_eq!(&out[..4], &[y; 4], "Y of {rgb:?}");
            assert_eq!((out[4], out[5]), (u, v), "UV of {rgb:?}");
        }
    }

    #[test]
    fn bgra_and_rgba_agree() {
        let rgba: Vec<u8> = (0..16 * 4 * 4).map(|i| (i * 37 % 256) as u8).collect();
        let mut bgra = rgba.clone();
        for px in bgra.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
        let mut a = vec![0_u8; frame_bytes(16, 4)];
        let mut b = vec![0_u8; frame_bytes(16, 4)];
        from_rgba(&rgba, 16, 4, 64, PixelOrder::Rgba, &mut a);
        from_rgba(&bgra, 16, 4, 64, PixelOrder::Bgra, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn round_trip_is_close_for_smooth_images() {
        let (w, h) = (64_u32, 32_u32);
        let mut src = vec![0_u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) * 4) as usize;
                src[o] = (x * 4) as u8;
                src[o + 1] = (y * 8) as u8;
                src[o + 2] = 128;
                src[o + 3] = 255;
            }
        }
        let mut nv = vec![0_u8; frame_bytes(w, h)];
        from_rgba(&src, w, h, (w * 4) as usize, PixelOrder::Rgba, &mut nv);
        let back = to_rgba(&nv, w, h);
        let p = psnr(&src, &back);
        assert!(p > 38.0, "round trip PSNR {p:.1} dB");
    }
}
