//! Frame snapshots and a **perceptual** image diff.
//!
//! # Why not a byte compare
//!
//! A byte compare of two rendered frames answers the wrong question in both directions. It
//! fails on differences no human could see — a ±1 least-significant bit from a different
//! rasteriser's rounding, a driver that resolves multisampling a hair differently — and it
//! passes nothing useful once it fails, because "1,048,576 bytes differ" says nothing about
//! *how*. Meanwhile it is entirely blind to structure: an image shifted one pixel to the right
//! is catastrophically wrong and byte-different in exactly the same way that an imperceptible
//! dither is byte-different.
//!
//! # The two metrics, and why both
//!
//! **CIEDE2000 (ΔE₀₀)** — the CIE's 2000 colour-difference formula, computed per pixel after
//! sRGB → linear → CIEXYZ(D65) → CIELAB. It is the standard answer to "would a person call
//! these two colours different", and it is calibrated so that **ΔE₀₀ ≈ 1 is one just-noticeable
//! difference**. That calibration is the whole point: it gives the threshold a meaning
//! ("nothing in this frame is visibly the wrong colour") instead of an arbitrary tolerance in
//! byte units. It was chosen over the older CIE76 ΔE*ab, which is badly non-uniform in the
//! blues and would wave through exactly the sky-colour errors this project cares about, and
//! over full perceptual image metrics, which are more accurate but far more implementation than
//! a test harness can carry as hand-written code.
//!
//! **SSIM** — structural similarity, over 8×8 windows on Rec.709 luma. ΔE₀₀ is per pixel and
//! therefore structure-blind: shift an image one pixel and every pixel is a little wrong, which
//! averages down to something that looks fine. SSIM collapses under exactly that failure, and
//! is insensitive to the uniform brightness/contrast offsets ΔE₀₀ already catches. The pair
//! covers what either alone misses; [`PerceptualDiff::is_within`] requires both.
//!
//! # Reference evidence
//!
//! Synthetic pairs and analytical GPU clear colours validate these metrics. Java-versus-Rust
//! parity additionally requires matched-scene PNGs, identical camera/settings, and retained
//! diff reports. A Rust frame compared with its own output establishes regression coverage.
//! The compare_snapshots example compares supplied PNGs using default thresholds.

use std::path::Path;

use mc_render::Gpu;

use crate::ParityError;

/// A captured RGBA8 image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Tightly packed RGBA8 rows, top row first — the layout
    /// [`Gpu::read_pixels`](mc_render::Gpu::read_pixels) produces.
    pub rgba: Vec<u8>,
}

impl Snapshot {
    /// Wrap raw pixel data.
    ///
    /// # Errors
    /// [`ParityError::ShapeMismatch`] if `rgba` is not exactly `width * height * 4` bytes.
    pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> Result<Snapshot, ParityError> {
        let expected = width as usize * height as usize * 4;
        if rgba.len() != expected {
            return Err(ParityError::ShapeMismatch(format!(
                "a {width}×{height} RGBA8 image needs {expected} bytes, got {}",
                rgba.len()
            )));
        }
        Ok(Snapshot {
            width,
            height,
            rgba,
        })
    }

    /// A solid-colour image, for building synthetic references.
    #[must_use]
    pub fn solid(width: u32, height: u32, colour: [u8; 4]) -> Snapshot {
        Snapshot {
            width,
            height,
            rgba: colour.repeat(width as usize * height as usize),
        }
    }

    /// Read the offscreen colour target back from a headless [`Gpu`].
    ///
    /// # Errors
    /// [`ParityError::Render`] if the context has a surface (so there is no offscreen texture)
    /// or the readback fails, and [`ParityError::ShapeMismatch`] if the returned buffer is not
    /// the size the surface configuration implies.
    pub fn read_back(gpu: &Gpu) -> Result<Snapshot, ParityError> {
        let rgba = gpu.read_pixels()?;
        Snapshot::new(gpu.config.width, gpu.config.height, rgba)
    }

    /// One pixel, or `None` when out of bounds.
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let i = (y as usize * self.width as usize + x as usize) * 4;
        Some([
            self.rgba[i],
            self.rgba[i + 1],
            self.rgba[i + 2],
            self.rgba[i + 3],
        ])
    }

    /// Overwrite one pixel. Out-of-bounds coordinates are ignored.
    pub fn set_pixel(&mut self, x: u32, y: u32, colour: [u8; 4]) {
        if x >= self.width || y >= self.height {
            return;
        }
        let i = (y as usize * self.width as usize + x as usize) * 4;
        self.rgba[i..i + 4].copy_from_slice(&colour);
    }

    /// Decode a PNG.
    ///
    /// # Errors
    /// [`ParityError::Io`] if the file cannot be read, [`ParityError::Image`] if it is not a
    /// decodable PNG.
    pub fn load_png(path: impl AsRef<Path>) -> Result<Snapshot, ParityError> {
        let bytes = std::fs::read(path)?;
        let image = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
            .map_err(|e| ParityError::Image(e.to_string()))?
            .to_rgba8();
        let (width, height) = image.dimensions();
        Snapshot::new(width, height, image.into_raw())
    }

    /// Encode as a PNG.
    ///
    /// # Errors
    /// [`ParityError::Image`] if encoding fails, [`ParityError::Io`] if the file cannot be
    /// written.
    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<(), ParityError> {
        let buffer: image::RgbaImage =
            image::ImageBuffer::from_raw(self.width, self.height, self.rgba.clone()).ok_or_else(
                || ParityError::Image("pixel buffer does not match the declared size".into()),
            )?;
        buffer
            .save_with_format(path, image::ImageFormat::Png)
            .map_err(|e| ParityError::Image(e.to_string()))
    }
}

/// What the two metrics said.
#[derive(Debug, Clone, PartialEq)]
pub struct PerceptualDiff {
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// The largest per-pixel CIEDE2000 difference.
    pub max_delta_e: f64,
    /// The mean per-pixel CIEDE2000 difference.
    pub mean_delta_e: f64,
    /// How many pixels exceeded one just-noticeable difference (ΔE₀₀ > 1).
    pub pixels_over_jnd: usize,
    /// `pixels_over_jnd` as a fraction of the image.
    pub fraction_over_jnd: f64,
    /// Mean structural similarity over 8×8 luma windows. `1.0` is identical.
    pub ssim: f64,
    /// The largest absolute difference in any single 8-bit channel, alpha included. Reported
    /// so a byte-level regression is still visible next to the perceptual verdict.
    pub max_channel_delta: u8,
    /// The largest absolute alpha difference on its own — a transparency regression is
    /// invisible to a colour metric that only looks at RGB.
    pub max_alpha_delta: u8,
    /// How many pixels differ in any byte at all.
    pub differing_pixels: usize,
    /// Where `max_delta_e` was found.
    pub worst_pixel: (u32, u32),
}

/// The thresholds a comparison has to satisfy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiffThresholds {
    /// No pixel may exceed this ΔE₀₀. `1.0` is one just-noticeable difference.
    pub max_delta_e: f64,
    /// At most this fraction of pixels may exceed one JND. A few isolated pixels over the line
    /// is normal for a different rasteriser's edge coverage; a percent of them is not.
    pub max_fraction_over_jnd: f64,
    /// Mean SSIM must be at least this.
    pub min_ssim: f64,
    /// Alpha must not move by more than this many 8-bit steps.
    pub max_alpha_delta: u8,
}

impl Default for DiffThresholds {
    /// Tight enough that a real difference fails, loose enough that quantisation noise does
    /// not: one JND anywhere, and structure essentially intact.
    fn default() -> Self {
        DiffThresholds {
            max_delta_e: 1.0,
            max_fraction_over_jnd: 0.0,
            min_ssim: 0.99,
            max_alpha_delta: 1,
        }
    }
}

impl PerceptualDiff {
    /// Whether this comparison passes `thresholds`.
    #[must_use]
    pub fn is_within(&self, thresholds: &DiffThresholds) -> bool {
        self.max_delta_e <= thresholds.max_delta_e
            && self.fraction_over_jnd <= thresholds.max_fraction_over_jnd
            && self.ssim >= thresholds.min_ssim
            && self.max_alpha_delta <= thresholds.max_alpha_delta
    }

    /// Whether the two images were byte-identical.
    #[must_use]
    pub fn is_identical(&self) -> bool {
        self.differing_pixels == 0
    }

    /// A one-line summary suitable for a test's stdout.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{}×{}: max ΔE00 {:.4} at {:?}, mean {:.4}, {} px over JND ({:.4}%), SSIM {:.6}, \
             max channel Δ {}, {} differing px",
            self.width,
            self.height,
            self.max_delta_e,
            self.worst_pixel,
            self.mean_delta_e,
            self.pixels_over_jnd,
            self.fraction_over_jnd * 100.0,
            self.ssim,
            self.max_channel_delta,
            self.differing_pixels
        )
    }
}

/// Compare two snapshots perceptually.
///
/// # Errors
/// [`ParityError::ShapeMismatch`] if the two images are different sizes, or if either is empty.
pub fn perceptual_diff(a: &Snapshot, b: &Snapshot) -> Result<PerceptualDiff, ParityError> {
    if a.width != b.width || a.height != b.height {
        return Err(ParityError::ShapeMismatch(format!(
            "{}×{} against {}×{}",
            a.width, a.height, b.width, b.height
        )));
    }
    if a.width == 0 || a.height == 0 {
        return Err(ParityError::ShapeMismatch(
            "an empty image has nothing to compare".into(),
        ));
    }

    let mut max_delta_e = 0.0f64;
    let mut sum_delta_e = 0.0f64;
    let mut pixels_over_jnd = 0usize;
    let mut max_channel_delta = 0u8;
    let mut max_alpha_delta = 0u8;
    let mut differing_pixels = 0usize;
    let mut worst_pixel = (0u32, 0u32);

    for y in 0..a.height {
        for x in 0..a.width {
            let i = (y as usize * a.width as usize + x as usize) * 4;
            let pa = [a.rgba[i], a.rgba[i + 1], a.rgba[i + 2], a.rgba[i + 3]];
            let pb = [b.rgba[i], b.rgba[i + 1], b.rgba[i + 2], b.rgba[i + 3]];

            if pa != pb {
                differing_pixels += 1;
                for c in 0..4 {
                    max_channel_delta = max_channel_delta.max(pa[c].abs_diff(pb[c]));
                }
                max_alpha_delta = max_alpha_delta.max(pa[3].abs_diff(pb[3]));
            }

            let de = ciede2000(
                rgb_to_lab([pa[0], pa[1], pa[2]]),
                rgb_to_lab([pb[0], pb[1], pb[2]]),
            );
            sum_delta_e += de;
            if de > max_delta_e {
                max_delta_e = de;
                worst_pixel = (x, y);
            }
            if de > 1.0 {
                pixels_over_jnd += 1;
            }
        }
    }

    let pixels = a.width as usize * a.height as usize;
    Ok(PerceptualDiff {
        width: a.width,
        height: a.height,
        max_delta_e,
        mean_delta_e: sum_delta_e / pixels as f64,
        pixels_over_jnd,
        fraction_over_jnd: pixels_over_jnd as f64 / pixels as f64,
        ssim: ssim(a, b),
        max_channel_delta,
        max_alpha_delta,
        differing_pixels,
        worst_pixel,
    })
}

/// Mean structural similarity over 8×8 luma windows, stepping 4 pixels.
///
/// Returns `1.0` for identical images. Images smaller than one window are compared as a single
/// window rather than skipped, so a tiny test image still gets an answer.
#[must_use]
pub fn ssim(a: &Snapshot, b: &Snapshot) -> f64 {
    // The standard 8-bit constants: C1 = (0.01·L)², C2 = (0.03·L)², L = 255.
    const C1: f64 = 6.502_5;
    const C2: f64 = 58.522_5;
    const WINDOW: u32 = 8;
    const STEP: u32 = 4;

    let luma = |s: &Snapshot, x: u32, y: u32| -> f64 {
        let i = (y as usize * s.width as usize + x as usize) * 4;
        0.212_6 * f64::from(s.rgba[i])
            + 0.715_2 * f64::from(s.rgba[i + 1])
            + 0.072_2 * f64::from(s.rgba[i + 2])
    };

    let w = WINDOW.min(a.width);
    let h = WINDOW.min(a.height);
    let mut total = 0.0f64;
    let mut windows = 0usize;

    let mut y0 = 0;
    while y0 + h <= a.height {
        let mut x0 = 0;
        while x0 + w <= a.width {
            let n = f64::from(w * h);
            let (mut sum_a, mut sum_b) = (0.0, 0.0);
            for y in y0..y0 + h {
                for x in x0..x0 + w {
                    sum_a += luma(a, x, y);
                    sum_b += luma(b, x, y);
                }
            }
            let (mean_a, mean_b) = (sum_a / n, sum_b / n);

            let (mut var_a, mut var_b, mut cov) = (0.0, 0.0, 0.0);
            for y in y0..y0 + h {
                for x in x0..x0 + w {
                    let (da, db) = (luma(a, x, y) - mean_a, luma(b, x, y) - mean_b);
                    var_a += da * da;
                    var_b += db * db;
                    cov += da * db;
                }
            }
            // Sample variance (n-1), which is what the reference implementation uses.
            let denom = (n - 1.0).max(1.0);
            var_a /= denom;
            var_b /= denom;
            cov /= denom;

            let numerator = (2.0 * mean_a * mean_b + C1) * (2.0 * cov + C2);
            let denominator = (mean_a * mean_a + mean_b * mean_b + C1) * (var_a + var_b + C2);
            total += numerator / denominator;
            windows += 1;

            x0 += STEP;
        }
        y0 += STEP;
    }

    if windows == 0 {
        return 1.0;
    }
    total / windows as f64
}

/// A CIELAB colour: `L*`, `a*`, `b*`.
pub type Lab = [f64; 3];

/// Convert an 8-bit sRGB triple to CIELAB under the D65 white point.
#[must_use]
pub fn rgb_to_lab(rgb: [u8; 3]) -> Lab {
    // sRGB IEC 61966-2-1 transfer function.
    let linear = |c: u8| {
        let c = f64::from(c) / 255.0;
        if c <= 0.040_45 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let (r, g, b) = (linear(rgb[0]), linear(rgb[1]), linear(rgb[2]));

    // sRGB primaries → CIEXYZ, D65.
    let x = 0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b;
    let y = 0.212_672_9 * r + 0.715_152_2 * g + 0.072_175_0 * b;
    let z = 0.019_333_9 * r + 0.119_192_0 * g + 0.950_304_1 * b;

    // D65. These are the row sums of the matrix above rather than the rounded book values:
    // the published sRGB matrix's middle row sums to 1.0000001, not 1, so dividing by a flat
    // `Yn = 1.0` puts pure white at L* = 100.0000039 instead of 100. The error is ~4e-8
    // relative and could not matter against a JND of 1, but a white point that is not the
    // matrix's own white point is simply an inconsistency, and it is free to not have one.
    const XN: f64 = 0.412_456_4 + 0.357_576_1 + 0.180_437_5;
    const YN: f64 = 0.212_672_9 + 0.715_152_2 + 0.072_175_0;
    const ZN: f64 = 0.019_333_9 + 0.119_192_0 + 0.950_304_1;
    // δ = 6/29; the linear segment below δ³ keeps the function differentiable at the join.
    let f = |t: f64| {
        const DELTA: f64 = 6.0 / 29.0;
        if t > DELTA * DELTA * DELTA {
            t.cbrt()
        } else {
            t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
        }
    };
    let (fx, fy, fz) = (f(x / XN), f(y / YN), f(z / ZN));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// The CIEDE2000 colour-difference ΔE₀₀ between two CIELAB colours, with `kL = kC = kH = 1`.
///
/// Implemented from the CIE's formulation as published by Sharma, Wu and Dalal (2005),
/// *"The CIEDE2000 color-difference formula: implementation notes, supplementary test data,
/// and mathematical observations"*. That paper's 34-pair supplementary test set is what
/// `tests/snapshot.rs` checks this against — it exists specifically because the formula has
/// three discontinuity traps (the hue-mean quadrant rule, the `Δh'` wrap, and the
/// `C1'·C2' = 0` special case) that a naive transcription gets wrong on pairs a natural test
/// would never think to try.
#[must_use]
pub fn ciede2000(lab1: Lab, lab2: Lab) -> f64 {
    let (l1, a1, b1) = (lab1[0], lab1[1], lab1[2]);
    let (l2, a2, b2) = (lab2[0], lab2[1], lab2[2]);

    let c1 = a1.hypot(b1);
    let c2 = a2.hypot(b2);
    let c_bar = (c1 + c2) / 2.0;

    let c_bar7 = c_bar.powi(7);
    let g = 0.5 * (1.0 - (c_bar7 / (c_bar7 + 25f64.powi(7))).sqrt());

    let a1p = (1.0 + g) * a1;
    let a2p = (1.0 + g) * a2;
    let c1p = a1p.hypot(b1);
    let c2p = a2p.hypot(b2);

    // atan2 in degrees, folded to [0, 360). The `a' = b = 0` case is defined as 0 by the
    // standard, which `atan2(0, 0) = 0` already gives.
    let hue = |b: f64, ap: f64| {
        if b == 0.0 && ap == 0.0 {
            0.0
        } else {
            let d = b.atan2(ap).to_degrees();
            if d < 0.0 {
                d + 360.0
            } else {
                d
            }
        }
    };
    let h1p = hue(b1, a1p);
    let h2p = hue(b2, a2p);

    let dlp = l2 - l1;
    let dcp = c2p - c1p;

    let dhp = if c1p * c2p == 0.0 {
        0.0
    } else {
        let d = h2p - h1p;
        if d > 180.0 {
            d - 360.0
        } else if d < -180.0 {
            d + 360.0
        } else {
            d
        }
    };
    let dhp_big = 2.0 * (c1p * c2p).sqrt() * (dhp.to_radians() / 2.0).sin();

    let lp_bar = (l1 + l2) / 2.0;
    let cp_bar = (c1p + c2p) / 2.0;

    let hp_bar = if c1p * c2p == 0.0 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        (h1p + h2p) / 2.0
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) / 2.0
    } else {
        (h1p + h2p - 360.0) / 2.0
    };

    let t = 1.0 - 0.17 * (hp_bar - 30.0).to_radians().cos()
        + 0.24 * (2.0 * hp_bar).to_radians().cos()
        + 0.32 * (3.0 * hp_bar + 6.0).to_radians().cos()
        - 0.20 * (4.0 * hp_bar - 63.0).to_radians().cos();

    let d_theta = 30.0 * (-(((hp_bar - 275.0) / 25.0).powi(2))).exp();
    let cp_bar7 = cp_bar.powi(7);
    let rc = 2.0 * (cp_bar7 / (cp_bar7 + 25f64.powi(7))).sqrt();

    let sl = 1.0 + (0.015 * (lp_bar - 50.0).powi(2)) / (20.0 + (lp_bar - 50.0).powi(2)).sqrt();
    let sc = 1.0 + 0.045 * cp_bar;
    let sh = 1.0 + 0.015 * cp_bar * t;
    let rt = -(2.0 * d_theta).to_radians().sin() * rc;

    let term_l = dlp / sl;
    let term_c = dcp / sc;
    let term_h = dhp_big / sh;

    (term_l * term_l + term_c * term_c + term_h * term_h + rt * term_c * term_h).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_endpoints_land_on_the_lab_values_they_must() {
        let white = rgb_to_lab([255, 255, 255]);
        assert!((white[0] - 100.0).abs() < 1e-6, "white L* = {}", white[0]);
        assert!(white[1].abs() < 0.01 && white[2].abs() < 0.02, "{white:?}");

        let black = rgb_to_lab([0, 0, 0]);
        assert!(black[0].abs() < 1e-9, "black L* = {}", black[0]);

        // Mid grey 128/255 is well-known to sit near L* 53.585 — a sanity check that the
        // transfer function is applied and not skipped.
        let grey = rgb_to_lab([128, 128, 128]);
        assert!((grey[0] - 53.585).abs() < 0.01, "grey L* = {}", grey[0]);
        assert!(grey[1].abs() < 0.01 && grey[2].abs() < 0.02, "{grey:?}");
    }

    #[test]
    fn a_colour_has_zero_difference_from_itself() {
        for rgb in [[0, 0, 0], [255, 255, 255], [17, 200, 91], [200, 30, 30]] {
            let lab = rgb_to_lab(rgb);
            assert_eq!(ciede2000(lab, lab), 0.0, "{rgb:?}");
        }
    }

    #[test]
    fn ciede2000_is_symmetric() {
        let a = rgb_to_lab([12, 200, 77]);
        let b = rgb_to_lab([90, 40, 210]);
        let (ab, ba) = (ciede2000(a, b), ciede2000(b, a));
        assert!((ab - ba).abs() < 1e-12, "{ab} vs {ba}");
    }

    #[test]
    fn identical_images_are_identical_on_both_metrics() {
        let a = Snapshot::solid(16, 16, [40, 90, 160, 255]);
        let d = perceptual_diff(&a, &a).expect("diff");
        assert!(d.is_identical());
        assert_eq!(d.max_delta_e, 0.0);
        assert!((d.ssim - 1.0).abs() < 1e-9, "ssim {}", d.ssim);
        assert!(d.is_within(&DiffThresholds::default()));
    }

    #[test]
    fn images_of_different_sizes_are_a_shape_error() {
        let a = Snapshot::solid(4, 4, [0, 0, 0, 255]);
        let b = Snapshot::solid(4, 5, [0, 0, 0, 255]);
        assert!(matches!(
            perceptual_diff(&a, &b),
            Err(ParityError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn alpha_is_reported_separately_because_the_colour_metric_ignores_it() {
        let a = Snapshot::solid(8, 8, [10, 20, 30, 255]);
        let b = Snapshot::solid(8, 8, [10, 20, 30, 0]);
        let d = perceptual_diff(&a, &b).expect("diff");
        assert_eq!(d.max_delta_e, 0.0, "RGB really is unchanged");
        assert_eq!(d.max_alpha_delta, 255);
        assert!(
            !d.is_within(&DiffThresholds::default()),
            "a fully transparent frame must not pass as identical"
        );
    }

    #[test]
    fn snapshot_new_rejects_a_wrongly_sized_buffer() {
        assert!(matches!(
            Snapshot::new(2, 2, vec![0; 15]),
            Err(ParityError::ShapeMismatch(_))
        ));
        assert!(Snapshot::new(2, 2, vec![0; 16]).is_ok());
    }

    #[test]
    fn pixel_access_is_bounds_checked() {
        let mut s = Snapshot::solid(3, 2, [1, 2, 3, 4]);
        assert_eq!(s.pixel(2, 1), Some([1, 2, 3, 4]));
        assert_eq!(s.pixel(3, 1), None);
        assert_eq!(s.pixel(0, 2), None);
        s.set_pixel(9, 9, [9, 9, 9, 9]);
        assert_eq!(s.rgba, [1, 2, 3, 4].repeat(6), "out of bounds must no-op");
    }
}
