//! S22 task 3: render snapshot tests with a perceptual diff.
//!
//! # What is proven here, and what is not
//!
//! **Proven, and it needs no GPU:** the perceptual metric itself. `ciede2000` is checked
//! against the 34-pair supplementary test set from Sharma, Wu and Dalal (2005) — a reference
//! published outside this project specifically because the formula has three discontinuity
//! traps that a plausible-looking transcription gets wrong. Then the whole
//! `Snapshot` + `perceptual_diff` pipeline is demonstrated on synthetic pairs whose difference
//! is known in advance: an imperceptible dither must pass where a byte compare fails, a local
//! recolour must fail on colour, and a one-pixel shift must fail on *structure* even though
//! every pixel is individually within one just-noticeable difference.
//!
//! **Proven when a GPU is present:** that a real offscreen frame can be rendered, read back,
//! and compared. The reference for that comparison is computed analytically from the clear
//! colour rather than loaded from a committed PNG, because a committed frame is only a valid
//! reference on hardware that reproduces it and two GPUs do not agree bit for bit.
//!
//! **Not proven, and not provable here: parity with vanilla.** There is no vanilla screenshot
//! to diff against — no display, no running Minecraft client, no network to fetch one. S11 hit
//! the same wall and was explicit that its committed reference is its own output, so it proves
//! *no-regression*, not *parity*. Everything below builds the instrument; the measurement
//! needs a screenshot that does not exist in this environment.

use mc_parity::snapshot::{ciede2000, perceptual_diff, rgb_to_lab, ssim, DiffThresholds, Snapshot};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, RenderError};

/// The Sharma/Wu/Dalal supplementary test data: `(L1, a1, b1, L2, a2, b2, ΔE₀₀)`.
///
/// Pairs 4–6 and 21–24 are constructed to be exactly one unit apart, and pairs 9–16 straddle
/// the hue-quadrant and `Δh'` wrap discontinuities — those eight are the ones a naive
/// implementation fails.
#[rustfmt::skip]
const SHARMA: [[f64; 7]; 34] = [
    [50.0000,   2.6772, -79.7751, 50.0000,  0.0000, -82.7485,  2.0425],
    [50.0000,   3.1571, -77.2803, 50.0000,  0.0000, -82.7485,  2.8615],
    [50.0000,   2.8361, -74.0200, 50.0000,  0.0000, -82.7485,  3.4412],
    [50.0000,  -1.3802, -84.2814, 50.0000,  0.0000, -82.7485,  1.0000],
    [50.0000,  -1.1848, -84.8006, 50.0000,  0.0000, -82.7485,  1.0000],
    [50.0000,  -0.9009, -85.5211, 50.0000,  0.0000, -82.7485,  1.0000],
    [50.0000,   0.0000,   0.0000, 50.0000, -1.0000,   2.0000,  2.3669],
    [50.0000,  -1.0000,   2.0000, 50.0000,  0.0000,   0.0000,  2.3669],
    [50.0000,   2.4900,  -0.0010, 50.0000, -2.4900,   0.0009,  7.1792],
    [50.0000,   2.4900,  -0.0010, 50.0000, -2.4900,   0.0010,  7.1792],
    [50.0000,   2.4900,  -0.0010, 50.0000, -2.4900,   0.0011,  7.2195],
    [50.0000,   2.4900,  -0.0010, 50.0000, -2.4900,   0.0012,  7.2195],
    [50.0000,  -0.0010,   2.4900, 50.0000,  0.0009,  -2.4900,  4.8045],
    [50.0000,  -0.0010,   2.4900, 50.0000,  0.0010,  -2.4900,  4.8045],
    [50.0000,  -0.0010,   2.4900, 50.0000,  0.0011,  -2.4900,  4.7461],
    [50.0000,   2.5000,   0.0000, 50.0000,  0.0000,  -2.5000,  4.3065],
    [50.0000,   2.5000,   0.0000, 73.0000, 25.0000, -18.0000, 27.1492],
    [50.0000,   2.5000,   0.0000, 61.0000, -5.0000,  29.0000, 22.8977],
    [50.0000,   2.5000,   0.0000, 56.0000,-27.0000,  -3.0000, 31.9030],
    [50.0000,   2.5000,   0.0000, 58.0000, 24.0000,  15.0000, 19.4535],
    [50.0000,   2.5000,   0.0000, 50.0000,  3.1736,   0.5854,  1.0000],
    [50.0000,   2.5000,   0.0000, 50.0000,  3.2972,   0.0000,  1.0000],
    [50.0000,   2.5000,   0.0000, 50.0000,  1.8634,   0.5757,  1.0000],
    [50.0000,   2.5000,   0.0000, 50.0000,  3.2592,   0.3350,  1.0000],
    [60.2574, -34.0099,  36.2677, 60.4626,-34.1751,  39.4387,  1.2644],
    [63.0109, -31.0961,  -5.8663, 62.8187,-29.7946,  -4.0864,  1.2630],
    [61.2901,   3.7196,  -5.3901, 61.4292,  2.2480,  -4.9620,  1.8731],
    [35.0831, -44.1164,   3.7933, 35.0232,-40.0716,   1.5901,  1.8645],
    [22.7233,  20.0904, -46.6940, 23.0331, 14.9730, -42.5619,  2.0373],
    [36.4612,  47.8580,  18.3852, 36.2715, 50.5065,  21.2231,  1.4146],
    [90.8027,  -2.0831,   1.4410, 91.1528, -1.6435,   0.0447,  1.4441],
    [90.9257,  -0.5406,  -0.9208, 88.6381, -0.8985,  -0.7239,  1.5381],
    [ 6.7747,  -0.2908,  -2.4247,  5.8714, -0.0985,  -2.2286,  0.6377],
    [ 2.0776,   0.0795,  -1.1350,  0.9033, -0.0636,  -0.5514,  0.9082],
];

#[test]
fn ciede2000_matches_the_published_test_data() {
    for (i, row) in SHARMA.iter().enumerate() {
        let got = ciede2000([row[0], row[1], row[2]], [row[3], row[4], row[5]]);
        let want = row[6];
        assert!(
            (got - want).abs() < 1e-4,
            "pair {} ({:?} vs {:?}): got {got:.4}, published {want:.4}",
            i + 1,
            &row[0..3],
            &row[3..6]
        );
    }
}

#[test]
fn the_discontinuity_pairs_are_actually_exercised() {
    // Pairs 9-16 exist because a naive `Δh'` or hue-mean gets them wrong; if the table is ever
    // trimmed, this is what notices.
    assert_eq!(SHARMA.len(), 34);
    let unit_pairs = SHARMA.iter().filter(|r| (r[6] - 1.0).abs() < 1e-9).count();
    assert_eq!(
        unit_pairs, 7,
        "the published set has seven pairs constructed to be exactly one JND apart"
    );
}

/// A deterministic textured image: a diagonal gradient with a few solid patches, so SSIM has
/// real structure to measure rather than a flat field.
fn scene(width: u32, height: u32) -> Snapshot {
    let mut s = Snapshot::solid(width, height, [0, 0, 0, 255]);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width.max(1)) as u8;
            let g = ((y * 255) / height.max(1)) as u8;
            let b = (((x + y) * 255) / (width + height).max(1)) as u8;
            s.set_pixel(x, y, [r, g, b, 255]);
        }
    }
    for y in 8..24 {
        for x in 8..24 {
            s.set_pixel(x, y, [200, 40, 40, 255]);
        }
    }
    s
}

#[test]
fn an_imperceptible_dither_passes_perceptually_where_a_byte_compare_fails() {
    let a = scene(64, 64);
    let mut b = a.clone();
    for y in 0..b.height {
        for x in 0..b.width {
            let p = b.pixel(x, y).expect("in bounds");
            // +1 LSB on every channel: the largest change that is still nowhere near visible.
            b.set_pixel(
                x,
                y,
                [
                    p[0].saturating_add(1),
                    p[1].saturating_add(1),
                    p[2].saturating_add(1),
                    p[3],
                ],
            );
        }
    }

    assert_ne!(a.rgba, b.rgba, "a byte compare would fail here");

    let d = perceptual_diff(&a, &b).expect("diff");
    println!("dither: {}", d.summary());
    assert_eq!(d.differing_pixels, 64 * 64, "every pixel really did change");
    assert!(
        d.max_delta_e < 1.0,
        "a one-LSB change must stay under one JND, got {}",
        d.max_delta_e
    );
    assert!(
        d.is_within(&DiffThresholds::default()),
        "the default thresholds must let quantisation noise through: {}",
        d.summary()
    );
}

#[test]
fn a_local_recolour_fails_on_colour() {
    let a = scene(64, 64);
    let mut b = a.clone();
    for y in 30..40 {
        for x in 30..40 {
            b.set_pixel(x, y, [10, 200, 90, 255]);
        }
    }

    let d = perceptual_diff(&a, &b).expect("diff");
    println!("recolour: {}", d.summary());
    assert!(d.max_delta_e > 10.0, "{}", d.summary());
    assert_eq!(d.pixels_over_jnd, 100, "exactly the patch");
    assert!(!d.is_within(&DiffThresholds::default()));
    assert!(
        (30..40).contains(&d.worst_pixel.0) && (30..40).contains(&d.worst_pixel.1),
        "the report must point at the patch, got {:?}",
        d.worst_pixel
    );
}

#[test]
fn a_one_pixel_shift_fails_on_structure_even_though_every_pixel_is_within_one_jnd() {
    // This is the case that justifies carrying SSIM at all. A low-contrast checkerboard shifted
    // by one pixel changes every pixel by two 8-bit steps — individually invisible — while
    // destroying the local correlation completely.
    let (w, h) = (64u32, 64u32);
    let value = |x: u32, y: u32| -> u8 {
        if (x + y).is_multiple_of(2) {
            128
        } else {
            130
        }
    };
    let mut a = Snapshot::solid(w, h, [0, 0, 0, 255]);
    let mut b = Snapshot::solid(w, h, [0, 0, 0, 255]);
    for y in 0..h {
        for x in 0..w {
            let va = value(x, y);
            let vb = value(x + 1, y);
            a.set_pixel(x, y, [va, va, va, 255]);
            b.set_pixel(x, y, [vb, vb, vb, 255]);
        }
    }

    let d = perceptual_diff(&a, &b).expect("diff");
    println!("shift: {}", d.summary());
    assert!(
        d.max_delta_e < 1.0,
        "every individual pixel is inside one JND, got {}",
        d.max_delta_e
    );
    assert_eq!(
        d.pixels_over_jnd, 0,
        "so the colour metric alone would pass"
    );
    assert!(
        d.ssim < 0.99,
        "but the structure is gone and SSIM must say so, got {}",
        d.ssim
    );
    assert!(
        !d.is_within(&DiffThresholds::default()),
        "requiring both metrics is what catches this: {}",
        d.summary()
    );
}

#[test]
fn ssim_is_one_for_identical_images_and_falls_for_inverted_ones() {
    let a = scene(32, 32);
    assert!((ssim(&a, &a) - 1.0).abs() < 1e-9);

    let mut inverted = a.clone();
    for y in 0..a.height {
        for x in 0..a.width {
            let p = a.pixel(x, y).expect("in bounds");
            inverted.set_pixel(x, y, [255 - p[0], 255 - p[1], 255 - p[2], p[3]]);
        }
    }
    let s = ssim(&a, &inverted);
    assert!(s < 0.0, "an inverted image must anti-correlate, got {s}");
}

#[test]
fn a_snapshot_survives_a_png_round_trip() {
    let a = scene(40, 24);
    let path = std::env::temp_dir().join(format!(
        "mc-parity-{}-{}.png",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    a.save_png(&path).expect("save");
    let back = Snapshot::load_png(&path).expect("load");
    let _ = std::fs::remove_file(&path);
    assert_eq!(back, a, "PNG is lossless; the reference path must be too");
}

#[test]
fn lab_conversion_is_monotone_in_lightness() {
    let mut last = -1.0;
    for v in 0..=255u8 {
        let l = rgb_to_lab([v, v, v])[0];
        assert!(l > last, "L* must increase with grey level at {v}");
        last = l;
    }
    assert!((last - 100.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------------------------
// The GPU half. Skips itself when no adapter exists, which is normal on a CI runner.
// ---------------------------------------------------------------------------------------------

const W: u32 = 96;
const H: u32 = 64;

fn headless() -> Option<Gpu> {
    match Gpu::headless(
        W,
        H,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(g) => Some(g),
        Err(RenderError::NoAdapter) => {
            eprintln!("skip: no wgpu adapter on this machine");
            None
        }
        Err(e) => panic!("headless context: {e}"),
    }
}

/// Render one frame cleared to `(r, g, b)` in linear 0..1 and read it back.
fn render_clear(gpu: &mut Gpu, r: f64, g: f64, b: f64) -> Snapshot {
    let mut phases = PhaseList::new();
    // Field assignment rather than a struct literal: `clear_color` is a `wgpu::Color`, and
    // `mc-parity` deliberately does not depend on wgpu.
    phases.clear_color.r = r;
    phases.clear_color.g = g;
    phases.clear_color.b = b;
    phases.clear_color.a = 1.0;

    let frame = gpu.acquire().expect("acquire the offscreen frame");
    phases.render(gpu, frame.view(), &Camera::default(), 0.0);
    Snapshot::read_back(gpu).expect("readback")
}

#[test]
fn a_headless_frame_reads_back_matching_an_analytically_computed_reference() {
    let Some(mut gpu) = headless() else { return };

    // Rgba8Unorm stores the clear value verbatim, so the expected image is arithmetic, not a
    // committed PNG: 0.25 → 64, 0.5 → 128, 0.75 → 191.
    let actual = render_clear(&mut gpu, 0.25, 0.5, 0.75);
    let expected = Snapshot::solid(W, H, [64, 128, 191, 255]);

    // The budget is derived, not guessed: the perceptual cost of being one 8-bit step off on
    // every channel at exactly this colour. Two of those is the tolerance.
    let one_lsb = Snapshot::solid(W, H, [65, 129, 192, 255]);
    let budget = perceptual_diff(&expected, &one_lsb)
        .expect("budget")
        .max_delta_e
        * 2.0;

    let d = perceptual_diff(&expected, &actual).expect("diff");
    println!("gpu clear: {} (budget ΔE00 {budget:.4})", d.summary());
    assert!(
        d.max_delta_e <= budget,
        "the readback is further than two LSB from the requested clear colour: {}",
        d.summary()
    );
    assert_eq!(d.max_alpha_delta, 0, "the frame must be fully opaque");
}

#[test]
fn rendering_the_same_frame_twice_gives_the_same_pixels() {
    let Some(mut gpu) = headless() else { return };
    let a = render_clear(&mut gpu, 0.043, 0.070, 0.145);
    let b = render_clear(&mut gpu, 0.043, 0.070, 0.145);
    let d = perceptual_diff(&a, &b).expect("diff");
    assert!(
        d.is_identical(),
        "snapshot capture must be deterministic on one machine: {}",
        d.summary()
    );
}

#[test]
fn a_changed_frame_is_detected_by_the_perceptual_diff() {
    // The GPU counterpart of the synthetic tests: if the harness could not tell two genuinely
    // different frames apart, every green snapshot run above would be meaningless.
    let Some(mut gpu) = headless() else { return };
    let a = render_clear(&mut gpu, 0.043, 0.070, 0.145);
    let b = render_clear(&mut gpu, 0.043, 0.070, 0.400);
    let d = perceptual_diff(&a, &b).expect("diff");
    println!("gpu changed: {}", d.summary());
    assert!(!d.is_within(&DiffThresholds::default()), "{}", d.summary());
    assert!(d.max_delta_e > 1.0);
}

#[test]
fn a_captured_frame_survives_the_png_reference_path() {
    let Some(mut gpu) = headless() else { return };
    let captured = render_clear(&mut gpu, 0.2, 0.4, 0.6);
    let path = std::env::temp_dir().join(format!("mc-parity-gpu-{}.png", std::process::id()));
    captured.save_png(&path).expect("save");
    let back = Snapshot::load_png(&path).expect("load");
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        back, captured,
        "a reference written from a captured frame must reload identically"
    );
}
