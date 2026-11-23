//! The colour-grading LUT: every colour operation of a pack's grade
//! baked into one 33³ table, applied with one filtered lookup per pixel. Usable by every pack.
//!
//! # Use
//!
//! 1. Build a [`GradeLut`] the first time a pack needs it (it allocates a 287 KB texture and one
//!    small uniform; nothing exists before that).
//! 2. Give it the grade with [`GradeLut::set_params`] whenever the settings change. It compares,
//!    so calling it every frame with the same value costs a compare.
//! 3. Call [`GradeLut::record_bake`] in the frame encoder before the pass that reads the table.
//!    It records a tiny compute pass only when the parameters changed since the last bake.
//! 4. In the pass's bind group, bind [`GradeLut::view`] (a `texture_3d<f32>`, filterable) and
//!    [`GradeLut::sampler`] (linear, clamp); [`GradeLut::texture_entry`] and
//!    [`GradeLut::sampler_entry`] give the layout entries. Append [`LUT_APPLY_WGSL`] to the shader
//!    and call `grade_lut_apply(lut, lut_sampler, rgb)` on a display-encoded colour.
//!
//! The table's input and output are display-encoded RGB in `[0, 1]`: the frame's own byte domain
//! (the surface is `Bgra8Unorm` and vanilla blends in gamma space). A pack that renders HDR applies
//! it after its tone map, on display values, which is where a "look" LUT belongs.
//!
//! # The grade
//!
//! [`GradeParams`] is the whole grade: exposure, white balance (temperature, tint), contrast,
//! saturation, vibrance, hue shift, lift / gamma / gain as colour wheels, split toning, a
//! three-stop gradient map and fade. `grade.wgsl` documents the order. [`GradeParams::apply`] is the
//! CPU reference: it runs the same steps on the same derived uniform ([`GradeUniform`]) in `f32`,
//! and the GPU test `tw_sb_grade.rs` compares the baked table with it entry by entry.
//!
//! Every parameter is neutral at [`GradeParams::IDENTITY`], and the uniform's flags skip the steps
//! that would change nothing, so the identity bakes an exact identity table (also tested).
//!
//! # Cost
//!
//! A bake is 35,937 compute invocations, a few microseconds of GPU time, and happens only when a
//! setting changes. The lookup is one trilinear fetch from a 287 KB table that stays in cache
//! because neighbouring pixels have similar colours.

use glam::{DMat3, DVec2, DVec3, Mat3, Vec3};

use crate::Gpu;

/// Entries per axis. 33 is the industry's usual size (a 32-step grid plus the end point): a
/// trilinear lookup reproduces every smooth grade to well under one 8-bit level.
pub const LUT_SIZE: u32 = 33;

/// The table's format: 16-bit float, so steep curves do not band. Writable as a storage texture
/// on every WebGPU adapter and filterable.
pub const LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// How far the lift wheel moves black, in display units, at full strength.
pub const LIFT_RANGE: f32 = 0.25;
/// How far the gamma wheel moves a channel's exponent, at full strength (1 ± 0.5).
pub const GAMMA_RANGE: f32 = 0.5;
/// How far the gain wheel scales a channel, at full strength (1 ± 0.5).
pub const GAIN_RANGE: f32 = 0.5;
/// The black level at full fade, in display units.
pub const FADE_MAX: f32 = 0.25;

/// Rec. 709 luma weights.
pub const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// The WGSL helper a pack appends to its own shader to read the table:
///
/// ```wgsl
/// let graded = grade_lut_apply(my_lut, my_lut_sampler, colour.rgb);
/// ```
///
/// `lut` must be [`GradeLut::view`] and `lut_sampler` [`GradeLut::sampler`] (any linear, clamping
/// sampler works). The input is clamped to `[0, 1]`; the half-texel remap puts 0 and 1 on the
/// first and last entries' centres, where the bake wrote them.
pub const LUT_APPLY_WGSL: &str = r"
fn grade_lut_apply(lut: texture_3d<f32>, lut_sampler: sampler, c: vec3<f32>) -> vec3<f32> {
    let size = f32(textureDimensions(lut).x);
    let uvw = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)) * ((size - 1.0) / size) + vec3<f32>(0.5 / size);
    return textureSampleLevel(lut, lut_sampler, uvw, 0.0).rgb;
}
";

/// One colour wheel: a hue and how far to push toward it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wheel {
    /// Degrees, 0 = red, 120 = green, 240 = blue.
    pub hue: f32,
    /// 0 = neutral, 1 = the wheel's full range ([`LIFT_RANGE`], [`GAMMA_RANGE`], [`GAIN_RANGE`]).
    pub strength: f32,
}

impl Wheel {
    /// The wheel that changes nothing.
    pub const NEUTRAL: Wheel = Wheel {
        hue: 0.0,
        strength: 0.0,
    };

    /// The push as an RGB direction with zero luma, scaled by the strength: what the wheel adds
    /// to (lift) or multiplies into (gamma, gain) each channel at full range.
    #[must_use]
    pub fn vector(self) -> Vec3 {
        if self.strength == 0.0 {
            return Vec3::ZERO;
        }
        let rgb = hue_rgb(self.hue);
        let luma = rgb.dot(Vec3::from(LUMA));
        (rgb - Vec3::splat(luma)) * self.strength
    }
}

/// The fully saturated colour at `hue` degrees (HSV with S = V = 1).
#[must_use]
pub fn hue_rgb(hue: f32) -> Vec3 {
    let h = (hue / 60.0).rem_euclid(6.0);
    Vec3::new(
        ((h - 3.0).abs() - 1.0).clamp(0.0, 1.0),
        (2.0 - (h - 2.0).abs()).clamp(0.0, 1.0),
        (2.0 - (h - 4.0).abs()).clamp(0.0, 1.0),
    )
}

/// A whole grade. Every field's neutral value is in [`GradeParams::IDENTITY`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradeParams {
    /// Stops of exposure (2^EV), 0 = unchanged.
    pub exposure: f32,
    /// White balance, -1 (cool) to 1 (warm), 0 = unchanged.
    pub temperature: f32,
    /// White balance across the locus, -1 (green) to 1 (magenta), 0 = unchanged.
    pub tint: f32,
    /// A power around linear mid grey 0.18, 1 = unchanged. Black stays black.
    pub contrast: f32,
    /// OKLab chroma scale, 1 = unchanged, 0 = grey.
    pub saturation: f32,
    /// Extra chroma for muted colours (-1 to 1), 0 = unchanged. Saturated colours move least.
    pub vibrance: f32,
    /// Rotation of the OKLab hue, in degrees, 0 = unchanged.
    pub hue_shift: f32,
    /// Shadows wheel: an offset at black that fades out toward white.
    pub lift: Wheel,
    /// Midtones wheel: a per-channel gamma.
    pub gamma: Wheel,
    /// Highlights wheel: a per-channel gain.
    pub gain: Wheel,
    /// Split toning: the colour shadows lean toward (display RGB, 0..1).
    pub split_shadow: [f32; 3],
    /// Split toning: the colour highlights lean toward (display RGB, 0..1).
    pub split_highlight: [f32; 3],
    /// Split toning: -1 favours the shadow colour, 1 the highlight colour.
    pub split_balance: f32,
    /// Split toning strength, 0 = off, 1 = full.
    pub split_strength: f32,
    /// Gradient map stops over luma: dark, mid, light (display RGB, 0..1).
    pub gradient: [[f32; 3]; 3],
    /// How much of the gradient map replaces the colour, 0 = off.
    pub gradient_mix: f32,
    /// Lifted blacks, 0 = off, 1 = black at [`FADE_MAX`].
    pub fade: f32,
}

impl Default for GradeParams {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl GradeParams {
    /// The grade that changes nothing. It bakes an exact identity table.
    pub const IDENTITY: GradeParams = GradeParams {
        exposure: 0.0,
        temperature: 0.0,
        tint: 0.0,
        contrast: 1.0,
        saturation: 1.0,
        vibrance: 0.0,
        hue_shift: 0.0,
        lift: Wheel::NEUTRAL,
        gamma: Wheel::NEUTRAL,
        gain: Wheel::NEUTRAL,
        split_shadow: [0.5; 3],
        split_highlight: [0.5; 3],
        split_balance: 0.0,
        split_strength: 0.0,
        gradient: [[0.0; 3], [0.5; 3], [1.0; 3]],
        gradient_mix: 0.0,
        fade: 0.0,
    };

    /// Whether this grade changes nothing (every field neutral; stops and colours that are
    /// switched off do not count).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.uniform().is_identity()
    }

    /// The grade made safe to bake: non-finite numbers become neutral, every value clamped to the
    /// range its control offers (and a little beyond where the maths allows).
    #[must_use]
    pub fn sanitized(self) -> GradeParams {
        let f = |value: f32, neutral: f32, lo: f32, hi: f32| {
            if value.is_finite() {
                value.clamp(lo, hi)
            } else {
                neutral
            }
        };
        let colour = |c: [f32; 3], neutral: f32| c.map(|v| f(v, neutral, 0.0, 1.0));
        let wheel = |w: Wheel| Wheel {
            hue: f(w.hue, 0.0, -360.0, 720.0),
            strength: f(w.strength, 0.0, 0.0, 1.0),
        };
        GradeParams {
            exposure: f(self.exposure, 0.0, -4.0, 4.0),
            temperature: f(self.temperature, 0.0, -1.0, 1.0),
            tint: f(self.tint, 0.0, -1.0, 1.0),
            contrast: f(self.contrast, 1.0, 0.25, 2.0),
            saturation: f(self.saturation, 1.0, 0.0, 2.0),
            vibrance: f(self.vibrance, 0.0, -1.0, 1.0),
            hue_shift: f(self.hue_shift, 0.0, -180.0, 180.0),
            lift: wheel(self.lift),
            gamma: wheel(self.gamma),
            gain: wheel(self.gain),
            split_shadow: colour(self.split_shadow, 0.5),
            split_highlight: colour(self.split_highlight, 0.5),
            split_balance: f(self.split_balance, 0.0, -1.0, 1.0),
            split_strength: f(self.split_strength, 0.0, 0.0, 1.0),
            gradient: [
                colour(self.gradient[0], 0.0),
                colour(self.gradient[1], 0.5),
                colour(self.gradient[2], 1.0),
            ],
            gradient_mix: f(self.gradient_mix, 0.0, 0.0, 1.0),
            fade: f(self.fade, 0.0, 0.0, 1.0),
        }
    }

    /// The uniform the bake reads, with every derived number worked out here once.
    #[must_use]
    pub fn uniform(&self) -> GradeUniform {
        GradeUniform::from_params(&self.sanitized())
    }

    /// The CPU reference of the bake: one display-encoded colour through the whole grade, the same
    /// steps as `grade.wgsl` on the same [`GradeUniform`], in `f32`.
    #[must_use]
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        self.uniform().apply(rgb)
    }
}

// ------------------------------------------------------------------------------ white balance
//
// Temperature and tint pick the colour that white light takes in the picture. Temperature slides
// that white along the Planckian locus (the colours of glowing black bodies) in steps of mired,
// one million over the colour temperature in kelvin; tint pushes it off the locus by a Duv, the
// signed distance across it in CIE 1960 UCS. Every colour is then carried from the display's own
// white (D65) to the picked white by the Bradford chromatic-adaptation transform, the way the eye
// re-balances under a new light. All of it is worked out here, in f64, only when a setting
// changes; what reaches the bake is one 3x3 matrix on linear sRGB (with the exposure folded in).

/// sRGB's white point, D65, in CIE 1931 xy (IEC 61966-2-1).
const D65_XY: DVec2 = DVec2::new(0.3127, 0.3290);

/// The sRGB (ITU-R BT.709) primaries red, green and blue, in CIE 1931 xy.
const SRGB_PRIMARIES_XY: [DVec2; 3] = [
    DVec2::new(0.64, 0.33),
    DVec2::new(0.30, 0.60),
    DVec2::new(0.15, 0.06),
];

/// D65's correlated colour temperature: where the temperature control starts.
const D65_KELVIN: f64 = 6504.0;

/// Mired added to white at full warmth (+1): a 3,650 K black body's colour.
const WARM_MIRED: f64 = 120.0;

/// How fast cooling heads for the cold end of the locus (0 mired, an infinitely hot black body):
/// white keeps `1 - tanh(COOL_RATE * cooling)` of D65's mired value. That is about 2.5 mired per
/// percent near neutral, and it never runs off the end of the locus.
const COOL_RATE: f64 = 1.65;

/// The Duv at full tint: green (-1) lifts white this far above the locus, magenta (+1) drops it
/// this far below.
const TINT_DUV: f64 = 0.04;

/// The Bradford cone-response matrix (rows: XYZ to the three sharpened cone signals), as
/// published for chromatic adaptation (ICC.1, Annex E).
const BRADFORD_ROWS: [[f64; 3]; 3] = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

/// The XYZ of the colour with chromaticity `xy`, at luminance Y = 1.
fn xy_to_xyz(xy: DVec2) -> DVec3 {
    DVec3::new(xy.x / xy.y, 1.0, (1.0 - xy.x - xy.y) / xy.y)
}

/// CIE 1931 xy to CIE 1960 UCS uv.
fn xy_to_uv(xy: DVec2) -> DVec2 {
    let d = -2.0 * xy.x + 12.0 * xy.y + 3.0;
    DVec2::new(4.0 * xy.x / d, 6.0 * xy.y / d)
}

/// CIE 1960 UCS uv to CIE 1931 xy.
fn uv_to_xy(uv: DVec2) -> DVec2 {
    let d = 2.0 * uv.x - 8.0 * uv.y + 4.0;
    DVec2::new(3.0 * uv.x / d, 2.0 * uv.y / d)
}

/// Linear sRGB to XYZ, built from the primaries and the white point: each primary's XYZ at unit
/// luminance, scaled so that equal amounts of the three make the white at Y = 1.
fn srgb_to_xyz() -> DMat3 {
    let [r, g, b] = SRGB_PRIMARIES_XY.map(xy_to_xyz);
    let primaries = DMat3::from_cols(r, g, b);
    let amounts = primaries.inverse() * xy_to_xyz(D65_XY);
    primaries * DMat3::from_diagonal(amounts)
}

/// A black body's colour at `kelvin`, in CIE 1960 UCS: the published rational approximation of
/// the Planckian locus (1985). Within about 1e-4 of the exact locus from 1,000 K to 15,000 K,
/// and smooth (no poles) at every positive temperature beyond.
fn planckian_uv(kelvin: f64) -> DVec2 {
    let t = kelvin;
    let u = (0.860_117_757 + 1.541_182_54e-4 * t + 1.286_412_12e-7 * t * t)
        / (1.0 + 8.424_202_35e-4 * t + 7.081_451_63e-7 * t * t);
    let v = (0.317_398_726 + 4.228_062_45e-5 * t + 4.204_816_91e-8 * t * t)
        / (1.0 - 2.897_418_16e-5 * t + 1.614_560_53e-7 * t * t);
    DVec2::new(u, v)
}

/// The locus point at `mired`.
fn locus_at_mired(mired: f64) -> DVec2 {
    planckian_uv(1.0e6 / mired.max(1.0e-3))
}

/// The mired value white takes for a temperature control in -1 (cool) ..= 1 (warm).
fn white_mired(temperature: f64) -> f64 {
    let start = 1.0e6 / D65_KELVIN;
    if temperature >= 0.0 {
        start + WARM_MIRED * temperature
    } else {
        start * (1.0 - (COOL_RATE * -temperature).tanh())
    }
}

/// The white the controls pick, in CIE 1960 UCS: D65 carried along the locus by the
/// temperature's mired shift, then across it by the tint's Duv (positive tint is magenta).
fn picked_white_uv(temperature: f64, tint: f64) -> DVec2 {
    let start = 1.0e6 / D65_KELVIN;
    let mired = white_mired(temperature);
    let along = locus_at_mired(mired) - locus_at_mired(start);
    // The locus' direction toward warmer light, turned a quarter to its green side: the side
    // D65 itself sits on, so zero tint keeps D65's own offset.
    let slope = locus_at_mired(mired + 0.5) - locus_at_mired(mired - 0.5);
    let greener = DVec2::new(-slope.y, slope.x).normalize();
    xy_to_uv(D65_XY) + along - greener * (TINT_DUV * tint)
}

/// The white balance for a temperature and tint in -1..1, as one matrix on linear sRGB: the
/// Bradford adaptation from D65 to the picked white. The display's white lands exactly on that
/// white, at the same luminance. Neutral settings return the exact identity.
#[must_use]
pub fn white_balance_matrix(temperature: f32, tint: f32) -> Mat3 {
    if temperature == 0.0 && tint == 0.0 {
        return Mat3::IDENTITY;
    }
    let picked = xy_to_xyz(uv_to_xy(picked_white_uv(
        f64::from(temperature),
        f64::from(tint),
    )));
    let cones = DMat3::from_cols_array_2d(&BRADFORD_ROWS).transpose();
    let scale = DMat3::from_diagonal((cones * picked) / (cones * xy_to_xyz(D65_XY)));
    let to_xyz = srgb_to_xyz();
    (to_xyz.inverse() * cones.inverse() * scale * cones * to_xyz).as_mat3()
}

/// The bake's uniform, `GradeUniform` in `grade.wgsl` (13 `vec4`s, 208 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GradeUniform {
    /// Columns (xyz) of the linear-light colour matrix: the white balance times the exposure
    /// multiplier. The first column's w is 1 when the matrix is not the identity.
    pub colour: [[f32; 4]; 3],
    /// x = contrast, y = saturation, z = vibrance, w = hue shift in radians.
    pub tone: [f32; 4],
    /// x = OKLab step on, y = shoulder maximum, z = fade level, w = gradient mix.
    pub flags: [f32; 4],
    /// rgb = lift offset, w unused.
    pub lift: [f32; 4],
    /// rgb = per-channel gamma, w = display-space steps on.
    pub gamma: [f32; 4],
    /// rgb = per-channel gain.
    pub gain: [f32; 4],
    /// rgb = shadow colour, a = split strength.
    pub split_shadow: [f32; 4],
    /// rgb = highlight colour, a = split balance.
    pub split_highlight: [f32; 4],
    /// Gradient stops.
    pub grad_dark: [f32; 4],
    /// Gradient stops.
    pub grad_mid: [f32; 4],
    /// Gradient stops.
    pub grad_light: [f32; 4],
}

fn v4(v: Vec3, w: f32) -> [f32; 4] {
    [v.x, v.y, v.z, w]
}

fn c4(c: [f32; 3], w: f32) -> [f32; 4] {
    [c[0], c[1], c[2], w]
}

fn columns(m: Mat3) -> [[f32; 4]; 3] {
    [v4(m.x_axis, 0.0), v4(m.y_axis, 0.0), v4(m.z_axis, 0.0)]
}

impl GradeUniform {
    /// Derive the uniform from (already sanitized) parameters.
    #[must_use]
    pub fn from_params(p: &GradeParams) -> GradeUniform {
        let colour = white_balance_matrix(p.temperature, p.tint) * p.exposure.exp2();
        let mut colour_columns = columns(colour);
        colour_columns[0][3] = f32::from(u8::from(colour != Mat3::IDENTITY));
        let oklab_on = p.hue_shift != 0.0 || p.saturation != 1.0 || p.vibrance != 0.0;
        let lift = p.lift.vector() * LIFT_RANGE;
        let gamma = Vec3::ONE + p.gamma.vector() * GAMMA_RANGE;
        let gain = Vec3::ONE + p.gain.vector() * GAIN_RANGE;
        let split = if p.split_strength > 0.0 {
            p.split_strength
        } else {
            0.0
        };
        let display_on = lift != Vec3::ZERO
            || gamma != Vec3::ONE
            || gain != Vec3::ONE
            || split > 0.0
            || p.gradient_mix > 0.0
            || p.fade > 0.0;
        let mut uniform = GradeUniform {
            colour: colour_columns,
            tone: [
                p.contrast,
                p.saturation,
                p.vibrance,
                p.hue_shift.to_radians(),
            ],
            flags: [
                f32::from(u8::from(oklab_on)),
                1.0,
                p.fade * FADE_MAX,
                p.gradient_mix,
            ],
            lift: v4(lift, 0.0),
            gamma: v4(gamma, f32::from(u8::from(display_on))),
            gain: v4(gain, 0.0),
            split_shadow: c4(p.split_shadow, split),
            split_highlight: c4(p.split_highlight, p.split_balance),
            grad_dark: c4(p.gradient[0], 0.0),
            grad_mid: c4(p.gradient[1], 0.0),
            grad_light: c4(p.gradient[2], 0.0),
        };
        // The shoulder lands the brightest channel of white on 1, and only when exposure, white
        // balance or contrast can push white above 1; otherwise it is the identity.
        let white = uniform.grade_linear(Vec3::ONE);
        uniform.flags[1] = white.max_element().max(1.0);
        uniform
    }

    /// Whether the bake would be the identity: no step switched on (the colour matrix is the
    /// identity) and unit contrast.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.colour[0][3] == 0.0
            && self.flags[0] == 0.0
            && self.gamma[3] == 0.0
            && self.tone[0] == 1.0
    }

    /// `grade_linear` of `grade.wgsl`.
    fn grade_linear(&self, lin: Vec3) -> Vec3 {
        let mut lin = lin;
        if self.colour[0][3] > 0.5 {
            lin = mat(self.colour) * lin;
        }
        lin = lin.max(Vec3::ZERO);
        let contrast = self.tone[0];
        if contrast != 1.0 {
            lin = (lin / 0.18).powf(contrast) * 0.18;
        }
        lin
    }

    /// `grade` of `grade.wgsl`: the CPU reference for one display-encoded colour.
    #[must_use]
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let c = Vec3::from(rgb).clamp(Vec3::ZERO, Vec3::ONE);
        let mut lin = self.grade_linear(srgb_to_linear(c));
        if self.flags[0] > 0.5 {
            let lab = linear_to_oklab(lin);
            let (sn, cs) = self.tone[3].sin_cos();
            let mut a = lab.y * cs - lab.z * sn;
            let mut b = lab.y * sn + lab.z * cs;
            let chroma = (a * a + b * b).sqrt();
            let vib = 1.0 + self.tone[2] * (1.0 - (chroma / 0.3).clamp(0.0, 1.0));
            let scale = (self.tone[1] * vib).max(0.0);
            a *= scale;
            b *= scale;
            lin = oklab_to_linear(Vec3::new(lab.x, a, b)).max(Vec3::ZERO);
        }
        let m = self.flags[1];
        lin = Vec3::new(shoulder(lin.x, m), shoulder(lin.y, m), shoulder(lin.z, m));
        let mut g = linear_to_srgb(lin).clamp(Vec3::ZERO, Vec3::ONE);
        if self.gamma[3] > 0.5 {
            let gain = Vec3::new(self.gain[0], self.gain[1], self.gain[2]);
            let lift = Vec3::new(self.lift[0], self.lift[1], self.lift[2]);
            let gamma = Vec3::new(self.gamma[0], self.gamma[1], self.gamma[2]);
            g = g * gain + lift * (Vec3::ONE - g);
            let g_clamped = g.clamp(Vec3::ZERO, Vec3::ONE);
            g = Vec3::new(
                g_clamped.x.powf(1.0 / gamma.x),
                g_clamped.y.powf(1.0 / gamma.y),
                g_clamped.z.powf(1.0 / gamma.z),
            );
            let strength = self.split_shadow[3];
            if strength > 0.0 {
                let t = (g.dot(Vec3::from(LUMA)) + self.split_highlight[3] * 0.5).clamp(0.0, 1.0);
                let shadow = Vec3::new(
                    self.split_shadow[0],
                    self.split_shadow[1],
                    self.split_shadow[2],
                );
                let highlight = Vec3::new(
                    self.split_highlight[0],
                    self.split_highlight[1],
                    self.split_highlight[2],
                );
                let shadow_blend = Vec3::splat(0.5).lerp(shadow, strength * (1.0 - t));
                let highlight_blend = Vec3::splat(0.5).lerp(highlight, strength * t);
                g = soft_light(g, shadow_blend);
                g = soft_light(g, highlight_blend);
            }
            let mix = self.flags[3];
            if mix > 0.0 {
                let y = g.dot(Vec3::from(LUMA)).clamp(0.0, 1.0);
                let dark = Vec3::new(self.grad_dark[0], self.grad_dark[1], self.grad_dark[2]);
                let mid = Vec3::new(self.grad_mid[0], self.grad_mid[1], self.grad_mid[2]);
                let light = Vec3::new(self.grad_light[0], self.grad_light[1], self.grad_light[2]);
                let mapped = if y < 0.5 {
                    dark.lerp(mid, (y * 2.0).clamp(0.0, 1.0))
                } else {
                    mid.lerp(light, (y * 2.0 - 1.0).clamp(0.0, 1.0))
                };
                g = g.lerp(mapped, mix);
            }
            let fade = self.flags[2];
            g = Vec3::splat(fade) + g * (1.0 - fade);
        }
        g.clamp(Vec3::ZERO, Vec3::ONE).to_array()
    }
}

fn mat(cols: [[f32; 4]; 3]) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(cols[0][0], cols[0][1], cols[0][2]),
        Vec3::new(cols[1][0], cols[1][1], cols[1][2]),
        Vec3::new(cols[2][0], cols[2][1], cols[2][2]),
    )
}

/// The sRGB transfer function, display to linear.
#[must_use]
pub fn srgb_to_linear(c: Vec3) -> Vec3 {
    let f = |x: f32| {
        if x <= 0.040_45 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    };
    Vec3::new(f(c.x), f(c.y), f(c.z))
}

/// The sRGB transfer function, linear to display.
#[must_use]
pub fn linear_to_srgb(c: Vec3) -> Vec3 {
    let f = |x: f32| {
        let x = x.max(0.0);
        if x <= 0.003_130_8 {
            x * 12.92
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        }
    };
    Vec3::new(f(c.x), f(c.y), f(c.z))
}

/// Linear sRGB to OKLab.
#[must_use]
pub fn linear_to_oklab(c: Vec3) -> Vec3 {
    let l = 0.412_221_46 * c.x + 0.536_332_55 * c.y + 0.051_445_995 * c.z;
    let m = 0.211_903_5 * c.x + 0.680_699_5 * c.y + 0.107_396_96 * c.z;
    let s = 0.088_302_46 * c.x + 0.281_718_85 * c.y + 0.629_978_7 * c.z;
    let (l, m, s) = (l.cbrt(), m.cbrt(), s.cbrt());
    Vec3::new(
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    )
}

/// OKLab to linear sRGB.
#[must_use]
pub fn oklab_to_linear(lab: Vec3) -> Vec3 {
    let l = lab.x + 0.396_337_78 * lab.y + 0.215_803_76 * lab.z;
    let m = lab.x - 0.105_561_346 * lab.y - 0.063_854_17 * lab.z;
    let s = lab.x - 0.089_484_18 * lab.y - 1.291_485_5 * lab.z;
    let (l, m, s) = (l * l * l, m * m * m, s * s * s);
    Vec3::new(
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    )
}

/// `shoulder` of `grade.wgsl`.
fn shoulder(x: f32, m: f32) -> f32 {
    let knee = 0.75;
    if m <= 1.0 || x <= knee {
        return x;
    }
    let n = (m - knee) / (1.0 - knee);
    let t = ((x - knee) / (m - knee)).clamp(0.0, 1.0);
    knee + (1.0 - knee) * (1.0 - (1.0 - t).powf(n))
}

/// `soft_light` of `grade.wgsl` (W3C), per channel.
fn soft_light(a: Vec3, b: Vec3) -> Vec3 {
    let one = |a: f32, b: f32| {
        if b <= 0.5 {
            a - (1.0 - 2.0 * b) * a * (1.0 - a)
        } else {
            let d = if a <= 0.25 {
                ((16.0 * a - 12.0) * a + 4.0) * a
            } else {
                a.sqrt()
            };
            a + (2.0 * b - 1.0) * (d - a)
        }
    };
    Vec3::new(one(a.x, b.x), one(a.y, b.y), one(a.z, b.z))
}

/// The baked table and what bakes it. See the [module docs](self).
pub struct GradeLut {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    /// The parameters the next frame should show.
    wanted: GradeParams,
    /// The parameters the table holds, `None` before the first bake.
    baked: Option<GradeParams>,
    bakes: u64,
}

impl std::fmt::Debug for GradeLut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradeLut")
            .field("wanted", &self.wanted)
            .field("baked", &self.baked.is_some())
            .field("bakes", &self.bakes)
            .finish_non_exhaustive()
    }
}

impl GradeLut {
    /// The bake's WGSL source.
    pub const BAKE_WGSL: &'static str = include_str!("grade.wgsl");

    /// Build the table (unbaked; the first [`GradeLut::record_bake`] fills it), its sampler and
    /// the bake pipeline. Call it when a pack first needs a grade, not before.
    #[must_use]
    pub fn new(gpu: &Gpu) -> GradeLut {
        let device = &gpu.device;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shaderpack/grade lut"),
            size: wgpu::Extent3d {
                width: LUT_SIZE,
                height: LUT_SIZE,
                depth_or_array_layers: LUT_SIZE,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: LUT_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shaderpack/grade lut"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaderpack/grade bake"),
            source: wgpu::ShaderSource::Wgsl(Self::BAKE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shaderpack/grade bake"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: LUT_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shaderpack/grade bake"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shaderpack/grade bake"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_bake"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shaderpack/grade bake"),
            size: std::mem::size_of::<GradeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shaderpack/grade bake"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });
        GradeLut {
            texture,
            view,
            sampler,
            pipeline,
            group,
            uniform,
            wanted: GradeParams::IDENTITY,
            baked: None,
            bakes: 0,
        }
    }

    /// Set the grade the table should hold. Sanitized and compared: an unchanged grade costs a
    /// compare and bakes nothing.
    pub fn set_params(&mut self, params: GradeParams) {
        self.wanted = params.sanitized();
    }

    /// The grade the table will hold after the next [`GradeLut::record_bake`].
    #[must_use]
    pub fn params(&self) -> GradeParams {
        self.wanted
    }

    /// Whether the next [`GradeLut::record_bake`] has work to do.
    #[must_use]
    pub fn needs_bake(&self) -> bool {
        self.baked != Some(self.wanted)
    }

    /// Record the bake into `enc` if the parameters changed since the last one. Returns whether a
    /// bake was recorded. Passes recorded after it in the same encoder see the new table.
    pub fn record_bake(&mut self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder) -> bool {
        if !self.needs_bake() {
            return false;
        }
        let uniform = self.wanted.uniform();
        gpu.write_buffer_staged(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shaderpack/grade bake"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        let groups = LUT_SIZE.div_ceil(4);
        pass.dispatch_workgroups(groups, groups, groups);
        drop(pass);
        self.baked = Some(self.wanted);
        self.bakes += 1;
        true
    }

    /// The table, for a `texture_3d<f32>` binding (filterable float).
    #[must_use]
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// The table's texture (tests read it back).
    #[must_use]
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// A linear, clamping sampler for the table.
    #[must_use]
    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// How many bakes have been recorded.
    #[must_use]
    pub fn bakes(&self) -> u64 {
        self.bakes
    }

    /// The layout entry for [`GradeLut::view`] at `binding`, visible to fragment shaders.
    #[must_use]
    pub fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        }
    }

    /// The layout entry for [`GradeLut::sampler`] at `binding`, visible to fragment shaders.
    #[must_use]
    pub fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(source: &str) {
        let module = wgpu::naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("validates");
    }

    /// `cargo check` never parses WGSL; this does, with the validator wgpu runs at pipeline
    /// creation.
    #[test]
    fn the_bake_and_the_apply_helper_parse_and_validate() {
        validate(GradeLut::BAKE_WGSL);
        let apply = format!(
            "@group(0) @binding(0) var t: texture_3d<f32>;\n\
             @group(0) @binding(1) var s: sampler;\n{LUT_APPLY_WGSL}\n\
             @fragment fn fs() -> @location(0) vec4<f32> {{\n\
                 return vec4<f32>(grade_lut_apply(t, s, vec3<f32>(0.5)), 1.0);\n}}"
        );
        validate(&apply);
    }

    #[test]
    fn the_uniform_matches_the_wgsl_struct_size() {
        assert_eq!(std::mem::size_of::<GradeUniform>(), 13 * 16);
    }

    fn grid() -> impl Iterator<Item = [f32; 3]> {
        (0..=16).flat_map(|r| {
            (0..=16).flat_map(move |g| (0..=16).map(move |b| [r, g, b].map(|v| v as f32 / 16.0)))
        })
    }

    fn max_error(a: [f32; 3], b: [f32; 3]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// The neutral grade switches every step off and is the identity to f32 rounding.
    #[test]
    fn the_identity_switches_every_step_off_and_changes_nothing() {
        let uniform = GradeParams::IDENTITY.uniform();
        assert!(uniform.is_identity(), "{uniform:?}");
        assert_eq!(uniform.flags[1], 1.0, "no shoulder at the identity");
        for c in grid() {
            let out = GradeParams::IDENTITY.apply(c);
            assert!(max_error(out, c) < 1e-5, "{c:?} -> {out:?}");
        }
    }

    /// Each control on its own moves the grade off the identity, and back at neutral it is the
    /// identity again (every step's neutral value really is neutral).
    #[test]
    fn every_control_is_live_and_neutral_at_its_neutral_value() {
        let probe = [0.8, 0.45, 0.2];
        let cases: Vec<(&str, GradeParams)> = vec![
            (
                "exposure",
                GradeParams {
                    exposure: 0.5,
                    ..GradeParams::IDENTITY
                },
            ),
            (
                "temperature",
                GradeParams {
                    temperature: 0.4,
                    ..GradeParams::IDENTITY
                },
            ),
            (
                "tint",
                GradeParams {
                    tint: -0.4,
                    ..GradeParams::IDENTITY
                },
            ),
            (
                "contrast",
                GradeParams {
                    contrast: 1.3,
                    ..GradeParams::IDENTITY
                },
            ),
            (
                "saturation",
                GradeParams {
                    saturation: 0.5,
                    ..GradeParams::IDENTITY
                },
            ),
            (
                "vibrance",
                GradeParams {
                    vibrance: 0.6,
                    ..GradeParams::IDENTITY
                },
            ),
            (
                "hue",
                GradeParams {
                    hue_shift: 40.0,
                    ..GradeParams::IDENTITY
                },
            ),
            (
                "lift",
                GradeParams {
                    lift: Wheel {
                        hue: 220.0,
                        strength: 0.5,
                    },
                    ..GradeParams::IDENTITY
                },
            ),
            (
                "gamma",
                GradeParams {
                    gamma: Wheel {
                        hue: 30.0,
                        strength: 0.5,
                    },
                    ..GradeParams::IDENTITY
                },
            ),
            (
                "gain",
                GradeParams {
                    gain: Wheel {
                        hue: 300.0,
                        strength: 0.5,
                    },
                    ..GradeParams::IDENTITY
                },
            ),
            (
                "split",
                GradeParams {
                    split_shadow: [0.2, 0.3, 0.8],
                    split_highlight: [1.0, 0.7, 0.4],
                    split_strength: 0.6,
                    ..GradeParams::IDENTITY
                },
            ),
            (
                "gradient",
                GradeParams {
                    gradient: [[0.1, 0.0, 0.3], [0.8, 0.3, 0.5], [1.0, 0.9, 0.7]],
                    gradient_mix: 0.5,
                    ..GradeParams::IDENTITY
                },
            ),
            (
                "fade",
                GradeParams {
                    fade: 0.5,
                    ..GradeParams::IDENTITY
                },
            ),
        ];
        for (name, params) in cases {
            assert!(!params.is_identity(), "{name} reads as the identity");
            let out = params.apply(probe);
            assert!(max_error(out, probe) > 0.01, "{name} did nothing: {out:?}");
        }
        // Neutral colours: split toning at 0 strength and a gradient at 0 mix are the identity
        // whatever their colours.
        let parked = GradeParams {
            split_shadow: [0.0, 0.0, 1.0],
            split_highlight: [1.0, 0.0, 0.0],
            gradient: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            lift: Wheel {
                hue: 123.0,
                strength: 0.0,
            },
            ..GradeParams::IDENTITY
        };
        assert!(parked.is_identity());
        assert!(max_error(parked.apply(probe), probe) < 1e-5);
    }

    /// Warm white balance makes white warmer (more red than blue), cool the reverse, and white
    /// keeps roughly its brightness.
    #[test]
    fn temperature_warms_and_cools_white() {
        let warm = GradeParams {
            temperature: 0.5,
            ..GradeParams::IDENTITY
        }
        .apply([0.7; 3]);
        let cool = GradeParams {
            temperature: -0.5,
            ..GradeParams::IDENTITY
        }
        .apply([0.7; 3]);
        assert!(warm[0] > warm[2] + 0.02, "{warm:?}");
        assert!(cool[2] > cool[0] + 0.02, "{cool:?}");
    }

    /// Neutral settings are the exact identity (the bake skips the matrix), and the matrix
    /// leaves the identity continuously: the smallest step changes almost nothing.
    #[test]
    fn neutral_white_balance_is_the_exact_identity() {
        assert_eq!(white_balance_matrix(0.0, 0.0), Mat3::IDENTITY);
        assert_eq!(white_balance_matrix(-0.0, 0.0), Mat3::IDENTITY);
        assert_eq!(GradeParams::IDENTITY.uniform().colour[0][3], 0.0);
        for (temperature, tint) in [(1.0e-4, 0.0), (-1.0e-4, 0.0), (0.0, 1.0e-4), (0.0, -1.0e-4)] {
            let m = white_balance_matrix(temperature, tint);
            assert_ne!(m, Mat3::IDENTITY, "({temperature}, {tint}) changes nothing");
            let off = (m - Mat3::IDENTITY).to_cols_array().map(f32::abs);
            let worst = off.iter().copied().fold(0.0, f32::max);
            assert!(worst < 1.0e-3, "({temperature}, {tint}) jumps by {worst}");
        }
        // Exposure alone is a plain multiplier on the same matrix step.
        let bright = GradeParams {
            exposure: 1.0,
            ..GradeParams::IDENTITY
        }
        .uniform();
        assert_eq!(bright.colour[0][3], 1.0);
        assert_eq!(mat(bright.colour), Mat3::IDENTITY * 2.0);
    }

    /// The linear sRGB to XYZ matrix built from the primaries matches the one IEC 61966-2-1
    /// prints (to its four decimals), and equal RGB is the white point.
    #[test]
    fn the_srgb_matrix_built_from_the_primaries_matches_the_standard() {
        let printed = [
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505],
        ];
        let m = srgb_to_xyz();
        for (row, values) in printed.iter().enumerate() {
            for (col, want) in values.iter().enumerate() {
                let got = m.col(col)[row];
                assert!(
                    (got - want).abs() < 1.0e-4,
                    "[{row}][{col}] {got} vs {want}"
                );
            }
        }
        let white = m * DVec3::ONE;
        assert!((white - xy_to_xyz(D65_XY)).abs().max_element() < 1.0e-12);
    }

    /// The locus approximation meets black bodies of known chromaticity: CIE illuminant A
    /// (defined as a black body at 2856 K) and tabulated points from 1,000 K to 10,000 K.
    #[test]
    fn the_locus_approximation_meets_known_black_bodies() {
        for (kelvin, x, y) in [
            (1000.0, 0.6528, 0.3444),
            (2856.0, 0.447_57, 0.407_45),
            (4000.0, 0.3805, 0.3768),
            (6500.0, 0.3135, 0.3237),
            (10_000.0, 0.2807, 0.2884),
        ] {
            let got = planckian_uv(kelvin);
            let want = xy_to_uv(DVec2::new(x, y));
            let error = (got - want).abs().max_element();
            assert!(error < 2.0e-4, "{kelvin} K: {got:?} vs {want:?}");
        }
        // The cold end: smooth and finite far past the fitted range.
        let far = planckian_uv(1.0e9);
        assert!(far.is_finite() && (far - planckian_uv(1.0e8)).length() < 1.0e-3);
    }

    /// The display's white lands on the picked white, at unit luminance, for every setting.
    #[test]
    fn white_lands_on_the_picked_white_at_the_same_luminance() {
        let to_xyz = srgb_to_xyz();
        for temperature in [-1.0, -0.6, -0.24, -0.04, 0.0, 0.08, 0.26, 0.7, 1.0] {
            for tint in [-1.0, -0.3, 0.0, 0.03, 0.14, 1.0] {
                if temperature == 0.0 && tint == 0.0 {
                    continue;
                }
                let m = white_balance_matrix(temperature, tint).as_dmat3();
                let white = to_xyz * (m * DVec3::ONE);
                let want = uv_to_xy(picked_white_uv(f64::from(temperature), f64::from(tint)));
                let xy = DVec2::new(white.x, white.y) / white.element_sum();
                assert!(
                    (xy - want).abs().max_element() < 2.0e-6,
                    "({temperature}, {tint}): white at {xy:?}, picked {want:?}"
                );
                assert!((white.y - 1.0).abs() < 2.0e-6, "luminance {}", white.y);
            }
        }
    }

    /// Temperature moves white along the locus (warmer: more red against blue, monotonic over
    /// the whole range, never past either end), tint moves it across: magenta takes green out of
    /// white, green puts it in, by the same Duv at the same strength.
    #[test]
    fn temperature_follows_the_locus_and_tint_crosses_it() {
        let ratio = |temperature: f32| {
            let white = white_balance_matrix(temperature, 0.0) * Vec3::ONE;
            white.x / white.z
        };
        let mut last = 0.0;
        for step in -100..=100 {
            let r = ratio(step as f32 / 100.0);
            assert!(r.is_finite() && r > last, "not warmer at {step}");
            last = r;
        }
        assert!(white_mired(-1.0) > 5.0, "full cooling stays on the locus");
        assert!((white_mired(1.0) - (1.0e6 / D65_KELVIN + WARM_MIRED)).abs() < 1.0e-9);
        // Tint: the offset from the untinted white is TINT_DUV * tint, across the locus.
        for temperature in [-0.5, 0.0, 0.5] {
            let plain = picked_white_uv(temperature, 0.0);
            for tint in [-1.0, -0.25, 0.25, 1.0] {
                let moved = picked_white_uv(temperature, tint) - plain;
                assert!((moved.length() - TINT_DUV * tint.abs()).abs() < 1.0e-12);
                let mired = white_mired(temperature);
                let slope = locus_at_mired(mired + 0.5) - locus_at_mired(mired - 0.5);
                assert!(moved.normalize().dot(slope.normalize()).abs() < 1.0e-9);
            }
        }
        let magenta = white_balance_matrix(0.0, 0.5) * Vec3::ONE;
        let green = white_balance_matrix(0.0, -0.5) * Vec3::ONE;
        assert!(magenta.y < magenta.x.min(magenta.z), "{magenta:?}");
        assert!(green.y > green.x.max(green.z), "{green:?}");
    }

    /// The shoulder is the identity when nothing can exceed 1, keeps slope 1 at the knee, lands
    /// the maximum on 1 and never decreases.
    #[test]
    fn the_shoulder_is_smooth_monotonic_and_lands_the_maximum_on_one() {
        for i in 0..=100 {
            let x = i as f32 / 100.0;
            assert_eq!(shoulder(x, 1.0), x);
        }
        let m = 1.8;
        assert!((shoulder(m, m) - 1.0).abs() < 1e-6);
        let mut last = 0.0;
        for i in 0..=1000 {
            let y = shoulder(i as f32 / 1000.0 * m, m);
            assert!(y >= last - 1e-7, "not monotonic at {i}");
            last = y;
        }
        let eps = 1e-3;
        let slope = (shoulder(0.75 + eps, m) - shoulder(0.75, m)) / eps;
        assert!((slope - 1.0).abs() < 0.01, "{slope}");
        // A strong exposure lift engages it, and white lands on 1, not past it.
        let bright = GradeParams {
            exposure: 1.0,
            ..GradeParams::IDENTITY
        };
        assert!(bright.uniform().flags[1] > 1.9);
        let white = bright.apply([1.0; 3]);
        assert!(white.iter().all(|&v| (v - 1.0).abs() < 1e-4), "{white:?}");
        let grey = bright.apply([0.3; 3]);
        assert!(grey[0] > 0.35, "{grey:?}");
    }

    #[test]
    fn oklab_round_trips() {
        for c in grid() {
            let lin = srgb_to_linear(Vec3::from(c));
            let back = oklab_to_linear(linear_to_oklab(lin));
            assert!((back - lin).abs().max_element() < 2e-5, "{c:?}");
        }
    }

    #[test]
    fn sanitize_turns_garbage_into_neutral_values() {
        let garbage = GradeParams {
            exposure: f32::NAN,
            contrast: f32::INFINITY,
            saturation: -3.0,
            fade: 7.0,
            ..GradeParams::IDENTITY
        }
        .sanitized();
        assert_eq!(garbage.exposure, 0.0);
        assert_eq!(garbage.contrast, 1.0);
        assert_eq!(garbage.saturation, 0.0);
        assert_eq!(garbage.fade, 1.0);
    }
}
