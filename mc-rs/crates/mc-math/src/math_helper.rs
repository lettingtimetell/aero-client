//! Bit-exact port of `net/minecraft/util/MathHelper.java` (Minecraft 1.8.9).
//!
//! Every function here reproduces the Java expression *as written*, including its evaluation
//! order, its narrowing casts, and its integer overflow. `sin`/`cos` are a 65,536-entry lookup table, not libm. Substituting `f32::sin` desyncs
//! movement over a few hundred ticks and gets the client kicked by anticheat.
//!
//! ## Java cast semantics used throughout
//!
//! Java's `(int)`/`(long)` narrowing conversion from a floating-point value rounds toward zero,
//! saturates at the target's `MIN`/`MAX`, and maps `NaN` to `0`. Rust's `as` on floats has had
//! exactly these semantics since 1.45, so `value as i32` is a faithful translation and is used
//! directly. Java's *integer* arithmetic wraps on overflow, so every `+`/`-`/`*`/`neg` that can
//! overflow is written with the explicit `wrapping_*` form — a debug-build panic there would be
//! a behaviour change, not a bug catch.

use std::sync::LazyLock;

// java: MathHelper.java:13 — `private static final float[] SIN_TABLE = new float[65536];`
//
// The table is built in the class initialiser as `(float)Math.sin(i * PI * 2 / 65536)`, i.e.
// computed in `double` and stored as `float`. `Math.sin` is only specified to within 1 ulp, and
// a 1-ulp difference in the `double` can change the `float` it rounds to, so the table is *not*
// recomputed here with `f64::sin` (which would make the result depend on the host libm).
// Instead the exact 65,536 words dumped from a real JVM are embedded. Regenerate with
// `javaref/generate.ps1`. `tests/differential.rs` re-checks them against the same dump.
static SIN_TABLE_BYTES: &[u8] = include_bytes!("tables/sin_table.bin");

/// The 65,536-entry sine lookup table, exactly as a JVM builds it.
static SIN_TABLE: LazyLock<Box<[f32; 65536]>> = LazyLock::new(|| {
    let v: Vec<f32> = SIN_TABLE_BYTES
        .chunks_exact(4)
        .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect();
    v.into_boxed_slice()
        .try_into()
        .expect("sin_table.bin must contain exactly 65536 little-endian f32 words")
});

// java: MathHelper.java:552-562 — the `atan2` support tables. `field_181164_e[j] = asin(j/256)`
// and `field_181165_f[j] = cos(asin(j/256))`. Same reasoning as SIN_TABLE: `Math.asin` and
// `Math.cos` are 1-ulp-specified, so the JVM's values are embedded rather than recomputed.
static ATAN2_TABLE_BYTES: &[u8] = include_bytes!("tables/atan2_tables.bin");

/// `(asin_table, cos_asin_table)`, 257 entries each.
type Atan2Tables = (Box<[f64; 257]>, Box<[f64; 257]>);

/// java: `field_181164_e`, `field_181165_f`.
static ATAN2_TABLES: LazyLock<Atan2Tables> = LazyLock::new(|| {
    let v: Vec<f64> = ATAN2_TABLE_BYTES
        .chunks_exact(8)
        .map(|c| {
            f64::from_bits(u64::from_le_bytes([
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
            ]))
        })
        .collect();
    assert_eq!(
        v.len(),
        514,
        "atan2_tables.bin must contain 257+257 f64 words"
    );
    let asin: Box<[f64; 257]> = v[..257].to_vec().into_boxed_slice().try_into().unwrap();
    let cos: Box<[f64; 257]> = v[257..].to_vec().into_boxed_slice().try_into().unwrap();
    (asin, cos)
});

// java: MathHelper.java:552 — `Double.longBitsToDouble(4805340802404319232L)`.
const ATAN2_BIAS: f64 = f64::from_bits(4_805_340_802_404_319_232_u64);

// java: MathHelper.java:551
const MULTIPLY_DE_BRUIJN_BIT_POSITION: [i32; 32] = [
    0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 31, 27, 13, 23, 21, 19, 16, 7, 26,
    12, 18, 6, 11, 5, 10, 9,
];

/// `MathHelper.SQRT_2` — `sqrt_float(2.0F)`. java: MathHelper.java:8
///
/// `(float)Math.sqrt(2.0)` is the correctly-rounded `f32` nearest to sqrt(2), so it is exactly
/// `f32::consts::SQRT_2`; `sqrt_2_constant_matches_the_function` asserts that bit-for-bit.
pub const SQRT_2: f32 = std::f32::consts::SQRT_2;

// =========================================================================================
// trigonometry
// =========================================================================================

/// Sine, looked up in the 65,536-entry table. java: `MathHelper.sin` (MathHelper.java:30)
///
/// **This is not `f32::sin`.** The table is sampled at `2*PI / 65536` and indexed by
/// `(int)(f * 10430.378F) & 65535`. Hazard H1 — do not "improve" it.
#[inline]
#[must_use]
pub fn sin(f: f32) -> f32 {
    SIN_TABLE[((f * 10430.378_f32) as i32 & 65535) as usize]
}

/// Cosine, looked up in the sine table with a quarter-turn offset. java: `MathHelper.cos`
#[inline]
#[must_use]
pub fn cos(f: f32) -> f32 {
    SIN_TABLE[((f * 10430.378_f32 + 16384.0_f32) as i32 & 65535) as usize]
}

/// Square root of a `f32`, computed in `f64` and narrowed. java: `MathHelper.sqrt_float`
#[inline]
#[must_use]
pub fn sqrt_f(f: f32) -> f32 {
    (f64::from(f)).sqrt() as f32
}

/// Square root of a `f64`, **returned as `f32`**. java: `MathHelper.sqrt_double`
///
/// The narrowing is deliberate and load-bearing: `Vec3::lengthVector` and friends round-trip
/// through `float` here, and skipping that step changes movement results.
#[inline]
#[must_use]
pub fn sqrt_d(d: f64) -> f32 {
    d.sqrt() as f32
}

/// Vanilla's fast `atan2`. java: `MathHelper.atan2` (MathHelper.java:411)
///
/// A polynomial approximation over the embedded `asin` tables — noticeably different from
/// `f64::atan2`, and what entity look-angles are computed with.
///
/// # Panics
/// Non-finite inputs can drive the table index out of range, exactly as they raise
/// `ArrayIndexOutOfBoundsException` in Java. Callers pass finite coordinates.
#[must_use]
pub fn atan2(y: f64, x: f64) -> f64 {
    let mut y = y;
    let mut x = x;
    let d0 = x * x + y * y;

    if d0.is_nan() {
        return f64::NAN;
    }

    let flag = y < 0.0;
    if flag {
        y = -y;
    }

    let flag1 = x < 0.0;
    if flag1 {
        x = -x;
    }

    let flag2 = y > x;
    if flag2 {
        std::mem::swap(&mut x, &mut y);
    }

    let (asin_table, cos_table) = &*ATAN2_TABLES;
    let d9 = fast_inv_sqrt(d0);
    x *= d9;
    y *= d9;
    let d2 = ATAN2_BIAS + y;
    let i = d2.to_bits() as i32;
    let d3 = asin_table[i as usize];
    let d4 = cos_table[i as usize];
    let d5 = d2 - ATAN2_BIAS;
    let d6 = y * d4 - x * d5;
    let d7 = (6.0 + d6 * d6) * d6 * 0.166_666_666_666_666_66;
    let mut d8 = d3 + d7;

    if flag2 {
        d8 = std::f64::consts::FRAC_PI_2 - d8;
    }
    if flag1 {
        d8 = std::f64::consts::PI - d8;
    }
    if flag {
        d8 = -d8;
    }
    d8
}

/// One Newton step of the `double` fast inverse square root. java: `MathHelper.func_181161_i`
#[inline]
#[must_use]
pub fn fast_inv_sqrt(value: f64) -> f64 {
    let d0 = 0.5 * value;
    let i = value.to_bits() as i64;
    let i = 6_910_469_410_427_058_090_i64.wrapping_sub(i >> 1);
    let v = f64::from_bits(i as u64);
    v * (1.5 - d0 * v * v)
}

// =========================================================================================
// rounding
// =========================================================================================

/// Greatest integer `<=` the `f32` argument. java: `MathHelper.floor_float`
#[inline]
#[must_use]
pub fn floor_f32(value: f32) -> i32 {
    let i = value as i32;
    if value < i as f32 {
        i.wrapping_sub(1)
    } else {
        i
    }
}

/// Greatest integer `<=` the `f64` argument. java: `MathHelper.floor_double`
///
/// Not `f64::floor() as i32`: the Java form truncates first and *then* corrects, so its
/// saturation and overflow behaviour at the `i32` boundary differs from a naive port.
#[inline]
#[must_use]
pub fn floor_f64(value: f64) -> i32 {
    let i = value as i32;
    if value < i as f64 {
        i.wrapping_sub(1)
    } else {
        i
    }
}

/// `i64` version of [`floor_f64`]. java: `MathHelper.floor_double_long`
#[inline]
#[must_use]
pub fn floor_f64_i64(value: f64) -> i64 {
    let i = value as i64;
    if value < i as f64 {
        i.wrapping_sub(1)
    } else {
        i
    }
}

/// Smallest integer `>=` the `f32` argument. java: `MathHelper.ceiling_float_int`
#[inline]
#[must_use]
pub fn ceil_f32(value: f32) -> i32 {
    let i = value as i32;
    if value > i as f32 {
        i.wrapping_add(1)
    } else {
        i
    }
}

/// Smallest integer `>=` the `f64` argument. java: `MathHelper.ceiling_double_int`
#[inline]
#[must_use]
pub fn ceil_f64(value: f64) -> i32 {
    let i = value as i32;
    if value > i as f64 {
        i.wrapping_add(1)
    } else {
        i
    }
}

/// `(int)(value + 1024.0) - 1024` — a truncation capped near `i32::MAX`.
/// java: `MathHelper.truncateDoubleToInt`
#[inline]
#[must_use]
pub fn truncate_f64_to_i32(value: f64) -> i32 {
    ((value + 1024.0) as i32).wrapping_sub(1024)
}

/// `(int)(value >= 0 ? value : -value + 1)` — folds negatives above the positives.
/// java: `MathHelper.func_154353_e`
#[inline]
#[must_use]
pub fn fold_sign_to_i32(value: f64) -> i32 {
    (if value >= 0.0 { value } else { -value + 1.0 }) as i32
}

/// Fractional part: `value - floor(value)`. java: `MathHelper.func_181162_h`
#[inline]
#[must_use]
pub fn frac_f64(value: f64) -> f64 {
    value - value.floor()
}

// =========================================================================================
// absolute value, clamping, interpolation
// =========================================================================================

/// `value >= 0 ? value : -value`. java: `MathHelper.abs`
///
/// Note this is **not** `f32::abs`: `abs_f32(-0.0)` is `-0.0` (because `-0.0 >= 0.0`), and a
/// `NaN` comes back with its sign bit flipped.
#[inline]
#[must_use]
pub fn abs_f32(value: f32) -> f32 {
    if value >= 0.0 {
        value
    } else {
        -value
    }
}

/// `value >= 0 ? value : -value`, wrapping at `i32::MIN`. java: `MathHelper.abs_int`
#[inline]
#[must_use]
pub fn abs_i32(value: i32) -> i32 {
    if value >= 0 {
        value
    } else {
        value.wrapping_neg()
    }
}

/// Larger of the two magnitudes. java: `MathHelper.abs_max`
#[inline]
#[must_use]
pub fn abs_max(a: f64, b: f64) -> f64 {
    let mut a = a;
    let mut b = b;
    if a < 0.0 {
        a = -a;
    }
    if b < 0.0 {
        b = -b;
    }
    if a > b {
        a
    } else {
        b
    }
}

/// `num` clamped to `[min, max]`. java: `MathHelper.clamp_int`
#[inline]
#[must_use]
pub fn clamp_i32(num: i32, min: i32, max: i32) -> i32 {
    if num < min {
        min
    } else if num > max {
        max
    } else {
        num
    }
}

/// `num` clamped to `[min, max]`. java: `MathHelper.clamp_float`
///
/// A `NaN` `num` fails both comparisons and is returned unchanged — unlike `f32::clamp`,
/// which panics if the bounds are unordered.
#[inline]
#[must_use]
pub fn clamp_f32(num: f32, min: f32, max: f32) -> f32 {
    if num < min {
        min
    } else if num > max {
        max
    } else {
        num
    }
}

/// `num` clamped to `[min, max]`. java: `MathHelper.clamp_double`
#[inline]
#[must_use]
pub fn clamp_f64(num: f64, min: f64, max: f64) -> f64 {
    if num < min {
        min
    } else if num > max {
        max
    } else {
        num
    }
}

// =========================================================================================
// integer helpers
// =========================================================================================

/// Reduce a `f32` angle in degrees to `[-180, 180)`. java: `MathHelper.wrapAngleTo180_float`
#[inline]
#[must_use]
pub fn wrap_degrees_f32(value: f32) -> f32 {
    // HotSpot's `frem` result needs an explicit spelling here.  LLVM's `frem`
    // preserves a NaN sign where the JVM clears it, and it chooses a different
    // quiet NaN for infinities.  Keep the finite path as the direct Java
    // source translation below.
    if value.is_nan() {
        return f32::from_bits((value.to_bits() & 0x007f_ffff) | 0x7fc0_0000);
    }
    if value.is_infinite() {
        return f32::from_bits(0xffc0_0000);
    }
    let mut value = value % 360.0_f32;
    if value >= 180.0 {
        value -= 360.0;
    }
    if value < -180.0 {
        value += 360.0;
    }
    value
}

/// Reduce a `f64` angle in degrees to `[-180, 180)`. java: `MathHelper.wrapAngleTo180_double`
#[inline]
#[must_use]
pub fn wrap_degrees_f64(value: f64) -> f64 {
    // See `wrap_degrees_f32`: mirror HotSpot's non-finite `drem` bit pattern
    // before entering the otherwise literal finite Java translation.
    if value.is_nan() {
        return f64::from_bits((value.to_bits() & 0x000f_ffff_ffff_ffff) | 0x7ff8_0000_0000_0000);
    }
    if value.is_infinite() {
        return f64::from_bits(0xfff8_0000_0000_0000);
    }
    let mut value = value % 360.0_f64;
    if value >= 180.0 {
        value -= 360.0;
    }
    if value < -180.0 {
        value += 360.0;
    }
    value
}

/// Round up to the next power of two, Java-style (wraps, and maps `0` to `0`).
/// java: `MathHelper.roundUpToPowerOfTwo`
#[inline]
#[must_use]
pub fn round_up_to_power_of_two(value: i32) -> i32 {
    let mut i = value.wrapping_sub(1);
    i |= i >> 1;
    i |= i >> 2;
    i |= i >> 4;
    i |= i >> 8;
    i |= i >> 16;
    i.wrapping_add(1)
}

/// java: `MathHelper.isPowerOfTwo`
#[inline]
#[must_use]
fn is_power_of_two(value: i32) -> bool {
    value != 0 && (value & value.wrapping_sub(1)) == 0
}

/// java: `MathHelper.calculateLogBaseTwoDeBruijn`
#[inline]
fn calculate_log_base_two_de_bruijn(value: i32) -> i32 {
    let value = if is_power_of_two(value) {
        value
    } else {
        round_up_to_power_of_two(value)
    };
    let idx = ((i64::from(value).wrapping_mul(125_613_361) >> 27) as i32) & 31;
    MULTIPLY_DE_BRUIJN_BIT_POSITION[idx as usize]
}

/// Floor of the base-2 logarithm — the index of the highest set bit.
/// java: `MathHelper.calculateLogBaseTwo`
#[inline]
#[must_use]
pub fn calculate_log_base_two(value: i32) -> i32 {
    calculate_log_base_two_de_bruijn(value).wrapping_sub(i32::from(!is_power_of_two(value)))
}

/// Round `value` up to the next multiple of `step`. java: `MathHelper.roundUp`
#[inline]
#[must_use]
pub fn round_up(value: i32, step: i32) -> i32 {
    if step == 0 {
        return 0;
    }
    if value == 0 {
        return step;
    }
    let step = if value < 0 {
        step.wrapping_mul(-1)
    } else {
        step
    };
    let i = value.wrapping_rem(step);
    if i == 0 {
        value
    } else {
        value.wrapping_add(step).wrapping_sub(i)
    }
}

/// A deterministic hash of a block coordinate, used to seed per-position randomness.
/// java: `MathHelper.getCoordinateRandom`
#[inline]
#[must_use]
pub fn coordinate_random(x: i32, y: i32, z: i32) -> i64 {
    let mut i = i64::from(x.wrapping_mul(3_129_871))
        ^ i64::from(z).wrapping_mul(116_129_781)
        ^ i64::from(y);
    i = i
        .wrapping_mul(i)
        .wrapping_mul(42_317_861)
        .wrapping_add(i.wrapping_mul(11));
    i
}

// =========================================================================================
// colour packing
// =========================================================================================

/// Pack three 0..255 channels into `0xRRGGBB`. java: `MathHelper.func_180181_b`
#[inline]
#[must_use]
pub fn pack_color(r: i32, g: i32, b: i32) -> i32 {
    let v = (r << 8).wrapping_add(g);
    (v << 8).wrapping_add(b)
}

/// Pack three 0..1 float channels into `0xRRGGBB`. java: `MathHelper.func_180183_b`
#[inline]
#[must_use]
pub fn pack_color_f32(r: f32, g: f32, b: f32) -> i32 {
    pack_color(
        floor_f32(r * 255.0),
        floor_f32(g * 255.0),
        floor_f32(b * 255.0),
    )
}

/// Per-channel multiply of two packed colours, keeping `a`'s alpha.
/// java: `MathHelper.func_180188_d`
#[inline]
#[must_use]
pub fn multiply_color(a: i32, b: i32) -> i32 {
    let i = (a & 16_711_680) >> 16;
    let j = (b & 16_711_680) >> 16;
    let k = (a & 65_280) >> 8;
    let l = (b & 65_280) >> 8;
    let i1 = a & 255;
    let j1 = b & 255;
    let k1 = (i as f32 * j as f32 / 255.0_f32) as i32;
    let l1 = (k as f32 * l as f32 / 255.0_f32) as i32;
    let i2 = (i1 as f32 * j1 as f32 / 255.0_f32) as i32;
    (a & -16_777_216) | (k1 << 16) | (l1 << 8) | i2
}

/// HSV (each component in `0..1`) to a packed `0xRRGGBB`. java: `MathHelper.hsvToRGB`
///
/// # Panics
/// If `(int)(hue * 6) % 6` falls outside `0..=5`, exactly as Java throws a `RuntimeException`.
#[must_use]
pub fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> i32 {
    let i = ((hue * 6.0_f32) as i32) % 6;
    let f = hue * 6.0_f32 - i as f32;
    let f1 = value * (1.0 - saturation);
    let f2 = value * (1.0 - f * saturation);
    let f3 = value * (1.0 - (1.0 - f) * saturation);

    let (f4, f5, f6) = match i {
        0 => (value, f3, f1),
        1 => (f2, value, f1),
        2 => (f1, value, f3),
        3 => (f1, f2, value),
        4 => (f3, f1, value),
        5 => (value, f1, f2),
        _ => panic!(
            "Something went wrong when converting from HSV to RGB. Input was {hue}, \
             {saturation}, {value}"
        ),
    };

    let j = clamp_i32((f4 * 255.0_f32) as i32, 0, 255);
    let k = clamp_i32((f5 * 255.0_f32) as i32, 0, 255);
    let l = clamp_i32((f6 * 255.0_f32) as i32, 0, 255);
    (j << 16) | (k << 8) | l
}

// =========================================================================================
// java.lang.Math compatibility
// =========================================================================================

/// `java.lang.Math.min(double, double)`.
///
/// Differs from `f64::min` on two inputs that matter: a `NaN` argument propagates instead of
/// being ignored, and `min(0.0, -0.0)` is `-0.0`. `AxisAlignedBB`'s constructor uses this, so
/// getting it wrong perturbs collision boxes built from unordered corners.
#[inline]
#[must_use]
pub fn java_min_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        return a;
    }
    if a == 0.0 && b == 0.0 && b.to_bits() == (-0.0_f64).to_bits() {
        return b;
    }
    if a <= b {
        a
    } else {
        b
    }
}

/// `java.lang.Math.max(double, double)`. See [`java_min_f64`] for why this is not `f64::max`.
#[inline]
#[must_use]
pub fn java_max_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        return a;
    }
    if a == 0.0 && b == 0.0 && a.to_bits() == (-0.0_f64).to_bits() {
        return b;
    }
    if a >= b {
        a
    } else {
        b
    }
}

// `strict_log` below is transliterated from fdlibm 5.3 `e_log.c`, whose notice is kept here as
// its licence requires:
//
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunSoft, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================

/// `java.lang.StrictMath.log` — a port of fdlibm 5.3 `__ieee754_log`.
///
/// `java.util.Random.nextGaussian` calls `StrictMath.log`, which is specified to be fdlibm and
/// therefore identical on every JVM. `f64::ln` is *not*: it defers to the platform libm, which
/// differs from fdlibm in the last bit often enough to break bit-exactness (and would make our
/// own output host-dependent). This is that fdlibm routine, transliterated.
#[allow(clippy::excessive_precision)]
#[must_use]
pub fn strict_log(x: f64) -> f64 {
    const LN2_HI: f64 = 6.93147180369123816490e-01;
    const LN2_LO: f64 = 1.90821492927058770002e-10;
    const TWO54: f64 = 1.80143985094819840000e+16;
    const LG1: f64 = 6.666666666666735130e-01;
    const LG2: f64 = 3.999999999940941908e-01;
    const LG3: f64 = 2.857142874366239149e-01;
    const LG4: f64 = 2.222219843214978396e-01;
    const LG5: f64 = 1.818357216161805012e-01;
    const LG6: f64 = 1.531383769920937332e-01;
    const LG7: f64 = 1.479819860511658591e-01;

    let mut x = x;
    let mut hx = (x.to_bits() >> 32) as u32 as i32;
    let lx = x.to_bits() as u32;
    let mut k: i32 = 0;

    if hx < 0x0010_0000 {
        // x < 2**-1022
        if ((hx & 0x7fff_ffff) as u32 | lx) == 0 {
            return f64::NEG_INFINITY; // log(+-0) = -inf
        }
        if hx < 0 {
            return f64::NAN; // log(-#) = NaN
        }
        k -= 54;
        x *= TWO54; // subnormal: scale up
        hx = (x.to_bits() >> 32) as u32 as i32;
    }
    if hx >= 0x7ff0_0000 {
        return x + x;
    }
    k += (hx >> 20) - 1023;
    hx &= 0x000f_ffff;
    let i = (hx + 0x0009_5f64) & 0x0010_0000;
    // __HI(x) = hx | (i ^ 0x3ff00000): normalise x into [1,2) or [2,4)/2
    x = f64::from_bits(
        ((((hx | (i ^ 0x3ff0_0000)) as u32) as u64) << 32) | u64::from(x.to_bits() as u32),
    );
    k += i >> 20;
    let f = x - 1.0;

    if (0x000f_ffff & (2 + hx)) < 3 {
        // |f| < 2**-20
        if f == 0.0 {
            if k == 0 {
                return 0.0;
            }
            let dk = f64::from(k);
            return dk * LN2_HI + dk * LN2_LO;
        }
        let r = f * f * (0.5 - 0.33333333333333333 * f);
        if k == 0 {
            return f - r;
        }
        let dk = f64::from(k);
        return dk * LN2_HI - ((r - dk * LN2_LO) - f);
    }

    let s = f / (2.0 + f);
    let dk = f64::from(k);
    let z = s * s;
    let mut i = hx - 0x0006_147a;
    let w = z * z;
    let j = 0x0006_b851 - hx;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    i |= j;
    let r = t2 + t1;

    if i > 0 {
        let hfsq = 0.5 * f * f;
        if k == 0 {
            f - (hfsq - s * (hfsq + r))
        } else {
            dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f)
        }
    } else if k == 0 {
        f - s * (f - r)
    } else {
        dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sin_cos_are_the_table_not_libm() {
        // The table is sampled, so almost every angle differs from libm in the low bits.
        // That difference *is* the parity contract (H1) — count it rather than tolerate it.
        let mut differing = 0;
        for i in 0..1000 {
            let a = i as f32 * 0.007;
            if sin(a).to_bits() != a.sin().to_bits() {
                differing += 1;
            }
            // ...but never by more than the sample spacing.
            assert!(
                (sin(a) - a.sin()).abs() < 1.0e-4,
                "sin({a}) drifted too far"
            );
        }
        assert!(
            differing > 900,
            "the LUT must differ from f32::sin nearly everywhere, got {differing}/1000"
        );

        // The quarter-turn sample points land exactly, because the table is 65536 samples of
        // a full turn and 16384 of them is exactly PI/2.
        assert_eq!(sin(0.0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(cos(0.0).to_bits(), 1.0_f32.to_bits());
        assert_eq!(
            sin(std::f32::consts::FRAC_PI_2).to_bits(),
            1.0_f32.to_bits()
        );
        assert!(sin(-1.0) < 0.0 && cos(-1.0) > 0.0);
    }

    #[test]
    fn sqrt_2_constant_matches_the_function() {
        assert_eq!(SQRT_2.to_bits(), sqrt_f(2.0).to_bits());
    }

    #[test]
    fn floor_matches_java_on_negatives_and_edges() {
        assert_eq!(floor_f64(-0.5), -1);
        assert_eq!(floor_f64(-1.0), -1);
        assert_eq!(floor_f64(0.5), 0);
        assert_eq!(floor_f64(-0.0), 0);
        assert_eq!(floor_f64(f64::NAN), 0);
        assert_eq!(floor_f32(-0.5), -1);
        assert_eq!(floor_f64_i64(-0.5), -1);
        // (int)3e9 saturates to i32::MAX and no correction applies.
        assert_eq!(floor_f64(3.0e9), i32::MAX);
        // (int)-3e9 saturates to i32::MIN and the correction wraps, as it does in Java.
        assert_eq!(floor_f64(-3.0e9), i32::MAX);
    }

    #[test]
    fn ceil_matches_java() {
        assert_eq!(ceil_f64(0.5), 1);
        assert_eq!(ceil_f64(-0.5), 0);
        assert_eq!(ceil_f64(1.0), 1);
        assert_eq!(ceil_f32(0.5), 1);
        assert_eq!(ceil_f64(f64::NAN), 0);
    }

    #[test]
    fn abs_keeps_negative_zero() {
        // MathHelper.abs is `v >= 0 ? v : -v`, and -0.0 >= 0.0 is true.
        assert_eq!(abs_f32(-0.0).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(abs_f32(-1.0), 1.0);
        assert_eq!(abs_i32(i32::MIN), i32::MIN); // wraps, like Java
    }

    #[test]
    fn wrap_degrees_handles_non_finite() {
        // Bit-level vectors captured from the Java fixture's literal
        // `wrapAngleTo180_*` methods on HotSpot.  These catch the tempting
        // `value % 360` port, which produces a different NaN sign/payload in
        // LLVM for both a negative NaN and infinity.
        for (input, expected) in [
            (0x7fc0_0000, 0x7fc0_0000), // quiet NaN
            (0x7fc0_0001, 0x7fc0_0001), // payload survives
            (0x7fa0_0001, 0x7fe0_0001), // signaling NaN is quieted
            (0xffc1_2345, 0x7fc1_2345), // NaN sign is cleared
            (0x7f80_0000, 0xffc0_0000), // +infinity
            (0xff80_0000, 0xffc0_0000), // -infinity
        ] {
            assert_eq!(wrap_degrees_f32(f32::from_bits(input)).to_bits(), expected);
        }
        for (input, expected) in [
            (0x7ff8_0000_0000_0000, 0x7ff8_0000_0000_0000), // quiet NaN
            (0x7ff8_0000_0000_0001, 0x7ff8_0000_0000_0001), // payload survives
            (0x7ff0_0000_0000_0001, 0x7ff8_0000_0000_0001), // signaling NaN is quieted
            (0xfff8_1234_5678_9abc, 0x7ff8_1234_5678_9abc), // NaN sign is cleared
            (0x7ff0_0000_0000_0000, 0xfff8_0000_0000_0000), // +infinity
            (0xfff0_0000_0000_0000, 0xfff8_0000_0000_0000), // -infinity
        ] {
            assert_eq!(wrap_degrees_f64(f64::from_bits(input)).to_bits(), expected);
        }
        // The finite behaviour is the part that matters and is fully pinned by the fixture.
        assert_eq!(wrap_degrees_f32(370.0), 10.0);
        assert_eq!(wrap_degrees_f32(-190.0), 170.0);
        assert_eq!(wrap_degrees_f32(180.0), -180.0);
        assert_eq!(wrap_degrees_f64(540.0), -180.0);
    }

    #[test]
    fn java_min_max_differ_from_rust() {
        assert!(java_min_f64(f64::NAN, 1.0).is_nan());
        assert_eq!(f64::min(f64::NAN, 1.0), 1.0);
        assert_eq!(java_min_f64(0.0, -0.0).to_bits(), (-0.0_f64).to_bits());
        assert_eq!(java_max_f64(-0.0, 0.0).to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn log2_table() {
        assert_eq!(calculate_log_base_two(1), 0);
        assert_eq!(calculate_log_base_two(2), 1);
        assert_eq!(calculate_log_base_two(1024), 10);
        assert_eq!(calculate_log_base_two(1023), 9);
        assert_eq!(round_up_to_power_of_two(30_000_000), 33_554_432);
    }

    #[test]
    fn strict_log_agrees_with_libm_on_easy_values() {
        // Sanity only; `tests/differential.rs` does the bit-exact check against a JVM.
        for i in 1..1000 {
            let x = f64::from(i) / 500.0;
            assert!((strict_log(x) - x.ln()).abs() < 1e-15);
        }
        assert_eq!(strict_log(0.0), f64::NEG_INFINITY);
        assert!(strict_log(-1.0).is_nan());
        assert_eq!(strict_log(1.0), 0.0);
    }
}
