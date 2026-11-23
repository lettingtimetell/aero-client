//! `f64` vector maths with Java `Vec3` semantics.
//!
//! java: `net/minecraft/util/Vec3.java`
//!
//! [`glam::DVec3`] is used as the storage type — it is a plain `#[repr(C)]` struct of three
//! `f64`s with no SIMD lowering, so its layout and `+`/`-`/`*` are safe to rely on. The
//! *derived* operations below are hand-rolled rather than taken from `glam`, because vanilla's
//! versions are not the textbook ones:
//!
//! * `lengthVector` / `distanceTo` round-trip through `MathHelper.sqrt_double`, which returns a
//!   **`float`**. `DVec3::length()` would keep full `f64` precision and drift from the client.
//! * `normalize` compares against `1.0E-4` and returns the zero vector below that, instead of
//!   producing `NaN`.
//! * `rotatePitch` / `rotateYaw` use the [`crate::sin`] / [`crate::cos`] lookup table.
//!
//! `glam::Vec3` (`f32`) is re-exported for render-side code only. Hazard **H7**: anything that
//! feeds physics is `f64` end to end.

use crate::math_helper::{cos, sin, sqrt_d};
use glam::DVec3;

/// Construct a vector with Java `Vec3`'s constructor semantics.
///
/// Vanilla normalises `-0.0` to `+0.0` on every component (java: Vec3.java:16-29), which
/// changes the sign of downstream products. Use this wherever a `Vec3` would be constructed in
/// the Java, and plain [`DVec3::new`] for a raw triple.
#[inline]
#[must_use]
pub fn vec3(x: f64, y: f64, z: f64) -> DVec3 {
    DVec3::new(
        denormalize_zero(x),
        denormalize_zero(y),
        denormalize_zero(z),
    )
}

#[inline]
fn denormalize_zero(v: f64) -> f64 {
    if v == -0.0 {
        0.0
    } else {
        v
    }
}

/// Squared length. java: implicit in `Vec3.lengthVector`
#[inline]
#[must_use]
pub fn length_squared(v: DVec3) -> f64 {
    v.x * v.x + v.y * v.y + v.z * v.z
}

/// Length, via `MathHelper.sqrt_double` — i.e. **narrowed to `f32` and widened back**.
/// java: `Vec3.lengthVector` (Vec3.java:120)
#[inline]
#[must_use]
pub fn length(v: DVec3) -> f64 {
    f64::from(sqrt_d(length_squared(v)))
}

/// Squared distance between two points. java: `Vec3.squareDistanceTo`
#[inline]
#[must_use]
pub fn square_distance_to(a: DVec3, b: DVec3) -> f64 {
    let d0 = b.x - a.x;
    let d1 = b.y - a.y;
    let d2 = b.z - a.z;
    d0 * d0 + d1 * d1 + d2 * d2
}

/// Distance between two points, via `MathHelper.sqrt_double`. java: `Vec3.distanceTo`
#[inline]
#[must_use]
pub fn distance_to(a: DVec3, b: DVec3) -> f64 {
    f64::from(sqrt_d(square_distance_to(a, b)))
}

/// Unit vector, or the zero vector if shorter than `1.0E-4`. java: `Vec3.normalize`
#[inline]
#[must_use]
pub fn normalize(v: DVec3) -> DVec3 {
    let d0 = length(v);
    if d0 < 1.0e-4 {
        vec3(0.0, 0.0, 0.0)
    } else {
        vec3(v.x / d0, v.y / d0, v.z / d0)
    }
}

/// Rotate about the X axis by `pitch` **radians**, using the sine table.
/// java: `Vec3.rotatePitch` (Vec3.java:193)
#[inline]
#[must_use]
pub fn rotate_pitch(v: DVec3, pitch: f32) -> DVec3 {
    let f = f64::from(cos(pitch));
    let f1 = f64::from(sin(pitch));
    vec3(v.x, v.y * f + v.z * f1, v.z * f - v.y * f1)
}

/// Rotate about the Y axis by `yaw` **radians**, using the sine table.
/// java: `Vec3.rotateYaw` (Vec3.java:203)
#[inline]
#[must_use]
pub fn rotate_yaw(v: DVec3, yaw: f32) -> DVec3 {
    let f = f64::from(cos(yaw));
    let f1 = f64::from(sin(yaw));
    vec3(v.x * f + v.z * f1, v.y, v.z * f - v.x * f1)
}

/// The point on segment `a -> b` with the given `x`, if the segment crosses that plane.
/// java: `Vec3.getIntermediateWithXValue`
#[must_use]
pub fn intermediate_with_x(a: DVec3, b: DVec3, x: f64) -> Option<DVec3> {
    let d0 = b.x - a.x;
    let d1 = b.y - a.y;
    let d2 = b.z - a.z;
    if d0 * d0 < 1.000_000_011_686_097_4E-7 {
        return None;
    }
    let d3 = (x - a.x) / d0;
    if (0.0..=1.0).contains(&d3) {
        Some(vec3(a.x + d0 * d3, a.y + d1 * d3, a.z + d2 * d3))
    } else {
        None
    }
}

/// The point on segment `a -> b` with the given `y`, if the segment crosses that plane.
/// java: `Vec3.getIntermediateWithYValue`
#[must_use]
pub fn intermediate_with_y(a: DVec3, b: DVec3, y: f64) -> Option<DVec3> {
    let d0 = b.x - a.x;
    let d1 = b.y - a.y;
    let d2 = b.z - a.z;
    if d1 * d1 < 1.000_000_011_686_097_4E-7 {
        return None;
    }
    let d3 = (y - a.y) / d1;
    if (0.0..=1.0).contains(&d3) {
        Some(vec3(a.x + d0 * d3, a.y + d1 * d3, a.z + d2 * d3))
    } else {
        None
    }
}

/// The point on segment `a -> b` with the given `z`, if the segment crosses that plane.
/// java: `Vec3.getIntermediateWithZValue`
#[must_use]
pub fn intermediate_with_z(a: DVec3, b: DVec3, z: f64) -> Option<DVec3> {
    let d0 = b.x - a.x;
    let d1 = b.y - a.y;
    let d2 = b.z - a.z;
    if d2 * d2 < 1.000_000_011_686_097_4E-7 {
        return None;
    }
    let d3 = (z - a.z) / d2;
    if (0.0..=1.0).contains(&d3) {
        Some(vec3(a.x + d0 * d3, a.y + d1 * d3, a.z + d2 * d3))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_folds_negative_zero() {
        let v = vec3(-0.0, -0.0, -0.0);
        assert_eq!(v.x.to_bits(), 0.0_f64.to_bits());
        assert_eq!(v.y.to_bits(), 0.0_f64.to_bits());
        assert_eq!(v.z.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn length_narrows_through_f32() {
        // 1/3 is not representable; the f32 round-trip is observable.
        let v = DVec3::new(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
        let java_like = length(v);
        assert_eq!(
            java_like.to_bits(),
            f64::from(sqrt_d(length_squared(v))).to_bits()
        );
        assert_ne!(
            java_like.to_bits(),
            length_squared(v).sqrt().to_bits(),
            "vanilla's length must lose precision through float"
        );
    }

    #[test]
    fn normalize_zeroes_below_epsilon() {
        assert_eq!(normalize(DVec3::new(1e-9, 0.0, 0.0)), DVec3::ZERO);
        let n = normalize(DVec3::new(3.0, 4.0, 0.0));
        assert!((length(n) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rotation_uses_the_lookup_table() {
        let v = DVec3::new(1.0, 0.0, 0.0);
        let r = rotate_yaw(v, std::f32::consts::FRAC_PI_2);
        // The LUT is not exact at PI/2, so x is near zero but not zero.
        assert!(r.x.abs() < 1e-4);
        assert_ne!(r.x, 0.0);
    }
}
