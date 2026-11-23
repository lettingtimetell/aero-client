//! Axis-aligned bounding boxes and the three collision-sweep functions.
//!
//! java: `net/minecraft/util/AxisAlignedBB.java`
//!
//! # Argument-order convention — read this before using the sweeps
//!
//! In vanilla, `Entity.moveEntity` calls
//!
//! ```text
//! for (AxisAlignedBB box : collidingBoxes) { y = box.calculateYOffset(entityBox, y); }
//! ```
//!
//! so **the receiver is the world/block box and the argument is the moving entity's box**.
//! [`Aabb::calculate_x_offset`] and friends keep that order: `self` is the obstacle, `other` is
//! the mover, and the returned value is the (possibly shortened) distance the mover may travel.
//! Swapping them silently inverts every collision. Hazard **H7**: everything here is `f64`.

use crate::facing::Facing;
use crate::math_helper::{java_max_f64, java_min_f64};
use crate::pos::BlockPos;
use glam::DVec3;
use std::fmt::{self, Write};

/// An axis-aligned box. java: `AxisAlignedBB`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    /// Minimum corner (`minX`, `minY`, `minZ`).
    pub min: DVec3,
    /// Maximum corner (`maxX`, `maxY`, `maxZ`).
    pub max: DVec3,
}

impl Aabb {
    /// Build a box from two opposite corners, normalising per axis with **Java's** `Math.min`
    /// / `Math.max` (which propagate `NaN` and order `-0.0` below `0.0`).
    /// java: `AxisAlignedBB(double x6)` / `AxisAlignedBB.fromBounds`
    #[inline]
    #[must_use]
    pub fn from_bounds(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64) -> Aabb {
        Aabb {
            min: DVec3::new(
                java_min_f64(x1, x2),
                java_min_f64(y1, y2),
                java_min_f64(z1, z2),
            ),
            max: DVec3::new(
                java_max_f64(x1, x2),
                java_max_f64(y1, y2),
                java_max_f64(z1, z2),
            ),
        }
    }

    /// Build a box from two opposite corners, normalising them like Java's six-double
    /// `AxisAlignedBB` constructor.
    #[inline]
    #[must_use]
    pub fn new(min: DVec3, max: DVec3) -> Aabb {
        Aabb::from_bounds(min.x, min.y, min.z, max.x, max.y, max.z)
    }

    /// Build a box from two block positions without reordering their corners.
    ///
    /// This deliberately differs from [`Aabb::from_bounds`]: vanilla's
    /// `AxisAlignedBB(BlockPos, BlockPos)` copies each position directly, so callers which
    /// pass the corners in reverse retain an inverted box.
    #[inline]
    #[must_use]
    pub const fn from_block_positions(first: BlockPos, second: BlockPos) -> Aabb {
        Aabb {
            min: DVec3::new(first.x as f64, first.y as f64, first.z as f64),
            max: DVec3::new(second.x as f64, second.y as f64, second.z as f64),
        }
    }

    /// Extend the box to cover a movement delta, growing whichever face the motion faces.
    /// java: `AxisAlignedBB.addCoord`
    #[must_use]
    pub fn add_coord(&self, x: f64, y: f64, z: f64) -> Aabb {
        let mut d0 = self.min.x;
        let mut d1 = self.min.y;
        let mut d2 = self.min.z;
        let mut d3 = self.max.x;
        let mut d4 = self.max.y;
        let mut d5 = self.max.z;

        if x < 0.0 {
            d0 += x;
        } else if x > 0.0 {
            d3 += x;
        }
        if y < 0.0 {
            d1 += y;
        } else if y > 0.0 {
            d4 += y;
        }
        if z < 0.0 {
            d2 += z;
        } else if z > 0.0 {
            d5 += z;
        }

        Aabb::from_bounds(d0, d1, d2, d3, d4, d5)
    }

    /// Grow the box by the given amount on every side (negative shrinks).
    /// java: `AxisAlignedBB.expand`
    #[inline]
    #[must_use]
    pub fn expand(&self, x: f64, y: f64, z: f64) -> Aabb {
        Aabb::from_bounds(
            self.min.x - x,
            self.min.y - y,
            self.min.z - z,
            self.max.x + x,
            self.max.y + y,
            self.max.z + z,
        )
    }

    /// Shrink the box by the given amount on every side. java: `AxisAlignedBB.contract`
    #[inline]
    #[must_use]
    pub fn contract(&self, x: f64, y: f64, z: f64) -> Aabb {
        Aabb::from_bounds(
            self.min.x + x,
            self.min.y + y,
            self.min.z + z,
            self.max.x - x,
            self.max.y - y,
            self.max.z - z,
        )
    }

    /// Translate the box. java: `AxisAlignedBB.offset`
    #[inline]
    #[must_use]
    pub fn offset(&self, x: f64, y: f64, z: f64) -> Aabb {
        Aabb::from_bounds(
            self.min.x + x,
            self.min.y + y,
            self.min.z + z,
            self.max.x + x,
            self.max.y + y,
            self.max.z + z,
        )
    }

    /// Smallest box containing both. java: `AxisAlignedBB.union`
    #[inline]
    #[must_use]
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb::from_bounds(
            java_min_f64(self.min.x, other.min.x),
            java_min_f64(self.min.y, other.min.y),
            java_min_f64(self.min.z, other.min.z),
            java_max_f64(self.max.x, other.max.x),
            java_max_f64(self.max.y, other.max.y),
            java_max_f64(self.max.z, other.max.z),
        )
    }

    /// How far `other` may actually move along X before hitting `self`.
    ///
    /// If the two boxes do not overlap in Y **and** Z, `offset_x` is returned unchanged.
    /// Otherwise the value is clipped to the gap between the facing sides. See the module docs
    /// for the receiver/argument convention. java: `AxisAlignedBB.calculateXOffset`
    #[must_use]
    pub fn calculate_x_offset(&self, other: &Aabb, offset_x: f64) -> f64 {
        let mut offset_x = offset_x;
        if other.max.y > self.min.y
            && other.min.y < self.max.y
            && other.max.z > self.min.z
            && other.min.z < self.max.z
        {
            if offset_x > 0.0 && other.max.x <= self.min.x {
                let d1 = self.min.x - other.max.x;
                if d1 < offset_x {
                    offset_x = d1;
                }
            } else if offset_x < 0.0 && other.min.x >= self.max.x {
                let d0 = self.max.x - other.min.x;
                if d0 > offset_x {
                    offset_x = d0;
                }
            }
        }
        offset_x
    }

    /// How far `other` may actually move along Y before hitting `self`.
    /// java: `AxisAlignedBB.calculateYOffset`
    #[must_use]
    pub fn calculate_y_offset(&self, other: &Aabb, offset_y: f64) -> f64 {
        let mut offset_y = offset_y;
        if other.max.x > self.min.x
            && other.min.x < self.max.x
            && other.max.z > self.min.z
            && other.min.z < self.max.z
        {
            if offset_y > 0.0 && other.max.y <= self.min.y {
                let d1 = self.min.y - other.max.y;
                if d1 < offset_y {
                    offset_y = d1;
                }
            } else if offset_y < 0.0 && other.min.y >= self.max.y {
                let d0 = self.max.y - other.min.y;
                if d0 > offset_y {
                    offset_y = d0;
                }
            }
        }
        offset_y
    }

    /// How far `other` may actually move along Z before hitting `self`.
    /// java: `AxisAlignedBB.calculateZOffset`
    #[must_use]
    pub fn calculate_z_offset(&self, other: &Aabb, offset_z: f64) -> f64 {
        let mut offset_z = offset_z;
        if other.max.x > self.min.x
            && other.min.x < self.max.x
            && other.max.y > self.min.y
            && other.min.y < self.max.y
        {
            if offset_z > 0.0 && other.max.z <= self.min.z {
                let d1 = self.min.z - other.max.z;
                if d1 < offset_z {
                    offset_z = d1;
                }
            } else if offset_z < 0.0 && other.min.z >= self.max.z {
                let d0 = self.max.z - other.min.z;
                if d0 > offset_z {
                    offset_z = d0;
                }
            }
        }
        offset_z
    }

    /// Strict overlap on all three axes (touching faces do not intersect).
    /// java: `AxisAlignedBB.intersectsWith`
    #[inline]
    #[must_use]
    pub fn intersects_with(&self, other: &Aabb) -> bool {
        other.max.x > self.min.x
            && other.min.x < self.max.x
            && other.max.y > self.min.y
            && other.min.y < self.max.y
            && other.max.z > self.min.z
            && other.min.z < self.max.z
    }

    /// Strictly inside the box on all three axes. java: `AxisAlignedBB.isVecInside`
    #[inline]
    #[must_use]
    pub fn is_vec_inside(&self, v: DVec3) -> bool {
        v.x > self.min.x
            && v.x < self.max.x
            && v.y > self.min.y
            && v.y < self.max.y
            && v.z > self.min.z
            && v.z < self.max.z
    }

    /// Mean of the three edge lengths. java: `AxisAlignedBB.getAverageEdgeLength`
    #[inline]
    #[must_use]
    pub fn average_edge_length(&self) -> f64 {
        let d0 = self.max.x - self.min.x;
        let d1 = self.max.y - self.min.y;
        let d2 = self.max.z - self.min.z;
        (d0 + d1 + d2) / 3.0
    }

    /// java: `AxisAlignedBB.hasNaN`
    #[inline]
    #[must_use]
    pub fn has_nan(&self) -> bool {
        self.min.x.is_nan()
            || self.min.y.is_nan()
            || self.min.z.is_nan()
            || self.max.x.is_nan()
            || self.max.y.is_nan()
            || self.max.z.is_nan()
    }

    /// Whether an entity with this box is within Java's entity render-distance gate.
    ///
    /// This is the exact box-dependent half of `Entity.isInRangeToRenderDist`: a NaN edge
    /// length becomes `1.0`, then vanilla squares `averageEdgeLength * 64 * weight` and compares
    /// the supplied squared distance strictly. The entity/render owners still choose the weight
    /// and must call this method; it is deliberately kept here so the numeric rule has one oracle.
    #[inline]
    #[must_use]
    pub fn is_in_render_range(&self, distance_squared: f64, render_distance_weight: f64) -> bool {
        let average_edge_length = self.average_edge_length();
        let average_edge_length = if average_edge_length.is_nan() {
            1.0
        } else {
            average_edge_length
        };
        let range = average_edge_length * 64.0 * render_distance_weight;
        distance_squared < range * range
    }

    /// Return this box, or vanilla `Render.shouldRender`'s ±2-block fallback around `position`.
    ///
    /// Java replaces the frustum box when it contains NaN **or** has zero average edge length;
    /// the range gate still evaluates the entity's original bounding box separately.
    #[inline]
    #[must_use]
    pub fn render_box_or_fallback(&self, position: DVec3) -> Aabb {
        if self.has_nan() || self.average_edge_length() == 0.0 {
            Aabb::from_bounds(
                position.x - 2.0,
                position.y - 2.0,
                position.z - 2.0,
                position.x + 2.0,
                position.y + 2.0,
                position.z + 2.0,
            )
        } else {
            *self
        }
    }

    /// Ray/box intersection: the first hit point on the segment `a -> b` and the face hit.
    /// java: `AxisAlignedBB.calculateIntercept`
    ///
    /// Vanilla's tie-breaking order (`-X, +X, -Y, +Y, -Z, +Z`, strictly-closer wins) is
    /// preserved, because the block-outline highlight depends on it.
    #[must_use]
    pub fn calculate_intercept(&self, a: DVec3, b: DVec3) -> Option<(DVec3, Facing)> {
        use crate::vec::square_distance_to;
        use crate::vec::{intermediate_with_x, intermediate_with_y, intermediate_with_z};

        let v0 = intermediate_with_x(a, b, self.min.x).filter(|v| self.is_vec_in_yz(*v));
        let v1 = intermediate_with_x(a, b, self.max.x).filter(|v| self.is_vec_in_yz(*v));
        let v2 = intermediate_with_y(a, b, self.min.y).filter(|v| self.is_vec_in_xz(*v));
        let v3 = intermediate_with_y(a, b, self.max.y).filter(|v| self.is_vec_in_xz(*v));
        let v4 = intermediate_with_z(a, b, self.min.z).filter(|v| self.is_vec_in_xy(*v));
        let v5 = intermediate_with_z(a, b, self.max.z).filter(|v| self.is_vec_in_xy(*v));

        let candidates = [
            (v0, Facing::West),
            (v1, Facing::East),
            (v2, Facing::Down),
            (v3, Facing::Up),
            (v4, Facing::North),
            (v5, Facing::South),
        ];

        let mut best: Option<(DVec3, Facing)> = None;
        for (candidate, facing) in candidates {
            if let Some(v) = candidate {
                let better = match best {
                    None => true,
                    Some((cur, _)) => square_distance_to(a, v) < square_distance_to(a, cur),
                };
                if better {
                    best = Some((v, facing));
                }
            }
        }
        best
    }

    #[inline]
    fn is_vec_in_yz(&self, v: DVec3) -> bool {
        v.y >= self.min.y && v.y <= self.max.y && v.z >= self.min.z && v.z <= self.max.z
    }

    #[inline]
    fn is_vec_in_xz(&self, v: DVec3) -> bool {
        v.x >= self.min.x && v.x <= self.max.x && v.z >= self.min.z && v.z <= self.max.z
    }

    #[inline]
    fn is_vec_in_xy(&self, v: DVec3) -> bool {
        v.x >= self.min.x && v.x <= self.max.x && v.y >= self.min.y && v.y <= self.max.y
    }
}

impl fmt::Display for Aabb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "box[")
            .and_then(|()| write_java_double(f, self.min.x))
            .and_then(|()| f.write_str(", "))
            .and_then(|()| write_java_double(f, self.min.y))
            .and_then(|()| f.write_str(", "))
            .and_then(|()| write_java_double(f, self.min.z))
            .and_then(|()| f.write_str(" -> "))
            .and_then(|()| write_java_double(f, self.max.x))
            .and_then(|()| f.write_str(", "))
            .and_then(|()| write_java_double(f, self.max.y))
            .and_then(|()| f.write_str(", "))
            .and_then(|()| write_java_double(f, self.max.z))
            .and_then(|()| f.write_char(']'))
    }
}

/// Write an `f64` with Java 8 `Double.toString`'s surface notation.
///
/// Both Java and Rust first produce a shortest round-tripping decimal; this only reshapes the
/// spelling differences relevant to `AxisAlignedBB.toString`: Java always keeps a fractional
/// digit, uses `E` outside `[1e-3, 1e7)`, and calls infinities `Infinity`.
fn write_java_double(f: &mut fmt::Formatter<'_>, value: f64) -> fmt::Result {
    // Rust's shortest representation spells the least positive subnormal `5e-324`, while
    // Java 8 `Double.toString` deliberately emits `4.9E-324`. This is observable through
    // AxisAlignedBB.toString (and the sign is preserved), so handle the two raw values before
    // the otherwise-shared shortest-decimal reshaping below.
    match value.to_bits() {
        0x0000_0000_0000_0001 => return f.write_str("4.9E-324"),
        0x8000_0000_0000_0001 => return f.write_str("-4.9E-324"),
        _ => {}
    }
    let raw = value.to_string();
    match raw.as_str() {
        "NaN" => return f.write_str("NaN"),
        "inf" => return f.write_str("Infinity"),
        "-inf" => return f.write_str("-Infinity"),
        _ => {}
    }

    let (sign, raw) = raw
        .strip_prefix('-')
        .map_or(("", raw.as_str()), |unsigned| ("-", unsigned));
    let (mantissa, explicit_exponent) = raw
        .split_once(['e', 'E'])
        .map_or((raw, 0), |(digits, exponent)| {
            (digits, exponent.parse::<i32>().expect("f64 exponent"))
        });
    let point = mantissa.find('.').unwrap_or(mantissa.len()) as i32;
    let mut digits: String = mantissa.chars().filter(|&ch| ch != '.').collect();
    let leading_zeroes = digits.bytes().take_while(|&byte| byte == b'0').count();
    let decimal_position = point + explicit_exponent - leading_zeroes as i32;
    digits.drain(..leading_zeroes);

    if digits.is_empty() {
        return write!(f, "{sign}0.0");
    }
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }

    let scientific_exponent = decimal_position - 1;
    if (-3..7).contains(&scientific_exponent) {
        f.write_str(sign)?;
        if decimal_position <= 0 {
            f.write_str("0.")?;
            for _ in 0..-decimal_position {
                f.write_char('0')?;
            }
            f.write_str(&digits)
        } else {
            let decimal_position = decimal_position as usize;
            if decimal_position >= digits.len() {
                f.write_str(&digits)?;
                for _ in digits.len()..decimal_position {
                    f.write_char('0')?;
                }
                f.write_str(".0")
            } else {
                f.write_str(&digits[..decimal_position])?;
                f.write_char('.')?;
                f.write_str(&digits[decimal_position..])
            }
        }
    } else {
        write!(f, "{sign}{}.", &digits[..1])?;
        if digits.len() == 1 {
            f.write_char('0')?;
        } else {
            f.write_str(&digits[1..])?;
        }
        write!(f, "E{scientific_exponent}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_at(x: f64, y: f64, z: f64) -> Aabb {
        Aabb::from_bounds(x, y, z, x + 1.0, y + 1.0, z + 1.0)
    }

    #[test]
    fn constructor_orders_corners() {
        let b = Aabb::from_bounds(1.0, 2.0, 3.0, -1.0, -2.0, -3.0);
        assert_eq!(b.min, DVec3::new(-1.0, -2.0, -3.0));
        assert_eq!(b.max, DVec3::new(1.0, 2.0, 3.0));

        // `new` is the constructor used by collision/state call sites; it must not be an
        // unchecked escape hatch from the Java constructor's ordering contract.
        let b = Aabb::new(DVec3::new(1.0, 2.0, 3.0), DVec3::new(-1.0, -2.0, -3.0));
        assert_eq!(b.min, DVec3::new(-1.0, -2.0, -3.0));
        assert_eq!(b.max, DVec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn block_position_constructor_keeps_its_argument_order() {
        let first = BlockPos::new(12, -4, 7);
        let second = BlockPos::new(-3, 19, -8);
        let b = Aabb::from_block_positions(first, second);

        assert_eq!(b.min, DVec3::new(12.0, -4.0, 7.0));
        assert_eq!(b.max, DVec3::new(-3.0, 19.0, -8.0));
        assert!(
            b.min.x > b.max.x && b.min.z > b.max.z,
            "a sorting mutant would erase vanilla's raw BlockPos-pair constructor ordering"
        );
    }

    #[test]
    fn display_uses_java_double_to_string_notation() {
        let b = Aabb {
            min: DVec3::new(1.0e-9, 1.0e20, -1.0e-7),
            max: DVec3::new(f64::NAN, 0.001, f64::NEG_INFINITY),
        };
        assert_eq!(
            b.to_string(),
            "box[1.0E-9, 1.0E20, -1.0E-7 -> NaN, 0.001, -Infinity]"
        );
        assert_eq!(
            Aabb::from_bounds(0.0, 0.0, 0.0, 1.0e7, 1.0e-4, 1.0).to_string(),
            "box[0.0, 0.0, 0.0 -> 1.0E7, 1.0E-4, 1.0]",
            "the Java notation boundaries bite decimal-point, exponent, and Infinity mutants"
        );
    }

    #[test]
    fn union_average_edge_length_and_nan_match_java() {
        let a = Aabb::from_bounds(-0.0, 5.0, 8.0, 2.0, 9.0, 14.0);
        let b = Aabb::from_bounds(-3.0, 6.0, 1.0, 1.0, 12.0, 10.0);
        let union = a.union(&b);
        assert_eq!(union.min, DVec3::new(-3.0, 5.0, 1.0));
        assert_eq!(union.max, DVec3::new(2.0, 12.0, 14.0));
        assert_eq!(a.average_edge_length(), 4.0);
        assert!(!a.has_nan());

        let nan = Aabb {
            min: DVec3::new(f64::NAN, 0.0, 0.0),
            max: DVec3::new(1.0, 1.0, 1.0),
        };
        assert!(nan.has_nan(), "a min-only NaN bites max-only scan mutants");
        assert!(nan.average_edge_length().is_nan());
    }

    #[test]
    fn entity_render_distance_uses_nan_one_and_a_strict_square_limit() {
        let b = Aabb::from_bounds(0.0, 0.0, 0.0, 2.0, 4.0, 6.0);
        // Average edge length = 4, so Java's (4 * 64 * 0.5)^2 gate is 16,384.
        assert!(b.is_in_render_range(16_383.999, 0.5));
        assert!(
            !b.is_in_render_range(16_384.0, 0.5),
            "a <= comparison mutant accepts the exact Java boundary"
        );

        let nan = Aabb {
            min: DVec3::new(f64::NAN, 0.0, 0.0),
            max: DVec3::new(1.0, 1.0, 1.0),
        };
        assert!(nan.is_in_render_range(4_095.0, 1.0));
        assert!(
            !nan.is_in_render_range(4_096.0, 1.0),
            "the NaN-to-1.0 fallback, not NaN comparison semantics, controls this gate"
        );
    }

    #[test]
    fn render_fallback_replaces_nan_and_zero_size_boxes_only() {
        let position = DVec3::new(-17.5, 64.0, 3.25);
        let zero = Aabb::from_bounds(1.0, 2.0, 3.0, 1.0, 2.0, 3.0);
        assert_eq!(
            zero.render_box_or_fallback(position),
            Aabb::from_bounds(-19.5, 62.0, 1.25, -15.5, 66.0, 5.25),
            "zero average edge length is a fallback condition even without NaN"
        );

        let nan = Aabb {
            min: DVec3::new(f64::NAN, 0.0, 0.0),
            max: DVec3::new(1.0, 1.0, 1.0),
        };
        assert_eq!(
            nan.render_box_or_fallback(position).min,
            DVec3::new(-19.5, 62.0, 1.25)
        );

        let normal = Aabb::from_bounds(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
        assert_eq!(normal.render_box_or_fallback(position), normal);
    }

    #[test]
    fn sweep_clips_to_the_gap() {
        // Obstacle at x=1..2, mover at x=-1..0 travelling +2 in X. It may only travel 1.0.
        let obstacle = unit_at(1.0, 0.0, 0.0);
        let mover = Aabb::from_bounds(-1.0, 0.0, 0.0, 0.0, 1.0, 1.0);
        assert_eq!(obstacle.calculate_x_offset(&mover, 2.0), 1.0);
        // Travelling away is unaffected.
        assert_eq!(obstacle.calculate_x_offset(&mover, -2.0), -2.0);
        // No Y/Z overlap: the offset passes straight through.
        let elsewhere = Aabb::from_bounds(-1.0, 5.0, 0.0, 0.0, 6.0, 1.0);
        assert_eq!(obstacle.calculate_x_offset(&elsewhere, 2.0), 2.0);
    }

    #[test]
    fn sweep_handles_all_three_axes() {
        let floor = Aabb::from_bounds(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
        let falling = Aabb::from_bounds(0.2, 2.0, 0.2, 0.8, 3.8, 0.8);
        assert_eq!(floor.calculate_y_offset(&falling, -5.0), -1.0);
        let wall = unit_at(0.0, 0.0, 1.0);
        let mover = Aabb::from_bounds(0.0, 0.0, -1.0, 1.0, 1.0, 0.0);
        assert_eq!(wall.calculate_z_offset(&mover, 3.0), 1.0);
    }

    #[test]
    fn touching_faces_do_not_intersect() {
        let a = unit_at(0.0, 0.0, 0.0);
        let b = unit_at(1.0, 0.0, 0.0);
        assert!(!a.intersects_with(&b));
        assert!(a.intersects_with(&unit_at(0.5, 0.0, 0.0)));
    }

    #[test]
    fn add_coord_grows_the_leading_face() {
        let a = unit_at(0.0, 0.0, 0.0);
        let grown = a.add_coord(2.0, -3.0, 0.0);
        assert_eq!(grown.min, DVec3::new(0.0, -3.0, 0.0));
        assert_eq!(grown.max, DVec3::new(3.0, 1.0, 1.0));
    }

    #[test]
    fn intercept_picks_the_near_face() {
        let a = unit_at(0.0, 0.0, 0.0);
        let hit = a.calculate_intercept(DVec3::new(-2.0, 0.5, 0.5), DVec3::new(2.0, 0.5, 0.5));
        let (point, facing) = hit.expect("ray must hit");
        assert_eq!(facing, Facing::West);
        assert_eq!(point.x, 0.0);
    }

    #[test]
    fn intercept_corner_ties_retain_the_first_java_face() {
        let box_ = unit_at(0.0, 0.0, 0.0);
        let hit = box_.calculate_intercept(DVec3::new(-1.0, -1.0, 0.5), DVec3::new(1.0, 1.0, 0.5));
        assert_eq!(
            hit,
            Some((DVec3::new(0.0, 0.0, 0.5), Facing::West)),
            "West is tested before Down and equal distance must not replace it"
        );
        assert!(
            box_.calculate_intercept(DVec3::new(-2.0, 2.0, 0.5), DVec3::new(-1.0, 2.0, 0.5))
                .is_none(),
            "an open ray remains a miss"
        );
    }
}
