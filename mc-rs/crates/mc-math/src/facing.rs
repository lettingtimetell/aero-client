//! `Facing`, `Axis`, and `Plane`.
//!
//! java: `net/minecraft/util/EnumFacing.java`
//!
//! The discriminants are **vanilla ordinal order** — `Down, Up, North, South, West, East` =
//! `0..5`. That order goes on the wire (block-place, block-break, painting/item-frame spawn,
//! and every `EnumFacing` written as a byte), so it is not free to change.

use crate::math_helper::{abs_i32, floor_f64};
use crate::pos::BlockPos;
use crate::MathError;

/// A block face / cardinal direction. java: `EnumFacing`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Facing {
    /// -Y. Ordinal 0.
    Down = 0,
    /// +Y. Ordinal 1.
    Up = 1,
    /// -Z. Ordinal 2.
    North = 2,
    /// +Z. Ordinal 3.
    South = 3,
    /// -X. Ordinal 4.
    West = 4,
    /// +X. Ordinal 5.
    East = 5,
}

/// The axis a [`Facing`] lies on. java: `EnumFacing.Axis`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
    /// West/East.
    X,
    /// Down/Up.
    Y,
    /// North/South.
    Z,
}

/// Horizontal or vertical grouping of axes. java: `EnumFacing.Plane`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Plane {
    /// The X and Z axes.
    Horizontal,
    /// The Y axis.
    Vertical,
}

impl Axis {
    /// Lowercase serialised name (`"x"`, `"y"`, `"z"`). java: `Axis.getName`
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Axis::X => "x",
            Axis::Y => "y",
            Axis::Z => "z",
        }
    }

    /// Parse a lowercase axis name. java: `Axis.byName`
    ///
    /// # Errors
    /// [`MathError::UnknownAxis`] if the name is not `x`, `y`, or `z`.
    pub fn by_name(name: &str) -> Result<Axis, MathError> {
        match name.to_ascii_lowercase().as_str() {
            "x" => Ok(Axis::X),
            "y" => Ok(Axis::Y),
            "z" => Ok(Axis::Z),
            _ => Err(MathError::UnknownAxis(name.to_owned())),
        }
    }

    /// java: `Axis.getPlane`
    #[inline]
    #[must_use]
    pub const fn plane(self) -> Plane {
        match self {
            Axis::X | Axis::Z => Plane::Horizontal,
            Axis::Y => Plane::Vertical,
        }
    }

    /// java: `Axis.isHorizontal`
    #[inline]
    #[must_use]
    pub const fn is_horizontal(self) -> bool {
        matches!(self.plane(), Plane::Horizontal)
    }

    /// java: `Axis.isVertical`
    #[inline]
    #[must_use]
    pub const fn is_vertical(self) -> bool {
        matches!(self.plane(), Plane::Vertical)
    }
}

impl Facing {
    /// All six facings in vanilla ordinal order (D-U-N-S-W-E). java: `EnumFacing.VALUES`
    pub const VALUES: [Facing; 6] = [
        Facing::Down,
        Facing::Up,
        Facing::North,
        Facing::South,
        Facing::West,
        Facing::East,
    ];

    /// The four horizontal facings in vanilla's `horizontalIndex` order (S-W-N-E).
    /// java: `EnumFacing.HORIZONTALS`
    pub const HORIZONTALS: [Facing; 4] = [Facing::South, Facing::West, Facing::North, Facing::East];

    /// Vanilla ordinal, `0..=5`, D-U-N-S-W-E. java: `EnumFacing.getIndex`
    #[inline]
    #[must_use]
    pub const fn index(self) -> u8 {
        self as u8
    }

    /// Facing from a vanilla ordinal, wrapping like Java's `getFront`.
    /// java: `EnumFacing.getFront` — `VALUES[abs_int(index % 6)]`
    #[inline]
    #[must_use]
    pub fn from_index(index: i32) -> Facing {
        Facing::VALUES[abs_i32(index.wrapping_rem(6)) as usize % 6]
    }

    /// Horizontal ordinal, `0..=3`, S-W-N-E. java: `EnumFacing.getHorizontalIndex`
    ///
    /// Returns `None` for [`Facing::Up`] and [`Facing::Down`], which vanilla stores as `-1`.
    #[inline]
    #[must_use]
    pub const fn horizontal_index(self) -> Option<u8> {
        match self {
            Facing::South => Some(0),
            Facing::West => Some(1),
            Facing::North => Some(2),
            Facing::East => Some(3),
            Facing::Down | Facing::Up => None,
        }
    }

    /// Facing from a horizontal ordinal, wrapping like Java's `getHorizontal`.
    /// java: `EnumFacing.getHorizontal`
    #[inline]
    #[must_use]
    pub fn from_horizontal(index: i32) -> Facing {
        Facing::HORIZONTALS[abs_i32(index.wrapping_rem(4)) as usize % 4]
    }

    /// The opposite facing. java: `EnumFacing.getOpposite`
    #[inline]
    #[must_use]
    pub const fn opposite(self) -> Facing {
        match self {
            Facing::Down => Facing::Up,
            Facing::Up => Facing::Down,
            Facing::North => Facing::South,
            Facing::South => Facing::North,
            Facing::West => Facing::East,
            Facing::East => Facing::West,
        }
    }

    /// java: `EnumFacing.getAxis`
    #[inline]
    #[must_use]
    pub const fn axis(self) -> Axis {
        match self {
            Facing::Down | Facing::Up => Axis::Y,
            Facing::North | Facing::South => Axis::Z,
            Facing::West | Facing::East => Axis::X,
        }
    }

    /// The unit offset `(dx, dy, dz)` for this facing.
    /// java: `EnumFacing.getFrontOffsetX/Y/Z`, `getDirectionVec`
    #[inline]
    #[must_use]
    pub const fn offset(self) -> (i32, i32, i32) {
        match self {
            Facing::Down => (0, -1, 0),
            Facing::Up => (0, 1, 0),
            Facing::North => (0, 0, -1),
            Facing::South => (0, 0, 1),
            Facing::West => (-1, 0, 0),
            Facing::East => (1, 0, 0),
        }
    }

    /// The unit offset as a [`BlockPos`]. java: `EnumFacing.getDirectionVec`
    #[inline]
    #[must_use]
    pub const fn direction_vec(self) -> BlockPos {
        let (x, y, z) = self.offset();
        BlockPos { x, y, z }
    }

    /// java: `EnumFacing.getFrontOffsetX`
    #[inline]
    #[must_use]
    pub const fn front_offset_x(self) -> i32 {
        self.offset().0
    }

    /// java: `EnumFacing.getFrontOffsetY`
    #[inline]
    #[must_use]
    pub const fn front_offset_y(self) -> i32 {
        self.offset().1
    }

    /// java: `EnumFacing.getFrontOffsetZ`
    #[inline]
    #[must_use]
    pub const fn front_offset_z(self) -> i32 {
        self.offset().2
    }

    /// Rotate clockwise about +Y (N -> E -> S -> W). Vertical facings are returned unchanged
    /// by [`Facing::rotate_around`]; calling this directly on one is a programming error.
    /// java: `EnumFacing.rotateY`
    #[inline]
    #[must_use]
    pub const fn rotate_y(self) -> Facing {
        match self {
            Facing::North => Facing::East,
            Facing::East => Facing::South,
            Facing::South => Facing::West,
            Facing::West => Facing::North,
            Facing::Up | Facing::Down => {
                panic!("cannot rotate a vertical facing directly around Y")
            }
        }
    }

    /// Rotate counter-clockwise about +Y (N -> W -> S -> E). java: `EnumFacing.rotateYCCW`
    #[inline]
    #[must_use]
    pub const fn rotate_y_ccw(self) -> Facing {
        match self {
            Facing::North => Facing::West,
            Facing::West => Facing::South,
            Facing::South => Facing::East,
            Facing::East => Facing::North,
            Facing::Up | Facing::Down => {
                panic!("cannot rotate a vertical facing directly around Y")
            }
        }
    }

    /// Rotate about the given axis clockwise, returning `self` when the facing lies on that
    /// axis. java: `EnumFacing.rotateAround`
    #[inline]
    #[must_use]
    pub const fn rotate_around(self, axis: Axis) -> Facing {
        match axis {
            Axis::X => match self {
                Facing::North => Facing::Down,
                Facing::South => Facing::Up,
                Facing::Up => Facing::North,
                Facing::Down => Facing::South,
                other => other, // WEST/EAST
            },
            Axis::Y => match self {
                Facing::Up | Facing::Down => self,
                other => other.rotate_y(),
            },
            Axis::Z => match self {
                Facing::East => Facing::Down,
                Facing::West => Facing::Up,
                Facing::Up => Facing::East,
                Facing::Down => Facing::West,
                other => other, // NORTH/SOUTH
            },
        }
    }

    /// Horizontal facing for a yaw in degrees. `0` is South, `90` is West.
    /// java: `EnumFacing.fromAngle` — `getHorizontal(floor_double(angle / 90 + 0.5) & 3)`
    #[inline]
    #[must_use]
    pub fn from_angle(angle: f64) -> Facing {
        Facing::from_horizontal(floor_f64(angle / 90.0 + 0.5) & 3)
    }

    /// The facing whose direction vector has the largest dot product with `(x, y, z)`.
    /// java: `EnumFacing.getFacingFromVector`
    ///
    /// Faithfully reproduces vanilla's two oddities: the scan starts at `NORTH` with
    /// `Float.MIN_VALUE` (`f32::from_bits(1)`, the smallest positive *subnormal*, deliberately
    /// distinct from `f32::MIN_POSITIVE`, the smallest positive normal), and it iterates in
    /// ordinal order so ties resolve to the earlier facing.
    #[must_use]
    pub fn from_vector(x: f32, y: f32, z: f32) -> Facing {
        let mut best = Facing::North;
        // java: `float f = Float.MIN_VALUE;` — the smallest positive subnormal, not -inf.
        let mut f = f32::from_bits(1);
        for facing in Facing::VALUES {
            let (dx, dy, dz) = facing.offset();
            let f1 = x * dx as f32 + y * dy as f32 + z * dz as f32;
            if f1 > f {
                f = f1;
                best = facing;
            }
        }
        best
    }

    /// Lowercase serialised name. java: `EnumFacing.getName`
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Facing::Down => "down",
            Facing::Up => "up",
            Facing::North => "north",
            Facing::South => "south",
            Facing::West => "west",
            Facing::East => "east",
        }
    }

    /// Parse a facing name, case-insensitively. java: `EnumFacing.byName`
    ///
    /// # Errors
    /// [`MathError::UnknownFacing`] if the name is not one of the six.
    pub fn by_name(name: &str) -> Result<Facing, MathError> {
        match name.to_ascii_lowercase().as_str() {
            "down" => Ok(Facing::Down),
            "up" => Ok(Facing::Up),
            "north" => Ok(Facing::North),
            "south" => Ok(Facing::South),
            "west" => Ok(Facing::West),
            "east" => Ok(Facing::East),
            _ => Err(MathError::UnknownFacing(name.to_owned())),
        }
    }
}

impl std::fmt::Display for Facing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::fmt::Display for Axis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinals_are_wire_order() {
        assert_eq!(Facing::Down.index(), 0);
        assert_eq!(Facing::Up.index(), 1);
        assert_eq!(Facing::North.index(), 2);
        assert_eq!(Facing::South.index(), 3);
        assert_eq!(Facing::West.index(), 4);
        assert_eq!(Facing::East.index(), 5);
        for (i, f) in Facing::VALUES.iter().enumerate() {
            assert_eq!(Facing::from_index(i as i32), *f);
            assert_eq!(usize::from(f.index()), i);
        }
    }

    #[test]
    fn opposites_and_offsets_agree() {
        for f in Facing::VALUES {
            assert_eq!(f.opposite().opposite(), f);
            let (x, y, z) = f.offset();
            let (ox, oy, oz) = f.opposite().offset();
            assert_eq!((x + ox, y + oy, z + oz), (0, 0, 0));
            assert_eq!(f.axis(), f.opposite().axis());
        }
    }

    #[test]
    fn horizontal_order_is_s_w_n_e() {
        assert_eq!(Facing::HORIZONTALS[0], Facing::South);
        assert_eq!(Facing::HORIZONTALS[1], Facing::West);
        assert_eq!(Facing::HORIZONTALS[2], Facing::North);
        assert_eq!(Facing::HORIZONTALS[3], Facing::East);
        assert_eq!(Facing::from_angle(0.0), Facing::South);
        assert_eq!(Facing::from_angle(90.0), Facing::West);
        assert_eq!(Facing::from_angle(180.0), Facing::North);
        assert_eq!(Facing::from_angle(270.0), Facing::East);
    }

    #[test]
    fn rotation_cycles() {
        assert_eq!(Facing::North.rotate_y(), Facing::East);
        assert_eq!(Facing::North.rotate_y_ccw(), Facing::West);
        assert_eq!(Facing::Up.rotate_around(Axis::Y), Facing::Up);
        assert_eq!(Facing::East.rotate_around(Axis::X), Facing::East);
        assert_eq!(Facing::North.rotate_around(Axis::X), Facing::Down);
    }

    #[test]
    fn direct_y_rotation_rejects_vertical_facings_but_rotate_around_guards_them() {
        for facing in [Facing::Down, Facing::Up] {
            assert!(std::panic::catch_unwind(|| facing.rotate_y()).is_err());
            assert!(std::panic::catch_unwind(|| facing.rotate_y_ccw()).is_err());
            assert_eq!(facing.rotate_around(Axis::Y), facing);
        }
    }

    #[test]
    fn from_vector_matches_vanilla_seed_order_and_float_rules() {
        // All-zero input never beats Float.MIN_VALUE, so vanilla falls through to NORTH.
        assert_eq!(Facing::from_vector(0.0, 0.0, 0.0), Facing::North);
        // Two subnormal units beat Java's Float.MIN_VALUE seed. Replacing it with Rust's
        // MIN_POSITIVE (the smallest normal) would incorrectly leave this at NORTH.
        assert_eq!(
            Facing::from_vector(f32::from_bits(2), 0.0, 0.0),
            Facing::East
        );
        assert_eq!(Facing::from_vector(1.0, 0.0, 0.0), Facing::East);
        assert_eq!(Facing::from_vector(0.0, -1.0, 0.0), Facing::Down);

        // The scan is D-U-N-S-W-E and replaces only on a *strict* improvement.  A `>=`
        // mutant, or a scan in horizontal-index order, would choose EAST here rather than UP.
        assert_eq!(Facing::from_vector(1.0, 1.0, 0.0), Facing::Up);

        // Java evaluates every `component * direction-component` as f32.  Thus even a NaN
        // multiplied by a zero direction component poisons every candidate; `fcmpl` then
        // rejects each comparison and preserves the NORTH initial value. Treating NaN as zero
        // (or using an `is_nan` fallback) incorrectly chooses EAST.
        assert_eq!(Facing::from_vector(f32::NAN, 0.0, 0.0), Facing::North);

        // Infinity is different: it only poisons candidates with a zero X component, while the
        // EAST dot product remains +Infinity and wins. This distinguishes the JVM arithmetic
        // rule from a blanket non-finite fallback.
        assert_eq!(Facing::from_vector(f32::INFINITY, 0.0, 0.0), Facing::East);
    }

    #[test]
    fn names_round_trip() {
        for f in Facing::VALUES {
            assert_eq!(Facing::by_name(f.name()).unwrap(), f);
        }
        assert!(Facing::by_name("sideways").is_err());
    }
}
