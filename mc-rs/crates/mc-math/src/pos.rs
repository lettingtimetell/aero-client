//! Integer positions: [`BlockPos`], [`ChunkPos`], [`ChunkSectionPos`].
//!
//! java: `net/minecraft/util/BlockPos.java`, `util/Vec3i.java`,
//! `world/ChunkCoordIntPair.java`

use crate::facing::Facing;
use crate::math_helper::floor_f64;
use glam::DVec3;
use std::cmp::Ordering;

// java: BlockPos.java:11-18. `NUM_X_BITS = 1 + calculateLogBaseTwo(roundUpToPowerOfTwo(30000000))`
// resolves to 26, so the packing is 26/12/26 — the same layout as protocol-47's `Position`
// (hazard H5), and it is sign-extended on the way out.
const NUM_X_BITS: i32 = 26;
const NUM_Z_BITS: i32 = 26;
const NUM_Y_BITS: i32 = 64 - NUM_X_BITS - NUM_Z_BITS;
const Y_SHIFT: i32 = NUM_Z_BITS;
const X_SHIFT: i32 = Y_SHIFT + NUM_Y_BITS;
const X_MASK: i64 = (1i64 << NUM_X_BITS) - 1;
const Y_MASK: i64 = (1i64 << NUM_Y_BITS) - 1;
const Z_MASK: i64 = (1i64 << NUM_Z_BITS) - 1;

/// An integer block coordinate. java: `BlockPos` / `Vec3i`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BlockPos {
    /// Block X.
    pub x: i32,
    /// Block Y (`0..=255` in a loaded 1.8.9 world, but not enforced).
    pub y: i32,
    /// Block Z.
    pub z: i32,
}

// Vec3i.compareTo orders positions vertically first, then Z, then X. Do not use the field-order
// derive here: `BlockPos { x, y, z }` would silently make X the primary key.
impl Ord for BlockPos {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.y
            .cmp(&other.y)
            .then_with(|| self.z.cmp(&other.z))
            .then_with(|| self.x.cmp(&other.x))
    }
}

impl PartialOrd for BlockPos {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl BlockPos {
    /// `(0, 0, 0)`. java: `BlockPos.ORIGIN`
    pub const ORIGIN: BlockPos = BlockPos { x: 0, y: 0, z: 0 };

    /// Construct from integer coordinates.
    #[inline]
    #[must_use]
    pub const fn new(x: i32, y: i32, z: i32) -> BlockPos {
        BlockPos { x, y, z }
    }

    /// Construct by flooring, exactly as `Vec3i(double, double, double)` does.
    /// java: Vec3i.java:26 — uses `MathHelper.floor_double`, **not** `(int)`.
    #[inline]
    #[must_use]
    pub fn from_f64(x: f64, y: f64, z: f64) -> BlockPos {
        BlockPos {
            x: floor_f64(x),
            y: floor_f64(y),
            z: floor_f64(z),
        }
    }

    /// Construct by flooring a vector. java: `new BlockPos(Vec3)`
    #[inline]
    #[must_use]
    pub fn from_vec3(v: DVec3) -> BlockPos {
        BlockPos::from_f64(v.x, v.y, v.z)
    }

    /// The corner of this block as a `f64` vector.
    #[inline]
    #[must_use]
    pub fn to_vec3(self) -> DVec3 {
        DVec3::new(f64::from(self.x), f64::from(self.y), f64::from(self.z))
    }

    /// Component-wise add, wrapping. java: `BlockPos.add`
    #[inline]
    #[must_use]
    pub const fn add(self, x: i32, y: i32, z: i32) -> BlockPos {
        BlockPos {
            x: self.x.wrapping_add(x),
            y: self.y.wrapping_add(y),
            z: self.z.wrapping_add(z),
        }
    }

    /// Component-wise subtract, wrapping. java: `BlockPos.subtract`.
    ///
    /// Java implements this as `add(-other.x, -other.y, -other.z)`, where every operand is an
    /// `int`; in particular, negating `Integer.MIN_VALUE` remains `Integer.MIN_VALUE`.  A
    /// wrapping subtraction is equivalent, including that edge case.
    #[inline]
    #[must_use]
    pub const fn sub(self, other: BlockPos) -> BlockPos {
        BlockPos {
            x: self.x.wrapping_sub(other.x),
            y: self.y.wrapping_sub(other.y),
            z: self.z.wrapping_sub(other.z),
        }
    }

    /// Move `n` blocks along `facing`. java: `BlockPos.offset(EnumFacing, int)`
    #[inline]
    #[must_use]
    pub const fn offset(self, facing: Facing, n: i32) -> BlockPos {
        let (dx, dy, dz) = facing.offset();
        BlockPos {
            x: self.x.wrapping_add(dx.wrapping_mul(n)),
            y: self.y.wrapping_add(dy.wrapping_mul(n)),
            z: self.z.wrapping_add(dz.wrapping_mul(n)),
        }
    }

    /// One block along `facing`. java: `BlockPos.offset(EnumFacing)`
    #[inline]
    #[must_use]
    pub const fn relative(self, facing: Facing) -> BlockPos {
        self.offset(facing, 1)
    }

    /// One block up. java: `BlockPos.up`
    #[inline]
    #[must_use]
    pub const fn up(self) -> BlockPos {
        self.offset(Facing::Up, 1)
    }

    /// One block down. java: `BlockPos.down`
    #[inline]
    #[must_use]
    pub const fn down(self) -> BlockPos {
        self.offset(Facing::Down, 1)
    }

    /// One block north (-Z). java: `BlockPos.north`
    #[inline]
    #[must_use]
    pub const fn north(self) -> BlockPos {
        self.offset(Facing::North, 1)
    }

    /// One block south (+Z). java: `BlockPos.south`
    #[inline]
    #[must_use]
    pub const fn south(self) -> BlockPos {
        self.offset(Facing::South, 1)
    }

    /// One block west (-X). java: `BlockPos.west`
    #[inline]
    #[must_use]
    pub const fn west(self) -> BlockPos {
        self.offset(Facing::West, 1)
    }

    /// One block east (+X). java: `BlockPos.east`
    #[inline]
    #[must_use]
    pub const fn east(self) -> BlockPos {
        self.offset(Facing::East, 1)
    }

    /// Right-handed cross product, wrapping each Java `int` intermediate. java:
    /// `BlockPos.crossProduct`
    #[inline]
    #[must_use]
    pub const fn cross(self, other: BlockPos) -> BlockPos {
        BlockPos {
            x: self
                .y
                .wrapping_mul(other.z)
                .wrapping_sub(self.z.wrapping_mul(other.y)),
            y: self
                .z
                .wrapping_mul(other.x)
                .wrapping_sub(self.x.wrapping_mul(other.z)),
            z: self
                .x
                .wrapping_mul(other.y)
                .wrapping_sub(self.y.wrapping_mul(other.x)),
        }
    }

    /// Pack into the 26/12/26 `i64` layout. java: `BlockPos.toLong`
    #[inline]
    #[must_use]
    pub const fn to_long(self) -> i64 {
        ((self.x as i64 & X_MASK) << X_SHIFT)
            | ((self.y as i64 & Y_MASK) << Y_SHIFT)
            | (self.z as i64 & Z_MASK)
    }

    /// Unpack from the 26/12/26 `i64` layout, sign-extending each field.
    /// java: `BlockPos.fromLong`
    #[inline]
    #[must_use]
    pub const fn from_long(serialized: i64) -> BlockPos {
        BlockPos {
            x: ((serialized << (64 - X_SHIFT - NUM_X_BITS)) >> (64 - NUM_X_BITS)) as i32,
            y: ((serialized << (64 - Y_SHIFT - NUM_Y_BITS)) >> (64 - NUM_Y_BITS)) as i32,
            z: ((serialized << (64 - NUM_Z_BITS)) >> (64 - NUM_Z_BITS)) as i32,
        }
    }

    /// Squared distance to a point. java: `Vec3i.distanceSq`
    #[inline]
    #[must_use]
    pub fn distance_sq(self, x: f64, y: f64, z: f64) -> f64 {
        let d0 = f64::from(self.x) - x;
        let d1 = f64::from(self.y) - y;
        let d2 = f64::from(self.z) - z;
        d0 * d0 + d1 * d1 + d2 * d2
    }

    /// Squared distance from a point to this block's centre. java: `Vec3i.distanceSqToCenter`
    #[inline]
    #[must_use]
    pub fn distance_sq_to_center(self, x: f64, y: f64, z: f64) -> f64 {
        let d0 = f64::from(self.x) + 0.5 - x;
        let d1 = f64::from(self.y) + 0.5 - y;
        let d2 = f64::from(self.z) + 0.5 - z;
        d0 * d0 + d1 * d1 + d2 * d2
    }

    /// Java's `Vec3i.hashCode`, for anything that has to reproduce vanilla hash ordering.
    ///
    /// This is deliberately separate from Rust's derived [`Hash`]: it is the Java 32-bit value
    /// `(y + z * 31) * 31 + x`, with Java `int` overflow at both arithmetic stages.
    /// java: Vec3i.java:48
    #[inline]
    #[must_use]
    pub const fn java_hash_code(self) -> i32 {
        self.y
            .wrapping_add(self.z.wrapping_mul(31))
            .wrapping_mul(31)
            .wrapping_add(self.x)
    }

    /// The chunk column containing this block.
    #[inline]
    #[must_use]
    pub const fn chunk_pos(self) -> ChunkPos {
        ChunkPos {
            x: self.x >> 4,
            z: self.z >> 4,
        }
    }

    /// The 16x16x16 chunk section containing this block.
    #[inline]
    #[must_use]
    pub const fn section_pos(self) -> ChunkSectionPos {
        ChunkSectionPos {
            x: self.x >> 4,
            y: self.y >> 4,
            z: self.z >> 4,
        }
    }

    /// Every block position in the inclusive box between `from` and `to`, in vanilla's
    /// x-fastest order. java: `BlockPos.getAllInBox`
    pub fn all_in_box(from: BlockPos, to: BlockPos) -> impl Iterator<Item = BlockPos> {
        let lo = BlockPos::new(from.x.min(to.x), from.y.min(to.y), from.z.min(to.z));
        let hi = BlockPos::new(from.x.max(to.x), from.y.max(to.y), from.z.max(to.z));
        AllInBox::new(lo, hi)
    }
}

/// State for [`BlockPos::all_in_box`].
///
/// This mirrors the state machine in Java's `BlockPos.getAllInBox`: X advances first, then Y,
/// then Z.  Keeping the current value rather than using arithmetic over a flattened volume is
/// important: the latter overflows for legitimate `i32` endpoints before the iterator itself has
/// a chance to decide whether a coordinate may advance.
#[derive(Debug, Clone)]
struct AllInBox {
    lo: BlockPos,
    hi: BlockPos,
    next: Option<BlockPos>,
}

impl AllInBox {
    const fn new(lo: BlockPos, hi: BlockPos) -> Self {
        Self {
            lo,
            hi,
            next: Some(lo),
        }
    }
}

impl Iterator for AllInBox {
    type Item = BlockPos;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.next?;

        // Do the terminal comparison before any increment.  In particular, an endpoint of
        // `i32::MAX` is yielded once and never incremented, matching Java's iterator without a
        // Rust debug-build overflow panic or a wrapped extra coordinate.
        self.next = if current == self.hi {
            None
        } else if current.x < self.hi.x {
            Some(BlockPos::new(current.x + 1, current.y, current.z))
        } else if current.y < self.hi.y {
            Some(BlockPos::new(self.lo.x, current.y + 1, current.z))
        } else {
            debug_assert!(current.z < self.hi.z);
            Some(BlockPos::new(self.lo.x, self.lo.y, current.z + 1))
        };

        Some(current)
    }
}

/// A chunk column coordinate. java: `ChunkCoordIntPair`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct ChunkPos {
    /// Chunk X (block X `>> 4`).
    pub x: i32,
    /// Chunk Z (block Z `>> 4`).
    pub z: i32,
}

impl ChunkPos {
    /// Construct from chunk coordinates.
    #[inline]
    #[must_use]
    pub const fn new(x: i32, z: i32) -> ChunkPos {
        ChunkPos { x, z }
    }

    /// Pack into an `i64` key. java: `ChunkCoordIntPair.chunkXZ2Int`
    #[inline]
    #[must_use]
    pub const fn to_long(self) -> i64 {
        (self.x as i64 & 0xFFFF_FFFF) | ((self.z as i64 & 0xFFFF_FFFF) << 32)
    }

    /// X of the block at the chunk's centre. java: `ChunkCoordIntPair.getCenterXPos`
    #[inline]
    #[must_use]
    pub const fn center_x(self) -> i32 {
        (self.x << 4) + 8
    }

    /// Z of the block at the chunk's centre. java: `ChunkCoordIntPair.getCenterZPosition`
    #[inline]
    #[must_use]
    pub const fn center_z(self) -> i32 {
        (self.z << 4) + 8
    }

    /// The block at this chunk's `(0, y, 0)` corner.
    #[inline]
    #[must_use]
    pub const fn origin(self, y: i32) -> BlockPos {
        BlockPos::new(self.x << 4, y, self.z << 4)
    }

    /// Java's `ChunkCoordIntPair.hashCode`.
    ///
    /// This is the exact pair of Java LCG mixes, not a generic Rust hash.  Its two `int`
    /// multiplications and additions wrap before the final XOR. java: ChunkCoordIntPair.java:28
    #[inline]
    #[must_use]
    pub const fn java_hash_code(self) -> i32 {
        let i = 1_664_525i32
            .wrapping_mul(self.x)
            .wrapping_add(1_013_904_223);
        let j = 1_664_525i32
            .wrapping_mul(self.z ^ -559_038_737)
            .wrapping_add(1_013_904_223);
        i ^ j
    }
}

/// A 16x16x16 chunk-section coordinate. Not a vanilla type — `ExtendedBlockStorage` is
/// addressed by `(ChunkCoordIntPair, yIndex)` — but the mesher (S11) wants one value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct ChunkSectionPos {
    /// Section X (block X `>> 4`).
    pub x: i32,
    /// Section Y, `0..=15` for a 1.8.9 world.
    pub y: i32,
    /// Section Z (block Z `>> 4`).
    pub z: i32,
}

impl ChunkSectionPos {
    /// Construct from section coordinates.
    #[inline]
    #[must_use]
    pub const fn new(x: i32, y: i32, z: i32) -> ChunkSectionPos {
        ChunkSectionPos { x, y, z }
    }

    /// The chunk column this section belongs to.
    #[inline]
    #[must_use]
    pub const fn chunk_pos(self) -> ChunkPos {
        ChunkPos {
            x: self.x,
            z: self.z,
        }
    }

    /// The block at this section's minimum corner.
    #[inline]
    #[must_use]
    pub const fn origin(self) -> BlockPos {
        BlockPos::new(self.x << 4, self.y << 4, self.z << 4)
    }

    /// The block at `(x, y, z)` within this section, where each local coordinate is `0..=15`.
    #[inline]
    #[must_use]
    pub const fn block(self, x: i32, y: i32, z: i32) -> BlockPos {
        BlockPos::new((self.x << 4) + x, (self.y << 4) + y, (self.z << 4) + z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tiny reference functions use i64 for the intermediate calculation and narrow after
    // every Java `int` operation.  Keeping them separate from the production expressions makes
    // the overflow fixtures below useful mutation controls rather than a second spelling of the
    // same Rust operations.
    fn java_add(a: i32, b: i32) -> i32 {
        (i64::from(a) + i64::from(b)) as i32
    }

    fn java_sub(a: i32, b: i32) -> i32 {
        (i64::from(a) - i64::from(b)) as i32
    }

    fn java_mul(a: i32, b: i32) -> i32 {
        (i64::from(a) * i64::from(b)) as i32
    }

    fn java_neg(a: i32) -> i32 {
        (-i64::from(a)) as i32
    }

    fn java_block_pos_hash(p: BlockPos) -> i32 {
        java_add(java_mul(java_add(p.y, java_mul(p.z, 31)), 31), p.x)
    }

    fn java_chunk_pos_hash(p: ChunkPos) -> i32 {
        let i = java_add(java_mul(1_664_525, p.x), 1_013_904_223);
        let j = java_add(java_mul(1_664_525, p.z ^ -559_038_737), 1_013_904_223);
        i ^ j
    }

    #[test]
    fn packing_is_26_12_26() {
        assert_eq!(NUM_X_BITS, 26);
        assert_eq!(NUM_Y_BITS, 12);
        assert_eq!(NUM_Z_BITS, 26);
        assert_eq!(X_SHIFT, 38);
        assert_eq!(Y_SHIFT, 26);
    }

    #[test]
    fn long_round_trip_sign_extends() {
        for p in [
            BlockPos::new(0, 0, 0),
            BlockPos::new(1, 2, 3),
            BlockPos::new(-1, -1, -1),
            BlockPos::new(-3_000_000, 200, 3_000_000),
            BlockPos::new(30_000_000, 255, -30_000_000),
        ] {
            assert_eq!(BlockPos::from_long(p.to_long()), p, "round trip for {p:?}");
        }
    }

    #[test]
    fn from_f64_floors_negatives() {
        assert_eq!(
            BlockPos::from_f64(-0.5, -0.5, -0.5),
            BlockPos::new(-1, -1, -1)
        );
        assert_eq!(BlockPos::from_f64(0.9, 0.9, 0.9), BlockPos::ORIGIN);
    }

    #[test]
    fn offsets_follow_facing() {
        let p = BlockPos::new(10, 20, 30);
        assert_eq!(p.up(), BlockPos::new(10, 21, 30));
        assert_eq!(p.down(), BlockPos::new(10, 19, 30));
        assert_eq!(p.north(), BlockPos::new(10, 20, 29));
        assert_eq!(p.south(), BlockPos::new(10, 20, 31));
        assert_eq!(p.west(), BlockPos::new(9, 20, 30));
        assert_eq!(p.east(), BlockPos::new(11, 20, 30));
        assert_eq!(p.offset(Facing::East, 5), BlockPos::new(15, 20, 30));
    }

    #[test]
    fn subtract_matches_java_add_of_negated_components_at_integer_edges() {
        for (left, right) in [
            (BlockPos::new(10, -20, 30), BlockPos::new(-4, 5, 6)),
            (
                BlockPos::new(i32::MAX, i32::MIN, -1),
                BlockPos::new(-1, i32::MIN, i32::MAX),
            ),
            (
                BlockPos::new(i32::MIN, i32::MAX, i32::MIN),
                BlockPos::new(1, -1, i32::MIN),
            ),
        ] {
            let expected = BlockPos::new(
                java_add(left.x, java_neg(right.x)),
                java_add(left.y, java_neg(right.y)),
                java_add(left.z, java_neg(right.z)),
            );
            assert_eq!(left.sub(right), expected, "{left:?} - {right:?}");
        }

        // This bites a non-wrapping implementation and proves Java's surprising MIN_VALUE case.
        assert_eq!(
            BlockPos::new(i32::MIN, 0, 0).sub(BlockPos::new(1, 0, 0)),
            BlockPos::new(i32::MAX, 0, 0)
        );
    }

    #[test]
    fn cross_is_right_handed_and_wraps_each_java_int_operation() {
        let x = BlockPos::new(1, 0, 0);
        let y = BlockPos::new(0, 1, 0);
        assert_eq!(x.cross(y), BlockPos::new(0, 0, 1));
        assert_eq!(y.cross(x), BlockPos::new(0, 0, -1));

        let a = BlockPos::new(-2, 3, 5);
        let b = BlockPos::new(7, -11, 13);
        assert_eq!(a.cross(b), BlockPos::new(94, 61, 1));

        let overflow_a = BlockPos::new(i32::MAX, i32::MAX, 0);
        let overflow_b = BlockPos::new(0, i32::MAX, i32::MAX);
        let expected = BlockPos::new(
            java_sub(
                java_mul(overflow_a.y, overflow_b.z),
                java_mul(overflow_a.z, overflow_b.y),
            ),
            java_sub(
                java_mul(overflow_a.z, overflow_b.x),
                java_mul(overflow_a.x, overflow_b.z),
            ),
            java_sub(
                java_mul(overflow_a.x, overflow_b.y),
                java_mul(overflow_a.y, overflow_b.x),
            ),
        );
        assert_eq!(expected, BlockPos::new(1, -1, 1));
        assert_eq!(overflow_a.cross(overflow_b), expected);
    }

    #[test]
    fn java_position_hashes_match_independent_oracles_and_reject_order_mutants() {
        for p in [
            BlockPos::ORIGIN,
            BlockPos::new(1, 2, 3),
            BlockPos::new(-2, -3, 5),
            BlockPos::new(i32::MAX, i32::MIN, 1),
        ] {
            let actual = p.java_hash_code();
            assert_eq!(actual, java_block_pos_hash(p), "Vec3i hash for {p:?}");
        }
        for p in [
            BlockPos::new(1, 2, 3),
            BlockPos::new(-2, -3, 5),
            BlockPos::new(i32::MAX, i32::MIN, 1),
        ] {
            // A tempting x/y/z field-order hash is not vanilla's y/z/x formula.
            let actual = p.java_hash_code();
            let x_first = java_add(java_mul(java_add(java_mul(p.x, 31), p.y), 31), p.z);
            assert_ne!(
                actual, x_first,
                "fixture must kill x-first hash mutant: {p:?}"
            );
        }
        assert_eq!(BlockPos::new(1, 2, 3).java_hash_code(), 2_946);
        assert_eq!(
            BlockPos::new(i32::MAX, i32::MIN, i32::MAX).java_hash_code(),
            2_147_482_686
        );

        for p in [
            ChunkPos::new(0, 0),
            ChunkPos::new(1, -1),
            ChunkPos::new(-123_456, 654_321),
            ChunkPos::new(i32::MIN, i32::MAX),
        ] {
            let actual = p.java_hash_code();
            assert_eq!(
                actual,
                java_chunk_pos_hash(p),
                "ChunkCoordIntPair hash for {p:?}"
            );
        }
        for p in [
            ChunkPos::new(1, -1),
            ChunkPos::new(-123_456, 654_321),
            ChunkPos::new(i32::MIN, i32::MAX),
        ] {
            let actual = p.java_hash_code();
            // Swapping axes changes the Java bucket hash.
            assert_ne!(
                actual,
                java_chunk_pos_hash(ChunkPos::new(p.z, p.x)),
                "fixture must kill swapped-axis chunk hash mutant: {p:?}"
            );
        }
    }

    #[test]
    fn chunk_math() {
        assert_eq!(BlockPos::new(-1, 0, -1).chunk_pos(), ChunkPos::new(-1, -1));
        assert_eq!(BlockPos::new(16, 0, 16).chunk_pos(), ChunkPos::new(1, 1));
        assert_eq!(ChunkPos::new(2, 3).center_x(), 40);
        assert_eq!(ChunkPos::new(2, 3).center_z(), 56);
        assert_eq!(
            BlockPos::new(17, 33, 49).section_pos(),
            ChunkSectionPos::new(1, 2, 3)
        );
    }

    #[test]
    fn all_in_box_is_x_fastest() {
        let v: Vec<_> =
            BlockPos::all_in_box(BlockPos::new(0, 0, 0), BlockPos::new(1, 1, 1)).collect();
        assert_eq!(v.len(), 8);
        assert_eq!(v[0], BlockPos::new(0, 0, 0));
        assert_eq!(v[1], BlockPos::new(1, 0, 0));
        assert_eq!(v[2], BlockPos::new(0, 1, 0));
    }

    #[test]
    fn all_in_box_normalizes_corners_and_keeps_java_axis_order() {
        let actual: Vec<_> =
            BlockPos::all_in_box(BlockPos::new(2, 4, 7), BlockPos::new(1, 3, 6)).collect();

        // Java normalizes each endpoint independently, then advances x, y, z.  This full
        // fixture catches both a caller-order interpretation and every swapped-axis mutant.
        assert_eq!(
            actual,
            [
                BlockPos::new(1, 3, 6),
                BlockPos::new(2, 3, 6),
                BlockPos::new(1, 4, 6),
                BlockPos::new(2, 4, 6),
                BlockPos::new(1, 3, 7),
                BlockPos::new(2, 3, 7),
                BlockPos::new(1, 4, 7),
                BlockPos::new(2, 4, 7),
            ]
        );
    }

    #[test]
    fn all_in_box_handles_max_endpoint_without_wrapping() {
        let mut positions = BlockPos::all_in_box(
            BlockPos::new(i32::MAX, -3, 8),
            BlockPos::new(i32::MAX - 1, -3, 8),
        );
        let first = positions.next().expect("inclusive lower endpoint");
        let second = positions.next().expect("inclusive upper endpoint");

        assert_eq!(first, BlockPos::new(i32::MAX - 1, -3, 8));
        assert_eq!(second, BlockPos::new(i32::MAX, -3, 8));
        assert_eq!(positions.next(), None, "must not wrap past i32::MAX");
    }

    #[test]
    fn all_in_box_yields_independent_immutable_snapshots() {
        let mut positions =
            BlockPos::all_in_box(BlockPos::new(10, 20, 30), BlockPos::new(11, 20, 30));
        let first = positions.next().unwrap();
        let second = positions.next().unwrap();

        // Unlike the later mutable iterator API, 1.8.9's getAllInBox yields ordinary immutable
        // BlockPos values.  Advancing the iterator cannot retroactively change an earlier value.
        assert_eq!(first, BlockPos::new(10, 20, 30));
        assert_eq!(second, BlockPos::new(11, 20, 30));
    }

    #[test]
    fn ordering_matches_vec3i_y_then_z_then_x() {
        let mut positions = [
            BlockPos::new(-100, 1, -100),
            BlockPos::new(10, 0, 1),
            BlockPos::new(1, 0, 0),
            BlockPos::new(0, 0, 0),
        ];
        positions.sort();
        assert_eq!(
            positions,
            [
                BlockPos::new(0, 0, 0),
                BlockPos::new(1, 0, 0),
                BlockPos::new(10, 0, 1),
                BlockPos::new(-100, 1, -100),
            ]
        );

        // The old field-order derive says Greater here because it compares X first.
        assert_eq!(
            BlockPos::new(1, 0, 0).cmp(&BlockPos::new(0, 1, 0)),
            Ordering::Less
        );
    }
}
