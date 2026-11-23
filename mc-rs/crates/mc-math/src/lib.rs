#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Vectors, AABBs, and arithmetic that matches Java **bit for bit**.
//!
//! This is the quiet foundation of movement parity:
//! `mc-protocol`, `mc-world`, `mc-render` and above all `mc-physics` compute on top of it, and
//! a one-ulp difference here becomes visible desync and anticheat kicks a few hundred ticks
//! later.
//!
//! # What "bit-exact" means here
//!
//! Hazard **H1**: [`sin`] and [`cos`] are a **65,536-entry `f32` lookup table**, not libm. The
//! table is what a JVM builds in `MathHelper`'s class initialiser, embedded verbatim rather
//! than recomputed, so it cannot drift with the host's `sin` implementation. [`floor_f64`],
//! [`ceil_f64`] and friends reproduce Java's truncate-then-correct form including its overflow
//! wrap. [`JavaRandom`] is the real 48-bit LCG. [`strict_log`] is fdlibm, because
//! `Random.nextGaussian` calls `StrictMath.log`.
//!
//! Everything is checked against ~1.36 million reference values dumped from a real JVM;
//! see `tests/differential.rs` and `javaref/`.
//!
//! # Layout
//!
//! | module | java source |
//! |---|---|
//! | [`math_helper`] | `util/MathHelper.java` |
//! | [`vec`] | `util/Vec3.java` |
//! | [`pos`] | `util/BlockPos.java`, `util/Vec3i.java`, `world/ChunkCoordIntPair.java` |
//! | [`aabb`] | `util/AxisAlignedBB.java` |
//! | [`facing`] | `util/EnumFacing.java` |
//! | [`random`] | `java.util.Random` |
//!
//! # Example
//!
//! ```
//! use mc_math::{Aabb, BlockPos, Facing, floor_f64, sin};
//!
//! // Vanilla floors, it does not truncate.
//! assert_eq!(floor_f64(-0.5), -1);
//!
//! // The block a player at y = 64.9 is standing in.
//! assert_eq!(BlockPos::from_f64(0.5, 64.9, -0.5), BlockPos::new(0, 64, -1));
//!
//! // A block's collision box stops a mover 1.0 short of it.
//! let block = Aabb::from_bounds(1.0, 0.0, 0.0, 2.0, 1.0, 1.0);
//! let player = Aabb::from_bounds(-1.0, 0.0, 0.0, 0.0, 1.0, 1.0);
//! assert_eq!(block.calculate_x_offset(&player, 2.0), 1.0);
//!
//! // Facing ordinals are the wire values.
//! assert_eq!(Facing::North.index(), 2);
//!
//! // ...and sin is the sampled table, so it disagrees with libm almost everywhere.
//! assert_ne!(sin(1.0).to_bits(), 1.0_f32.sin().to_bits());
//! ```

use thiserror::Error;

pub mod aabb;
pub mod facing;
pub mod math_helper;
pub mod pos;
pub mod random;
pub mod vec;

pub use aabb::Aabb;
pub use facing::{Axis, Facing, Plane};
pub use math_helper::{
    abs_f32, abs_i32, abs_max, atan2, calculate_log_base_two, ceil_f32, ceil_f64, clamp_f32,
    clamp_f64, clamp_i32, coordinate_random, cos, fast_inv_sqrt, floor_f32, floor_f64,
    floor_f64_i64, fold_sign_to_i32, frac_f64, hsv_to_rgb, java_max_f64, java_min_f64,
    multiply_color, pack_color, pack_color_f32, round_up, round_up_to_power_of_two, sin, sqrt_d,
    sqrt_f, strict_log, truncate_f64_to_i32, wrap_degrees_f32, wrap_degrees_f64, SQRT_2,
};
pub use pos::{BlockPos, ChunkPos, ChunkSectionPos};
pub use random::JavaRandom;
pub use vec::vec3;

/// `f64` 3-vector — the type all physics and collision maths uses (hazard H7).
///
/// Re-exported from `glam`. It is a plain struct of three `f64`s with no SIMD lowering, so its
/// arithmetic is IEEE-predictable; the *derived* operations vanilla defines differently live in
/// [`vec`] rather than being taken from `glam`.
pub use glam::DVec3;

/// `f32` 3-vector, for render-side data only.
///
/// Never use this for anything that feeds movement: `glam`'s `f32` vectors may take SIMD paths
/// whose rounding differs, and vanilla's physics is `f64` throughout.
pub use glam::Vec3;

/// Errors returned by `mc-math`.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MathError {
    /// A string that is not one of the six [`Facing`] names.
    #[error("mc-math: `{0}` is not a facing name (expected one of down/up/north/south/west/east)")]
    UnknownFacing(String),
    /// A string that is not one of the three [`Axis`] names.
    #[error("mc-math: `{0}` is not an axis name (expected one of x/y/z)")]
    UnknownAxis(String),
}
