//! `java.util.Random`, reproduced exactly.
//!
//! Particles (S18), block-break dust, sound-pitch jitter and every other client-side effect
//! that vanilla seeds from a shared `Random` must draw the same sequence, or the client looks
//! visibly different from vanilla even when the world state agrees. This is the real 48-bit
//! linear congruential generator with Java's exact `nextInt(bound)` rejection loop, its
//! two-draw `nextDouble`, and the polar-method `nextGaussian` with its cached second value.
//!
//! `tests/differential.rs` checks every method against sequences dumped from a real JVM.

use crate::math_helper::strict_log;

const MULTIPLIER: i64 = 0x5_DEEC_E66D;
const ADDEND: i64 = 0xB;
const MASK: i64 = (1 << 48) - 1;
// java.util.Random.DOUBLE_UNIT = 0x1.0p-53
const DOUBLE_UNIT: f64 = f64::from_bits(0x3CA0_0000_0000_0000);

/// A bit-exact `java.util.Random`.
///
/// Not thread-safe and deliberately not synchronised — vanilla's `synchronized` on
/// `nextGaussian` is irrelevant when the generator is owned by one system.
#[derive(Debug, Clone, PartialEq)]
pub struct JavaRandom {
    seed: i64,
    next_next_gaussian: f64,
    have_next_next_gaussian: bool,
}

impl JavaRandom {
    /// Seed the generator. java: `Random(long)` — the seed is scrambled on construction.
    #[inline]
    #[must_use]
    pub fn new(seed: i64) -> JavaRandom {
        JavaRandom {
            seed: (seed ^ MULTIPLIER) & MASK,
            next_next_gaussian: 0.0,
            have_next_next_gaussian: false,
        }
    }

    /// Re-seed, discarding any cached gaussian. java: `Random.setSeed`
    #[inline]
    pub fn set_seed(&mut self, seed: i64) {
        self.seed = (seed ^ MULTIPLIER) & MASK;
        self.have_next_next_gaussian = false;
        self.next_next_gaussian = 0.0;
    }

    /// The raw 48-bit scrambled seed, for callers that must persist generator state.
    #[inline]
    #[must_use]
    pub const fn raw_seed(&self) -> i64 {
        self.seed
    }

    /// Advance the LCG and take the top `bits`. java: `Random.next(int)`
    #[inline]
    fn next(&mut self, bits: u32) -> i32 {
        self.seed = self.seed.wrapping_mul(MULTIPLIER).wrapping_add(ADDEND) & MASK;
        // The seed is masked to 48 bits, so the logical shift matches Java's `>>>`.
        ((self.seed as u64) >> (48 - bits)) as i32
    }

    /// A uniform `i32` over the whole range. java: `Random.nextInt()`
    #[inline]
    pub fn next_i32(&mut self) -> i32 {
        self.next(32)
    }

    /// A uniform `i32` in `0..bound`. java: `Random.nextInt(int)`
    ///
    /// Reproduces the power-of-two fast path *and* the rejection loop, both of which consume a
    /// different number of LCG steps than a naive modulo would.
    ///
    /// # Panics
    /// If `bound` is not positive, as Java throws `IllegalArgumentException`.
    pub fn next_i32_bound(&mut self, bound: i32) -> i32 {
        assert!(bound > 0, "bound must be positive, got {bound}");
        let mut r = self.next(31);
        let m = bound - 1;
        if (bound & m) == 0 {
            // bound is a power of two
            r = ((i64::from(bound).wrapping_mul(i64::from(r))) >> 31) as i32;
        } else {
            // java: `for (int u = r; u - (r = u % bound) + m < 0; u = next(31));`
            let mut u = r;
            loop {
                r = u % bound;
                if u.wrapping_sub(r).wrapping_add(m) >= 0 {
                    break;
                }
                u = self.next(31);
            }
        }
        r
    }

    /// A uniform `i64`, built from two 32-bit draws. java: `Random.nextLong()`
    #[inline]
    pub fn next_i64(&mut self) -> i64 {
        let hi = i64::from(self.next(32));
        let lo = i64::from(self.next(32));
        (hi << 32).wrapping_add(lo)
    }

    /// A uniform `bool` from a single bit. java: `Random.nextBoolean()`
    #[inline]
    pub fn next_bool(&mut self) -> bool {
        self.next(1) != 0
    }

    /// A uniform `f32` in `[0, 1)`, from a 24-bit draw. java: `Random.nextFloat()`
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        self.next(24) as f32 / (1 << 24) as f32
    }

    /// A uniform `f64` in `[0, 1)`, from a 26-bit and a 27-bit draw.
    /// java: `Random.nextDouble()`
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let hi = i64::from(self.next(26)) << 27;
        let lo = i64::from(self.next(27));
        (hi.wrapping_add(lo)) as f64 * DOUBLE_UNIT
    }

    /// A standard-normal `f64` by the Marsaglia polar method, with the second value cached
    /// for the next call. java: `Random.nextGaussian()`
    ///
    /// Uses [`strict_log`] rather than `f64::ln`, because Java calls `StrictMath.log` here and
    /// the platform libm is not bit-identical to fdlibm.
    pub fn next_gaussian(&mut self) -> f64 {
        if self.have_next_next_gaussian {
            self.have_next_next_gaussian = false;
            return self.next_next_gaussian;
        }
        loop {
            let v1 = 2.0 * self.next_f64() - 1.0;
            let v2 = 2.0 * self.next_f64() - 1.0;
            let s = v1 * v1 + v2 * v2;
            if s < 1.0 && s != 0.0 {
                let multiplier = (-2.0 * strict_log(s) / s).sqrt();
                self.next_next_gaussian = v2 * multiplier;
                self.have_next_next_gaussian = true;
                return v1 * multiplier;
            }
        }
    }

    /// `MathHelper.getRandomIntegerInRange` — inclusive on both ends.
    /// java: MathHelper.java:172
    #[inline]
    pub fn range_i32(&mut self, min: i32, max: i32) -> i32 {
        if min >= max {
            min
        } else {
            self.next_i32_bound(max.wrapping_sub(min).wrapping_add(1))
                .wrapping_add(min)
        }
    }

    /// `MathHelper.randomFloatClamp`. java: MathHelper.java:177
    #[inline]
    pub fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        if min >= max {
            min
        } else {
            self.next_f32() * (max - min) + min
        }
    }

    /// `MathHelper.getRandomDoubleInRange`. java: MathHelper.java:182
    #[inline]
    pub fn range_f64(&mut self, min: f64, max: f64) -> f64 {
        if min >= max {
            min
        } else {
            self.next_f64() * (max - min) + min
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_java_sequences() {
        // `new Random(0)` in any JVM produces these; they are the canonical smoke test.
        let mut r = JavaRandom::new(0);
        assert_eq!(r.next_i32(), -1_155_484_576);
        assert_eq!(r.next_i32(), -723_955_400);
        assert_eq!(r.next_i32(), 1_033_096_058);

        let mut r = JavaRandom::new(42);
        assert_eq!(r.next_i32_bound(100), 30);
        assert_eq!(r.next_i32_bound(100), 63);

        let mut r = JavaRandom::new(0);
        assert_eq!(r.next_i64(), -4_962_768_465_676_381_896);

        let mut r = JavaRandom::new(0);
        assert_eq!(r.next_f64().to_bits(), 0x3fe7_6416_8ea6_ca89);

        let mut r = JavaRandom::new(0);
        assert_eq!(
            r.next_gaussian().to_bits(),
            0.802_533_063_739_030_5_f64.to_bits()
        );
    }

    #[test]
    fn seed_is_scrambled_on_construction() {
        assert_eq!(JavaRandom::new(0).raw_seed(), 0x5_DEEC_E66D);
    }

    #[test]
    fn gaussian_caches_its_partner() {
        let mut a = JavaRandom::new(7);
        let first = a.next_gaussian();
        let second = a.next_gaussian();
        // The second value comes from the cache, so no LCG step happened between them.
        let mut b = JavaRandom::new(7);
        assert_eq!(b.next_gaussian().to_bits(), first.to_bits());
        assert_eq!(b.next_gaussian().to_bits(), second.to_bits());
        assert_ne!(first.to_bits(), second.to_bits());
    }

    #[test]
    fn bounded_draws_stay_in_range() {
        let mut r = JavaRandom::new(-1);
        for _ in 0..10_000 {
            let v = r.next_i32_bound(97);
            assert!((0..97).contains(&v));
        }
    }
}
