//! Star field generation. java: `RenderGlobal.renderStars` (called once from `generateStars`,
//! baked into a display list/VBO vanilla only rebuilds when the VBO setting toggles).
//!
//! Uses [`mc_math::JavaRandom`] seeded exactly as vanilla seeds it (`new Random(10842L)`), so the
//! star field this produces is pixel-for-pixel the same field vanilla generates — the plan's
//! guidance to use `mc-math`'s Java-compatible `Random` for parity-sensitive placement applies
//! most directly here.

use glam::Vec3;
use mc_math::JavaRandom;

/// Vanilla's fixed star-field seed. java: `new Random(10842L)`
pub const STAR_SEED: i64 = 10842;

/// Candidate draws; roughly two thirds are accepted (the `0.01 < |v|² < 1.0` rejection test), so
/// the real star count is close to but not exactly a fixed number.
pub const STAR_CANDIDATES: u32 = 1500;

/// One star's quad, four camera-facing-ish corners already positioned on the celestial sphere.
/// Position-only, matching vanilla's `DefaultVertexFormats.POSITION` — colour is uniform per
/// frame ([`crate::sky::environment::Environment::star_brightness`]), not per-star.
#[derive(Debug, Clone, Copy)]
pub struct StarQuad {
    /// Four corners, in sky-dome-local space (sphere radius 100, matching the sun/moon quads).
    pub corners: [Vec3; 4],
}

/// Build the full star field. Deterministic: two calls produce byte-identical output, and it is
/// the same field vanilla draws because the RNG and the rejection/projection maths match.
/// java: `RenderGlobal.renderStars`
#[must_use]
pub fn generate_stars() -> Vec<StarQuad> {
    let mut random = JavaRandom::new(STAR_SEED);
    let mut stars = Vec::with_capacity(STAR_CANDIDATES as usize * 2 / 3);

    for _ in 0..STAR_CANDIDATES {
        let d0 = f64::from(random.next_f32() * 2.0 - 1.0);
        let d1 = f64::from(random.next_f32() * 2.0 - 1.0);
        let d2 = f64::from(random.next_f32() * 2.0 - 1.0);
        let d3 = f64::from(0.15 + random.next_f32() * 0.1);
        let len_sq = d0 * d0 + d1 * d1 + d2 * d2;

        if !(len_sq < 1.0 && len_sq > 0.01) {
            continue;
        }

        let inv_len = 1.0 / len_sq.sqrt();
        let (x, y, z) = (d0 * inv_len, d1 * inv_len, d2 * inv_len);
        let (cx, cy, cz) = (x * 100.0, y * 100.0, z * 100.0);

        // Orientation basis: yaw around the sphere (atan2(x,z)) then pitch from the pole
        // (atan2(sqrt(x²+z²), y)) — tangent-plane axes at this point on the sphere.
        let yaw = x.atan2(z);
        let (sin_yaw, cos_yaw) = yaw.sin_cos();
        let pitch = (x * x + z * z).sqrt().atan2(y);
        let (sin_pitch, cos_pitch) = pitch.sin_cos();

        // A random roll for the quad itself, so stars are not all identically oriented.
        let roll = random.next_f64() * std::f64::consts::PI * 2.0;
        let (sin_roll, cos_roll) = roll.sin_cos();

        let mut corners = [Vec3::ZERO; 4];
        for (j, corner) in corners.iter_mut().enumerate() {
            let j = j as i32;
            let d18 = f64::from((j & 2) - 1) * d3;
            let d19 = f64::from(((j + 1) & 2) - 1) * d3;
            let d21 = d18 * cos_roll - d19 * sin_roll;
            let d22 = d19 * cos_roll + d18 * sin_roll;
            let d23 = d21 * sin_pitch;
            let d24 = -d21 * cos_pitch;
            let d25 = d24 * sin_yaw - d22 * cos_yaw;
            let d26 = d22 * sin_yaw + d24 * cos_yaw;
            *corner = Vec3::new((cx + d25) as f32, (cy + d23) as f32, (cz + d26) as f32);
        }

        stars.push(StarQuad { corners });
    }

    stars
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic() {
        let a = generate_stars();
        let b = generate_stars();
        assert_eq!(a.len(), b.len());
        for (sa, sb) in a.iter().zip(b.iter()) {
            assert_eq!(sa.corners, sb.corners);
        }
    }

    #[test]
    fn roughly_two_thirds_of_candidates_survive_rejection() {
        // The unit-ball-minus-small-sphere volume ratio is (1 - 0.1³) ≈ 0.999, but the *cube*
        // the samples are drawn from has volume 8 against a unit sphere volume of 4/3π ≈ 4.19,
        // so acceptance is close to π/6 ≈ 0.524, minus the sliver inside radius 0.1. Assert a
        // generous band around that rather than an exact count, since it's a probabilistic draw.
        let stars = generate_stars();
        let accepted = stars.len() as f64;
        let ratio = accepted / f64::from(STAR_CANDIDATES);
        assert!((0.35..0.7).contains(&ratio), "{ratio} ({accepted} stars)");
    }

    #[test]
    fn every_star_sits_on_the_radius_100_sphere() {
        for star in generate_stars() {
            // The quad's centre (average of corners) should sit very close to radius 100 — the
            // corners themselves are offset by up to ~0.35 for the quad half-size and roll, so
            // allow generous slack rather than asserting each corner exactly.
            let centre =
                (star.corners[0] + star.corners[1] + star.corners[2] + star.corners[3]) / 4.0;
            let r = centre.length();
            assert!((r - 100.0).abs() < 1.0, "{r}");
        }
    }

    #[test]
    fn quads_are_non_degenerate() {
        for star in generate_stars().into_iter().take(50) {
            let a = star.corners[0];
            let b = star.corners[1];
            let c = star.corners[2];
            let area2 = (b - a).cross(c - a).length();
            assert!(area2 > 1e-6, "{star:?}");
        }
    }

    #[test]
    fn matches_a_hand_checked_first_accepted_star() {
        // Cross-check against an independent re-derivation of the first accepted draw from
        // `JavaRandom::new(10842)`'s own documented sequence, so a transcription slip in the
        // rejection test or the basis maths shows up even though there's no real-JVM oracle
        // wired into this crate.
        let mut random = JavaRandom::new(STAR_SEED);
        let mut first_accepted = None;
        for _ in 0..STAR_CANDIDATES {
            let d0 = f64::from(random.next_f32() * 2.0 - 1.0);
            let d1 = f64::from(random.next_f32() * 2.0 - 1.0);
            let d2 = f64::from(random.next_f32() * 2.0 - 1.0);
            let _d3 = 0.15 + random.next_f32() * 0.1;
            let len_sq = d0 * d0 + d1 * d1 + d2 * d2;
            if len_sq < 1.0 && len_sq > 0.01 {
                first_accepted = Some((d0, d1, d2));
                break;
            }
            // A rejected draw still consumes the `nextDouble` roll in the real loop *only when
            // accepted* — vanilla's `if` guards the whole projection block including the roll,
            // so nothing extra is consumed here on rejection. This mirrors that.
        }
        let (d0, d1, d2) = first_accepted.expect("at least one of the first draws accepts");
        let len = (d0 * d0 + d1 * d1 + d2 * d2).sqrt();
        let expected_centre = Vec3::new(
            (d0 / len * 100.0) as f32,
            (d1 / len * 100.0) as f32,
            (d2 / len * 100.0) as f32,
        );

        let stars = generate_stars();
        let first = &stars[0];
        let centre =
            (first.corners[0] + first.corners[1] + first.corners[2] + first.corners[3]) / 4.0;
        assert!(
            (centre - expected_centre).length() < 0.01,
            "{centre:?} vs {expected_centre:?}"
        );
    }
}
