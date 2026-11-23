//! Back-to-front sorting of the translucent layer.
//!
//! Alpha blending is order-dependent, so water seen through water only looks right if the far
//! quad is drawn first. Vanilla re-sorts a section's translucent quads by the squared distance
//! from the camera to each quad's **centroid**, descending, and does it again whenever the
//! camera moves more than one block away from the last sort position
//! (`RenderGlobal.renderBlockLayer`'s squared-Euclidean-distance check).
//
// java: client/renderer/WorldRenderer.java sortVertexData/getDistanceSq,
//       client/renderer/chunk/RenderChunk.java resortTransparency,
//       client/renderer/RenderGlobal.java renderBlockLayer

use mc_math::{ChunkSectionPos, DVec3};

use super::vertex::Buffer;

/// Squared distance from `eye` to the centroid of quad `i`, in the same section-local space
/// the buffer stores. java: `WorldRenderer.getDistanceSq`.
///
/// Since T1.1 the stored position is a `Unorm16` code, so the centroid is formed from the
/// *decoded* coordinates. The decode is affine and identical for all four vertices, so the
/// ordering this produces is the ordering the 28-byte format produced except where two quads'
/// centroids were within one quantum (1/2048 block) of each other — a distinction alpha
/// blending cannot see.
#[must_use]
pub fn quad_distance_sq(buf: &Buffer, i: usize, eye: [f32; 3]) -> f32 {
    let q = buf.quad(i);
    let p = [
        q[0].position(),
        q[1].position(),
        q[2].position(),
        q[3].position(),
    ];
    let cx = (p[0][0] + p[1][0] + p[2][0] + p[3][0]) * 0.25 - eye[0];
    let cy = (p[0][1] + p[1][1] + p[2][1] + p[3][1]) * 0.25 - eye[1];
    let cz = (p[0][2] + p[1][2] + p[2][2] + p[3][2]) * 0.25 - eye[2];
    cx * cx + cy * cy + cz * cz
}

/// Sort a section's translucent quads back to front for an eye at `camera` (world space).
///
/// `section` is the section the buffer belongs to, so the world-space camera can be brought
/// into the buffer's section-local space — vanilla does the equivalent by tracking the
/// `WorldRenderer`'s translation offset.
pub fn sort_translucent(buf: &mut Buffer, section: ChunkSectionPos, camera: DVec3) {
    let origin = section.origin();
    let eye = [
        (camera.x - f64::from(origin.x)) as f32,
        (camera.y - f64::from(origin.y)) as f32,
        (camera.z - f64::from(origin.z)) as f32,
    ];
    sort_translucent_local(buf, eye);
}

/// [`sort_translucent`] with the eye already in section-local space.
pub fn sort_translucent_local(buf: &mut Buffer, eye: [f32; 3]) {
    let n = buf.quad_count();
    if n < 2 {
        return;
    }
    let mut order: Vec<(f32, usize)> = (0..n).map(|i| (quad_distance_sq(buf, i, eye), i)).collect();
    // Descending: farthest first. `Floats.compare(afloat[b], afloat[a])` in the Java.
    order.sort_by(|a, b| b.0.total_cmp(&a.0));

    let src = buf.vertices().to_vec();
    let dst = buf.vertices_mut();
    for (slot, &(_, from)) in order.iter().enumerate() {
        dst[slot * 4..slot * 4 + 4].copy_from_slice(&src[from * 4..from * 4 + 4]);
    }
}

/// Whether the camera has moved far enough since `last` that the translucent layers need
/// re-sorting. java: `RenderGlobal.renderBlockLayer` — squared Euclidean distance greater than
/// one block squared. The strict `>` is load-bearing: exactly one block does not re-sort.
#[must_use]
pub fn needs_resort(last: DVec3, now: DVec3) -> bool {
    let dx = now.x - last.x;
    let dy = now.y - last.y;
    let dz = now.z - last.z;
    dx * dx + dy * dy + dz * dz > 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::vertex::TerrainVertex;

    fn quad_at(y: f32) -> [TerrainVertex; 4] {
        let v = |x: f32, z: f32| TerrainVertex::new([x, y, z], [0.0; 2], [0; 4], [0; 4]);
        [v(0.0, 0.0), v(1.0, 0.0), v(1.0, 1.0), v(0.0, 1.0)]
    }

    #[test]
    fn quads_come_out_farthest_first() {
        let mut b = Buffer::new();
        for y in [0.0f32, 8.0, 4.0, 12.0] {
            b.push_quad(quad_at(y));
        }
        // Eye just below the stack: 12 is farthest, 0 nearest.
        sort_translucent_local(&mut b, [0.5, -1.0, 0.5]);
        let ys: Vec<f32> = (0..b.quad_count())
            .map(|i| b.quad(i)[0].position()[1])
            .collect();
        assert_eq!(ys, [12.0, 8.0, 4.0, 0.0]);
    }

    #[test]
    fn sorting_is_stable_under_repetition() {
        let mut b = Buffer::new();
        for y in [3.0f32, 1.0, 2.0] {
            b.push_quad(quad_at(y));
        }
        sort_translucent_local(&mut b, [0.5, 0.0, 0.5]);
        let first: Vec<f32> = (0..b.quad_count())
            .map(|i| b.quad(i)[0].position()[1])
            .collect();
        sort_translucent_local(&mut b, [0.5, 0.0, 0.5]);
        let second: Vec<f32> = (0..b.quad_count())
            .map(|i| b.quad(i)[0].position()[1])
            .collect();
        assert_eq!(first, second);
        assert_eq!(first, [3.0, 2.0, 1.0]);
    }

    #[test]
    fn moving_the_camera_flips_the_order() {
        let mut b = Buffer::new();
        b.push_quad(quad_at(0.0));
        b.push_quad(quad_at(10.0));
        sort_translucent_local(&mut b, [0.5, -5.0, 0.5]);
        assert_eq!(b.quad(0)[0].position()[1], 10.0);
        sort_translucent_local(&mut b, [0.5, 15.0, 0.5]);
        assert_eq!(b.quad(0)[0].position()[1], 0.0);
    }

    #[test]
    fn world_space_sorting_accounts_for_the_section_origin() {
        let mut b = Buffer::new();
        b.push_quad(quad_at(0.0));
        b.push_quad(quad_at(15.0));
        // Section (0, 4, 0) spans y 64..80. A camera at y = 100 is above both.
        sort_translucent(
            &mut b,
            ChunkSectionPos::new(0, 4, 0),
            DVec3::new(0.5, 100.0, 0.5),
        );
        assert_eq!(
            b.quad(0)[0].position()[1],
            0.0,
            "the lower quad is farther away"
        );
    }

    #[test]
    fn resort_threshold_is_one_block_of_euclidean_distance() {
        let a = DVec3::new(0.0, 0.0, 0.0);
        assert!(!needs_resort(a, DVec3::new(0.9, 0.0, 0.0)));
        assert!(!needs_resort(a, DVec3::new(1.0, 0.0, 0.0)));
        assert!(needs_resort(a, DVec3::new(1.1, 0.0, 0.0)));
        assert!(needs_resort(a, DVec3::new(0.0, -1.5, 0.0)));
    }

    #[test]
    fn diagonal_motion_resorts_when_its_total_distance_exceeds_one_block() {
        let origin = DVec3::new(0.0, 0.0, 0.0);
        // Every individual axis is below one, so the old per-axis cube test returned false;
        // Euclidean distance squared is 1.62, so Java returns true.
        assert!(needs_resort(origin, DVec3::new(0.9, 0.0, 0.9)));

        // Exercise all three axes on both sides of the spherical boundary. These values are
        // deliberately below one per axis, keeping the old implementation's answer false for
        // both and proving the positive case reaches the divergent region.
        assert!(!needs_resort(origin, DVec3::new(0.57, -0.57, 0.57)));
        assert!(needs_resort(origin, DVec3::new(0.58, -0.58, 0.58)));
    }

    #[test]
    fn resort_distance_is_translation_invariant() {
        let last = DVec3::new(30_000_000.0, -64.0, -30_000_000.0);
        assert!(!needs_resort(
            last,
            DVec3::new(30_000_000.6, -63.6, -29_999_999.6)
        ));
        assert!(needs_resort(
            last,
            DVec3::new(30_000_000.8, -63.2, -30_000_000.0)
        ));
    }

    #[test]
    fn an_empty_or_single_quad_buffer_is_left_alone() {
        let mut b = Buffer::new();
        sort_translucent_local(&mut b, [0.0; 3]);
        assert!(b.is_empty());
        b.push_quad(quad_at(1.0));
        sort_translucent_local(&mut b, [0.0; 3]);
        assert_eq!(b.quad_count(), 1);
    }
}
