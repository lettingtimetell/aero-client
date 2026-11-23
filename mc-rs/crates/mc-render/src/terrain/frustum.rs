//! View-frustum culling.
//!
//! Vanilla pulls the six clipping planes out of `projection * modelview` with the standard
//! Gribb/Hartmann extraction and tests a box by asking, for each plane, whether *all eight*
//! corners are behind it. We do the same, except the matrix comes from
//! [`Camera::view_proj`](crate::Camera) rather than from `glGetFloat`, and — because S10's
//! geometry is camera-relative and its view matrix is rotation-only — the box has to be moved
//! into camera-relative space first, which is exactly what vanilla's `Frustum` wrapper does
//! with its `xPosition/yPosition/zPosition` subtraction.
//
// java: client/renderer/culling/ClippingHelper.java, ClippingHelperImpl.java, Frustum.java

use glam::Mat4;
use mc_math::{ChunkSectionPos, DVec3};

/// Six normalised clipping planes, each `ax + by + cz + d`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
    planes: [[f32; 4]; 6],
    origin: DVec3,
}

impl Frustum {
    /// Extract the planes from a camera-relative `view_proj` and remember the eye position the
    /// geometry is relative to.
    ///
    /// java: `ClippingHelperImpl.init` (the extraction) + `Frustum.setPosition` (the origin).
    #[must_use]
    pub fn from_view_proj(view_proj: Mat4, origin: DVec3) -> Self {
        // glam is column-major: `m.col(c)[r]` is the Java `clippingMatrix[c * 4 + r]`.
        let m = view_proj.to_cols_array();
        let at = |c: usize, r: usize| m[c * 4 + r];

        let row = |r: usize| [at(0, r), at(1, r), at(2, r), at(3, r)];
        let sub = |a: [f32; 4], b: [f32; 4]| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
        let add = |a: [f32; 4], b: [f32; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
        let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));

        // Right, left, bottom, top, far, near — vanilla's order, which only matters for
        // debugging since the test is a conjunction.
        //
        // The near plane is `row2` alone rather than vanilla's `row3 + row2`: S10's projection
        // is `glam::camera::rh::proj::directx::perspective`, so clip-space depth runs 0..w
        // (wgpu/D3D) instead of OpenGL's -w..w. Using the GL form here culls everything in
        // front of the camera.
        let mut planes = [
            sub(r3, r0),
            add(r3, r0),
            add(r3, r1),
            sub(r3, r1),
            sub(r3, r2),
            r2,
        ];
        for p in &mut planes {
            normalize(p);
        }
        Self { planes, origin }
    }

    /// Build a frustum straight from a [`Camera`](crate::Camera).
    #[must_use]
    pub fn from_camera(cam: &crate::Camera) -> Self {
        Self::from_view_proj(cam.view_proj(), cam.pos)
    }

    /// The eye position this frustum's camera-relative space is measured from.
    #[must_use]
    pub fn origin(&self) -> DVec3 {
        self.origin
    }

    /// java: `ClippingHelper.isBoxInFrustum` — a box is culled only when every corner is
    /// strictly outside one single plane. Corners are in **world** space; the origin
    /// subtraction happens here.
    #[must_use]
    pub fn contains_box(&self, min: DVec3, max: DVec3) -> bool {
        let lo = (min - self.origin).as_vec3();
        let hi = (max - self.origin).as_vec3();
        for p in &self.planes {
            let dot = |x: f32, y: f32, z: f32| p[0] * x + p[1] * y + p[2] * z + p[3];
            let outside = dot(lo.x, lo.y, lo.z) <= 0.0
                && dot(hi.x, lo.y, lo.z) <= 0.0
                && dot(lo.x, hi.y, lo.z) <= 0.0
                && dot(hi.x, hi.y, lo.z) <= 0.0
                && dot(lo.x, lo.y, hi.z) <= 0.0
                && dot(hi.x, lo.y, hi.z) <= 0.0
                && dot(lo.x, hi.y, hi.z) <= 0.0
                && dot(hi.x, hi.y, hi.z) <= 0.0;
            if outside {
                return false;
            }
        }
        true
    }

    /// Whether a 16³ chunk section is potentially visible.
    /// java: `RenderChunk.boundingBox` tested by `ICamera.isBoundingBoxInFrustum`.
    #[must_use]
    pub fn contains_section(&self, pos: ChunkSectionPos) -> bool {
        let o = pos.origin();
        let min = DVec3::new(f64::from(o.x), f64::from(o.y), f64::from(o.z));
        self.contains_box(min, min + DVec3::splat(16.0))
    }
}

/// Beyond this half-width the table is not built and every query falls back to
/// [`Frustum::contains_section`]; the walk's own dense sets stop at the same bound.
const MAX_TABLE_RADIUS: u32 = 64;

/// [`Frustum::contains_section`] for the sections of one terrain walk, with every plane's
/// per-axis products computed once per walk instead of once per box corner (#7 follow-up).
///
/// After the walk stopped hashing, this test is its largest per-section cost: one call per
/// newly reached in-range neighbour, ~5,000 per walk at render distance 16 on a void map.
///
/// **Bit-identical** to `contains_section`, by construction. There, a corner's plane value is
/// `p[0] * x + p[1] * y + p[2] * z + p[3]`, which Rust evaluates as `((p0·x + p1·y) + p2·z) +
/// p3` in `f32` with no reassociation and no FMA contraction, and each operand comes from one
/// axis only: `x = (f64(origin.x) [+ 16.0] - eye.x) as f32`. So the three products are
/// tabulated per section coordinate (`lo` and `hi` corner) with the same `f64` arithmetic and
/// the same `f32` multiplications, and the same three additions run in the same order. The
/// "all eight corners outside one plane" test is a conjunction, so corner order is free.
#[derive(Debug, Default)]
pub struct SectionFrustum {
    /// The frustum the table was built from, for queries outside it.
    frustum: Option<Frustum>,
    /// Section x and z of the table's first entry, and its width (`0` when not built).
    x0: i32,
    z0: i32,
    side: i32,
    /// Per section x: `[plane k lo, plane k hi]` products `p[k][0] * x`, flattened `2k + corner`.
    px: Vec<[f32; 12]>,
    /// Per section y `0..16`: the `p[k][1] * y` products.
    py: Vec<[f32; 12]>,
    /// Per section z: the `p[k][2] * z` products.
    pz: Vec<[f32; 12]>,
    /// `p[k][3]`.
    d: [f32; 6],
}

impl SectionFrustum {
    /// Tabulate `frustum` for the sections within `radius` columns of `center` (any `y` in the
    /// world's sixteen). Reuses the previous allocation.
    pub fn rebuild(&mut self, frustum: &Frustum, center: ChunkSectionPos, radius: u32) {
        self.frustum = Some(*frustum);
        self.px.clear();
        self.py.clear();
        self.pz.clear();
        if radius > MAX_TABLE_RADIUS {
            self.side = 0;
            return;
        }
        let r = radius as i32;
        self.x0 = center.x - r;
        self.z0 = center.z - r;
        self.side = 2 * r + 1;
        let planes = &frustum.planes;
        let eye = frustum.origin;
        for (k, plane) in planes.iter().enumerate() {
            self.d[k] = plane[3];
        }
        // `contains_box(min, min + 16)` with `min = f64(section origin)`: per axis, the corner
        // operands are `(min - eye) as f32` and `((min + 16.0) - eye) as f32`.
        let products = |axis: usize, min: f64, eye: f64| {
            let lo = (min - eye) as f32;
            let hi = ((min + 16.0) - eye) as f32;
            let mut out = [0.0f32; 12];
            for (k, plane) in planes.iter().enumerate() {
                out[2 * k] = plane[axis] * lo;
                out[2 * k + 1] = plane[axis] * hi;
            }
            out
        };
        for i in 0..self.side {
            let origin = ChunkSectionPos::new(self.x0 + i, 0, self.z0 + i).origin();
            self.px.push(products(0, f64::from(origin.x), eye.x));
            self.pz.push(products(2, f64::from(origin.z), eye.z));
        }
        for y in 0..16 {
            let origin = ChunkSectionPos::new(0, y, 0).origin();
            self.py.push(products(1, f64::from(origin.y), eye.y));
        }
    }

    /// Exactly [`Frustum::contains_section`] of the frustum last passed to
    /// [`SectionFrustum::rebuild`].
    ///
    /// # Panics
    ///
    /// If called before the first [`SectionFrustum::rebuild`].
    #[inline]
    #[must_use]
    pub fn contains_section(&self, pos: ChunkSectionPos) -> bool {
        let (ix, iz) = (pos.x.wrapping_sub(self.x0), pos.z.wrapping_sub(self.z0));
        if ix < 0 || iz < 0 || ix >= self.side || iz >= self.side || !(0..16).contains(&pos.y) {
            return self
                .frustum
                .as_ref()
                .expect("SectionFrustum::rebuild before use")
                .contains_section(pos);
        }
        let (ax, by, cz) = (
            &self.px[ix as usize],
            &self.py[pos.y as usize],
            &self.pz[iz as usize],
        );
        for (k, &d) in self.d.iter().enumerate() {
            let v = |cx: usize, cy: usize, cz_: usize| {
                ax[2 * k + cx] + by[2 * k + cy] + cz[2 * k + cz_] + d
            };
            let outside = v(0, 0, 0) <= 0.0
                && v(1, 0, 0) <= 0.0
                && v(0, 1, 0) <= 0.0
                && v(1, 1, 0) <= 0.0
                && v(0, 0, 1) <= 0.0
                && v(1, 0, 1) <= 0.0
                && v(0, 1, 1) <= 0.0
                && v(1, 1, 1) <= 0.0;
            if outside {
                return false;
            }
        }
        true
    }
}

fn normalize(p: &mut [f32; 4]) {
    // java: ClippingHelperImpl.normalize — divides all four terms, including `d`.
    let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    if len > 0.0 {
        p[0] /= len;
        p[1] /= len;
        p[2] /= len;
        p[3] /= len;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Camera;

    fn camera_at(pos: DVec3, yaw: f32, pitch: f32) -> Camera {
        let mut c = Camera {
            pos,
            yaw,
            pitch,
            aspect: 1.0,
            far: 512.0,
            ..Camera::default()
        };
        c.update();
        c
    }

    /// #7 follow-up: the per-walk table must answer `contains_section` exactly — same `bool` for
    /// every section of the walk's square and every world `y`, and the fallback outside it —
    /// for cameras near and far from the origin (where the `f64 → f32` step matters), at odd
    /// angles, FOVs, aspects and far planes, and on section and block boundaries.
    #[test]
    fn the_section_table_answers_contains_section_bit_for_bit() {
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut table = SectionFrustum::default();
        let (mut inside, mut outside) = (0usize, 0usize);
        for case in 0..120 {
            let r = next();
            let scale = [1.0, 16.0, 1.0e4, 3.0e7][(r % 4) as usize];
            let coord = |bits: u64| ((bits % 20_001) as f64 / 10_000.0 - 1.0) * scale;
            let mut pos = DVec3::new(
                coord(r >> 2),
                (r >> 20) as f64 % 300.0 - 20.0,
                coord(r >> 30),
            );
            if case % 7 == 0 {
                // Exactly on a section corner.
                pos = DVec3::new(
                    (pos.x / 16.0).round() * 16.0,
                    (pos.y / 16.0).round() * 16.0,
                    (pos.z / 16.0).round() * 16.0,
                );
            }
            let mut cam = Camera {
                pos,
                yaw: (next() % 7200) as f32 / 20.0 - 180.0,
                pitch: (next() % 1800) as f32 / 10.0 - 90.0,
                aspect: [1.0, 16.0 / 9.0, 21.0 / 9.0, 0.5][(next() % 4) as usize],
                far: [64.0, 181.0, 362.0, 1024.0][(next() % 4) as usize],
                ..Camera::default()
            };
            cam.fov = [30.0, 70.0, 110.0][(next() % 3) as usize];
            cam.update();
            let f = Frustum::from_camera(&cam);
            let radius = [0u32, 1, 4, 8, 12][(next() % 5) as usize];
            let center = mc_math::BlockPos::from_vec3(cam.pos).section_pos();
            table.rebuild(&f, center, radius);
            let r = radius as i32;
            for dx in -r - 1..=r + 1 {
                for dz in -r - 1..=r + 1 {
                    for y in -1..=16 {
                        let section = ChunkSectionPos::new(center.x + dx, y, center.z + dz);
                        let expected = f.contains_section(section);
                        assert_eq!(
                            table.contains_section(section),
                            expected,
                            "case {case}: {section:?} from {:?}",
                            cam.pos
                        );
                        if expected {
                            inside += 1;
                        } else {
                            outside += 1;
                        }
                    }
                }
            }
        }
        assert!(inside > 10_000 && outside > 10_000, "{inside} / {outside}");
        // Above the table bound it is not built, and still answers.
        let cam = camera_at(DVec3::new(8.0, 70.0, 8.0), 0.0, 0.0);
        let f = Frustum::from_camera(&cam);
        table.rebuild(&f, ChunkSectionPos::new(0, 4, 0), MAX_TABLE_RADIUS + 1);
        let s = ChunkSectionPos::new(0, 4, 3);
        assert_eq!(table.contains_section(s), f.contains_section(s));
    }

    #[test]
    fn a_box_in_front_of_the_camera_is_visible() {
        // Vanilla yaw 0 looks towards +Z.
        let cam = camera_at(DVec3::ZERO, 0.0, 0.0);
        let f = Frustum::from_camera(&cam);
        assert!(f.contains_box(DVec3::new(-1.0, -1.0, 9.0), DVec3::new(1.0, 1.0, 11.0)));
    }

    #[test]
    fn a_box_behind_the_camera_is_culled() {
        let cam = camera_at(DVec3::ZERO, 0.0, 0.0);
        let f = Frustum::from_camera(&cam);
        assert!(!f.contains_box(DVec3::new(-1.0, -1.0, -11.0), DVec3::new(1.0, 1.0, -9.0)));
    }

    #[test]
    fn a_box_far_to_the_side_is_culled() {
        let cam = camera_at(DVec3::ZERO, 0.0, 0.0);
        let f = Frustum::from_camera(&cam);
        assert!(!f.contains_box(DVec3::new(199.0, -1.0, 9.0), DVec3::new(201.0, 1.0, 11.0)));
    }

    #[test]
    fn turning_around_swaps_which_side_is_visible() {
        let front = DVec3::new(9.0, -1.0, -1.0);
        let back = front + DVec3::splat(2.0);
        let cam = camera_at(DVec3::ZERO, -90.0, 0.0); // vanilla yaw -90 faces +X
        let f = Frustum::from_camera(&cam);
        assert!(f.contains_box(front, back));
        let cam = camera_at(DVec3::ZERO, 90.0, 0.0);
        let f = Frustum::from_camera(&cam);
        assert!(!f.contains_box(front, back));
    }

    #[test]
    fn beyond_the_far_plane_is_culled() {
        let mut cam = camera_at(DVec3::ZERO, 0.0, 0.0);
        cam.far = 64.0;
        cam.update();
        let f = Frustum::from_camera(&cam);
        assert!(f.contains_box(DVec3::new(-1.0, -1.0, 30.0), DVec3::new(1.0, 1.0, 32.0)));
        assert!(!f.contains_box(DVec3::new(-1.0, -1.0, 1000.0), DVec3::new(1.0, 1.0, 1002.0)));
    }

    #[test]
    fn section_test_uses_the_full_sixteen_cube() {
        let cam = camera_at(DVec3::new(8.0, 8.0, 8.0), 0.0, 0.0);
        let f = Frustum::from_camera(&cam);
        // The section the camera is inside is always visible.
        assert!(f.contains_section(ChunkSectionPos::new(0, 0, 0)));
        // One far behind it is not.
        assert!(!f.contains_section(ChunkSectionPos::new(0, 0, -20)));
    }

    #[test]
    fn planes_are_unit_length() {
        let cam = camera_at(DVec3::ZERO, 37.0, 12.0);
        let f = Frustum::from_camera(&cam);
        for p in &f.planes {
            let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-4, "plane not normalised: {p:?}");
        }
    }

    #[test]
    fn directx_plane_extraction_uses_rows_and_preserves_all_six_signs() {
        // Deliberately asymmetric row values catch a column/row transpose while keeping
        // every expected normalized plane easy to state.  This matrix is not a camera
        // projection; it is an extraction oracle for the same column-major convention used
        // by glam and by Java's FloatBuffer/OpenGL matrices.
        let matrix = Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0, // column 0
            0.0, 2.0, 0.0, 0.0, // column 1
            0.0, 0.0, 3.0, 0.0, // column 2
            0.0, 0.0, 4.0, 5.0, // column 3
        ]);
        let f = Frustum::from_view_proj(matrix, DVec3::ZERO);

        // right, left, bottom, top, far, near.  The normalizer divides all four terms.
        let expected = [
            [-1.0, 0.0, 0.0, 5.0],
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 1.0, 0.0, 2.5],
            [0.0, -1.0, 0.0, 2.5],
            [0.0, 0.0, -1.0, 1.0 / 3.0],
            [0.0, 0.0, 1.0, 4.0 / 3.0],
        ];
        for (actual, want) in f.planes.iter().zip(expected) {
            for (got, expected) in actual.iter().zip(want) {
                assert!(
                    (got - expected).abs() < 1e-6,
                    "got {actual:?}, expected {want:?}"
                );
            }
        }

        // Exact Java boundary behavior: a box is culled only when every corner is <= 0 on
        // one plane.  These controls exercise the DirectX near plane (z >= 0), the opposite
        // far plane (z <= 5 here), and the left/right planes independently.
        assert!(!f.contains_box(
            DVec3::new(-0.5, -0.5, -2.0),
            DVec3::new(0.5, 0.5, -4.0 / 3.0),
        ));
        assert!(f.contains_box(DVec3::new(-0.5, -0.5, 0.0), DVec3::new(0.5, 0.5, 0.2)));
        assert!(!f.contains_box(DVec3::new(-0.5, -0.5, 1.0), DVec3::new(0.5, 0.5, 2.0)));
        assert!(!f.contains_box(DVec3::new(-6.0, -0.5, 1.0), DVec3::new(-5.0, 0.5, 2.0)));
        assert!(!f.contains_box(DVec3::new(5.0, -0.5, 1.0), DVec3::new(6.0, 0.5, 2.0)));
    }

    #[test]
    fn near_plane_matches_the_actual_zero_to_one_clip_depth() {
        // `Camera::proj` is the right-handed DirectX/WebGPU projection: a point at the
        // near distance maps to clip z = 0, while the same point closer to the eye maps
        // to clip z < 0.  The near plane must therefore be `row2`, not GL's `row3 + row2`.
        let mut cam = camera_at(DVec3::ZERO, 0.0, 0.0);
        cam.near = 0.05;
        cam.far = 100.0;
        cam.update();
        let f = Frustum::from_camera(&cam);

        let clip_depth = |point: DVec3| {
            let clip = cam.view_proj() * cam.relative(point).extend(1.0);
            (clip.z, clip.w, clip.z / clip.w)
        };

        let before_near = DVec3::new(0.0, 0.0, 0.049);
        let after_near = DVec3::new(0.0, 0.0, 0.051);
        let (before_z, before_w, before_ndc) = clip_depth(before_near);
        let (after_z, after_w, after_ndc) = clip_depth(after_near);
        assert!(before_w > 0.0 && after_w > 0.0);
        assert!(
            before_z < 0.0 && before_ndc < 0.0,
            "before near: {before_z}/{before_w}"
        );
        assert!(
            after_z > 0.0 && after_ndc > 0.0,
            "after near: {after_z}/{after_w}"
        );

        // Keep the boxes non-degenerate so this exercises the same eight-corner AABB oracle
        // used by RenderChunk, while keeping every corner on the same side of the plane.
        assert!(!f.contains_box(
            DVec3::new(-0.001, -0.001, 0.040),
            DVec3::new(0.001, 0.001, 0.049),
        ));
        assert!(f.contains_box(
            DVec3::new(-0.001, -0.001, 0.051),
            DVec3::new(0.001, 0.001, 0.060),
        ));

        // A box crossing the plane is retained because vanilla rejects only when all eight
        // corners are outside one plane.  This also pins the `<= 0.0` boundary convention.
        assert!(f.contains_box(
            DVec3::new(-0.001, -0.001, 0.049),
            DVec3::new(0.001, 0.001, 0.051),
        ));
    }

    #[test]
    fn clip_depth_oracle_keeps_far_and_side_controls_independent() {
        let mut cam = camera_at(DVec3::ZERO, 0.0, 0.0);
        cam.near = 0.05;
        cam.far = 100.0;
        cam.update();
        let f = Frustum::from_camera(&cam);

        let clip_depth = |point: DVec3| {
            let clip = cam.view_proj() * cam.relative(point).extend(1.0);
            clip.z / clip.w
        };

        let inside = DVec3::new(0.0, 0.0, 50.0);
        let beyond_far = DVec3::new(0.0, 0.0, 101.0);
        assert!(clip_depth(inside) > 0.0 && clip_depth(inside) < 1.0);
        assert!(clip_depth(beyond_far) > 1.0);
        assert!(f.contains_box(DVec3::new(-0.5, -0.5, 49.0), DVec3::new(0.5, 0.5, 51.0),));
        assert!(!f.contains_box(DVec3::new(-0.5, -0.5, 100.5), DVec3::new(0.5, 0.5, 101.5),));

        // At z = 1, x = 100 is far outside the horizontal frustum even though its depth
        // remains between the near and far planes.
        let side = DVec3::new(100.0, 0.0, 1.0);
        assert!(clip_depth(side) > 0.0 && clip_depth(side) < 1.0);
        assert!(!f.contains_box(DVec3::new(99.0, -0.5, 0.5), DVec3::new(101.0, 0.5, 1.5),));
    }
}
