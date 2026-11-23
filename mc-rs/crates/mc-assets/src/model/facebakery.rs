//! Element + face -> [`BakedQuad`] geometry. Hazard **H12**: `uvlock` and rotation are where
//! most reimplementations get subtly wrong output.
//!
// java: FaceBakery.java, EnumFaceDirection.java, ModelRotation.java
//!
//! # Rotation convention (derived, not guessed)
//!
//! Both `ModelRotation` (blockstate `x`/`y`) and `BlockPartRotation` (an element's own
//! `rotation`) build their transform with LWJGL's `Matrix4f.rotate(angle, axis, ...)`, which
//! is the standard active-rotation (Rodrigues) formula. Expanding that formula per axis and
//! cross-checking against the one unambiguous fact of the model format — "`y: 90` rotates the
//! model 90° **clockwise viewed from above**" — pins down the sign for
//! all three axes at once, since LWJGL uses one formula shape for every axis:
//!
//! ```text
//! rotate about X by θ: (y, z) -> (y cosθ - z sinθ,  y sinθ + z cosθ)
//! rotate about Y by θ: (x, z) -> (x cosθ + z sinθ, -x sinθ + z cosθ)
//! rotate about Z by θ: (x, y) -> (x cosθ - y sinθ,  x sinθ + y cosθ)
//! ```
//!
//! `ModelRotation` calls this with `θ = -angle_from_json` (java negates x/y before building the
//! matrix); `BlockPartRotation` calls it with `θ = +angle_from_json`. Both apply about the
//! block center `(0.5, 0.5, 0.5)`; `BlockPartRotation` applies about its own JSON `origin`
//! instead, and axis-aligned rotation is X-then-Y (`ModelRotation` builds `Y * X`).
//!
//! # Vertex slot order
//!
//! Java does not store a rotated quad's vertices in the order it built them. `makeBakedQuad`
//! runs three steps this port now reproduces in the same sequence:
//!
//! 1. [`rotate_vertex_slot`] (java: `ModelRotation.rotateVertex`) shifts each vertex's storage
//!    slot cyclically by a `k` shared by all four vertices of the quad.
//! 2. `lockUv` (here, the [`lock_vertex_uv`] branch of [`bake_quad`]) re-derives each slot's UV
//!    from its final *position* and writes it to slot `(i + 4 - rotation/90) % 4` — java's
//!    `BlockFaceUV.func_178345_c`, the inverse of the `func_178347_d` shift the unlocked path
//!    uses, so `uvlock` combined with a per-face `"rotation"` lands where java lands it.
//! 3. [`apply_facing`] (java: `FaceBakery.applyFacing`, applied only when the element has no
//!    `rotation` of its own) rebuilds every slot's position from the *final* facing's
//!    `EnumFaceDirection` table over the quad's own bounding box, and re-pairs each slot's UV
//!    to it by position.
//!
//! Because step 3 overwrites step 1's layout whenever it runs, and step 1's `k` is not step 3's
//! shift, an earlier pass that skipped both did **not** land on java's slot order for a rotated
//! variant — e.g. `stone`'s four y-rotated variants. Winding was never affected (all three
//! layouts trace the same loop in the same rotational sense, so back-face culling and the
//! quad's geometric facing are identical either way), but any consumer that indexes a vertex by
//! *slot* — vanilla's own `BlockModelRenderer.VertexTranslations` ambient-occlusion table, and
//! anything that reproduces it — reads the wrong corner. That is what these three steps close.
//!
//! Per-face **shade colour** (`FaceBakery.getFaceShadeColor`) is not computed here:
//! there is no per-vertex colour in [`crate::model::Vertex`], by design — [`BakedQuad::shade`]
//! carries the element's `shade` flag through unchanged, and combining it with the quad's
//! facing into an actual brightness (and blending that against ambient occlusion) is the
//! terrain mesher's job (S11), done once per world position rather than once per bake.
//!
//! [`BakedQuad::face`] stays the *label* rotation ([`rotate_face`]) rather than java's
//! geometric `getFacingFromVertexData`. The two agree exactly whenever the element has no
//! `rotation` of its own — [`facing_from_vertex_data`] exists and
//! `the_geometric_and_label_facings_agree_for_unrotated_elements` proves it — and the
//! divergence for an obliquely rotated element (java falls back to `UP`; this port keeps the
//! JSON face's rotated label) is the pre-existing documented one, deliberately unchanged here
//! because S11's mesher keys neighbour culling and AO off that field.

use mc_math::facing::Axis;
use mc_math::Facing;

use super::blockmodel::{RawElement, RawFace, RawRotation};
use super::{BakedQuad, Vertex};
use crate::sprite::UvRect;

/// `x`/`y` model rotation in quarter turns, plus whether faces should stay texture-locked to
/// world space rather than spinning with the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BakeRotation {
    /// Quarter turns about X, applied first.
    pub quarters_x: u8,
    /// Quarter turns about Y, applied second.
    pub quarters_y: u8,
    /// `variants[].uvlock`.
    pub uvlock: bool,
}

/// Rotates `facing` the way `ModelRotation.rotateFace` does: `quarters_x` steps about X, then
/// (only if that didn't land on the Y axis) `quarters_y` steps about Y.
#[must_use]
pub fn rotate_face(facing: Facing, quarters_x: u8, quarters_y: u8) -> Facing {
    let mut f = facing;
    for _ in 0..(quarters_x % 4) {
        f = f.rotate_around(Axis::X);
    }
    if f.axis() != Axis::Y {
        for _ in 0..(quarters_y % 4) {
            f = f.rotate_around(Axis::Y);
        }
    }
    f
}

/// LWJGL 2's `org.lwjgl.util.vector.Matrix4f`, stored the way LWJGL stores it: `m[c][r]` is
/// its field `m<c><r>` (column `c`, row `r`).
///
/// Both `FaceBakery.rotatePart` and `ModelRotation`'s constructor build their transform with
/// this class in `float` arithmetic, so a quarter turn is not an exact permutation in vanilla:
/// `(float)Math.cos(-90° in float radians)` is `-4.371139E-8`, and `1 - c + c` rounds to
/// `0.99999994`. Reproducing the class term for term (operation order, the float trig, the
/// separate X and Y matrices multiplied together) is what makes every baked vertex
/// bit-identical to Java's; see `tests/face_bakery_java_parity.rs`. Bake time only.
type LwjglMatrix = [[f32; 4]; 4];

/// `Matrix4f.setIdentity()`.
const LWJGL_IDENTITY: LwjglMatrix = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// java: the `0.017453292F` degree-to-radian literal both callers multiply by in `float`.
const DEGREES_TO_RADIANS: f32 = f32::from_bits(0x3C8E_FA35);

/// java: LWJGL `Matrix4f.rotate(angle, axis, src, dest)`, term for term — including
/// `(float)Math.cos(angle)` on the float angle widened to double, and the left-to-right sums
/// whose `0 * x` terms decide signed zeros.
fn lwjgl_rotate(angle: f32, axis: [f32; 3], src: &LwjglMatrix) -> LwjglMatrix {
    let c = f64::from(angle).cos() as f32;
    let s = f64::from(angle).sin() as f32;
    let oneminusc = 1.0 - c;
    let [ax, ay, az] = axis;
    let xy = ax * ay;
    let yz = ay * az;
    let xz = ax * az;
    let xs = ax * s;
    let ys = ay * s;
    let zs = az * s;
    let f = [
        [
            ax * ax * oneminusc + c,
            xy * oneminusc + zs,
            xz * oneminusc - ys,
        ],
        [
            xy * oneminusc - zs,
            ay * ay * oneminusc + c,
            yz * oneminusc + xs,
        ],
        [
            xz * oneminusc + ys,
            yz * oneminusc - xs,
            az * az * oneminusc + c,
        ],
    ];
    let mut dest = *src;
    for (column, fc) in f.iter().enumerate() {
        for row in 0..4 {
            dest[column][row] = src[0][row] * fc[0] + src[1][row] * fc[1] + src[2][row] * fc[2];
        }
    }
    dest
}

/// java: LWJGL `Matrix4f.mul(left, right, dest)`.
fn lwjgl_mul(left: &LwjglMatrix, right: &LwjglMatrix) -> LwjglMatrix {
    let mut dest = [[0.0; 4]; 4];
    for (column, out) in dest.iter_mut().enumerate() {
        for (row, value) in out.iter_mut().enumerate() {
            *value = left[0][row] * right[column][0]
                + left[1][row] * right[column][1]
                + left[2][row] * right[column][2]
                + left[3][row] * right[column][3];
        }
    }
    dest
}

/// java: `FaceBakery.rotateScale` — LWJGL `Matrix4f.transform` of `(p - origin, 1)`, then the
/// per-axis scale, then `+ origin`.
fn rotate_scale(p: [f32; 3], origin: [f32; 3], matrix: &LwjglMatrix, scale: [f32; 3]) -> [f32; 3] {
    let v = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2], 1.0];
    let row = |r: usize| {
        matrix[0][r] * v[0] + matrix[1][r] * v[1] + matrix[2][r] * v[2] + matrix[3][r] * v[3]
    };
    [
        row(0) * scale[0] + origin[0],
        row(1) * scale[1] + origin[1],
        row(2) * scale[2] + origin[2],
    ]
}

/// java: the `ModelRotation(int x, int y)` constructor — `rotate(-x * DEG, X)` and
/// `rotate(-y * DEG, Y)` on two identity matrices, then `Matrix4f.mul(rotY, rotX)`.
fn model_rotation_matrix(quarters_x: u8, quarters_y: u8) -> LwjglMatrix {
    let x = -(i32::from(quarters_x % 4) * 90) as f32;
    let y = -(i32::from(quarters_y % 4) * 90) as f32;
    let rot_x = lwjgl_rotate(x * DEGREES_TO_RADIANS, [1.0, 0.0, 0.0], &LWJGL_IDENTITY);
    let rot_y = lwjgl_rotate(y * DEGREES_TO_RADIANS, [0.0, 1.0, 0.0], &LWJGL_IDENTITY);
    lwjgl_mul(&rot_y, &rot_x)
}

// java: FaceBakery.SCALE_ROTATION_22_5 = 1.0F / (float)Math.cos(0.39269909262657166D) - 1.0F,
// every step in float: 0.08239222 (0x3DA8BD40). An earlier hand-typed 0.0204313 shrank every
// rescaled 22.5° element (all fire planes, the attached tripwire hook) to a quarter of
// vanilla's stretch.
const SCALE_ROTATION_22_5: f32 = f32::from_bits(0x3DA8_BD40);
// java: FaceBakery.SCALE_ROTATION_GENERAL = 1.0F / (float)Math.cos(Math.PI / 4D) - 1.0F:
// 0.41421354 (0x3ED413CC), one ulp under the decimal 1/cos(45°) - 1 typed here before.
const SCALE_ROTATION_GENERAL: f32 = f32::from_bits(0x3ED4_13CC);

/// Applies an element's own `rotation` (java: `FaceBakery.rotatePart`), including the
/// `rescale` bounding-box correction.
fn apply_part_rotation(p: [f32; 3], rot: &RawRotation) -> [f32; 3] {
    // java: `BlockPart.Deserializer.parseRotation` scales the origin by 0.0625F (exact).
    let origin = [
        rot.origin[0] / 16.0,
        rot.origin[1] / 16.0,
        rot.origin[2] / 16.0,
    ];
    let axis = super::blockmodel::parse_axis(&rot.axis).unwrap_or(Axis::Y);
    let (axis_vec, scale_mask) = match axis {
        Axis::X => ([1.0, 0.0, 0.0], [0.0, 1.0, 1.0]),
        Axis::Y => ([0.0, 1.0, 0.0], [1.0, 0.0, 1.0]),
        Axis::Z => ([0.0, 0.0, 1.0], [1.0, 1.0, 0.0]),
    };
    let matrix = lwjgl_rotate(rot.angle * DEGREES_TO_RADIANS, axis_vec, &LWJGL_IDENTITY);
    // java: `vector3f.scale(SCALE_*)` then `Vector3f.add(vector3f, (1, 1, 1))`, or `(1, 1, 1)`
    // when `rescale` is off. Either way `rotateScale` multiplies before adding the origin back.
    let scale = if rot.rescale {
        let extra = if rot.angle.abs() == 22.5 {
            SCALE_ROTATION_22_5
        } else {
            SCALE_ROTATION_GENERAL
        };
        [
            scale_mask[0] * extra + 1.0,
            scale_mask[1] * extra + 1.0,
            scale_mask[2] * extra + 1.0,
        ]
    } else {
        [1.0; 3]
    };
    rotate_scale(p, origin, &matrix, scale)
}

/// Applies the blockstate variant's `x`/`y` rotation about the block center (java:
/// `FaceBakery.rotateVertex`, which returns early for `ModelRotation.X0_Y0` and otherwise calls
/// `rotateScale(position, (0.5, 0.5, 0.5), modelRotation.getMatrix4d(), (1, 1, 1))`).
fn apply_model_rotation(p: [f32; 3], rot: BakeRotation) -> [f32; 3] {
    if rot.quarters_x.is_multiple_of(4) && rot.quarters_y.is_multiple_of(4) {
        return p;
    }
    let matrix = model_rotation_matrix(rot.quarters_x, rot.quarters_y);
    rotate_scale(p, [0.5; 3], &matrix, [1.0; 3])
}

/// The four `(x_facing, y_facing, z_facing)` source-index triples for one output facing,
/// each entry naming which of the six `positions_div16` slots to read. java:
/// `EnumFaceDirection`'s per-facing vertex tables.
fn face_vertex_table(facing: Facing) -> [(Facing, Facing, Facing); 4] {
    use Facing::{Down, East, North, South, Up, West};
    match facing {
        Down => [
            (West, Down, South),
            (West, Down, North),
            (East, Down, North),
            (East, Down, South),
        ],
        Up => [
            (West, Up, North),
            (West, Up, South),
            (East, Up, South),
            (East, Up, North),
        ],
        North => [
            (East, Up, North),
            (East, Down, North),
            (West, Down, North),
            (West, Up, North),
        ],
        South => [
            (West, Up, South),
            (West, Down, South),
            (East, Down, South),
            (East, Up, South),
        ],
        West => [
            (West, Up, North),
            (West, Down, North),
            (West, Down, South),
            (West, Up, South),
        ],
        East => [
            (East, Up, South),
            (East, Down, South),
            (East, Down, North),
            (East, Up, North),
        ],
    }
}

/// `from`/`to` (model-space `0..16`) as the six-slot array `FaceBakery.getPositionsDiv16`
/// builds, indexed by `Facing::index()`.
fn positions_div16(from: [f32; 3], to: [f32; 3]) -> [f32; 6] {
    let mut arr = [0.0f32; 6];
    arr[Facing::West.index() as usize] = from[0] / 16.0;
    arr[Facing::Down.index() as usize] = from[1] / 16.0;
    arr[Facing::North.index() as usize] = from[2] / 16.0;
    arr[Facing::East.index() as usize] = to[0] / 16.0;
    arr[Facing::Up.index() as usize] = to[1] / 16.0;
    arr[Facing::South.index() as usize] = to[2] / 16.0;
    arr
}

/// The element's default per-face UV rect when the JSON face has no explicit `"uv"` (java:
/// `BlockPart.getFaceUvs`).
fn default_face_uv(facing: Facing, from: [f32; 3], to: [f32; 3]) -> [f32; 4] {
    match facing {
        Facing::Down | Facing::Up => [from[0], from[2], to[0], to[2]],
        Facing::North | Facing::South => [from[0], 16.0 - to[1], to[0], 16.0 - from[1]],
        Facing::West | Facing::East => [from[2], 16.0 - to[1], to[2], 16.0 - from[1]],
    }
}

/// One UV corner of a (possibly `rotation`-shifted) face rect (java: `BlockFaceUV`).
fn uv_corner(uvs: [f32; 4], rotation_deg: i32, vertex_index: usize) -> (f32, f32) {
    let i = (vertex_index as i32 + rotation_deg / 90).rem_euclid(4) as usize;
    let u = if i != 0 && i != 1 { uvs[2] } else { uvs[0] };
    let v = if i != 0 && i != 3 { uvs[3] } else { uvs[1] };
    (u, v)
}

/// Where `uvlock` stores the UV it derived from the vertex sitting in slot `vertex_index`.
/// java: `BlockFaceUV.func_178345_c` — `(i + (4 - rotation / 90)) % 4`, the inverse of the
/// shift [`uv_corner`] applies, so a face that is both `uvlock`ed and `"rotation"`-shifted
/// lands the same way vanilla lands it.
fn uv_store_slot(rotation_deg: i32, vertex_index: usize) -> usize {
    (vertex_index as i32 + (4 - rotation_deg / 90)).rem_euclid(4) as usize
}

/// The storage slot a vertex built at index `vertex_index` ends up in, for a quad on `facing`
/// baked under an `x`/`y` model rotation. java: `ModelRotation.rotateVertex`.
///
/// Always `(vertex_index + k) % 4` for a `k` shared by all four vertices of one quad, so it is
/// a cyclic relabelling and never changes the winding — see the module docs.
#[must_use]
pub fn rotate_vertex_slot(facing: Facing, vertex_index: usize, rot: BakeRotation) -> usize {
    if rot.quarters_x == 0 && rot.quarters_y == 0 {
        // java: FaceBakery.rotateVertex short-circuits on ModelRotation.X0_Y0.
        return vertex_index;
    }
    let qx = usize::from(rot.quarters_x % 4);
    let qy = usize::from(rot.quarters_y % 4);
    let mut i = vertex_index;
    if facing.axis() == Axis::X {
        i = (i + qx) % 4;
    }
    let mut f = facing;
    for _ in 0..qx {
        f = f.rotate_around(Axis::X);
    }
    if f.axis() == Axis::Y {
        i = (i + qy) % 4;
    }
    i
}

/// java: `MathHelper.epsilonEquals` — `abs(b - a) < 1.0E-5`.
fn epsilon_equals(a: f32, b: f32) -> bool {
    (b - a).abs() < 1.0e-5
}

/// java: `FaceBakery.getFacingFromVertexData` — the facing whose direction vector has the
/// largest positive dot with the quad's own normal, computed from slots 0/1/2.
///
/// Vanilla falls back to `UP` when no facing scores a positive dot, which for a real quad only
/// happens if the cross product is degenerate (a zero-area quad makes it `NaN`); this returns
/// `None` there and lets the caller decide, so the fallback is visible rather than silent.
#[must_use]
pub fn facing_from_vertex_data(verts: &[Vertex; 4]) -> Option<Facing> {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let d0 = sub(verts[0].pos, verts[1].pos);
    let d1 = sub(verts[2].pos, verts[1].pos);
    // java: Vector3f.cross(vector3f4, vector3f3, ..) — (v2 - v1) x (v0 - v1).
    let n = [
        d1[1] * d0[2] - d1[2] * d0[1],
        d1[2] * d0[0] - d1[0] * d0[2],
        d1[0] * d0[1] - d1[1] * d0[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len == 0.0 || !len.is_finite() {
        return None;
    }
    let n = [n[0] / len, n[1] / len, n[2] / len];
    let mut best: Option<Facing> = None;
    let mut best_dot = 0.0f32;
    for f in Facing::VALUES {
        let (dx, dy, dz) = f.offset();
        let dot = n[0] * dx as f32 + n[1] * dy as f32 + n[2] * dz as f32;
        if dot >= 0.0 && dot > best_dot {
            best_dot = dot;
            best = Some(f);
        }
    }
    best
}

/// java: `FaceBakery.applyFacing` — rebuild every slot's position from `facing`'s
/// `EnumFaceDirection` table over the quad's own bounding box, then re-pair each slot's UV to
/// its new position by searching the pre-canonicalisation vertices.
///
/// Vanilla runs this only when the element has no `rotation` of its own, and it is what makes a
/// rotated variant's slot order identical to an unrotated one's. A slot whose rebuilt position
/// matches no original vertex keeps whatever UV it already had, exactly as java does (it only
/// writes the UV words inside the `epsilonEquals` branch).
fn apply_facing(verts: &mut [Vertex; 4], facing: Facing) {
    let original = *verts;
    // java: afloat[] seeded with 999/-999 per EnumFaceDirection.Constants index.
    let mut bounds = [0.0f32; 6];
    bounds[Facing::West.index() as usize] = 999.0;
    bounds[Facing::Down.index() as usize] = 999.0;
    bounds[Facing::North.index() as usize] = 999.0;
    bounds[Facing::East.index() as usize] = -999.0;
    bounds[Facing::Up.index() as usize] = -999.0;
    bounds[Facing::South.index() as usize] = -999.0;
    for v in &original {
        bounds[Facing::West.index() as usize] = bounds[Facing::West.index() as usize].min(v.pos[0]);
        bounds[Facing::Down.index() as usize] = bounds[Facing::Down.index() as usize].min(v.pos[1]);
        bounds[Facing::North.index() as usize] =
            bounds[Facing::North.index() as usize].min(v.pos[2]);
        bounds[Facing::East.index() as usize] = bounds[Facing::East.index() as usize].max(v.pos[0]);
        bounds[Facing::Up.index() as usize] = bounds[Facing::Up.index() as usize].max(v.pos[1]);
        bounds[Facing::South.index() as usize] =
            bounds[Facing::South.index() as usize].max(v.pos[2]);
    }

    for (i, (xf, yf, zf)) in face_vertex_table(facing).into_iter().enumerate() {
        let p = [
            bounds[xf.index() as usize],
            bounds[yf.index() as usize],
            bounds[zf.index() as usize],
        ];
        verts[i].pos = p;
        for o in &original {
            if epsilon_equals(p[0], o.pos[0])
                && epsilon_equals(p[1], o.pos[1])
                && epsilon_equals(p[2], o.pos[2])
            {
                verts[i].uv = o.uv;
            }
        }
    }
}

/// Re-derives UV purely from a vertex's final world-space position, so a rotated model keeps
/// its texture aligned to world space instead of spinning with it (java:
/// `FaceBakery.lockVertexUv`).
///
/// `pub` (rather than the `bake_quad`-private helper it started as) because a second caller
/// needs the exact same formula: java's `BreakingFour.remapVert` — the block-breaking overlay's
/// per-quad texture substitution (`SimpleBakedModel.Builder(bakedModel, destroyStageSprite)`) —
/// derives each vertex's crack-texture UV this same position-only way, **not** from the quad's
/// original texture mapping. That is `mc-render`'s [`crate::model`]-facing half of the
/// block-breaking overlay: the natural-sounding alternative design (store each
/// quad's original atlas sprite rect and re-normalise its baked UV through it) is not what
/// vanilla does and would disagree with it for any quad whose face has a non-default `"uv"`
/// (common — stair/slab/fence side faces routinely map a sub-rect of their texture), whereas
/// reusing this exact function is both simpler and provably matches java's own algorithm.
///
/// This wrapping behavior is specific to `FaceBakery.lockVertexUv`; use
/// [`breaking_vertex_uv`] for `BreakingFour.remapVert`.
#[must_use]
pub fn lock_vertex_uv(pos: [f32; 3], final_facing: Facing) -> (f32, f32) {
    let wrap = |mut f: f32| -> f32 {
        if !(-0.1..1.1).contains(&f) {
            f -= f.floor();
        }
        f
    };
    vertex_uv([wrap(pos[0]), wrap(pos[1]), wrap(pos[2])], final_facing)
}

/// Projects a baked vertex onto the block-breaking texture. java:
/// `BreakingFour.remapVert`.
///
/// It shares FaceBakery's six-face projection table but deliberately does *not* wrap oversized
/// coordinates: `BreakingFour` does not call `lockVertexUv`.
#[must_use]
pub fn breaking_vertex_uv(pos: [f32; 3], final_facing: Facing) -> (f32, f32) {
    vertex_uv(pos, final_facing)
}

/// The six-face position-to-UV projection table shared by `lockVertexUv` and
/// `BreakingFour.remapVert`.
fn vertex_uv([f, f1, f2]: [f32; 3], final_facing: Facing) -> (f32, f32) {
    match final_facing {
        Facing::Down => (f * 16.0, (1.0 - f2) * 16.0),
        Facing::Up => (f * 16.0, f2 * 16.0),
        Facing::North => ((1.0 - f) * 16.0, (1.0 - f1) * 16.0),
        Facing::South => (f * 16.0, (1.0 - f1) * 16.0),
        Facing::West => (f2 * 16.0, (1.0 - f1) * 16.0),
        Facing::East => ((1.0 - f2) * 16.0, (1.0 - f1) * 16.0),
    }
}

/// Bakes one `elements[].faces.<name>` entry into a [`BakedQuad`].
///
/// `facing` is the JSON face key (`"down"`, `"north"`, ...). `sprite_uv` is the texture's
/// rect within the atlas — [`BlockPartFace::texture`](RawFace::texture) must already have
/// been resolved to a concrete [`crate::ResourceLocation`] and looked up before calling this.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn bake_quad(
    element: &RawElement,
    facing: Facing,
    face: &RawFace,
    sprite_uv: UvRect,
    rot: BakeRotation,
) -> BakedQuad {
    let positions = positions_div16(element.from, element.to);
    // java: `fillVertexData` derives the baked shade colour from
    // `modelRotation.rotateFace(jsonFace)`, before an element-local rotation changes the
    // geometry. `makeBakedQuad` below separately stores `getFacingFromVertexData`.
    let shade_facing = rotate_face(facing, rot.quarters_x, rot.quarters_y);
    let uv_rect = face
        .uv
        .unwrap_or_else(|| default_face_uv(facing, element.from, element.to));

    let table = face_vertex_table(facing);

    // java: makeQuadVertexData -> fillVertexData. Position and UV both come from the *build*
    // index `i`, but land together in the rotated storage slot `rotate_vertex_slot(i)`.
    let mut verts = [Vertex::default(); 4];
    for (i, (xf, yf, zf)) in table.into_iter().enumerate() {
        let mut p = [
            positions[xf.index() as usize],
            positions[yf.index() as usize],
            positions[zf.index() as usize],
        ];
        if let Some(part_rot) = &element.rotation {
            p = apply_part_rotation(p, part_rot);
        }
        p = apply_model_rotation(p, rot);
        let (u, v) = uv_corner(uv_rect, face.rotation, i);
        verts[rotate_vertex_slot(facing, i, rot)] = Vertex {
            pos: p,
            uv: [sprite_uv.interpolate_u(u), sprite_uv.interpolate_v(v)],
        };
    }

    // java: `EnumFacing enumfacing = getFacingFromVertexData(aint)` — the *geometric* facing,
    // which is what lockUv and applyFacing are keyed on. It equals `shade_facing` for every
    // element without its own `rotation` (asserted by a test); the fallback for a degenerate
    // quad is java's `EnumFacing.UP`.
    let geometric_facing = facing_from_vertex_data(&verts).unwrap_or(Facing::Up);

    // java: `if (uvLocked) this.lockUv(..)` — note it reads each *slot's* final position and
    // writes to `func_178345_c(slot)`, so it has to run over the whole quad at once.
    if rot.uvlock {
        let mut locked = [[0.0f32; 2]; 4];
        for i in 0..4 {
            let (u, v) = lock_vertex_uv(verts[i].pos, geometric_facing);
            locked[uv_store_slot(face.rotation, i)] =
                [sprite_uv.interpolate_u(u), sprite_uv.interpolate_v(v)];
        }
        for (v, uv) in verts.iter_mut().zip(locked) {
            v.uv = uv;
        }
    }

    // java: `if (partRotation == null) this.applyFacing(aint, enumfacing)`.
    if element.rotation.is_none() {
        apply_facing(&mut verts, geometric_facing);
    }

    BakedQuad {
        verts,
        // java: `new BakedQuad(..., getFacingFromVertexData(aint))`, including the UP fallback
        // for a degenerate cross product. This is the face BlockModelRenderer uses for both
        // fillQuadBounds and AmbientOcclusionFace's neighbour table.
        face: Some(geometric_facing),
        shade_face: Some(shade_facing),
        tint_index: (face.tintindex >= 0).then_some(face.tintindex as u8),
        shade: element.shade,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn full_cube_element() -> RawElement {
        let mut faces = HashMap::new();
        for name in ["down", "up", "north", "south", "west", "east"] {
            faces.insert(
                name.to_string(),
                RawFace {
                    uv: None,
                    texture: "#all".to_string(),
                    cullface: Some(name.to_string()),
                    rotation: 0,
                    tintindex: -1,
                },
            );
        }
        RawElement {
            from: [0.0, 0.0, 0.0],
            to: [16.0, 16.0, 16.0],
            rotation: None,
            faces,
            shade: true,
        }
    }

    const IDENTITY_UV: UvRect = UvRect {
        min_u: 0.0,
        max_u: 1.0,
        min_v: 0.0,
        max_v: 1.0,
    };

    #[test]
    fn full_cube_up_face_spans_the_whole_texture() {
        let el = full_cube_element();
        let f = el.faces["up"].clone();
        let q = bake_quad(&el, Facing::Up, &f, IDENTITY_UV, BakeRotation::default());
        assert_eq!(q.face, Some(Facing::Up));
        for v in &q.verts {
            assert!(
                (v.pos[1] - 1.0).abs() < 1e-6,
                "up face must sit at y=1: {v:?}"
            );
        }
        let us: Vec<f32> = q.verts.iter().map(|v| v.uv[0]).collect();
        let vs: Vec<f32> = q.verts.iter().map(|v| v.uv[1]).collect();
        assert!(us.iter().any(|&u| (u - 0.0).abs() < 1e-6));
        assert!(us.iter().any(|&u| (u - 1.0).abs() < 1e-6));
        assert!(vs.iter().any(|&v| (v - 0.0).abs() < 1e-6));
        assert!(vs.iter().any(|&v| (v - 1.0).abs() < 1e-6));
    }

    /// y:90 is documented as clockwise viewed from above: a quad on the "north" face (facing
    /// out toward -Z) must end up facing "east" (+X) after the rotation.
    #[test]
    fn y90_rotates_clockwise_from_above() {
        assert_eq!(rotate_face(Facing::North, 0, 1), Facing::East);
        assert_eq!(rotate_face(Facing::East, 0, 1), Facing::South);
        assert_eq!(rotate_face(Facing::South, 0, 1), Facing::West);
        assert_eq!(rotate_face(Facing::West, 0, 1), Facing::North);
        assert_eq!(rotate_face(Facing::Up, 0, 1), Facing::Up);
        assert_eq!(rotate_face(Facing::Down, 0, 1), Facing::Down);
    }

    #[test]
    fn y90_moves_geometry_consistently_with_facing() {
        let el = full_cube_element();
        let f = el.faces["north"].clone();
        let rot = BakeRotation {
            quarters_x: 0,
            quarters_y: 1,
            uvlock: false,
        };
        let q = bake_quad(&el, Facing::North, &f, IDENTITY_UV, rot);
        assert_eq!(q.face, Some(Facing::East));
        for v in &q.verts {
            assert!(
                (v.pos[0] - 1.0).abs() < 1e-5,
                "rotated face must sit at x=1: {v:?}"
            );
        }
    }

    #[test]
    fn x90_matches_mc_math_rotate_around() {
        // mc_math::Facing::rotate_around(Axis::X) (ported from and differential-tested
        // against the real EnumFacing.rotateX) is the ground truth: North->Down, Up->North.
        // This checks the *geometric* rotation matrix agrees with it — if the two ever
        // disagree, a quad's `face` label would point somewhere its vertices don't.
        assert_eq!(rotate_face(Facing::Up, 1, 0), Facing::North);
        assert_eq!(rotate_face(Facing::North, 1, 0), Facing::Down);
        assert_eq!(Facing::North.rotate_around(Axis::X), Facing::Down);
        assert_eq!(Facing::Up.rotate_around(Axis::X), Facing::North);
    }

    /// Cross-checks the geometric position formula (`model_rotation_matrix`, used by
    /// `apply_model_rotation`) against `rotate_face` (which just wraps
    /// `mc_math::Facing::rotate_around`) for every axis/quarter combination: rotating a
    /// direction vector must land on the same facing as rotating the label. A mismatch here
    /// would mean a baked quad's `face` disagrees with where its vertices actually ended up.
    #[test]
    fn geometry_and_facing_rotation_agree_on_every_step() {
        fn direction_vec(f: Facing) -> [f32; 3] {
            let (dx, dy, dz) = f.offset();
            [dx as f32, dy as f32, dz as f32]
        }
        fn facing_from_vec(v: [f32; 3]) -> Facing {
            [
                Facing::Down,
                Facing::Up,
                Facing::North,
                Facing::South,
                Facing::West,
                Facing::East,
            ]
            .into_iter()
            .max_by(|a, b| {
                let da = direction_vec(*a);
                let db = direction_vec(*b);
                let dot_a = da[0] * v[0] + da[1] * v[1] + da[2] * v[2];
                let dot_b = db[0] * v[0] + db[1] * v[1] + db[2] * v[2];
                dot_a.partial_cmp(&dot_b).unwrap()
            })
            .unwrap()
        }

        const ORIGIN: [f32; 3] = [0.0, 0.0, 0.0];
        for facing in [
            Facing::Down,
            Facing::Up,
            Facing::North,
            Facing::South,
            Facing::West,
            Facing::East,
        ] {
            for qx in 0u8..4 {
                for qy in 0u8..4 {
                    let rot = BakeRotation {
                        quarters_x: qx,
                        quarters_y: qy,
                        uvlock: false,
                    };
                    let by_geometry = {
                        let v = direction_vec(facing);
                        let matrix = model_rotation_matrix(qx, qy);
                        facing_from_vec(rotate_scale(v, ORIGIN, &matrix, [1.0; 3]))
                    };
                    let by_label = rotate_face(facing, rot.quarters_x, rot.quarters_y);
                    assert_eq!(by_geometry, by_label, "facing={facing:?} qx={qx} qy={qy}");
                }
            }
        }
    }

    #[test]
    fn uvlock_keeps_texture_aligned_to_world_space() {
        let el = full_cube_element();
        let f = el.faces["up"].clone();
        let rot_locked = BakeRotation {
            quarters_x: 0,
            quarters_y: 1,
            uvlock: true,
        };
        let rot_unlocked = BakeRotation {
            quarters_x: 0,
            quarters_y: 1,
            uvlock: false,
        };
        let locked = bake_quad(&el, Facing::Up, &f, IDENTITY_UV, rot_locked);
        let unlocked = bake_quad(&el, Facing::Up, &f, IDENTITY_UV, rot_unlocked);
        // Both must still cover the full [0,1] UV square (a locked *and* rotated square top
        // face just gets its corners relabelled), but the corner->uv assignment must differ
        // between the two for at least one vertex, proving uvlock actually re-derives UVs.
        let locked_uv: Vec<[f32; 2]> = locked.verts.iter().map(|v| v.uv).collect();
        let unlocked_uv: Vec<[f32; 2]> = unlocked.verts.iter().map(|v| v.uv).collect();
        assert_ne!(locked_uv, unlocked_uv);
    }

    /// `FaceBakery.lockVertexUv` wraps outside its tolerance window; `BreakingFour.remapVert`
    /// uses the same table without that step. The oversized position is the negative control.
    #[test]
    fn breaking_uv_does_not_inherit_lock_uv_coordinate_wrapping() {
        let pos = [1.2, 0.5, -0.25];
        let locked = lock_vertex_uv(pos, Facing::Up);
        let breaking = breaking_vertex_uv(pos, Facing::Up);
        assert!((locked.0 - 3.2).abs() < 1.0e-5 && (locked.1 - 12.0).abs() < 1.0e-5);
        assert!((breaking.0 - 19.2).abs() < 1.0e-5 && (breaking.1 + 4.0).abs() < 1.0e-5);
        assert_ne!(locked, breaking);

        let in_range = [0.75, 0.5, 0.25];
        assert_eq!(
            lock_vertex_uv(in_range, Facing::North),
            breaking_vertex_uv(in_range, Facing::North)
        );
    }

    /// java: `ModelRotation.rotateVertex` returns `(vertexIndex + k) % 4` for one `k` shared by
    /// the whole quad. Asserting the *shape* rather than a table of values is what makes the
    /// "a cyclic shift cannot flip the winding" argument in the module docs a checked fact.
    #[test]
    fn rotate_vertex_slot_is_one_cyclic_shift_for_the_whole_quad() {
        for facing in Facing::VALUES {
            for qx in 0u8..4 {
                for qy in 0u8..4 {
                    let rot = BakeRotation {
                        quarters_x: qx,
                        quarters_y: qy,
                        uvlock: false,
                    };
                    let k = (rotate_vertex_slot(facing, 0, rot) + 4) % 4;
                    for i in 0..4 {
                        assert_eq!(
                            rotate_vertex_slot(facing, i, rot),
                            (i + k) % 4,
                            "facing={facing:?} qx={qx} qy={qy} i={i}"
                        );
                    }
                }
            }
        }
        // And it is a permutation, never a collision.
        for facing in Facing::VALUES {
            let rot = BakeRotation {
                quarters_x: 1,
                quarters_y: 3,
                uvlock: false,
            };
            let mut seen = [false; 4];
            for i in 0..4 {
                seen[rotate_vertex_slot(facing, i, rot)] = true;
            }
            assert!(seen.iter().all(|b| *b), "facing={facing:?}");
        }
    }

    /// The decisive regression for the S11 -> S09 request: a rotated variant must store its
    /// vertices in the *same* slot order as the unrotated model it rotates onto.
    ///
    /// A full cube's top face is invariant under `y: 180` as a set of four points, so the two
    /// bakes must agree slot for slot. Before `rotateVertex` + `applyFacing` landed, the
    /// rotated bake's slot 0 held the corner that belongs in slot 2 — the failure mode vanilla's
    /// fixed `VertexTranslations` AO table would have read straight through.
    #[test]
    fn a_rotated_variant_stores_the_same_slot_order_as_the_unrotated_one() {
        let el = full_cube_element();
        let cases: [(&str, Facing, u8, u8); 4] = [
            // (face key, its facing, qx, qy) — each pair maps the face onto itself.
            ("up", Facing::Up, 0, 2),
            ("down", Facing::Down, 0, 2),
            ("east", Facing::East, 1, 0),
            ("west", Facing::West, 2, 0),
        ];
        for (name, facing, qx, qy) in cases {
            let f = el.faces[name].clone();
            let plain = bake_quad(&el, facing, &f, IDENTITY_UV, BakeRotation::default());
            let rot = BakeRotation {
                quarters_x: qx,
                quarters_y: qy,
                uvlock: false,
            };
            let rotated = bake_quad(&el, facing, &f, IDENTITY_UV, rot);
            assert_eq!(
                rotated.face, plain.face,
                "{name} qx={qx} qy={qy} should map onto itself"
            );
            for i in 0..4 {
                for c in 0..3 {
                    assert!(
                        (rotated.verts[i].pos[c] - plain.verts[i].pos[c]).abs() < 1e-5,
                        "{name} qx={qx} qy={qy}: slot {i} is {:?}, expected {:?}",
                        rotated.verts[i].pos,
                        plain.verts[i].pos
                    );
                }
            }
        }

        // ...and the test has teeth: the pre-fix layout (rotate each table entry in place and
        // keep it at its build index) really did differ. For `up` under `y: 180`, build index 0
        // is `(West, Up, North)` = (0,1,0), which the rotation carries to (1,1,1) — so a bake
        // that skipped `rotateVertex`/`applyFacing` put the *south-east* corner in slot 0 where
        // vanilla puts the north-west one. Asserting the wrong answer is absent is the same
        // discipline S11's AO tests use.
        let f = el.faces["up"].clone();
        let rotated = bake_quad(
            &el,
            Facing::Up,
            &f,
            IDENTITY_UV,
            BakeRotation {
                quarters_x: 0,
                quarters_y: 2,
                uvlock: false,
            },
        );
        let naive = apply_model_rotation(
            [0.0, 1.0, 0.0],
            BakeRotation {
                quarters_x: 0,
                quarters_y: 2,
                uvlock: false,
            },
        );
        assert!(
            (naive[0] - 1.0).abs() < 1e-5 && (naive[2] - 1.0).abs() < 1e-5,
            "sanity: y:180 must carry (0,1,0) to (1,1,1), got {naive:?}"
        );
        assert!(
            (rotated.verts[0].pos[0] - naive[0]).abs() > 0.5,
            "slot 0 must not be the rotate-in-place answer {naive:?}"
        );
    }

    /// java's `getFacingFromVertexData` and `ModelRotation.rotateFace` must agree wherever
    /// vanilla uses the geometric one to canonicalise — i.e. for every element without its own
    /// `rotation`. `BakedQuad::face` is the geometric result in both cases.
    #[test]
    fn the_geometric_and_label_facings_agree_for_unrotated_elements() {
        let el = full_cube_element();
        for name in ["down", "up", "north", "south", "west", "east"] {
            let facing = super::super::blockmodel::parse_facing(name).unwrap();
            let f = el.faces[name].clone();
            for qx in 0u8..4 {
                for qy in 0u8..4 {
                    let rot = BakeRotation {
                        quarters_x: qx,
                        quarters_y: qy,
                        uvlock: false,
                    };
                    let q = bake_quad(&el, facing, &f, IDENTITY_UV, rot);
                    assert_eq!(
                        facing_from_vertex_data(&q.verts),
                        q.face,
                        "{name} qx={qx} qy={qy}"
                    );
                }
            }
        }
    }

    /// Winding: every baked quad's first triangle (`0,1,2` — the index pattern S11 uploads)
    /// must be counter-clockwise *seen from outside*, i.e. its normal must point along the
    /// quad's own facing. This is the CPU half of the evidence for turning terrain back-face
    /// culling on; the GPU half is in `mc-render`'s `s11_cull.rs`.
    /// `FaceBakery.makeBakedQuad` stores `getFacingFromVertexData(aint)`, not the face label
    /// that `fillVertexData` passes to `ModelRotation.rotateFace` for its shade byte. A 45-degree
    /// X rotation of south is the tie-sensitive negative control: its normal is equally down
    /// and south, and Java's D-U-N-S-W-E scan keeps the earlier `DOWN` tie winner. A
    /// label-derived implementation would incorrectly retain `SOUTH` and would also lock UVs
    /// with SOUTH's projection rather than DOWN's.
    #[test]
    fn rotated_element_uses_geometric_face_not_label_for_quad_and_uvlock() {
        let mut el = full_cube_element();
        el.rotation = Some(RawRotation {
            origin: [8.0, 8.0, 8.0],
            axis: "x".to_string(),
            angle: 45.0,
            rescale: false,
        });
        let f = el.faces["south"].clone();
        let q = bake_quad(
            &el,
            Facing::South,
            &f,
            IDENTITY_UV,
            BakeRotation {
                quarters_x: 0,
                quarters_y: 0,
                uvlock: true,
            },
        );

        assert_eq!(q.face, Some(Facing::Down));
        assert_eq!(q.shade_face, Some(Facing::South));
        assert_ne!(q.face, q.shade_face, "negative control: the label is SOUTH");
        assert_eq!(facing_from_vertex_data(&q.verts), q.face);

        let mut differs_from_label_projection = false;
        for vertex in q.verts {
            let (u, v) = lock_vertex_uv(vertex.pos, Facing::Down);
            let expected = [IDENTITY_UV.interpolate_u(u), IDENTITY_UV.interpolate_v(v)];
            assert!(
                (vertex.uv[0] - expected[0]).abs() < 1e-6
                    && (vertex.uv[1] - expected[1]).abs() < 1e-6,
                "UV must use the geometric DOWN projection: {vertex:?} expected {expected:?}"
            );
            let (label_u, label_v) = lock_vertex_uv(vertex.pos, Facing::South);
            let label = [
                IDENTITY_UV.interpolate_u(label_u),
                IDENTITY_UV.interpolate_v(label_v),
            ];
            differs_from_label_projection |=
                (vertex.uv[0] - label[0]).abs() > 1e-6 || (vertex.uv[1] - label[1]).abs() > 1e-6;
        }
        assert!(
            differs_from_label_projection,
            "negative control must distinguish DOWN's geometric UV projection from SOUTH's label"
        );
    }

    #[test]
    fn every_baked_quad_winds_counter_clockwise_about_its_own_facing() {
        let el = full_cube_element();
        for name in ["down", "up", "north", "south", "west", "east"] {
            let facing = super::super::blockmodel::parse_facing(name).unwrap();
            let f = el.faces[name].clone();
            for qx in 0u8..4 {
                for qy in 0u8..4 {
                    let rot = BakeRotation {
                        quarters_x: qx,
                        quarters_y: qy,
                        uvlock: false,
                    };
                    let q = bake_quad(&el, facing, &f, IDENTITY_UV, rot);
                    let v = &q.verts;
                    let a = [
                        v[1].pos[0] - v[0].pos[0],
                        v[1].pos[1] - v[0].pos[1],
                        v[1].pos[2] - v[0].pos[2],
                    ];
                    let b = [
                        v[2].pos[0] - v[0].pos[0],
                        v[2].pos[1] - v[0].pos[1],
                        v[2].pos[2] - v[0].pos[2],
                    ];
                    let n = [
                        a[1] * b[2] - a[2] * b[1],
                        a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0],
                    ];
                    let (dx, dy, dz) = q.face.unwrap().offset();
                    let dot = n[0] * dx as f32 + n[1] * dy as f32 + n[2] * dz as f32;
                    assert!(
                        dot > 0.5,
                        "{name} qx={qx} qy={qy}: normal {n:?} does not point along {:?} (dot {dot})",
                        q.face
                    );
                }
            }
        }
    }

    #[test]
    fn cube_stays_within_unit_bounds_after_rotation() {
        let el = full_cube_element();
        for &(qx, qy) in &[(0u8, 0u8), (0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0)] {
            for face_name in ["down", "up", "north", "south", "west", "east"] {
                let f = el.faces[face_name].clone();
                let facing = super::super::blockmodel::parse_facing(face_name).unwrap();
                let rot = BakeRotation {
                    quarters_x: qx,
                    quarters_y: qy,
                    uvlock: false,
                };
                let q = bake_quad(&el, facing, &f, IDENTITY_UV, rot);
                for v in &q.verts {
                    for c in v.pos {
                        assert!(
                            (-1e-4..=1.0001).contains(&c),
                            "{qx},{qy},{face_name}: {v:?}"
                        );
                    }
                }
            }
        }
    }
}
