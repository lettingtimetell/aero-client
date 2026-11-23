//! Smooth lighting and ambient occlusion — a term-for-term port of vanilla's
//! `BlockModelRenderer.AmbientOcclusionFace`.
//!
//! **Hazard H8.** This is the part of the mesher that cannot be approximated. Each of a
//! quad's four vertices is lit from the three blocks touching that corner plus the block the
//! face points into, and the corner sample is *skipped* when both of the side blocks already
//! block light — vanilla's `if (!flag2 && !flag)` arms below. Getting that wrong produces
//! terrain that looks plausible in a screenshot and shimmers wrongly the moment the camera
//! moves, because the error is per-corner rather than per-face.
//!
//! Two paths exist, exactly as in vanilla:
//!
//! * **Full-face** (`bounds_flags[1]` clear, or a top/bottom face): the four corner values are
//!   used directly.
//! * **Partial-face** (`bounds_flags[1]` set on a side face): the four corner values are
//!   bilinearly weighted by where the quad actually sits inside the block, using the
//!   `Orientation` lookup tables. This is what makes a torch, a fence post or a half-slab pick
//!   up the light of the block it is embedded in rather than the corners of the full cube.
//
// java: client/renderer/BlockModelRenderer.java:359-532 (AmbientOcclusionFace),
//       :534-599 (EnumNeighborInfo, Orientation), :601-637 (VertexTranslations)

use mc_assets::model::{BakedQuad, Vertex};
use mc_data::{block_props, BlockState};
use mc_math::{BlockPos, Facing};
use mc_world::World;

use super::cull_face::MeshSettings;
use super::light::mixed_brightness;

// ---- Orientation indices into `quad_bounds` ------------------------------------------------
//
// java: Orientation.field_178229_m = facing.getIndex() + (flip ? 6 : 0)

const D: usize = 0;
const U: usize = 1;
const N: usize = 2;
const S: usize = 3;
const W: usize = 4;
const E: usize = 5;
const FD: usize = 6;
const FU: usize = 7;
const FN: usize = 8;
const FS: usize = 9;
const FW: usize = 10;
const FE: usize = 11;

/// Per-face neighbour tables. java: `BlockModelRenderer.EnumNeighborInfo`.
struct NeighborInfo {
    /// The four in-plane neighbours, in vanilla's order. java: `field_178276_g`.
    neighbors: [Facing; 4],
    /// Whether the partial-face interpolation applies to this face. Only the four side faces
    /// carry orientation tables; `DOWN`/`UP` always take the flat path.
    /// java: `field_178289_i`.
    interpolated: bool,
    /// Orientation pairs for vertex 0..3. java: `field_178286_j`, `_k`, `_l`, `_m`.
    orientations: [[usize; 8]; 4],
}

const EMPTY: [[usize; 8]; 4] = [[0; 8]; 4];

/// Indexed by [`Facing::index`] (DOWN, UP, NORTH, SOUTH, WEST, EAST).
static NEIGHBOR_INFO: [NeighborInfo; 6] = [
    // DOWN
    NeighborInfo {
        neighbors: [Facing::West, Facing::East, Facing::North, Facing::South],
        interpolated: false,
        orientations: EMPTY,
    },
    // UP
    NeighborInfo {
        neighbors: [Facing::East, Facing::West, Facing::North, Facing::South],
        interpolated: false,
        orientations: EMPTY,
    },
    // NORTH
    NeighborInfo {
        neighbors: [Facing::Up, Facing::Down, Facing::East, Facing::West],
        interpolated: true,
        orientations: [
            [U, FW, U, W, FU, W, FU, FW],
            [U, FE, U, E, FU, E, FU, FE],
            [D, FE, D, E, FD, E, FD, FE],
            [D, FW, D, W, FD, W, FD, FW],
        ],
    },
    // SOUTH
    NeighborInfo {
        neighbors: [Facing::West, Facing::East, Facing::Down, Facing::Up],
        interpolated: true,
        orientations: [
            [U, FW, FU, FW, FU, W, U, W],
            [D, FW, FD, FW, FD, W, D, W],
            [D, FE, FD, FE, FD, E, D, E],
            [U, FE, FU, FE, FU, E, U, E],
        ],
    },
    // WEST
    NeighborInfo {
        neighbors: [Facing::Up, Facing::Down, Facing::North, Facing::South],
        interpolated: true,
        orientations: [
            [U, S, U, FS, FU, FS, FU, S],
            [U, N, U, FN, FU, FN, FU, N],
            [D, N, D, FN, FD, FN, FD, N],
            [D, S, D, FS, FD, FS, FD, S],
        ],
    },
    // EAST
    NeighborInfo {
        neighbors: [Facing::Down, Facing::Up, Facing::North, Facing::South],
        interpolated: true,
        orientations: [
            [FD, S, FD, FS, D, FS, D, S],
            [FD, N, FD, FN, D, FN, D, N],
            [FU, N, FU, FN, U, FN, U, N],
            [FU, S, FU, FS, U, FS, U, S],
        ],
    },
];

/// Which quad vertex each of the four computed corners lands on, per face.
/// java: `BlockModelRenderer.VertexTranslations`.
///
/// This is vanilla's fixed table, and it is correct **only** for a quad whose four vertices
/// are stored in `EnumFaceDirection`'s canonical order.
///
/// When this was written, `mc_assets`' `FaceBakery` port emitted that order for an unrotated
/// model but reproduced neither `ModelRotation.rotateVertex` nor `FaceBakery.applyFacing`, so a
/// rotated variant's slot `k` did **not** mean the same geometric corner and reading this table
/// straight would have landed the AO on the wrong corners. That is why [`corner_slots`] derives
/// the mapping from the vertex positions instead.
///
/// S09 has since closed the gap (the S11 -> S09 §13 request), so the two conventions now
/// coincide — but the geometric derivation is deliberately kept rather than replaced by the
/// table, because it is also correct for the one case vanilla itself leaves non-canonical: an
/// element with its own `rotation`, where `applyFacing` never runs.
/// `vertex_translations_agree_with_the_geometric_mapping` pins that they agree.
static VERTEX_TRANSLATIONS: [[usize; 4]; 6] = [
    [0, 1, 2, 3], // DOWN
    [2, 3, 0, 1], // UP
    [3, 0, 1, 2], // NORTH
    [0, 1, 2, 3], // SOUTH
    [3, 0, 1, 2], // WEST
    [1, 2, 3, 0], // EAST
];

/// Map each of the four computed corners to the quad vertex that physically sits there.
///
/// Corner `k` is defined by the pair of in-plane neighbours it lies between:
///
/// | slot | lies towards |
/// |---|---|
/// | 0 | `neighbors[0]` and `neighbors[3]` |
/// | 1 | `neighbors[0]` and `neighbors[2]` |
/// | 2 | `neighbors[1]` and `neighbors[2]` |
/// | 3 | `neighbors[1]` and `neighbors[3]` |
///
/// Classifying each vertex by which side of the quad's own centroid it falls on, along the
/// two in-plane axes, reproduces [`VERTEX_TRANSLATIONS`] exactly for a canonically ordered
/// quad (`vertex_translations_agree_with_the_geometric_mapping` proves it) while also
/// surviving a rotated one. Returns `None` when the quad is degenerate along an in-plane
/// axis, in which case the caller falls back to the table.
fn corner_slots(verts: &[Vertex; 4], face: Facing) -> Option<[usize; 4]> {
    let info = &NEIGHBOR_INFO[face.index() as usize];
    let axis_a = info.neighbors[0].direction_vec();
    let axis_b = info.neighbors[2].direction_vec();
    let dot = |v: &Vertex, d: mc_math::BlockPos| {
        v.pos[0] * d.x as f32 + v.pos[1] * d.y as f32 + v.pos[2] * d.z as f32
    };

    let mean_a = verts.iter().map(|v| dot(v, axis_a)).sum::<f32>() * 0.25;
    let mean_b = verts.iter().map(|v| dot(v, axis_b)).sum::<f32>() * 0.25;
    const EPS: f32 = 1.0E-5;

    let mut slots = [usize::MAX; 4];
    for (i, v) in verts.iter().enumerate() {
        let a = dot(v, axis_a) - mean_a;
        let b = dot(v, axis_b) - mean_b;
        if a.abs() < EPS || b.abs() < EPS {
            return None;
        }
        let slot = match (a > 0.0, b > 0.0) {
            (true, false) => 0,
            (true, true) => 1,
            (false, true) => 2,
            (false, false) => 3,
        };
        if slots[slot] != usize::MAX {
            return None; // two vertices in the same corner: not a well-formed quad
        }
        slots[slot] = i;
    }
    Some(slots)
}

/// Vanilla's fixed per-face brightness, applied at bake time in Java and at mesh time here
/// because [`BakedQuad::shade`] is deliberately the raw JSON flag, not a baked colour.
///
/// java: `client/renderer/block/model/FaceBakery.getFaceBrightness`, and the identical values
/// in `EnumNeighborInfo.field_178288_h`.
#[must_use]
pub fn face_brightness(face: Facing) -> f32 {
    match face {
        Facing::Down => 0.5,
        Facing::Up => 1.0,
        Facing::North | Facing::South => 0.8,
        Facing::West | Facing::East => 0.6,
    }
}

/// The shade multiplier for one baked quad: [`face_brightness`] when the model asked to be
/// shaded, `1.0` when it did not (`"shade": false` — torches, redstone dust, ...).
#[must_use]
pub fn quad_shade(quad: &BakedQuad) -> f32 {
    match (quad.shade, quad.shade_face.or(quad.face)) {
        (true, Some(f)) => face_brightness(f),
        _ => 1.0,
    }
}

/// `fillQuadBounds`' output: the quad's extent along each axis, plus its complement.
///
/// Indexed by [`Facing::index`] for the low edge and `index + 6` for `1 - low`; the pairing is
/// what lets the `Orientation` tables express "how far along this axis is the quad" as a plain
/// array lookup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadBounds {
    /// `[west, east, down, up, north, south, 1-west, 1-east, 1-down, 1-up, 1-north, 1-south]`
    /// where `west = min_x`, `east = max_x`, and so on.
    pub bounds: [f32; 12],
    /// `bounds_flags[0]`: the quad is flush against its face, so lighting is sampled from the
    /// *neighbouring* block rather than from this one. java: `boundsFlags.set(0, ...)`.
    pub flush: bool,
    /// `bounds_flags[1]`: the quad does not span the whole face, so the interpolated AO path
    /// applies. java: `boundsFlags.set(1, ...)`.
    pub partial: bool,
}

/// java: `BlockModelRenderer.fillQuadBounds`.
///
/// `full_cube` is the *rendered* block's `Block.isFullCube()`.
#[must_use]
pub fn fill_quad_bounds(quad: &BakedQuad, face: Facing, full_cube: bool) -> QuadBounds {
    let mut min_x = 32.0f32;
    let mut min_y = 32.0f32;
    let mut min_z = 32.0f32;
    let mut max_x = -32.0f32;
    let mut max_y = -32.0f32;
    let mut max_z = -32.0f32;
    for v in &quad.verts {
        min_x = min_x.min(v.pos[0]);
        min_y = min_y.min(v.pos[1]);
        min_z = min_z.min(v.pos[2]);
        max_x = max_x.max(v.pos[0]);
        max_y = max_y.max(v.pos[1]);
        max_z = max_z.max(v.pos[2]);
    }

    let mut bounds = [0.0f32; 12];
    bounds[W] = min_x;
    bounds[E] = max_x;
    bounds[D] = min_y;
    bounds[U] = max_y;
    bounds[N] = min_z;
    bounds[S] = max_z;
    bounds[W + 6] = 1.0 - min_x;
    bounds[E + 6] = 1.0 - max_x;
    bounds[D + 6] = 1.0 - min_y;
    bounds[U + 6] = 1.0 - max_y;
    bounds[N + 6] = 1.0 - min_z;
    bounds[S + 6] = 1.0 - max_z;

    const LO: f32 = 1.0E-4;
    const HI: f32 = 0.9999;

    let (partial, flush) = match face {
        Facing::Down => (
            min_x >= LO || min_z >= LO || max_x <= HI || max_z <= HI,
            (min_y < LO || full_cube) && min_y == max_y,
        ),
        Facing::Up => (
            min_x >= LO || min_z >= LO || max_x <= HI || max_z <= HI,
            (max_y > HI || full_cube) && min_y == max_y,
        ),
        Facing::North => (
            min_x >= LO || min_y >= LO || max_x <= HI || max_y <= HI,
            (min_z < LO || full_cube) && min_z == max_z,
        ),
        Facing::South => (
            min_x >= LO || min_y >= LO || max_x <= HI || max_y <= HI,
            (max_z > HI || full_cube) && min_z == max_z,
        ),
        Facing::West => (
            min_y >= LO || min_z >= LO || max_y <= HI || max_z <= HI,
            (min_x < LO || full_cube) && min_x == max_x,
        ),
        Facing::East => (
            min_y >= LO || min_z >= LO || max_y <= HI || max_z <= HI,
            (max_x > HI || full_cube) && min_x == max_x,
        ),
    };

    QuadBounds {
        bounds,
        flush,
        partial,
    }
}

/// The per-vertex result of the AO pass, in quad-vertex order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AoFace {
    /// The ambient-occlusion multiplier for each vertex, `0.2..=1.0`.
    /// java: `AmbientOcclusionFace.vertexColorMultiplier`.
    pub color_multiplier: [f32; 4],
    /// The packed brightness for each vertex, in [`super::light`]'s layout.
    /// java: `AmbientOcclusionFace.vertexBrightness`.
    pub brightness: [u32; 4],
}

/// java: `AmbientOcclusionFace.getAoBrightness`.
///
/// A zero brightness means "unlit or unloaded"; vanilla substitutes the face's own brightness
/// for it so that a corner poking into an unloaded chunk does not punch a black hole.
#[inline]
fn ao_brightness(br1: u32, br2: u32, br3: u32, br4: u32) -> u32 {
    let br1 = if br1 == 0 { br4 } else { br1 };
    let br2 = if br2 == 0 { br4 } else { br2 };
    let br3 = if br3 == 0 { br4 } else { br3 };
    ((br1 + br2 + br3 + br4) >> 2) & 0x00FF_00FF
}

/// java: `AmbientOcclusionFace.getVertexBrightness` — a weighted blend of four packed
/// brightnesses, done per byte, with Java's truncating float-to-int conversion.
///
/// The four products are summed strictly left to right, matching Java's evaluation order, so
/// that the truncation lands on the same integer.
#[inline]
#[allow(clippy::suboptimal_flops)]
fn vertex_brightness(b: [u32; 4], w: [f32; 4]) -> u32 {
    let byte = |shift: u32, i: usize| ((b[i] >> shift) & 255) as f32;
    let sky = (byte(16, 0) * w[0] + byte(16, 1) * w[1] + byte(16, 2) * w[2] + byte(16, 3) * w[3])
        as i32 as u32
        & 255;
    let block = (byte(0, 0) * w[0] + byte(0, 1) * w[1] + byte(0, 2) * w[2] + byte(0, 3) * w[3])
        as i32 as u32
        & 255;
    sky << 16 | block
}

/// `Block.getAmbientOcclusionLightValue()` for the state at `pos`: `0.2` for a normal cube,
/// `1.0` for anything light passes through.
#[inline]
fn ao_light(world: &World, pos: BlockPos) -> f32 {
    block_props(world.block_at(pos)).ambient_occlusion_light
}

/// `Block.isTranslucent()` for the state at `pos` — i.e. `!material.blocksLight()`.
#[inline]
fn translucent(world: &World, pos: BlockPos) -> bool {
    block_props(world.block_at(pos)).translucent
}

/// The whole of `AmbientOcclusionFace.updateVertexBrightness`, hazard **H8**.
///
/// `rendered` is the state being drawn (it selects the `getMixedBrightnessForBlock` override),
/// `pos` its world position, `face` the quad's own facing, and `qb` the output of
/// [`fill_quad_bounds`] for that quad. `settings` carries the fancy-graphics flag that
/// [`super::cull_face::is_opaque_cube`] needs — see the centre-sample comment below (the S11 ->
/// S01 §13 row).
#[must_use]
#[allow(clippy::too_many_lines)] // A faithful port; splitting it would obscure the ordering.
pub fn update_vertex_brightness(
    world: &World,
    rendered: BlockState,
    pos: BlockPos,
    face: Facing,
    qb: &QuadBounds,
    verts: Option<&[Vertex; 4]>,
    settings: MeshSettings,
) -> AoFace {
    let mixed = |p: BlockPos| mixed_brightness(world, rendered, p);

    // When the quad is flush with the face, every sample is taken one block out.
    let base = if qb.flush { pos.relative(face) } else { pos };
    let info = &NEIGHBOR_INFO[face.index() as usize];

    let p1 = base.relative(info.neighbors[0]);
    let p2 = base.relative(info.neighbors[1]);
    let p3 = base.relative(info.neighbors[2]);
    let p4 = base.relative(info.neighbors[3]);

    let (br1, br2, br3, br4) = (mixed(p1), mixed(p2), mixed(p3), mixed(p4));
    let (ao1, ao2, ao3, ao4) = (
        ao_light(world, p1),
        ao_light(world, p2),
        ao_light(world, p3),
        ao_light(world, p4),
    );

    // "Does light already get through past this side block?" — the H8 special case. When both
    // side blocks block light the corner sample is skipped entirely and the side value reused.
    let t1 = translucent(world, p1.relative(face));
    let t2 = translucent(world, p2.relative(face));
    let t3 = translucent(world, p3.relative(face));
    let t4 = translucent(world, p4.relative(face));

    let (corner13_ao, corner13_br) = if !t3 && !t1 {
        (ao1, br1)
    } else {
        let c = p1.relative(info.neighbors[2]);
        (ao_light(world, c), mixed(c))
    };
    let (corner14_ao, corner14_br) = if !t4 && !t1 {
        (ao1, br1)
    } else {
        let c = p1.relative(info.neighbors[3]);
        (ao_light(world, c), mixed(c))
    };
    let (corner23_ao, corner23_br) = if !t3 && !t2 {
        (ao2, br2)
    } else {
        let c = p2.relative(info.neighbors[2]);
        (ao_light(world, c), mixed(c))
    };
    let (corner24_ao, corner24_br) = if !t4 && !t2 {
        (ao2, br2)
    } else {
        let c = p2.relative(info.neighbors[3]);
        (ao_light(world, c), mixed(c))
    };

    // The centre sample. A quad flush with its face, or one facing a non-opaque neighbour,
    // takes the neighbour's light instead of its own.
    //
    // `is_opaque_cube` (not the raw `opaque_cube` field) because `BlockLeaves.isOpaqueCube()`
    // is `!fancyGraphics` rather than a constant, and `mc_data` only recorded the fast-graphics
    // value. Reading the field directly here would make a quad next to fancy-graphics leaves
    // keep sampling its own block instead of the leaves' light — the S11 -> S01 §13 row this
    // closes.
    //
    // Only the sample that is kept is computed: `mixed` is a pure read, and
    // every flush quad — every full-cube face — used to pay five hashed world reads for a
    // `mixed(pos)` it then overwrote.
    let centre_br = if qb.flush
        || !super::cull_face::is_opaque_cube(world.block_at(pos.relative(face)), settings)
    {
        mixed(pos.relative(face))
    } else {
        mixed(pos)
    };
    let centre_ao = if qb.flush {
        ao_light(world, base)
    } else {
        ao_light(world, pos)
    };

    let vt = verts
        .and_then(|v| corner_slots(v, face))
        .unwrap_or(VERTEX_TRANSLATIONS[face.index() as usize]);

    // The four corner AO values, in vanilla's (f29..f32 / f9..f12) order.
    let c0 = (ao4 + ao1 + corner14_ao + centre_ao) * 0.25;
    let c1 = (ao3 + ao1 + corner13_ao + centre_ao) * 0.25;
    let c2 = (ao3 + ao2 + corner23_ao + centre_ao) * 0.25;
    let c3 = (ao4 + ao2 + corner24_ao + centre_ao) * 0.25;

    let corner_br = [
        ao_brightness(br4, br1, corner14_br, centre_br),
        ao_brightness(br3, br1, corner13_br, centre_br),
        ao_brightness(br3, br2, corner23_br, centre_br),
        ao_brightness(br4, br2, corner24_br, centre_br),
    ];

    let mut out = AoFace {
        color_multiplier: [0.0; 4],
        brightness: [0; 4],
    };

    if qb.partial && info.interpolated {
        let weights = |o: &[usize; 8]| {
            [
                qb.bounds[o[0]] * qb.bounds[o[1]],
                qb.bounds[o[2]] * qb.bounds[o[3]],
                qb.bounds[o[4]] * qb.bounds[o[5]],
                qb.bounds[o[6]] * qb.bounds[o[7]],
            ]
        };
        for (slot, orientation) in info.orientations.iter().enumerate() {
            let w = weights(orientation);
            let target = vt[slot];
            out.color_multiplier[target] = c0 * w[0] + c1 * w[1] + c2 * w[2] + c3 * w[3];
            out.brightness[target] = vertex_brightness(corner_br, w);
        }
    } else {
        let flat = [c0, c1, c2, c3];
        for slot in 0..4 {
            out.color_multiplier[vt[slot]] = flat[slot];
            out.brightness[vt[slot]] = corner_br[slot];
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_brightness_matches_vanilla_constants() {
        // java: FaceBakery.getFaceBrightness
        assert_eq!(face_brightness(Facing::Down), 0.5);
        assert_eq!(face_brightness(Facing::Up), 1.0);
        assert_eq!(face_brightness(Facing::North), 0.8);
        assert_eq!(face_brightness(Facing::South), 0.8);
        assert_eq!(face_brightness(Facing::West), 0.6);
        assert_eq!(face_brightness(Facing::East), 0.6);
    }

    #[test]
    fn ao_brightness_substitutes_the_face_value_for_zeroes() {
        // java: getAoBrightness — br1..br3 of 0 are replaced by br4 before averaging.
        let face = super::super::light::pack(15, 0);
        assert_eq!(ao_brightness(0, 0, 0, face), face & 0x00FF_00FF);
        // Averaging is a plain >>2 over the packed value, masked to the two live bytes.
        let a = super::super::light::pack(8, 4);
        let b = super::super::light::pack(4, 0);
        let expected = ((a + a + b + b) >> 2) & 0x00FF_00FF;
        assert_eq!(ao_brightness(a, a, b, b), expected);
    }

    #[test]
    fn vertex_brightness_is_a_per_byte_weighted_blend() {
        let full = super::super::light::pack(15, 15);
        let dark = 0;
        // All the weight on the fully lit corner.
        let got = vertex_brightness([full, dark, dark, dark], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(got >> 16 & 255, 15 * 16);
        assert_eq!(got & 255, 15 * 16);
        // Half and half truncates, as Java's (int) cast does.
        let got = vertex_brightness([full, dark, dark, dark], [0.5, 0.5, 0.0, 0.0]);
        assert_eq!(got >> 16 & 255, 15 * 16 / 2);
    }

    /// `F-P2-client.renderer-02-2`: out-of-cube piston-head geometry supplies negative AO
    /// weights. Java truncates the float sum to a signed `int` before masking the low byte;
    /// casting straight to Rust `u32` saturates the negative value to zero instead.
    #[test]
    fn negative_piston_head_weight_wraps_each_brightness_byte_like_java() {
        let full = super::super::light::pack(15, 15);
        let dark = 0;
        // Reachable piston-head WEST/EAST slot-0 weights. Lighting only the two negative corners
        // produces 240 * (-0.15625 - 0.09375) = -60 in both packed-light channels.
        let got = vertex_brightness(
            [dark, full, full, dark],
            [0.78125, -0.15625, -0.09375, 0.46875],
        );
        assert_eq!(got, 0x00C4_00C4, "-60 as Java int, then & 255, is 196");

        // Negative control: the ordinary all-positive weighted path remains a plain truncating
        // blend rather than wrapping.
        let got = vertex_brightness([full, dark, dark, dark], [0.25, 0.25, 0.25, 0.25]);
        assert_eq!(got, 0x003C_003C);
    }

    #[test]
    fn vertex_translations_are_a_permutation_for_every_face() {
        for vt in &VERTEX_TRANSLATIONS {
            let mut seen = [false; 4];
            for &i in vt {
                assert!(!seen[i], "vertex translation {vt:?} is not a permutation");
                seen[i] = true;
            }
        }
    }

    /// The geometric corner mapping must reproduce vanilla's fixed table exactly, for a quad
    /// stored in `EnumFaceDirection`'s canonical order. That equivalence is what lets the
    /// mesher use the geometric form (which also survives `mc_assets` not reproducing
    /// `ModelRotation.rotateVertex`) without diverging from vanilla.
    #[test]
    fn vertex_translations_agree_with_the_geometric_mapping() {
        // java: EnumFaceDirection — (x, y, z) picked from the West/East, Down/Up, North/South
        // slots of the unit cube, in this exact order per face.
        use Facing::{Down, East, North, South, Up, West};
        let table = |f: Facing| -> [(Facing, Facing, Facing); 4] {
            match f {
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
        };
        let coord = |f: Facing| -> f32 {
            match f {
                West | Down | North => 0.0,
                East | Up | South => 1.0,
            }
        };
        for face in Facing::VALUES {
            let verts: [Vertex; 4] = std::array::from_fn(|i| {
                let (x, y, z) = table(face)[i];
                Vertex {
                    pos: [coord(x), coord(y), coord(z)],
                    uv: [0.0, 0.0],
                }
            });
            let geometric = corner_slots(&verts, face).expect("a unit face is well formed");
            assert_eq!(
                geometric,
                VERTEX_TRANSLATIONS[face.index() as usize],
                "face {face:?}"
            );
        }
    }

    #[test]
    fn a_degenerate_quad_falls_back_to_the_table() {
        // All four vertices in one spot: no in-plane extent, so no geometric classification.
        let verts = [Vertex {
            pos: [0.5, 1.0, 0.5],
            uv: [0.0, 0.0],
        }; 4];
        assert_eq!(corner_slots(&verts, Facing::Up), None);
    }

    #[test]
    fn orientation_indices_are_in_range_and_only_on_side_faces() {
        for (i, info) in NEIGHBOR_INFO.iter().enumerate() {
            let face = Facing::from_index(i as i32);
            // java: EnumNeighborInfo — only the four side faces carry orientation tables.
            assert_eq!(
                info.interpolated,
                !matches!(face, Facing::Down | Facing::Up)
            );
            if info.interpolated {
                for table in &info.orientations {
                    for &idx in table {
                        assert!(idx < 12, "orientation index {idx} out of range");
                    }
                }
            }
        }
    }
}
