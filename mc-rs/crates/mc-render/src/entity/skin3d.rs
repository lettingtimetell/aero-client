//! 3D skins: the outer skin layer as voxels.
//!
//! Vanilla draws a player's second skin layer (hat, jacket, sleeves, pants) as six flat boxes a
//! little larger than the body (`ModelPlayer`'s `*wear` parts, inflated 0.5 for the hat and 0.25
//! for the rest). The "3D Skins" mod replaces each of them with a voxel shell: every opaque texel
//! becomes a block one texel wide that stands proud of the body, so hair, hoods and jackets get
//! real depth.
//!
//! # The shell
//!
//! Each wear box (`W x H x D` texels, at the base body's exact position) gets a one-cell shell:
//! a grid of `(W+2)(H+2)(D+2)` cells whose interior is the body itself. Cells are one texel wide
//! along the face and [`WearPart::thickness`] deep, so a face's texels line up with the base
//! layer's pixels exactly (vanilla's flat layer stretches them by its inflation instead).
//!
//! * A **face cell** (one coordinate outside the body) is solid when its texel is opaque, with the
//!   entity shader's own test: alpha > 0.1 ([`OPAQUE_ALPHA`]), the texel `GL_GREATER 0.1` keeps.
//! * An **edge cell** (two coordinates outside) is solid when both neighbouring face texels are,
//!   so a brim that wraps a corner stays closed and a feature on one face only ends cleanly.
//! * A **corner cell** is solid when its three edge cells are.
//!
//! Only faces between a solid cell and an empty cell (or the outside) are emitted; faces toward
//! the body never are (the base layer is opaque by `ImageBufferDownload.setAreaOpaque`).
//!
//! # Keeping the vertex count low
//!
//! * The outward faces of a whole face are **one quad**, alpha-tested by the texture exactly like
//!   the flat layer (an empty texel is a discarded one): 6 quads per part instead of one per texel.
//! * Side walls are merged into runs along a row (the UV runs linearly along the row, so each
//!   segment still samples its own texel); edge strips merge the same way.
//! * Meshes are built once per skin texture ([`SkinLayerMesh::build`], ~0.1 ms) and cached by the
//!   caller; a frame only transforms the cached quads by the posed part matrices, in the same
//!   vertex batch, texture and draw as the body. No extra draw call, no GPU resource.
//!
//! Thicknesses keep armour outside the voxels: the hat is 0.75 deep (a helmet sits at 1.0), the
//! jacket, sleeves and pants 0.4 (leggings sit at 0.5, a chestplate and boots at 1.0). A worn
//! skull (`LayerCustomHead`, 9.5 texels wide) would meet the 0.75 hat exactly, so the caller
//! keeps the flat hat under any head item that is not armour (skull, pumpkin, block).

use std::collections::HashMap;
use std::sync::Arc;

use glam::{Mat4, Vec3};

use crate::entity::anim::{AnimParams, EntityModel};
use crate::entity::model::{Model, ModelPart, PartId};
use crate::entity::models::biped::PlayerModel;
use crate::entity::EntityVertex;

/// The smallest texel alpha the entity shader draws: it discards `a / 255 <= 0.1` (java:
/// `alphaFunc(GL_GREATER, 0.1)`), so 26 is the first kept value.
pub const OPAQUE_ALPHA: u8 = 26;

/// The six outer-layer parts, in [`SkinLayerMesh`] order. The discriminant is the bit index in a
/// [`WearMask`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WearPart {
    /// java: `bipedHeadwear` (`EnumPlayerModelParts.HAT`)
    Hat = 0,
    /// java: `bipedBodyWear` (`JACKET`)
    Jacket = 1,
    /// java: `bipedLeftArmwear` (`LEFT_SLEEVE`)
    LeftSleeve = 2,
    /// java: `bipedRightArmwear` (`RIGHT_SLEEVE`)
    RightSleeve = 3,
    /// java: `bipedLeftLegwear` (`LEFT_PANTS_LEG`)
    LeftPants = 4,
    /// java: `bipedRightLegwear` (`RIGHT_PANTS_LEG`)
    RightPants = 5,
}

/// A set of [`WearPart`]s, one bit each (`1 << part as u8`).
pub type WearMask = u8;

/// Every wear part.
pub const ALL_WEAR: WearMask = 0b11_1111;

impl WearPart {
    /// All six, in mesh order.
    pub const ALL: [Self; 6] = [
        Self::Hat,
        Self::Jacket,
        Self::LeftSleeve,
        Self::RightSleeve,
        Self::LeftPants,
        Self::RightPants,
    ];

    /// This part's bit in a [`WearMask`].
    #[must_use]
    pub const fn bit(self) -> WearMask {
        1 << self as u8
    }

    /// The part's handle in a [`PlayerModel`].
    #[must_use]
    pub fn part_id(self, model: &PlayerModel) -> PartId {
        match self {
            Self::Hat => model.biped.parts.headwear,
            Self::Jacket => model.wear.body,
            Self::LeftSleeve => model.wear.left_arm,
            Self::RightSleeve => model.wear.right_arm,
            Self::LeftPants => model.wear.left_leg,
            Self::RightPants => model.wear.right_leg,
        }
    }

    /// How far the voxels stand proud of the body, in texels (model units). Chosen to stay
    /// inside the armour that can cover the part: a helmet at 1.0 over the hat; leggings at 0.5
    /// over the jacket and pants; a chestplate and boots at 1.0.
    #[must_use]
    pub const fn thickness(self) -> f32 {
        match self {
            Self::Hat => 0.75,
            _ => 0.4,
        }
    }
}

/// One quad of a voxel layer in part-local model units (Y down, before the draw scale), with
/// its texture coordinates already normalised to the skin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelQuad {
    /// The four corners, wound as a fan (`0-1-2`, `0-2-3`).
    pub pos: [[f32; 3]; 4],
    /// Texture coordinate per corner.
    pub uv: [[f32; 2]; 4],
    /// Outward unit normal (an axis direction).
    pub normal: [f32; 3],
    /// The normal as an index: `axis * 2 + positive` (0 = -X, 1 = +X, 2 = -Y, ..., 5 = +Z).
    pub facing: u8,
}

impl VoxelQuad {
    /// The axis the quad faces along.
    #[must_use]
    pub const fn axis(&self) -> usize {
        (self.facing >> 1) as usize
    }

    /// Whether it faces the positive direction of [`Self::axis`].
    #[must_use]
    pub const fn positive(&self) -> bool {
        self.facing & 1 == 1
    }

    /// The quad's plane coordinate along its axis.
    #[must_use]
    pub fn plane(&self) -> f32 {
        self.pos[0][self.axis()]
    }

    /// Whether a viewer at `eye` (part-local model units) sees the quad's front. The voxel
    /// layer and the opaque body under it form a closed surface, so a quad seen from behind is
    /// always hidden behind a front-facing one: skipping it changes no pixel.
    #[must_use]
    pub fn faces(&self, eye: [f32; 3]) -> bool {
        let plane = self.plane();
        if self.positive() {
            eye[self.axis()] > plane
        } else {
            eye[self.axis()] < plane
        }
    }
}

/// One skin's six voxel layers, built for one arm width.
#[derive(Clone, Debug, PartialEq)]
pub struct SkinLayerMesh {
    /// Built for slim (3-texel) arms.
    pub slim: bool,
    parts: [Vec<VoxelQuad>; 6],
}

impl SkinLayerMesh {
    /// Builds the voxel layers of a normalised 64x64 skin for the given arm width. `None` for a
    /// texture of another size (an HD resource-pack default skin keeps the flat layer).
    #[must_use]
    pub fn build(rgba: &[u8], width: u32, height: u32, slim: bool) -> Option<Self> {
        if width != 64 || height != 64 || rgba.len() != 64 * 64 * 4 {
            return None;
        }
        let texels = Texels {
            rgba,
            width: width as i32,
            height: height as i32,
        };
        let model = PlayerModel::new(0.0, slim);
        let parts = WearPart::ALL.map(|part| {
            let mut quads = Vec::new();
            let source = model.biped.model.get(part.part_id(&model));
            if let Some(spec) = BoxSpec::from_part(source, part.thickness()) {
                build_box(&texels, &spec, &mut quads);
            }
            quads
        });
        Some(Self { slim, parts })
    }

    /// One part's quads.
    #[must_use]
    pub fn part(&self, part: WearPart) -> &[VoxelQuad] {
        &self.parts[part as usize]
    }

    /// Quads over all six parts.
    #[must_use]
    pub fn quad_count(&self) -> usize {
        self.parts.iter().map(Vec::len).sum()
    }

    /// Vertices the six parts emit per frame (six per quad), for the stats in the report.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.quad_count() * 6
    }
}

// ---------------------------------------------------------------------------------------------
// The builder
// ---------------------------------------------------------------------------------------------

struct Texels<'a> {
    rgba: &'a [u8],
    width: i32,
    height: i32,
}

impl Texels<'_> {
    fn alpha(&self, x: i32, y: i32) -> u8 {
        if x < 0 || y < 0 || x >= self.width || y >= self.height {
            return 0;
        }
        self.rgba[((y * self.width + x) * 4 + 3) as usize]
    }
}

/// One wear box in part-local model units: `ModelBox`'s texture offset, `pos1` and size.
struct BoxSpec {
    tex: [f32; 2],
    origin: [f32; 3],
    dims: [i32; 3],
    thickness: f32,
    tex_size: [f32; 2],
}

/// The axis and side of the six `ModelBox` faces, in vanilla's order: `+X, -X, -Y, +Y, -Z, +Z`.
const FACE_AXIS: [usize; 6] = [0, 0, 1, 1, 2, 2];
const FACE_MAX: [bool; 6] = [true, false, false, true, false, true];

const fn face_of(axis: usize, max: bool) -> usize {
    match (axis, max) {
        (0, true) => 0,
        (0, false) => 1,
        (1, false) => 2,
        (1, true) => 3,
        (2, false) => 4,
        _ => 5,
    }
}

const fn other_axes(axis: usize) -> (usize, usize) {
    match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

impl BoxSpec {
    fn from_part(part: &ModelPart, thickness: f32) -> Option<Self> {
        let cube = part.cubes.first()?;
        let dims = [
            (cube.pos2[0] - cube.pos1[0]) as i32,
            (cube.pos2[1] - cube.pos1[1]) as i32,
            (cube.pos2[2] - cube.pos1[2]) as i32,
        ];
        (dims.iter().all(|&d| d > 0)).then(|| Self {
            tex: [part.texture_offset[0] as f32, part.texture_offset[1] as f32],
            origin: cube.pos1,
            dims,
            thickness,
            tex_size: [part.texture_width, part.texture_height],
        })
    }

    fn dim(&self, axis: usize) -> f32 {
        self.dims[axis] as f32
    }

    /// The model coordinate of cell boundary `b` along `axis` (`b` in `-1..=dims + 1`): the
    /// shell's outer planes sit `thickness` outside the body, every inner boundary on a texel.
    fn bound(&self, axis: usize, b: i32) -> f32 {
        if b < 0 {
            self.origin[axis] - self.thickness
        } else if b > self.dims[axis] {
            self.origin[axis] + self.dim(axis) + self.thickness
        } else {
            self.origin[axis] + b as f32
        }
    }

    /// The texel position (in pixels) a point on face `face`'s plane maps to: `ModelBox`'s UV
    /// layout for that face, extended linearly. Ignores the face's own normal coordinate.
    fn face_pixel(&self, face: usize, p: [f32; 3]) -> [f32; 2] {
        let [tx, ty] = self.tex;
        let [x0, y0, z0] = self.origin;
        let (w, d) = (self.dim(0), self.dim(2));
        match face {
            0 => [tx + d + w + (p[2] - z0), ty + d + (p[1] - y0)],
            1 => [tx + (z0 + d - p[2]), ty + d + (p[1] - y0)],
            2 => [tx + d + (p[0] - x0), ty + (z0 + d - p[2])],
            3 => [tx + d + w + (p[0] - x0), ty + (z0 + d - p[2])],
            4 => [tx + d + (p[0] - x0), ty + d + (p[1] - y0)],
            _ => [tx + d + w + d + (x0 + w - p[0]), ty + d + (p[1] - y0)],
        }
    }

    fn uv(&self, pixel: [f32; 2]) -> [f32; 2] {
        [pixel[0] / self.tex_size[0], pixel[1] / self.tex_size[1]]
    }

    /// The centre of cell `c` along `axis`.
    fn centre(&self, axis: usize, c: i32) -> f32 {
        (self.bound(axis, c) + self.bound(axis, c + 1)) * 0.5
    }

    /// The texel face `face` shows at cell `c`, with every other coordinate clamped onto the
    /// face (so an edge or corner cell samples the face's border texel).
    fn texel(&self, face: usize, c: [i32; 3]) -> [i32; 2] {
        let mut p = [0.0; 3];
        for (axis, value) in p.iter_mut().enumerate() {
            let clamped = c[axis].clamp(0, self.dims[axis] - 1);
            *value = self.origin[axis] + clamped as f32 + 0.5;
        }
        let px = self.face_pixel(face, p);
        [px[0].floor() as i32, px[1].floor() as i32]
    }

    /// The UV of texel `t`'s centre: the constant coordinate small faces use.
    fn texel_centre_uv(&self, t: [i32; 2]) -> [f32; 2] {
        self.uv([t[0] as f32 + 0.5, t[1] as f32 + 0.5])
    }
}

/// The shell's solid cells.
struct Grid {
    dims: [i32; 3],
    solid: Vec<bool>,
}

impl Grid {
    fn index(&self, c: [i32; 3]) -> usize {
        let [w, h, _] = self.dims;
        ((c[0] + 1) + (c[1] + 1) * (w + 2) + (c[2] + 1) * (w + 2) * (h + 2)) as usize
    }

    fn inside(&self, c: [i32; 3]) -> bool {
        (0..3).all(|a| c[a] >= -1 && c[a] <= self.dims[a])
    }

    fn solid(&self, c: [i32; 3]) -> bool {
        self.inside(c) && self.solid[self.index(c)]
    }

    fn out_axes(&self, c: [i32; 3]) -> [bool; 3] {
        core::array::from_fn(|a| c[a] < 0 || c[a] >= self.dims[a])
    }

    /// Every shell cell (at least one coordinate outside the body).
    fn shell_cells(&self) -> impl Iterator<Item = [i32; 3]> + '_ {
        let [w, h, d] = self.dims;
        (-1..=d).flat_map(move |z| {
            (-1..=h).flat_map(move |y| {
                (-1..=w)
                    .map(move |x| [x, y, z])
                    .filter(move |&c| self.out_axes(c).iter().any(|&o| o))
            })
        })
    }
}

/// One step inward along `axis` from a coordinate outside the body.
fn step_in(spec: &BoxSpec, mut c: [i32; 3], axis: usize) -> [i32; 3] {
    c[axis] = if c[axis] < 0 { 0 } else { spec.dims[axis] - 1 };
    c
}

fn push_quad(
    out: &mut Vec<VoxelQuad>,
    normal_axis: usize,
    positive: bool,
    plane: f32,
    (u_axis, u0, u1): (usize, f32, f32),
    (v_axis, v0, v1): (usize, f32, f32),
    uv: impl Fn([f32; 3]) -> [f32; 2],
) {
    let corner = |u: f32, v: f32| {
        let mut p = [0.0; 3];
        p[normal_axis] = plane;
        p[u_axis] = u;
        p[v_axis] = v;
        p
    };
    let pos = [
        corner(u0, v0),
        corner(u1, v0),
        corner(u1, v1),
        corner(u0, v1),
    ];
    let mut normal = [0.0; 3];
    normal[normal_axis] = if positive { 1.0 } else { -1.0 };
    out.push(VoxelQuad {
        pos,
        uv: pos.map(&uv),
        normal,
        facing: (normal_axis as u8) * 2 + u8::from(positive),
    });
}

/// Builds one wear box's voxel shell into `out`.
fn build_box(texels: &Texels<'_>, spec: &BoxSpec, out: &mut Vec<VoxelQuad>) {
    let grid = solid_grid(texels, spec);
    for face in 0..6 {
        emit_face(spec, &grid, face, out);
    }
    emit_edges(spec, &grid, out);
    emit_corners(spec, &grid, out);
}

/// Classifies every shell cell of one box.
fn solid_grid(texels: &Texels<'_>, spec: &BoxSpec) -> Grid {
    let dims = spec.dims;
    let mut grid = Grid {
        dims,
        solid: vec![false; ((dims[0] + 2) * (dims[1] + 2) * (dims[2] + 2)) as usize],
    };
    let cells: Vec<[i32; 3]> = grid.shell_cells().collect();
    // Face cells first, then edges (two outside axes), then corners (three): each rule reads
    // only the previous class.
    for outside in 1..=3 {
        for &c in &cells {
            let out_axes = grid.out_axes(c);
            if out_axes.iter().filter(|&&o| o).count() != outside {
                continue;
            }
            let solid = if outside == 1 {
                let axis = out_axes.iter().position(|&o| o).expect("one outside axis");
                let face = face_of(axis, c[axis] >= 0);
                let t = spec.texel(face, c);
                texels.alpha(t[0], t[1]) >= OPAQUE_ALPHA
            } else {
                (0..3)
                    .filter(|&a| out_axes[a])
                    .all(|a| grid.solid(step_in(spec, c, a)))
            };
            let index = grid.index(c);
            grid.solid[index] = solid;
        }
    }
    grid
}

/// One face: its single alpha-tested outward quad, then its merged side walls.
fn emit_face(spec: &BoxSpec, grid: &Grid, face: usize, out: &mut Vec<VoxelQuad>) {
    let a = FACE_AXIS[face];
    let max = FACE_MAX[face];
    let slab = if max { spec.dims[a] } else { -1 };
    let (p, q) = other_axes(a);
    let mut any = false;
    for pi in 0..spec.dims[p] {
        for qi in 0..spec.dims[q] {
            let mut c = [0; 3];
            c[a] = slab;
            c[p] = pi;
            c[q] = qi;
            any |= grid.solid(c);
        }
    }
    if !any {
        return;
    }
    // The outward face, exactly the flat layer's UV rectangle at the shell's outer plane.
    let outer = if max {
        spec.bound(a, spec.dims[a] + 1)
    } else {
        spec.bound(a, -1)
    };
    push_quad(
        out,
        a,
        max,
        outer,
        (p, spec.origin[p], spec.origin[p] + spec.dim(p)),
        (q, spec.origin[q], spec.origin[q] + spec.dim(q)),
        |pt| spec.uv(spec.face_pixel(face, pt)),
    );
    // Side walls, merged along the run axis `r`; `n` is the wall's normal axis.
    let slab_lo = spec.bound(a, slab);
    let slab_hi = spec.bound(a, slab + 1);
    for (n, r) in [(p, q), (q, p)] {
        for positive in [false, true] {
            let step = if positive { 1 } else { -1 };
            for ni in 0..spec.dims[n] {
                let mut run_start = None;
                for ri in 0..=spec.dims[r] {
                    let exposed = ri < spec.dims[r] && {
                        let mut c = [0; 3];
                        c[a] = slab;
                        c[n] = ni;
                        c[r] = ri;
                        let mut neighbour = c;
                        neighbour[n] += step;
                        grid.solid(c) && !grid.solid(neighbour)
                    };
                    match (exposed, run_start) {
                        (true, None) => run_start = Some(ri),
                        (false, Some(r0)) => {
                            let plane = spec.bound(n, if positive { ni + 1 } else { ni });
                            let centre_n = spec.centre(n, ni);
                            push_quad(
                                out,
                                n,
                                positive,
                                plane,
                                (r, spec.bound(r, r0), spec.bound(r, ri)),
                                (a, slab_lo, slab_hi),
                                |mut pt| {
                                    // Constant across the wall, linear along the run: each
                                    // segment samples its own texel's centre column/row.
                                    pt[n] = centre_n;
                                    spec.uv(spec.face_pixel(face, pt))
                                },
                            );
                            run_start = None;
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

/// The twelve edges: merged outward strips on both faces, and the end walls between edge cells.
fn emit_edges(spec: &BoxSpec, grid: &Grid, out: &mut Vec<VoxelQuad>) {
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        let e = 3 - a - b;
        for a_max in [false, true] {
            for b_max in [false, true] {
                let sa = if a_max { spec.dims[a] } else { -1 };
                let sb = if b_max { spec.dims[b] } else { -1 };
                let cell = |ei: i32| {
                    let mut c = [0; 3];
                    c[a] = sa;
                    c[b] = sb;
                    c[e] = ei;
                    c
                };
                let face_a = face_of(a, a_max);
                let face_b = face_of(b, b_max);
                let (a_lo, a_hi) = (spec.bound(a, sa), spec.bound(a, sa + 1));
                let (b_lo, b_hi) = (spec.bound(b, sb), spec.bound(b, sb + 1));
                let outer_a = if a_max { a_hi } else { a_lo };
                let outer_b = if b_max { b_hi } else { b_lo };
                // Outward strips, merged along the edge.
                let mut run_start = None;
                for ei in 0..=spec.dims[e] {
                    let solid = ei < spec.dims[e] && grid.solid(cell(ei));
                    match (solid, run_start) {
                        (true, None) => run_start = Some(ei),
                        (false, Some(e0)) => {
                            let span_e = (e, spec.bound(e, e0), spec.bound(e, ei));
                            // On face A's plane: the border texel column along `b`.
                            let border_b = spec.origin[b] + sb.clamp(0, spec.dims[b] - 1) as f32;
                            push_quad(out, a, a_max, outer_a, span_e, (b, b_lo, b_hi), |mut p| {
                                p[b] = border_b + 0.5;
                                spec.uv(spec.face_pixel(face_a, p))
                            });
                            let border_a = spec.origin[a] + sa.clamp(0, spec.dims[a] - 1) as f32;
                            push_quad(out, b, b_max, outer_b, span_e, (a, a_lo, a_hi), |mut p| {
                                p[a] = border_a + 0.5;
                                spec.uv(spec.face_pixel(face_b, p))
                            });
                            run_start = None;
                        }
                        _ => {}
                    }
                }
                // End walls where a solid edge cell meets an empty edge or corner cell. The
                // side faces (+-Z, then +-X) own the colour of these little squares.
                let owner = if FACE_AXIS[face_a] == 2 || (FACE_AXIS[face_b] != 2 && a == 0) {
                    face_a
                } else {
                    face_b
                };
                for ei in -1..spec.dims[e] {
                    let (lo, hi) = (cell(ei), cell(ei + 1));
                    let (solid_lo, solid_hi) = (grid.solid(lo), grid.solid(hi));
                    if solid_lo == solid_hi {
                        continue;
                    }
                    let solid_cell = if solid_lo { lo } else { hi };
                    let uv = spec.texel_centre_uv(spec.texel(owner, solid_cell));
                    push_quad(
                        out,
                        e,
                        solid_lo,
                        spec.bound(e, ei + 1),
                        (a, a_lo, a_hi),
                        (b, b_lo, b_hi),
                        |_| uv,
                    );
                }
            }
        }
    }
}

/// The eight corners: three outward squares each, every one in its own face's corner texel.
fn emit_corners(spec: &BoxSpec, grid: &Grid, out: &mut Vec<VoxelQuad>) {
    for x_max in [false, true] {
        for y_max in [false, true] {
            for z_max in [false, true] {
                let max = [x_max, y_max, z_max];
                let c: [i32; 3] = core::array::from_fn(|a| if max[a] { spec.dims[a] } else { -1 });
                if !grid.solid(c) {
                    continue;
                }
                for (a, &positive) in max.iter().enumerate() {
                    let face = face_of(a, positive);
                    let (u, v) = other_axes(a);
                    let plane = if positive {
                        spec.bound(a, spec.dims[a] + 1)
                    } else {
                        spec.bound(a, -1)
                    };
                    let uv = spec.texel_centre_uv(spec.texel(face, c));
                    push_quad(
                        out,
                        a,
                        positive,
                        plane,
                        (u, spec.bound(u, c[u]), spec.bound(u, c[u] + 1)),
                        (v, spec.bound(v, c[v]), spec.bound(v, c[v] + 1)),
                        |_| uv,
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------------------------

/// Transforms `quads` by `base * part.local_matrix(scale)` (the matrix [`Model::emit_part`]
/// draws the part under) and appends the ones facing the camera, six vertices per quad,
/// untinted like every model vertex. Respects the part's visibility exactly as `emit_part`
/// does. `base` is camera-relative (the entity pipeline's convention), so the camera is the
/// origin.
pub fn emit_wear_voxels(
    model: &Model,
    id: PartId,
    quads: &[VoxelQuad],
    scale: f32,
    base: Mat4,
    out: &mut Vec<EntityVertex>,
) {
    let part = model.get(id);
    if part.is_hidden || !part.show_model {
        return;
    }
    emit_quads(quads, base * part.local_matrix(scale), scale, out);
}

/// Appends the camera-facing `quads` transformed by `m` (positions are scaled by `scale`
/// first; the camera is the origin of `m`'s output space).
///
/// Per part: one matrix inverse (the camera in part-local units) and the six axis normals.
/// Per quad: one compare, then four corner transforms for the visible ones.
pub fn emit_quads(quads: &[VoxelQuad], m: Mat4, scale: f32, out: &mut Vec<EntityVertex>) {
    emit_quads_with(quads, m, scale, true, out);
}

/// [`emit_quads`] without the camera cull, for a vertex space whose origin is not the camera
/// (the first-person hand).
pub fn emit_quads_unculled(quads: &[VoxelQuad], m: Mat4, scale: f32, out: &mut Vec<EntityVertex>) {
    emit_quads_with(quads, m, scale, false, out);
}

fn emit_quads_with(
    quads: &[VoxelQuad],
    m: Mat4,
    scale: f32,
    cull: bool,
    out: &mut Vec<EntityVertex>,
) {
    if quads.is_empty() {
        return;
    }
    let scaled = m * Mat4::from_scale(Vec3::splat(scale));
    let eye = eye_in(scaled);
    let normals: [[f32; 3]; 6] = core::array::from_fn(|facing| {
        let mut n = Vec3::ZERO;
        n[facing >> 1] = if facing & 1 == 1 { 1.0 } else { -1.0 };
        m.transform_vector3(n).normalize_or_zero().into()
    });
    out.reserve(if cull {
        quads.len() * 3
    } else {
        quads.len() * 6
    });
    for quad in quads {
        if cull && !quad.faces(eye) {
            continue;
        }
        let normal = normals[usize::from(quad.facing)];
        let v = |i: usize| EntityVertex {
            pos: scaled.transform_point3(Vec3::from(quad.pos[i])).into(),
            uv: quad.uv[i],
            normal,
            color: EntityVertex::WHITE,
        };
        let (v0, v1, v2, v3) = (v(0), v(1), v(2), v(3));
        out.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
    }
}

/// The camera (the origin of `scaled`'s output space) in the part-local units it maps from.
fn eye_in(scaled: Mat4) -> [f32; 3] {
    scaled.inverse().transform_point3(Vec3::ZERO).into()
}

/// Poses and emits a player body exactly like `EntityModel::pose_and_render`, except that the
/// wear parts in `voxels` are drawn from `mesh` instead of as flat boxes. A part the skin-part
/// settings (or a spectator view) hide stays hidden; every other part, the pose and the vertex
/// order of the flat parts are untouched. A mesh built for the other arm width, or a child
/// model, falls back to the flat body.
pub fn emit_player_body(
    model: &mut PlayerModel,
    params: &AnimParams,
    base: Mat4,
    mesh: &SkinLayerMesh,
    voxels: WearMask,
    out: &mut Vec<EntityVertex>,
) {
    model.pose_only(params);
    if voxels == 0 || model.biped.model.is_child || mesh.slim != model.small_arms {
        model.render(params, base, out);
        return;
    }
    let mut replaced: WearMask = 0;
    for part in WearPart::ALL {
        if voxels & part.bit() == 0 {
            continue;
        }
        let id = part.part_id(model);
        let p = model.biped.model.get_mut(id);
        if p.show_model && !p.is_hidden {
            p.show_model = false;
            replaced |= part.bit();
        }
    }
    model.render(params, base, out);
    // java: `ModelBiped.render`/`ModelPlayer.render` draw every adult part, wear included,
    // under the sneak-lowered matrix.
    let m = if params.entity.sneaking {
        base * Mat4::from_translation(Vec3::new(0.0, 0.2, 0.0))
    } else {
        base
    };
    for part in WearPart::ALL {
        if replaced & part.bit() == 0 {
            continue;
        }
        let id = part.part_id(model);
        model.biped.model.get_mut(id).show_model = true;
        emit_wear_voxels(
            &model.biped.model,
            id,
            mesh.part(part),
            params.scale,
            m,
            out,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------------------------

/// One cached skin: per arm width (`[classic, slim]`), `None` = not built yet, `Some(None)` =
/// this texture cannot have voxel layers (not 64x64).
type CacheSlots = [Option<Option<Arc<SkinLayerMesh>>>; 2];

/// Built meshes by skin texture key and arm width. Bounded: past [`Self::CAPACITY`] skins it
/// starts over (a mesh is ~0.1 ms to rebuild).
#[derive(Debug, Default)]
pub struct Skin3dCache {
    meshes: HashMap<String, CacheSlots>,
    builds: u64,
}

impl Skin3dCache {
    /// Distinct skins kept before the cache starts over.
    pub const CAPACITY: usize = 256;

    /// The mesh for `key` (building it from `pixels` the first time it is asked for). `pixels`
    /// is only called on a miss and returns the skin's `(rgba, width, height)`, or `None` while
    /// it is not available (nothing is remembered then, so a later frame retries). A hit is one
    /// borrowed-`str` hash probe and an `Arc` clone: no allocation.
    pub fn get_or_build<'p>(
        &mut self,
        key: &str,
        slim: bool,
        pixels: impl FnOnce() -> Option<(&'p [u8], u32, u32)>,
    ) -> Option<Arc<SkinLayerMesh>> {
        let slot = usize::from(slim);
        if let Some(Some(mesh)) = self.meshes.get(key).map(|slots| &slots[slot]) {
            return mesh.clone();
        }
        let (rgba, width, height) = pixels()?;
        if self.meshes.len() >= Self::CAPACITY && !self.meshes.contains_key(key) {
            self.meshes.clear();
        }
        let mesh = SkinLayerMesh::build(rgba, width, height, slim).map(Arc::new);
        self.builds += 1;
        self.meshes.entry(key.to_owned()).or_default()[slot] = Some(mesh.clone());
        mesh
    }

    /// Meshes built so far (each skin and arm width once, unless the cache was cleared).
    #[must_use]
    pub fn builds(&self) -> u64 {
        self.builds
    }

    /// Forgets every mesh (a resource reload can change the default skins' pixels).
    pub fn clear(&mut self) {
        self.meshes.clear();
    }

    /// Meshes held (including skins that cannot have one).
    #[must_use]
    pub fn len(&self) -> usize {
        self.meshes.len()
    }

    /// True when nothing is cached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.meshes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The six wear boxes' texture rectangles, `(u, v, w, h)` in pixels (both arm widths share
    /// the 16-wide sleeve rectangles; slim leaves the last column unused).
    const OUTER_RECTS: [(usize, usize, usize, usize); 6] = [
        (32, 0, 32, 16),  // hat
        (16, 32, 24, 16), // jacket
        (48, 48, 16, 16), // left sleeve
        (40, 32, 16, 16), // right sleeve
        (0, 48, 16, 16),  // left pants
        (0, 32, 16, 16),  // right pants
    ];

    fn skin(alpha_at: impl Fn(usize, usize) -> u8) -> Vec<u8> {
        let mut rgba = vec![0u8; 64 * 64 * 4];
        for y in 0..64 {
            for x in 0..64 {
                let i = (y * 64 + x) * 4;
                rgba[i] = (x * 4) as u8;
                rgba[i + 1] = (y * 4) as u8;
                rgba[i + 2] = 128;
                rgba[i + 3] = alpha_at(x, y);
            }
        }
        rgba
    }

    fn in_outer_layer(x: usize, y: usize) -> bool {
        OUTER_RECTS
            .iter()
            .any(|&(u, v, w, h)| x >= u && x < u + w && y >= v && y < v + h)
    }

    /// A deterministic pseudo-random outer layer: ~half its texels opaque.
    fn noisy_skin(seed: u64) -> Vec<u8> {
        skin(|x, y| {
            let mut h = seed ^ ((x as u64) << 32 | y as u64);
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
            h ^= h >> 33;
            if in_outer_layer(x, y) && h & 1 == 0 {
                255
            } else if in_outer_layer(x, y) {
                (h >> 8) as u8 % OPAQUE_ALPHA
            } else {
                255
            }
        })
    }

    fn spec_for(part: WearPart, slim: bool) -> BoxSpec {
        let model = PlayerModel::new(0.0, slim);
        BoxSpec::from_part(
            model.biped.model.get(part.part_id(&model)),
            part.thickness(),
        )
        .expect("wear parts have one box")
    }

    fn area(q: &VoxelQuad) -> f32 {
        let a = Vec3::from(q.pos[1]) - Vec3::from(q.pos[0]);
        let b = Vec3::from(q.pos[3]) - Vec3::from(q.pos[0]);
        a.cross(b).length()
    }

    fn is_big_outward(spec: &BoxSpec, q: &VoxelQuad) -> bool {
        let axis = q
            .normal
            .iter()
            .position(|n| n.abs() == 1.0)
            .expect("axis normal");
        let (p, r) = other_axes(axis);
        let ext = |k: usize| {
            let lo = q.pos.iter().map(|v| v[k]).fold(f32::MAX, f32::min);
            let hi = q.pos.iter().map(|v| v[k]).fold(f32::MIN, f32::max);
            (lo, hi)
        };
        ext(p) == (spec.origin[p], spec.origin[p] + spec.dim(p))
            && ext(r) == (spec.origin[r], spec.origin[r] + spec.dim(r))
    }

    #[test]
    fn a_transparent_outer_layer_has_no_voxels() {
        let rgba = skin(|x, y| if in_outer_layer(x, y) { 0 } else { 255 });
        let mesh = SkinLayerMesh::build(&rgba, 64, 64, false).expect("64x64");
        assert_eq!(mesh.quad_count(), 0);
        // Alpha at the shader's discard boundary is still empty (`GL_GREATER 0.1`).
        let rgba = skin(|x, y| if in_outer_layer(x, y) { 25 } else { 255 });
        assert_eq!(
            SkinLayerMesh::build(&rgba, 64, 64, true)
                .unwrap()
                .quad_count(),
            0
        );
        assert!(
            SkinLayerMesh::build(&rgba, 64, 32, true).is_none(),
            "only 64x64 skins"
        );
    }

    /// Fully opaque: each part is a closed box of 6 face quads, 12 edges x 2 strips and 8
    /// corners x 3 squares, with no internal walls.
    #[test]
    fn a_full_outer_layer_is_a_closed_shell_of_54_quads_per_part() {
        let rgba = skin(|_, _| 255);
        for slim in [false, true] {
            let mesh = SkinLayerMesh::build(&rgba, 64, 64, slim).unwrap();
            for part in WearPart::ALL {
                assert_eq!(mesh.part(part).len(), 54, "{part:?} slim={slim}");
                // The shell's extents are the body box grown by the part's thickness.
                let spec = spec_for(part, slim);
                for axis in 0..3 {
                    let lo = mesh.part(part).iter().flat_map(|q| q.pos).map(|p| p[axis]);
                    let min = lo.clone().fold(f32::MAX, f32::min);
                    let max = lo.fold(f32::MIN, f32::max);
                    assert_eq!(
                        min,
                        spec.origin[axis] - part.thickness(),
                        "{part:?} axis {axis}"
                    );
                    assert_eq!(
                        max,
                        spec.origin[axis] + spec.dim(axis) + part.thickness(),
                        "{part:?} axis {axis}"
                    );
                }
            }
            assert_eq!(mesh.vertex_count(), 6 * 54 * 6);
        }
        // Slim sleeves are three texels wide.
        let slim = spec_for(WearPart::LeftSleeve, true);
        assert_eq!(slim.dims, [3, 12, 4]);
        assert_eq!(spec_for(WearPart::LeftSleeve, false).dims, [4, 12, 4]);
    }

    /// One opaque texel in the middle of the hat's front: the face quad plus four walls, all on
    /// that texel's cell and sampling that texel.
    #[test]
    fn a_single_texel_is_a_raised_block_with_four_walls() {
        // Hat front (-Z) texels start at (32 + 8, 0 + 8); texel (3, 4) -> pixel (43, 12).
        let rgba = skin(|x, y| {
            if (x, y) == (43, 12) {
                255
            } else if in_outer_layer(x, y) {
                0
            } else {
                255
            }
        });
        let mesh = SkinLayerMesh::build(&rgba, 64, 64, false).unwrap();
        let hat = mesh.part(WearPart::Hat);
        assert_eq!(hat.len(), 5, "{hat:#?}");
        assert_eq!(mesh.quad_count(), 5);
        let t = WearPart::Hat.thickness();
        let spec = spec_for(WearPart::Hat, false);
        let walls: Vec<_> = hat.iter().filter(|q| !is_big_outward(&spec, q)).collect();
        assert_eq!(walls.len(), 4);
        for wall in walls {
            assert!(
                (area(wall) - t).abs() < 1e-6,
                "a wall is one texel by the thickness"
            );
            for p in wall.pos {
                // Cell (3, 4, -1): x in -1..0, y in -4..-3, z in -4.75..-4.
                assert!((-1.0..=0.0).contains(&p[0]), "{p:?}");
                assert!((-4.0..=-3.0).contains(&p[1]), "{p:?}");
                assert!((-4.0 - t..=-4.0).contains(&p[2]), "{p:?}");
            }
            // Corners sit on texel boundaries; the wall's interior samples the texel.
            let centre = wall.uv.iter().fold([0.0f32; 2], |s, uv| {
                [s[0] + uv[0] / 4.0, s[1] + uv[1] / 4.0]
            });
            assert_eq!(
                ((centre[0] * 64.0) as usize, (centre[1] * 64.0) as usize),
                (43, 12)
            );
        }
    }

    /// Merging must not change the surface: per direction, the emitted wall/strip/corner area
    /// equals a brute-force count over every exposed cell face, and every face quad exists
    /// exactly when its face has an opaque texel. No two quads overlap on one plane, and every
    /// small quad samples an opaque texel.
    #[test]
    fn merged_quads_cover_exactly_the_voxel_surface() {
        for seed in 0..6 {
            let rgba = noisy_skin(seed);
            let texels = Texels {
                rgba: &rgba,
                width: 64,
                height: 64,
            };
            for slim in [false, true] {
                for part in WearPart::ALL {
                    let spec = spec_for(part, slim);
                    let grid = solid_grid(&texels, &spec);
                    let mut quads = Vec::new();
                    build_box(&texels, &spec, &mut quads);

                    // Brute force: [axis][positive] exposed area, excluding the outward faces
                    // of face cells (those are the alpha-tested face quads).
                    let mut want = [[0.0f64; 2]; 3];
                    let mut faces_with_texels = [false; 6];
                    let [w, h, d] = spec.dims;
                    for z in -1..=d {
                        for y in -1..=h {
                            for x in -1..=w {
                                let c = [x, y, z];
                                let out = grid.out_axes(c);
                                if !out.iter().any(|&o| o) || !grid.solid(c) {
                                    continue;
                                }
                                for axis in 0..3 {
                                    for positive in [false, true] {
                                        let mut n = c;
                                        n[axis] += if positive { 1 } else { -1 };
                                        let neighbour_out = grid.out_axes(n);
                                        let interior =
                                            grid.inside(n) && !neighbour_out.iter().any(|&o| o);
                                        let exposed =
                                            !grid.inside(n) || (!interior && !grid.solid(n));
                                        if !exposed {
                                            continue;
                                        }
                                        let face_cell_outward =
                                            out.iter().filter(|&&o| o).count() == 1 && out[axis];
                                        if face_cell_outward {
                                            faces_with_texels[face_of(axis, positive)] = true;
                                            continue;
                                        }
                                        let (u, v) = other_axes(axis);
                                        let len = |k: usize| {
                                            f64::from(spec.bound(k, c[k] + 1) - spec.bound(k, c[k]))
                                        };
                                        want[axis][usize::from(positive)] += len(u) * len(v);
                                    }
                                }
                            }
                        }
                    }
                    let mut got = [[0.0f64; 2]; 3];
                    let mut big = [false; 6];
                    for q in &quads {
                        let axis = q.normal.iter().position(|n| n.abs() == 1.0).unwrap();
                        let positive = q.normal[axis] > 0.0;
                        if is_big_outward(&spec, q) {
                            let face = face_of(axis, positive);
                            assert!(!big[face], "one face quad per face");
                            big[face] = true;
                            continue;
                        }
                        got[axis][usize::from(positive)] += f64::from(area(q));
                        // Every small quad samples an opaque texel at its centre.
                        let uv = q.uv.iter().fold([0.0f32; 2], |s, uv| {
                            [s[0] + uv[0] / 4.0, s[1] + uv[1] / 4.0]
                        });
                        let (px, py) = ((uv[0] * 64.0) as i32, (uv[1] * 64.0) as i32);
                        assert!(
                            texels.alpha(px, py) >= OPAQUE_ALPHA,
                            "{part:?} quad {q:?} samples a transparent texel"
                        );
                    }
                    assert_eq!(big, faces_with_texels, "{part:?} seed {seed}");
                    for axis in 0..3 {
                        for side in 0..2 {
                            assert!(
                                (got[axis][side] - want[axis][side]).abs() < 1e-3,
                                "{part:?} slim={slim} seed {seed} axis {axis} side {side}: \
                                 {} vs {}",
                                got[axis][side],
                                want[axis][side]
                            );
                        }
                    }
                    // No coplanar overlap (z-fighting) between any two quads.
                    for (i, a) in quads.iter().enumerate() {
                        for b in &quads[i + 1..] {
                            let axis = a.normal.iter().position(|n| n.abs() == 1.0).unwrap();
                            if b.normal[axis] == 0.0 || a.pos[0][axis] != b.pos[0][axis] {
                                continue;
                            }
                            let (u, v) = other_axes(axis);
                            let range = |q: &VoxelQuad, k: usize| {
                                let lo = q.pos.iter().map(|p| p[k]).fold(f32::MAX, f32::min);
                                let hi = q.pos.iter().map(|p| p[k]).fold(f32::MIN, f32::max);
                                (lo, hi)
                            };
                            let overlap = |k: usize| {
                                let (a0, a1) = range(a, k);
                                let (b0, b1) = range(b, k);
                                a1.min(b1) - a0.max(b0)
                            };
                            assert!(
                                overlap(u) <= 1e-6 || overlap(v) <= 1e-6,
                                "{part:?} overlapping quads {a:?} {b:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Worst case: random noise (half the outer texels opaque, nothing to merge in long runs).
    /// Merging still beats one quad per exposed cell face, and the full-outer-layer best case is
    /// 54 quads a part.
    #[test]
    fn noise_is_the_worst_case_and_merging_still_beats_one_quad_per_cell_face() {
        let rgba = noisy_skin(1);
        let texels = Texels {
            rgba: &rgba,
            width: 64,
            height: 64,
        };
        let mesh = SkinLayerMesh::build(&rgba, 64, 64, false).unwrap();
        let mut unmerged = 0;
        for part in WearPart::ALL {
            let spec = spec_for(part, false);
            let grid = solid_grid(&texels, &spec);
            let [w, h, d] = spec.dims;
            for z in -1..=d {
                for y in -1..=h {
                    for x in -1..=w {
                        let c = [x, y, z];
                        let out = grid.out_axes(c);
                        if !out.iter().any(|&o| o) || !grid.solid(c) {
                            continue;
                        }
                        for axis in 0..3 {
                            for step in [-1, 1] {
                                let mut n = c;
                                n[axis] += step;
                                let interior =
                                    grid.inside(n) && !grid.out_axes(n).iter().any(|&o| o);
                                if !grid.inside(n) || (!interior && !grid.solid(n)) {
                                    unmerged += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        let merged = mesh.quad_count();
        assert!(merged < unmerged, "merged {merged} vs unmerged {unmerged}");
        assert!(merged > 300, "{merged}");
    }

    fn still_params() -> AnimParams {
        let mut p = AnimParams::still();
        p.entity.child = false;
        p
    }

    #[test]
    fn with_no_voxel_parts_the_body_is_bit_for_bit_the_flat_body() {
        let rgba = noisy_skin(3);
        let mesh = SkinLayerMesh::build(&rgba, 64, 64, false).unwrap();
        let base = Mat4::from_translation(Vec3::new(1.5, -0.25, 3.0)) * Mat4::from_rotation_y(0.4);
        let mut p = still_params();
        p.limb_swing = 2.0;
        p.limb_swing_amount = 0.6;
        p.entity.sneaking = true;
        let (mut flat_model, mut voxel_model) =
            (PlayerModel::new(0.0, false), PlayerModel::new(0.0, false));
        let (mut flat, mut same) = (Vec::new(), Vec::new());
        flat_model.pose_and_render(&p, base, &mut flat);
        emit_player_body(&mut voxel_model, &p, base, &mesh, 0, &mut same);
        assert_eq!(
            bytemuck::cast_slice::<_, u32>(&flat[..]),
            bytemuck::cast_slice::<_, u32>(&same[..])
        );
    }

    #[test]
    fn voxel_parts_replace_the_flat_wear_and_restore_the_shared_model() {
        let transparent = skin(|x, y| if in_outer_layer(x, y) { 0 } else { 255 });
        let empty = SkinLayerMesh::build(&transparent, 64, 64, false).unwrap();
        let mut model = PlayerModel::new(0.0, false);
        let p = still_params();
        let mut out = Vec::new();
        emit_player_body(&mut model, &p, Mat4::IDENTITY, &empty, ALL_WEAR, &mut out);
        // Only the six base boxes remain: head, body, two arms, two legs.
        assert_eq!(out.len(), 6 * 36);
        // The shared model draws all twelve boxes again for the next (flat) player.
        let mut next = Vec::new();
        model.pose_and_render(&p, Mat4::IDENTITY, &mut next);
        assert_eq!(next.len(), 12 * 36);

        // A part the skin-part settings hide gets no voxels either. (In front of the camera:
        // only quads facing it are emitted, counted here with the same eye.)
        let full = SkinLayerMesh::build(&skin(|_, _| 255), 64, 64, false).unwrap();
        let jacket = model.wear.body;
        model.biped.model.get_mut(jacket).show_model = false;
        let base = Mat4::from_translation(Vec3::new(0.3, -1.0, -4.0));
        let mut hidden = Vec::new();
        emit_player_body(&mut model, &p, base, &full, ALL_WEAR, &mut hidden);
        let facing: usize = WearPart::ALL
            .iter()
            .filter(|&&part| part != WearPart::Jacket)
            .map(|&part| {
                let local = model
                    .biped
                    .model
                    .get(part.part_id(&model))
                    .local_matrix(p.scale);
                let eye = eye_in(base * local * Mat4::from_scale(Vec3::splat(p.scale)));
                full.part(part).iter().filter(|q| q.faces(eye)).count()
            })
            .sum();
        assert!(facing > 0 && facing < 5 * 54, "{facing}");
        assert_eq!(hidden.len(), 6 * 36 + facing * 6);
        assert!(!model.biped.model.get(jacket).show_model, "stays hidden");

        // A slim-arm mesh on a classic model falls back to the flat body.
        let slim = SkinLayerMesh::build(&skin(|_, _| 255), 64, 64, true).unwrap();
        model.biped.model.get_mut(jacket).show_model = true;
        let mut fallback = Vec::new();
        emit_player_body(
            &mut model,
            &p,
            Mat4::IDENTITY,
            &slim,
            ALL_WEAR,
            &mut fallback,
        );
        assert_eq!(fallback.len(), 12 * 36);
    }

    /// Report numbers over real skins: `MC_SKIN3D_STATS_DIR` = a vanilla `assets/skins` folder
    /// (PNG files, any depth), `MC_SKIN3D_STATS_MAX` = how many to read (default 400). Prints
    /// the quad distribution, the one-off build time and the per-frame emission time of 20
    /// players (flat vs voxel). Without the variable it does nothing. Timings depend on the
    /// profile and the machine's load: provisional.
    #[test]
    fn real_skin_stats() {
        let Some(dir) = std::env::var_os("MC_SKIN3D_STATS_DIR") else {
            return;
        };
        let max: usize = std::env::var("MC_SKIN3D_STATS_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(400);
        let mut files = Vec::new();
        let mut stack = vec![std::path::PathBuf::from(dir)];
        while let Some(path) = stack.pop() {
            if files.len() >= max {
                break;
            }
            if path.is_dir() {
                let mut entries: Vec<_> = std::fs::read_dir(&path)
                    .into_iter()
                    .flatten()
                    .flatten()
                    .map(|e| e.path())
                    .collect();
                entries.sort();
                entries.reverse();
                stack.extend(entries);
            } else {
                files.push(path);
            }
        }
        let mut meshes = Vec::new();
        let mut build_time = std::time::Duration::ZERO;
        for file in &files {
            let Ok(bytes) = std::fs::read(file) else {
                continue;
            };
            let Ok(texture) =
                crate::entity::skin::decode_skin(&bytes, crate::entity::SkinModel::Classic)
            else {
                continue;
            };
            let start = std::time::Instant::now();
            let mesh = SkinLayerMesh::build(&texture.rgba, 64, 64, false);
            build_time += start.elapsed();
            if let Some(mesh) = mesh {
                meshes.push(mesh);
            }
        }
        assert!(!meshes.is_empty(), "no decodable skins under the folder");
        let mut quads: Vec<usize> = meshes.iter().map(SkinLayerMesh::quad_count).collect();
        quads.sort_unstable();
        let pct = |p: f64| quads[((quads.len() - 1) as f64 * p).round() as usize];
        let mean = quads.iter().sum::<usize>() as f64 / quads.len() as f64;
        let no_layer = quads.iter().filter(|&&q| q == 0).count();
        println!(
            "skin3d stats over {} skins: quads mean {mean:.0}, p50 {}, p90 {}, p95 {}, max {}, \
             no outer layer {no_layer}; build {:.1} us per skin",
            quads.len(),
            pct(0.5),
            pct(0.9),
            pct(0.95),
            quads[quads.len() - 1],
            build_time.as_secs_f64() * 1e6 / meshes.len() as f64,
        );
        // 20 players: the 20 heaviest decile skins (a pessimistic lobby).
        let mut by_size: Vec<&SkinLayerMesh> = meshes.iter().collect();
        by_size.sort_by_key(|m| m.quad_count());
        let heavy: Vec<&SkinLayerMesh> = by_size.iter().rev().take(20).copied().collect();
        let typical: Vec<&SkinLayerMesh> = by_size
            .iter()
            .skip(by_size.len() / 2)
            .take(20)
            .copied()
            .collect();
        let mut p = still_params();
        p.limb_swing = 3.0;
        p.limb_swing_amount = 0.8;
        let base = Mat4::from_translation(Vec3::new(2.0, -1.5, 6.0));
        let mut model = PlayerModel::new(0.0, false);
        let mut out = Vec::with_capacity(1 << 20);
        let frames = 200;
        let time = |model: &mut PlayerModel,
                    out: &mut Vec<EntityVertex>,
                    set: Option<&[&SkinLayerMesh]>| {
            let start = std::time::Instant::now();
            let mut verts = 0;
            for _ in 0..frames {
                out.clear();
                for i in 0..20 {
                    match set {
                        None => model.pose_and_render(&p, base, out),
                        Some(set) => {
                            emit_player_body(model, &p, base, set[i % set.len()], ALL_WEAR, out)
                        }
                    }
                }
                verts = out.len();
            }
            (
                start.elapsed().as_secs_f64() * 1e6 / f64::from(frames),
                verts,
            )
        };
        let (flat_us, flat_v) = time(&mut model, &mut out, None);
        let (typ_us, typ_v) = time(&mut model, &mut out, Some(&typical));
        let (heavy_us, heavy_v) = time(&mut model, &mut out, Some(&heavy));
        println!(
            "20 players per frame: flat {flat_us:.0} us / {flat_v} verts; typical (p50..) \
             {typ_us:.0} us / {typ_v} verts; heaviest 20 {heavy_us:.0} us / {heavy_v} verts"
        );
    }

    /// The per-part cull keeps exactly the quads whose camera-space normal points at the camera
    /// (the origin): checked against the transformed geometry itself, under the entity flip.
    #[test]
    fn the_cull_keeps_exactly_the_camera_facing_quads() {
        let mesh = SkinLayerMesh::build(&noisy_skin(9), 64, 64, false).unwrap();
        let hat = mesh.part(WearPart::Hat);
        for base in [
            Mat4::from_translation(Vec3::new(0.4, 1.2, -3.0)),
            Mat4::from_translation(Vec3::new(-2.0, -0.5, 1.5))
                * Mat4::from_rotation_y(2.1)
                * Mat4::from_scale(Vec3::new(-0.9375, -0.9375, 0.9375)),
        ] {
            let mut out = Vec::new();
            emit_quads(hat, base, 0.0625, &mut out);
            let scaled = base * Mat4::from_scale(Vec3::splat(0.0625));
            let mut kept = 0;
            for q in hat {
                let p = scaled.transform_point3(Vec3::from(q.pos[0]));
                let centre = q
                    .pos
                    .iter()
                    .fold(Vec3::ZERO, |s, v| s + Vec3::from(*v) / 4.0);
                let n = scaled.transform_point3(centre + Vec3::from(q.normal) * 0.01)
                    - scaled.transform_point3(centre);
                let front = n.dot(p) < 0.0;
                assert_eq!(q.faces(eye_in(scaled)), front, "{q:?}");
                kept += usize::from(front);
            }
            assert_eq!(out.len(), kept * 6);
            assert!(kept > 0 && kept < hat.len());
        }
    }

    #[test]
    fn the_cache_builds_each_skin_and_arm_width_once() {
        let rgba = noisy_skin(5);
        let mut cache = Skin3dCache::default();
        let mut asked = 0;
        for _ in 0..3 {
            let mesh = cache.get_or_build("skins/abc", false, || {
                asked += 1;
                Some((&rgba[..], 64, 64))
            });
            assert!(mesh.is_some());
        }
        assert_eq!((asked, cache.builds()), (1, 1));
        let _ = cache.get_or_build("skins/abc", true, || Some((&rgba[..], 64, 64)));
        assert_eq!(cache.builds(), 2, "the other arm width is its own mesh");
        // Pixels not ready: nothing is remembered.
        assert!(cache.get_or_build("skins/later", false, || None).is_none());
        assert_eq!(cache.len(), 1);
        // A texture that cannot have layers is remembered as such.
        assert!(cache
            .get_or_build("hd", false, || Some((&rgba[..], 128, 32)))
            .is_none());
        let builds = cache.builds();
        assert!(cache
            .get_or_build("hd", false, || panic!("not rebuilt"))
            .is_none());
        assert_eq!(cache.builds(), builds);
    }
}
