//! Vanilla's occlusion-culling data: which faces of a section can see which other faces.
//!
//! A section is opaque enough to hide what is behind it far more often than a frustum test
//! alone can tell. Vanilla answers "can a ray entering this section through face A leave
//! through face B?" by flood-filling the section's non-opaque cells from each boundary cell
//! and recording which faces each connected component touches. The renderer then walks the
//! chunk graph breadth-first from the camera's own section, only stepping into a neighbour
//! when the section it came from says those two faces are connected.
//!
//! This is what stops a cave system or the far side of a mountain from being submitted at all,
//! and it is the single biggest draw-call saving in the 1.8 renderer.
//
// java: client/renderer/chunk/VisGraph.java, client/renderer/chunk/SetVisibility.java

use mc_math::Facing;

/// Index of `(x, y, z)` inside a section, in vanilla's bit packing.
/// java: `VisGraph.getIndex` — `x << 0 | y << 8 | z << 4`.
#[inline]
#[must_use]
pub const fn index(x: i32, y: i32, z: i32) -> usize {
    // java: `x << 0 | y << 8 | z << 4` — the `<< 0` is kept only as a comment.
    ((x & 15) as usize) | (((y & 15) as usize) << 8) | (((z & 15) as usize) << 4)
}

const STEP_X: usize = 1; // 16^0
const STEP_Z: usize = 16; // 16^1
const STEP_Y: usize = 256; // 16^2

/// The 6×6 face-connectivity matrix for one section. java: `SetVisibility`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetVisibility {
    bits: u64, // 36 bits used: facing.index() + facing2.index() * 6
}

impl Default for SetVisibility {
    fn default() -> Self {
        Self::all(false)
    }
}

impl SetVisibility {
    /// All pairs visible, or none. java: `SetVisibility.setAllVisible`.
    #[must_use]
    pub const fn all(visible: bool) -> Self {
        Self {
            bits: if visible { (1u64 << 36) - 1 } else { 0 },
        }
    }

    /// Mark `a`↔`b` mutually visible. java: `SetVisibility.setVisible` (symmetric).
    pub fn set_visible(&mut self, a: Facing, b: Facing) {
        let i = a.index() as u32 + b.index() as u32 * 6;
        let j = b.index() as u32 + a.index() as u32 * 6;
        self.bits |= (1 << i) | (1 << j);
    }

    /// Mark every pair drawn from `faces` mutually visible.
    /// java: `SetVisibility.setManyVisible`.
    pub fn set_many_visible(&mut self, faces: &[Facing]) {
        for &a in faces {
            for &b in faces {
                self.set_visible(a, b);
            }
        }
    }

    /// Can a ray entering through `a` leave through `b`? java: `SetVisibility.isVisible`.
    #[must_use]
    pub fn is_visible(&self, a: Facing, b: Facing) -> bool {
        self.bits & (1 << (a.index() as u32 + b.index() as u32 * 6)) != 0
    }

    /// The 15 unordered pairs of *distinct* faces, one bit each, at [`pair_bit`].
    ///
    /// This is the whole matrix as far as the terrain walk is concerned: every constructor sets
    /// `a`↔`b` symmetrically, and the walk never asks about a face paired with itself (its
    /// rule 1 rejects stepping back out of the face it entered through, which is the only way
    /// `is_visible(entered.opposite(), step)` could have `a == b`). Two bytes per section is
    /// what lets [`super::cull`]'s dense index stay small.
    #[must_use]
    pub const fn pair_mask(self) -> u16 {
        // Row `b` of the matrix holds `is_visible(a, b)` at bit `a + 6b`; its `a < b` part is
        // `b` bits wide and lands at `pair_bit`'s offset `b(b-1)/2`.
        let mut mask = 0u16;
        let mut b: u32 = 1;
        while b < 6 {
            let row = (self.bits >> (6 * b)) & ((1 << b) - 1);
            mask |= (row as u16) << (b * (b - 1) / 2);
            b += 1;
        }
        mask
    }
}

/// The bit for the unordered pair `{a, b}` in a [`SetVisibility::pair_mask`]; `0` when
/// `a == b`, which the mask does not represent.
#[inline]
#[must_use]
pub const fn pair_bit(a: Facing, b: Facing) -> u16 {
    let (a, b) = (a.index() as u32, b.index() as u32);
    if a == b {
        return 0;
    }
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    1 << (lo + hi * (hi - 1) / 2)
}

/// What `RenderGlobal.setupTerrain` actually reads from `getVisibleFacings(camera)`: whether
/// the set is empty, the one face when it has exactly one (for the view-facing removal), or
/// that it has two or more — which no removal can empty, so which faces they are never
/// matters. [`SectionOcclusion::camera_reach`] can therefore stop flooding at the second face.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraReach {
    /// No section face is reachable from the camera's cell.
    Sealed,
    /// Exactly this one face is reachable.
    One(Facing),
    /// Two or more faces are reachable.
    Many,
}

impl CameraReach {
    /// Classify a full `getVisibleFacings` answer.
    #[must_use]
    pub fn of(faces: &[Facing]) -> Self {
        match faces {
            [] => Self::Sealed,
            [face] => Self::One(*face),
            _ => Self::Many,
        }
    }
}

/// The opaque cells of one section, packed for the camera-cell query.
///
/// `RenderGlobal.getVisibleFacings` rebuilds a `VisGraph` for the section containing the
/// camera, then calls `func_178609_b` at the camera block.  The renderer has already done that
/// opacity scan while meshing, so retaining these 4096 bits (512 bytes, rather than a second
/// copy of the block states) lets the culling walk make the same query later.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SectionOcclusion {
    opaque: [u64; 64],
}

impl Default for SectionOcclusion {
    fn default() -> Self {
        Self { opaque: [0; 64] }
    }
}

impl SectionOcclusion {
    /// Whether the section-local cell is opaque according to the mesh-time `VisGraph` scan.
    #[must_use]
    pub fn is_opaque(&self, x: i32, y: i32, z: i32) -> bool {
        let i = index(x, y, z);
        self.opaque[i / 64] & (1u64 << (i % 64)) != 0
    }

    /// The boundary faces reachable from one cell. java: `VisGraph.func_178609_b`.
    ///
    /// This intentionally reconstructs the flood-fill scratch for every query. The renderer
    /// asks it for the camera section only, once per terrain walk; keeping it compact in every
    /// resident section is more valuable than saving that one 4096-byte temporary.
    #[must_use]
    pub fn visible_facings_from(&self, x: i32, y: i32, z: i32) -> Vec<Facing> {
        let mut seen = [false; 4096];
        for (word_index, &word) in self.opaque.iter().enumerate() {
            for bit in 0..64 {
                if word & (1u64 << bit) != 0 {
                    seen[word_index * 64 + bit] = true;
                }
            }
        }
        let mut faces = Vec::with_capacity(6);
        let mut queue = Vec::with_capacity(64);
        // Vanilla deliberately seeds an opaque starting cell too.
        fill(&mut seen, &mut queue, &mut faces, index(x, y, z));
        faces.sort_unstable_by_key(|facing| facing.index());
        faces
    }

    /// Whether no cell is opaque: an all-air (or never-scanned) section.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.opaque.iter().all(|&word| word == 0)
    }

    /// [`Self::visible_facings_from`] reduced to the [`CameraReach`] the walk reads (#34).
    ///
    /// Same fill, same opaque-start seeding, but on the packed bits with a fixed-size queue —
    /// no `[bool; 4096]` expansion and no heap allocation — and it stops at the second face.
    /// An all-air section answers [`CameraReach::Many`] without filling at all: with nothing
    /// opaque the fill from any cell reaches all six faces. On a void map the camera section
    /// is usually exactly that. `CameraReach::of(&self.visible_facings_from(x, y, z))` is
    /// the reference this must equal for every input.
    #[must_use]
    pub fn camera_reach(&self, x: i32, y: i32, z: i32) -> CameraReach {
        if self.is_empty() {
            return CameraReach::Many;
        }
        let mut seen = self.opaque;
        let mut queue = [0u16; 4096];
        let start = index(x, y, z);
        seen[start / 64] |= 1u64 << (start % 64);
        queue[0] = start as u16;
        let (mut head, mut tail) = (0usize, 1usize);
        let mut faces = 0u8;
        while head < tail {
            let i = queue[head] as usize;
            head += 1;
            faces |= touched_face_mask(i);
            if faces.count_ones() >= 2 {
                return CameraReach::Many;
            }
            for facing in Facing::VALUES {
                if let Some(j) = neighbour(i, facing) {
                    let bit = 1u64 << (j % 64);
                    if seen[j / 64] & bit == 0 {
                        seen[j / 64] |= bit;
                        // Each cell is queued at most once, so `tail` never passes 4096.
                        queue[tail] = j as u16;
                        tail += 1;
                    }
                }
            }
        }
        if faces == 0 {
            CameraReach::Sealed
        } else {
            CameraReach::One(Facing::VALUES[faces.trailing_zeros() as usize])
        }
    }
}

/// One section's opacity bitmap, flood-filled into a [`SetVisibility`].
/// java: `VisGraph`.
#[derive(Clone, Debug)]
pub struct VisGraph {
    opaque: [bool; 4096],
    /// java: `field_178611_f`, the count of *non*-opaque cells left.
    empty_count: u32,
}

impl Default for VisGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl VisGraph {
    /// An all-transparent section.
    #[must_use]
    pub fn new() -> Self {
        Self {
            opaque: [false; 4096],
            empty_count: 4096,
        }
    }

    /// Record that the cell at section-local `(x, y, z)` is an opaque cube.
    /// java: `VisGraph.func_178606_a`.
    pub fn set_opaque(&mut self, x: i32, y: i32, z: i32) {
        let i = index(x, y, z);
        if !self.opaque[i] {
            self.opaque[i] = true;
            self.empty_count -= 1;
        }
    }

    /// A compact copy retained with the uploaded mesh for the camera-cell culling query.
    #[must_use]
    pub fn occlusion(&self) -> SectionOcclusion {
        let mut opaque = [0u64; 64];
        for (i, &is_opaque) in self.opaque.iter().enumerate() {
            if is_opaque {
                opaque[i / 64] |= 1u64 << (i % 64);
            }
        }
        SectionOcclusion { opaque }
    }

    /// Flood-fill from every boundary cell and produce the face-connectivity matrix.
    /// java: `VisGraph.computeVisibility`.
    ///
    /// Two shortcuts come straight from vanilla: fewer than 256 opaque cells means "assume
    /// everything is visible" (the fill would not pay for itself), and a completely full
    /// section means nothing is visible at all.
    #[must_use]
    pub fn compute_visibility(&self) -> SetVisibility {
        let opaque_count = 4096 - self.empty_count;
        if opaque_count < 256 {
            return SetVisibility::all(true);
        }
        if self.empty_count == 0 {
            return SetVisibility::all(false);
        }

        let mut vis = SetVisibility::all(false);
        let mut seen = self.opaque;
        let mut faces: Vec<Facing> = Vec::with_capacity(6);
        let mut queue: Vec<usize> = Vec::with_capacity(64);

        for start in boundary_cells() {
            if seen[start] {
                continue;
            }
            faces.clear();
            fill(&mut seen, &mut queue, &mut faces, start);
            vis.set_many_visible(&faces);
        }
        vis
    }

    /// The faces reachable from one specific cell, used by the renderer to decide which of the
    /// camera section's own faces are worth stepping through.
    /// java: `VisGraph.func_178609_b`.
    #[must_use]
    pub fn visible_facings_from(&self, x: i32, y: i32, z: i32) -> Vec<Facing> {
        self.occlusion().visible_facings_from(x, y, z)
    }
}

/// java: `VisGraph.func_178604_a` — breadth-first fill recording which section faces the
/// component touches.
fn fill(seen: &mut [bool; 4096], queue: &mut Vec<usize>, faces: &mut Vec<Facing>, start: usize) {
    queue.clear();
    queue.push(start);
    seen[start] = true;
    let mut head = 0;
    while head < queue.len() {
        let i = queue[head];
        head += 1;
        touched_faces(i, faces);
        for facing in Facing::VALUES {
            if let Some(j) = neighbour(i, facing) {
                if !seen[j] {
                    seen[j] = true;
                    queue.push(j);
                }
            }
        }
    }
}

/// java: `VisGraph.func_178610_a` — which of the section's six faces this cell lies on.
fn touched_faces(i: usize, out: &mut Vec<Facing>) {
    let push = |out: &mut Vec<Facing>, f: Facing| {
        if !out.contains(&f) {
            out.push(f);
        }
    };
    let x = i & 15;
    if x == 0 {
        push(out, Facing::West);
    } else if x == 15 {
        push(out, Facing::East);
    }
    let y = (i >> 8) & 15;
    if y == 0 {
        push(out, Facing::Down);
    } else if y == 15 {
        push(out, Facing::Up);
    }
    let z = (i >> 4) & 15;
    if z == 0 {
        push(out, Facing::North);
    } else if z == 15 {
        push(out, Facing::South);
    }
}

/// [`touched_faces`] as a bitmask over [`Facing::index`].
fn touched_face_mask(i: usize) -> u8 {
    let bit = |f: Facing| 1u8 << f.index();
    let mut mask = 0;
    match i & 15 {
        0 => mask |= bit(Facing::West),
        15 => mask |= bit(Facing::East),
        _ => {}
    }
    match (i >> 8) & 15 {
        0 => mask |= bit(Facing::Down),
        15 => mask |= bit(Facing::Up),
        _ => {}
    }
    match (i >> 4) & 15 {
        0 => mask |= bit(Facing::North),
        15 => mask |= bit(Facing::South),
        _ => {}
    }
    mask
}

/// java: `VisGraph.func_178603_a`.
fn neighbour(i: usize, facing: Facing) -> Option<usize> {
    match facing {
        Facing::Down => ((i >> 8) & 15 != 0).then(|| i - STEP_Y),
        Facing::Up => ((i >> 8) & 15 != 15).then(|| i + STEP_Y),
        Facing::North => ((i >> 4) & 15 != 0).then(|| i - STEP_Z),
        Facing::South => ((i >> 4) & 15 != 15).then(|| i + STEP_Z),
        Facing::West => ((i & 15) != 0).then(|| i - STEP_X),
        Facing::East => ((i & 15) != 15).then(|| i + STEP_X),
    }
}

/// The 1352 cells on the surface of a 16³ box, in vanilla's iteration order.
/// java: `VisGraph`'s static initialiser.
fn boundary_cells() -> impl Iterator<Item = usize> {
    (0..16).flat_map(|x| {
        (0..16).flat_map(move |y| {
            (0..16)
                .filter(move |&z| x == 0 || x == 15 || y == 0 || y == 15 || z == 0 || z == 15)
                .map(move |z| index(x, y, z))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn there_are_exactly_1352_boundary_cells() {
        // java: VisGraph.field_178613_e is `new int[1352]` and the static block fills it all.
        let cells: Vec<_> = boundary_cells().collect();
        assert_eq!(cells.len(), 1352);
        let mut sorted = cells.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 1352, "boundary cells must be distinct");
        assert_eq!(16 * 16 * 16 - 14 * 14 * 14, 1352);
    }

    #[test]
    fn index_packing_matches_vanilla() {
        // java: getIndex(x, y, z) = x << 0 | y << 8 | z << 4
        assert_eq!(index(1, 0, 0), 1);
        assert_eq!(index(0, 1, 0), 256);
        assert_eq!(index(0, 0, 1), 16);
        assert_eq!(index(15, 15, 15), 4095);
    }

    #[test]
    fn an_empty_section_connects_every_face_to_every_other() {
        let g = VisGraph::new();
        let v = g.compute_visibility();
        for a in Facing::VALUES {
            for b in Facing::VALUES {
                assert!(v.is_visible(a, b), "{a:?} -> {b:?}");
            }
        }
    }

    #[test]
    fn a_solid_section_connects_nothing() {
        let mut g = VisGraph::new();
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    g.set_opaque(x, y, z);
                }
            }
        }
        let v = g.compute_visibility();
        for a in Facing::VALUES {
            for b in Facing::VALUES {
                assert!(!v.is_visible(a, b));
            }
        }
    }

    #[test]
    fn a_solid_wall_separates_the_two_halves() {
        // Fill everything except two disjoint pockets: a north slab and a south slab, split by
        // a solid plane at z == 8. North must not see South.
        let mut g = VisGraph::new();
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    // Leave the z == 0 and z == 15 planes open, wall off everything else.
                    if z != 0 && z != 15 {
                        g.set_opaque(x, y, z);
                    }
                }
            }
        }
        let v = g.compute_visibility();
        assert!(v.is_visible(Facing::North, Facing::North));
        assert!(v.is_visible(Facing::South, Facing::South));
        assert!(
            !v.is_visible(Facing::North, Facing::South),
            "a solid wall must break north-south visibility"
        );
    }

    #[test]
    fn a_tunnel_connects_exactly_its_two_ends() {
        // Solid section with a single 1x1 tunnel along z.
        let mut g = VisGraph::new();
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    if !(x == 8 && y == 8) {
                        g.set_opaque(x, y, z);
                    }
                }
            }
        }
        let v = g.compute_visibility();
        assert!(v.is_visible(Facing::North, Facing::South));
        assert!(v.is_visible(Facing::South, Facing::North));
        assert!(!v.is_visible(Facing::Up, Facing::Down));
        assert!(!v.is_visible(Facing::West, Facing::East));
        assert!(!v.is_visible(Facing::North, Facing::Up));
    }

    #[test]
    fn a_sparse_section_short_circuits_to_all_visible() {
        // java: `if (4096 - field_178611_f < 256) setAllVisible(true)`.
        let mut g = VisGraph::new();
        for i in 0..255 {
            g.set_opaque(i % 16, i / 16, 0);
        }
        assert!(g.compute_visibility().is_visible(Facing::Up, Facing::Down));
    }

    #[test]
    fn an_opaque_start_cell_still_floods_into_adjacent_air() {
        // java: `func_178604_a` always enqueues and visits its start index, even when
        // `field_178612_d` already contains it.  The old Rust early return produced no faces.
        // Keeping the start away from the boundary proves that the surrounding air, rather than
        // the opaque cell itself, is what carries the fill to all six section faces.
        let mut g = VisGraph::new();
        g.set_opaque(8, 8, 8);

        let faces = g.visible_facings_from(8, 8, 8);

        for facing in Facing::VALUES {
            assert!(faces.contains(&facing), "missing {facing:?}: {faces:?}");
        }
        assert_eq!(faces.len(), 6, "the result is an EnumSet in vanilla");
    }

    #[test]
    fn visible_facings_iterate_in_enumset_ordinal_order() {
        // `touched_faces` discovers this corner as W, D, N, but Java stores those insertions in
        // an EnumSet and therefore exposes them as D, N, W.  Returning discovery order is an
        // observable difference now that the Rust port represents the set with a Vec.
        let faces = VisGraph::new().visible_facings_from(0, 0, 0);
        assert_eq!(faces, Facing::VALUES);
    }

    #[test]
    fn compact_camera_snapshot_preserves_the_opaque_start_fill() {
        let mut graph = VisGraph::new();
        // A nearly solid section with a narrow north-to-south tunnel. The camera is placed in
        // an opaque cell next to it, exercising Java's unusual unconditional seed as well as
        // the packed snapshot's bit indexing.
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    if !(x == 8 && y == 8) {
                        graph.set_opaque(x, y, z);
                    }
                }
            }
        }
        graph.set_opaque(7, 8, 8);
        let snapshot = graph.occlusion();
        assert!(snapshot.is_opaque(7, 8, 8));
        assert!(!snapshot.is_opaque(8, 8, 8));
        assert_eq!(
            snapshot.visible_facings_from(7, 8, 8),
            vec![Facing::North, Facing::South],
            "the retained 512-byte mesh snapshot must answer VisGraph.func_178609_b exactly"
        );
    }

    #[test]
    fn an_opaque_camera_cell_can_reach_exactly_one_face() {
        let mut graph = VisGraph::new();
        // The camera starts in the solid cell at (8,8,8). Its north neighbour opens onto a
        // one-block-wide tunnel, while every other cell is solid. `func_178604_a` still seeds
        // that opaque camera cell and reaches exactly NORTH through the adjacent air.
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    if !(x == 8 && y == 8 && z < 8) {
                        graph.set_opaque(x, y, z);
                    }
                }
            }
        }
        assert_eq!(
            graph.occlusion().visible_facings_from(8, 8, 8),
            vec![Facing::North]
        );
    }

    /// A small deterministic generator for the randomized equivalence tests below (no `rand`
    /// in this crate's dev-dependencies, and a fixed stream keeps failures reproducible).
    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Every `SetVisibility` the code can build, as a function of random connected-face groups:
    /// `compute_visibility` only ever calls `set_many_visible` per component.
    fn random_visibility(rng: &mut XorShift) -> SetVisibility {
        match rng.next() % 8 {
            0 => SetVisibility::all(true),
            1 => SetVisibility::all(false),
            _ => {
                let mut vis = SetVisibility::all(false);
                for _ in 0..(rng.next() % 4) {
                    let group: Vec<Facing> = Facing::VALUES
                        .into_iter()
                        .filter(|_| rng.next().is_multiple_of(3))
                        .collect();
                    vis.set_many_visible(&group);
                }
                vis
            }
        }
    }

    /// #7: the walk reads `pair_mask & pair_bit(a, b)` instead of `is_visible(a, b)`. They must
    /// agree on every distinct pair, in both orders, for every matrix the code can build.
    #[test]
    fn the_pair_mask_answers_is_visible_for_every_distinct_pair() {
        let mut bits_seen = std::collections::HashSet::new();
        for a in Facing::VALUES {
            for b in Facing::VALUES {
                let bit = pair_bit(a, b);
                assert_eq!(bit, pair_bit(b, a), "unordered");
                if a == b {
                    assert_eq!(bit, 0);
                } else {
                    assert_eq!(bit.count_ones(), 1);
                    bits_seen.insert(bit);
                }
            }
        }
        assert_eq!(bits_seen.len(), 15, "15 distinct pairs, 15 distinct bits");
        assert_eq!(SetVisibility::all(true).pair_mask(), 0x7fff);
        assert_eq!(SetVisibility::all(false).pair_mask(), 0);

        let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
        for _ in 0..5_000 {
            let vis = random_visibility(&mut rng);
            let mask = vis.pair_mask();
            assert_eq!(
                mask & 0x8000,
                0,
                "bit 15 is free for the dense index's presence flag"
            );
            for a in Facing::VALUES {
                for b in Facing::VALUES {
                    if a != b {
                        assert_eq!(
                            mask & pair_bit(a, b) != 0,
                            vis.is_visible(a, b),
                            "{vis:?} {a:?}->{b:?}"
                        );
                    }
                }
            }
        }
    }

    /// #34: the allocation-free, early-exit camera query must classify exactly as the full
    /// `getVisibleFacings` port does, for every cell of a spread of opacity patterns —
    /// including all-air (the fast path), all-solid, and the tunnel and opaque-start shapes
    /// the tests above pin.
    #[test]
    fn camera_reach_classifies_exactly_like_visible_facings_from() {
        let mut patterns: Vec<VisGraph> = vec![VisGraph::new()];
        let mut solid = VisGraph::new();
        for i in 0..4096 {
            solid.set_opaque(i & 15, (i >> 8) & 15, (i >> 4) & 15);
        }
        patterns.push(solid);
        let mut tunnel = VisGraph::new();
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    if !(x == 8 && y == 8) {
                        tunnel.set_opaque(x, y, z);
                    }
                }
            }
        }
        patterns.push(tunnel);
        // Random densities, from a few blocks (an island's edge) to nearly solid.
        let mut rng = XorShift(0x2545_f491_4f6c_dd1d);
        for density in [1u64, 5, 20, 45, 60, 75, 90, 97] {
            let mut g = VisGraph::new();
            for i in 0..4096 {
                if rng.next() % 100 < density {
                    g.set_opaque(i & 15, (i >> 8) & 15, (i >> 4) & 15);
                }
            }
            patterns.push(g);
        }
        // A ground slab with air above, which is what standing on an island looks like.
        let mut ground = VisGraph::new();
        for x in 0..16 {
            for z in 0..16 {
                for y in 0..4 {
                    ground.set_opaque(x, y, z);
                }
            }
        }
        patterns.push(ground);

        let mut classes = [0usize; 3];
        for g in &patterns {
            let occ = g.occlusion();
            for i in 0..4096 {
                let (x, y, z) = (i & 15, (i >> 8) & 15, (i >> 4) & 15);
                let reference = CameraReach::of(&occ.visible_facings_from(x, y, z));
                assert_eq!(occ.camera_reach(x, y, z), reference, "cell ({x},{y},{z})");
                classes[match reference {
                    CameraReach::Sealed => 0,
                    CameraReach::One(_) => 1,
                    CameraReach::Many => 2,
                }] += 1;
            }
        }
        assert!(
            classes.iter().all(|&n| n > 0),
            "the sweep must exercise every class: {classes:?}"
        );
    }

    #[test]
    fn set_visibility_is_symmetric() {
        let mut v = SetVisibility::all(false);
        v.set_visible(Facing::North, Facing::Up);
        assert!(v.is_visible(Facing::North, Facing::Up));
        assert!(v.is_visible(Facing::Up, Facing::North));
        assert!(!v.is_visible(Facing::North, Facing::Down));
    }
}
