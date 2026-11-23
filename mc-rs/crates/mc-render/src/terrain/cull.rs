//! Which sections to draw: vanilla's frustum + occlusion walk.
//!
//! `RenderGlobal.setupTerrain` does a breadth-first search over the chunk-section graph
//! starting at the camera's own section. A neighbour is visited only when
//!
//! 1. the step is not back through a face the path already came *out* of (`setFacing`),
//! 2. the section being stepped out of says those two faces are connected
//!    ([`SetVisibility`](super::visgraph::SetVisibility)),
//! 3. the neighbour has not already been reached this frame, and
//! 4. the neighbour survives the frustum test.
//!
//! Rule 1 is the part that is easy to miss and the part that does most of the work: it forbids
//! a path from doubling back, which is what keeps the search from leaking round corners into
//! the whole world.
//!
//! # Cost (S23)
//!
//! This walk was measured at **~60% of terrain's per-frame CPU at render distance 12** (0.102 ms
//! of a 0.168 ms share; 0.198 ms at render distance 16), which made it the largest remaining
//! frame-loop cost. Two things were done about it, and they are deliberately different in kind:
//!
//! * [`SectionWalk`] keeps its scratch buffers between frames and replaces the per-frame
//!   `HashSet<ChunkSectionPos>` with a **dense bitset** over the render-distance cube. A
//!   render-distance-12 walk touches up to 25 × 25 × 16 = 10,000 sections, so the visited set is
//!   the walk's inner loop; a bit test replaces a `SipHash` of twelve bytes. This changes no
//!   answer at all — it is the same set, stored differently — and it applies on **every** frame.
//! * [`SectionWalk::run`] also *caches* the result. This one is a correctness risk (a stale
//!   visible set drops chunks the player can see), so the invalidation rule is vanilla's own and
//!   nothing looser: `RenderGlobal.setupTerrain` recomputes `renderInfos` whenever the camera's
//!   position **or** rotation moved at all, or the rebuild queue is non-empty. The key here is
//!   the camera position's bit pattern, all six frustum planes, the render distance, the
//!   occlusion toggle, and a caller-supplied generation counter for the loaded set — so a camera
//!   that moved by one ten-thousandth of a block, a camera that turned, a chunk that loaded, a
//!   chunk that unloaded and a section that re-meshed each force a full re-walk.
//
// java: client/renderer/RenderGlobal.java:774-940 (setupTerrain, getVisibleFacings,
//       getRenderChunkOffset) and RenderGlobal.ContainerLocalRenderInformation

use std::collections::{HashSet, VecDeque};

use mc_math::{BlockPos, ChunkSectionPos, DVec3, Facing};

use super::frustum::Frustum;
use super::visgraph::{pair_bit, CameraReach, SetVisibility};

/// A face set as a bitmask over [`Facing::index`]. java: `ContainerLocalRenderInformation.setFacing`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct FaceSet(u8);

impl FaceSet {
    fn contains(self, f: Facing) -> bool {
        self.0 & (1 << f.index()) != 0
    }
    fn with(self, f: Facing) -> Self {
        Self(self.0 | (1 << f.index()))
    }
}

/// One section the renderer decided to draw, and how the search reached it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisibleSection {
    /// The section itself.
    pub pos: ChunkSectionPos,
    /// The face the search entered through, or `None` for the camera's own section.
    pub entered_through: Option<Facing>,
    /// Breadth-first depth from the camera's section. java: `ContainerLocalRenderInformation.counter`.
    pub depth: u32,
}

/// What the walk needs to know about the world it is walking.
pub trait SectionSource {
    /// The section's visibility matrix, or `None` when it has no mesh yet.
    ///
    /// `None` **blocks** the walk when smart culling is on, for every section except the seed.
    /// java: the step test is `renderchunk3.getCompiledChunk().isVisible(from, to)`
    /// (`RenderGlobal.java:895`), and a `RenderChunk` holds `CompiledChunk.DUMMY` until its
    /// geometry is built — `DUMMY.isVisible` is a hard-coded `return false`
    /// (`CompiledChunk.java:11-25`). The `enumfacing2 == null` disjunct is what exempts the
    /// camera's own section, so a camera standing in an un-meshed section still sees out.
    ///
    /// This doc used to claim the opposite ("`None` means pass through"), reasoning from
    /// `ViewFrustum` allocating a `RenderChunk` per section. That premise is true and the
    /// conclusion does not follow: the gate is the *compiled* chunk, not the `RenderChunk`.
    ///
    /// An all-**air** section is a different case and must still return `Some`: vanilla compiles
    /// it normally and `computeVisibility` on an empty `VisGraph` yields all-visible.
    /// `TerrainRenderer::upload` registers empty meshes for exactly this reason.
    fn visibility(&self, pos: ChunkSectionPos) -> Option<SetVisibility>;

    /// [`Self::visibility`] as a [`SetVisibility::pair_mask`], which is all the walk reads.
    ///
    /// This is the per-section query of the walk's inner loop. The default derives it from
    /// [`Self::visibility`]; the live renderer overrides it with a dense index so the walk does
    /// not hash a position per visited section (#7). An override must answer exactly
    /// `self.visibility(pos).map(SetVisibility::pair_mask)`.
    fn visibility_pairs(&self, pos: ChunkSectionPos) -> Option<u16> {
        self.visibility(pos).map(SetVisibility::pair_mask)
    }

    /// Boundary faces the camera can reach from its exact block inside `pos`.
    ///
    /// This is the one camera-section-only query in `RenderGlobal.setupTerrain`:
    /// `getVisibleFacings(camera_block)`. Most lightweight callers only retain the compiled
    /// section's [`SetVisibility`], so the conservative default preserves the ordinary graph
    /// walk. The live terrain source retains the compact mesh-time opacity snapshot and returns
    /// the exact faces.
    fn camera_visible_facings(
        &self,
        _pos: ChunkSectionPos,
        _local_x: i32,
        _local_y: i32,
        _local_z: i32,
    ) -> Option<Vec<Facing>> {
        None
    }

    /// [`Self::camera_visible_facings`] reduced to what the walk reads (#34). The default
    /// classifies the full answer; a source that keeps the camera section's opacity bits can
    /// answer it without building the face list (see
    /// [`SectionOcclusion::camera_reach`](super::visgraph::SectionOcclusion::camera_reach)).
    /// An override must answer exactly
    /// `self.camera_visible_facings(..).map(|faces| CameraReach::of(&faces))`.
    fn camera_reach(
        &self,
        pos: ChunkSectionPos,
        local_x: i32,
        local_y: i32,
        local_z: i32,
    ) -> Option<CameraReach> {
        self.camera_visible_facings(pos, local_x, local_y, local_z)
            .map(|faces| CameraReach::of(&faces))
    }

    /// Whether the camera's cell is an opaque cube, when the source knows.
    fn camera_cell_opaque(
        &self,
        _pos: ChunkSectionPos,
        _local_x: i32,
        _local_y: i32,
        _local_z: i32,
    ) -> bool {
        false
    }
}

impl<F> SectionSource for F
where
    F: Fn(ChunkSectionPos) -> Option<SetVisibility>,
{
    fn visibility(&self, pos: ChunkSectionPos) -> Option<SetVisibility> {
        self(pos)
    }
}

/// Sections in one chunk column: the world is 256 blocks tall.
const COLUMN_SECTIONS: i32 = 16;

/// Above this render distance the dense visited-set stops being worth its memory and the walk
/// falls back to a hash set. Vanilla's maximum render distance is 32, so nothing in ordinary use
/// reaches this; it exists because [`super::phase::TerrainRenderer::render_distance`] is a public
/// `u32` and a bitset sized from an unvalidated number is an allocation waiting to go wrong.
///
/// At 64 the dense set is `129 × 129 × 16` bits = 33 KiB, allocated once and reused.
const MAX_DENSE_DISTANCE: u32 = 64;

/// The walk queue's "no dense index" marker: every entry has one while the visited set is
/// dense, none in the sparse fallback.
const NO_INDEX: u32 = u32::MAX;

/// "Already reached this walk", as a dense bitset over the render-distance cube.
///
/// The walk's inner loop asks this question once per candidate neighbour — up to six times per
/// visited section — so at render distance 12 it runs tens of thousands of times per frame. A
/// `HashSet<ChunkSectionPos>` answers it with a `SipHash` of twelve bytes plus a probe; this
/// answers it with a shift and a mask, and reuses one allocation across frames instead of
/// growing a new table every frame.
///
/// It is **exactly** the same set: [`Visited::insert`] returns what
/// [`HashSet::insert`] returns for every input, including inputs outside the cube, which fall
/// through to the `overflow` set rather than being silently accepted twice.
#[derive(Debug, Default)]
struct Visited {
    bits: Vec<u64>,
    /// Positions the dense cube cannot address. Empty in practice — every caller filters to the
    /// render distance before inserting — but it is what makes [`Visited::insert`] total.
    overflow: HashSet<ChunkSectionPos>,
    dense: bool,
    start: ChunkSectionPos,
    /// `2 * render_distance + 1`.
    side: i32,
    /// `render_distance`, the offset that maps a signed delta onto `0..side`.
    half: i32,
}

impl Visited {
    /// Empty the set and re-centre it on `start` for a walk at `render_distance`.
    fn reset(&mut self, start: ChunkSectionPos, render_distance: u32) {
        self.start = start;
        self.overflow.clear();
        self.dense = render_distance <= MAX_DENSE_DISTANCE;
        if self.dense {
            self.half = render_distance as i32;
            self.side = self.half * 2 + 1;
            let cells = (self.side as usize) * (self.side as usize) * (COLUMN_SECTIONS as usize);
            let words = cells.div_ceil(64);
            // `clear` + `resize` keeps the allocation and zeroes it, which is the whole point:
            // this is a memset of ~1.2 KiB at render distance 12, not a fresh hash table.
            self.bits.clear();
            self.bits.resize(words, 0);
        } else {
            self.bits.clear();
        }
    }

    /// Bit index of `pos`, or `None` when it is outside the cube this set can address.
    fn dense_index(&self, pos: ChunkSectionPos) -> Option<usize> {
        let dx = pos.x.checked_sub(self.start.x)?.checked_add(self.half)?;
        let dz = pos.z.checked_sub(self.start.z)?.checked_add(self.half)?;
        if dx < 0 || dz < 0 || dx >= self.side || dz >= self.side {
            return None;
        }
        if !(0..COLUMN_SECTIONS).contains(&pos.y) {
            return None;
        }
        let plane = (dx as usize) * (self.side as usize) + (dz as usize);
        Some(plane * (COLUMN_SECTIONS as usize) + pos.y as usize)
    }

    /// Insert `pos`, returning `true` when it was not already present — `HashSet::insert`'s
    /// contract, which is what the walk's rule 3 is written against.
    fn insert(&mut self, pos: ChunkSectionPos) -> bool {
        if self.dense {
            if let Some(i) = self.dense_index(pos) {
                let word = i / 64;
                let mask = 1u64 << (i % 64);
                if self.bits[word] & mask != 0 {
                    return false;
                }
                self.bits[word] |= mask;
                return true;
            }
        }
        self.overflow.insert(pos)
    }

    /// `pos`'s dense index as the walk carries it in its queue, or [`NO_INDEX`] when the set
    /// cannot address it (the sparse fallback, or outside the cube).
    fn queue_index(&self, pos: ChunkSectionPos) -> u32 {
        if !self.dense {
            return NO_INDEX;
        }
        self.dense_index(pos).map_or(NO_INDEX, |i| i as u32)
    }

    /// How far one step along `facing` moves a dense index: the index is linear in the section
    /// delta, `(dx · side + dz) · 16 + y`, so a neighbour that is still inside the cube is its
    /// parent's index plus this.
    fn step_stride(&self, facing: Facing) -> isize {
        let column = COLUMN_SECTIONS as isize;
        facing.front_offset_x() as isize * self.side as isize * column
            + facing.front_offset_z() as isize * column
            + facing.front_offset_y() as isize
    }

    /// [`Visited::insert`] for a dense index the caller already has (see
    /// [`Visited::queue_index`]): the same answer without re-deriving the index.
    fn insert_index(&mut self, i: usize) -> bool {
        debug_assert!(self.dense);
        let (word, mask) = (i / 64, 1u64 << (i % 64));
        if self.bits[word] & mask != 0 {
            return false;
        }
        self.bits[word] |= mask;
        true
    }

    /// Whether `pos` was inserted since the last [`Visited::reset`].
    fn contains(&self, pos: ChunkSectionPos) -> bool {
        if self.dense {
            if let Some(i) = self.dense_index(pos) {
                return self.bits[i / 64] & (1u64 << (i % 64)) != 0;
            }
        }
        self.overflow.contains(&pos)
    }
}

/// Everything [`visible_sections`]' answer is a function of.
///
/// The camera is compared by **bit pattern** rather than by value, so `-0.0` and `0.0` are
/// different keys and a `NaN` camera never matches itself. Both are the conservative direction:
/// a key that fails to match costs a re-walk, a key that matches when it should not costs a
/// wrong frame.
#[derive(Clone, Copy, Debug, PartialEq)]
struct WalkKey {
    camera: [u64; 3],
    view: Option<[u64; 3]>,
    frustum: Frustum,
    render_distance: u32,
    smart_culling: bool,
    spectator: bool,
    world_generation: u64,
}

/// A reusable terrain walk: the scratch buffers, plus a cache of the last result.
///
/// [`visible_sections`] is this with a fresh instance every call. The renderer keeps one instead,
/// which is what makes the buffer reuse and the result cache possible. See the module docs for
/// why the two are different kinds of change.
#[derive(Debug, Default)]
pub struct SectionWalk {
    visited: Visited,
    /// The sections in [`SectionWalk::out`], for constant-time membership. Not `visited`: java's
    /// `setFrameIndex` marks a section before its frustum test, so that set also holds sections
    /// the walk rejected.
    reached: Visited,
    /// This walk's frustum, tabulated per section coordinate. Bit-identical to
    /// [`Frustum::contains_section`], and the walk asks it once per new neighbour.
    frustum_table: super::frustum::SectionFrustum,
    /// Each entry carries its section's dense index (or [`NO_INDEX`]), so a neighbour's index is
    /// one addition instead of a bounds-checked re-derivation per step.
    queue: VecDeque<(VisibleSection, FaceSet, u32)>,
    out: Vec<VisibleSection>,
    /// What [`SectionWalk::out`] is the answer to. `None` means "nothing usable is cached".
    key: Option<WalkKey>,
    hits: u64,
    misses: u64,
}

impl SectionWalk {
    /// A walk with no cached result.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Run the walk, or return the cached answer when nothing it depends on has moved.
    ///
    /// `world_generation` is the caller's **loaded-set version**: bump it whenever a section is
    /// added, removed, or re-meshed with a different visibility matrix. It is the one input this
    /// module cannot observe for itself, and getting it wrong is exactly the stale-visible-set
    /// failure the cache risks — so it is a required parameter rather than something with a
    /// default. [`super::phase::TerrainRenderer`] bumps it in `upload`, `remove` and on eviction.
    ///
    /// Every other input is compared here: the camera position, all six frustum planes (which is
    /// how rotation, field of view, aspect and the far plane are covered), the render distance
    /// and the occlusion toggle.
    pub fn run(
        &mut self,
        source: &impl SectionSource,
        frustum: &Frustum,
        camera: DVec3,
        render_distance: u32,
        smart_culling: bool,
        world_generation: u64,
    ) -> &[VisibleSection] {
        self.run_with_view(
            source,
            frustum,
            camera,
            None,
            render_distance,
            smart_culling,
            false,
            world_generation,
        )
    }

    /// As [`Self::run`], with the camera's world-space view vector for vanilla's one-face
    /// camera-section shortcut. Callers without retained camera-cell opacity can use
    /// [`Self::run`] unchanged; sources that do provide it receive the exact shortcut here.
    // These parameters are the explicit inputs to vanilla's setupTerrain cache key; collapsing
    // them would make the invalidation contract less visible and harder to audit.
    #[allow(clippy::too_many_arguments)]
    pub fn run_with_view(
        &mut self,
        source: &impl SectionSource,
        frustum: &Frustum,
        camera: DVec3,
        view: Option<DVec3>,
        render_distance: u32,
        smart_culling: bool,
        spectator: bool,
        world_generation: u64,
    ) -> &[VisibleSection] {
        let key = WalkKey {
            camera: [camera.x.to_bits(), camera.y.to_bits(), camera.z.to_bits()],
            view: view.map(|v| [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]),
            frustum: *frustum,
            render_distance,
            smart_culling,
            spectator,
            world_generation,
        };
        if self.key == Some(key) {
            self.hits += 1;
            return &self.out;
        }
        mc_core::profile_scope!("terrain.cull_walk");
        self.misses += 1;
        // Dropped *before* the walk, not replaced after it: if the walk panics or the source
        // misbehaves, the next frame must re-walk rather than serve a half-built list.
        self.key = None;
        // The walk marks `reached` as it emits each section into `out`, so the two are the same
        // set; it shares `visited`'s layout (same start section, same render distance).
        self.reached
            .reset(BlockPos::from_vec3(camera).section_pos(), render_distance);
        self.walk(
            source,
            frustum,
            camera,
            view,
            render_distance,
            smart_culling,
            spectator,
        );
        self.key = Some(key);
        &self.out
    }

    /// The last result, without re-running anything. Empty before the first [`SectionWalk::run`].
    #[must_use]
    pub fn sections(&self) -> &[VisibleSection] {
        &self.out
    }

    /// Whether the last walk reached `pos`, or `None` when no walk result is held.
    /// java: membership of `RenderGlobal.renderInfos`.
    #[must_use]
    pub fn reached(&self, pos: ChunkSectionPos) -> Option<bool> {
        self.key.is_some().then(|| self.reached.contains(pos))
    }

    /// Force the next [`SectionWalk::run`] to recompute.
    pub fn invalidate(&mut self) {
        self.key = None;
    }

    /// How many [`SectionWalk::run`] calls were answered from the cache, and how many walked.
    /// For the F3 overlay and for tests that need to prove the cache is doing what it claims.
    #[must_use]
    pub fn cache_counts(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    /// Consume the walk and take its result.
    #[must_use]
    pub fn into_sections(self) -> Vec<VisibleSection> {
        self.out
    }
}

/// Run vanilla's terrain walk.
///
/// `camera` is the eye position in world space, `render_distance` the setting in chunks, and
/// `smart_culling` vanilla's `Minecraft.renderChunksMany` (the "advanced OpenGL"/occlusion
/// toggle) — with it off, only the frustum and the render distance apply.
///
/// Returns the sections to draw, in the order vanilla submits them (near to far, because the
/// search is breadth-first from the camera).
///
/// This allocates a fresh [`SectionWalk`] per call, so a caller running it every frame should
/// keep one instead — see [`SectionWalk::run`]. The two produce identical output, which
/// `the_cached_walk_answers_exactly_what_the_one_shot_walk_answers` asserts over a sweep.
#[must_use]
pub fn visible_sections(
    source: &impl SectionSource,
    frustum: &Frustum,
    camera: DVec3,
    render_distance: u32,
    smart_culling: bool,
) -> Vec<VisibleSection> {
    let mut walk = SectionWalk::new();
    walk.run(source, frustum, camera, render_distance, smart_culling, 0);
    walk.into_sections()
}

/// [`visible_sections`] with the camera's view vector. The live terrain renderer uses this
/// form, matching `RenderGlobal.setupTerrain`'s camera-cell early-out.
#[must_use]
pub fn visible_sections_with_view(
    source: &impl SectionSource,
    frustum: &Frustum,
    camera: DVec3,
    view: DVec3,
    render_distance: u32,
    smart_culling: bool,
) -> Vec<VisibleSection> {
    let mut walk = SectionWalk::new();
    walk.run_with_view(
        source,
        frustum,
        camera,
        Some(view),
        render_distance,
        smart_culling,
        false,
        0,
    );
    walk.into_sections()
}

impl SectionWalk {
    /// The walk itself, into this instance's reused buffers.
    ///
    /// Byte for byte the algorithm S11 wrote; only where `seen`, `queue` and `out` live changed.
    // The walk keeps Java's branch inputs separate because each one controls a distinct gate.
    #[allow(clippy::too_many_arguments)]
    fn walk(
        &mut self,
        source: &impl SectionSource,
        frustum: &Frustum,
        camera: DVec3,
        view: Option<DVec3>,
        render_distance: u32,
        mut smart_culling: bool,
        spectator: bool,
    ) {
        let (seen, reached, queue, out) = (
            &mut self.visited,
            &mut self.reached,
            &mut self.queue,
            &mut self.out,
        );
        let eye = BlockPos::from_vec3(camera);
        let start = eye.section_pos();
        out.clear();
        queue.clear();
        seen.reset(start, render_distance);
        let strides = Facing::VALUES.map(|facing| seen.step_stride(facing));
        // Every section the walk tests is within `render_distance` columns of `start`.
        let section_frustum = &mut self.frustum_table;
        section_frustum.rebuild(frustum, start, render_distance);
        let section_frustum = &*section_frustum;

        // java: when the camera is inside the world, the search starts at its own section
        // unconditionally — it is never frustum-culled, because it surrounds the viewer.
        if (0..16).contains(&start.y) {
            seen.insert(start);
            let local_x = eye.x & 15;
            let local_y = eye.y & 15;
            let local_z = eye.z & 15;
            let camera_section_only = source
                .camera_reach(start, local_x, local_y, local_z)
                .is_some_and(|reach| match reach {
                    CameraReach::Sealed => true,
                    // java: `if (set1.size() == 1) set1.remove(getFacingFromVector(view).opposite())`.
                    // Crucially, this does *not* constrain a multi-face seed; it only turns an
                    // empty result into the one-section fast path below.
                    CameraReach::One(face) => view.is_some_and(|view| {
                        face == Facing::from_vector(view.x as f32, view.y as f32, view.z as f32)
                            .opposite()
                    }),
                    // Two or more faces: one removal cannot empty the set.
                    CameraReach::Many => false,
                });
            if camera_section_only && !spectator {
                // The player cannot see out of this section. `RenderGlobal` appends only its
                // `ContainerLocalRenderInformation` and skips the BFS. A spectator inside an
                // opaque block is the one exception: Java seeds every facing and continues the
                // ordinary walk so a clipped camera cannot hide the surrounding world.
                out.push(VisibleSection {
                    pos: start,
                    entered_through: None,
                    depth: 0,
                });
                reached.insert(start);
                return;
            }
            // java: `RenderGlobal.setupTerrain` (`:853`) sets `flag1 = false` for a spectator
            // whose eye is in an opaque cube, so the search keeps only its frustum test.
            if spectator && source.camera_cell_opaque(start, local_x, local_y, local_z) {
                smart_culling = false;
            }
            queue.push_back((
                VisibleSection {
                    pos: start,
                    entered_through: None,
                    depth: 0,
                },
                FaceSet::default(),
                seen.queue_index(start),
            ));
        } else {
            // java: the camera is above or below the world; seed from the top/bottom slab of
            // every column in range instead.
            let y = if eye.y > 0 { 15 } else { 0 };
            let r = render_distance as i32;
            for dx in -r..=r {
                for dz in -r..=r {
                    let pos = ChunkSectionPos::new(start.x + dx, y, start.z + dz);
                    if section_frustum.contains_section(pos) && seen.insert(pos) {
                        queue.push_back((
                            VisibleSection {
                                pos,
                                entered_through: None,
                                depth: 0,
                            },
                            FaceSet::default(),
                            seen.queue_index(pos),
                        ));
                    }
                }
            }
        }

        while let Some((info, path, index)) = queue.pop_front() {
            out.push(info);
            if index == NO_INDEX {
                reached.insert(info.pos);
            } else {
                reached.insert_index(index as usize);
            }
            // Only rule 2 reads it, so with smart culling off the source is not asked at all.
            let vis = if smart_culling && info.entered_through.is_some() {
                source.visibility_pairs(info.pos)
            } else {
                None
            };

            for step in Facing::VALUES {
                // Rule 1: never double back along the path already taken.
                if smart_culling && path.contains(step.opposite()) {
                    continue;
                }
                // Rule 2: the section must connect the face we came in through to the one we are
                // leaving through.
                if smart_culling {
                    if let Some(entered) = info.entered_through {
                        // java: `... || enumfacing2 == null ||
                        // renderchunk3.getCompiledChunk().isVisible(enumfacing2.getOpposite(),
                        // enumfacing1)`. An un-meshed section holds `CompiledChunk.DUMMY`, whose
                        // `isVisible` is `false`, so it blocks every step out of itself.
                        //
                        // The pair mask has no diagonal, and needs none: `entered` is on `path`,
                        // so rule 1 has already rejected `step == entered.opposite()`.
                        match vis {
                            Some(pairs) if pairs & pair_bit(entered.opposite(), step) != 0 => {}
                            _ => continue,
                        }
                    }
                }
                let next = ChunkSectionPos::new(
                    info.pos.x + step.front_offset_x(),
                    info.pos.y + step.front_offset_y(),
                    info.pos.z + step.front_offset_z(),
                );
                // java: getRenderChunkOffset — out of the world vertically, or beyond the render
                // distance horizontally, is not a chunk at all.
                if !(0..16).contains(&next.y) {
                    continue;
                }
                if (next.x - start.x).unsigned_abs() > render_distance
                    || (next.z - start.z).unsigned_abs() > render_distance
                {
                    continue;
                }
                // The two tests above keep `next` inside the dense cube, so its index is the
                // parent's plus the step's stride — the same bit `seen.insert(next)` would test.
                let next_index = if index == NO_INDEX {
                    seen.queue_index(next)
                } else {
                    (index as isize + strides[step.index() as usize]) as u32
                };
                let fresh = if next_index == NO_INDEX {
                    seen.insert(next)
                } else {
                    seen.insert_index(next_index as usize)
                };
                if !fresh {
                    continue;
                }
                if !section_frustum.contains_section(next) {
                    continue;
                }
                queue.push_back((
                    VisibleSection {
                        pos: next,
                        entered_through: Some(step),
                        depth: info.depth + 1,
                    },
                    path.with(step),
                    next_index,
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Camera;

    fn camera_at(pos: DVec3, yaw: f32) -> Camera {
        let mut c = Camera {
            pos,
            yaw,
            aspect: 1.0,
            far: 1024.0,
            ..Camera::default()
        };
        c.update();
        c
    }

    /// Every section in a 16-high column range exists and is fully see-through.
    fn open_world(pos: ChunkSectionPos) -> Option<SetVisibility> {
        (0..16).contains(&pos.y).then(|| SetVisibility::all(true))
    }

    /// A fully-open section graph with an explicit `RenderGlobal.getVisibleFacings` answer for
    /// the camera section. This lets the regressions isolate the Java seed shortcut from the
    /// normal compiled-section matrix used after the first step.
    struct CameraFaces {
        faces: Vec<Facing>,
    }

    impl SectionSource for CameraFaces {
        fn visibility(&self, pos: ChunkSectionPos) -> Option<SetVisibility> {
            open_world(pos)
        }

        fn camera_visible_facings(
            &self,
            _pos: ChunkSectionPos,
            _local_x: i32,
            _local_y: i32,
            _local_z: i32,
        ) -> Option<Vec<Facing>> {
            Some(self.faces.clone())
        }
    }

    /// `reached` must answer for the walk's output exactly. The frustum rejects sections the
    /// walk has already marked as seen, so a section behind the camera is the case that separates
    /// the two sets.
    #[test]
    fn reached_is_exactly_the_walk_output_not_the_seen_set() {
        let cam = camera_at(DVec3::new(8.0, 72.0, 8.0), 0.0);
        let frustum = Frustum::from_camera(&cam);
        let mut walk = SectionWalk::new();
        let start = BlockPos::from_vec3(cam.pos).section_pos();
        assert_eq!(walk.reached(start), None, "no answer before the first walk");

        let output: HashSet<ChunkSectionPos> = walk
            .run(&open_world, &frustum, cam.pos, 6, true, 1)
            .iter()
            .map(|section| section.pos)
            .collect();
        assert!(output.contains(&start));

        let mut rejected = 0;
        for x in -7..=7 {
            for z in -7..=7 {
                for y in -1..=16 {
                    let pos = ChunkSectionPos::new(start.x + x, y, start.z + z);
                    assert_eq!(walk.reached(pos), Some(output.contains(&pos)), "{pos:?}");
                    let seen = walk.visited.contains(pos);
                    if seen && !output.contains(&pos) {
                        rejected += 1;
                    }
                }
            }
        }
        assert!(
            rejected > 0,
            "the frustum must have rejected a section the walk had seen"
        );

        // A cached run keeps the answer, and invalidating withdraws it.
        walk.run(&open_world, &frustum, cam.pos, 6, true, 1);
        assert_eq!(walk.cache_counts(), (1, 1));
        assert_eq!(walk.reached(start), Some(true));
        walk.invalidate();
        assert_eq!(walk.reached(start), None);
    }

    #[test]
    fn the_camera_section_is_always_included() {
        let cam = camera_at(DVec3::new(8.0, 72.0, 8.0), 0.0);
        let f = Frustum::from_camera(&cam);
        let v = visible_sections(&open_world, &f, cam.pos, 2, true);
        assert_eq!(v[0].pos, ChunkSectionPos::new(0, 4, 0));
        assert_eq!(v[0].depth, 0);
        assert_eq!(v[0].entered_through, None);
    }

    #[test]
    fn one_camera_face_opposite_the_view_submits_only_the_camera_section() {
        let cam = camera_at(DVec3::new(8.0, 72.0, 8.0), 0.0);
        let f = Frustum::from_camera(&cam);
        // `Facing::from_vector(+Z)` is SOUTH, so Java removes NORTH from this singleton set.
        let source = CameraFaces {
            faces: vec![Facing::North],
        };
        let visible =
            visible_sections_with_view(&source, &f, cam.pos, DVec3::new(0.0, 0.0, 1.0), 4, true);
        assert_eq!(
            visible,
            vec![VisibleSection {
                pos: ChunkSectionPos::new(0, 4, 0),
                entered_through: None,
                depth: 0,
            }],
            "the one reachable face is behind the camera, so RenderGlobal skips its BFS"
        );

        // The shortcut is independent of `renderChunksMany`; it executes before `flag1` is
        // consulted in the neighbour loop. Removing that early condition would incorrectly
        // visit a frustum-shaped world in this control.
        let loose =
            visible_sections_with_view(&source, &f, cam.pos, DVec3::new(0.0, 0.0, 1.0), 4, false);
        assert_eq!(loose, visible);
    }

    #[test]
    fn camera_seed_does_not_turn_a_single_face_into_a_general_bfs_filter() {
        let cam = camera_at(DVec3::new(8.0, 72.0, 8.0), 0.0);
        let f = Frustum::from_camera(&cam);
        // SOUTH is the view-facing direction and is retained; Java then performs its ordinary
        // six-direction BFS, not a SOUTH-only traversal.
        let one_front_face = CameraFaces {
            faces: vec![Facing::South],
        };
        let ordinary = visible_sections_with_view(
            &one_front_face,
            &f,
            cam.pos,
            DVec3::new(0.0, 0.0, 1.0),
            4,
            true,
        );
        assert!(
            ordinary.len() > 1,
            "a retained singleton must not stop the walk"
        );
        assert!(
            ordinary.iter().any(|section| section.pos.x != 0),
            "the first hop is not restricted to the singleton face"
        );

        // An already-empty camera-cell flood-fill has the same one-section result, but must not
        // need a view vector at all.
        let sealed = CameraFaces { faces: vec![] };
        let only_camera =
            visible_sections_with_view(&sealed, &f, cam.pos, DVec3::new(0.0, 0.0, 1.0), 4, true);
        assert_eq!(only_camera.len(), 1);
    }

    /// Every section blocks the search, and the camera's cell is an opaque cube.
    struct OpaqueCameraCell;

    impl SectionSource for OpaqueCameraCell {
        fn visibility(&self, _pos: ChunkSectionPos) -> Option<SetVisibility> {
            Some(SetVisibility::default())
        }

        fn camera_visible_facings(
            &self,
            _pos: ChunkSectionPos,
            _local_x: i32,
            _local_y: i32,
            _local_z: i32,
        ) -> Option<Vec<Facing>> {
            Some(Vec::new())
        }

        fn camera_cell_opaque(
            &self,
            _pos: ChunkSectionPos,
            _local_x: i32,
            _local_y: i32,
            _local_z: i32,
        ) -> bool {
            true
        }
    }

    #[test]
    fn a_spectator_inside_an_opaque_cube_sees_every_section_in_the_frustum() {
        let cam = camera_at(DVec3::new(8.0, 72.0, 8.0), 0.0);
        let f = Frustum::from_camera(&cam);
        let view = Some(DVec3::new(0.0, 0.0, 1.0));
        let walk_with = |spectator| {
            SectionWalk::new()
                .run_with_view(&OpaqueCameraCell, &f, cam.pos, view, 4, true, spectator, 0)
                .len()
        };
        let frustum_only = SectionWalk::new()
            .run_with_view(&OpaqueCameraCell, &f, cam.pos, view, 4, false, true, 0)
            .len();
        assert_eq!(
            walk_with(false),
            1,
            "a survival player stays in the camera section"
        );
        assert_eq!(walk_with(true), frustum_only);
        assert!(frustum_only > 1);
    }

    #[test]
    fn spectator_inside_an_opaque_camera_cell_bypasses_only_the_early_return() {
        let cam = camera_at(DVec3::new(8.0, 72.0, 8.0), 0.0);
        let f = Frustum::from_camera(&cam);
        let sealed = CameraFaces { faces: vec![] };
        let mut walk = SectionWalk::new();

        let survival = walk
            .run_with_view(
                &sealed,
                &f,
                cam.pos,
                Some(DVec3::new(0.0, 0.0, 1.0)),
                4,
                true,
                false,
                0,
            )
            .to_vec();
        assert_eq!(survival.len(), 1);

        let spectator = walk
            .run_with_view(
                &sealed,
                &f,
                cam.pos,
                Some(DVec3::new(0.0, 0.0, 1.0)),
                4,
                true,
                true,
                0,
            )
            .to_vec();
        assert!(
            spectator.len() > survival.len(),
            "spectator must continue the ordinary BFS out of the opaque camera cell"
        );
        assert_eq!(
            walk.cache_counts(),
            (0, 2),
            "the spectator mode edge is part of the cached walk key"
        );
    }

    #[test]
    fn opaque_sections_stop_the_walk() {
        let cam = camera_at(DVec3::new(8.0, 72.0, 8.0), 0.0);
        let f = Frustum::from_camera(&cam);
        // Everything is solid except the camera's own section.
        let sealed = |pos: ChunkSectionPos| {
            if !(0..16).contains(&pos.y) {
                None
            } else if pos == ChunkSectionPos::new(0, 4, 0) {
                Some(SetVisibility::all(true))
            } else {
                Some(SetVisibility::all(false))
            }
        };
        let v = visible_sections(&sealed, &f, cam.pos, 8, true);
        // The camera's section plus its six immediate neighbours: the walk reaches them (they
        // are what the camera can see), but cannot pass *through* them.
        assert!(
            v.len() <= 7,
            "walk leaked past sealed sections: {}",
            v.len()
        );
        assert!(v.iter().all(|s| s.depth <= 1));
    }

    /// java: `RenderGlobal.java:895`'s step test reads
    /// `renderchunk3.getCompiledChunk().isVisible(...)`, and an un-compiled section holds
    /// `CompiledChunk.DUMMY`, whose `isVisible` is `false`. So an unbuilt section is opaque to
    /// the walk. The camera's own section is exempt via `enumfacing2 == null`.
    ///
    /// Negative control: the same world with the ring *built* (`Some(all(true))`) must let the
    /// walk straight through, and with smart culling off the unbuilt ring must not block either
    /// — vanilla's whole test is behind `!flag1 ||`.
    #[test]
    fn an_unmeshed_section_blocks_the_walk_the_way_compiledchunk_dummy_does() {
        let cam = camera_at(DVec3::new(8.0, 72.0, 8.0), 0.0);
        let f = Frustum::from_camera(&cam);
        let here = ChunkSectionPos::new(0, 4, 0);
        // A shell of un-meshed sections one step out from the camera, open beyond it.
        let unbuilt_ring = |pos: ChunkSectionPos| {
            if !(0..16).contains(&pos.y) {
                return None;
            }
            let d = (pos.x - here.x).abs() + (pos.y - here.y).abs() + (pos.z - here.z).abs();
            if d == 1 {
                None
            } else {
                Some(SetVisibility::all(true))
            }
        };
        let blocked = visible_sections(&unbuilt_ring, &f, cam.pos, 8, true);
        assert!(
            blocked.iter().all(|s| s.depth <= 1),
            "the walk crossed an un-meshed section: {:?}",
            blocked.iter().map(|s| s.depth).max()
        );

        // Negative control 1: build the ring and the walk runs away.
        let built = visible_sections(&open_world, &f, cam.pos, 8, true);
        assert!(
            built.iter().any(|s| s.depth > 1),
            "the control world does not exercise the walk at all"
        );
        assert!(built.len() > blocked.len());

        // Negative control 2: with smart culling off nothing is consulted, so the unbuilt ring
        // is irrelevant and both worlds agree.
        let a = visible_sections(&unbuilt_ring, &f, cam.pos, 8, false).len();
        let b = visible_sections(&open_world, &f, cam.pos, 8, false).len();
        assert_eq!(a, b);
        assert!(a > blocked.len());
    }

    #[test]
    fn smart_culling_off_falls_back_to_the_frustum_only() {
        let cam = camera_at(DVec3::new(8.0, 72.0, 8.0), 0.0);
        let f = Frustum::from_camera(&cam);
        let sealed =
            |pos: ChunkSectionPos| (0..16).contains(&pos.y).then(|| SetVisibility::all(false));
        let strict = visible_sections(&sealed, &f, cam.pos, 4, true);
        let loose = visible_sections(&sealed, &f, cam.pos, 4, false);
        assert!(
            loose.len() > strict.len(),
            "disabling occlusion culling must submit more, got {} vs {}",
            loose.len(),
            strict.len()
        );
    }

    #[test]
    fn the_walk_never_leaves_the_render_distance() {
        let cam = camera_at(DVec3::new(8.0, 72.0, 8.0), 0.0);
        let f = Frustum::from_camera(&cam);
        let v = visible_sections(&open_world, &f, cam.pos, 3, true);
        for s in &v {
            assert!(s.pos.x.abs() <= 3 && s.pos.z.abs() <= 3, "{:?}", s.pos);
            assert!((0..16).contains(&s.pos.y));
        }
    }

    #[test]
    fn the_walk_is_breadth_first_and_visits_each_section_once() {
        let cam = camera_at(DVec3::new(8.0, 72.0, 8.0), 0.0);
        let f = Frustum::from_camera(&cam);
        let v = visible_sections(&open_world, &f, cam.pos, 4, true);
        let mut seen = HashSet::new();
        for s in &v {
            assert!(seen.insert(s.pos), "{:?} visited twice", s.pos);
        }
        assert!(v.windows(2).all(|w| w[0].depth <= w[1].depth));
    }

    // ---- S23: the reusable walk, the dense visited set, and the result cache ----------------

    /// The dense bitset must answer exactly what the `HashSet` it replaced answered, for every
    /// input — including the ones outside the cube it can address, which is where a bitset
    /// silently goes wrong.
    #[test]
    fn the_dense_visited_set_answers_exactly_what_a_hash_set_would() {
        let start = ChunkSectionPos::new(-3, 7, 11);
        for rd in [0u32, 1, 2, 5, 12] {
            let mut dense = Visited::default();
            dense.reset(start, rd);
            let mut reference: HashSet<ChunkSectionPos> = HashSet::new();
            let r = rd as i32;
            // Sweep the whole cube twice (so every second insert must answer `false`), plus a
            // margin outside it in all three axes to exercise the overflow path.
            for _round in 0..2 {
                for dx in (-r - 2)..=(r + 2) {
                    for dz in (-r - 2)..=(r + 2) {
                        for y in -2..(COLUMN_SECTIONS + 2) {
                            let pos = ChunkSectionPos::new(start.x + dx, y, start.z + dz);
                            assert_eq!(
                                dense.insert(pos),
                                reference.insert(pos),
                                "rd {rd}, {pos:?}"
                            );
                        }
                    }
                }
            }
            // And `reset` really empties it, or the second frame would drop every section.
            dense.reset(start, rd);
            assert!(dense.insert(start), "reset must clear the set");
        }
    }

    /// Above `MAX_DENSE_DISTANCE` the walk falls back to a hash set; that path must answer the
    /// same way, or a player with an absurd render distance gets a different world.
    #[test]
    fn the_sparse_fallback_answers_the_same_as_the_dense_set() {
        let start = ChunkSectionPos::new(0, 4, 0);
        let mut dense = Visited::default();
        dense.reset(start, MAX_DENSE_DISTANCE);
        let mut sparse = Visited::default();
        sparse.reset(start, MAX_DENSE_DISTANCE + 1);
        assert!(dense.dense, "control: this one is the dense path");
        assert!(!sparse.dense, "and this one is not");
        for dx in -4..=4 {
            for dz in -4..=4 {
                for y in 0..COLUMN_SECTIONS {
                    let pos = ChunkSectionPos::new(dx, y, dz);
                    assert_eq!(dense.insert(pos), sparse.insert(pos));
                    assert_eq!(dense.insert(pos), sparse.insert(pos), "and on the repeat");
                }
            }
        }
    }

    /// The load-bearing equality: a cached walk must answer exactly what a fresh one does.
    #[test]
    fn the_cached_walk_answers_exactly_what_the_one_shot_walk_answers() {
        let mut walk = SectionWalk::new();
        let mut generation = 0u64;
        for (x, z, yaw, pitch, rd, smart) in [
            (8.0, 8.0, 0.0f32, 0.0f32, 2u32, true),
            (8.0, 8.0, 0.0, 0.0, 4, true),
            (8.0, 8.0, 90.0, 0.0, 4, true),
            (8.0, 8.0, 90.0, 0.0, 4, false),
            (40.5, -71.25, 137.0, -22.0, 6, true),
            (40.5, -71.25, 137.0, -22.0, 6, true), // repeated: this one must be a cache hit
            (-200.0, 350.0, 12.0, 40.0, 8, true),
        ] {
            let mut cam = Camera {
                pos: DVec3::new(x, 72.0, z),
                yaw,
                pitch,
                aspect: 1.0,
                far: 1024.0,
                ..Camera::default()
            };
            cam.update();
            let f = Frustum::from_camera(&cam);
            let expected = visible_sections(&open_world, &f, cam.pos, rd, smart);
            let got = walk.run(&open_world, &f, cam.pos, rd, smart, generation);
            assert_eq!(got, expected.as_slice(), "yaw {yaw} rd {rd}");
            generation += 0; // the loaded set does not move in this sweep
        }
        let (hits, misses) = walk.cache_counts();
        assert_eq!(hits, 1, "exactly the one repeated key should have hit");
        assert_eq!(misses, 6);
    }

    /// Invalidation, case by case. Each is a way the world can change under a cached walk, and
    /// each must force a re-walk. The control is the *unchanged* re-run at the end: without it,
    /// a cache that never hits at all would pass every assertion above it.
    #[test]
    fn every_way_the_answer_can_change_invalidates_the_cache() {
        let base = |pos: DVec3, yaw: f32| {
            let mut c = Camera {
                pos,
                yaw,
                aspect: 1.0,
                far: 1024.0,
                ..Camera::default()
            };
            c.update();
            c
        };

        // Start just inside section (0, 4, 0), one block from the +X boundary.
        let cam = base(DVec3::new(15.5, 72.0, 8.0), 0.0);
        let f = Frustum::from_camera(&cam);
        let mut walk = SectionWalk::new();
        let first = walk.run(&open_world, &f, cam.pos, 4, true, 0).to_vec();
        assert_eq!(walk.cache_counts(), (0, 1));

        // (a) the camera crosses a section boundary.
        let moved = base(DVec3::new(16.5, 72.0, 8.0), 0.0);
        let fm = Frustum::from_camera(&moved);
        let after_move = walk.run(&open_world, &fm, moved.pos, 4, true, 0).to_vec();
        assert_eq!(walk.cache_counts(), (0, 2), "a moved camera must re-walk");
        assert_eq!(
            after_move,
            visible_sections(&open_world, &fm, moved.pos, 4, true)
        );
        assert_ne!(
            first[0].pos, after_move[0].pos,
            "the walk really did start from a different section"
        );

        // (b) the camera turns without moving. Same position, different frustum.
        let turned = base(DVec3::new(16.5, 72.0, 8.0), 180.0);
        let ft = Frustum::from_camera(&turned);
        let after_turn = walk.run(&open_world, &ft, turned.pos, 4, true, 0).to_vec();
        assert_eq!(walk.cache_counts(), (0, 3), "a turned camera must re-walk");
        assert_ne!(
            after_move, after_turn,
            "turning 180 degrees must change what is visible"
        );

        // (c) a sub-section camera move — the case a "same section" cache key would get wrong.
        let nudged = base(DVec3::new(16.5001, 72.0, 8.0), 180.0);
        let fn_ = Frustum::from_camera(&nudged);
        walk.run(&open_world, &fn_, nudged.pos, 4, true, 0);
        assert_eq!(
            walk.cache_counts(),
            (0, 4),
            "a camera that moved within its own section must still re-walk"
        );

        // (d) a chunk loads / unloads / goes dirty: the caller bumps the generation.
        walk.run(&open_world, &fn_, nudged.pos, 4, true, 0);
        assert_eq!(walk.cache_counts(), (1, 4), "control: identical inputs hit");
        for generation in [1u64, 2, 3] {
            walk.run(&open_world, &fn_, nudged.pos, 4, true, generation);
        }
        assert_eq!(
            walk.cache_counts(),
            (1, 7),
            "each loaded-set change must re-walk"
        );

        // (e) render distance and the occlusion toggle.
        walk.run(&open_world, &fn_, nudged.pos, 5, true, 3);
        walk.run(&open_world, &fn_, nudged.pos, 5, false, 3);
        assert_eq!(walk.cache_counts(), (1, 9));

        // Control: with every input held still, it hits — so the eight misses above are the
        // invalidation rules firing rather than a cache that never hits.
        for _ in 0..5 {
            walk.run(&open_world, &fn_, nudged.pos, 5, false, 3);
        }
        assert_eq!(walk.cache_counts(), (6, 9));
    }

    /// A stale visibility source must not be able to hide behind the cache: when the source
    /// changes shape but the caller bumps the generation, the new answer wins.
    #[test]
    fn a_section_going_opaque_changes_the_answer_once_the_generation_moves() {
        let mut cam = Camera {
            pos: DVec3::new(8.0, 72.0, 8.0),
            aspect: 1.0,
            far: 1024.0,
            ..Camera::default()
        };
        cam.update();
        let f = Frustum::from_camera(&cam);
        let sealed = |pos: ChunkSectionPos| {
            if !(0..16).contains(&pos.y) {
                None
            } else if pos == ChunkSectionPos::new(0, 4, 0) {
                Some(SetVisibility::all(true))
            } else {
                Some(SetVisibility::all(false))
            }
        };

        let mut walk = SectionWalk::new();
        let open = walk.run(&open_world, &f, cam.pos, 6, true, 0).len();
        // Same camera, same frustum, different world — only the generation says so.
        let stale = walk.run(&sealed, &f, cam.pos, 6, true, 0).len();
        assert_eq!(
            stale, open,
            "with the generation held still the cache is (correctly) trusted"
        );
        let fresh = walk.run(&sealed, &f, cam.pos, 6, true, 1).len();
        assert!(
            fresh < open,
            "a bumped generation must re-walk the new world: {fresh} vs {open}"
        );
        assert_eq!(fresh, visible_sections(&sealed, &f, cam.pos, 6, true).len());
    }

    // ---- #7 / #34: the pair-mask walk and the camera-reach query, against the S23 walk -------

    /// The walk exactly as it stood before #7/#34 (pristine `SectionWalk::walk`), kept verbatim
    /// as the reference: full `SetVisibility` per dequeued section, `is_visible`, and the full
    /// `camera_visible_facings` list with the one-face removal.
    #[allow(clippy::too_many_arguments)]
    fn reference_walk(
        source: &impl SectionSource,
        frustum: &Frustum,
        camera: DVec3,
        view: Option<DVec3>,
        render_distance: u32,
        mut smart_culling: bool,
        spectator: bool,
    ) -> Vec<VisibleSection> {
        let mut seen: HashSet<ChunkSectionPos> = HashSet::new();
        let mut queue: VecDeque<(VisibleSection, FaceSet)> = VecDeque::new();
        let mut out = Vec::new();
        let eye = BlockPos::from_vec3(camera);
        let start = eye.section_pos();
        if (0..16).contains(&start.y) {
            seen.insert(start);
            let (local_x, local_y, local_z) = (eye.x & 15, eye.y & 15, eye.z & 15);
            let camera_section_only = source
                .camera_visible_facings(start, local_x, local_y, local_z)
                .map(|mut faces| {
                    if faces.len() == 1 {
                        if let Some(view) = view {
                            let facing =
                                Facing::from_vector(view.x as f32, view.y as f32, view.z as f32)
                                    .opposite();
                            faces.retain(|&face| face != facing);
                        }
                    }
                    faces.is_empty()
                })
                .unwrap_or(false);
            let seed = VisibleSection {
                pos: start,
                entered_through: None,
                depth: 0,
            };
            if camera_section_only && !spectator {
                return vec![seed];
            }
            if spectator && source.camera_cell_opaque(start, local_x, local_y, local_z) {
                smart_culling = false;
            }
            queue.push_back((seed, FaceSet::default()));
        } else {
            let y = if eye.y > 0 { 15 } else { 0 };
            let r = render_distance as i32;
            for dx in -r..=r {
                for dz in -r..=r {
                    let pos = ChunkSectionPos::new(start.x + dx, y, start.z + dz);
                    if frustum.contains_section(pos) && seen.insert(pos) {
                        let seed = VisibleSection {
                            pos,
                            entered_through: None,
                            depth: 0,
                        };
                        queue.push_back((seed, FaceSet::default()));
                    }
                }
            }
        }
        while let Some((info, path)) = queue.pop_front() {
            out.push(info);
            let vis = source.visibility(info.pos);
            for step in Facing::VALUES {
                if smart_culling && path.contains(step.opposite()) {
                    continue;
                }
                if smart_culling {
                    if let Some(entered) = info.entered_through {
                        match vis {
                            Some(vis) if vis.is_visible(entered.opposite(), step) => {}
                            _ => continue,
                        }
                    }
                }
                let next = ChunkSectionPos::new(
                    info.pos.x + step.front_offset_x(),
                    info.pos.y + step.front_offset_y(),
                    info.pos.z + step.front_offset_z(),
                );
                if !(0..16).contains(&next.y) {
                    continue;
                }
                if (next.x - start.x).unsigned_abs() > render_distance
                    || (next.z - start.z).unsigned_abs() > render_distance
                {
                    continue;
                }
                if !seen.insert(next) {
                    continue;
                }
                if !frustum.contains_section(next) {
                    continue;
                }
                queue.push_back((
                    VisibleSection {
                        pos: next,
                        entered_through: Some(step),
                        depth: info.depth + 1,
                    },
                    path.with(step),
                ));
            }
        }
        out
    }

    fn splitmix(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    /// A deterministic pseudo-random world: per section, unmeshed (`None`), all-open, sealed,
    /// or a random set of connected face groups — the shapes `compute_visibility` produces —
    /// plus a random camera-cell face set and opacity, so every walk branch is reachable.
    struct RandomWorld {
        seed: u64,
        /// Out of 16: how often a section is unmeshed / sealed / open; the rest are mixed.
        none: u64,
        sealed: u64,
        open: u64,
    }

    impl RandomWorld {
        fn roll(&self, pos: ChunkSectionPos, salt: u64) -> u64 {
            let key =
                (pos.x as u32 as u64) ^ ((pos.z as u32 as u64) << 32) ^ ((pos.y as u64) << 59);
            splitmix(self.seed ^ splitmix(key ^ salt))
        }
    }

    impl SectionSource for RandomWorld {
        fn visibility(&self, pos: ChunkSectionPos) -> Option<SetVisibility> {
            let r = self.roll(pos, 1);
            let class = r % 16;
            if class < self.none {
                return None;
            }
            if class < self.none + self.sealed {
                return Some(SetVisibility::all(false));
            }
            if class < self.none + self.sealed + self.open {
                return Some(SetVisibility::all(true));
            }
            let mut vis = SetVisibility::all(false);
            let mut bits = r >> 4;
            for _ in 0..3 {
                let group: Vec<Facing> = Facing::VALUES
                    .into_iter()
                    .filter(|f| (bits >> f.index()) & 1 == 1)
                    .collect();
                vis.set_many_visible(&group);
                bits >>= 6;
            }
            Some(vis)
        }

        fn camera_visible_facings(
            &self,
            pos: ChunkSectionPos,
            local_x: i32,
            local_y: i32,
            local_z: i32,
        ) -> Option<Vec<Facing>> {
            let cell = (local_x | (local_y << 4) | (local_z << 8)) as u64;
            let r = self.roll(pos, 2 ^ (cell << 8));
            if r.is_multiple_of(7) {
                return None;
            }
            // Bias towards 0 and 1 faces, the cases the one-face removal decides.
            let faces = match (r >> 3) % 4 {
                0 => vec![],
                1 => vec![Facing::VALUES[((r >> 8) % 6) as usize]],
                _ => Facing::VALUES
                    .into_iter()
                    .filter(|f| (r >> (12 + f.index())) & 1 == 1)
                    .collect(),
            };
            Some(faces)
        }

        fn camera_cell_opaque(
            &self,
            pos: ChunkSectionPos,
            local_x: i32,
            local_y: i32,
            local_z: i32,
        ) -> bool {
            let cell = (local_x | (local_y << 4) | (local_z << 8)) as u64;
            self.roll(pos, 3 ^ (cell << 8)).is_multiple_of(3)
        }
    }

    /// #7/#34 parity: the rewritten walk (pair masks, `camera_reach`, no source query when rule 2
    /// cannot read it) returns exactly the reference walk's list — same sections, same order,
    /// same depth and entry face — across random worlds, cameras inside, above and below the
    /// world, both culling modes and the spectator edge. A reused `SectionWalk` is checked too,
    /// so its buffer reuse is covered.
    #[test]
    fn the_rewritten_walk_matches_the_reference_walk_exactly() {
        let mut walk = SectionWalk::new();
        let mut generation = 0u64;
        let mut compared = 0usize;
        let mut nontrivial = 0usize;
        for case in 0..240u64 {
            let r = splitmix(case);
            let world = RandomWorld {
                seed: splitmix(r),
                none: r % 3,
                sealed: (r >> 2) % 4,
                open: 3 + (r >> 4) % 8,
            };
            let x = ((r >> 8) % 2000) as f64 / 7.0 - 140.0;
            let z = ((r >> 20) % 2000) as f64 / 7.0 - 140.0;
            // Mostly inside the world; sometimes above or below it (the slab-seed branch).
            let y = match (r >> 32) % 10 {
                0 => 300.0,
                1 => -20.0,
                _ => ((r >> 36) % 2560) as f64 / 10.0,
            };
            let yaw = ((r >> 40) % 360) as f32;
            let pitch = ((r >> 49) % 180) as f32 - 90.0;
            let mut cam = Camera {
                pos: DVec3::new(x, y, z),
                yaw,
                pitch,
                aspect: 16.0 / 9.0,
                far: 1024.0,
                ..Camera::default()
            };
            cam.update();
            let frustum = Frustum::from_camera(&cam);
            let view = cam.look_vec();
            let rd = [2u32, 4, 6, 8][(r >> 58) as usize % 4];
            for smart in [true, false] {
                for spectator in [false, true] {
                    let expected =
                        reference_walk(&world, &frustum, cam.pos, Some(view), rd, smart, spectator);
                    generation += 1;
                    let got = walk.run_with_view(
                        &world,
                        &frustum,
                        cam.pos,
                        Some(view),
                        rd,
                        smart,
                        spectator,
                        generation,
                    );
                    assert_eq!(
                        got,
                        expected.as_slice(),
                        "case {case} smart {smart} spectator {spectator}"
                    );
                    compared += 1;
                    if expected.len() > 1 {
                        nontrivial += 1;
                    }
                    // `reached` must still be exactly the output set.
                    for s in &expected {
                        assert_eq!(walk.reached(s.pos), Some(true));
                    }
                }
            }
        }
        assert_eq!(compared, 960);
        assert!(
            nontrivial > 400,
            "the sweep must mostly exercise real walks: {nontrivial}"
        );
    }

    /// Above `MAX_DENSE_DISTANCE` the walk runs on the sparse fallback, where queue entries
    /// carry no dense index; it must still match the reference walk, `reached` included.
    #[test]
    fn the_sparse_fallback_walk_matches_the_reference_walk() {
        let world = RandomWorld {
            seed: 99,
            none: 1,
            sealed: 2,
            open: 8,
        };
        let mut walk = SectionWalk::new();
        let rd = MAX_DENSE_DISTANCE + 6;
        for (i, (y, yaw, pitch)) in [
            (72.0, 0.0f32, 0.0f32),
            (72.0, 135.0, 30.0),
            (300.0, 20.0, 80.0),
        ]
        .into_iter()
        .enumerate()
        {
            let mut cam = Camera {
                pos: DVec3::new(8.5, y, 8.5),
                yaw,
                pitch,
                aspect: 16.0 / 9.0,
                far: 48.0,
                ..Camera::default()
            };
            cam.update();
            let f = Frustum::from_camera(&cam);
            let view = Some(cam.look_vec());
            let expected = reference_walk(&world, &f, cam.pos, view, rd, true, false);
            let got = walk
                .run_with_view(&world, &f, cam.pos, view, rd, true, false, i as u64)
                .to_vec();
            assert_eq!(got, expected, "camera {i}");
            assert!(!walk.visited.dense, "control: this is the sparse path");
            for s in &expected {
                assert_eq!(walk.reached(s.pos), Some(true));
            }
        }
    }

    /// #7 cost structure on a void map, counted on the real walk (counts, not time; run with
    /// `--nocapture` for the table). Everything in range is air, columns beyond `loaded` are
    /// absent from the world (the live source then answers settled air after a second probe).
    /// "hashed probes" is what the pre-#7 live source paid: one `chunks` probe per query, plus
    /// one `world_columns` probe when the section is not resident.
    #[test]
    fn void_map_walk_cost_structure() {
        use std::cell::Cell;
        struct Void {
            loaded: i32,
            queries: Cell<u64>,
            absent: Cell<u64>,
        }
        impl SectionSource for Void {
            fn visibility(&self, pos: ChunkSectionPos) -> Option<SetVisibility> {
                self.queries.set(self.queries.get() + 1);
                if pos.x.abs() > self.loaded || pos.z.abs() > self.loaded {
                    self.absent.set(self.absent.get() + 1);
                }
                (0..16).contains(&pos.y).then(|| SetVisibility::all(true))
            }
        }
        let mut walk = SectionWalk::new();
        let mut generation = 0u64;
        for rd in [8u32, 12, 16] {
            for loaded in [rd as i32, 8] {
                for (yaw, pitch) in [(0.0f32, 0.0f32), (45.0, 0.0), (45.0, -30.0), (0.0, 70.0)] {
                    let mut cam = Camera {
                        pos: DVec3::new(8.5, 81.6, 8.5),
                        yaw,
                        pitch,
                        ..Camera::default()
                    };
                    cam.far = (rd * 16) as f32 * std::f32::consts::SQRT_2;
                    cam.update();
                    let f = Frustum::from_camera(&cam);
                    let view = Some(cam.look_vec());
                    let src = Void {
                        loaded,
                        queries: Cell::new(0),
                        absent: Cell::new(0),
                    };
                    let old = reference_walk(&src, &f, cam.pos, view, rd, true, false);
                    let (old_queries, old_absent) = (src.queries.get(), src.absent.get());
                    src.queries.set(0);
                    src.absent.set(0);
                    generation += 1;
                    let new = walk
                        .run_with_view(&src, &f, cam.pos, view, rd, true, false, generation)
                        .to_vec();
                    assert_eq!(new, old);
                    let unloaded_ring = old
                        .iter()
                        .filter(|s| s.pos.x.abs() > loaded || s.pos.z.abs() > loaded)
                        .count();
                    eprintln!(
                        "walk-cost rd={rd:2} loaded={loaded:2} yaw={yaw:3} pitch={pitch:4}: \
                         sections={:5} unloaded_ring={:5} old_queries={old_queries:5} \
                         old_hashed_probes={:5} new_queries={:5} (dense-grid loads, 0 hashed)",
                        old.len(),
                        unloaded_ring,
                        old_queries + old_absent,
                        src.queries.get(),
                    );
                    assert_eq!(old_queries as usize, old.len(), "one query per section");
                    assert_eq!(src.queries.get() as usize, old.len() - 1, "not the seed");
                }
            }
        }
    }

    #[test]
    fn looking_one_way_does_not_submit_sections_behind_the_camera() {
        // Vanilla yaw 0 faces +Z, so the -Z column should be culled by the frustum.
        let cam = camera_at(DVec3::new(8.0, 72.0, 8.0), 0.0);
        let f = Frustum::from_camera(&cam);
        let v = visible_sections(&open_world, &f, cam.pos, 6, true);
        assert!(
            v.iter().all(|s| s.pos.z >= -1),
            "a section well behind the camera was submitted"
        );
        assert!(v.iter().any(|s| s.pos.z > 2), "nothing ahead was submitted");
    }
}
