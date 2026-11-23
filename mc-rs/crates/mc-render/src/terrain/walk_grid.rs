//! A dense, camera-centred index of the one thing the terrain walk reads per section (#7).
//!
//! [`SectionWalk`](super::cull::SectionWalk) asks its source for a section's face connectivity
//! once per section it dequeues. On a void map nothing occludes, so that is every section in
//! the frustum out to the render distance — thousands per walk, and the walk re-runs on nearly
//! every PvP frame (any camera motion, any upload). The live source answered each one with a
//! `SipHash` probe into `HashMap<ChunkSectionPos, ChunkBuffers>` (entries of ~830 B), and a
//! second probe into the world-column set when the column is unloaded.
//!
//! This grid holds the same answers as one `u16` per section — the 15-pair
//! [`SetVisibility::pair_mask`] plus a presence bit — in a `(x mod S, z mod S, y)` ring over the
//! render-distance square, which is vanilla's own `ViewFrustum` layout. A lookup is a range
//! check, two masks, an owner compare and a load.
//!
//! It is a **cache** of `AuthoritativeSectionSource::visibility` (see [`authoritative`]),
//! never a second source of truth:
//!
//! * the renderer re-derives a section's cell whenever it inserts or removes that section
//!   ([`WalkGrid::set_section`]), and a column's cells whenever that column's world membership
//!   changes ([`WalkGrid::refresh_column`]);
//! * when the camera crosses a column boundary the window recentres and only the columns that
//!   entered it are filled ([`WalkGrid::recenter`]), as `ViewFrustum.updateChunkPositions` does
//!   — a strip of `2r + 1` columns, not the square;
//! * a render-distance change or a replaced world-column set discards it whole.
//!
//! Every slot records which column it holds, and a slot is kept current for as long as it
//! holds that column, inside the window or not, so a column that leaves and re-enters the
//! window without being overwritten is still exact. Debug builds additionally check every
//! answer the walk reads against the hashed maps (`GridSectionSource` in `phase.rs`).

use std::collections::HashSet;

use mc_math::{ChunkPos, ChunkSectionPos};

use super::visgraph::SetVisibility;

/// Sections per column: the world is 256 blocks tall.
const COLUMN_SECTIONS: usize = 16;

/// Above this render distance the grid stays empty and the walk keeps the hashed source. The
/// same bound as the walk's own dense visited set; at 64 the ring is 256 × 256 columns, 2 MiB.
const MAX_RADIUS: u32 = 64;

/// Cell flag: the source answers `Some` (a compiled matrix, or settled air in an absent
/// column). Clear means `None`, the DUMMY hard stop. The low 15 bits are the pair mask.
pub(crate) const CELL_PRESENT: u16 = 1 << 15;

/// One cell for an authoritative answer.
#[inline]
#[must_use]
pub(crate) fn encode(visibility: Option<SetVisibility>) -> u16 {
    visibility.map_or(0, |vis| CELL_PRESENT | vis.pair_mask())
}

/// The walk's answer from one cell: the pair mask, or `None` for a hard stop.
#[inline]
#[must_use]
pub(crate) fn decode(cell: u16) -> Option<u16> {
    (cell & CELL_PRESENT != 0).then_some(cell & !CELL_PRESENT)
}

/// The rule the grid caches: a resident compiled section wins; otherwise, when the world's
/// column membership is authoritative, a section of an absent column is settled all-air;
/// otherwise there is no compiled chunk. `AuthoritativeSectionSource::visibility` is this.
#[must_use]
pub(crate) fn authoritative(
    resident: Option<SetVisibility>,
    world_columns: Option<&HashSet<ChunkPos>>,
    pos: ChunkSectionPos,
) -> Option<SetVisibility> {
    if resident.is_some() {
        return resident;
    }
    if let Some(world_columns) = world_columns {
        if (0..16).contains(&pos.y) && !world_columns.contains(&pos.chunk_pos()) {
            return Some(SetVisibility::all(true));
        }
    }
    None
}

/// Fill one column's 16 cells from the authoritative maps. `resident` is the renderer's
/// section lookup. The world-column probe is made once per column rather than per section.
pub(crate) fn fill_column(
    col: ChunkPos,
    cells: &mut [u16],
    world_columns: Option<&HashSet<ChunkPos>>,
    resident: impl Fn(ChunkSectionPos) -> Option<SetVisibility>,
) {
    let absent_column = world_columns.is_some_and(|columns| !columns.contains(&col));
    for (y, cell) in cells.iter_mut().enumerate() {
        let pos = ChunkSectionPos::new(col.x, y as i32, col.z);
        *cell = encode(match resident(pos) {
            Some(vis) => Some(vis),
            None if absent_column => Some(SetVisibility::all(true)),
            None => None,
        });
    }
}

/// The ring. See the module docs.
#[derive(Debug, Default)]
pub(crate) struct WalkGrid {
    /// The column the window is centred on; `None` before the first fill, after
    /// [`WalkGrid::invalidate`], and while the render distance is above [`MAX_RADIUS`].
    center: Option<ChunkPos>,
    /// Window half-width in columns: the walk's render distance.
    radius: u32,
    /// `log2` of the ring side, which is the power of two at or above `2 * radius + 1`, so the
    /// modulo is a mask.
    shift: u32,
    /// The column each ring slot holds, if any.
    owners: Vec<Option<ChunkPos>>,
    /// [`COLUMN_SECTIONS`] cells per slot, `y` fastest.
    cells: Vec<u16>,
    /// Columns filled from the hashed maps so far: what recentring has cost.
    column_fills: u64,
}

impl WalkGrid {
    /// An empty grid; the first [`WalkGrid::recenter`] fills it.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn slot(&self, x: i32, z: i32) -> usize {
        let mask = (1i32 << self.shift) - 1;
        (((x & mask) as usize) << self.shift) | ((z & mask) as usize)
    }

    #[inline]
    fn covers(&self, col: ChunkPos) -> bool {
        self.center.is_some_and(|center| {
            col.x.abs_diff(center.x) <= self.radius && col.z.abs_diff(center.z) <= self.radius
        })
    }

    /// The cached cell for `pos`, or `None` when the grid does not cover it (outside the window
    /// or the world's 16 sections, or no window yet). The caller then asks the hashed maps.
    #[inline]
    #[must_use]
    pub(crate) fn get(&self, pos: ChunkSectionPos) -> Option<u16> {
        let col = pos.chunk_pos();
        if !(0..16).contains(&pos.y) || !self.covers(col) {
            return None;
        }
        let slot = self.slot(pos.x, pos.z);
        // Always true after `recenter`; checked anyway so a broken invariant falls back to the
        // hashed answer instead of serving another column's cells.
        if self.owners[slot] != Some(col) {
            return None;
        }
        Some(self.cells[slot * COLUMN_SECTIONS + pos.y as usize])
    }

    /// Centre the window on `center` with half-width `radius`, filling every column of it the
    /// ring does not already hold. `fill(col, cells)` must write the authoritative cells of all
    /// 16 sections of `col` (see [`fill_column`]). O(1) when nothing moved.
    pub(crate) fn recenter(
        &mut self,
        center: ChunkPos,
        radius: u32,
        mut fill: impl FnMut(ChunkPos, &mut [u16]),
    ) {
        if radius > MAX_RADIUS {
            *self = Self {
                column_fills: self.column_fills,
                ..Self::default()
            };
            return;
        }
        if self.center == Some(center) && self.radius == radius {
            return;
        }
        if self.radius != radius || self.owners.is_empty() {
            let side = (2 * radius + 1).next_power_of_two();
            let slots = (side * side) as usize;
            self.shift = side.trailing_zeros();
            self.radius = radius;
            self.owners.clear();
            self.owners.resize(slots, None);
            self.cells.clear();
            self.cells.resize(slots * COLUMN_SECTIONS, 0);
        }
        self.center = Some(center);
        let r = radius as i32;
        for dx in -r..=r {
            for dz in -r..=r {
                let col = ChunkPos::new(center.x + dx, center.z + dz);
                let slot = self.slot(col.x, col.z);
                if self.owners[slot] != Some(col) {
                    let base = slot * COLUMN_SECTIONS;
                    fill(col, &mut self.cells[base..base + COLUMN_SECTIONS]);
                    self.owners[slot] = Some(col);
                    self.column_fills += 1;
                }
            }
        }
    }

    /// Re-derive one section's cell after the renderer inserted or removed it. A section whose
    /// column no slot holds is ignored: it is filled fresh when its column enters the window.
    pub(crate) fn set_section(&mut self, pos: ChunkSectionPos, cell: u16) {
        if self.owners.is_empty() || !(0..16).contains(&pos.y) {
            return;
        }
        let slot = self.slot(pos.x, pos.z);
        if self.owners[slot] == Some(pos.chunk_pos()) {
            self.cells[slot * COLUMN_SECTIONS + pos.y as usize] = cell;
        }
    }

    /// Re-derive one column's cells after its world membership changed.
    pub(crate) fn refresh_column(
        &mut self,
        col: ChunkPos,
        fill: impl FnOnce(ChunkPos, &mut [u16]),
    ) {
        if self.owners.is_empty() {
            return;
        }
        let slot = self.slot(col.x, col.z);
        if self.owners[slot] == Some(col) {
            let base = slot * COLUMN_SECTIONS;
            fill(col, &mut self.cells[base..base + COLUMN_SECTIONS]);
            self.column_fills += 1;
        }
    }

    /// Forget every slot, keeping the allocation; the next [`WalkGrid::recenter`] refills the
    /// whole window. For changes that move every cell's answer at once (a replaced world-column
    /// set).
    pub(crate) fn invalidate(&mut self) {
        self.center = None;
        self.owners.iter_mut().for_each(|owner| *owner = None);
    }

    /// Columns filled from the hashed maps since creation.
    #[must_use]
    pub(crate) fn column_fills(&self) -> u64 {
        self.column_fills
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % (hi - lo + 1) as u64) as i32
        }
    }

    /// What the renderer holds: resident sections and, optionally, authoritative membership.
    #[derive(Default)]
    struct Model {
        resident: HashMap<ChunkSectionPos, SetVisibility>,
        world_columns: Option<HashSet<ChunkPos>>,
    }

    impl Model {
        fn answer(&self, pos: ChunkSectionPos) -> u16 {
            encode(authoritative(
                self.resident.get(&pos).copied(),
                self.world_columns.as_ref(),
                pos,
            ))
        }
        fn recenter(&self, grid: &mut WalkGrid, center: ChunkPos, radius: u32) {
            grid.recenter(center, radius, |col, cells| {
                fill_column(col, cells, self.world_columns.as_ref(), |pos| {
                    self.resident.get(&pos).copied()
                });
            });
        }
        fn refresh_column(&self, grid: &mut WalkGrid, col: ChunkPos) {
            grid.refresh_column(col, |col, cells| {
                fill_column(col, cells, self.world_columns.as_ref(), |pos| {
                    self.resident.get(&pos).copied()
                });
            });
        }
    }

    /// Every covered cell equals the authoritative answer, and the grid covers exactly the
    /// window (so the walk, which never leaves it, never falls back).
    fn assert_exact(grid: &WalkGrid, model: &Model, center: ChunkPos, radius: u32, step: usize) {
        let r = radius as i32;
        for dx in -r - 2..=r + 2 {
            for dz in -r - 2..=r + 2 {
                for y in -1..=16 {
                    let pos = ChunkSectionPos::new(center.x + dx, y, center.z + dz);
                    let inside = dx.abs() <= r && dz.abs() <= r && (0..16).contains(&y);
                    let got = grid.get(pos);
                    if inside {
                        assert_eq!(got, Some(model.answer(pos)), "step {step}: {pos:?}");
                    } else {
                        assert_eq!(got, None, "step {step}: {pos:?} is outside the window");
                    }
                }
            }
        }
    }

    #[test]
    fn encode_and_decode_round_trip_the_walks_answer() {
        assert_eq!(decode(encode(None)), None);
        assert_eq!(decode(encode(Some(SetVisibility::all(false)))), Some(0));
        assert_eq!(decode(encode(Some(SetVisibility::all(true)))), Some(0x7fff));
        let mut vis = SetVisibility::all(false);
        vis.set_visible(mc_math::Facing::North, mc_math::Facing::Up);
        assert_eq!(decode(encode(Some(vis))), Some(vis.pair_mask()));
    }

    #[test]
    fn fill_column_is_the_authoritative_rule_cell_by_cell() {
        let mut model = Model::default();
        let col = ChunkPos::new(-3, 7);
        model
            .resident
            .insert(ChunkSectionPos::new(-3, 5, 7), SetVisibility::all(false));
        let mut cells = [0xffffu16; 16];
        for world_columns in [None, Some(HashSet::new()), Some(HashSet::from([col]))] {
            model.world_columns = world_columns;
            fill_column(col, &mut cells, model.world_columns.as_ref(), |pos| {
                model.resident.get(&pos).copied()
            });
            for (y, &cell) in cells.iter().enumerate() {
                assert_eq!(cell, model.answer(ChunkSectionPos::new(-3, y as i32, 7)));
            }
        }
    }

    /// The maintenance contract under a long random history of everything the renderer does:
    /// uploads and removals (in and out of the window), world-column loads and unloads, a
    /// replaced world-column set, camera moves by one column, short hops, teleports, moves
    /// back into previously-held columns, and render-distance changes. After every step the
    /// grid must equal the hashed answer for the whole window.
    #[test]
    fn the_grid_stays_exact_under_random_maintenance() {
        let mut rng = XorShift(0x5851_f42d_4c95_7f2d);
        let mut model = Model::default();
        let mut grid = WalkGrid::new();
        let mut center = ChunkPos::new(0, 0);
        let mut radius = 4u32;
        model.recenter(&mut grid, center, radius);
        assert_exact(&grid, &model, center, radius, 0);

        for step in 1..=3_000usize {
            match rng.next() % 12 {
                // Upload (insert or replace) a section near the camera or far from it.
                0..=3 => {
                    let span = if rng.next().is_multiple_of(4) {
                        40
                    } else {
                        radius as i32 + 2
                    };
                    let pos = ChunkSectionPos::new(
                        center.x + rng.range(-span, span),
                        rng.range(0, 15),
                        center.z + rng.range(-span, span),
                    );
                    let vis = match rng.next() % 3 {
                        0 => SetVisibility::all(true),
                        1 => SetVisibility::all(false),
                        _ => {
                            let mut v = SetVisibility::all(false);
                            v.set_visible(
                                mc_math::Facing::VALUES[(rng.next() % 6) as usize],
                                mc_math::Facing::VALUES[(rng.next() % 6) as usize],
                            );
                            v
                        }
                    };
                    model.resident.insert(pos, vis);
                    grid.set_section(pos, model.answer(pos));
                }
                // Remove a resident section.
                4 | 5 => {
                    let n = (rng.next() % (model.resident.len().max(1) as u64)) as usize;
                    let victim = model.resident.keys().nth(n).copied();
                    if let Some(pos) = victim {
                        model.resident.remove(&pos);
                        grid.set_section(pos, model.answer(pos));
                    }
                }
                // A column loads or unloads in the world.
                6 => {
                    let col = ChunkPos::new(
                        center.x + rng.range(-(radius as i32) - 3, radius as i32 + 3),
                        center.z + rng.range(-(radius as i32) - 3, radius as i32 + 3),
                    );
                    if let Some(columns) = model.world_columns.as_mut() {
                        let changed = if rng.next().is_multiple_of(2) {
                            columns.insert(col)
                        } else {
                            columns.remove(&col)
                        };
                        if changed {
                            model.refresh_column(&mut grid, col);
                        }
                    }
                }
                // The world-column set is replaced (join, respawn, dimension change).
                7 => {
                    let mut next = HashSet::new();
                    for _ in 0..rng.next() % 60 {
                        next.insert(ChunkPos::new(
                            center.x + rng.range(-8, 8),
                            center.z + rng.range(-8, 8),
                        ));
                    }
                    if model.world_columns.as_ref() != Some(&next) {
                        model.world_columns = Some(next);
                        grid.invalidate();
                    }
                }
                // The camera crosses into a neighbouring column.
                8 | 9 => {
                    center =
                        ChunkPos::new(center.x + rng.range(-1, 1), center.z + rng.range(-1, 1));
                }
                // A hop (ender pearl) or a teleport across the map.
                10 => {
                    let span = if rng.next().is_multiple_of(3) { 500 } else { 6 };
                    center = ChunkPos::new(
                        center.x + rng.range(-span, span),
                        center.z + rng.range(-span, span),
                    );
                }
                // The render distance changes.
                _ => radius = [0u32, 1, 2, 4, 6, 8][(rng.next() % 6) as usize],
            }
            model.recenter(&mut grid, center, radius);
            assert_exact(&grid, &model, center, radius, step);
        }
    }

    /// The point of a ring rather than a re-centred array: crossing one column boundary fills
    /// one strip of `2r + 1` columns, not the whole `(2r + 1)²` square.
    #[test]
    fn crossing_a_column_boundary_fills_one_strip() {
        let model = Model::default();
        let mut grid = WalkGrid::new();
        let radius = 12u32;
        model.recenter(&mut grid, ChunkPos::new(0, 0), radius);
        assert_eq!(grid.column_fills(), 25 * 25);
        model.recenter(&mut grid, ChunkPos::new(0, 0), radius);
        assert_eq!(grid.column_fills(), 25 * 25, "no move, no fill");
        model.recenter(&mut grid, ChunkPos::new(1, 0), radius);
        assert_eq!(grid.column_fills(), 25 * 25 + 25, "one strip");
        model.recenter(&mut grid, ChunkPos::new(1, 1), radius);
        assert_eq!(grid.column_fills(), 25 * 25 + 50);
        // Back where it started: the columns that left were not overwritten (the ring side is
        // 32 > 25), so they are still held and still exact.
        model.recenter(&mut grid, ChunkPos::new(0, 0), radius);
        assert_eq!(grid.column_fills(), 25 * 25 + 50);
        assert_exact(&grid, &model, ChunkPos::new(0, 0), radius, 0);
    }

    #[test]
    fn an_absurd_render_distance_leaves_the_grid_off() {
        let model = Model::default();
        let mut grid = WalkGrid::new();
        model.recenter(&mut grid, ChunkPos::new(0, 0), MAX_RADIUS + 1);
        assert_eq!(grid.get(ChunkSectionPos::new(0, 4, 0)), None);
        grid.set_section(ChunkSectionPos::new(0, 4, 0), CELL_PRESENT);
        model.recenter(&mut grid, ChunkPos::new(0, 0), 2);
        assert_exact(&grid, &model, ChunkPos::new(0, 0), 2, 0);
    }
}
