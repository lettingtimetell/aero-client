//! Differential tests for round-2 09-relight.
//!
//! * The resolved-column light engine ([`World`]'s default) against the original
//!   hashed-lookup transcription ([`World::set_reference_light_engine`]), on void worlds with
//!   islands, bridges over the void, emitters, removals, random light noise and explosion craters.
//! * The section-aware chunk descents against their cell-by-cell originals, on random chunks.
//!
//! Every comparison is exact: section presence, every block, both light nibble arrays, both
//! height maps, the height minimum, the gap flags, the relight cursor and the mesher dirty set.

use std::collections::BTreeSet;

use mc_data::{BlockId, BlockState};
use mc_math::{BlockPos, ChunkPos, ChunkSectionPos};
use proptest::prelude::*;

use crate::{Chunk, SkyBlock, World};

/// Every branch the light code distinguishes: air, opacity 255 / 3 / 1 / 0 solids, the
/// neighbour-brightness slab, transparent and opaque emitters, and a dim emitter.
fn palette(i: u8) -> BlockState {
    let (id, meta) = match i % 12 {
        0 => (0, 0),   // air
        1 => (1, 0),   // stone: opacity 255
        2 => (20, 0),  // glass: solid, opacity 0
        3 => (18, 0),  // leaves: opacity 1
        4 => (9, 0),   // still water: opacity 3, liquid
        5 => (89, 0),  // glowstone: emission 15
        6 => (50, 5),  // torch: emission 14, opacity 0
        7 => (44, 0),  // stone slab: use_neighbor_brightness
        8 => (91, 0),  // jack o'lantern: opaque emitter
        9 => (79, 0),  // ice: opacity 3
        10 => (76, 5), // redstone torch: emission 7
        _ => (3, 0),   // dirt
    };
    BlockState::new(BlockId(id), meta)
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self((seed ^ 0x9e37_79b9_7f4a_7c15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, lo: i32, hi_exclusive: i32) -> i32 {
        lo + (self.next() % (hi_exclusive - lo) as u64) as i32
    }
}

/// `(2r+1)²` resident, section-less columns around chunk (0, 0): a void map.
fn void_world(r: i32) -> World {
    let mut w = World::new();
    for cx in -r..=r {
        for cz in -r..=r {
            w.load_empty_chunk(ChunkPos::new(cx, cz));
        }
    }
    w.drain_dirty_sections();
    w
}

/// Two independent copies of `base` sharing nothing writable: one per engine.
fn fork(base: &World) -> (World, World) {
    let mut reference = base.render_snapshot();
    reference.set_reference_light_engine(true);
    let fast = base.render_snapshot();
    (reference, fast)
}

fn first_difference(a: &World, b: &World) -> String {
    for (pos, ca) in &a.chunks {
        let Some(cb) = b.chunks.get(pos) else {
            return format!("column {pos:?} resident in only one world");
        };
        if ca.same_storage(cb) {
            continue;
        }
        for si in 0..16u8 {
            match (ca.section(si), cb.section(si)) {
                (None, None) => {}
                (Some(_), None) | (None, Some(_)) => {
                    return format!("column {pos:?} section {si}: present in only one world");
                }
                (Some(sa), Some(sb)) => {
                    for y in 0..16u8 {
                        for z in 0..16u8 {
                            for x in 0..16u8 {
                                let cell = (
                                    sa.block(x, y, z),
                                    sa.sky_light(x, y, z),
                                    sa.block_light(x, y, z),
                                );
                                let other = (
                                    sb.block(x, y, z),
                                    sb.sky_light(x, y, z),
                                    sb.block_light(x, y, z),
                                );
                                if cell != other {
                                    let p = BlockPos::new(
                                        pos.x * 16 + i32::from(x),
                                        i32::from(si) * 16 + i32::from(y),
                                        pos.z * 16 + i32::from(z),
                                    );
                                    return format!(
                                        "{p:?}: (block, sky, block light) reference {cell:?} vs fast {other:?}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        return format!("column {pos:?}: height maps / gap flags / relight cursor differ");
    }
    "no storage difference".to_owned()
}

/// Exact equality of every resident column and of the dirty set (drained from both).
fn assert_same(reference: &mut World, fast: &mut World, context: &str) {
    assert_eq!(
        reference.chunks.len(),
        fast.chunks.len(),
        "{context}: residency"
    );
    let same = reference.chunks.iter().all(|(pos, column)| {
        fast.chunks
            .get(pos)
            .is_some_and(|other| column.same_storage(other))
    });
    assert!(same, "{context}: {}", first_difference(reference, fast));
    let dirty_reference: BTreeSet<ChunkSectionPos> = reference.drain_dirty_sections();
    let dirty_fast: BTreeSet<ChunkSectionPos> = fast.drain_dirty_sections();
    assert_eq!(dirty_reference, dirty_fast, "{context}: dirty sets differ");
}

#[derive(Debug, Clone)]
enum Edit {
    /// One `set_block`.
    Place(BlockPos, u8),
    /// A line of placements over the void, one block at a time (speed-bridging).
    Bridge {
        start: BlockPos,
        dir: u8,
        len: u8,
        block: u8,
    },
    /// Every position within `radius` of `centre` set to air one at a time, in a scrambled
    /// order — `S27PacketExplosion` through `set_block`, as `dispatch.rs` applies it.
    Explosion {
        centre: BlockPos,
        radius: u8,
        seed: u64,
    },
    /// The same crater through `apply_block_changes` (the S22 batch path), both engines.
    BatchExplosion {
        centre: BlockPos,
        radius: u8,
        seed: u64,
    },
    /// Arbitrary stored light, so later checks run the darkening pass against stale values.
    Noise { seed: u64, cells: u8 },
    /// Direct `check_light` / `recheck_gaps` / relight-sweep calls.
    Repair { at: BlockPos },
}

fn crater(centre: BlockPos, radius: u8, seed: u64) -> Vec<BlockPos> {
    let r = i32::from(radius);
    let mut out = Vec::new();
    for x in -r..=r {
        for y in -r..=r {
            for z in -r..=r {
                if x * x + y * y + z * z <= r * r + r {
                    out.push(BlockPos::new(centre.x + x, centre.y + y, centre.z + z));
                }
            }
        }
    }
    let mut rng = Rng::new(seed);
    for i in (1..out.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        out.swap(i, j);
    }
    out
}

fn apply(w: &mut World, edit: &Edit) {
    match *edit {
        Edit::Place(p, block) => w.set_block(p, palette(block)),
        Edit::Bridge {
            start,
            dir,
            len,
            block,
        } => {
            let (dx, dz) = [(1, 0), (-1, 0), (0, 1), (0, -1)][usize::from(dir % 4)];
            for i in 0..i32::from(len) {
                w.set_block(
                    BlockPos::new(start.x + dx * i, start.y, start.z + dz * i),
                    palette(block),
                );
            }
        }
        Edit::Explosion {
            centre,
            radius,
            seed,
        } => {
            for p in crater(centre, radius, seed) {
                w.set_block(p, BlockState::AIR);
            }
        }
        Edit::BatchExplosion {
            centre,
            radius,
            seed,
        } => {
            let changes: Vec<_> = crater(centre, radius, seed)
                .into_iter()
                .map(|p| (p, BlockState::AIR))
                .collect();
            w.apply_block_changes(&changes);
        }
        Edit::Noise { seed, cells } => {
            let mut rng = Rng::new(seed);
            for _ in 0..cells {
                let p = BlockPos::new(rng.range(-24, 40), rng.range(30, 100), rng.range(-24, 40));
                let kind = if rng.next() & 1 == 0 {
                    SkyBlock::Sky
                } else {
                    SkyBlock::Block
                };
                w.set_light_for(kind, p, (rng.next() % 16) as u8);
            }
        }
        Edit::Repair { at } => {
            w.check_light(at);
            w.recheck_gaps(at.chunk_pos(), false);
            if let Some(chunk) = w.chunks.get_mut(&at.chunk_pos()) {
                std::sync::Arc::make_mut(chunk).queued_light_checks = 0;
            }
            w.enqueue_relight_checks(at.chunk_pos());
        }
    }
}

/// A void map with a few random islands (built once, then forked to both engines).
fn island_world(seed: u64, islands: u8) -> World {
    let mut w = void_world(3);
    let mut rng = Rng::new(seed);
    for _ in 0..islands {
        let (x0, y0, z0) = (rng.range(-28, 36), rng.range(40, 80), rng.range(-28, 36));
        let (sx, sy, sz) = (rng.range(1, 10), rng.range(1, 6), rng.range(1, 10));
        let main = (rng.next() % 12) as u8;
        let mut changes = Vec::new();
        for x in x0..x0 + sx {
            for y in y0..y0 + sy {
                for z in z0..z0 + sz {
                    let block = if rng.next().is_multiple_of(5) {
                        (rng.next() % 12) as u8
                    } else {
                        main
                    };
                    changes.push((BlockPos::new(x, y, z), palette(block)));
                }
            }
        }
        w.apply_block_changes(&changes);
    }
    w.drain_dirty_sections();
    w
}

fn arb_pos() -> impl Strategy<Value = BlockPos> {
    (-30i32..46, 35i32..95, -30i32..46).prop_map(|(x, y, z)| BlockPos::new(x, y, z))
}

fn arb_edit() -> impl Strategy<Value = Edit> {
    prop_oneof![
        3 => (arb_pos(), any::<u8>()).prop_map(|(p, b)| Edit::Place(p, b)),
        4 => (arb_pos(), any::<u8>(), 1u8..28, prop_oneof![Just(1u8), any::<u8>()]).prop_map(
            |(start, dir, len, block)| Edit::Bridge { start, dir, len, block }
        ),
        2 => (arb_pos(), 1u8..4, any::<u64>())
            .prop_map(|(centre, radius, seed)| Edit::Explosion { centre, radius, seed }),
        1 => (arb_pos(), 1u8..4, any::<u64>())
            .prop_map(|(centre, radius, seed)| Edit::BatchExplosion { centre, radius, seed }),
        1 => (any::<u64>(), 1u8..40).prop_map(|(seed, cells)| Edit::Noise { seed, cells }),
        1 => arb_pos().prop_map(|at| Edit::Repair { at }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 24,
        // Each case replays dozens of light floods twice; bound shrinking so a failure reports
        // promptly instead of timing out the job.
        max_shrink_iters: 24,
        ..ProptestConfig::default()
    })]

    /// The resolved-column engine and the reference engine end every edit with identical
    /// storage, light and dirty sets.
    #[test]
    fn resolved_engine_matches_reference_on_random_void_edits(
        seed in any::<u64>(),
        islands in 0u8..5,
        edits in prop::collection::vec(arb_edit(), 1..7),
    ) {
        let base = island_world(seed, islands);
        let (mut reference, mut fast) = fork(&base);
        for (i, edit) in edits.iter().enumerate() {
            apply(&mut reference, edit);
            apply(&mut fast, edit);
            assert_same(&mut reference, &mut fast, &format!("edit {i}: {edit:?}"));
        }
    }

    /// The section-aware `generate_height_map` / `generate_skylight_map` / `relight_begin` equal
    /// the cell-by-cell originals on arbitrary storage, including stale height maps (vanilla never
    /// clears a column it finds nothing opaque in) and stale sky nibbles.
    #[test]
    fn section_aware_descents_match_the_cell_by_cell_originals(
        seed in any::<u64>(),
        presence in any::<u16>(),
        density in 0u32..64,
        has_sky in any::<bool>(),
    ) {
        let mut rng = Rng::new(seed);
        let mut chunk = Chunk::empty();
        for si in 0..16u8 {
            if presence & (1 << si) == 0 {
                continue;
            }
            chunk.section_or_insert(si);
            for _ in 0..density * 8 {
                let (x, y, z) = (rng.range(0, 16) as u8, rng.range(0, 16) as u8, rng.range(0, 16) as u8);
                let section = chunk.section_or_insert(si);
                section.set_block(x, y, z, palette((rng.next() % 12) as u8));
                section.set_sky_light(x, y, z, (rng.next() % 16) as u8);
            }
        }
        for idx in 0..256 {
            if rng.next().is_multiple_of(3) {
                chunk.height_map[idx] = rng.range(0, 257);
            }
        }
        let mut reference = chunk.clone();
        let mut fast = chunk.clone();
        let written_reference = reference.generate_skylight_map_reference(has_sky);
        let written_fast = fast.generate_skylight_map(has_sky);
        prop_assert_eq!(written_reference, written_fast);
        prop_assert!(reference.same_storage(&fast), "generate_skylight_map storage differs");
        // Same copy-on-write decisions: a section shared with `chunk` (as with a render
        // snapshot) is detached by one engine exactly when it is by the other.
        for si in 0..16 {
            let detached = |c: &Chunk| match (&chunk.sections()[si], &c.sections()[si]) {
                (Some(a), Some(b)) => !std::sync::Arc::ptr_eq(a, b),
                _ => false,
            };
            prop_assert_eq!(detached(&reference), detached(&fast), "copy-on-write, section {}", si);
        }

        let mut reference = chunk.clone();
        let mut fast = chunk;
        reference.generate_height_map_reference();
        fast.generate_height_map();
        prop_assert!(reference.same_storage(&fast), "generate_height_map storage differs");

        for x in 0..16u8 {
            for z in 0..16u8 {
                for y in [0, 1, 15, 16, 17, 63, 64, 65, 128, 255, 256, rng.range(0, 257)] {
                    prop_assert_eq!(
                        fast.relight_begin(x, y, z),
                        fast.relight_begin_reference(x, y, z),
                        "relight_begin({}, {}, {})", x, y, z
                    );
                }
            }
        }
    }
}

/// The sky descent through sections that hold
/// only air (as light fills create them below a void bridge) equals the cell-by-cell original,
/// whether a column enters such a section with full sky (15) or already dimmed by a block above
/// (each air cell then costs 1), with stale sky nibbles left in the air sections, stale height
/// maps, sections shared with a snapshot (copy-on-write) and section 0's never-written `y = 0`.
#[test]
fn air_sections_descend_like_the_cell_by_cell_original() {
    let (mut dimmed_air_runs, mut full_air_runs) = (0, 0);
    for seed in 0..300u64 {
        let mut rng = Rng::new(seed ^ 0x0c0f_fee0);
        let mut chunk = Chunk::empty();
        let presence = (rng.next() as u16) | (1 << (rng.range(0, 16) as u16));
        let mut layout = [0u8; 16]; // 0 absent, 1 all air, 2 blocks
        for si in 0..16u8 {
            if presence & (1 << si) == 0 {
                continue;
            }
            let dense = rng.next().is_multiple_of(2);
            layout[si as usize] = if dense { 2 } else { 1 };
            let section = chunk.section_or_insert(si);
            for _ in 0..rng.range(0, 300) {
                let (x, y, z) = (
                    rng.range(0, 16) as u8,
                    rng.range(0, 16) as u8,
                    rng.range(0, 16) as u8,
                );
                section.set_sky_light(x, y, z, (rng.next() % 16) as u8);
            }
            if dense {
                for _ in 0..rng.range(1, 400) {
                    let (x, y, z) = (
                        rng.range(0, 16) as u8,
                        rng.range(0, 16) as u8,
                        rng.range(0, 16) as u8,
                    );
                    section.set_block(x, y, z, palette((rng.next() % 12) as u8));
                }
            }
        }
        for idx in 0..256 {
            if rng.next().is_multiple_of(4) {
                chunk.height_map[idx] = rng.range(0, 257);
            }
        }
        // Count what the fixture exercises: an air section below one with blocks (columns enter
        // it dimmed or blocked) and one with nothing above it but air (full sky).
        let top = (0..16).rev().find(|&si| layout[si] != 0).unwrap_or(0);
        for si in 0..=top {
            if layout[si] == 1 {
                if layout[si + 1..=top].contains(&2) {
                    dimmed_air_runs += 1;
                } else {
                    full_air_runs += 1;
                }
            }
        }
        for has_sky in [true, false] {
            let mut reference = chunk.clone();
            let mut fast = chunk.clone();
            let written_reference = reference.generate_skylight_map_reference(has_sky);
            let written_fast = fast.generate_skylight_map(has_sky);
            assert_eq!(written_reference, written_fast, "seed {seed}");
            assert!(
                reference.same_storage(&fast),
                "seed {seed}: storage differs"
            );
            for si in 0..16 {
                let detached = |c: &Chunk| match (&chunk.sections()[si], &c.sections()[si]) {
                    (Some(a), Some(b)) => !std::sync::Arc::ptr_eq(a, b),
                    _ => false,
                };
                assert_eq!(
                    detached(&reference),
                    detached(&fast),
                    "seed {seed}: section {si}"
                );
            }
        }
    }
    assert!(
        dimmed_air_runs > 100 && full_air_runs > 100,
        "{dimmed_air_runs} {full_air_runs}"
    );
}

/// The void-bridge fixture round 1 asked for: two bridges over section-less void columns, the
/// first entering fresh chunks (light-created sections, whole-chunk skylight regeneration), the
/// second alongside it once those sections exist, then both broken back out — compared after
/// every single block.
#[test]
fn void_bridges_relight_identically_block_by_block() {
    let base = void_world(3);
    let (mut reference, mut fast) = fork(&base);
    let stone = palette(1);
    for (z, state) in [
        (0, stone),
        (2, stone),
        (1, palette(3)),
        (0, BlockState::AIR),
    ] {
        for x in -30..30 {
            let p = BlockPos::new(x, 64, z);
            reference.set_block(p, state);
            fast.set_block(p, state);
            assert_same(
                &mut reference,
                &mut fast,
                &format!("bridge block {p:?} = {state:?}"),
            );
        }
    }
    // The fixture must exercise the light-created-section path it exists for.
    assert!(fast
        .chunk(ChunkPos::new(0, 0))
        .unwrap()
        .section(0)
        .is_some());
    assert!(fast.light_for(SkyBlock::Sky, BlockPos::new(5, 30, 2)) > 0);
}

/// An explosion crater on an island, applied per block (the S27 path), compared after every
/// block; plus an emitter in the crater.
#[test]
fn island_crater_relights_identically_block_by_block() {
    let mut base = void_world(3);
    let mut changes = Vec::new();
    for x in -10..=10 {
        for z in -10..=10 {
            for y in 56..=63 {
                changes.push((
                    BlockPos::new(x, y, z),
                    palette(if y == 63 { 11 } else { 1 }),
                ));
            }
        }
    }
    base.apply_block_changes(&changes);
    base.drain_dirty_sections();
    let (mut reference, mut fast) = fork(&base);
    for (i, p) in crater(BlockPos::new(2, 62, -1), 4, 7)
        .into_iter()
        .enumerate()
    {
        reference.set_block(p, BlockState::AIR);
        fast.set_block(p, BlockState::AIR);
        assert_same(
            &mut reference,
            &mut fast,
            &format!("crater block {i} {p:?}"),
        );
    }
    for (p, state) in [
        (BlockPos::new(2, 60, -1), palette(5)),
        (BlockPos::new(3, 59, 0), palette(6)),
        (BlockPos::new(2, 60, -1), BlockState::AIR),
    ] {
        reference.set_block(p, state);
        fast.set_block(p, state);
        assert_same(
            &mut reference,
            &mut fast,
            &format!("emitter {p:?} = {state:?}"),
        );
    }
}

/// Probes whether `apply_block_changes` (the S22 batch path, and round 1's proposed explosion
/// path) leaves the same light as applying the same writes one `set_block` at a time — the
/// order vanilla's `Explosion.doExplosionB` and `handleMultiBlockChange` use. It reports rather
/// than asserts: the answer decides whether explosions may be batched (see findings 09-relight).
#[test]
fn report_whether_batched_craters_match_sequential_light() {
    let mut mismatches = 0;
    let mut first = None;
    let cases = 24u64;
    for seed in 0..cases {
        let mut base = island_world(seed, 4);
        if seed % 2 == 1 {
            // Stale light, as a client holds after server-sent columns: the darkening pass and
            // origin order then matter.
            apply(&mut base, &Edit::Noise { seed, cells: 120 });
            base.drain_dirty_sections();
        }
        let (_, mut sequential) = fork(&base);
        let (_, mut batched) = fork(&base);
        let mut rng = Rng::new(seed ^ 0xabcd);
        let centre = BlockPos::new(rng.range(-20, 30), rng.range(45, 80), rng.range(-20, 30));
        let positions = crater(centre, 3, seed);
        for &p in &positions {
            sequential.set_block(p, BlockState::AIR);
        }
        let changes: Vec<_> = positions.iter().map(|&p| (p, BlockState::AIR)).collect();
        batched.apply_block_changes(&changes);
        let same = sequential
            .chunks
            .iter()
            .all(|(pos, c)| batched.chunks.get(pos).is_some_and(|o| c.same_storage(o)));
        if !same {
            mismatches += 1;
            first.get_or_insert_with(|| {
                format!(
                    "seed {seed} centre {centre:?}: {}",
                    first_difference(&sequential, &batched)
                )
            });
        }
    }
    println!(
        "batched vs sequential craters: {mismatches}/{cases} differ; first: {}",
        first.unwrap_or_else(|| "none".to_owned())
    );
}
