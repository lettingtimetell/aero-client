//! A frame-budget probe for GAP A12-01's light propagation.
//!
//! ```text
//! cargo run -p mc-world --release --example light_perf
//! ```
//!
//! The client keeps a **ms/frame** budget, and a flood fill on the game thread is exactly
//! the kind of thing that blows one. This measures the three shapes that matter, in both
//! wall-clock (noisy) and *dirty sections produced* (exact, and the number that actually reaches
//! the mesher):
//!
//! 1. a chunk-packet burst — the join path, which must stay free of propagation entirely;
//! 2. single block placements/removals — the interactive path, one per player action;
//! 3. `tick_chunk_light` — the periodic repair pass, which runs every tick forever.
//!
//! Counts and section totals are deterministic and are the real evidence. Treat every
//! microsecond figure as provisional unless the machine was quiet when it was taken.

use std::time::Instant;

use mc_data::{block_by_name, BlockState};
use mc_math::{BlockPos, ChunkPos};
use mc_world::{expected_chunk_data_len, World};

fn state(name: &str) -> BlockState {
    BlockState::new(block_by_name(name).expect("known block"), 0)
}

/// A full-column payload: terrain up to `surface`, block light 0, and **server-correct sky
/// light** — 15 above the surface, 0 in the rock. That last part matters: with an all-dark
/// sky-light region every block change would kick off a full 17-radius sky flood correcting the
/// server's "mistake", which is not a cost a real client pays and would put this probe an order
/// of magnitude out.
fn column_payload(surface: i32, stone: BlockState) -> Vec<u8> {
    let mut out = Vec::with_capacity(expected_chunk_data_len(0xFFFF, true, true));
    for s in 0..16i32 {
        for y in 0..16i32 {
            for _z in 0..16 {
                for _x in 0..16 {
                    let id = if (s << 4) + y <= surface { stone.id } else { 0 };
                    out.push((id & 0xFF) as u8);
                    out.push((id >> 8) as u8);
                }
            }
        }
    }
    out.resize(out.len() + 16 * 2048, 0); // block light
    for s in 0..16i32 {
        // A nibble array is `y << 8 | z << 4 | x`, two cells per byte; a whole Y layer is 128
        // bytes, and every cell in it has the same answer here.
        for y in 0..16i32 {
            let lit = (s << 4) + y > surface;
            out.resize(out.len() + 128, if lit { 0xFF } else { 0x00 });
        }
    }
    out.resize(out.len() + 256, 1); // biomes
    debug_assert_eq!(out.len(), expected_chunk_data_len(0xFFFF, true, true));
    out
}

fn main() {
    let stone = state("stone");
    let torch = BlockState::new(block_by_name("torch").expect("torch"), 5);
    // A view-distance-8-ish disc: 17x17 columns, which is what a real join streams in.
    const R: i32 = 8;
    let payload = column_payload(63, stone);

    // ---- 1. the join path -----------------------------------------------------------------
    let t0 = Instant::now();
    let mut w = World::new();
    for cx in -R..=R {
        for cz in -R..=R {
            w.apply_chunk_data(ChunkPos::new(cx, cz), true, 0xFFFF, &payload)
                .expect("payload");
        }
    }
    let load = t0.elapsed();
    let columns = (2 * R + 1) * (2 * R + 1);
    let dirty_on_load = w.drain_dirty_sections().len();
    println!(
        "chunk load   : {columns} columns in {load:?} ({:?}/column), {dirty_on_load} dirty sections",
        load / columns as u32
    );
    println!("               (no flood fill runs here — `fillChunk` only rebuilds the heightmap)");

    // ---- 2. the interactive path ----------------------------------------------------------
    // Torches on the open surface: the worst realistic single-block case, because both channels
    // run and sky light is live all around.
    let mut total = std::time::Duration::ZERO;
    let mut dirty_total = 0usize;
    let n = 200;
    for i in 0..n {
        let p = BlockPos::new((i % 40) - 20, 64, (i / 40) - 2);
        let t = Instant::now();
        w.set_block(p, torch);
        total += t.elapsed();
        dirty_total += w.drain_dirty_sections().len();
    }
    println!(
        "place torch  : {n} placements, {:?} total, {:?} each, {:.1} dirty sections each",
        total,
        total / n as u32,
        dirty_total as f64 / f64::from(n)
    );

    // Digging into rock: the sky-light case, where `relightBlock` rewrites a whole column.
    let mut total = std::time::Duration::ZERO;
    let mut dirty_total = 0usize;
    let n = 60;
    for i in 0..n {
        let p = BlockPos::new(100 + (i % 8), 63 - (i / 8), 100);
        let t = Instant::now();
        w.set_block(p, BlockState::AIR);
        total += t.elapsed();
        dirty_total += w.drain_dirty_sections().len();
    }
    println!(
        "mine block   : {n} removals, {:?} total, {:?} each, {:.1} dirty sections each",
        total,
        total / n as u32,
        dirty_total as f64 / f64::from(n)
    );

    // ---- 3. the periodic repair pass ------------------------------------------------------
    let ticks = 200;
    let t = Instant::now();
    let mut visited = 0;
    for _ in 0..ticks {
        visited += w.tick_chunk_light(10);
    }
    let tick_time = t.elapsed();
    println!(
        "tick_chunk_light: {ticks} ticks ({visited} column visits) in {tick_time:?}, {:?}/tick",
        tick_time / ticks
    );
    println!(
        "               a tick is 50 ms; this pass is {:.4}% of one",
        (tick_time.as_secs_f64() / f64::from(ticks) / 0.050) * 100.0
    );
    void_scenarios(stone);

    println!(
        "\nAll timings PROVISIONAL — re-measure on a quiet machine. The section counts above are\n\
         deterministic and are what actually reaches the mesher."
    );
}

/// FNV-1a over every block-light and sky-light value in a box: a cheap way to show two worlds
/// ended with identical light without printing it.
fn light_digest(w: &World, lo: BlockPos, hi: BlockPos) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for x in lo.x..=hi.x {
        for y in lo.y..=hi.y {
            for z in lo.z..=hi.z {
                let p = BlockPos::new(x, y, z);
                for v in [w.sky_light_at(p), w.block_light_at(p)] {
                    h = (h ^ u64::from(v)).wrapping_mul(0x0100_0000_01b3);
                }
            }
        }
    }
    h
}

/// The void-map shapes round 1 never measured (09-relight): SkyWars/BedWars bridges over empty
/// columns, and an explosion crater on an island. Public API only, so the same file measures
/// any engine revision.
fn void_scenarios(stone: BlockState) {
    // 9x9 empty columns: every chunk is resident (the server sends void columns) but has no
    // sections, which is what `isAreaLoaded(p, 17, false)` and the sky floods actually see.
    const V: i32 = 4;
    let fresh_void = || {
        let mut w = World::new();
        for cx in -V..=V {
            for cz in -V..=V {
                w.load_empty_chunk(ChunkPos::new(cx, cz));
            }
        }
        w.drain_dirty_sections();
        w
    };

    // ---- 4. bridging over the void ------------------------------------------------------------
    let mut w = fresh_void();
    for (label, z) in [
        ("void bridge (fresh chunks)", 0),
        ("void bridge (sections exist)", 2),
    ] {
        let mut total = std::time::Duration::ZERO;
        let mut worst = std::time::Duration::ZERO;
        let mut dirty_total = 0usize;
        let n = 96;
        for i in 0..n {
            let p = BlockPos::new(-48 + i, 64, z);
            let t = Instant::now();
            w.set_block(p, stone);
            let e = t.elapsed();
            total += e;
            worst = worst.max(e);
            dirty_total += w.drain_dirty_sections().len();
        }
        println!(
            "{label:<29}: {n} blocks, {total:?} total, {:?} each, worst {worst:?}, {:.1} dirty sections each",
            total / n as u32,
            dirty_total as f64 / f64::from(n)
        );
    }
    let bridge_digest = light_digest(&w, BlockPos::new(-50, 0, -3), BlockPos::new(50, 80, 5));
    println!("void bridge light digest  : {bridge_digest:016x}");

    // Break the second bridge back out, block by block (the "column got shorter" path).
    let mut total = std::time::Duration::ZERO;
    let n = 96;
    for i in 0..n {
        let t = Instant::now();
        w.set_block(BlockPos::new(-48 + i, 64, 2), BlockState::AIR);
        total += t.elapsed();
        w.drain_dirty_sections();
    }
    println!(
        "void bridge break-out     : {n} blocks, {total:?} total, {:?} each",
        total / n as u32
    );

    // ---- 5. an emitter at an island edge -------------------------------------------------------
    let glowstone = BlockState::new(block_by_name("glowstone").expect("glowstone"), 0);
    let mut total = std::time::Duration::ZERO;
    let n = 20;
    for i in 0..n {
        let p = BlockPos::new(-40 + 4 * i, 65, 0);
        let t = Instant::now();
        w.set_block(p, glowstone);
        w.set_block(p, BlockState::AIR);
        total += t.elapsed();
        w.drain_dirty_sections();
    }
    println!(
        "glowstone place+remove    : {n} pairs over the bridge, {total:?} total, {:?} per pair",
        total / n as u32
    );

    // ---- 6. a TNT-sized crater on a void island -----------------------------------------------
    let island = || {
        let mut w = fresh_void();
        let mut blocks = Vec::new();
        for x in -12..=12 {
            for z in -12..=12 {
                for y in 56..=63 {
                    blocks.push((BlockPos::new(x, y, z), stone));
                }
            }
        }
        w.apply_block_changes(&blocks);
        w.drain_dirty_sections();
        w
    };
    // A radius-4 sphere centred on the surface, in a scrambled order like a server HashSet's.
    let mut crater: Vec<BlockPos> = Vec::new();
    for x in -4i32..=4 {
        for y in -4i32..=4 {
            for z in -4i32..=4 {
                if x * x + y * y + z * z <= 16 {
                    crater.push(BlockPos::new(x, 63 + y, z));
                }
            }
        }
    }
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    for i in (1..crater.len()).rev() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        crater.swap(i, (seed % (i as u64 + 1)) as usize);
    }
    let probe = (BlockPos::new(-22, 40, -22), BlockPos::new(22, 80, 22));

    let mut sequential = island();
    let t = Instant::now();
    for &p in &crater {
        sequential.set_block(p, BlockState::AIR);
    }
    let seq_time = t.elapsed();
    let seq_dirty = sequential.drain_dirty_sections().len();
    let seq_digest = light_digest(&sequential, probe.0, probe.1);
    println!(
        "explosion (per-block)     : {} positions, {seq_time:?}, {seq_dirty} dirty sections, light {seq_digest:016x}",
        crater.len()
    );

    let mut batched = island();
    let changes: Vec<_> = crater.iter().map(|&p| (p, BlockState::AIR)).collect();
    let t = Instant::now();
    batched.apply_block_changes(&changes);
    let batch_time = t.elapsed();
    let batch_dirty = batched.drain_dirty_sections().len();
    let batch_digest = light_digest(&batched, probe.0, probe.1);
    println!(
        "explosion (batched)       : {} positions, {batch_time:?}, {batch_dirty} dirty sections, light {batch_digest:016x} ({})",
        crater.len(),
        if batch_digest == seq_digest {
            "same light as per-block"
        } else {
            "DIFFERENT light from per-block"
        }
    );
}
