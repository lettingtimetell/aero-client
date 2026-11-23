//! S23's frame-loop profiler: where the time in a terrain frame actually goes.
//!
//! "Profile the frame loop; fix the top 5 hotspots". This is a `harness = false`
//! bench, i.e. a plain binary — there is no `criterion` in this workspace's offline cargo cache,
//! and hand-rolled timing with the run count and the spread printed is what the plan's honesty
//! convention asks for anyway.
//!
//! Run it with:
//!
//! ```text
//! cargo bench -p mc-render --bench frame_loop
//! cargo bench -p mc-render --bench frame_loop -- --radius 6 --frames 150
//! cargo bench -p mc-render --bench frame_loop -- --radius 12 --frames 150
//! ```
//!
//! # What it measures, and what each number is worth
//!
//! * **`phases.render` CPU time** — recording the frame: the culling walk, the per-chunk offset
//!   upload, and one `draw_indexed` per visible section per layer. This is the number a CPU-side
//!   optimisation moves.
//! * **GPU completion time** — `device.poll` after the submission, so the two halves are
//!   separated rather than averaged into one meaningless "frame time".
//! * **An empty-`PhaseList` control** — the same loop with nothing registered. Without it, a
//!   "terrain costs X ms" claim silently includes the acquire/clear/submit floor, and on a fast
//!   GPU that floor is most of it.
//! * **The mesher**, the **queue**, and the **upload path**, each on its own, because they run at
//!   different rates (per dirty section, per tick, per frame) and averaging them hides which one
//!   is the problem.
//!
//! Everything here runs against the real 1.8.9 jar and a real wgpu adapter. With neither it
//! prints `skip:` and exits 0, the same convention the S11/S18/S20 GPU tests use.
//!
//! # Honesty notes
//!
//! * These are **single-machine** numbers from one desktop GPU, not a certified benchmark. The
//!   profile is printed with every run so a reader can tell `dev` (`opt-level = 1`, deps at 3)
//!   from `release`.
//! * The first measured frame in a fresh process is confounded by one-time pipeline and shader
//!   compilation — S18 documented the same effect at ~11 ms for particles. Warm-up frames are
//!   run and discarded, and the discarded first-frame cost is printed separately rather than
//!   hidden.
//! * Mean, standard deviation, min and max are all printed. A single best-of-N number is not
//!   evidence.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mc_data::BlockState;
use mc_math::{BlockPos, ChunkSectionPos, DVec3};
use mc_render::terrain::{fixture, jobs, MeshSettings, TerrainPhase, TerrainRenderer};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, RenderError};
use mc_world::World;

const WIDTH: u32 = 854;
const HEIGHT: u32 = 480;

// ---------------------------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------------------------

/// A sample set with the spread kept, because a mean on its own cannot be checked.
struct Samples {
    ms: Vec<f64>,
}

impl Samples {
    fn new() -> Self {
        Samples { ms: Vec::new() }
    }

    fn push(&mut self, d: Duration) {
        self.ms.push(d.as_secs_f64() * 1000.0);
    }

    fn mean(&self) -> f64 {
        if self.ms.is_empty() {
            return 0.0;
        }
        self.ms.iter().sum::<f64>() / self.ms.len() as f64
    }

    fn sd(&self) -> f64 {
        if self.ms.len() < 2 {
            return 0.0;
        }
        let m = self.mean();
        let var =
            self.ms.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (self.ms.len() - 1) as f64;
        var.sqrt()
    }

    fn min(&self) -> f64 {
        self.ms.iter().copied().fold(f64::INFINITY, f64::min)
    }

    fn max(&self) -> f64 {
        self.ms.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    /// The sample set sorted ascending, which is what every percentile below indexes into.
    fn sorted(&self) -> Vec<f64> {
        let mut v = self.ms.clone();
        v.sort_by(f64::total_cmp);
        v
    }

    /// The `q`-quantile by **nearest rank**: the smallest sample `v` with at least `ceil(q * n)`
    /// samples `<= v`. Exact — unlike [`mc_core::profile`]'s histogram this keeps every sample,
    /// because a bench holds a few hundred of them rather than a few million.
    ///
    /// Nearest rank, not interpolation, so every printed figure is a frame time the run actually
    /// produced. **Resolution is `1/n`**: with `n = 200` the p99.9 column cannot be anything but
    /// the maximum, and with `n = 8` (the mesher rounds) neither can p95. That is a property of
    /// the sample count, not of the scene, which is why `n` is printed on the line above.
    fn percentile_of(sorted: &[f64], q: f64) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let rank =
            ((q.clamp(0.0, 1.0) * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
        sorted[rank - 1]
    }

    fn report(&self, label: &str) {
        println!(
            "  {label:<38} {:>8.3} ms  sd {:>6.3}  min {:>7.3}  max {:>7.3}  (n = {})",
            self.mean(),
            self.sd(),
            self.min(),
            self.max(),
            self.ms.len()
        );
        // A mean cannot see a stutter. Printed on its own line so the line above
        // stays byte-comparable with every figure already recorded.
        let sorted = self.sorted();
        println!(
            "  {:<38} {:>8} p50 {:>7.3}  p95 {:>7.3}  p99 {:>7.3}  p99.9 {:>7.3}",
            "",
            "",
            Self::percentile_of(&sorted, 0.50),
            Self::percentile_of(&sorted, 0.95),
            Self::percentile_of(&sorted, 0.99),
            Self::percentile_of(&sorted, 0.999),
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------------------------

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

/// Bake and install the vanilla assets. `None` means "skip the whole run".
fn install_vanilla() -> Option<mc_assets::Atlas> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        println!("skip: 1.8.9.jar not found; set MC_JARS_DIR");
        return None;
    }
    let mut rm = mc_assets::ResourceManager::new();
    rm.push_pack(Box::new(
        mc_assets::ZipResourcePack::open(&jar).expect("open 1.8.9.jar"),
    ));
    let started = Instant::now();
    let (assets, report) = mc_assets::bake_all(&rm, 2048).expect("bake_all");
    let bake = started.elapsed();
    println!(
        "  bake_all: {:.1} ms for {} model states, atlas {}x{}",
        bake.as_secs_f64() * 1000.0,
        report.model_states_considered,
        assets.atlas.w,
        assets.atlas.h
    );
    let atlas = assets.atlas.clone();
    mc_assets::model::install(assets);
    mc_render::terrain::tint::load_colormaps(&rm).expect("colormaps");
    Some(atlas)
}

fn gpu() -> Option<Gpu> {
    match Gpu::headless(
        WIDTH,
        HEIGHT,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(g) => Some(g),
        Err(RenderError::NoAdapter) => {
            println!("skip: no wgpu adapter available");
            None
        }
        Err(e) => panic!("headless context: {e}"),
    }
}

/// A dense scene: stone up to y = 47 with a chequer of air pockets, a water table, and a scatter
/// of leaves and glass so all four render layers are populated. Neither the all-solid nor the
/// all-air fast path dominates, which is what makes the mesher numbers representative.
///
/// This is `s11_perf.rs`'s `quarry` with the non-solid layers added — one shape, so the two
/// files' numbers are comparable.
fn quarry(radius: i32) -> World {
    let mut w = fixture::blank_world(radius);
    let stone = fixture::state_of("minecraft:stone");
    let water = fixture::state_of("minecraft:water");
    let leaves = fixture::state_of("minecraft:leaves");
    let glass = fixture::state_of("minecraft:glass");
    let lo = -radius * 16;
    let hi = radius * 16 + 15;
    fixture::fill(
        &mut w,
        BlockPos::new(lo, 0, lo),
        BlockPos::new(hi, 47, hi),
        stone,
    );
    for x in (lo..=hi).step_by(4) {
        for z in (lo..=hi).step_by(4) {
            for y in (8..48).step_by(4) {
                w.set_block(BlockPos::new(x, y, z), BlockState::AIR);
            }
        }
    }
    // A water table and some cutout geometry above the stone.
    for x in (lo..=hi).step_by(2) {
        for z in (lo..=hi).step_by(2) {
            w.set_block(BlockPos::new(x, 48, z), water);
            if (x + z) % 6 == 0 {
                w.set_block(BlockPos::new(x, 49, z), leaves);
            }
            if (x + z) % 10 == 0 {
                w.set_block(BlockPos::new(x, 50, z), glass);
            }
        }
    }
    w
}

// ---------------------------------------------------------------------------------------------
// The `survival` fixture
// ---------------------------------------------------------------------------------------------
//
// `quarry` above is kept **unchanged and default**, because every recorded baseline figure was
// taken on it and it is the control for the per-layer split. This is an additional fixture,
// selected with `--fixture survival`.
//
// The premise T0.2 states is that quarry's translucent layer is near-empty, so every
// translucent-path measurement in §1.6 is weakly loaded. Whether that premise holds is a
// measurement, not an assumption — the bench prints the split for **both** fixtures whenever the
// survival one is selected, so the claim can be checked rather than believed.
//
// What this fixture is, and what it is not: it is a *plausible* survival-terrain shape, not a
// reimplementation of vanilla's world generator. Nothing here claims to match `ChunkProviderGenerate`.
// What it does claim is load in all four render layers from the causes that produce it in a real
// world — a heightmap surface, a sea, carved caves with interiors and mouths, trees, ground
// cover, and built structures with glass in them.

/// Sea level. Everything below it that is not solid is water.
const SEA: i32 = 42;

/// A deterministic 32-bit hash of a block position.
///
/// Not cryptographic and not meant to be. The only property required is that two runs of the
/// bench build the identical world, so an A/B across a code change is not also an A/B across a
/// different scene.
fn hash3(x: i32, y: i32, z: i32, salt: u32) -> u32 {
    let mut h = salt.wrapping_add(0x9E37_79B9);
    for v in [x as u32, y as u32, z as u32] {
        h ^= v.wrapping_mul(0x85EB_CA6B);
        h = h.rotate_left(13).wrapping_mul(0xC2B2_AE35);
        h ^= h >> 16;
    }
    h
}

/// A hash mapped into `0.0..=1.0`.
fn unit(h: u32) -> f64 {
    f64::from(h) / f64::from(u32::MAX)
}

/// Hermite smoothstep, so the value-noise lattice does not show as a grid of creases.
fn smooth(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// One octave of 2-D value noise on a `cell`-block lattice, in `0.0..=1.0`.
fn noise2(x: f64, z: f64, cell: f64, salt: u32) -> f64 {
    let (fx, fz) = (x / cell, z / cell);
    let (x0, z0) = (fx.floor(), fz.floor());
    let (tx, tz) = (smooth(fx - x0), smooth(fz - z0));
    let at = |ix: f64, iz: f64| unit(hash3(ix as i32, 0, iz as i32, salt));
    lerp(
        lerp(at(x0, z0), at(x0 + 1.0, z0), tx),
        lerp(at(x0, z0 + 1.0), at(x0 + 1.0, z0 + 1.0), tx),
        tz,
    )
}

/// Ground height at `(x, z)`, roughly `34..=52`, centred one block above [`SEA`].
///
/// Three octaves, so there are both hills and metre-scale roughness — the second is what stops
/// whole 16×16 section faces being flat and therefore trivially culled, which would make the
/// fixture easier than a real world rather than harder.
///
/// **The octave sizes are chosen against the smallest radius the bench runs at, not for realism.**
/// The largest is 26 blocks, so even `--radius 2` (an 80-block span) contains three of them and
/// the terrain crosses sea level several times. A first attempt used a 96-block first octave —
/// far more plausible terrain — and produced a radius-2 fixture that was *entirely underwater*,
/// with zero leaves, zero grass and a cutout-mipped layer of exactly nothing. A fixture whose
/// layer loading depends on which radius you happen to pass is worse than an unrealistic one.
///
/// Each octave is centred on zero (`noise - 0.5`) so the mean height is the constant, not the
/// constant plus half the amplitudes: that is what puts a predictable ~55 % of the world above
/// water at every radius rather than leaving it to where the lattice happens to fall.
///
/// The ceiling matters: the bench meshes sections `y in 0..4`, i.e. blocks `0..=63`. Terrain tops
/// out near 52 and a tree adds at most 7, so nothing this fixture builds falls outside the meshed
/// volume and silently vanishes from the quad counts.
fn surface_height(x: i32, z: i32) -> i32 {
    let (xf, zf) = (f64::from(x), f64::from(z));
    (43.0
        + (noise2(xf, zf, 26.0, 0x51ED) - 0.5) * 11.0
        + (noise2(xf, zf, 11.0, 0x2A17) - 0.5) * 5.0
        + (noise2(xf, zf, 5.0, 0x77C3) - 0.5) * 2.5) as i32
}

/// Every block state this fixture places, resolved once. `state_of` panics on an unregistered
/// name, so resolving up front turns a typo into one failure at startup rather than a silent
/// hole in the middle of the world.
struct Palette {
    stone: BlockState,
    dirt: BlockState,
    grass: BlockState,
    sand: BlockState,
    gravel: BlockState,
    clay: BlockState,
    cobble: BlockState,
    mossy: BlockState,
    ores: [BlockState; 6],
    log: BlockState,
    leaves: BlockState,
    tallgrass: BlockState,
    red_flower: BlockState,
    yellow_flower: BlockState,
    torch: BlockState,
    glass: BlockState,
    ladder: BlockState,
    waterlily: BlockState,
    water: BlockState,
    ice: BlockState,
    stained_glass: BlockState,
}

impl Palette {
    fn new() -> Self {
        let s = fixture::state_of;
        Palette {
            stone: s("minecraft:stone"),
            dirt: s("minecraft:dirt"),
            grass: s("minecraft:grass"),
            sand: s("minecraft:sand"),
            gravel: s("minecraft:gravel"),
            clay: s("minecraft:clay"),
            cobble: s("minecraft:cobblestone"),
            mossy: s("minecraft:mossy_cobblestone"),
            ores: [
                s("minecraft:coal_ore"),
                s("minecraft:iron_ore"),
                s("minecraft:gold_ore"),
                s("minecraft:redstone_ore"),
                s("minecraft:lapis_ore"),
                s("minecraft:diamond_ore"),
            ],
            log: s("minecraft:log"),
            leaves: s("minecraft:leaves"),
            tallgrass: s("minecraft:tallgrass"),
            red_flower: s("minecraft:red_flower"),
            yellow_flower: s("minecraft:yellow_flower"),
            torch: s("minecraft:torch"),
            glass: s("minecraft:glass"),
            ladder: s("minecraft:ladder"),
            waterlily: s("minecraft:waterlily"),
            water: s("minecraft:water"),
            ice: s("minecraft:ice"),
            stained_glass: s("minecraft:stained_glass"),
        }
    }
}

/// Terrain heavy in **all four** render layers: a hilly heightmap with ores, carved caves that
/// both stay interior and break the surface, a sea with ice and lily pads, forest, ground cover,
/// torches, and small glass-windowed ruins.
///
/// Built in five passes, in this order, because each depends on the one before:
/// column stack → caves → water → surface cover and trees → structures.
///
/// Deterministic: no RNG, only [`hash3`] of the block position, so two runs mesh the same world.
fn survival(radius: i32) -> World {
    let mut w = fixture::blank_world(radius);
    let p = Palette::new();
    let lo = -radius * 16;
    let hi = radius * 16 + 15;

    // ---- 1. the column stack --------------------------------------------------------------
    // Stone to four below the surface, then a soil cap whose material depends on where the
    // column sits relative to the sea: grass inland, sand on the shore, gravel and clay in the
    // deeper water. Four materials rather than one because a single material would let the
    // mesher's texture lookup run entirely out of cache in a way a real world does not.
    let mut land_columns = 0u64;
    let mut all_columns = 0u64;
    for x in lo..=hi {
        for z in lo..=hi {
            let h = surface_height(x, z);
            all_columns += 1;
            if h > SEA + 1 {
                land_columns += 1;
            }
            let cap = if h > SEA + 1 {
                p.grass
            } else if h >= SEA - 2 {
                p.sand
            } else if hash3(x, 3, z, 0x0A11).is_multiple_of(3) {
                p.clay
            } else {
                p.gravel
            };
            for y in 0..=h {
                let state = if y >= h - 3 && h > SEA + 1 {
                    if y == h {
                        cap
                    } else {
                        p.dirt
                    }
                } else if y >= h - 2 {
                    cap
                } else {
                    p.stone
                };
                w.set_block(BlockPos::new(x, y, z), state);
            }
        }
    }

    // ---- 1b. ores ---------------------------------------------------------------------------
    // Small blobs on a coarse lattice. These do not change the quad count — an ore block is a
    // full cube exactly like the stone it replaces — so they are here for the *texture* variety
    // in the solid layer, and they only become visible geometry where a cave wall exposes them.
    for bx in (lo..=hi).step_by(5) {
        for bz in (lo..=hi).step_by(5) {
            let ceiling = surface_height(bx, bz) - 5;
            for by in (5..ceiling.max(6)).step_by(4) {
                let r = hash3(bx, by, bz, 0x0E17);
                if !r.is_multiple_of(4) {
                    continue;
                }
                // Rarer ores deeper, as in vanilla — diamond only near the bottom.
                let depth_bias = if by < 14 { 6 } else { 4 };
                let ore = p.ores[((r >> 8) % depth_bias) as usize];
                for dx in 0..2 {
                    for dy in 0..2 {
                        for dz in 0..2 {
                            let pos = BlockPos::new(bx + dx, by + dy, bz + dz);
                            if w.block_at(pos) == p.stone {
                                w.set_block(pos, ore);
                            }
                        }
                    }
                }
            }
        }
    }

    // ---- 2. caves ---------------------------------------------------------------------------
    // Random-walk tunnels rather than a 3-D noise field, because a noise field would evaluate
    // noise at every one of the ~8 million blocks in the volume while a walk only touches what it
    // carves. The shape is what matters here: an interior void with a floor, a ceiling and walls
    // gives the mesher real interior faces and the visibility graph a real reason to say no.
    //
    // One tunnel in five is allowed to break the surface, so some caves are reachable from
    // outside and the culling walk has to descend into them rather than writing every cave
    // section off as disconnected.
    let span = i64::from(hi - lo + 1);
    let tunnels = ((span * span) / 512).max(4) as u32;
    for t in 0..tunnels {
        let seed = hash3(t as i32, 0, 0, 0xCA7E);
        let mut cx = f64::from(lo) + unit(seed) * f64::from(hi - lo);
        let mut cz = f64::from(lo) + unit(hash3(t as i32, 1, 0, 0xCA7E)) * f64::from(hi - lo);
        let mut cy = 6.0 + unit(hash3(t as i32, 2, 0, 0xCA7E)) * 26.0;
        let mut yaw = unit(hash3(t as i32, 3, 0, 0xCA7E)) * std::f64::consts::TAU;
        let mut pitch = (unit(hash3(t as i32, 4, 0, 0xCA7E)) - 0.5) * 0.6;
        let breaches_surface = seed.is_multiple_of(5);
        let steps = 80 + (seed >> 3) % 120;
        for step in 0..steps {
            let s = hash3(t as i32, step as i32, 1, 0x5EED);
            yaw += (unit(s) - 0.5) * 0.45;
            pitch = (pitch + (unit(s >> 7) - 0.5) * 0.22).clamp(-0.5, 0.5);
            cx += yaw.cos() * pitch.cos();
            cz += yaw.sin() * pitch.cos();
            cy += pitch.sin();
            let (bx, by, bz) = (cx as i32, cy as i32, cz as i32);
            if bx < lo + 2 || bx > hi - 2 || bz < lo + 2 || bz > hi - 2 || !(4..56).contains(&by) {
                break;
            }
            let r = 1.6 + unit(s >> 13) * 1.6;
            let ri = r.ceil() as i32;
            for dx in -ri..=ri {
                for dy in -ri..=ri {
                    for dz in -ri..=ri {
                        if f64::from(dx * dx + dy * dy + dz * dz) > r * r {
                            continue;
                        }
                        let pos = BlockPos::new(bx + dx, by + dy, bz + dz);
                        let cap =
                            surface_height(pos.x, pos.z) - if breaches_surface { 0 } else { 3 };
                        if pos.y < 3 || pos.y > cap {
                            continue;
                        }
                        w.set_block(pos, BlockState::AIR);
                    }
                }
            }
            // A torch every dozen steps, on the tunnel floor, so the cutout layer is loaded
            // underground as well as on the surface.
            if step % 12 == 0 {
                let floor = BlockPos::new(bx, by - ri, bz);
                let below = BlockPos::new(bx, by - ri - 1, bz);
                if w.block_at(floor) == BlockState::AIR
                    && w.block_at(below) != BlockState::AIR
                    && w.block_at(below) != p.water
                {
                    w.set_block(floor, p.torch);
                }
            }
        }
    }

    // ---- 3. the sea -------------------------------------------------------------------------
    // Everything at or below `SEA` that is now air becomes water. This is a *volume*, not a
    // sheet: the translucent layer gets shorelines, a surface, and the sides of every cave the
    // carver opened into a lake bed.
    for x in lo..=hi {
        for z in lo..=hi {
            let h = surface_height(x, z);
            if h >= SEA {
                continue;
            }
            for y in (h + 1)..=SEA {
                let pos = BlockPos::new(x, y, z);
                if w.block_at(pos) == BlockState::AIR {
                    w.set_block(pos, p.water);
                }
            }
            // A cold band freezes the surface, and lily pads sit on the rest. Ice is translucent
            // and a full cube; a lily pad is cutout and a single horizontal quad. Both are here
            // because they load different paths.
            let cold = noise2(f64::from(x), f64::from(z), 140.0, 0x1CE0) < 0.36;
            let r = hash3(x, SEA, z, 0x11FA);
            if cold && !r.is_multiple_of(4) {
                w.set_block(BlockPos::new(x, SEA, z), p.ice);
            } else if !cold && r.is_multiple_of(128) {
                w.set_block(BlockPos::new(x, SEA + 1, z), p.waterlily);
            }
        }
    }

    // ---- 4. forest and ground cover ---------------------------------------------------------
    // Leaves are the fixture's main cutout-mipped load and the reason T2.1 (fast leaves) has
    // something to measure against. In fancy graphics a leaf block is not an opaque cube, so
    // interior faces of a canopy are *not* culled — which is exactly the cost a fast-leaves
    // option removes, and it has to be present here or that item cannot be evaluated.
    for x in lo..=hi {
        for z in lo..=hi {
            let h = surface_height(x, z);
            if h <= SEA + 1 || w.block_at(BlockPos::new(x, h, z)) != p.grass {
                continue;
            }
            let r = hash3(x, 7, z, 0x7A3E);
            if r % 1000 < 7 && x > lo + 3 && x < hi - 3 && z > lo + 3 && z < hi - 3 {
                let trunk = 4 + (r >> 10) % 3;
                for i in 1..=trunk as i32 {
                    w.set_block(BlockPos::new(x, h + i, z), p.log);
                }
                let top = h + trunk as i32;
                for (level, radius_xz) in [
                    (top - 2, 2i32),
                    (top - 1, 2i32),
                    (top, 1i32),
                    (top + 1, 1i32),
                ] {
                    for dx in -radius_xz..=radius_xz {
                        for dz in -radius_xz..=radius_xz {
                            if radius_xz == 2
                                && dx.abs() == 2
                                && dz.abs() == 2
                                && hash3(x + dx, level, z + dz, 0x1EA5).is_multiple_of(2)
                            {
                                continue; // ragged corners, like vanilla's oak
                            }
                            let pos = BlockPos::new(x + dx, level, z + dz);
                            if w.block_at(pos) == BlockState::AIR {
                                w.set_block(pos, p.leaves);
                            }
                        }
                    }
                }
                continue;
            }
            // Ground cover: cross-model plants, the cheapest possible cutout geometry, in the
            // quantity a real plains chunk carries.
            let above = BlockPos::new(x, h + 1, z);
            if w.block_at(above) != BlockState::AIR {
                continue;
            }
            match r % 1000 {
                7..=249 => w.set_block(above, p.tallgrass),
                250..=274 => w.set_block(above, p.red_flower),
                275..=299 => w.set_block(above, p.yellow_flower),
                300..=304 => w.set_block(above, p.torch),
                _ => {}
            }
        }
    }

    // ---- 5. ruins -----------------------------------------------------------------------------
    // Small built structures on a coarse lattice: cobblestone shells with glass windows, a
    // stained-glass roof, a ladder and a torch inside. Few in number and small, so they are a
    // *variety* contribution rather than a load one — but they are the only source of
    // `stained_glass` (translucent, and unlike water it is a full cube) and of `ladder` (cutout
    // with a non-cross model), and a fixture with neither would leave both paths untested.
    for bx in (lo + 8..hi - 8).step_by(64) {
        for bz in (lo + 8..hi - 8).step_by(64) {
            let h = surface_height(bx, bz);
            if h <= SEA + 1 {
                continue;
            }
            for dx in 0..6 {
                for dz in 0..6 {
                    for dy in 1..=4 {
                        let edge = dx == 0 || dx == 5 || dz == 0 || dz == 5;
                        let pos = BlockPos::new(bx + dx, h + dy, bz + dz);
                        if dy == 4 {
                            w.set_block(pos, p.stained_glass);
                        } else if edge {
                            let window = dy == 2 && (dx == 2 || dx == 3 || dz == 2 || dz == 3);
                            w.set_block(
                                pos,
                                if window {
                                    p.glass
                                } else if hash3(pos.x, pos.y, pos.z, 0x0B05).is_multiple_of(4) {
                                    p.mossy
                                } else {
                                    p.cobble
                                },
                            );
                        }
                    }
                }
            }
            w.set_block(BlockPos::new(bx + 2, h + 1, bz + 2), p.torch);
            w.set_block(BlockPos::new(bx + 4, h + 1, bz + 1), p.ladder);
            w.set_block(BlockPos::new(bx + 4, h + 2, bz + 1), p.ladder);
        }
    }

    // A census, not a decoration: if this is 0 % or 100 % the fixture has silently lost either
    // its forest or its sea, and the per-layer split below would be measuring something other
    // than what this function claims to build.
    println!(
        "  {:<38} {:>7.1} % of {all_columns} columns above sea level, {tunnels} cave tunnels",
        "fixture census",
        100.0 * land_columns as f64 / all_columns.max(1) as f64,
    );

    w
}

/// Quads and bytes in each of the four render layers, for a whole meshed fixture.
///
/// T0.2's verification is this split and nothing else: the claim under test is that quarry's
/// translucent layer is near-empty, and only a per-layer count can confirm or refute it.
fn layer_split(meshes: &[mc_render::terrain::ChunkMesh]) -> Vec<(&'static str, usize, usize)> {
    let names = ["solid", "cutout_mipped", "cutout", "translucent"];
    let mut out: Vec<(&'static str, usize, usize)> =
        names.iter().map(|n| (*n, 0usize, 0usize)).collect();
    for mesh in meshes {
        for (index, (_, buffer)) in mesh.layers().enumerate() {
            out[index].1 += buffer.quad_count();
            out[index].2 += buffer.byte_size();
        }
    }
    out
}

/// Print one fixture's split, and — when a control is supplied — the same rows for the fixture
/// the recorded baseline was measured on, side by side.
fn report_layer_split(
    label: &str,
    meshes: &[mc_render::terrain::ChunkMesh],
    control: Option<(&str, &[mc_render::terrain::ChunkMesh])>,
) {
    let rows = layer_split(meshes);
    let total: usize = rows.iter().map(|r| r.1).sum();
    let control_rows = control.map(|(_, m)| layer_split(m));
    let control_total: usize = control_rows
        .as_ref()
        .map_or(0, |r| r.iter().map(|x| x.1).sum());

    match control {
        None => println!(
            "  {:<16} {:>12} {:>8} {:>10}",
            "layer", "quads", "share", "MiB"
        ),
        Some((name, _)) => println!(
            "  {:<16} {:>12} {:>8} {:>10}   |{:>12} {:>8} {:>10}  <- control: {name}",
            "layer", label, "share", "MiB", name, "share", "MiB"
        ),
    }
    for (index, (name, quads, bytes)) in rows.iter().enumerate() {
        let share = 100.0 * *quads as f64 / total.max(1) as f64;
        let mib = *bytes as f64 / (1024.0 * 1024.0);
        match &control_rows {
            None => println!("  {name:<16} {quads:>12} {share:>7.2}% {mib:>10.2}"),
            Some(c) => {
                let (_, cq, cb) = c[index];
                println!(
                    "  {name:<16} {quads:>12} {share:>7.2}% {mib:>10.2}   |{cq:>12} {:>7.2}% {:>10.2}",
                    100.0 * cq as f64 / control_total.max(1) as f64,
                    cb as f64 / (1024.0 * 1024.0)
                );
            }
        }
    }
    match &control_rows {
        None => println!("  {:<16} {total:>12}", "total"),
        Some(_) => println!(
            "  {:<16} {total:>12} {:>8} {:>10}   |{control_total:>12}",
            "total", "", ""
        ),
    }
}

fn sections(radius: i32, height: i32) -> Vec<ChunkSectionPos> {
    (-radius..=radius)
        .flat_map(|x| {
            (-radius..=radius)
                .flat_map(move |z| (0..height).map(move |y| ChunkSectionPos::new(x, y, z)))
        })
        .collect()
}

fn camera() -> Camera {
    let mut c = Camera {
        pos: DVec3::new(8.0, 52.0, 8.0),
        yaw: 35.0,
        pitch: 10.0,
        fov: 70.0,
        near: 0.05,
        far: 512.0,
        aspect: WIDTH as f32 / HEIGHT as f32,
        ..Camera::default()
    };
    c.update();
    c
}

/// The culling walk **exactly as it stood before S23**: a fresh `HashSet<ChunkSectionPos>`, a
/// fresh `VecDeque` and a fresh output `Vec` per call.
///
/// This is the A/B control for the two changes S23 made to `cull.rs`. It is a transcription of
/// the previous `visible_sections` body, not a paraphrase — same rules, same order, same
/// early-outs — so that the only difference the measurement can be attributed to is where the
/// visited set lives. `2b` asserts it returns the same number of sections as the new walk before
/// timing either.
fn reference_walk(
    source: &impl mc_render::terrain::SectionSource,
    frustum: &mc_render::terrain::Frustum,
    camera: DVec3,
    render_distance: u32,
    smart_culling: bool,
) -> Vec<mc_render::terrain::VisibleSection> {
    use mc_math::Facing;
    use mc_render::terrain::VisibleSection;
    use std::collections::{HashSet, VecDeque};

    let contains = |set: u8, f: Facing| set & (1 << f.index()) != 0;
    let with = |set: u8, f: Facing| set | (1 << f.index());

    let eye = BlockPos::from_vec3(camera);
    let start = eye.section_pos();
    let mut out: Vec<VisibleSection> = Vec::new();
    let mut seen: HashSet<ChunkSectionPos> = HashSet::new();
    let mut queue: VecDeque<(VisibleSection, u8)> = VecDeque::new();

    if (0..16).contains(&start.y) {
        seen.insert(start);
        queue.push_back((
            VisibleSection {
                pos: start,
                entered_through: None,
                depth: 0,
            },
            0,
        ));
    } else {
        let y = if eye.y > 0 { 15 } else { 0 };
        let r = render_distance as i32;
        for dx in -r..=r {
            for dz in -r..=r {
                let pos = ChunkSectionPos::new(start.x + dx, y, start.z + dz);
                if frustum.contains_section(pos) && seen.insert(pos) {
                    queue.push_back((
                        VisibleSection {
                            pos,
                            entered_through: None,
                            depth: 0,
                        },
                        0,
                    ));
                }
            }
        }
    }

    while let Some((info, path)) = queue.pop_front() {
        out.push(info);
        let vis = source.visibility(info.pos);
        for step in Facing::VALUES {
            if smart_culling && contains(path, step.opposite()) {
                continue;
            }
            if smart_culling {
                if let (Some(entered), Some(vis)) = (info.entered_through, vis) {
                    if !vis.is_visible(entered.opposite(), step) {
                        continue;
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
                with(path, step),
            ));
        }
    }
    out
}

fn profile_name() -> &'static str {
    if cfg!(debug_assertions) {
        "dev (opt-level 1, deps at 3, debug_assertions on)"
    } else {
        "release-like (debug_assertions off)"
    }
}

// ---------------------------------------------------------------------------------------------
// Measurements
// ---------------------------------------------------------------------------------------------

/// Time one frame end to end, returning `(cpu record, gpu completion)`.
fn one_frame(gpu: &mut Gpu, phases: &mut PhaseList, cam: &Camera) -> (Duration, Duration) {
    let frame = gpu.acquire().expect("acquire offscreen frame");
    let t0 = Instant::now();
    let index = phases.render(gpu, frame.view(), cam, 0.0);
    let cpu = t0.elapsed();
    let t1 = Instant::now();
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .expect("poll");
    (cpu, t1.elapsed())
}

fn measure_frames(
    gpu: &mut Gpu,
    phases: &mut PhaseList,
    cam: &Camera,
    frames: usize,
    label: &str,
) -> (Samples, Samples) {
    // The first frame in a fresh process pays for pipeline and shader compilation. Measured and
    // reported rather than quietly dropped.
    let (first_cpu, first_gpu) = one_frame(gpu, phases, cam);
    for _ in 0..4 {
        one_frame(gpu, phases, cam);
    }
    let mut cpu = Samples::new();
    let mut wait = Samples::new();
    for _ in 0..frames {
        let (c, g) = one_frame(gpu, phases, cam);
        cpu.push(c);
        wait.push(g);
    }
    println!("  [{label}]");
    println!(
        "  {:<38} {:>8.3} ms cpu + {:>7.3} ms gpu  (one-time; excluded below)",
        "first frame of the process",
        first_cpu.as_secs_f64() * 1000.0,
        first_gpu.as_secs_f64() * 1000.0
    );
    cpu.report("phases.render (cpu record+submit)");
    wait.report("gpu completion (device.poll)");
    (cpu, wait)
}

fn main() {
    let mut frames = 200usize;
    let mut radius = 2i32;
    let mut fixture_name = "quarry".to_owned();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--frames" => {
                frames = args
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(frames);
                i += 2;
            }
            "--radius" => {
                radius = args
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(radius);
                i += 2;
            }
            // `quarry` is the default and stays the control: every recorded baseline figure was
            // taken on it.
            "--fixture" => {
                fixture_name = args.get(i + 1).cloned().unwrap_or(fixture_name);
                assert!(
                    matches!(fixture_name.as_str(), "quarry" | "survival"),
                    "--fixture takes `quarry` or `survival`, not `{fixture_name}`"
                );
                i += 2;
            }
            // `cargo bench` passes `--bench`; ignore anything unrecognised.
            _ => i += 1,
        }
    }

    println!("S23 frame-loop profile");
    println!("  profile: {}", profile_name());
    println!("  rayon threads: {}", rayon::current_num_threads());
    println!("  target: {WIDTH}x{HEIGHT} headless, no MSAA, no vsync");
    println!("  fixture: {fixture_name} (radius {radius})");

    let Some(atlas) = install_vanilla() else {
        return;
    };
    let Some(mut gpu) = gpu() else {
        return;
    };
    println!();

    let built = Instant::now();
    let world = if fixture_name == "survival" {
        survival(radius)
    } else {
        quarry(radius)
    };
    println!(
        "  {:<38} {:>8.1} ms",
        format!("built the `{fixture_name}` fixture in"),
        built.elapsed().as_secs_f64() * 1000.0
    );
    let all = sections(radius, 4);
    let cam = camera();

    // ---- 1. the mesher -----------------------------------------------------------------------
    println!("1. mesher (per section, {} sections per round)", all.len());
    let warm = jobs::mesh_all(&world, &all, MeshSettings::default());
    let quads: usize = warm
        .iter()
        .map(mc_render::terrain::ChunkMesh::quad_count)
        .sum();
    let bytes: usize = warm.iter().map(|m| m.byte_size()).sum();

    // ---- 1a. the per-layer split --------------------------------------------
    // The verification T0.2 asks for. When a fixture other than `quarry` is selected, `quarry` is
    // built and meshed here too, in the same process on the same machine, so the comparison is
    // interleaved rather than across two runs — §0 rule 1.
    println!("1a. per-layer quad split");
    if fixture_name == "quarry" {
        report_layer_split("quarry", &warm, None);
    } else {
        let control_world = quarry(radius);
        let control = jobs::mesh_all(&control_world, &all, MeshSettings::default());
        report_layer_split(&fixture_name, &warm, Some(("quarry", &control)));
    }
    println!();
    let mut mesh_rounds = Samples::new();
    for _ in 0..8 {
        let t = Instant::now();
        let out = jobs::mesh_all(&world, &all, MeshSettings::default());
        mesh_rounds.push(t.elapsed());
        std::hint::black_box(&out);
    }
    mesh_rounds.report("mesh_all (whole batch, parallel)");
    println!(
        "  {:<38} {:>8.3} ms/section, {quads} quads, {:.2} MiB of vertex data",
        "per section",
        mesh_rounds.mean() / all.len() as f64,
        bytes as f64 / (1024.0 * 1024.0)
    );

    // Serial, single-threaded, for a number that does not depend on the core count.
    let mut serial = Samples::new();
    for _ in 0..4 {
        let t = Instant::now();
        for pos in &all {
            std::hint::black_box(mc_render::terrain::mesh_section(&world, *pos));
        }
        serial.push(t.elapsed());
    }
    println!(
        "  {:<38} {:>8.3} ms/section (1 thread)",
        "mesh_section serial",
        serial.mean() / all.len() as f64
    );
    // The same walk with the per-block fast paths switched off, in the same process
    // and interleaved with the line above, so the ratio is their measured speed-up. Byte-identical
    // output is pinned by tests/mesher_fast_path_parity.rs.
    let mut reference = Samples::new();
    for _ in 0..4 {
        let t = Instant::now();
        for pos in &all {
            std::hint::black_box(mc_render::terrain::mesher::mesh_section_reference(
                &world,
                *pos,
                MeshSettings::default(),
                true,
            ));
        }
        reference.push(t.elapsed());
    }
    println!(
        "  {:<38} {:>8.3} ms/section (1 thread, fast paths {:.2}x faster)",
        "mesh_section reference (pre-#43 paths)",
        reference.mean() / all.len() as f64,
        reference.mean() / serial.mean()
    );
    println!();

    // ---- 1b. the vacant-section fast path ----------------------------------------------------
    // A render-distance-12 client holds 25 x 25 x 16 = 10,000 sections and most of them are sky.
    // The control is the same call on sections that are *not* vacant, which is what shows the
    // fast path is doing something rather than the sections being cheap anyway.
    println!("1b. vacant sections (the sky above the terrain)");
    {
        let sky: Vec<ChunkSectionPos> = sections(radius, 16)
            .into_iter()
            .filter(|p| p.y >= 4)
            .collect();
        let mut vacant = Samples::new();
        for _ in 0..8 {
            let t = Instant::now();
            for pos in &sky {
                std::hint::black_box(mc_render::terrain::mesh_section(&world, *pos));
            }
            vacant.push(t.elapsed());
        }
        println!(
            "  {:<38} {:>8.4} ms/section  ({} sections)",
            "vacant (fast path)",
            vacant.mean() / sky.len() as f64,
            sky.len()
        );
        println!(
            "  {:<38} {:>8.4} ms/section  <- control, same call",
            "non-vacant (full walk)",
            serial.mean() / all.len() as f64
        );
        println!(
            "  {:<38} {:>8.0}x",
            "ratio",
            (serial.mean() / all.len() as f64) / (vacant.mean() / sky.len() as f64).max(1e-9)
        );
    }
    println!();

    // ---- 2. the queue ------------------------------------------------------------------------
    // `enqueue_dirty` runs once a tick over everything the world marked dirty; on a render
    // distance change or a teleport that is every section at once.
    //
    // Both measurements are A/B against the algorithm S23 replaced, reimplemented here over the
    // same data on the same machine in the same run. A speedup with no "before" is an assertion,
    // not a measurement.
    println!("2. mesh queue (the per-tick path) — new vs. the algorithm S23 replaced");
    for n in [256usize, 4096, 16384] {
        let positions: Vec<ChunkSectionPos> = (0..n as i32)
            .map(|i| ChunkSectionPos::new(i % 64, (i / 64) % 16, i / 1024))
            .collect();
        let mut enq = Samples::new();
        let mut old = Samples::new();
        for _ in 0..5 {
            let mut q = mc_render::terrain::MeshQueue::new();
            let t = Instant::now();
            for p in &positions {
                q.enqueue(*p);
            }
            enq.push(t.elapsed());
            // The bounded active queue retains overflow in its deferred set.
            assert_eq!(q.len() + q.deferred_len(), n);

            // Before S23: `if !pending.contains(&pos) { pending.push(pos) }`.
            let mut pending: Vec<ChunkSectionPos> = Vec::new();
            let t = Instant::now();
            for p in &positions {
                if !pending.contains(p) {
                    pending.push(*p);
                }
            }
            old.push(t.elapsed());
            assert_eq!(pending.len(), n);
        }
        enq.report(&format!("enqueue {n}: HashSet dedup"));
        old.report(&format!("enqueue {n}: linear scan (before)"));
    }
    {
        let positions: Vec<ChunkSectionPos> = (0..16384i32)
            .map(|i| ChunkSectionPos::new(i % 64, (i / 64) % 16, i / 1024))
            .collect();
        let mut new = Samples::new();
        let mut old = Samples::new();
        for _ in 0..20 {
            let mut q = mc_render::terrain::MeshQueue::new();
            for p in &positions {
                q.enqueue(*p);
            }
            let t = Instant::now();
            let taken = q.take_nearest(cam.pos, jobs::DEFAULT_BUDGET);
            new.push(t.elapsed());
            assert_eq!(taken.len(), jobs::DEFAULT_BUDGET);

            // Before S23: sort the whole queue, then take the first `budget`.
            let mut pending = positions.clone();
            let t = Instant::now();
            pending.sort_by(|a, b| {
                jobs::distance_sq(*a, cam.pos).total_cmp(&jobs::distance_sq(*b, cam.pos))
            });
            let _: Vec<_> = pending.drain(..jobs::DEFAULT_BUDGET).collect();
            old.push(t.elapsed());
        }
        new.report("take 32 of 16384: select_nth_unstable");
        old.report("take 32 of 16384: full sort (before)");
    }
    println!();

    // ---- 2b. the culling walk ----------------------------------------------------------------
    // `begin_frame` runs this once a frame, so it is inside the per-frame cost measured below.
    // S23 measured this at ~60% of terrain's per-frame CPU at render distance 12 and changed two
    // things about it. Both are measured here against the **algorithm they replaced**,
    // reimplemented below over the same data in the same process — a speedup with no "before" is
    // an assertion, not a measurement.
    println!("2b. culling walk (inside begin_frame, once per frame) — new vs. what S23 replaced");
    {
        use mc_render::terrain::{cull, frustum::Frustum, SectionWalk, SetVisibility};
        let f = Frustum::from_camera(&cam);
        let open = |p: ChunkSectionPos| (0..16).contains(&p.y).then(|| SetVisibility::all(true));
        for distance in [8u32, 12, 16] {
            let mut before = Samples::new();
            let mut one_shot = Samples::new();
            let mut reused = Samples::new();
            let mut cached = Samples::new();
            let mut count = 0;
            let mut walk = SectionWalk::new();
            // Warm both paths so neither pays a first-touch allocation inside a timed region.
            let reference_count = reference_walk(&open, &f, cam.pos, distance, true).len();
            walk.run(&open, &f, cam.pos, distance, true, 0);
            assert_eq!(
                walk.sections().len(),
                reference_count,
                "the reused walk must answer what the algorithm it replaced answered"
            );

            for i in 0..40u64 {
                // Before S23: a fresh `HashSet`, `VecDeque` and `Vec` every frame.
                let t = Instant::now();
                let old = reference_walk(&open, &f, cam.pos, distance, true);
                before.push(t.elapsed());
                std::hint::black_box(&old);

                // After, one-shot (the free function, which still allocates per call).
                let t = Instant::now();
                count = cull::visible_sections(&open, &f, cam.pos, distance, true).len();
                one_shot.push(t.elapsed());

                // After, with the renderer's persistent buffers. The generation is bumped every
                // round so this is a genuine re-walk, never a cache hit.
                let t = Instant::now();
                walk.run(&open, &f, cam.pos, distance, true, i + 1);
                reused.push(t.elapsed());

                // And the cache hit, which is what a still camera costs.
                let t = Instant::now();
                walk.run(&open, &f, cam.pos, distance, true, i + 1);
                cached.push(t.elapsed());
            }
            let (hits, misses) = walk.cache_counts();
            println!("  [render distance {distance}: {count} sections]");
            before.report("HashSet + fresh buffers (before)");
            one_shot.report("visible_sections (one-shot)");
            reused.report("SectionWalk::run (persistent, re-walk)");
            cached.report("SectionWalk::run (cache hit)");
            println!(
                "  {:<38} {:>7.2}x re-walk, {:>7.0}x cache hit   ({hits} hits / {misses} walks)",
                "speedup vs before",
                before.mean() / reused.mean().max(1e-9),
                before.mean() / cached.mean().max(1e-9)
            );
        }
    }
    println!();

    // ---- 3. upload ---------------------------------------------------------------------------
    // A/B in one process on identical meshes: `upload` once per section (what the renderer did
    // before S23) against `upload_all` (the batched path). `s23_batching.rs` proves the two
    // render byte-identical frames; this is what the difference costs.
    println!(
        "3. gpu upload of {} meshed sections — per-section vs. batched",
        warm.len()
    );
    {
        let mut plain = Samples::new();
        let mut batched = Samples::new();
        let mut per_frame = Samples::new();
        let mut plain_buffers = 0;
        let mut batched_buffers = 0;
        let mut per_frame_buffers = 0;
        for _ in 0..12 {
            let mut r = TerrainRenderer::new(&gpu, &atlas);
            let t = Instant::now();
            for mesh in warm.iter().cloned() {
                r.upload(&gpu, mesh, cam.pos);
            }
            plain.push(t.elapsed());
            plain_buffers = r.gpu_buffer_count();

            let mut r = TerrainRenderer::new(&gpu, &atlas);
            let t = Instant::now();
            r.upload_all(&gpu, warm.clone(), cam.pos);
            batched.push(t.elapsed());
            batched_buffers = r.gpu_buffer_count();

            // What the client actually does: `MeshQueue::drain` hands back at most
            // `DEFAULT_BUDGET` finished meshes per frame, so a real batch is that size, not the
            // whole world. Measured separately because the two are different workloads and the
            // single-giant-batch number is the one that is *not* representative.
            let mut r = TerrainRenderer::new(&gpu, &atlas);
            let t = Instant::now();
            for chunk in warm.chunks(jobs::DEFAULT_BUDGET) {
                r.upload_all(&gpu, chunk.to_vec(), cam.pos);
            }
            per_frame.push(t.elapsed());
            per_frame_buffers = r.gpu_buffer_count();
        }
        // Where the batched path's extra time goes. Batching *adds* one memcpy of the whole
        // vertex payload — the per-section path hands each mesh's bytes straight to
        // `create_buffer_init`, while this one concatenates them into `UploadBatch::bytes` first
        // and `create_buffer_init` then copies that into mapped memory. Split out rather than
        // averaged, because "batching is slower to upload" and "concatenation is slower to
        // upload" are different claims and only the second one is true.
        let mut planning = Samples::new();
        for _ in 0..5 {
            let t = Instant::now();
            let plan = jobs::UploadPlan::plan(&warm, jobs::GpuBudget::new(u64::MAX));
            planning.push(t.elapsed());
            std::hint::black_box(&plan);
        }

        plain.report("upload per section (before)");
        batched.report("upload_all, one batch for everything");
        planning.report("  of which: UploadPlan::plan (cpu concat)");
        println!(
            "  {:<38} {:>8.3} ms  <- the 4 create_buffer_init calls",
            "  of which: gpu buffer creation",
            batched.mean() - planning.mean()
        );
        per_frame.report(&format!(
            "upload_all in {}-section batches (real)",
            jobs::DEFAULT_BUDGET
        ));
        println!(
            "  {:<38} {:>8.3} -> {:>7.3} ms/section (per-section -> real batches)",
            "per section",
            plain.mean() / warm.len().max(1) as f64,
            per_frame.mean() / warm.len().max(1) as f64
        );
        let plan = jobs::UploadPlan::plan(&warm, jobs::GpuBudget::new(u64::MAX));
        println!(
            "  {:<38} {plain_buffers} -> {per_frame_buffers} (real batches) \
             -> {batched_buffers} (one batch); plan says {} -> {}, {:.2} MiB",
            "gpu buffers, counted on the renderer",
            plan.buffers_unbatched,
            plan.batches.len(),
            plan.total_bytes as f64 / (1024.0 * 1024.0)
        );
    }
    println!();

    // ---- 4. the frame ------------------------------------------------------------------------
    println!("4. frame loop");
    // Control first: an empty phase list is the acquire/clear/submit floor. Anything below this
    // is not attributable to terrain.
    let mut empty = PhaseList::new();
    let (floor_cpu, floor_gpu) =
        measure_frames(&mut gpu, &mut empty, &cam, frames, "control: no phases");
    println!();

    // The same scene twice, differing only in how the geometry was uploaded. Both render
    // byte-identical frames (`s23_batching.rs`), so any difference here is the draw loop's.
    for batched in [false, true] {
        let mut phases = PhaseList::new();
        let shared = TerrainPhase::install(&gpu, &atlas, &mut phases);
        {
            let mut r = shared.borrow_mut();
            r.render_distance = 12;
            if batched {
                r.upload_all(&gpu, warm.clone(), cam.pos);
            } else {
                for mesh in warm.iter().cloned() {
                    r.upload(&gpu, mesh, cam.pos);
                }
            }
        }
        let label = if batched {
            "terrain, 4 layers, batched upload"
        } else {
            "terrain, 4 layers, per-section upload (before)"
        };
        let (cpu, wait) = measure_frames(&mut gpu, &mut phases, &cam, frames, label);
        let stats = shared.borrow().stats();
        let (hits, walks) = shared.borrow().cull_cache_counts();
        println!(
            "  {:<38} {} visible of {} loaded, {} draw calls, {} vertex-buffer binds, {} quads",
            "last frame",
            stats.visible_sections,
            stats.loaded_sections,
            stats.draw_calls,
            stats.vertex_buffer_binds,
            stats.quads
        );
        println!(
            "  {:<38} {} gpu buffers, {:.2} MiB resident, cull walk {hits} hits / {walks} walks",
            "state",
            shared.borrow().gpu_buffer_count(),
            shared.borrow().gpu_memory().bytes as f64 / (1024.0 * 1024.0),
        );
        println!(
            "  {:<38} {:>8.3} ms cpu + {:>7.3} ms gpu",
            "terrain's share (measured - control)",
            cpu.mean() - floor_cpu.mean(),
            wait.mean() - floor_gpu.mean()
        );
        let total = cpu.mean() + wait.mean();
        println!(
            "  {:<38} {:>8.3} ms => {:.0} fps if nothing else ran",
            "total frame",
            total,
            1000.0 / total.max(f64::MIN_POSITIVE)
        );

        // The stated verification: "the sum of per-pass timestamps reconciles with
        // the existing poll number, and the difference is named."
        //
        // The poll number lives here and nowhere else, so the reconciliation has to live here
        // too. Three quantities, and they are three different things:
        //
        // * `device.poll` — wall clock from submit to "the queue says this submission is done".
        //   It contains queue latency and driver bookkeeping as well as execution, which is
        //   exactly why T0.4 exists.
        // * `gpu.frame` — first to last timestamp *inside* the command buffer. Execution only.
        // * `poll − gpu.frame` — **submission-to-start latency plus completion signalling**.
        //   That is the named difference. It is not GPU work and never was.
        //
        // And the ratio worth reading off: CPU record time against `gpu.frame`. If recording
        // costs more than executing, the client is command-recording bound and no amount of
        // GPU-side work removal will show up in a frame time.
        let timings = phases.gpu_timings();
        if timings.is_empty() {
            println!(
                "  {:<38} {:>8} (set MC_GPU_TIMING=1 for the T0.4 reconciliation)",
                "gpu timestamps", "-"
            );
        } else {
            let frame_ms = timings.frame_ns.mean_ns() / 1_000_000.0;
            let attributed_ms = timings.attributed_mean_ns() / 1_000_000.0;
            println!(
                "  {:<38} {:>8.4} ms  ({} timed frames, {} discarded pairs)",
                "gpu.frame (timestamps, execution only)",
                frame_ms,
                timings.frames,
                timings.discarded_pairs
            );
            println!(
                "  {:<38} {:>8.4} ms  + {:.4} ms inter-pass = {:.4} ms",
                "  of which attributed to phases",
                attributed_ms,
                timings.unattributed_mean_ns() / 1_000_000.0,
                frame_ms
            );
            println!(
                "  {:<38} {:>8.4} ms  <- submission latency + completion signalling, not GPU work",
                "device.poll - gpu.frame",
                wait.mean() - frame_ms
            );
            println!(
                "  {:<38} {:>8.2}x  (cpu record {:.4} ms vs gpu execute {:.4} ms)",
                if cpu.mean() > frame_ms {
                    "=> command-recording bound by"
                } else {
                    "=> gpu bound by"
                },
                if cpu.mean() > frame_ms {
                    cpu.mean() / frame_ms.max(f64::MIN_POSITIVE)
                } else {
                    frame_ms / cpu.mean().max(f64::MIN_POSITIVE)
                },
                cpu.mean(),
                frame_ms
            );
        }
        println!();
    }

    // ---- 4b. a *moving* camera ---------------------------------------------------------------
    // Section 4 holds the camera still, so the culling walk is served from the cache every frame
    // and the numbers there isolate the draw loop. A player is not still. This runs the same
    // scene with the camera nudged every frame, which is exactly the case the cache cannot help
    // — vanilla's own rule (`RenderGlobal.setupTerrain`) invalidates on any movement at all — so
    // the walk is paid in full on every frame below.
    println!("4b. frame loop with a moving camera (the culling cache cannot help here)");
    {
        let mut phases = PhaseList::new();
        let shared = TerrainPhase::install(&gpu, &atlas, &mut phases);
        {
            let mut r = shared.borrow_mut();
            r.render_distance = 12;
            r.upload_all(&gpu, warm.clone(), cam.pos);
        }
        let mut moving = cam;
        for _ in 0..10 {
            one_frame(&mut gpu, &mut phases, &moving);
        }
        let mut cpu = Samples::new();
        for i in 0..frames {
            moving.pos.x = cam.pos.x + (i as f64) * 0.01;
            moving.update();
            let (c, _) = one_frame(&mut gpu, &mut phases, &moving);
            cpu.push(c);
        }
        let (hits, walks) = shared.borrow().cull_cache_counts();
        cpu.report("phases.render, camera moving");
        println!(
            "  {:<38} {:>8.3} ms cpu   ({hits} cache hits / {walks} walks — as designed, ~none)",
            "terrain's share (measured - control)",
            cpu.mean() - floor_cpu.mean()
        );
    }
    println!();

    // ---- 4c. the draw loop, A/B interleaved --------------------------------------------------
    // Section 4 measures the two upload paths in separate blocks, which is enough for the *counts*
    // (they are deterministic) but not for the microseconds: the per-frame difference is a few
    // percent and the block-to-block spread on this machine is larger than that. So both phase
    // lists are built up front and frames are alternated between them, which is the same control
    // for drift S23 used for the LTO measurement.
    println!("4c. draw loop A/B, interleaved frame by frame");
    {
        let mut per_section_phases = PhaseList::new();
        let per_section = TerrainPhase::install(&gpu, &atlas, &mut per_section_phases);
        {
            let mut r = per_section.borrow_mut();
            r.render_distance = 12;
            for mesh in warm.iter().cloned() {
                r.upload(&gpu, mesh, cam.pos);
            }
        }
        let mut batched_phases = PhaseList::new();
        let batched = TerrainPhase::install(&gpu, &atlas, &mut batched_phases);
        {
            let mut r = batched.borrow_mut();
            r.set_multi_draw_indirect(false);
            r.render_distance = 12;
            r.upload_all(&gpu, warm.clone(), cam.pos);
        }
        for _ in 0..10 {
            one_frame(&mut gpu, &mut per_section_phases, &cam);
            one_frame(&mut gpu, &mut batched_phases, &cam);
        }
        let mut a = Samples::new();
        let mut b = Samples::new();
        for _ in 0..frames {
            let (c, _) = one_frame(&mut gpu, &mut per_section_phases, &cam);
            a.push(c);
            let (c, _) = one_frame(&mut gpu, &mut batched_phases, &cam);
            b.push(c);
        }
        a.report("per-section buffers (before)");
        b.report("batched buffers");
        let (sa, sb) = (per_section.borrow().stats(), batched.borrow().stats());
        println!(
            "  {:<38} {} vs {} draws, {} vs {} vertex-buffer binds, {} vs {} gpu buffers",
            "counts (deterministic)",
            sa.draw_calls,
            sb.draw_calls,
            sa.vertex_buffer_binds,
            sb.vertex_buffer_binds,
            per_section.borrow().gpu_buffer_count(),
            batched.borrow().gpu_buffer_count(),
        );
        let diff = a.mean() - b.mean();
        // Standard error of the difference of two means, so the number can be read as
        // "distinguishable from zero" or not rather than taken on faith.
        let n = a.ms.len().min(b.ms.len()).max(1) as f64;
        let se = (a.sd() * a.sd() / n + b.sd() * b.sd() / n).sqrt();
        println!(
            "  {:<38} {:>8.4} ms/frame  (se {:.4}, {:.1} sigma, {:+.1}%)",
            "difference",
            diff,
            se,
            diff / se.max(f64::MIN_POSITIVE),
            -100.0 * diff / a.mean().max(f64::MIN_POSITIVE)
        );
    }
    println!();

    // ---- 4d. native indirect draw loop A/B, interleaved ------------------------------------
    // Compare the native feature path against the same batched geometry in one process. The
    // upload comparison above remains direct-vs-batched; this block reports actual API calls and
    // logical section items separately.
    println!("4d. native indirect draw loop A/B, interleaved");
    {
        let mut direct_phases = PhaseList::new();
        let direct = TerrainPhase::install(&gpu, &atlas, &mut direct_phases);
        {
            let mut r = direct.borrow_mut();
            r.set_multi_draw_indirect(false);
            r.render_distance = 12;
            r.upload_all(&gpu, warm.clone(), cam.pos);
        }
        let mut indirect_phases = PhaseList::new();
        let indirect = TerrainPhase::install(&gpu, &atlas, &mut indirect_phases);
        {
            let mut r = indirect.borrow_mut();
            r.set_multi_draw_indirect(true);
            r.render_distance = 12;
            r.upload_all(&gpu, warm.clone(), cam.pos);
        }
        if indirect.borrow().multi_draw_indirect_enabled() {
            for _ in 0..10 {
                one_frame(&mut gpu, &mut direct_phases, &cam);
                one_frame(&mut gpu, &mut indirect_phases, &cam);
            }
            let mut a = Samples::new();
            let mut b = Samples::new();
            for _ in 0..frames {
                let (cpu, _) = one_frame(&mut gpu, &mut direct_phases, &cam);
                a.push(cpu);
                let (cpu, _) = one_frame(&mut gpu, &mut indirect_phases, &cam);
                b.push(cpu);
            }
            a.report("batched direct draw API");
            b.report("batched native indirect API");
            let (sa, sb) = (direct.borrow().stats(), indirect.borrow().stats());
            println!(
                "  {:<38} {} -> {} API draws; {} logical items; {} -> {} binds",
                "counts (last frame)",
                sa.draw_calls,
                sb.draw_calls,
                sa.draw_items,
                sa.vertex_buffer_binds,
                sb.vertex_buffer_binds,
            );
            let n = a.ms.len().min(b.ms.len()).max(1) as f64;
            let se = (a.sd() * a.sd() / n + b.sd() * b.sd() / n).sqrt();
            println!(
                "  {:<38} {:>8.4} ms/frame (se {:.4}, {:+.1}%)",
                "indirect minus direct",
                b.mean() - a.mean(),
                se,
                100.0 * (b.mean() - a.mean()) / a.mean().max(f64::MIN_POSITIVE),
            );
        } else {
            println!("  native multi-draw unavailable; direct fallback selected");
        }
    }
    println!();

    // ---- 5. marginal draw-call cost ----------------------------------------------------------
    // The draw-call reduction task turns on this number: if a draw call is cheap, moving from
    // 4N draws to N is not worth a vertex-format change. Measured by drawing the same geometry
    // at a sweep of visible-section counts (the render distance selects how many) and regressing
    // the CPU record time on the draw count, rather than by asserting a figure.
    println!("5. marginal cost of a terrain draw call");
    {
        // The draw count is varied by varying how many sections are *resident*, not by the
        // render distance: at this scene size the frustum selects the same sections from render
        // distance 2 upwards, so a render-distance sweep holds the draw count constant and the
        // regression would be fitted to noise.
        let mut points: Vec<(f64, f64)> = Vec::new();
        let steps: Vec<usize> = [1usize, 2, 4, 8, 16, 32, 64, warm.len()]
            .into_iter()
            .filter(|k| *k <= warm.len())
            .collect();
        for k in steps {
            let mut phases = PhaseList::new();
            let shared = TerrainPhase::install(&gpu, &atlas, &mut phases);
            {
                let mut r = shared.borrow_mut();
                r.set_multi_draw_indirect(false);
                r.render_distance = 12;
                r.upload_all(&gpu, warm[..k].to_vec(), cam.pos);
            }
            for _ in 0..10 {
                one_frame(&mut gpu, &mut phases, &cam);
            }
            let mut s = Samples::new();
            for _ in 0..120 {
                let (c, _) = one_frame(&mut gpu, &mut phases, &cam);
                s.push(c);
            }
            let stats = shared.borrow().stats();
            println!(
                "  {:<20} {:>4} draws, {:>4} binds  {:>8.4} ms cpu  sd {:>6.4}",
                format!("{k} sections resident"),
                stats.draw_calls,
                stats.vertex_buffer_binds,
                s.mean(),
                s.sd()
            );
            points.push((stats.draw_calls as f64, s.mean()));
        }
        // Least-squares slope of cpu-time against draw count.
        let n = points.len() as f64;
        let mx = points.iter().map(|p| p.0).sum::<f64>() / n;
        let my = points.iter().map(|p| p.1).sum::<f64>() / n;
        let num: f64 = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
        let den: f64 = points.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
        let slope_ms = if den > 0.0 { num / den } else { 0.0 };
        println!(
            "  {:<38} {:>8.3} us per draw call (least squares over {} points)",
            "marginal cost",
            slope_ms * 1000.0,
            points.len()
        );
        if let Some((draws, _)) = points.last() {
            println!(
                "  {:<38} {:>8.3} ms for all {draws} draws",
                "so the whole draw loop costs",
                slope_ms * draws
            );
            println!(
                "  {:<38} {:>8.3} ms  <- what a *perfect* draw-call merge could save",
                "and one draw per layer would cost",
                slope_ms * 4.0
            );
        }
    }
}
