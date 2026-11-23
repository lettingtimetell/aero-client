//! Differential test of every block collision shape against the **real** 1.8.9 game.
//!
//! `javaref/BoxRefGen.java` runs the recompiled vanilla classes headless and records, for many
//! synthetic *client* worlds, exactly what the client's collision path returns:
//!
//! * per cell, `Block.addCollisionBoxesToList(world, pos, state, mask, list, mover)` — the body of
//!   `World.getCollidingBoundingBoxes`' inner loop (`World.java:1303`);
//! * per sweep mask, the whole `World.getCollidingBoundingBoxes(mover, mask)` list in order, and
//!   the mover's `outsideBorder` latch afterwards.
//!
//! This test rebuilds the same worlds and compares bit-for-bit through the functions the live
//! client sweep calls (`collision_boxes_at` is the per-block body of the sweep scan, and
//! `collision_boxes_in_with_border_and_entities` is the scan the predictor runs every tick).
//!
//! Scenes: every packed `(id, meta)` 0..4095 isolated, surrounded by stone and surrounded by
//! itself, near the origin and near `|x| = 2^23` and `30,000,000`; stairs against every
//! combination of four stair/air neighbours; doors against every legal/illegal/non-door half
//! above and below; fences, walls, panes and bars against every 4-neighbour combination of 12
//! block types; random decorative grids with sweep masks; a world border over loaded and
//! *unloaded* columns; and step-up neighbourhoods.
//!
//! Two classes read the block singleton's latched bounds in vanilla, so their real shape
//! depends on what the chunk mesher or the mouse-over ray last touched: `BlockAnvil` and
//! `BlockChest` (the generator proves that set). The fixture records mc-physics' deterministic
//! model for them (an anvil collides with its own facing's bounds; a chest with the single-chest
//! bounds).
//!
//! Hashed scenes store a count and an FNV-1a over the per-query result bytes. On a mismatch,
//! set `MC_BOXES_DUMP=<file>` to write this side's per-query hashes and boxes, then diff the
//! `h=` column against the generator's dump.
//!
//! Regenerate with `javaref/generate-l3.ps1` (it also writes the generator's own per-query dump).

use std::fmt::Write as _;
use std::path::PathBuf;

use mc_data::BlockState;
use mc_math::{Aabb, BlockPos, ChunkPos};
use mc_physics::collision::{
    collision_boxes_at, collision_boxes_in_with_border_and_entities, CollisionBorder,
};
use mc_world::World;

struct Reader<'a> {
    d: &'a [u8],
    o: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> &'a [u8] {
        let v = &self.d[self.o..self.o + n];
        self.o += n;
        v
    }
    fn u8(&mut self) -> u8 {
        self.bytes(1)[0]
    }
    fn u16(&mut self) -> u16 {
        u16::from_be_bytes(self.bytes(2).try_into().unwrap())
    }
    fn i32(&mut self) -> i32 {
        i32::from_be_bytes(self.bytes(4).try_into().unwrap())
    }
    fn u32(&mut self) -> u32 {
        self.i32() as u32
    }
    fn u64(&mut self) -> u64 {
        u64::from_be_bytes(self.bytes(8).try_into().unwrap())
    }
    fn f64(&mut self) -> f64 {
        f64::from_bits(self.u64())
    }
    fn string(&mut self) -> String {
        let n = self.u16() as usize;
        String::from_utf8(self.bytes(n).to_vec()).expect("utf-8")
    }
}

enum WorldSpec {
    Placements(Vec<(BlockPos, u16)>),
    AllStates {
        surround: u8,
        origin: BlockPos,
    },
    Stairs(u16),
    Doors {
        door: u16,
        wool: u16,
        illegal: u16,
    },
    Connect {
        centres: Vec<u16>,
        ks: Vec<u16>,
    },
    Grid {
        origin: BlockPos,
        size: (i32, i32, i32),
        cells: Vec<u16>,
    },
}

#[derive(Clone, Copy)]
enum Query {
    Cell(BlockPos),
    Sweep {
        mask: Aabb,
        entity_x: f64,
        entity_z: f64,
        outside_before: bool,
    },
}

enum Expect {
    /// Per-query result bytes, exactly as the generator wrote them.
    Full(Vec<Vec<u8>>),
    Hash {
        count: u32,
        fnv: u64,
    },
}

struct Scene {
    name: String,
    world: WorldSpec,
    loaded_rect: Option<(i32, i32, i32, i32)>,
    load_placed: bool,
    border: Option<CollisionBorder>,
    query_kind: u8,
    explicit: Vec<Query>,
    expect: Expect,
}

/// `java.util.Random`-free FNV-1a 64, the generator's hash.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn feed(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }
}

fn fixture() -> Vec<Scene> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("block_boxes.bin");
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut r = Reader { d: &data, o: 0 };
    assert_eq!(r.bytes(8), b"MCBOX001", "bad magic");
    let n = r.u32();
    let mut scenes = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let name = r.string();
        let world = match r.u8() {
            0 => {
                let k = r.u32();
                WorldSpec::Placements(
                    (0..k)
                        .map(|_| (BlockPos::new(r.i32(), r.i32(), r.i32()), r.u16()))
                        .collect(),
                )
            }
            1 => WorldSpec::AllStates {
                surround: r.u8(),
                origin: BlockPos::new(r.i32(), r.i32(), r.i32()),
            },
            2 => WorldSpec::Stairs(r.u16()),
            3 => WorldSpec::Doors {
                door: r.u16(),
                wool: r.u16(),
                illegal: r.u16(),
            },
            4 => {
                let nc = r.u16();
                let centres = (0..nc).map(|_| r.u16()).collect();
                let nk = r.u16();
                let ks = (0..nk).map(|_| r.u16()).collect();
                WorldSpec::Connect { centres, ks }
            }
            5 => {
                let origin = BlockPos::new(r.i32(), r.i32(), r.i32());
                let size = (r.i32(), r.i32(), r.i32());
                let cells = (0..size.0 * size.1 * size.2).map(|_| r.u16()).collect();
                WorldSpec::Grid {
                    origin,
                    size,
                    cells,
                }
            }
            k => panic!("{name}: unknown world kind {k}"),
        };
        let rect = (r.i32(), r.i32(), r.i32(), r.i32());
        let loaded_rect = (rect.0 <= rect.2 && rect.1 <= rect.3).then_some(rect);
        let load_placed = r.u8() != 0;
        let border = (r.u8() != 0).then(|| {
            let (cx, cz, diameter) = (r.f64(), r.f64(), r.f64());
            CollisionBorder::new(cx, cz, diameter, r.i32())
        });
        let query_kind = r.u8();
        let mut explicit = Vec::new();
        if query_kind != 0 {
            for _ in 0..r.u32() {
                explicit.push(match r.u8() {
                    0 => Query::Cell(BlockPos::new(r.i32(), r.i32(), r.i32())),
                    _ => {
                        let (x0, y0, z0, x1, y1, z1) =
                            (r.f64(), r.f64(), r.f64(), r.f64(), r.f64(), r.f64());
                        Query::Sweep {
                            mask: Aabb::from_bounds(x0, y0, z0, x1, y1, z1),
                            entity_x: r.f64(),
                            entity_z: r.f64(),
                            outside_before: r.u8() != 0,
                        }
                    }
                });
            }
        }
        let expect = match r.u8() {
            1 => Expect::Hash {
                count: r.u32(),
                fnv: r.u64(),
            },
            _ => {
                let mut scene = Scene {
                    name: name.clone(),
                    world,
                    loaded_rect,
                    load_placed,
                    border,
                    query_kind,
                    explicit,
                    expect: Expect::Full(Vec::new()),
                };
                let queries = implicit_count(&scene.world) * usize::from(scene.query_kind != 1)
                    + scene.explicit.len();
                let mut per_query = Vec::with_capacity(queries);
                for q in 0..queries {
                    let start = r.o;
                    let boxes = r.u32() as usize;
                    r.bytes(boxes * 48);
                    let is_sweep = q >= queries - scene.explicit.len()
                        && matches!(
                            scene.explicit[q - (queries - scene.explicit.len())],
                            Query::Sweep { .. }
                        );
                    if is_sweep && scene.border.is_some() {
                        r.u8();
                    }
                    per_query.push(r.d[start..r.o].to_vec());
                }
                scene.expect = Expect::Full(per_query);
                scenes.push(scene);
                continue;
            }
        };
        scenes.push(Scene {
            name,
            world,
            loaded_rect,
            load_placed,
            border,
            query_kind,
            explicit,
            expect,
        });
    }
    assert_eq!(r.o, data.len(), "trailing bytes in the fixture");
    scenes
}

fn implicit_count(world: &WorldSpec) -> usize {
    match world {
        WorldSpec::Placements(p) => p.len(),
        WorldSpec::AllStates { .. } => 4096,
        WorldSpec::Stairs(_) => STAIRS_CONFIGS,
        WorldSpec::Doors { .. } => DOOR_CONFIGS,
        WorldSpec::Connect { centres, ks } => centres.len() * ks.len().pow(4),
        WorldSpec::Grid { cells, .. } => cells.len(),
    }
}

const STAIRS_CONFIGS: usize = 8 * 9 * 9 * 9 * 9;
const DOOR_CONFIGS: usize = 16 * 18 * 18;
const BATCH: usize = 4096;

fn all_states_pos(packed: u16, origin: BlockPos) -> BlockPos {
    let id = i32::from(packed >> 4);
    let meta = i32::from(packed & 15);
    BlockPos::new(
        origin.x + (id & 15) * 3,
        origin.y + 4 + meta * 3,
        origin.z + (id >> 4) * 3,
    )
}

fn batch_pos(j: usize, y: i32) -> BlockPos {
    BlockPos::new((j % 64) as i32 * 3, y, (j / 64) as i32 * 3)
}

fn door_pos(i: usize) -> BlockPos {
    BlockPos::new((i % 64) as i32 * 2, 65, (i / 64) as i32 * 2)
}

fn stairs_config(i: usize, stairs: u16) -> [u16; 5] {
    let mut r = i / 8;
    let mut out = [0u16; 5];
    out[0] = stairs << 4 | (i % 8) as u16;
    for slot in out.iter_mut().skip(1) {
        let v = r % 9;
        r /= 9;
        *slot = if v == 0 {
            0
        } else {
            stairs << 4 | (v - 1) as u16
        };
    }
    out
}

fn door_neighbour(v: usize, wool: u16, illegal: u16) -> u16 {
    match v {
        0 => 0,
        17 => illegal,
        _ => wool << 4 | (v - 1) as u16,
    }
}

fn connect_config(i: usize, centres: &[u16], ks: &[u16]) -> [u16; 5] {
    let mut r = i / centres.len();
    let mut out = [0u16; 5];
    out[0] = centres[i % centres.len()];
    for slot in out.iter_mut().skip(1) {
        *slot = ks[r % ks.len()];
        r /= ks.len();
    }
    out
}

/// Builds one world from raw packed placements, the way a chunk section stores them.
fn build(placements: &[(BlockPos, u16)], scene: &Scene) -> World {
    let mut w = World::new();
    if let Some((cx0, cz0, cx1, cz1)) = scene.loaded_rect {
        for cx in cx0..=cx1 {
            for cz in cz0..=cz1 {
                mc_physics::scene::ensure_chunk(&mut w, ChunkPos::new(cx, cz));
            }
        }
    }
    assert!(
        scene.load_placed || placements.is_empty(),
        "{}: unloaded placements are not representable",
        scene.name
    );
    mc_physics::scene::fill_static(
        &mut w,
        placements
            .iter()
            .map(|&(p, packed)| (p, p, BlockState::from_raw(packed))),
    );
    w
}

fn cell_mask(p: BlockPos) -> Aabb {
    Aabb::from_bounds(
        f64::from(p.x - 2),
        f64::from(p.y - 2),
        f64::from(p.z - 2),
        f64::from(p.x + 3),
        f64::from(p.y + 4),
        f64::from(p.z + 3),
    )
}

/// The live client always queries through a border; without an S44 it is vanilla's default.
const DEFAULT_BORDER: CollisionBorder = CollisionBorder::new(0.0, 0.0, 6.0e7, 29_999_984);

fn run_query(w: &World, scene: &Scene, q: Query) -> Vec<u8> {
    let (boxes, latch) = match q {
        Query::Cell(p) => (collision_boxes_at(w, p, &cell_mask(p)), None),
        Query::Sweep {
            mask,
            entity_x,
            entity_z,
            outside_before,
        } => {
            let mut outside = outside_before;
            let mut out = Vec::new();
            collision_boxes_in_with_border_and_entities(
                w,
                &mask,
                scene.border.unwrap_or(DEFAULT_BORDER),
                entity_x,
                entity_z,
                &mut outside,
                &[],
                &mut out,
            );
            (out, scene.border.is_some().then_some(outside))
        }
    };
    let mut bytes = Vec::with_capacity(4 + boxes.len() * 48 + 1);
    bytes.extend_from_slice(&(boxes.len() as u32).to_be_bytes());
    for b in &boxes {
        for v in [b.min.x, b.min.y, b.min.z, b.max.x, b.max.y, b.max.z] {
            bytes.extend_from_slice(&v.to_bits().to_be_bytes());
        }
    }
    if let Some(outside) = latch {
        bytes.push(u8::from(outside));
    }
    bytes
}

/// Every query's result bytes, in fixture order, rebuilding batched worlds as Java does.
fn evaluate(scene: &Scene) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let implicit = scene.query_kind != 1;
    match &scene.world {
        WorldSpec::Placements(placements) => {
            let w = build(placements, scene);
            if implicit {
                for &(p, _) in placements {
                    out.push(run_query(&w, scene, Query::Cell(p)));
                }
            }
            for &q in &scene.explicit {
                out.push(run_query(&w, scene, q));
            }
        }
        WorldSpec::AllStates { surround, origin } => {
            let mut placements = Vec::new();
            for packed in 0..4096u16 {
                let p = all_states_pos(packed, *origin);
                placements.push((p, packed));
                if *surround != 0 {
                    let around = if *surround == 1 { 1 << 4 } else { packed };
                    for n in [p.down(), p.up(), p.north(), p.south(), p.west(), p.east()] {
                        placements.push((n, around));
                    }
                }
            }
            let w = build(&placements, scene);
            for packed in 0..4096u16 {
                out.push(run_query(
                    &w,
                    scene,
                    Query::Cell(all_states_pos(packed, *origin)),
                ));
            }
        }
        WorldSpec::Stairs(stairs) => {
            for base in (0..STAIRS_CONFIGS).step_by(BATCH) {
                let end = STAIRS_CONFIGS.min(base + BATCH);
                let mut placements = Vec::new();
                for i in base..end {
                    let c = stairs_config(i, *stairs);
                    let p = batch_pos(i - base, 64);
                    placements.extend([
                        (p, c[0]),
                        (p.north(), c[1]),
                        (p.south(), c[2]),
                        (p.west(), c[3]),
                        (p.east(), c[4]),
                    ]);
                }
                let w = build(&placements, scene);
                for i in base..end {
                    out.push(run_query(&w, scene, Query::Cell(batch_pos(i - base, 64))));
                }
            }
        }
        WorldSpec::Doors {
            door,
            wool,
            illegal,
        } => {
            let mut placements = Vec::new();
            for i in 0..DOOR_CONFIGS {
                let r = i / 16;
                let p = door_pos(i);
                placements.extend([
                    (p, door << 4 | (i % 16) as u16),
                    (p.down(), door_neighbour(r % 18, *wool, *illegal)),
                    (p.up(), door_neighbour(r / 18, *wool, *illegal)),
                ]);
            }
            let w = build(&placements, scene);
            for i in 0..DOOR_CONFIGS {
                out.push(run_query(&w, scene, Query::Cell(door_pos(i))));
            }
        }
        WorldSpec::Connect { centres, ks } => {
            let total = centres.len() * ks.len().pow(4);
            for base in (0..total).step_by(BATCH) {
                let end = total.min(base + BATCH);
                let mut placements = Vec::new();
                for i in base..end {
                    let c = connect_config(i, centres, ks);
                    let p = batch_pos(i - base, 64);
                    placements.extend([
                        (p, c[0]),
                        (p.north(), c[1]),
                        (p.south(), c[2]),
                        (p.west(), c[3]),
                        (p.east(), c[4]),
                    ]);
                }
                let w = build(&placements, scene);
                for i in base..end {
                    out.push(run_query(&w, scene, Query::Cell(batch_pos(i - base, 64))));
                }
            }
        }
        WorldSpec::Grid {
            origin,
            size,
            cells,
        } => {
            let mut placements = Vec::with_capacity(cells.len());
            let mut idx = 0;
            for x in 0..size.0 {
                for y in 0..size.1 {
                    for z in 0..size.2 {
                        placements.push((
                            BlockPos::new(origin.x + x, origin.y + y, origin.z + z),
                            cells[idx],
                        ));
                        idx += 1;
                    }
                }
            }
            let w = build(&placements, scene);
            if implicit {
                for &(p, _) in &placements {
                    out.push(run_query(&w, scene, Query::Cell(p)));
                }
            }
            for &q in &scene.explicit {
                out.push(run_query(&w, scene, q));
            }
        }
    }
    out
}

fn describe(bytes: &[u8]) -> String {
    let n = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut s = format!("{n}");
    for b in 0..n {
        let v: Vec<f64> = (0..6)
            .map(|k| {
                let o = 4 + b * 48 + k * 8;
                f64::from_bits(u64::from_be_bytes(bytes[o..o + 8].try_into().unwrap()))
            })
            .collect();
        let _ = write!(
            s,
            " [{},{},{} .. {},{},{}]",
            v[0], v[1], v[2], v[3], v[4], v[5]
        );
    }
    if bytes.len() > 4 + n * 48 {
        let _ = write!(s, " outsideAfter={}", bytes[4 + n * 48]);
    }
    s
}

fn hash_one(bytes: &[u8]) -> u64 {
    let mut h = Fnv::new();
    h.feed(bytes);
    h.0
}

#[test]
fn every_block_collision_shape_matches_java() {
    let dump_path = std::env::var_os("MC_BOXES_DUMP");
    let mut dump = String::new();
    let mut failures = Vec::new();

    for scene in fixture() {
        let results = evaluate(&scene);
        if dump_path.is_some() {
            for (i, r) in results.iter().enumerate() {
                let _ = writeln!(
                    dump,
                    "{} #{i} h={:016x} -> {}",
                    scene.name,
                    hash_one(r),
                    describe(r)
                );
            }
        }
        match &scene.expect {
            Expect::Full(expected) => {
                assert_eq!(results.len(), expected.len(), "{}: query count", scene.name);
                let mut bad = 0;
                for (i, (got, want)) in results.iter().zip(expected).enumerate() {
                    if got != want {
                        bad += 1;
                        if bad <= 12 {
                            failures.push(format!(
                                "{} #{i}\n    java: {}\n    rust: {}",
                                scene.name,
                                describe(want),
                                describe(got)
                            ));
                        }
                    }
                }
                if bad > 12 {
                    failures.push(format!("{}: {bad} mismatching queries in all", scene.name));
                }
            }
            Expect::Hash { count, fnv } => {
                let mut h = Fnv::new();
                for r in &results {
                    h.feed(r);
                }
                if results.len() != *count as usize || h.0 != *fnv {
                    failures.push(format!(
                        "{}: {} queries fnv {:016x}, java {count} queries fnv {fnv:016x} \
                         (MC_BOXES_DUMP=<file> to locate)",
                        scene.name,
                        results.len(),
                        h.0
                    ));
                }
            }
        }
    }

    if let Some(path) = dump_path {
        std::fs::write(&path, dump).expect("write MC_BOXES_DUMP");
    }
    assert!(
        failures.is_empty(),
        "{} collision mismatches against 1.8.9:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The fixture must actually exercise what the header promises, or a bug in the generator
/// could pass vacuously.
#[test]
fn the_box_fixture_covers_what_it_claims() {
    let scenes = fixture();
    let names: Vec<&str> = scenes.iter().map(|s| s.name.as_str()).collect();
    for want in [
        "isolated_all_states",
        "stone_surround_all_states",
        "self_surround_all_states",
        "far_all_states",
        "edge_all_states",
        "stairs_exhaustive",
        "doors_exhaustive_oak",
        "doors_exhaustive_iron",
        "connect_exhaustive",
        "border_loaded",
        "border_unloaded",
        "step_contexts",
    ] {
        assert!(names.contains(&want), "missing scene {want}");
    }
    assert!(
        names
            .iter()
            .filter(|n| n.starts_with("grid_sweep_"))
            .count()
            >= 20
    );

    // The isolated scene has real shapes in it: count the states with a non-empty box list.
    let isolated = scenes
        .iter()
        .find(|s| s.name == "isolated_all_states")
        .unwrap();
    let Expect::Full(results) = &isolated.expect else {
        panic!("isolated scene must be stored in full");
    };
    let with_boxes = results
        .iter()
        .filter(|r| u32::from_be_bytes(r[0..4].try_into().unwrap()) > 0)
        .count();
    // 910 of the 4096 packed values collide in 1.8.9; the rest are unregistered, illegal
    // metadata (air on a client), or blocks without a box.
    assert!(with_boxes > 900, "only {with_boxes} colliding states");

    // The unloaded-border scene places no blocks, yet Java collides: border stone over columns
    // the server never sent.
    let unloaded = scenes.iter().find(|s| s.name == "border_unloaded").unwrap();
    let Expect::Full(results) = &unloaded.expect else {
        panic!("border scenes must be stored in full");
    };
    assert!(results
        .iter()
        .any(|r| u32::from_be_bytes(r[0..4].try_into().unwrap()) > 0));
}
