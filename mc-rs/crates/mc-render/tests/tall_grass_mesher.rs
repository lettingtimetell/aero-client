//! The Tall Grass mod in the production mesher.
//!
//! * **On (vanilla):** the mesher produces exactly the bytes it produced before this change.
//!   The golden hashes are `ctm_off_parity.rs`'s, printed by the mesher *before* connected
//!   textures or this mod existed; the scatter worlds hold every renderable state, grass, ferns
//!   and both double-plant halves included.
//! * **Off:** the off mesh is the on mesh with only the hidden plants' quads taken out; every
//!   other quad keeps its exact bytes, lighting included.
//!
//! Off is deliberately *not* "as if the grass were air". Grass is a `useNeighborBrightness`
//! block (`Block.registerBlocks` sets it for every block with light opacity 0), so vanilla's
//! `ChunkCache.getLightForExt` samples a grass cell as the brightest of its six neighbours, and
//! a stone face next to a fern is lit from that. Replacing the fern with air would relight the
//! stone; hiding it only stops drawing the fern, so the rest of the frame stays vanilla.
//!
//! Installs the checked-in bake cache (`mc-client/cache/baked-assets.bin`) in its own test
//! binary and skips, loudly, when it is missing or from another format version.

use std::sync::OnceLock;

use mc_data::{BlockState, RenderLayer, RenderType};
use mc_math::{BlockPos, ChunkSectionPos};
use mc_render::terrain::mesher::mesh_section_with_alternatives;
use mc_render::terrain::{fixture, is_hidden_tall_grass, ChunkMesh, MeshSettings, VERTEX_SIZE};
use mc_world::World;

/// `ctm_off_parity.rs`: earlier mesher output for the scatter worlds (re-baselined there by
/// H7, 2026-11-11, for the Java-exact `FaceBakery` bake; see the comment there).
const GOLDEN_SCATTER: u64 = 0xf043_9651_d4f9_422d;
/// `ctm_off_parity.rs`: earlier mesher output for the glass world.
const GOLDEN_GLASS: u64 = 0x1464_8c37_a263_6a65;
/// `ctm_off_parity.rs`: quads in both.
const GOLDEN_QUADS: usize = 892_836;

const SECTION: ChunkSectionPos = ChunkSectionPos::new(0, 4, 0);

fn install_cached_bake() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../mc-client/cache/baked-assets.bin");
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("SKIP: {} not found", path.display());
            return false;
        };
        let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let u64_at = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
        if bytes.len() < 16 || &bytes[..8] != b"MCRSBAKE" {
            eprintln!("SKIP: {} is not a bake cache", path.display());
            return false;
        }
        if u32_at(8) != mc_assets::cache::FORMAT_VERSION {
            eprintln!("SKIP: {} is from another cache format", path.display());
            return false;
        }
        let payload_at = 16 + u32_at(12) as usize;
        let payload_len = u64_at(payload_at) as usize;
        let payload = &bytes[payload_at + 16..payload_at + 16 + payload_len];
        let snapshot = mc_assets::cache::BakedSnapshot::decode(payload).expect("decode");
        mc_assets::model::install(snapshot.into_assets());
        true
    })
}

struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

fn renderable_states() -> Vec<BlockState> {
    mc_data::legal_states()
        .filter(|s| {
            matches!(
                s.props().render_type,
                RenderType::Model | RenderType::Liquid
            )
        })
        .collect()
}

/// `ctm_off_parity.rs`'s scatter world, unchanged (the golden hash depends on it).
fn scatter_world(states: &[BlockState], density_percent: u64, seed: u64) -> World {
    let mut w = fixture::blank_world(1);
    let mut rng = Lcg(seed);
    for x in -1..17 {
        for y in 63..81 {
            for z in -1..17 {
                if rng.next_u64() % 100 < density_percent {
                    let state = states[(rng.next_u64() as usize) % states.len()];
                    w.set_block(BlockPos::new(x, y, z), state);
                }
            }
        }
    }
    w
}

/// `ctm_off_parity.rs`'s glass world, unchanged.
fn glass_world() -> World {
    let mut w = fixture::blank_world(1);
    let glass = fixture::state_of("minecraft:glass");
    let pane = fixture::state_of("minecraft:glass_pane");
    let stained = BlockState::new(mc_data::block_by_name("stained_glass").unwrap(), 14);
    let stained_pane = BlockState::new(mc_data::block_by_name("stained_glass_pane").unwrap(), 3);
    let stone = fixture::state_of("minecraft:stone");
    let fill = |w: &mut World, a: (i32, i32, i32), b: (i32, i32, i32), s: BlockState| {
        fixture::fill(
            w,
            BlockPos::new(a.0, a.1, a.2),
            BlockPos::new(b.0, b.1, b.2),
            s,
        );
    };
    fill(&mut w, (0, 64, 0), (15, 64, 15), stone);
    fill(&mut w, (1, 65, 2), (5, 69, 2), glass);
    fill(&mut w, (7, 65, 2), (9, 67, 2), stained);
    fill(&mut w, (1, 65, 6), (1, 68, 9), pane);
    fill(&mut w, (4, 65, 6), (7, 67, 6), stained_pane);
    fill(&mut w, (10, 65, 8), (10, 67, 8), glass);
    fill(&mut w, (10, 65, 9), (10, 67, 11), pane);
    w.set_block(BlockPos::new(12, 65, 12), glass);
    w.set_block(BlockPos::new(13, 66, 12), glass);
    w
}

fn fnv(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= u64::from(b);
        *hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
}

fn mesh_hash(mesh: &ChunkMesh, hash: &mut u64) {
    for (_, buffer) in mesh.layers() {
        fnv(hash, &(buffer.as_bytes().len() as u64).to_le_bytes());
        fnv(hash, buffer.as_bytes());
    }
}

/// The four graphics combinations, each with the given Tall Grass switch.
fn all_settings(hide_tall_grass: bool) -> [MeshSettings; 4] {
    [(true, true), (true, false), (false, true), (false, false)].map(|(fancy, ao)| MeshSettings {
        fancy_graphics: fancy,
        ambient_occlusion: ao,
        hide_tall_grass,
        ..MeshSettings::default()
    })
}

/// The section's hidden-grass cells, by the same predicate the mesher uses.
fn hidden_cells(world: &World) -> Vec<BlockPos> {
    let origin = SECTION.origin();
    let mut out = Vec::new();
    for y in 0..16 {
        for z in 0..16 {
            for x in 0..16 {
                let pos = origin.add(x, y, z);
                if is_hidden_tall_grass(world, world.block_at(pos), pos) {
                    out.push(pos);
                }
            }
        }
    }
    out
}

/// One quad's bytes.
fn quads(bytes: &[u8]) -> impl Iterator<Item = &[u8]> {
    bytes.chunks_exact(4 * VERTEX_SIZE)
}

/// The world cell a plant quad belongs to: the floor of its centroid. Every plant model here is
/// a cross whose centroid sits mid-cell, and the random offsets move it by at most 0.25.
fn cell_of(mesh: &ChunkMesh, layer: RenderLayer, index: usize) -> BlockPos {
    let buffer = mesh.layers().find(|(l, _)| *l == layer).unwrap().1;
    let mut sum = [0.0f32; 3];
    for v in buffer.quad(index) {
        for (axis, value) in v.position().into_iter().enumerate() {
            sum[axis] += value / 4.0;
        }
    }
    let origin = SECTION.origin();
    origin.add(
        sum[0].floor() as i32,
        sum[1].floor() as i32,
        sum[2].floor() as i32,
    )
}

/// The quads `visible` has and `hidden` lacks, per layer, as `(layer, quad index in visible)`.
/// Panics unless `hidden` is `visible` with some quads taken out and nothing added or changed.
fn dropped_quads(
    visible: &ChunkMesh,
    hidden: &ChunkMesh,
    label: &str,
) -> Vec<(RenderLayer, usize)> {
    let mut dropped = Vec::new();
    for ((layer, on), (_, off)) in visible.layers().zip(hidden.layers()) {
        let mut on_quads = quads(on.as_bytes()).enumerate();
        for (n, kept) in quads(off.as_bytes()).enumerate() {
            loop {
                let (index, candidate) = on_quads.next().unwrap_or_else(|| {
                    panic!("{label}: {layer:?} quad {n} of the off mesh is not in the on mesh")
                });
                if candidate == kept {
                    break;
                }
                dropped.push((layer, index));
            }
        }
        dropped.extend(on_quads.map(|(index, _)| (layer, index)));
    }
    dropped
}

#[test]
fn on_is_byte_identical_to_the_mesher_before_the_mod_wave() {
    if !install_cached_bake() {
        return;
    }
    let states = renderable_states();
    let grass = states
        .iter()
        .filter(|s| matches!(s.id >> 4, 31 | 175))
        .count();
    assert!(
        grass >= 13,
        "the scatter draws from every grass state ({grass})"
    );
    let mut scatter = 0xcbf2_9ce4_8422_2325_u64;
    let mut quads = 0usize;
    for (density, seed) in [(15u64, 1u64), (50, 2), (90, 3), (100, 4)] {
        let w = scatter_world(&states, density, seed);
        for settings in all_settings(false) {
            for alternatives in [true, false] {
                let mesh = mesh_section_with_alternatives(&w, SECTION, settings, alternatives);
                quads += mesh.quad_count();
                mesh_hash(&mesh, &mut scatter);
            }
        }
    }
    let mut glass = 0xcbf2_9ce4_8422_2325_u64;
    let w = glass_world();
    for settings in all_settings(false) {
        let mesh = mesh_section_with_alternatives(&w, SECTION, settings, true);
        quads += mesh.quad_count();
        mesh_hash(&mesh, &mut glass);
    }
    assert_eq!(quads, GOLDEN_QUADS);
    assert_eq!(
        scatter, GOLDEN_SCATTER,
        "grass shown must mesh as before the mod"
    );
    assert_eq!(glass, GOLDEN_GLASS);
}

/// A meadow: grass, ferns, both double-grass halves, and every plant that must stay (flowers,
/// all six double plants, the shrub, the dead bush, saplings) on a grass floor, with a few
/// solid blocks around so neighbour culling and smooth lighting are exercised.
fn meadow() -> World {
    let mut w = fixture::blank_world(1);
    let s = |name: &str, meta: u8| BlockState::new(mc_data::block_by_name(name).unwrap(), meta);
    fixture::fill(
        &mut w,
        BlockPos::new(-1, 64, -1),
        BlockPos::new(16, 64, 16),
        s("grass", 0),
    );
    let singles = [
        s("tallgrass", 1),
        s("tallgrass", 2),
        s("tallgrass", 0),
        s("deadbush", 0),
        s("yellow_flower", 0),
        s("red_flower", 0),
        s("red_flower", 4),
        s("red_flower", 8),
        s("sapling", 0),
        s("tallgrass", 1),
    ];
    let mut n = 0usize;
    for x in 0..16 {
        for z in 0..8 {
            if (x + z) % 3 != 0 {
                w.set_block(BlockPos::new(x, 65, z), singles[n % singles.len()]);
                n += 1;
            }
        }
    }
    // Every double plant, both halves, side by side, then a second row of grass and ferns.
    for (x, variant) in (0..6u8).enumerate() {
        let x = x as i32 * 2;
        w.set_block(BlockPos::new(x, 65, 10), s("double_plant", variant));
        w.set_block(BlockPos::new(x, 66, 10), s("double_plant", 10));
        let grassy = if x % 4 == 0 { 2 } else { 3 };
        w.set_block(BlockPos::new(x + 1, 65, 12), s("double_plant", grassy));
        w.set_block(BlockPos::new(x + 1, 66, 12), s("double_plant", 10));
    }
    // An orphan upper half (a peony top), and grass hemmed in by stone and under glass.
    w.set_block(BlockPos::new(14, 66, 14), s("double_plant", 10));
    w.set_block(BlockPos::new(3, 65, 14), s("tallgrass", 1));
    w.set_block(BlockPos::new(2, 65, 14), s("stone", 0));
    w.set_block(BlockPos::new(4, 65, 14), s("stone", 0));
    w.set_block(BlockPos::new(3, 66, 14), s("glass", 0));
    w.set_block(BlockPos::new(3, 65, 15), s("tallgrass", 2));
    w.set_block(BlockPos::new(3, 66, 15), s("stone", 0));
    w
}

/// The quads of `mesh` whose centroid lies strictly inside a hidden cell (not on a face of it,
/// where a neighbour's full face would sit). A plant cross's centroid is mid-cell.
fn quads_inside(mesh: &ChunkMesh, cells: &[BlockPos]) -> Vec<(RenderLayer, usize, BlockPos)> {
    let mut out = Vec::new();
    for (layer, buffer) in mesh.layers() {
        for index in 0..buffer.quad_count() {
            let mut sum = [0.0f32; 3];
            for v in buffer.quad(index) {
                for (axis, value) in v.position().into_iter().enumerate() {
                    sum[axis] += value / 4.0;
                }
            }
            let inside = sum.iter().all(|c| {
                let f = c - c.floor();
                f > 0.01 && f < 0.99
            });
            let cell = cell_of(mesh, layer, index);
            if inside && cells.contains(&cell) {
                out.push((layer, index, cell));
            }
        }
    }
    out
}

/// Off against on for one world: off is on minus exactly the hidden plants' quads (four per
/// plant, a cross, all cutout), nothing added, nothing else changed, no hidden plant left.
fn assert_off_is_on_minus_the_grass(world: &World, alternatives: bool, label: &str) {
    let hidden = hidden_cells(world);
    for (on_settings, off_settings) in all_settings(false).into_iter().zip(all_settings(true)) {
        let label = format!("{label}, {off_settings:?}, alternatives {alternatives}");
        let on = mesh_section_with_alternatives(world, SECTION, on_settings, alternatives);
        let off = mesh_section_with_alternatives(world, SECTION, off_settings, alternatives);
        // Panics unless every off quad is an on quad, in order, byte for byte.
        let dropped = dropped_quads(&on, &off, &label);
        assert_eq!(dropped.len(), 4 * hidden.len(), "{label}");
        for (layer, index) in dropped {
            assert_eq!(layer, RenderLayer::Cutout, "{label}");
            let cell = cell_of(&on, layer, index);
            assert!(
                hidden.contains(&cell),
                "{label}: a dropped quad belongs to {cell:?}, which is not hidden grass"
            );
        }
        let left = quads_inside(&off, &hidden);
        assert!(
            left.is_empty(),
            "{label}: hidden grass still drawn: {left:?}"
        );
        assert_eq!(
            on.visibility, off.visibility,
            "{label}: grass never occludes"
        );
        assert_eq!(on.occlusion, off.occlusion, "{label}");
    }
}

#[test]
fn off_takes_out_exactly_the_grass_quads_in_a_meadow() {
    if !install_cached_bake() {
        return;
    }
    let world = meadow();
    let hidden = hidden_cells(&world);
    // About a fifth of the field's plants are grass or ferns, plus both halves of the double
    // grass and ferns, plus the two hemmed-in plants.
    assert!(
        hidden.len() >= 20,
        "the meadow has grass to hide: {}",
        hidden.len()
    );
    let on = mesh_section_with_alternatives(&world, SECTION, MeshSettings::default(), true);
    assert_eq!(
        quads_inside(&on, &hidden).len(),
        4 * hidden.len(),
        "control: with the mod on every grass plant is drawn, four quads each"
    );
    for alternatives in [true, false] {
        assert_off_is_on_minus_the_grass(&world, alternatives, "meadow");
    }
}

#[test]
fn off_on_the_scatter_worlds_only_takes_out_grass_quads() {
    if !install_cached_bake() {
        return;
    }
    let states = renderable_states();
    for (density, seed) in [(15u64, 1u64), (50, 2), (90, 3), (100, 4)] {
        let world = scatter_world(&states, density, seed);
        assert_off_is_on_minus_the_grass(&world, true, &format!("scatter density {density}"));
    }
}
