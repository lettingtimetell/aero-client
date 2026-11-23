//! Connected textures: with the mod **off** the mesher must produce
//! exactly the bytes it produced before CTM existed.
//!
//! The golden hashes below were printed by this file's first version, run against the mesher
//! *before* any CTM code was added (2026-11-07 02:41).
//! The bake installed here also carries a CTM table (the pane-edge merge, the only part that
//! needs no CTM pack), so `Off` is shown to ignore a present table, not merely a missing one.
//!
//! Like `mesher_fast_path_parity.rs`, this installs the checked-in bake cache in its own test
//! binary and skips, loudly, when the cache is missing or from another format version.

use mc_data::{BlockState, RenderType};
use mc_math::{BlockPos, ChunkSectionPos};
use mc_render::terrain::mesher::mesh_section_with_alternatives;
use mc_render::terrain::{fixture, ChunkMesh, CtmMode, MeshSettings};
use mc_world::World;
use std::sync::OnceLock;

/// Pre-CTM mesher output for [`scatter_world`]s × [`all_settings`] × block alternatives.
///
/// Re-baselined by H7 (2026-11-11) from `0xe691_5e0e_efa6_49d9` when the *bake* became
/// bit-identical to vanilla `FaceBakery` (`mc-assets/tests/face_bakery_java_parity.rs`: the
/// rescale constants and LWJGL's float rotation matrices move vertex bytes of fire, cross plants
/// and every x/y-rotated variant). The mesher did not change: [`GOLDEN_QUADS`] still holds and
/// `mesher_fast_path_parity.rs` still finds the fast and reference meshers byte-identical.
///
/// For a while (2026-11-11 to 2026-11-22) wool and stained clay vertices carried a block id and
/// height in the two spare light bytes for the Build Height mod, and this hash was
/// `0x2796_7952_169e_02a5`. The mod is gone and those bytes are zero again, as vanilla's: the
/// hash is back to the value from before the mod, `0xf043_9651_d4f9_422d`, so nothing else in
/// the mesher's output changed.
const GOLDEN_SCATTER: u64 = 0xf043_9651_d4f9_422d;
/// Pre-CTM mesher output for [`glass_world`] × [`all_settings`].
const GOLDEN_GLASS: u64 = 0x1464_8c37_a263_6a65;
/// Quads in both, as a guard that the fixtures still produce the same amount of geometry.
const GOLDEN_QUADS: usize = 892_836;

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
            return false;
        }
        if u32_at(8) != mc_assets::cache::FORMAT_VERSION {
            return false;
        }
        let payload_at = 16 + u32_at(12) as usize;
        let payload_len = u64_at(payload_at) as usize;
        let payload = &bytes[payload_at + 16..payload_at + 16 + payload_len];
        let snapshot = mc_assets::cache::BakedSnapshot::decode(payload).expect("decode");
        let mut assets = snapshot.into_assets();
        // No CTM pack: no rules, but every pane state gets its edge-merge slot.
        assets.attach_ctm(&mc_assets::ResourceManager::new());
        let table = assets.ctm().expect("attach_ctm always builds a table");
        assert_eq!(table.rule_count(), 0);
        assert!(
            table.slot_count() >= 17,
            "glass and stained panes have edge slots"
        );
        mc_assets::model::install(assets);
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

fn glass_world() -> World {
    let mut w = fixture::blank_world(1);
    let glass = fixture::state_of("minecraft:glass");
    let pane = fixture::state_of("minecraft:glass_pane");
    let stained = BlockState::new(mc_data::block_by_name("stained_glass").unwrap(), 14);
    let stained_pane = BlockState::new(mc_data::block_by_name("stained_glass_pane").unwrap(), 3);
    let stone = fixture::state_of("minecraft:stone");
    fixture::fill(
        &mut w,
        BlockPos::new(0, 64, 0),
        BlockPos::new(15, 64, 15),
        stone,
    );
    fixture::fill(
        &mut w,
        BlockPos::new(1, 65, 2),
        BlockPos::new(5, 69, 2),
        glass,
    );
    fixture::fill(
        &mut w,
        BlockPos::new(7, 65, 2),
        BlockPos::new(9, 67, 2),
        stained,
    );
    fixture::fill(
        &mut w,
        BlockPos::new(1, 65, 6),
        BlockPos::new(1, 68, 9),
        pane,
    );
    fixture::fill(
        &mut w,
        BlockPos::new(4, 65, 6),
        BlockPos::new(7, 67, 6),
        stained_pane,
    );
    fixture::fill(
        &mut w,
        BlockPos::new(10, 65, 8),
        BlockPos::new(10, 67, 8),
        glass,
    );
    fixture::fill(
        &mut w,
        BlockPos::new(10, 65, 9),
        BlockPos::new(10, 67, 11),
        pane,
    );
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

/// The four graphics combinations with the given connected-textures mode.
fn all_settings(mode: CtmMode) -> [MeshSettings; 4] {
    let mut out = [MeshSettings::default(); 4];
    for (i, (fancy, ao)) in [(true, true), (true, false), (false, true), (false, false)]
        .into_iter()
        .enumerate()
    {
        out[i].fancy_graphics = fancy;
        out[i].ambient_occlusion = ao;
        out[i].connected_textures = mode;
    }
    out
}

/// `(scatter hash, glass hash, quads)` for one mode, over exactly the pre-CTM fixtures.
fn hashes(mode: CtmMode) -> (u64, u64, usize) {
    let section = ChunkSectionPos::new(0, 4, 0);
    let states = renderable_states();
    let mut scatter = 0xcbf2_9ce4_8422_2325_u64;
    let mut quads = 0usize;
    for (density, seed) in [(15u64, 1u64), (50, 2), (90, 3), (100, 4)] {
        let w = scatter_world(&states, density, seed);
        for settings in all_settings(mode) {
            for alternatives in [true, false] {
                let mesh = mesh_section_with_alternatives(&w, section, settings, alternatives);
                quads += mesh.quad_count();
                mesh_hash(&mesh, &mut scatter);
            }
        }
    }
    let mut glass = 0xcbf2_9ce4_8422_2325_u64;
    let w = glass_world();
    for settings in all_settings(mode) {
        let mesh = mesh_section_with_alternatives(&w, section, settings, true);
        quads += mesh.quad_count();
        mesh_hash(&mesh, &mut glass);
    }
    (scatter, glass, quads)
}

#[test]
fn off_is_byte_identical_to_the_mesher_before_connected_textures() {
    if !install_cached_bake() {
        return;
    }
    assert!(
        mc_assets::model::ctm_table().is_some(),
        "a table is installed"
    );
    let (scatter, glass, quads) = hashes(CtmMode::Off);
    assert_eq!(quads, GOLDEN_QUADS);
    assert_eq!(
        scatter, GOLDEN_SCATTER,
        "scatter fixtures changed with CTM off"
    );
    assert_eq!(glass, GOLDEN_GLASS, "glass fixture changed with CTM off");
}

#[test]
fn on_without_a_ctm_pack_only_merges_stacked_pane_edges() {
    if !install_cached_bake() {
        return;
    }
    // Control: the mod's machinery really runs, and does only what the format does without a
    // CTM pack (the stacked-pane edge merge): the glass fixture's pane stacks lose the edge quads
    // between panes, so it has fewer quads; nothing is remapped (no rules).
    for mode in [CtmMode::Fancy, CtmMode::Fast] {
        let (_, glass, quads) = hashes(mode);
        assert_ne!(
            glass, GOLDEN_GLASS,
            "{mode:?}: stacked pane edges must merge"
        );
        assert!(
            quads < GOLDEN_QUADS,
            "{mode:?}: merged edges are not emitted"
        );
    }
}
