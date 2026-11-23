//! Round-2 #43 (MESH-1/5/6/9): the mesher's per-block fast paths must be byte-identical to the
//! paths they replace.
//!
//! The fast paths are the dense `models[state.id]` lookup for blocks without a
//! `getActualState` override, the once-per-block tint, the class table behind face culling, the
//! live-block bounds shortcut and the cached fluid UVs. `mesh_section_reference` runs the same
//! walk with all of them switched off, so this compares the two on sections built from every
//! legal block state, under all four graphics combinations and both block-alternative settings.
//!
//! It needs real baked models, so it lives in its own test binary (the registry is
//! process-wide and first-install-wins) and installs the snapshot's checked-in bake cache. It
//! skips, loudly, when that cache is missing or from another format version.

use mc_data::{BlockState, RenderType};
use mc_math::{BlockPos, ChunkSectionPos};
use mc_render::terrain::mesher::{mesh_section_reference, mesh_section_with_alternatives};
use mc_render::terrain::{fixture, ChunkMesh, MeshSettings};
use mc_world::World;
use std::sync::OnceLock;

/// Install `crates/mc-client/cache/baked-assets.bin` once for this binary.
///
/// The file is `MAGIC(8) | version u32 | key len u32 | key | payload len u64 | checksum u64 |
/// payload` (`mc_assets::cache::save`); the key names the jar it was baked from, which this
/// sandbox does not have, so the payload is decoded directly rather than through `cache::load`.
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
            eprintln!("SKIP: not a bake cache");
            return false;
        }
        if u32_at(8) != mc_assets::cache::FORMAT_VERSION {
            eprintln!("SKIP: bake cache format {} is not current", u32_at(8));
            return false;
        }
        let payload_at = 16 + u32_at(12) as usize;
        if bytes.len() < payload_at + 16 {
            eprintln!("SKIP: truncated bake cache");
            return false;
        }
        let payload_len = u64_at(payload_at) as usize;
        let Some(payload) = bytes.get(payload_at + 16..payload_at + 16 + payload_len) else {
            eprintln!("SKIP: truncated bake cache");
            return false;
        };
        let snapshot = match mc_assets::cache::BakedSnapshot::decode(payload) {
            Ok(snapshot) => snapshot,
            Err(miss) => {
                eprintln!("SKIP: bake cache did not decode: {miss:?}");
                return false;
            }
        };
        mc_assets::model::install(snapshot.into_assets());
        true
    })
}

/// A tiny deterministic generator so the fixtures are the same on every run and platform.
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

/// Every legal state that the walk does something with (models and fluids).
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

/// A 3×3-column world whose section `(0, 4, 0)` and its whole neighbour shell are filled with
/// states drawn from `states` at the given density, so faces cull against real neighbours and
/// AO, fences, panes, stairs and grass all see mixed surroundings.
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

fn assert_identical(fast: &ChunkMesh, reference: &ChunkMesh, label: &str) {
    for ((layer, a), (_, b)) in fast.layers().zip(reference.layers()) {
        assert_eq!(
            a.as_bytes(),
            b.as_bytes(),
            "{label}: {layer:?} vertex bytes differ"
        );
    }
    assert_eq!(fast.visibility, reference.visibility, "{label}: visibility");
    assert_eq!(fast.occlusion, reference.occlusion, "{label}: occlusion");
}

#[test]
fn fast_paths_are_byte_identical_to_the_reference_mesher() {
    if !install_cached_bake() {
        return;
    }
    let states = renderable_states();
    assert!(states.len() > 100, "{} renderable states", states.len());
    let section = ChunkSectionPos::new(0, 4, 0);
    let all_settings = [
        MeshSettings {
            fancy_graphics: true,
            ambient_occlusion: true,
            ..MeshSettings::default()
        },
        MeshSettings {
            fancy_graphics: true,
            ambient_occlusion: false,
            ..MeshSettings::default()
        },
        MeshSettings {
            fancy_graphics: false,
            ambient_occlusion: true,
            ..MeshSettings::default()
        },
        MeshSettings {
            fancy_graphics: false,
            ambient_occlusion: false,
            ..MeshSettings::default()
        },
    ];
    let mut quads = 0usize;
    for (density, seed) in [(15u64, 1u64), (50, 2), (90, 3), (100, 4)] {
        let w = scatter_world(&states, density, seed);
        for settings in all_settings {
            for alternatives in [true, false] {
                let label =
                    format!("density {density}%, {settings:?}, alternatives {alternatives}");
                let fast = mesh_section_with_alternatives(&w, section, settings, alternatives);
                let reference = mesh_section_reference(&w, section, settings, alternatives);
                assert_identical(&fast, &reference, &label);
                quads += fast.quad_count();
            }
        }
    }
    assert!(
        quads > 10_000,
        "the fixtures must actually produce geometry ({quads} quads)"
    );
}

#[test]
fn each_state_alone_and_in_a_crowd_of_itself_matches_the_reference() {
    if !install_cached_bake() {
        return;
    }
    // One state per section: alone (every face exposed) and surrounded by copies of itself
    // (same-block culling: glass, ice, panes, leaves, slabs, fluids).
    let settings = MeshSettings::default();
    let centre = BlockPos::new(8, 72, 8);
    let section = ChunkSectionPos::new(0, 4, 0);
    let mut w = fixture::blank_world(1);
    for (index, state) in renderable_states().into_iter().enumerate() {
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    w.set_block(centre.add(dx, dy, dz), BlockState::AIR);
                }
            }
        }
        w.set_block(centre, state);
        let label = format!("#{index} {state:?} alone");
        assert_identical(
            &mesh_section_with_alternatives(&w, section, settings, true),
            &mesh_section_reference(&w, section, settings, true),
            &label,
        );
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    w.set_block(centre.add(dx, dy, dz), state);
                }
            }
        }
        let label = format!("#{index} {state:?} crowded");
        assert_identical(
            &mesh_section_with_alternatives(&w, section, settings, true),
            &mesh_section_reference(&w, section, settings, true),
            &label,
        );
    }
}

#[test]
fn a_section_snapshot_meshes_like_the_whole_world() {
    if !install_cached_bake() {
        return;
    }
    // `World::render_snapshot_for_sections` keeps only the 3×3 columns around each section. Make
    // every column *outside* that neighbourhood differ from "absent" (stone instead of air), so a
    // mesher read that strays past the neighbourhood would see stone in the whole world and air
    // in the snapshot — and the meshes would differ.
    let states = renderable_states();
    let mut w = fixture::blank_world(2);
    let stone = fixture::state_of("minecraft:stone");
    for x in -32..48 {
        for z in -32..48 {
            if (-16..32).contains(&x) && (-16..32).contains(&z) {
                continue;
            }
            for y in 60..84 {
                w.set_block(BlockPos::new(x, y, z), stone);
            }
        }
    }
    let mut rng = Lcg(11);
    for x in -16..32 {
        for y in 60..84 {
            for z in -16..32 {
                if rng.next_u64() % 100 < 60 {
                    let state = states[(rng.next_u64() as usize) % states.len()];
                    w.set_block(BlockPos::new(x, y, z), state);
                }
            }
        }
    }
    let section = ChunkSectionPos::new(0, 4, 0);
    let snapshot = w.render_snapshot_for_sections(&[section]);
    assert!(
        snapshot.chunk(mc_math::ChunkPos::new(2, 2)).is_none(),
        "the outer ring must not be in the section snapshot"
    );
    for settings in [
        MeshSettings::default(),
        MeshSettings {
            fancy_graphics: false,
            ambient_occlusion: false,
            ..MeshSettings::default()
        },
    ] {
        let label = format!("section snapshot, {settings:?}");
        assert_identical(
            &mesh_section_with_alternatives(&snapshot, section, settings, true),
            &mesh_section_with_alternatives(&w, section, settings, true),
            &label,
        );
    }
}
