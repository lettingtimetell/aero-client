//! X14 measurement, not a regression test: section mesh time on a
//! plains-like section (grass over three layers of dirt over stone, a few flowers and tall grass),
//! the case the "unchanged actual state" dense lookup in `render_model` targets.
//!
//! Run by hand: `cargo test -p mc-render --test x14_mesh_bench -- --ignored --nocapture`.
//! Timings are provisional on a busy machine.

use std::sync::OnceLock;
use std::time::Instant;

use mc_math::{BlockPos, ChunkSectionPos};
use mc_render::terrain::mesher::mesh_section_with_alternatives;
use mc_render::terrain::{fixture, MeshSettings};

/// Install `crates/mc-client/cache/baked-assets.bin` once (same loader as
/// `mesher_fast_path_parity.rs`).
fn install_cached_bake() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../mc-client/cache/baked-assets.bin");
        let Ok(bytes) = std::fs::read(&path) else {
            return false;
        };
        let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let u64_at = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
        if bytes.len() < 16
            || &bytes[..8] != b"MCRSBAKE"
            || u32_at(8) != mc_assets::cache::FORMAT_VERSION
        {
            return false;
        }
        let payload_at = 16 + u32_at(12) as usize;
        let payload_len = u64_at(payload_at) as usize;
        let Some(payload) = bytes.get(payload_at + 16..payload_at + 16 + payload_len) else {
            return false;
        };
        let Ok(snapshot) = mc_assets::cache::BakedSnapshot::decode(payload) else {
            return false;
        };
        mc_assets::model::install(snapshot.into_assets());
        true
    })
}

#[test]
#[ignore = "measurement; run by hand with --ignored --nocapture"]
fn plains_section_mesh_time() {
    assert!(install_cached_bake(), "needs the checked-in bake cache");
    let mut world = fixture::blank_world(1);
    let grass = fixture::state_of("minecraft:grass");
    let dirt = fixture::state_of("minecraft:dirt");
    let stone = fixture::state_of("minecraft:stone");
    let tall_grass =
        mc_data::BlockState::new(mc_data::block_by_name("minecraft:tallgrass").unwrap(), 1);
    let flower = fixture::state_of("minecraft:red_flower");
    for x in -1..17 {
        for z in -1..17 {
            for y in 60..76 {
                world.set_block(BlockPos::new(x, y, z), stone);
            }
            for y in 76..79 {
                world.set_block(BlockPos::new(x, y, z), dirt);
            }
            world.set_block(BlockPos::new(x, 79, z), grass);
            let hash = (x * 31 + z * 17).rem_euclid(13);
            if hash == 0 {
                world.set_block(BlockPos::new(x, 80, z), tall_grass);
            } else if hash == 5 {
                world.set_block(BlockPos::new(x, 80, z), flower);
            }
        }
    }
    // Section y 4 covers 64..80: stone 64..75, dirt 76..78, grass 79 (the surface section).
    let section = ChunkSectionPos::new(0, 4, 0);
    let settings = MeshSettings {
        fancy_graphics: true,
        ambient_occlusion: true,
        ..MeshSettings::default()
    };
    let mut quads = 0;
    for _ in 0..20 {
        quads = mesh_section_with_alternatives(&world, section, settings, true).quad_count();
    }
    let rounds = 400;
    let start = Instant::now();
    for _ in 0..rounds {
        let mesh = mesh_section_with_alternatives(&world, section, settings, true);
        std::hint::black_box(&mesh);
    }
    let per = start.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
    println!("plains surface section: {per:.1} us per mesh ({quads} quads)");
}
