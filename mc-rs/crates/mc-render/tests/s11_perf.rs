//! A coarse throughput check for the mesher and the culling walk.
//!
//! Not a benchmark — it runs in the debug profile and its thresholds are generous. It exists so
//! that a change which makes meshing an order of magnitude slower fails loudly instead of
//! quietly costing frames, and so the S11 handoff can quote a real number instead of a guess.

use std::path::{Path, PathBuf};
use std::time::Instant;

use mc_math::{BlockPos, ChunkSectionPos, DVec3};
use mc_render::terrain::{cull, fixture, frustum::Frustum, jobs, MeshSettings, SetVisibility};
use mc_render::Camera;

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn install() -> bool {
    let Some(jar) = jars_dir().map(|d| d.join("versions").join("1.8.9").join("1.8.9.jar")) else {
        eprintln!("SKIP: jars dir not found");
        return false;
    };
    if !jar.is_file() {
        eprintln!("SKIP: 1.8.9.jar not found");
        return false;
    }
    let mut rm = mc_assets::ResourceManager::new();
    rm.push_pack(Box::new(
        mc_assets::ZipResourcePack::open(&jar).expect("open jar"),
    ));
    let (assets, _) = mc_assets::bake_all(&rm, 2048).expect("bake_all");
    mc_assets::model::install(assets);
    mc_render::terrain::tint::load_colormaps(&rm).expect("colormaps");
    true
}

/// A dense, worst-ish case: solid stone up to y = 48 with a chequer of air pockets, so neither
/// the "all solid" nor the "all air" fast path dominates.
fn quarry(radius: i32) -> mc_world::World {
    let mut w = fixture::blank_world(radius);
    let stone = fixture::state_of("minecraft:stone");
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
                w.set_block(BlockPos::new(x, y, z), mc_data::BlockState::AIR);
            }
        }
    }
    w
}

#[test]
fn meshing_throughput_is_reported_and_not_pathological() {
    if !install() {
        return;
    }
    let world = quarry(1);
    let sections: Vec<ChunkSectionPos> = (-1..=1)
        .flat_map(|x| {
            (-1..=1).flat_map(move |z| (0..3).map(move |y| ChunkSectionPos::new(x, y, z)))
        })
        .collect();

    // Warm up the model cache and the allocator.
    let _ = jobs::mesh_all(&world, &sections, MeshSettings::default());

    let start = Instant::now();
    const ROUNDS: u32 = 4;
    let mut quads = 0usize;
    for _ in 0..ROUNDS {
        for m in jobs::mesh_all(&world, &sections, MeshSettings::default()) {
            quads += m.quad_count();
        }
    }
    let elapsed = start.elapsed();
    let total = sections.len() as u32 * ROUNDS;
    let per_section = elapsed.as_secs_f64() * 1000.0 / f64::from(total);
    eprintln!(
        "meshed {total} sections ({quads} quads) in {elapsed:?} = {per_section:.2} ms/section \
         (debug profile, {} rayon threads)",
        rayon::current_num_threads()
    );
    assert!(quads > 0);
    assert!(
        per_section < 250.0,
        "meshing collapsed to {per_section:.1} ms/section"
    );
}

#[test]
fn the_culling_walk_is_reported_and_bounded() {
    let mut cam = Camera {
        pos: DVec3::new(8.0, 72.0, 8.0),
        aspect: 16.0 / 9.0,
        far: 512.0,
        ..Camera::default()
    };
    cam.update();
    let f = Frustum::from_camera(&cam);
    let open = |pos: ChunkSectionPos| (0..16).contains(&pos.y).then(|| SetVisibility::all(true));

    let start = Instant::now();
    const ROUNDS: u32 = 200;
    let mut last = 0usize;
    for _ in 0..ROUNDS {
        last = cull::visible_sections(&open, &f, cam.pos, 12, true).len();
    }
    let per_walk = start.elapsed().as_secs_f64() * 1000.0 / f64::from(ROUNDS);
    eprintln!(
        "culling walk at render distance 12: {last} sections in {per_walk:.2} ms (debug profile)"
    );
    // 25x25 columns x 16 sections is the hard ceiling.
    assert!(last <= 25 * 25 * 16);
    assert!(per_walk < 100.0, "the culling walk took {per_walk:.1} ms");
}
