//! Renderer-level regression for the unloaded-camera-section visibility gap.
//!
//! This must stay an actual `TerrainPhase`/`TerrainRenderer` frame test.  A cull-only test would
//! merely mirror the bool fallback and would not cover the resident-map filter, walk cache, GPU
//! upload, or generation invalidation.

use mc_math::{BlockPos, ChunkSectionPos, DVec3};
use mc_render::terrain::{fixture, mesh_section, ChunkMesh, TerrainOptions, TerrainPhase};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, RenderError};
use std::path::{Path, PathBuf};

const WIDTH: u32 = 96;
const HEIGHT: u32 = 96;

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let path = PathBuf::from(dir);
        return path.is_dir().then_some(path);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let path = root.join("MCP-919-main").join("jars");
    path.is_dir().then_some(path)
}

fn vanilla_atlas() -> Option<mc_assets::Atlas> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        eprintln!("SKIP: 1.8.9.jar not found; set MC_JARS_DIR");
        return None;
    }
    let mut resources = mc_assets::ResourceManager::new();
    resources.push_pack(Box::new(
        mc_assets::ZipResourcePack::open(&jar).expect("open 1.8.9.jar"),
    ));
    let (assets, report) = mc_assets::bake_all(&resources, 2048).expect("bake_all");
    assert!(report.is_clean());
    let atlas = assets.atlas.clone();
    mc_assets::model::install(assets);
    let _ = mc_render::terrain::tint::load_colormaps(&resources);
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
        Ok(gpu) => Some(gpu),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available");
            None
        }
        Err(error) => panic!("headless context: {error}"),
    }
}

fn camera_in_b() -> Camera {
    let mut camera = Camera {
        // B is z=0; A is z=3.  yaw=0 faces +Z, putting remote A in the frustum.
        pos: DVec3::new(8.0, 8.0, 8.0),
        yaw: 0.0,
        pitch: 0.0,
        fov: 70.0,
        near: 0.05,
        far: 256.0,
        aspect: WIDTH as f32 / HEIGHT as f32,
        ..Camera::default()
    };
    camera.update();
    camera
}

fn frame(phases: &mut PhaseList, gpu: &mut Gpu, camera: &Camera) {
    let target = gpu.acquire().expect("acquire");
    phases.render(gpu, target.view(), camera, 0.0);
}

#[test]
fn missing_camera_section_uses_frustum_fallback_and_generation_recovers_smart_walk() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };
    let camera = camera_in_b();
    let a = ChunkSectionPos::new(0, 0, 3);
    let b = ChunkSectionPos::new(0, 0, 0);

    let mut world = fixture::blank_world(1);
    fixture::fill(
        &mut world,
        BlockPos::new(0, 0, 0),
        BlockPos::new(15, 15, 15),
        fixture::state_of("minecraft:stone"),
    );
    let opaque_b = mesh_section(&world, b);

    let mut phases = PhaseList::new();
    let shared = TerrainPhase::install_with(&gpu, &atlas, TerrainOptions::default(), &mut phases);
    {
        let mut renderer = shared.borrow_mut();
        renderer.render_distance = 4;
        renderer.smart_culling = true;
        renderer.upload(&gpu, ChunkMesh::empty(a), camera.pos);
    }

    // B and the two intermediate sections are absent, so the fallback must retain remote A.
    frame(&mut phases, &mut gpu, &camera);
    assert!(shared
        .borrow()
        .visible_section_positions()
        .any(|pos| pos == a));

    // Installing opaque B restores the normal smart walk; A is behind B and must be excluded.
    {
        shared.borrow_mut().upload(&gpu, opaque_b, camera.pos);
    }
    frame(&mut phases, &mut gpu, &camera);
    assert!(shared
        .borrow()
        .visible_section_positions()
        .any(|pos| pos == b));
    assert!(!shared
        .borrow()
        .visible_section_positions()
        .any(|pos| pos == a));

    // Removing B bumps chunk_generation; the next frame must not reuse the old smart result.
    shared.borrow_mut().remove(b);
    frame(&mut phases, &mut gpu, &camera);
    assert!(shared
        .borrow()
        .visible_section_positions()
        .any(|pos| pos == a));
}
