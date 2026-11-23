//! Scratch renderer-level regression for authoritative world-column membership.
//!
//! This must stay an actual `TerrainPhase`/`TerrainRenderer` frame test.  A cull-only test would
//! merely mirror the bool fallback and would not cover the resident-map filter, walk cache, GPU
//! upload, or generation invalidation.

use mc_math::{BlockPos, ChunkPos, ChunkSectionPos, DVec3};
use mc_render::terrain::{
    fixture, mesh_section, ChunkMesh, TerrainOptions, TerrainPhase, TerrainRenderer,
};
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

fn visible(
    shared: &std::rc::Rc<std::cell::RefCell<TerrainRenderer>>,
    pos: ChunkSectionPos,
) -> bool {
    shared
        .borrow()
        .visible_section_positions()
        .any(|candidate| candidate == pos)
}

fn frame(phases: &mut PhaseList, gpu: &mut Gpu, camera: &Camera) {
    let target = gpu.acquire().expect("acquire");
    phases.render(gpu, target.view(), camera, 0.0);
}

#[test]
fn authoritative_world_columns_cross_unloaded_corridor_and_preserve_dummy() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };
    let camera = camera_in_b();
    let camera_section = ChunkSectionPos::new(0, 0, 0);
    let corridor_section = ChunkSectionPos::new(0, 0, 1);
    let remote_section = ChunkSectionPos::new(0, 0, 3);
    let camera_column = ChunkPos::new(0, 0);
    let corridor_column = ChunkPos::new(0, 1);
    let remote_column = ChunkPos::new(0, 3);

    let mut world = fixture::blank_world(1);
    // blank_world(1) covers z=-1..1; explicitly load the remote z=3 column so the fill and
    // mesher cannot silently exercise an unloaded-column no-op.
    fixture::load_blank_chunk(&mut world, remote_column);
    fixture::fill(
        &mut world,
        BlockPos::new(0, 0, 16),
        BlockPos::new(15, 15, 31),
        fixture::state_of("minecraft:stone"),
    );
    fixture::fill(
        &mut world,
        BlockPos::new(0, 0, 48),
        BlockPos::new(15, 15, 63),
        fixture::state_of("minecraft:stone"),
    );
    let opaque_corridor = mesh_section(&world, corridor_section);
    let remote_mesh = mesh_section(&world, remote_section);
    assert!(opaque_corridor.quad_count() > 0);
    assert!(remote_mesh.quad_count() > 0);

    let mut phases = PhaseList::new();
    let shared = TerrainPhase::install_with(&gpu, &atlas, TerrainOptions::default(), &mut phases);
    {
        let mut renderer = shared.borrow_mut();
        renderer.render_distance = 4;
        renderer.smart_culling = true;
        renderer.set_world_columns([camera_column, corridor_column, remote_column]);
        renderer.upload(&gpu, ChunkMesh::empty(camera_section), camera.pos);
        // Start with a loaded air corridor. An all-visible empty mesh must let the remote
        // geometry through, proving this is an actual resident walk rather than a frustum-only
        // assertion.
        renderer.upload(&gpu, ChunkMesh::empty(corridor_section), camera.pos);
        renderer.upload(&gpu, remote_mesh.clone(), camera.pos);
    }

    // Loaded camera -> loaded air corridor -> remote geometry.
    frame(&mut phases, &mut gpu, &camera);
    assert!(visible(&shared, remote_section));

    // Remove the intermediate mesh and mark its world column absent. The authoritative source
    // supplies settled all-air visibility, so the remote geometry stays reachable.
    {
        let mut renderer = shared.borrow_mut();
        renderer.remove(corridor_section);
        renderer.mark_world_column_unloaded(corridor_column);
    }
    frame(&mut phases, &mut gpu, &camera);
    assert!(visible(&shared, remote_section));
    assert!(shared.borrow().visibility(corridor_section).is_none());

    // Reload the column before its replacement mesh arrives. Loaded-but-unmeshed is DUMMY and
    // must hard-stop smart traversal; loading must clear the air answer immediately.
    shared
        .borrow_mut()
        .mark_world_column_loaded(corridor_column);
    frame(&mut phases, &mut gpu, &camera);
    assert!(!visible(&shared, remote_section));

    // Once the opaque mesh arrives, the normal resident visibility path remains blocking.
    shared
        .borrow_mut()
        .upload(&gpu, opaque_corridor.clone(), camera.pos);
    frame(&mut phases, &mut gpu, &camera);
    assert!(!visible(&shared, remote_section));

    // Exercise GPU eviction while the opaque corridor is still resident and the world still says
    // it is loaded. The tiny cap must evict the corridor; eviction is cache-only and must not
    // turn it into settled air.
    let full_bytes = shared.borrow().gpu_memory().bytes;
    shared
        .borrow_mut()
        .set_gpu_budget(mc_render::terrain::jobs::GpuBudget::new(1));
    frame(&mut phases, &mut gpu, &camera);
    let evicted = shared.borrow_mut().take_evicted();
    assert!(evicted.contains(&corridor_section));
    assert!(shared.borrow().visibility(corridor_section).is_none());

    // Re-enable enough budget and restore only the camera/remote buffers. The evicted loaded
    // corridor remains DUMMY, so the remote must still be hidden.
    shared
        .borrow_mut()
        .set_gpu_budget(mc_render::terrain::jobs::GpuBudget::new(full_bytes.max(1)));
    shared
        .borrow_mut()
        .upload(&gpu, ChunkMesh::empty(camera_section), camera.pos);
    shared.borrow_mut().upload(&gpu, remote_mesh, camera.pos);
    assert!(shared.borrow().visibility(remote_section).is_some());
    frame(&mut phases, &mut gpu, &camera);
    assert!(!visible(&shared, remote_section));

    // Removing a mesh while the world still says the column is loaded must not silently turn it
    // into air. This is the cache-only remove invariant, exercised after an eviction too.
    shared
        .borrow_mut()
        .upload(&gpu, opaque_corridor, camera.pos);
    frame(&mut phases, &mut gpu, &camera);
    assert!(!visible(&shared, remote_section));
    shared.borrow_mut().remove(corridor_section);
    frame(&mut phases, &mut gpu, &camera);
    assert!(!visible(&shared, remote_section));

    // Replacing authority with an empty current-world set invalidates the walk generation and
    // makes the absent corridor settled air again. Restore the loaded membership afterward; the
    // still-unmeshed corridor must return to DUMMY hard-stop behavior.
    shared
        .borrow_mut()
        .set_world_columns(std::iter::empty::<ChunkPos>());
    frame(&mut phases, &mut gpu, &camera);
    assert!(visible(&shared, remote_section));
    shared
        .borrow_mut()
        .set_world_columns([camera_column, corridor_column, remote_column]);
    frame(&mut phases, &mut gpu, &camera);
    assert!(!visible(&shared, remote_section));
}
