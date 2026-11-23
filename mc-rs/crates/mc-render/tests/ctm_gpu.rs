//! Connected textures on the GPU: the jar plus the reference pack
//! (read-only, skipped when absent) baked **with** its CTM tiles, the resulting 4096×2048 atlas
//! uploaded through the real `TerrainPhase`, and a clear-glass wall rendered headlessly with the
//! mod off and on.
//!
//! The pack's clear-glass set (`glass/Normal`) has a 1-texel frame on unconnected edges only and a
//! fully transparent tile 26, so with CTM on the per-block grid lines inside the wall disappear:
//! fewer pixels are drawn. Set `M8_SHOTS_DIR` to also save both frames as PNGs.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use mc_math::{BlockPos, ChunkSectionPos, DVec3};
use mc_render::terrain::{fixture, mesh_section_with, CtmMode, MeshSettings, TerrainPhase};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, RenderError};
use mc_world::World;

const WIDTH: u32 = 768;
const HEIGHT: u32 = 576;

fn jar() -> Option<PathBuf> {
    let dir = std::env::var_os("MC_JARS_DIR").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../MCP-919-main/jars"),
        PathBuf::from,
    );
    let jar = dir.join("versions/1.8.9/1.8.9.jar");
    jar.is_file().then_some(jar)
}

/// The reference pack (never checked in): `MC_REFERENCE_PACK`, or else the only folder in the
/// workspace's `resourcepacks/`.
fn users_pack() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_REFERENCE_PACK") {
        return Some(dir.into());
    }
    let packs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../resourcepacks");
    let mut dirs = std::fs::read_dir(packs)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir());
    let only = dirs.next()?;
    dirs.next().is_none().then_some(only)
}

/// Bake jar + user pack with CTM, install it, and keep the atlas for the GPU upload.
fn ctm_atlas() -> Option<&'static mc_assets::Atlas> {
    static ATLAS: OnceLock<Option<mc_assets::Atlas>> = OnceLock::new();
    ATLAS
        .get_or_init(|| {
            let (Some(jar), Some(pack)) = (jar(), users_pack()) else {
                eprintln!("SKIP: the vanilla jar or the reference pack is not available");
                return None;
            };
            let mut rm = mc_assets::ResourceManager::new();
            rm.push_pack(Box::new(
                mc_assets::ZipResourcePack::open(&jar).expect("open jar"),
            ));
            rm.push_pack(Box::new(
                mc_assets::FolderResourcePack::open(&pack).expect("open pack"),
            ));
            let (assets, report) =
                mc_assets::model::bake_all_with_ctm(&rm, 8192, 4, true).expect("CTM bake");
            assert!(report.is_clean());
            assert!(assets.ctm().is_some_and(|t| t.rule_count() > 0));
            let atlas = assets.atlas.clone();
            eprintln!("CTM atlas {}x{}", atlas.w, atlas.h);
            mc_assets::model::install(assets);
            Some(atlas)
        })
        .as_ref()
}

fn gpu() -> Option<Gpu> {
    let options = GpuOptions {
        vsync: false,
        sample_count: 1,
        low_power: false,
    };
    match Gpu::headless(WIDTH, HEIGHT, options) {
        Ok(gpu) => {
            gpu.device.on_uncaptured_error(std::sync::Arc::new(|e| {
                panic!("wgpu validation error: {e}")
            }));
            Some(gpu)
        }
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter");
            None
        }
        Err(e) => panic!("headless context: {e}"),
    }
}

/// A 12×6 clear-glass wall at z = 8, seen face-on from five blocks away.
fn scene() -> World {
    let mut w = fixture::blank_world(1);
    fixture::fill(
        &mut w,
        BlockPos::new(2, 17, 8),
        BlockPos::new(13, 22, 8),
        fixture::state_of("minecraft:glass"),
    );
    w
}

fn camera() -> Camera {
    let mut c = Camera {
        pos: DVec3::new(8.0, 20.0, 3.0),
        yaw: 0.0, // faces +Z
        pitch: 0.0,
        fov: 70.0,
        near: 0.05,
        far: 256.0,
        aspect: WIDTH as f32 / HEIGHT as f32,
        ..Camera::default()
    };
    c.update();
    c
}

fn render(gpu: &mut Gpu, atlas: &mc_assets::Atlas, mode: CtmMode) -> Vec<u8> {
    let world = scene();
    let cam = camera();
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.10,
        g: 0.10,
        b: 0.10,
        a: 1.0,
    };
    let terrain = TerrainPhase::install(gpu, atlas, &mut phases);
    {
        let mut terrain = terrain.borrow_mut();
        terrain.render_distance = 8;
        let settings = MeshSettings {
            connected_textures: mode,
            ..MeshSettings::default()
        };
        let mesh = mesh_section_with(&world, ChunkSectionPos::new(0, 1, 0), settings);
        terrain.upload(gpu, mesh, cam.pos);
    }
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &cam, 0.0);
    gpu.read_pixels().expect("readback")
}

fn drawn(px: &[u8]) -> usize {
    px.chunks_exact(4)
        .filter(|p| p[0].abs_diff(26) > 3 || p[1].abs_diff(26) > 3 || p[2].abs_diff(26) > 3)
        .count()
}

#[test]
fn ctm_removes_the_grid_lines_inside_a_glass_wall_on_the_gpu() {
    let Some(atlas) = ctm_atlas() else {
        return;
    };
    let Some(mut gpu) = gpu() else {
        return;
    };
    let off = render(&mut gpu, atlas, CtmMode::Off);
    let fancy = render(&mut gpu, atlas, CtmMode::Fancy);
    if let Some(dir) = std::env::var_os("M8_SHOTS_DIR") {
        let dir = PathBuf::from(dir);
        for (name, px) in [("ctm_gpu_off.png", &off), ("ctm_gpu_fancy.png", &fancy)] {
            image::save_buffer(dir.join(name), px, WIDTH, HEIGHT, image::ColorType::Rgba8)
                .expect("save shot");
        }
    }
    let (off_drawn, fancy_drawn) = (drawn(&off), drawn(&fancy));
    println!("drawn pixels: off {off_drawn}, fancy {fancy_drawn}");
    assert!(
        off_drawn > 1000,
        "the vanilla glass must draw its per-block frames"
    );
    assert!(fancy_drawn > 0, "the wall's outer frame is still drawn");
    assert!(
        fancy_drawn < off_drawn,
        "connected glass has no grid lines inside the wall (off {off_drawn}, fancy {fancy_drawn})"
    );
}
