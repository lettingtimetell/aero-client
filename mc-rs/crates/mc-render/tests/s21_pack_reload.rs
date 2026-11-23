//! S21: a resource pack changing a **block's** texture must change what the world looks like,
//! without a restart — and removing it must restore the frame byte for byte.
//!
//! This is the GPU half of the hot-reload proof. `mc-client`'s `hot_reload.rs` covers the client
//! plumbing (the pack stack, the language table, GUI textures, the atlas re-stitch); this file
//! covers the part that needs a *world*: the real mesher reading the real global model registry,
//! the real atlas upload, and the real terrain pipelines.
//!
//! java: `Minecraft.refreshResources` → `ModelManager.onResourceManagerReload` re-bakes and
//! `RenderGlobal.loadRenderers()` throws the chunk renderers away, because the atlas they sampled
//! is a different GL texture now. `TerrainPhase` has the same property: the atlas is bound at
//! construction, so the phases are rebuilt rather than updated.
//!
//! **This file owns the process globals**, like `s11_render.rs` does: `mc_assets::model_for` is
//! process-wide, so everything is sequenced inside one `#[test]`.

use std::path::{Path, PathBuf};

use mc_math::{BlockPos, ChunkSectionPos, DVec3};
use mc_render::terrain::{fixture, mesh_section, TerrainPhase};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, RenderError};
use mc_world::World;

const WIDTH: u32 = 192;
const HEIGHT: u32 = 144;
const MAX_ATLAS: u32 = 2048;

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let path = PathBuf::from(dir);
        return path.is_dir().then_some(path);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let path = root.join("MCP-919-main").join("jars");
    path.is_dir().then_some(path)
}

fn jar() -> Option<PathBuf> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    jar.is_file().then_some(jar)
}

/// The vanilla stack, optionally with a folder pack on top.
fn stack(extra: Option<&Path>) -> Option<mc_assets::ResourceManager> {
    let mut manager = mc_assets::ResourceManager::new();
    manager.push_pack(Box::new(mc_assets::ZipResourcePack::open(jar()?).ok()?));
    if let Some(pack) = extra {
        manager.push_pack(Box::new(
            mc_assets::FolderResourcePack::open(pack).expect("folder pack"),
        ));
    }
    Some(manager)
}

/// A pack whose `blocks/stone.png` is one flat colour.
fn repaint_stone(dir: &Path, rgb: [u8; 3]) -> PathBuf {
    let pack = dir.join("Repaint");
    let blocks = pack.join("assets/minecraft/textures/blocks");
    std::fs::create_dir_all(&blocks).expect("mkdir");
    std::fs::write(
        pack.join("pack.mcmeta"),
        br#"{"pack":{"pack_format":1,"description":"s21"}}"#,
    )
    .expect("mcmeta");
    let mut image = image::RgbaImage::new(16, 16);
    for pixel in image.pixels_mut() {
        *pixel = image::Rgba([rgb[0], rgb[1], rgb[2], 255]);
    }
    image.save(blocks.join("stone.png")).expect("stone.png");
    pack
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
        Ok(gpu) => {
            gpu.device.on_uncaptured_error(std::sync::Arc::new(|error| {
                panic!("wgpu validation error: {error}")
            }));
            Some(gpu)
        }
        Err(RenderError::NoAdapter) => {
            eprintln!("skip: no wgpu adapter available on this machine");
            None
        }
        Err(error) => panic!("headless context: {error}"),
    }
}

/// A solid stone floor, filling the lower half of the frame.
fn scene() -> World {
    let mut world = fixture::blank_world(1);
    let stone = fixture::state_of("minecraft:stone");
    fixture::fill(
        &mut world,
        BlockPos::new(0, 16, 0),
        BlockPos::new(15, 16, 15),
        stone,
    );
    world
}

fn camera() -> Camera {
    let mut camera = Camera {
        pos: DVec3::new(8.0, 22.0, -8.0),
        yaw: 0.0,
        pitch: 25.0,
        fov: 70.0,
        near: 0.05,
        far: 256.0,
        aspect: WIDTH as f32 / HEIGHT as f32,
        ..Camera::default()
    };
    camera.update();
    camera
}

/// Build a whole fresh phase list over `atlas` — exactly what `Client::reinstall` does — mesh the
/// scene and read the frame back.
fn render(gpu: &mut Gpu, atlas: &mc_assets::Atlas) -> Vec<u8> {
    let world = scene();
    let cam = camera();
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.10,
        g: 0.10,
        b: 0.10,
        a: 1.0,
    };
    let shared = TerrainPhase::install(gpu, atlas, &mut phases);
    {
        let mut renderer = shared.borrow_mut();
        renderer.render_distance = 8;
        renderer.upload(
            gpu,
            mesh_section(&world, ChunkSectionPos::new(0, 1, 0)),
            cam.pos,
        );
    }
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &cam, 0.0);
    gpu.read_pixels().expect("readback")
}

/// Pixels whose red and blue are equal and clearly above green — vanilla's grey stone has
/// `r == g == b`, so this counts *only* the repainted magenta, shading and lightmap included.
fn magenta_pixels(pixels: &[u8]) -> usize {
    pixels
        .chunks_exact(4)
        .filter(|p| p[0] > 16 && p[0] == p[2] && p[1].saturating_add(16) < p[0])
        .count()
}

/// Pixels that are neutral grey and not the background — vanilla stone.
fn grey_pixels(pixels: &[u8]) -> usize {
    pixels
        .chunks_exact(4)
        .filter(|p| p[0] == p[1] && p[1] == p[2] && p[0] > 40)
        .count()
}

#[test]
fn a_pack_repainting_stone_changes_the_rendered_world_and_reverting_restores_it_byte_for_byte() {
    let Some(mut gpu) = gpu() else { return };
    let Some(vanilla) = stack(None) else {
        eprintln!("skip: 1.8.9.jar not found; set MC_JARS_DIR");
        return;
    };

    // ---- vanilla ----
    assert_eq!(
        mc_assets::model::installed_generation(),
        0,
        "this test binary must own the global registry"
    );
    let (baked, report) = mc_assets::bake_all(&vanilla, MAX_ATLAS).expect("bake");
    assert!(report.is_clean());
    let vanilla_atlas = baked.atlas.clone();
    mc_assets::model::install(baked);
    mc_render::terrain::tint::load_colormaps(&vanilla).expect("colormaps");

    let before = render(&mut gpu, &vanilla_atlas);
    assert!(
        grey_pixels(&before) > 2_000,
        "vanilla stone must actually be drawn: {} grey pixels",
        grey_pixels(&before)
    );
    assert_eq!(
        magenta_pixels(&before),
        0,
        "the control: nothing magenta before the pack"
    );

    // ---- the same bake reinstalled: byte-identical, which is the inertness claim ----
    let (again, _) = mc_assets::bake_all(&vanilla, MAX_ATLAS).expect("bake again");
    let again_atlas = again.atlas.clone();
    mc_assets::model::reinstall(again);
    mc_render::terrain::tint::reload_colormaps(&vanilla).expect("colormaps again");
    let unchanged = render(&mut gpu, &again_atlas);
    assert_eq!(
        before, unchanged,
        "reinstalling an identical bake must render a byte-identical frame — a client that \
         never swaps a pack is not affected by any of this"
    );

    // ---- pack on ----
    let dir = tempfile::tempdir().expect("scratch dir");
    let pack = repaint_stone(dir.path(), [255, 0, 255]);
    let packed = stack(Some(&pack)).expect("stacked");
    let (repainted, _) = mc_assets::bake_all(&packed, MAX_ATLAS).expect("bake with pack");
    let repainted_atlas = repainted.atlas.clone();
    mc_assets::model::reinstall(repainted);
    mc_render::terrain::tint::reload_colormaps(&packed).expect("colormaps with pack");

    let after = render(&mut gpu, &repainted_atlas);
    let magenta = magenta_pixels(&after);
    assert!(
        magenta > 2_000,
        "the pack must change the block actually drawn on the GPU: {magenta} magenta pixels"
    );
    assert_ne!(before, after);

    // ---- pack off ----
    let (restored, _) = mc_assets::bake_all(&vanilla, MAX_ATLAS).expect("bake vanilla again");
    let restored_atlas = restored.atlas.clone();
    mc_assets::model::reinstall(restored);
    mc_render::terrain::tint::reload_colormaps(&vanilla).expect("colormaps restored");
    let reverted = render(&mut gpu, &restored_atlas);
    assert_eq!(
        before, reverted,
        "removing the pack must restore the rendered world byte for byte"
    );
    assert!(mc_assets::model::installed_generation() >= 4);
}
