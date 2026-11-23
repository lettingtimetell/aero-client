//! Back-face culling: the GPU half of the evidence for turning `cull_mode` on in `terrain/`.
//!
//! S11 originally shipped `cull_mode: None` with an explicit reason — "S09's baked winding has
//! not been verified against vanilla's, and drawing both sides of closed opaque geometry is
//! visually identical while a wrong cull mode is catastrophic". That reason is now discharged
//! from both ends:
//!
//! * **CPU, in `mc-assets`** — `every_baked_quad_winds_counter_clockwise_about_its_own_facing`
//!   asserts that for all six faces x all sixteen `x`/`y` model rotations, the `0,1,2` triangle
//!   S11 uploads has a normal pointing *along* the quad's own facing. That is
//!   counter-clockwise-seen-from-outside, which is `wgpu::FrontFace::Ccw`.
//! * **GPU, here** — the same closed opaque scene rendered with culling on and off is
//!   **byte-identical** (so nothing visible was lost), while a deliberately reversed quad
//!   *disappears* under culling and reappears without it (so culling is genuinely active and it
//!   is the reversed winding, not ours, that it removes).
//!
//! The second half is what makes the first non-vacuous: "identical frames" alone would also be
//! the result of a `cull_mode` that quietly did nothing.
//!
//! # Why the scene here is opaque-only
//!
//! Culling on and off are *not* equivalent for translucent geometry, and that is correct rather
//! than a limitation. `terrain::fluid` emits vanilla's mirrored underside quad explicitly
//! (`BlockFluidRenderer` does the same), so with culling on exactly one of the pair survives
//! from any given side; with culling off both draw and the alpha blend applies twice. Vanilla
//! renders with `GL_CULL_FACE` enabled, so "on" is the vanilla answer and the difference over
//! water is the bug the old setting was hiding.

use mc_math::{BlockPos, ChunkSectionPos, DVec3};
use mc_render::terrain::{
    fixture, mesh_section, Buffer, ChunkMesh, TerrainOptions, TerrainPhase, TerrainVertex,
};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, RenderError};
use std::path::{Path, PathBuf};

const WIDTH: u32 = 192;
const HEIGHT: u32 = 144;

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn vanilla_atlas() -> Option<mc_assets::Atlas> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        eprintln!("SKIP: 1.8.9.jar not found; set MC_JARS_DIR");
        return None;
    }
    let mut rm = mc_assets::ResourceManager::new();
    rm.push_pack(Box::new(
        mc_assets::ZipResourcePack::open(&jar).expect("open 1.8.9.jar"),
    ));
    let (assets, report) = mc_assets::bake_all(&rm, 2048).expect("bake_all");
    assert!(report.is_clean());
    let atlas = assets.atlas.clone();
    mc_assets::model::install(assets);
    let _ = mc_render::terrain::tint::load_colormaps(&rm);
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
        Ok(g) => {
            g.device.on_uncaptured_error(std::sync::Arc::new(|e| {
                panic!("wgpu validation error: {e}")
            }));
            Some(g)
        }
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available");
            None
        }
        Err(e) => panic!("headless context: {e}"),
    }
}

fn camera() -> Camera {
    let mut c = Camera {
        pos: DVec3::new(8.0, 26.0, -12.0),
        yaw: 0.0,
        pitch: 25.0,
        fov: 70.0,
        near: 0.05,
        far: 256.0,
        aspect: WIDTH as f32 / HEIGHT as f32,
        ..Camera::default()
    };
    c.update();
    c
}

/// Opaque only, on purpose (see the module docs): a stone floor, a dirt pillar, and one
/// isolated block floating in air so that all six of *its* faces are emitted with no
/// neighbour culling to hide the back three.
fn opaque_scene() -> mc_world::World {
    let mut w = fixture::blank_world(1);
    let stone = fixture::state_of("minecraft:stone");
    let dirt = fixture::state_of("minecraft:dirt");
    let log = fixture::state_of("minecraft:log");
    fixture::fill(
        &mut w,
        BlockPos::new(0, 16, 0),
        BlockPos::new(15, 16, 15),
        stone,
    );
    fixture::fill(
        &mut w,
        BlockPos::new(4, 17, 8),
        BlockPos::new(4, 21, 8),
        dirt,
    );
    // Floating, so nothing culls its faces at mesh time.
    w.set_block(BlockPos::new(10, 22, 6), log);
    w
}

fn render_scene(gpu: &mut Gpu, atlas: &mc_assets::Atlas, cull: bool) -> Vec<u8> {
    let world = opaque_scene();
    let cam = camera();
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.1,
        g: 0.1,
        b: 0.1,
        a: 1.0,
    };
    let shared = TerrainPhase::install_with(
        gpu,
        atlas,
        TerrainOptions {
            cull_backfaces: cull,
            ..TerrainOptions::default()
        },
        &mut phases,
    );
    {
        let mut r = shared.borrow_mut();
        r.render_distance = 8;
        for y in 0..3 {
            r.upload(
                gpu,
                mesh_section(&world, ChunkSectionPos::new(0, y, 0)),
                cam.pos,
            );
        }
    }
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), &cam, 0.0);
    gpu.read_pixels().expect("readback")
}

fn drawn_pixels(px: &[u8]) -> usize {
    px.chunks_exact(4)
        .filter(|p| p[0].abs_diff(26) > 3 || p[1].abs_diff(26) > 3 || p[2].abs_diff(26) > 3)
        .count()
}

/// Nothing visible is lost. If the winding were reversed, back-face culling would remove every
/// outward face of the scene and this would go from "identical" to "almost empty".
#[test]
fn culling_back_faces_leaves_a_closed_opaque_scene_byte_identical() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };

    let culled = render_scene(&mut gpu, &atlas, true);
    let uncalled = render_scene(&mut gpu, &atlas, false);

    let drawn = drawn_pixels(&culled);
    eprintln!(
        "cull on: {drawn}/{} pixels drawn",
        (WIDTH * HEIGHT) as usize
    );
    assert!(
        drawn > (WIDTH * HEIGHT) as usize / 12,
        "the scene barely drew with culling on ({drawn} px) — the winding is probably inverted"
    );

    let differing = culled
        .chunks_exact(4)
        .zip(uncalled.chunks_exact(4))
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing, 0,
        "{differing} pixels changed when back-face culling was enabled; every face S09 bakes \
         should be front-facing from outside"
    );
}

/// ...and culling is not quietly a no-op: a quad wound the *other* way vanishes under it.
///
/// Both meshes are the same four corners of the same square, in the same place, with the same
/// UVs — only the vertex order differs. This is the "assert the plausible-wrong answer does not
/// occur" half, in the style of S11's own AO tests.
#[test]
fn a_reversed_quad_is_culled_and_the_correct_one_is_not() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };

    // A big square in the z = 8 plane, facing -Z (north) — i.e. toward a camera sitting at
    // negative z. Vanilla's `EnumFaceDirection` order for NORTH is
    // (E,U,N) (E,D,N) (W,D,N) (W,U,N), which is counter-clockwise seen from -Z.
    let stone_uv = mc_assets::sprite(&mc_assets::ResourceLocation::minecraft(
        "textures/blocks/stone",
    ))
    .uv;
    let corner = |x: f32, y: f32, u: f32, v: f32| {
        TerrainVertex::new([x, y, 8.0], [u, v], [255, 255, 255, 255], [240, 0, 0, 0])
    };
    let facing_camera = [
        corner(14.0, 14.0, stone_uv.max_u, stone_uv.min_v),
        corner(14.0, 2.0, stone_uv.max_u, stone_uv.max_v),
        corner(2.0, 2.0, stone_uv.min_u, stone_uv.max_v),
        corner(2.0, 14.0, stone_uv.min_u, stone_uv.min_v),
    ];
    let mut reversed = facing_camera;
    reversed.reverse();

    let shoot = |gpu: &mut Gpu, quad: [TerrainVertex; 4], cull: bool| -> usize {
        let mut cam = Camera {
            pos: DVec3::new(8.0, 24.0, -14.0),
            yaw: 0.0,
            pitch: 0.0,
            fov: 70.0,
            near: 0.05,
            far: 256.0,
            aspect: WIDTH as f32 / HEIGHT as f32,
            ..Camera::default()
        };
        cam.update();
        let mut phases = PhaseList::new();
        phases.clear_color = wgpu::Color {
            r: 0.1,
            g: 0.1,
            b: 0.1,
            a: 1.0,
        };
        let shared = TerrainPhase::install_with(
            gpu,
            &atlas,
            TerrainOptions {
                cull_backfaces: cull,
                ..TerrainOptions::default()
            },
            &mut phases,
        );
        {
            let mut solid = Buffer::new();
            solid.push_quad(quad);
            let mesh = ChunkMesh {
                solid,
                ..ChunkMesh::empty(ChunkSectionPos::new(0, 1, 0))
            };
            let mut r = shared.borrow_mut();
            r.render_distance = 8;
            r.upload(gpu, mesh, cam.pos);
        }
        let frame = gpu.acquire().expect("acquire");
        phases.render(gpu, frame.view(), &cam, 0.0);
        drawn_pixels(&gpu.read_pixels().expect("readback"))
    };

    let correct_culled = shoot(&mut gpu, facing_camera, true);
    let correct_uncalled = shoot(&mut gpu, facing_camera, false);
    let reversed_culled = shoot(&mut gpu, reversed, true);
    let reversed_uncalled = shoot(&mut gpu, reversed, false);
    eprintln!(
        "correct: {correct_culled} culled / {correct_uncalled} uncalled; \
         reversed: {reversed_culled} culled / {reversed_uncalled} uncalled"
    );

    assert!(
        correct_uncalled > 500,
        "the test quad did not draw at all ({correct_uncalled} px)"
    );
    assert_eq!(
        correct_culled, correct_uncalled,
        "vanilla's vertex order must survive FrontFace::Ccw + cull_mode Back"
    );
    assert_eq!(
        reversed_uncalled, correct_uncalled,
        "sanity: the reversed quad covers the same pixels when nothing is culled"
    );
    assert_eq!(
        reversed_culled, 0,
        "back-face culling is not actually removing anything — the identical-frames test above \
         would then be vacuous"
    );
}
