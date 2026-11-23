//! S11 task 10: mesh a chunk, render it headlessly, and snapshot-compare the result.
//!
//! This is the end-to-end check that the mesher, the pipelines, the camera-relative geometry
//! and the atlas upload all agree. It renders a fixed scene through the real
//! [`PhaseList`](mc_render::PhaseList) into an offscreen target and reads the pixels back,
//! then asserts three separate things:
//!
//! 1. **Structural** — the frame is not empty, the floor lands where the projection says it
//!    should, and the layers really do reach the target in the right order. These hold on a
//!    fresh checkout with no reference image.
//! 2. **Snapshot** — the frame is byte-compared (with a small tolerance) against
//!    `tests/snapshots/s11_terrain.png`, which is written on the first run and committed. This
//!    catches regressions in geometry, lighting, tint and UV assignment at once.
//! 3. **Determinism** — rendering the same scene twice produces the same pixels.
//!
//! # What this does *not* prove
//!
//! The reference image is this renderer's own output, not a vanilla screenshot. There is no
//! Minecraft client running in this environment to capture one from, so the snapshot proves
//! "nothing changed", not "this matches vanilla". Vanilla parity is asserted numerically
//! instead, in `s11_ao.rs`, `s11_mesher.rs`, `s11_fluid.rs` and `s11_lightmap_fog.rs`, against
//! values derived from the Java source. S22's parity harness is where a perceptual diff against
//! a real vanilla screenshot belongs.
//!
//! # Snapshot history
//!
//! The committed reference has been regenerated twice:
//!
//! 1. In session S09b, for three deliberate changes that all move the frame toward vanilla and
//!    are each proven by their own numeric test elsewhere: fluid quads now sample the real
//!    `water_still` texels instead of the whole atlas (`s11_fluid.rs`); terrain now samples
//!    S20's `EntityRenderer.updateLightmap` table instead of the shader's `light_curve`
//!    placeholder, which is very slightly darker at full daylight because vanilla's own table
//!    tops out at 250/255 rather than 255 (`s11_lightmap_fog.rs`); and back-face culling is on
//!    (`s11_cull.rs`, where the same *opaque* scene is proven byte-identical either way — the
//!    change here is confined to the water pool, whose mirrored underside quad vanilla also
//!    culls).
//! 2. On 2026-11-19, for the atlas colour-space fix. The 2026-09-10 reference was rendered while
//!    the terrain atlas was an `Rgba8UnormSrgb` texture, so every texel was linearised before
//!    the vertex colour and lightmap multiplied it. Parity run 2 (2026-09-10) switched the atlas
//!    to the plain `Rgba8Unorm` that Java's `TextureUtil.allocateTextureImpl` allocates
//!    (`GL_RGBA`; `JAVA_TERRAIN_ATLAS_FORMAT` and its GPU test in `terrain/phase.rs`), which
//!    brightened 6827 of the frame's 6987 geometry pixels by up to 72 per channel. The code of
//!    2026-11-19 with only the atlas format switched back to `Rgba8UnormSrgb` reproduced the old
//!    reference bit for bit (0 of 49 152 pixels differ), so nothing else in this scene changed.
//!
//! The headless target is `Rgba8Unorm`, like vanilla's default framebuffer: the comparison bytes
//! are the atlas bytes times the vertex colour and the lightmap, with no colour-space step in
//! between. A change to either texture's format therefore needs a regenerated reference.

use std::path::{Path, PathBuf};

use mc_math::{BlockPos, ChunkSectionPos, DVec3};
use mc_render::terrain::{fixture, mesh_section, TerrainPhase};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, RenderError};
use mc_world::World;

const WIDTH: u32 = 256;
const HEIGHT: u32 = 192;

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

/// Bake the vanilla jar, install it, and hand back the atlas. `None` means "skip".
fn vanilla_atlas() -> Option<mc_assets::Atlas> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        eprintln!("skip: 1.8.9.jar not found; set MC_JARS_DIR");
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
    mc_render::terrain::tint::load_colormaps(&rm).expect("colormaps");
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
            // Turn wgpu validation errors into test failures instead of silent no-draws.
            g.device.on_uncaptured_error(std::sync::Arc::new(|e| {
                panic!("wgpu validation error: {e}")
            }));
            Some(g)
        }
        Err(RenderError::NoAdapter) => {
            eprintln!("skip: no wgpu adapter available on this machine");
            None
        }
        Err(e) => panic!("headless context: {e}"),
    }
}

/// The scene every test in this file renders: a stone floor with a grass strip, a glass wall,
/// a leaf canopy, a water pool and a stair, all inside one section so the whole thing is one
/// `mesh_section` call.
fn scene() -> World {
    let mut w = fixture::blank_world(1);
    let stone = fixture::state_of("minecraft:stone");
    let grass = fixture::state_of("minecraft:grass");
    let glass = fixture::state_of("minecraft:glass");
    let leaves = fixture::state_of("minecraft:leaves");
    let water = fixture::state_of("minecraft:water");
    let log = fixture::state_of("minecraft:log");

    // Floor across the whole section.
    fixture::fill(
        &mut w,
        BlockPos::new(0, 16, 0),
        BlockPos::new(15, 16, 15),
        stone,
    );
    // Grass strip.
    fixture::fill(
        &mut w,
        BlockPos::new(0, 16, 0),
        BlockPos::new(15, 16, 5),
        grass,
    );
    // A glass wall.
    fixture::fill(
        &mut w,
        BlockPos::new(2, 17, 8),
        BlockPos::new(6, 20, 8),
        glass,
    );
    // A tree: log trunk with a leaf canopy.
    fixture::fill(
        &mut w,
        BlockPos::new(11, 17, 11),
        BlockPos::new(11, 21, 11),
        log,
    );
    fixture::fill(
        &mut w,
        BlockPos::new(9, 22, 9),
        BlockPos::new(13, 23, 13),
        leaves,
    );
    // A shallow pool.
    fixture::fill(
        &mut w,
        BlockPos::new(1, 16, 12),
        BlockPos::new(4, 16, 15),
        water,
    );
    // A stair, hazard H12's canary, in a spot the camera sees.
    let stair = mc_data::BlockState::new(
        mc_data::block_by_name("minecraft:oak_stairs").expect("oak_stairs"),
        0,
    );
    w.set_block(BlockPos::new(8, 17, 3), stair);
    w
}

fn camera() -> Camera {
    let mut c = Camera {
        // Outside the section, looking north-east and down at the scene.
        pos: DVec3::new(8.0, 26.0, -12.0),
        yaw: 0.0,    // vanilla yaw 0 faces +Z
        pitch: 25.0, // positive looks down
        fov: 70.0,
        near: 0.05,
        far: 256.0,
        aspect: WIDTH as f32 / HEIGHT as f32,
        ..Camera::default()
    };
    c.update();
    c
}

/// Render the scene once and return the RGBA8 pixels.
fn render(gpu: &mut Gpu, atlas: &mc_assets::Atlas) -> Vec<u8> {
    let world = scene();
    let cam = camera();

    let mut phases = PhaseList::new();
    // A flat mid-grey background, so "did anything draw" is unambiguous.
    phases.clear_color = wgpu::Color {
        r: 0.10,
        g: 0.10,
        b: 0.10,
        a: 1.0,
    };
    let shared = TerrainPhase::install(gpu, atlas, &mut phases);

    {
        let mut r = shared.borrow_mut();
        r.render_distance = 8;
        for y in 0..3 {
            let mesh = mesh_section(&world, ChunkSectionPos::new(0, y, 0));
            r.upload(gpu, mesh, cam.pos);
        }
    }

    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &cam, 0.0);
    gpu.read_pixels().expect("readback")
}

fn pixel(px: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * WIDTH + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2], px[i + 3]]
}

fn is_background(p: [u8; 4]) -> bool {
    p[0].abs_diff(26) <= 3 && p[1].abs_diff(26) <= 3 && p[2].abs_diff(26) <= 3
}

// ---- 1. structural ------------------------------------------------------------------------

#[test]
fn the_scene_actually_draws_geometry() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };
    let px = render(&mut gpu, &atlas);
    assert_eq!(px.len() as u32, WIDTH * HEIGHT * 4);

    let drawn = px
        .chunks_exact(4)
        .filter(|p| !is_background([p[0], p[1], p[2], p[3]]))
        .count();
    let total = (WIDTH * HEIGHT) as usize;
    eprintln!("terrain covers {drawn}/{total} pixels");
    // One 16x16 section seen from 20 blocks away at a 70 degree FOV occupies roughly a seventh
    // of the frame; everything else is empty space above and beside it.
    assert!(
        drawn > total / 12,
        "terrain covered only {drawn}/{total} pixels; the scene should be plainly visible"
    );

    // The corners look past the section into empty space.
    assert!(
        is_background(pixel(&px, 2, 2)),
        "the top-left corner should be sky, got {:?}",
        pixel(&px, 2, 2)
    );
    assert!(is_background(pixel(&px, WIDTH - 3, 2)));
}

/// Vanilla yaw 0 faces +Z (south), and facing south puts east (+X) on your **left**. Getting
/// this backwards mirrors the whole world, which is invisible in a symmetric test scene, so it
/// gets its own asymmetric one: a single block far to the +X side must land left of centre.
#[test]
fn positive_x_is_on_the_left_at_yaw_zero() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };

    let mut world = fixture::blank_world(1);
    fixture::fill(
        &mut world,
        BlockPos::new(14, 16, 6),
        BlockPos::new(15, 20, 8),
        fixture::state_of("minecraft:stone"),
    );

    let cam = camera();
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: 0.1,
        g: 0.1,
        b: 0.1,
        a: 1.0,
    };
    let shared = TerrainPhase::install(&gpu, &atlas, &mut phases);
    shared.borrow_mut().upload(
        &gpu,
        mesh_section(&world, ChunkSectionPos::new(0, 1, 0)),
        cam.pos,
    );
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    let px = gpu.read_pixels().expect("readback");

    let count = |x0: u32, x1: u32| {
        (0..HEIGHT)
            .flat_map(|y| (x0..x1).map(move |x| (x, y)))
            .filter(|&(x, y)| !is_background(pixel(&px, x, y)))
            .count()
    };
    let left = count(0, WIDTH / 2);
    let right = count(WIDTH / 2, WIDTH);
    assert!(left + right > 100, "the pillar did not render at all");
    assert!(
        left > right * 4,
        "a pillar at x = 14..15 must appear on the LEFT at yaw 0 (left {left}, right {right})"
    );
}

#[test]
fn the_grass_strip_is_visibly_tinted_green() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };
    let px = render(&mut gpu, &atlas);

    // The near half of the floor is the grass strip. Somewhere in the lower part of the frame
    // there must be a pixel whose green channel clearly dominates — that is the biome tint,
    // and it is absent if `tint_index` or the colormap lookup is wrong.
    let mut best = 0i32;
    for y in HEIGHT / 2..HEIGHT {
        for x in 0..WIDTH {
            let p = pixel(&px, x, y);
            let dominance = i32::from(p[1]) - i32::from(p[0]).max(i32::from(p[2]));
            best = best.max(dominance);
        }
    }
    // `grass_top.png` is grayscale, so an untinted top stays channel-equal here (the noon
    // lightmap's red and green terms are equal) and a clear margin proves the colormap reached
    // the framebuffer. The current frame reaches 28. While the atlas was sampled as
    // `Rgba8UnormSrgb` (see the snapshot history above) the tint was applied to linearised
    // texels and the frame reached only 16, so this canary had been lowered to 8; it is back at
    // its original 20, which that colour-space bug now fails.
    assert!(
        best > 20,
        "no clearly green pixel found (best green dominance {best}); biome tint is missing"
    );
}

#[test]
fn rendering_the_same_scene_twice_is_deterministic() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };
    let a = render(&mut gpu, &atlas);
    let b = render(&mut gpu, &atlas);
    assert_eq!(a, b, "the same scene rendered twice differed");
}

// ---- 2. snapshot ---------------------------------------------------------------------------

fn snapshot_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("snapshots")
        .join("s11_terrain.png")
}

#[test]
fn the_frame_matches_the_committed_snapshot() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };
    let px = render(&mut gpu, &atlas);

    let path = snapshot_path();
    if !path.is_file() {
        std::fs::create_dir_all(path.parent().expect("parent")).expect("create snapshots dir");
        image::save_buffer(&path, &px, WIDTH, HEIGHT, image::ColorType::Rgba8)
            .expect("write reference snapshot");
        eprintln!(
            "NOTE: wrote a new reference snapshot to {}. Commit it; subsequent runs compare \
             against it.",
            path.display()
        );
        return;
    }

    let reference = image::open(&path)
        .expect("open reference snapshot")
        .to_rgba8();
    assert_eq!(
        (reference.width(), reference.height()),
        (WIDTH, HEIGHT),
        "reference snapshot has the wrong size; delete it to regenerate"
    );

    // A handful of pixels may differ by a bit or two between driver versions on the edges of
    // alpha-tested geometry; a wholesale change is what this is looking for.
    let mut differing = 0usize;
    let mut worst = 0u8;
    for (i, r) in reference.as_raw().chunks_exact(4).enumerate() {
        let a = &px[i * 4..i * 4 + 4];
        let d = (0..4).map(|c| a[c].abs_diff(r[c])).max().unwrap_or(0);
        worst = worst.max(d);
        if d > 4 {
            differing += 1;
        }
    }
    let total = (WIDTH * HEIGHT) as usize;
    if differing * 200 > total {
        let out = path.with_extension("actual.png");
        image::save_buffer(&out, &px, WIDTH, HEIGHT, image::ColorType::Rgba8)
            .expect("write actual");
        panic!(
            "snapshot mismatch: {differing}/{total} pixels differ (worst channel delta {worst}). \
             Wrote the actual frame to {}. If the change is intended, delete the reference and \
             re-run.",
            out.display()
        );
    }
    eprintln!("snapshot matched ({differing}/{total} pixels differ, worst delta {worst})");
}

// ---- 3. the phases and the culling walk actually run --------------------------------------

#[test]
fn the_phase_list_registers_all_three_terrain_orders() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(gpu) = gpu() else { return };
    let mut phases = PhaseList::new();
    TerrainPhase::install(&gpu, &atlas, &mut phases);
    assert_eq!(
        phases.names().collect::<Vec<_>>(),
        ["terrain/solid", "terrain/cutout", "terrain/translucent"]
    );
}

#[test]
fn culling_reports_fewer_visible_sections_than_are_loaded() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };
    let world = scene();
    let cam = camera();
    let mut phases = PhaseList::new();
    let shared = TerrainPhase::install(&gpu, &atlas, &mut phases);
    {
        let mut r = shared.borrow_mut();
        r.render_distance = 8;
        for cx in -1..=1 {
            for cz in -1..=1 {
                for y in 0..16 {
                    let mesh = mesh_section(&world, ChunkSectionPos::new(cx, y, cz));
                    r.upload(&gpu, mesh, cam.pos);
                }
            }
        }
        assert_eq!(r.loaded_sections(), 9 * 16);
    }

    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &cam, 0.0);

    let forward = shared.borrow().stats();
    eprintln!("looking at the scene: {forward:?}");
    assert_eq!(forward.loaded_sections, 9 * 16);
    assert!(forward.visible_sections > 0, "nothing was submitted");
    assert!(forward.quads > 0, "no quads were drawn");

    // Turn the camera around: the same world, but the scene is now behind it, so the frustum
    // must reject nearly all of it. This is the decisive check that culling is doing work —
    // an unconditional "submit everything loaded" would give identical counts.
    let mut behind = camera();
    behind.yaw = 180.0;
    behind.update();
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &behind, 0.0);
    let back = shared.borrow().stats();
    eprintln!("looking away: {back:?}");
    assert!(
        back.visible_sections < forward.visible_sections,
        "turning 180 degrees did not reduce the submitted set ({} vs {})",
        back.visible_sections,
        forward.visible_sections
    );
}

#[test]
fn the_block_overlay_phase_builds_and_records() {
    let Some(atlas) = vanilla_atlas() else {
        return;
    };
    let Some(mut gpu) = gpu() else { return };
    let mut overlay = mc_render::terrain::BlockOverlayPhase::new(&gpu, Some(&atlas));
    assert!(
        overlay.cracks_available(),
        "the breaking overlay should be enabled once the atlas is supplied"
    );
    let stone = fixture::state_of("minecraft:stone");
    overlay.selection = Some(mc_render::terrain::SelectionBox::for_block(
        BlockPos::new(8, 17, 8),
        stone,
    ));
    overlay.breaking = vec![(
        mc_render::terrain::BreakProgress {
            pos: BlockPos::new(9, 17, 8),
            stage: 5,
        },
        mc_data::block::ActualBlockState::from_stored(stone),
    )];

    let mut phases = PhaseList::new();
    phases.insert(mc_render::PhaseOrder::BLOCK_OVERLAY, Box::new(overlay));
    let cam = camera();
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    let px = gpu.read_pixels().expect("readback");
    assert_eq!(px.len() as u32, WIDTH * HEIGHT * 4);
}

// The gap test that used to live here —
// `the_fluid_textures_are_known_to_be_missing_from_the_atlas` — has been **inverted and moved**
// to `tests/s11_fluid.rs`. `bake_all` now stitches `ModelBakery.LOCATIONS_BUILTIN_TEXTURES`, so
// `water_still`/`water_flow`/`lava_still`/`lava_flow` are in the atlas and fluid UVs address
// them; that file asserts it three ways, including a GPU readback checked against the real
// `water_still` texel.
