//! The shader packs' terrain variant, on the GPU.
//!
//! * **Off costs nothing.** The same scene rendered with no shader core, with a core attached but
//!   no pack switched on, and with a pack switched on but not running (no world on screen) is
//!   byte-identical, with identical draw calls, draw items, quads and vertex-buffer binds; and the
//!   vanilla mesher mode produces byte-identical meshes. A negative control (the pack running)
//!   proves the frames differ when the variant is really used.
//! * **Shadows dim only sky light.** With the Classic config running, a floating slab casts a
//!   shadow on the floor: floor pixels in its shadow are much darker than the same pixels with the
//!   slab removed, except where block light (a torch's light level 15) reaches the floor, which
//!   keeps its brightness exactly.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use glam::{DVec3, Vec3};
use mc_math::{BlockPos, ChunkSectionPos};
use mc_render::shaderpack::{
    order, CoreConfig, LightingSettings, PackFrame, PackId, ShaderCore, ShadowPhase, ShadowSettings,
};
use mc_render::terrain::{
    fixture, mesh_section_with, ChunkMesh, MeshSettings, TerrainOptions, TerrainPhase, TerrainStats,
};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, RenderError};
use mc_world::light::SkyBlock;
use mc_world::World;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn vanilla_atlas() -> Option<&'static mc_assets::Atlas> {
    static ATLAS: OnceLock<Option<mc_assets::Atlas>> = OnceLock::new();
    ATLAS
        .get_or_init(|| {
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
        })
        .as_ref()
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
            eprintln!("skip: no wgpu adapter available on this machine");
            None
        }
        Err(e) => panic!("headless context: {e}"),
    }
}

/// The slab floating over the floor.
const SLAB: (BlockPos, BlockPos) = (BlockPos::new(0, 27, 0), BlockPos::new(7, 27, 7));
/// Air above the floor with block light 15 (as if a torch were there): the north half of the
/// shadowed area and a strip in the open.
const LIT: (BlockPos, BlockPos) = (BlockPos::new(-4, 20, 0), BlockPos::new(14, 20, 3));

/// A grass floor, water, leaves, glass, and (optionally) a slab casting a shadow.
fn scene(slab: bool) -> World {
    let mut w = fixture::blank_world(1);
    let stone = fixture::state_of("minecraft:stone");
    fixture::fill(
        &mut w,
        BlockPos::new(-16, 16, -16),
        BlockPos::new(31, 18, 31),
        stone,
    );
    fixture::fill(
        &mut w,
        BlockPos::new(-16, 19, -16),
        BlockPos::new(31, 19, 31),
        fixture::state_of("minecraft:grass"),
    );
    fixture::fill(
        &mut w,
        BlockPos::new(-12, 19, 14),
        BlockPos::new(-4, 19, 22),
        fixture::state_of("minecraft:water"),
    );
    fixture::fill(
        &mut w,
        BlockPos::new(16, 20, 10),
        BlockPos::new(18, 22, 12),
        fixture::state_of("minecraft:leaves"),
    );
    fixture::fill(
        &mut w,
        BlockPos::new(20, 20, 14),
        BlockPos::new(24, 22, 14),
        fixture::state_of("minecraft:glass"),
    );
    if slab {
        fixture::fill(&mut w, SLAB.0, SLAB.1, stone);
    }
    for x in LIT.0.x..=LIT.1.x {
        for z in LIT.0.z..=LIT.1.z {
            w.set_light_for(SkyBlock::Block, BlockPos::new(x, LIT.0.y, z), 15);
        }
    }
    w
}

fn sections() -> Vec<ChunkSectionPos> {
    (-1..=1)
        .flat_map(|x| {
            (-1..=1).flat_map(move |z| (0..3).map(move |y| ChunkSectionPos::new(x, y, z)))
        })
        .collect()
}

fn meshes(world: &World, settings: MeshSettings) -> Vec<ChunkMesh> {
    sections()
        .into_iter()
        .map(|pos| mesh_section_with(world, pos, settings))
        .collect()
}

fn camera() -> Camera {
    let mut c = Camera {
        pos: DVec3::new(4.0, 40.0, -14.0),
        yaw: 0.0,
        pitch: 50.0,
        fov: 70.0,
        near: 0.05,
        far: 256.0,
        aspect: WIDTH as f32 / HEIGHT as f32,
        ..Camera::default()
    };
    c.update();
    c
}

/// The sun ~25 degrees from the zenith, in the west (shadows fall toward +x), on a path tilted
/// `SA_TEST_TILT` degrees toward the south (0 by default).
fn sun() -> Vec3 {
    let theta = 25f32.to_radians();
    mc_render::shaderpack::frame::sun_path_tilt(test_tilt())
        * Vec3::new(-theta.sin(), theta.cos(), 0.0)
}

/// `SA_TEST_TILT`, degrees (0 by default). The core config carries the same tilt, as the client's
/// does: the shadow map quantises the light along the configured path.
fn test_tilt() -> f32 {
    std::env::var("SA_TEST_TILT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0)
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Pack {
    /// No core attached to the terrain renderer.
    NoCore,
    /// A core attached, no pack switched on.
    Idle,
    /// Classic switched on, but no world on screen (`in_world` false).
    NotRunning,
    /// Classic running with shadows and sun shading.
    Running,
    /// Classic running with every terrain feature: shadows, shading, water, waving.
    Full,
    /// A pack running with exactly this core config.
    Custom(CoreConfig),
}

struct Frame {
    pixels: Vec<u8>,
    stats: TerrainStats,
    shadow_renders: u64,
}

fn classic() -> CoreConfig {
    CoreConfig {
        sun_tilt: test_tilt(),
        shadows: Some(ShadowSettings {
            distance: 48.0,
            ..ShadowSettings::default()
        }),
        lighting: Some(LightingSettings::default()),
        ..CoreConfig::OFF
    }
}

fn full() -> CoreConfig {
    CoreConfig {
        water: Some(mc_render::shaderpack::WaterSettings::default()),
        waving: Some(mc_render::shaderpack::WavingSettings::default()),
        ..classic()
    }
}

fn render(gpu: &mut Gpu, atlas: &mc_assets::Atlas, pack: Pack, slab: bool) -> Frame {
    let world = scene(slab);
    let cam = camera();
    let mut phases = PhaseList::new();
    let shared = TerrainPhase::install_with(gpu, atlas, TerrainOptions::default(), &mut phases);
    let core = ShaderCore::new();
    let config = match pack {
        Pack::Full => full(),
        Pack::Custom(config) => config,
        _ => classic(),
    };
    let running = matches!(pack, Pack::Running | Pack::Full | Pack::Custom(_));
    let mesh = if running {
        MeshSettings {
            pack: config.mesh(),
            ..MeshSettings::default()
        }
    } else {
        MeshSettings::default()
    };
    if pack != Pack::NoCore {
        shared.borrow_mut().set_shader_core(Some(core.clone()));
        phases.insert(
            order::SHADOW,
            Box::new(ShadowPhase::new(core.clone(), shared.clone())),
        );
    }
    if matches!(
        pack,
        Pack::NotRunning | Pack::Running | Pack::Full | Pack::Custom(_)
    ) {
        core.set_active(Some(PackId::Classic));
        core.set_config(config);
    }
    if pack != Pack::NoCore {
        let frame = PackFrame {
            in_world: running,
            sun_dir: sun(),
            moon_dir: -sun(),
            light_dir: sun(),
            light_is_sun: true,
            light_strength: 1.0,
            day: 1.0,
            eye: cam.pos,
            view: cam.view,
            proj: cam.proj,
            width: WIDTH,
            height: HEIGHT,
            ..PackFrame::default()
        };
        core.set_frame(gpu, frame);
    }
    {
        let mut r = shared.borrow_mut();
        r.render_distance = 8;
        let deferred = r.upload_all(gpu, meshes(&world, mesh), cam.pos);
        assert!(deferred.is_empty());
    }
    let frame = gpu.acquire().expect("acquire");
    let index = phases.render(gpu, frame.view(), &cam, 0.0);
    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .expect("poll");
    drop(frame);
    let pixels = gpu.read_pixels().expect("read pixels");
    let r = shared.borrow();
    Frame {
        pixels,
        stats: r.stats(),
        shadow_renders: r.shadow_pass_stats().1,
    }
}

/// Write a frame to `$SA_PNG_DIR/<name>.png` when that variable is set (for looking at it).
fn save(name: &str, pixels: &[u8]) {
    let Some(dir) = std::env::var_os("SA_PNG_DIR") else {
        return;
    };
    let path = PathBuf::from(dir).join(format!("{name}.png"));
    image::save_buffer(&path, pixels, WIDTH, HEIGHT, image::ColorType::Rgba8).expect("save png");
}

/// The pixel a world point lands on.
fn pixel_of(cam: &Camera, world: DVec3) -> (u32, u32) {
    let rel = (world - cam.pos).as_vec3();
    let clip = cam.view_proj() * rel.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    let x = ((ndc.x * 0.5 + 0.5) * WIDTH as f32) as u32;
    let y = ((0.5 - ndc.y * 0.5) * HEIGHT as f32) as u32;
    (x.min(WIDTH - 1), y.min(HEIGHT - 1))
}

/// Mean luma of a 5x5 patch around a world point on the floor's top face.
fn floor_luma(pixels: &[u8], x: f64, z: f64) -> f32 {
    let (px, py) = pixel_of(&camera(), DVec3::new(x, 20.0, z));
    let mut sum = 0.0;
    let mut n = 0.0;
    for dy in -2i32..=2 {
        for dx in -2i32..=2 {
            let xx = (px as i32 + dx).clamp(0, WIDTH as i32 - 1) as usize;
            let yy = (py as i32 + dy).clamp(0, HEIGHT as i32 - 1) as usize;
            let i = (yy * WIDTH as usize + xx) * 4;
            let p = &pixels[i..i + 4];
            sum += 0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2]);
            n += 1.0;
        }
    }
    sum / n
}

#[test]
fn with_no_pack_running_the_terrain_frame_is_byte_identical() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };
    let base = render(&mut gpu, atlas, Pack::NoCore, true);
    let non_clear = base
        .pixels
        .chunks_exact(4)
        .filter(|p| p[0] != base.pixels[0] || p[1] != base.pixels[1])
        .count();
    assert!(non_clear > (WIDTH * HEIGHT / 3) as usize, "the frame drew");
    for pack in [Pack::Idle, Pack::NotRunning] {
        let frame = render(&mut gpu, atlas, pack, true);
        assert!(frame.pixels == base.pixels, "{pack:?}: pixels differ");
        assert_eq!(frame.stats, base.stats, "{pack:?}: draw counts differ");
        assert_eq!(frame.shadow_renders, 0, "{pack:?}: no shadow pass");
    }
    // Negative control: the pack running really changes the frame and draws the shadow map.
    let on = render(&mut gpu, atlas, Pack::Running, true);
    assert!(
        on.pixels != base.pixels,
        "the running pack must change the frame"
    );
    assert_eq!(on.shadow_renders, 1, "one shadow render for a still scene");
    assert_eq!(
        (on.stats.draw_calls, on.stats.quads),
        (base.stats.draw_calls, base.stats.quads),
        "the variant draws the same geometry"
    );
    println!(
        "pack off: draw calls {}, items {}, quads {}, binds {}",
        base.stats.draw_calls,
        base.stats.draw_items,
        base.stats.quads,
        base.stats.vertex_buffer_binds
    );
}

#[test]
fn the_vanilla_mesher_mode_writes_identical_meshes() {
    let Some(_atlas) = vanilla_atlas() else {
        return;
    };
    let world = scene(true);
    let plain = meshes(&world, MeshSettings::default());
    let explicit = meshes(
        &world,
        MeshSettings {
            pack: mc_render::shaderpack::PackMesh::VANILLA,
            ..MeshSettings::default()
        },
    );
    assert!(plain == explicit);
    // Negative control: the pack mode does change the bytes (flat shade, flags).
    let packed = meshes(
        &world,
        MeshSettings {
            pack: classic().mesh(),
            ..MeshSettings::default()
        },
    );
    assert!(plain != packed);
}

#[test]
fn shadows_dim_sky_light_and_leave_block_light_alone() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };
    let shadowed = render(&mut gpu, atlas, Pack::Running, true);
    let open = render(&mut gpu, atlas, Pack::Running, false);
    let vanilla = render(&mut gpu, atlas, Pack::NoCore, true);
    save("terrain_pack_shadowed", &shadowed.pixels);
    save("terrain_pack_open", &open.pixels);
    save("terrain_vanilla", &vanilla.pixels);
    // The shadow falls toward +x: the slab's footprint (x 0..8) shifted by ~7 * tan(25) = 3.3
    // blocks. Sample at x = 10, outside the slab's own footprint, where vanilla's sky light is a
    // full 15 (under the slab vanilla's sky light is already lower, which would muddle this).
    let in_shadow_dark = (10.0, 6.0); // z = 6: no block light there
    let in_shadow_lit = (10.0, 1.5); // z = 1.5: block light 15
    let dark_with = floor_luma(&shadowed.pixels, in_shadow_dark.0, in_shadow_dark.1);
    let dark_without = floor_luma(&open.pixels, in_shadow_dark.0, in_shadow_dark.1);
    let lit_with = floor_luma(&shadowed.pixels, in_shadow_lit.0, in_shadow_lit.1);
    let lit_without = floor_luma(&open.pixels, in_shadow_lit.0, in_shadow_lit.1);
    println!(
        "sky-lit floor: {dark_without:.1} open, {dark_with:.1} in shadow; \
         block-lit floor: {lit_without:.1} open, {lit_with:.1} in shadow"
    );
    assert!(
        dark_with < dark_without * 0.9,
        "a shadow darkens the sky-lit floor ({dark_with} vs {dark_without})"
    );
    assert!(
        (lit_with - lit_without).abs() <= 2.0,
        "block light is untouched by the shadow ({lit_with} vs {lit_without})"
    );
    // And far from the slab both frames agree (the shadow is local).
    let far_with = floor_luma(&shadowed.pixels, -10.0, 8.0);
    let far_without = floor_luma(&open.pixels, -10.0, 8.0);
    assert!((far_with - far_without).abs() <= 2.0);
}

#[test]
fn shader_water_reads_the_scene_through_a_read_only_depth_pass() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };
    let full = render(&mut gpu, atlas, Pack::Full, true);
    let lit = render(&mut gpu, atlas, Pack::Running, true);
    let vanilla = render(&mut gpu, atlas, Pack::NoCore, true);
    save("terrain_pack_full", &full.pixels);
    // The pool (x -12..-4, z 14..22): shader water differs from the plain pack's water.
    let water_full = floor_luma(&full.pixels, -8.0, 18.0);
    let water_lit = floor_luma(&lit.pixels, -8.0, 18.0);
    let water_vanilla = floor_luma(&vanilla.pixels, -8.0, 18.0);
    println!(
        "water luma: vanilla {water_vanilla:.1}, lit {water_lit:.1}, shader water {water_full:.1}"
    );
    let (px, py) = pixel_of(&camera(), DVec3::new(-8.0, 20.0, 18.0));
    let i = (py as usize * WIDTH as usize + px as usize) * 4;
    let j = i;
    assert!(
        full.pixels[i..i + 4] != lit.pixels[j..j + 4],
        "shader water must change the pool"
    );
    // Same geometry either way.
    assert_eq!(full.stats.quads, lit.stats.quads);
}

/// The largest per-channel difference between two frames, and how many pixels differ by more
/// than `tolerance`.
fn frame_difference(a: &[u8], b: &[u8], tolerance: u8) -> (u8, usize) {
    let mut max = 0u8;
    let mut over = 0usize;
    for (p, q) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let d = (0..3).map(|i| p[i].abs_diff(q[i])).max().unwrap_or(0);
        max = max.max(d);
        if d > tolerance {
            over += 1;
        }
    }
    (max, over)
}

/// Mean luma of a 5x5 patch around a world point.
fn luma_at(pixels: &[u8], world: DVec3) -> f32 {
    let (px, py) = pixel_of(&camera(), world);
    let mut sum = 0.0;
    for dy in -2i32..=2 {
        for dx in -2i32..=2 {
            let x = (px as i32 + dx).clamp(0, WIDTH as i32 - 1) as usize;
            let y = (py as i32 + dy).clamp(0, HEIGHT as i32 - 1) as usize;
            let i = (y * WIDTH as usize + x) * 4;
            sum += 0.2126 * f32::from(pixels[i])
                + 0.7152 * f32::from(pixels[i + 1])
                + 0.0722 * f32::from(pixels[i + 2]);
        }
    }
    sum / 25.0
}

/// Shadows without sun shading (what a pack that wants only shadows gets): at strength 0 the
/// frame is vanilla's (only rounding from the linear-light round trip differs); at full strength
/// the floor in the slab's cast shadow darkens while a wall turned away from the sun (the slab's
/// north face) keeps all of its sky light. Before the fix every such wall lost 55%.
#[test]
fn shadows_alone_take_away_only_the_cast_shadow() {
    let Some(atlas) = vanilla_atlas() else { return };
    let Some(mut gpu) = gpu() else { return };
    let vanilla = render(&mut gpu, atlas, Pack::NoCore, true);
    let shadows = |strength: f32| CoreConfig {
        shadows: Some(ShadowSettings {
            distance: 48.0,
            strength,
            ..ShadowSettings::default()
        }),
        ..CoreConfig::OFF
    };
    let zero = render(&mut gpu, atlas, Pack::Custom(shadows(0.0)), true);
    save("terrain_shadows_only_strength0", &zero.pixels);
    let (max, over) = frame_difference(&vanilla.pixels, &zero.pixels, 2);
    println!("shadows only, strength 0 vs vanilla: max channel difference {max}, {over} px > 2");
    assert_eq!(
        over, 0,
        "strength 0 must be the unshadowed frame (max difference {max})"
    );

    let full = render(&mut gpu, atlas, Pack::Custom(shadows(1.0)), true);
    save("terrain_shadows_only_strength1", &full.pixels);
    // The slab's north face (z = 0, facing the camera; the sun is in the west and above, so the
    // face is turned away from it).
    let wall = DVec3::new(4.0, 27.5, -0.01);
    let wall_vanilla = luma_at(&vanilla.pixels, wall);
    let wall_full = luma_at(&full.pixels, wall);
    // The floor inside the cast shadow (see `shadows_dim_sky_light_and_leave_block_light_alone`).
    let floor = DVec3::new(10.0, 20.0, 6.0);
    let floor_vanilla = luma_at(&vanilla.pixels, floor);
    let floor_full = luma_at(&full.pixels, floor);
    println!(
        "wall turned away from the sun: vanilla {wall_vanilla:.1}, shadows {wall_full:.1}; \
         floor in the cast shadow: vanilla {floor_vanilla:.1}, shadows {floor_full:.1}"
    );
    assert!(
        (wall_full - wall_vanilla).abs() <= 2.0,
        "a wall turned away from the sun keeps its sky light"
    );
    assert!(
        floor_full < floor_vanilla * 0.85,
        "the cast shadow darkens the floor ({floor_full} vs {floor_vanilla})"
    );
    // ...and only there: the open floor far from the slab keeps vanilla's brightness. (This
    // catches a receiver sampling an empty or stale map, which shadows everything.)
    let open = DVec3::new(-10.0, 20.0, 8.0);
    let open_vanilla = luma_at(&vanilla.pixels, open);
    let open_full = luma_at(&full.pixels, open);
    println!("open floor: vanilla {open_vanilla:.1}, shadows {open_full:.1}");
    assert!(
        (open_full - open_vanilla).abs() <= 2.0,
        "the open floor is not shadowed ({open_full} vs {open_vanilla})"
    );
}
