//! Item 44 ("layers of translucent blocks aren't one to one
//! with vanilla"): translucent terrain fogged at its own depth.
//!
//! java: `renderWorldPass` fogs every draw as it happens (`GL_FOG`), so a stained-glass fragment
//! is fogged at its own distance and then blended over the already-fogged terrain behind it:
//! `out = fog(G, d_glass) * a + fog(S, d_stone) * (1 - a)`.
//!
//! mc-rs used to run its one screen-space fog pass *after* translucent terrain, over the depth of
//! the terrain behind (translucent terrain writes no depth): `out = fog(G * a + S * (1 - a),
//! d_stone)`. A window in front of distant terrain dissolved into the fog with it, and glass
//! against the sky was never fogged. The client now registers the pass at
//! `FogPhase::BEFORE_TRANSLUCENT_ORDER` and fogs the translucent layer per draw
//! (`TerrainRenderer::set_translucent_fog`).
//!
//! The scene, seen straight down: a stone floor 20 blocks below the eye, one red stained glass
//! block 4 blocks below the eye covering about a quarter of the view, under thick fog
//! (`dimension_always_fogs`, render distance 2: linear 1.6..16 blocks), so the floor is entirely
//! fog colour and the glass only a little fogged.

use mc_math::{BlockPos, ChunkSectionPos, DVec3};
use mc_render::sky::fog::{FogDistance, FogInputs, FogPhase, FogState};
use mc_render::sky::Environment;
use mc_render::terrain::{fixture, mesh_section, TerrainPhase};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};
use std::path::{Path, PathBuf};

const SIDE: u32 = 128;
const FLOOR_Y: i32 = 20;
const GLASS_Y: i32 = 36;
const EYE_Y: f64 = 41.0;

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn fixture_or_skip() -> Option<(mc_assets::Atlas, Gpu)> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        eprintln!("SKIP: 1.8.9.jar not found; set MC_JARS_DIR");
        return None;
    }
    let mut rm = mc_assets::ResourceManager::new();
    rm.push_pack(Box::new(
        mc_assets::ZipResourcePack::open(&jar).expect("open 1.8.9.jar"),
    ));
    let (assets, _report) = mc_assets::bake_all(&rm, 2048).expect("bake_all");
    let atlas = assets.atlas.clone();
    mc_assets::model::install(assets);
    let _ = mc_render::terrain::tint::load_colormaps(&rm);
    let gpu = match Gpu::headless(
        SIDE,
        SIDE,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => gpu,
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available");
            return None;
        }
        Err(e) => panic!("headless context: {e}"),
    };
    Some((atlas, gpu))
}

fn world(stone: bool, glass: bool) -> mc_world::World {
    let mut w = fixture::blank_world(1);
    if stone {
        fixture::fill(
            &mut w,
            BlockPos::new(0, FLOOR_Y, 0),
            BlockPos::new(15, FLOOR_Y, 15),
            fixture::state_of("minecraft:stone"),
        );
    }
    if glass {
        let red = mc_data::BlockState::new(
            mc_data::block_by_name("stained_glass").expect("stained glass"),
            14,
        );
        // One block, off the view's centre: about a quarter of the view is glass, the rest is
        // the bare floor.
        fixture::fill(
            &mut w,
            BlockPos::new(8, GLASS_Y, 8),
            BlockPos::new(8, GLASS_Y, 8),
            red,
        );
    }
    w
}

fn camera() -> Camera {
    let mut c = Camera {
        pos: DVec3::new(8.0, EYE_Y, 8.0),
        yaw: 0.0,
        pitch: 90.0,
        fov: 30.0,
        near: 0.05,
        far: 256.0,
        aspect: 1.0,
        ..Camera::default()
    };
    c.update();
    c
}

fn fog_inputs() -> FogInputs {
    FogInputs {
        render_distance_chunks: 2,
        dimension_always_fogs: true,
        ..FogInputs::default()
    }
}

/// One frame. `fog_order`: where the screen-space pass records; `translucent_fog`: whether the
/// translucent layer fogs itself. `clear`: the backdrop behind everything.
fn shoot(
    atlas: &mc_assets::Atlas,
    gpu: &mut Gpu,
    (stone, glass): (bool, bool),
    fog_order: PhaseOrder,
    translucent_fog: bool,
    clear: f64,
) -> Vec<u8> {
    let w = world(stone, glass);
    let cam = camera();
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color {
        r: clear,
        g: clear,
        b: clear,
        a: 1.0,
    };
    let shared = TerrainPhase::install(gpu, atlas, &mut phases);
    let mut fog_phase = FogPhase::new(gpu);
    fog_phase.environment = Environment::default();
    fog_phase.inputs = fog_inputs();
    let state = fog_phase.state();
    phases.insert(fog_order, Box::new(fog_phase));
    {
        let mut r = shared.borrow_mut();
        r.render_distance = 8;
        if translucent_fog {
            r.set_translucent_fog(Some((state, FogDistance::Planar)));
        }
        for section_y in [1, 2] {
            let mesh = mesh_section(&w, ChunkSectionPos::new(0, section_y, 0));
            r.upload(gpu, mesh, cam.pos);
        }
    }
    let frame = gpu.acquire().expect("acquire");
    phases.render(gpu, frame.view(), &cam, 0.0);
    gpu.read_pixels().expect("readback")
}

fn rgb(px: &[u8], i: usize) -> [i32; 3] {
    [
        i32::from(px[i * 4]),
        i32::from(px[i * 4 + 1]),
        i32::from(px[i * 4 + 2]),
    ]
}

#[test]
fn translucent_terrain_is_fogged_at_its_own_depth_like_vanilla() {
    let Some((atlas, mut gpu)) = fixture_or_skip() else {
        return;
    };
    let state: FogState = FogState::compute(&Environment::default(), fog_inputs(), 0.0);
    // The floor (20 blocks) is past the fog's end; the glass's top face (4 blocks) is not.
    assert_eq!(state.mode.factor(20.0), 0.0);
    let glass_factor = state.mode.factor(4.0);
    assert!(glass_factor > 0.7 && glass_factor < 0.95, "{glass_factor}");

    let before = PhaseOrder(640); // `FogPhase::ORDER`, the old registration
    let after = FogPhase::BEFORE_TRANSLUCENT_ORDER;
    let old = shoot(&atlas, &mut gpu, (true, true), before, false, 0.0);
    let new = shoot(&atlas, &mut gpu, (true, true), after, true, 0.0);
    let stone = shoot(&atlas, &mut gpu, (true, false), after, true, 0.0);
    // The glass alone over black and over white: `fog(G) * a` and `fog(G) * a + (1 - a)`.
    let on_black = shoot(&atlas, &mut gpu, (false, true), after, true, 0.0);
    let on_white = shoot(&atlas, &mut gpu, (false, true), after, true, 1.0);

    let fog_rgb = [state.color.x, state.color.y, state.color.z].map(|c| (c * 255.0).round() as i32);
    let mut glass_pixels = 0;
    let mut old_far_off = 0;
    for i in 0..(SIDE * SIDE) as usize {
        let covered = rgb(&on_black, i) != [0, 0, 0];
        if !covered {
            // No translucent surface: the very same fog pass over the very same depth.
            assert_eq!(rgb(&old, i), rgb(&new, i), "pixel {i} without glass");
            continue;
        }
        glass_pixels += 1;
        let (g, w, s) = (rgb(&on_black, i), rgb(&on_white, i), rgb(&stone, i));
        // vanilla: fog(G) * a + fog(S) * (1 - a), with (1 - a) = (white - black) / 255.
        let expected: [i32; 3] = core::array::from_fn(|c| g[c] + s[c] * (w[c] - g[c]) / 255);
        let got = rgb(&new, i);
        for c in 0..3 {
            assert!(
                (got[c] - expected[c]).abs() <= 3,
                "pixel {i} channel {c}: got {got:?}, vanilla {expected:?}"
            );
        }
        // The old order fogged the composite by the floor's depth: pure fog colour here.
        if (0..3).any(|c| (rgb(&old, i)[c] - expected[c]).abs() > 20) {
            old_far_off += 1;
        }
        let old_px = rgb(&old, i);
        assert!(
            (0..3).all(|c| (old_px[c] - fog_rgb[c]).abs() <= 1),
            "old order at pixel {i}: {old_px:?}, the fog colour {fog_rgb:?}"
        );
    }
    eprintln!(
        "glass pixels {glass_pixels}, of which the old order got {old_far_off} visibly wrong"
    );
    assert!(
        glass_pixels > 1000,
        "the glass is on screen ({glass_pixels} px)"
    );
    assert_eq!(
        old_far_off, glass_pixels,
        "the old order lost the window entirely"
    );
}

/// Glass against the sky: the old screen-space pass discarded sky pixels, so far glass was never
/// fogged; per draw it is.
#[test]
fn glass_against_the_sky_is_fogged_too() {
    let Some((atlas, mut gpu)) = fixture_or_skip() else {
        return;
    };
    let old = shoot(&atlas, &mut gpu, (false, true), PhaseOrder(640), false, 0.0);
    let new = shoot(
        &atlas,
        &mut gpu,
        (false, true),
        FogPhase::BEFORE_TRANSLUCENT_ORDER,
        true,
        0.0,
    );
    let mut glass = 0;
    let mut changed = 0;
    for i in 0..(SIDE * SIDE) as usize {
        if rgb(&old, i) != [0, 0, 0] {
            glass += 1;
            if rgb(&old, i) != rgb(&new, i) {
                changed += 1;
            }
        } else {
            assert_eq!(rgb(&new, i), [0, 0, 0], "the sky itself is never fogged");
        }
    }
    assert!(glass > 1000, "the glass is on screen ({glass} px)");
    assert_eq!(changed, glass, "every glass pixel against the sky now fogs");
}
