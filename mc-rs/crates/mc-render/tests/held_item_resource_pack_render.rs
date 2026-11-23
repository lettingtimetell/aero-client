//! The held items under a resource pack's item textures.
//!
//! A held item's shape is extruded from its sprite's opaque texels, so a pack that redraws a
//! sword moves and resizes the sword in the hand (vanilla does the same with the same pack). This
//! renders B1's eight `HeldItemRef` cases through the real `FirstPersonPhase` with the vanilla jar
//! plus the pack folder named by `MC_HELD_ITEM_PACK` stacked on top, and prints each coverage
//! mask. `MC_HELD_ITEM_PACK_DUMP=<dir>` also writes the renders as PNGs, to compare with
//! `generate-held-items.ps1 -Pack <dir>` (the real Java renderer with the same pack).
//!
//! It needs its own test binary: `mc_assets::install` is process-global, and the other held-item
//! tests install the vanilla bake. Without `MC_HELD_ITEM_PACK` it skips.

use std::path::{Path, PathBuf};

use mc_assets::{Atlas, BakedModel, FolderResourcePack, ResourceManager, ZipResourcePack};
use mc_render::pipeline::{FirstPersonPhase, FirstPersonUseAction};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

const W: u32 = 1920;
const H: u32 = 1080;
const MAX_ITEM_USE_DURATION: i32 = 72_000;

/// B1's `HeldItemRef` cases: name, item, swing, use count.
const CASES: &[(&str, &str, f32, i32)] = &[
    ("sword_idle", "diamond_sword", 0.0, 0),
    ("sword_mid_swing", "diamond_sword", 0.5, 0),
    ("sword_blocking", "diamond_sword", 0.0, 71_995),
    ("stone_block_idle", "stone", 0.0, 0),
    ("bow_idle", "bow", 0.0, 0),
    ("bow_drawn", "bow", 0.0, 71_980),
    ("apple_idle", "apple", 0.0, 0),
    ("fishing_rod_idle", "fishing_rod", 0.0, 0),
];

fn jar() -> Option<PathBuf> {
    let dir = match std::env::var_os("MC_JARS_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)?
            .join("MCP-919-main")
            .join("jars"),
    };
    let jar = dir.join("versions").join("1.8.9").join("1.8.9.jar");
    jar.is_file().then_some(jar)
}

fn held(item: &str, use_count: i32) -> (BakedModel, FirstPersonUseAction) {
    let id = mc_data::item_by_name(item).expect("a real 1.8.9 item");
    match (item, use_count) {
        (_, 0) => (
            mc_assets::item_model_for(id).clone(),
            FirstPersonUseAction::Idle,
        ),
        ("bow", _) => (
            mc_assets::item_model_for_variant(id, 3).clone(),
            FirstPersonUseAction::Bow,
        ),
        _ => (
            mc_assets::item_model_for(id).clone(),
            FirstPersonUseAction::Block,
        ),
    }
}

#[test]
fn held_items_render_under_a_resource_pack() {
    let Some(pack) = std::env::var_os("MC_HELD_ITEM_PACK").map(PathBuf::from) else {
        eprintln!("SKIP: set MC_HELD_ITEM_PACK to a resource-pack folder");
        return;
    };
    let Some(jar) = jar() else {
        eprintln!("SKIP: no vanilla jar; set MC_JARS_DIR");
        return;
    };
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).expect("the jar")));
    rm.push_pack(Box::new(
        FolderResourcePack::open(&pack).expect("the pack folder"),
    ));
    let (baked, _report) = mc_assets::bake_all(&rm, 4096).expect("bake with the pack");
    let atlas: Atlas = baked.atlas.clone();
    mc_assets::install(baked);

    let mut gpu = match Gpu::headless(
        W,
        H,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => gpu,
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            return;
        }
        Err(error) => panic!("headless context creation failed: {error}"),
    };
    let dump = std::env::var_os("MC_HELD_ITEM_PACK_DUMP").map(PathBuf::from);
    for &(name, item, swing, use_count) in CASES {
        let (model, action) = held(item, use_count);
        let gui_3d = model.gui_3d;
        let mut phase = FirstPersonPhase::new(&gpu);
        phase.set_atlas(&gpu, &atlas);
        phase.set_item_model(Some(model), gui_3d);
        phase.render_state.prev_equipped_progress = 1.0;
        phase.render_state.equipped_progress = 1.0;
        phase.render_state.prev_swing_progress = swing;
        phase.render_state.swing_progress = swing;
        phase.render_state.use_action = action;
        phase.render_state.item_use_count = use_count;
        phase.render_state.max_item_use_duration = MAX_ITEM_USE_DURATION;
        phase.partial_ticks = 0.0;
        phase.combined_light = mc_render::entity::render::FULL_BRIGHT_COMBINED_LIGHT;
        let mut camera = Camera::default();
        camera.set_viewport(W, H);
        camera.update();
        let mut phases = PhaseList::new();
        phases.clear_color = wgpu::Color {
            r: 1.0,
            g: 0.0,
            b: 1.0,
            a: 1.0,
        };
        phases.insert(PhaseOrder::FIRST_PERSON, Box::new(phase));
        let frame = gpu.acquire().expect("acquire offscreen frame");
        phases.render(&gpu, frame.view(), &camera, 0.0);
        let pixels = gpu.read_pixels().expect("readback");
        if let Some(dir) = &dump {
            std::fs::create_dir_all(dir).expect("dump dir");
            image::RgbaImage::from_raw(W, H, pixels.clone())
                .expect("rgba frame")
                .save(dir.join(format!("{name}.png")))
                .expect("write png");
        }
        let (mut count, mut sx, mut sy) = (0_u64, 0.0_f64, 0.0_f64);
        let (mut x0, mut x1, mut y0, mut y1) = (W, 0, H, 0);
        for (index, p) in pixels.chunks_exact(4).enumerate() {
            if p[0] == 255 && p[1] == 0 && p[2] == 255 {
                continue;
            }
            let (x, y) = (index as u32 % W, index as u32 / W);
            count += 1;
            sx += f64::from(x) + 0.5;
            sy += f64::from(y) + 0.5;
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
        }
        assert!(count > 0, "{name}: nothing drawn");
        eprintln!(
            "{name:18} n {count:6} x {x0:4}..{x1:4} y {y0:4}..{y1:4} c ({:8.2}, {:7.2})",
            sx / count as f64,
            sy / count as f64
        );
    }
}
