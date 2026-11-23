//! Bug report: "minecraft enchanting glint on armor looks like
//! stripes".
//!
//! Renders a real diamond chestplate (the production `RenderManager::emit_armor_slot_for_entity`
//! armour model and `item::append_armor_glint`) with the real `enchanted_item_glint.png` loaded
//! through `EntityPhase::load_texture`, i.e. with its `.mcmeta` sampler. `MC_X6_GLINT_PNG=<dir>`
//! saves the frames for inspection. Needs the vanilla jar (`MC_JARS_DIR`) and a wgpu adapter;
//! skips loudly without either.

use std::path::{Path, PathBuf};

use glam::Vec3;
use mc_assets::{ResourceManager, ZipResourcePack};
use mc_render::entity::anim::AnimParams;
use mc_render::entity::layers::{ArmorSlot, LayerBlend};
use mc_render::entity::manager::ArmorPoses;
use mc_render::entity::models::ModelKind;
use mc_render::entity::render::{entity_matrix, EntityPhase, EntityTransform};
use mc_render::entity::{item, EntityVertex, RenderManager};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

const SIZE: u32 = 256;

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn resources() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    let pack = ZipResourcePack::open(jar).ok()?;
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(pack));
    Some(rm)
}

fn headless() -> Option<Gpu> {
    match Gpu::headless(
        SIZE,
        SIZE,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(g) => Some(g),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter on this machine");
            None
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    }
}

const DIAMOND_CHEST: &str = "textures/models/armor/diamond_layer_1.png";

/// One frame of a chestplate facing the camera at `age` ticks; `glint` adds the foil.
fn render_chestplate(
    gpu: &mut Gpu,
    rm: &ResourceManager,
    age: f32,
    glint: bool,
) -> Option<Vec<u8>> {
    let mut cam = Camera::default();
    cam.set_viewport(SIZE, SIZE);
    cam.update();
    let mut phase = EntityPhase::new(gpu);
    if !phase.load_texture(gpu, rm, DIAMOND_CHEST)
        || !phase.load_texture(gpu, rm, item::ENCHANTED_ITEM_GLINT_TEXTURE)
    {
        eprintln!("SKIP: armour or glint texture missing from the jar");
        return None;
    }
    let mut models = RenderManager::new();
    let params = AnimParams::still();
    // The player's feet 1.4 blocks below eye level, 1.6 blocks ahead, body turned to face us.
    let base = entity_matrix(&EntityTransform {
        rel: Vec3::new(0.0, -1.15, 1.6),
        body_yaw: 180.0,
        ..EntityTransform::default()
    });
    let mut verts: Vec<EntityVertex> = Vec::new();
    models.emit_armor_slot_for_entity(
        ModelKind::Player,
        ArmorSlot::Chestplate,
        &params,
        base,
        &mut ArmorPoses::default(),
        &mut verts,
    );
    assert!(!verts.is_empty(), "the chestplate emits geometry");
    phase.begin();
    phase.push_blended_lit(
        DIAMOND_CHEST,
        LayerBlend::Opaque,
        mc_render::entity::render::FULL_BRIGHT_COMBINED_LIGHT,
        &verts,
    );
    if glint {
        let base_len = verts.len();
        item::append_armor_glint(&mut verts, base_len, age);
        phase.push_blended_lit(
            item::ENCHANTED_ITEM_GLINT_TEXTURE,
            LayerBlend::Glint,
            mc_render::entity::render::FULL_BRIGHT_COMBINED_LIGHT,
            &verts[base_len..],
        );
    }
    phase.finish(gpu);
    let mut list = PhaseList::new();
    list.clear_color = wgpu::Color::BLACK;
    list.insert(PhaseOrder::ENTITIES, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    list.render(gpu, frame.view(), &cam, 0.0);
    frame.present(&gpu.queue);
    Some(gpu.read_pixels().expect("read back the frame"))
}

/// The oracle's quad: the chestplate front panel in ModelBiped's 64x32 texture space.
const U0: f32 = 20.0 / 64.0;
const U1: f32 = 28.0 / 64.0;
const V0: f32 = 20.0 / 32.0;
/// `32 / 32`: the panel's bottom edge is the texture's.
const V1: f32 = 1.0;
/// `ArmorGlintRef`'s frame size.
const ORACLE_SIZE: u32 = 64;

fn oracle_frame(age_x10: u32) -> Option<image::RgbImage> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("snapshots")
        .join(format!("x6_armor_glint_gl-{age_x10}.png"));
    Some(image::open(path).ok()?.to_rgb8())
}

/// The production foil (`item::append_armor_glint` + `LayerBlend::Glint`, the glint texture
/// loaded with its `.mcmeta`) on one viewport-filling quad, against fixed-function GL drawing
/// `LayerArmorBase.renderGlint`'s exact state on the same UVs (`javaref/ArmorGlintRef.java`).
#[test]
fn armor_glint_matches_fixed_function_gl_pixel_for_pixel() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: vanilla jar not found; set MC_JARS_DIR");
        return;
    };
    let mut gpu = match Gpu::headless(
        ORACLE_SIZE,
        ORACLE_SIZE,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(g) => g,
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter on this machine");
            return;
        }
        Err(e) => panic!("headless context creation failed: {e}"),
    };
    let mut cam = Camera::default();
    cam.set_viewport(ORACLE_SIZE, ORACLE_SIZE);
    cam.update();
    // A fronto-parallel quad exactly filling the frustum at depth 2.
    let depth = 2.0;
    let half_w = depth / cam.proj.x_axis.x.abs();
    let half_h = depth / cam.proj.y_axis.y.abs();
    let screen = |p: Vec3| {
        let clip = cam.view_proj() * p.extend(1.0);
        (clip.x / clip.w, clip.y / clip.w)
    };
    // Give each corner the UV of the screen corner it lands on (top of the screen is V0).
    let corner = |x: f32, y: f32| {
        let pos = Vec3::new(x, y, depth);
        let (sx, sy) = screen(pos);
        EntityVertex {
            pos: pos.to_array(),
            uv: [
                if sx > 0.0 { U1 } else { U0 },
                if sy > 0.0 { V0 } else { V1 },
            ],
            normal: [0.0, 0.0, -1.0],
            color: [1.0; 4],
        }
    };
    let (a, b, c, d) = (
        corner(-half_w, -half_h),
        corner(half_w, -half_h),
        corner(half_w, half_h),
        corner(-half_w, half_h),
    );
    let quad = [a, b, c, a, c, d];

    let mut mismatches = Vec::new();
    let mut compared = 0;
    for age_x10 in [1000u32, 1175] {
        let Some(gl) = oracle_frame(age_x10) else {
            panic!("missing oracle frame {age_x10}; run javaref/generate-armor-glint.ps1");
        };
        let mut phase = EntityPhase::new(&gpu);
        assert!(phase.load_texture(&gpu, &rm, item::ENCHANTED_ITEM_GLINT_TEXTURE));
        phase.upload_texture(&gpu, "x6/black", &[0, 0, 0, 255], 1, 1);
        phase.set_lightmap(
            &gpu,
            &mc_render::sky::lightmap::Lightmap {
                rgba: [255u8; 4].repeat(256),
            },
        );
        phase.begin();
        // The armour's own depth for GL_EQUAL, drawn black like the oracle's clear.
        phase.push_blended_lit(
            "x6/black",
            LayerBlend::Opaque,
            mc_render::entity::render::FULL_BRIGHT_COMBINED_LIGHT,
            &quad,
        );
        let mut verts = quad.to_vec();
        item::append_armor_glint(&mut verts, quad.len(), age_x10 as f32 / 10.0);
        phase.push_blended_lit(
            item::ENCHANTED_ITEM_GLINT_TEXTURE,
            LayerBlend::Glint,
            mc_render::entity::render::FULL_BRIGHT_COMBINED_LIGHT,
            &verts[quad.len()..],
        );
        phase.finish(&gpu);
        let mut list = PhaseList::new();
        list.clear_color = wgpu::Color::BLACK;
        list.insert(PhaseOrder::ENTITIES, Box::new(phase));
        let frame = gpu.acquire().expect("acquire offscreen frame");
        list.render(&gpu, frame.view(), &cam, 0.0);
        frame.present(&gpu.queue);
        let pixels = gpu.read_pixels().expect("read back the frame");
        if let Some(dir) = std::env::var_os("MC_X6_GLINT_PNG") {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir).expect("output dir");
            image::save_buffer(
                dir.join(format!("wgpu-{age_x10}.png")),
                &pixels,
                ORACLE_SIZE,
                ORACLE_SIZE,
                image::ExtendedColorType::Rgba8,
            )
            .expect("write the PNG");
        }
        for y in 0..ORACLE_SIZE {
            for x in 0..ORACLE_SIZE {
                let i = ((y * ORACLE_SIZE + x) * 4) as usize;
                let got = [pixels[i], pixels[i + 1], pixels[i + 2]];
                let want = gl.get_pixel(x, y).0;
                compared += 1;
                if got.iter().zip(&want).any(|(g, w)| g.abs_diff(*w) > 3) {
                    mismatches.push(format!(
                        "age {age_x10} ({x},{y}): wgpu {got:?}, GL {want:?}"
                    ));
                }
            }
        }
    }
    assert!(
        mismatches.len() * 100 <= compared,
        "{} of {compared} pixels differ from fixed-function GL by more than 3:\n{}",
        mismatches.len(),
        mismatches
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn enchanted_chestplate_render() {
    let Some(rm) = resources() else {
        eprintln!("SKIP: vanilla jar not found; set MC_JARS_DIR");
        return;
    };
    let Some(mut gpu) = headless() else {
        return;
    };
    let Some(plain) = render_chestplate(&mut gpu, &rm, 100.0, false) else {
        return;
    };
    let Some(foil) = render_chestplate(&mut gpu, &rm, 100.0, true) else {
        return;
    };
    if let Some(dir) = std::env::var_os("MC_X6_GLINT_PNG") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("output dir");
        for (name, pixels) in [("plain.png", &plain), ("glint.png", &foil)] {
            image::save_buffer(
                dir.join(name),
                pixels,
                SIZE,
                SIZE,
                image::ExtendedColorType::Rgba8,
            )
            .expect("write the PNG");
        }
    }
    let brighter = plain
        .chunks_exact(4)
        .zip(foil.chunks_exact(4))
        .filter(|(a, b)| b[0] > a[0] || b[2] > a[2])
        .count();
    assert!(
        brighter > 100,
        "the foil brightens the chestplate: {brighter} pixels"
    );
    // A strip of frames, a tick apart, for eyeballing the motion.
    if let Some(dir) = std::env::var_os("MC_X6_GLINT_PNG") {
        let dir = PathBuf::from(dir);
        for step in 0..8 {
            let age = 100.0 + step as f32 * 5.0;
            if let Some(frame) = render_chestplate(&mut gpu, &rm, age, true) {
                image::save_buffer(
                    dir.join(format!("glint-{step}.png")),
                    &frame,
                    SIZE,
                    SIZE,
                    image::ExtendedColorType::Rgba8,
                )
                .expect("write the PNG");
            }
        }
    }
}
