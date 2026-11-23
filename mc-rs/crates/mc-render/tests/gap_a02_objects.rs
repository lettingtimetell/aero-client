//! The non-living entity renderers, on a real GPU.
//!
//! The unit tests beside [`mc_render::entity::objects`] check the arithmetic against the Java
//! numbers. Two things they cannot check live here:
//!
//! 1. **That the texture paths exist.** Every string in that module is transcribed from a
//!    `Render*` class's `ResourceLocation`. A typo compiles, passes every unit test, and shows up
//!    in a live session as an entity drawn against the 1x1 white fallback. They are resolved
//!    against the real `1.8.9.jar` here.
//! 2. **That the geometry reaches the screen.** Each renderer is emitted, uploaded and drawn into
//!    an offscreen target, and the pixels are read back. An unregistered renderer — which is what
//!    every one of these was before this pass — produces a frame identical to the empty one, so
//!    "differs from an empty scene" is a test that could not have passed before.
//!
//! Everything skips cleanly (printing `SKIP:`) when the jar or a GPU adapter is missing.

use std::path::{Path, PathBuf};

use glam::Vec3;
use mc_assets::{ResourceLocation, ResourceManager, ZipResourcePack};
use mc_render::entity::models::object::{ender_crystal_motion, EnderCrystalModel};
use mc_render::entity::objects;
use mc_render::entity::render::EntityPhase;
use mc_render::entity::{AnimParams, EntityModel, EntityVertex, Mat4};
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

// ------------------------------------------------------------------ fixtures

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
    if !jar.is_file() {
        return None;
    }
    let pack = ZipResourcePack::open(jar).ok()?;
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(pack));
    Some(rm)
}

macro_rules! require {
    ($opt:expr, $what:literal) => {
        match $opt {
            Some(v) => v,
            None => {
                eprintln!("SKIP: {} not available", $what);
                return;
            }
        }
    };
}

fn headless(width: u32, height: u32) -> Option<Gpu> {
    match Gpu::headless(
        width,
        height,
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

const SIZE: u32 = 96;

/// Draws one batch of geometry against a flat white texture and returns the frame.
fn draw(gpu: &mut Gpu, verts: &[EntityVertex]) -> Vec<u8> {
    let mut phase = EntityPhase::new(gpu);
    phase.upload_texture(gpu, "flat", &[220, 220, 220, 255], 1, 1);
    phase.begin();
    phase.push("flat", verts);
    phase.finish(gpu);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::ENTITIES, Box::new(phase));
    let mut cam = Camera::default();
    cam.set_viewport(SIZE, SIZE);
    cam.update();
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &cam, 0.0);
    frame.present(&gpu.queue);
    gpu.read_pixels().expect("read back the frame")
}

/// Pixels that are not the cleared background.
fn lit(pixels: &[u8]) -> usize {
    pixels
        .chunks(4)
        .filter(|p| p[0] > 8 || p[1] > 8 || p[2] > 8)
        .count()
}

// ------------------------------------------------------- the texture-path audit

/// Every `ResourceLocation` the non-living renderers name must exist in the real jar.
///
/// java: `RenderXPOrb.experienceOrbTextures`, `RenderArrow.arrowTextures`,
/// `RenderPainting.KRISTOFFER_PAINTING_TEXTURE`, `RenderFish.FISH_PARTICLES`,
/// `RenderWitherSkull`'s two sheets. A wrong path here is invisible until a live session.
#[test]
fn every_object_renderer_texture_exists_in_the_vanilla_jar() {
    let rm = require!(resources(), "1.8.9.jar");
    for path in [
        objects::XP_ORB_TEXTURE,
        objects::ARROW_TEXTURE,
        objects::PAINTING_TEXTURE,
        objects::FISH_TEXTURE,
        objects::WITHER_SKULL_TEXTURE,
        objects::WITHER_SKULL_INVULNERABLE_TEXTURE,
    ] {
        let loc = ResourceLocation::parse(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        assert!(
            rm.load(&loc).is_ok(),
            "{path} is named by a renderer and is not in the jar"
        );
    }
}

/// The block atlas has to carry the fireball's particle sprite, because `RenderFireball` binds
/// `TextureMap.locationBlocksTexture` and samples `items/fireball` out of it.
#[test]
fn the_fireball_sprite_is_an_item_texture_in_the_jar() {
    let rm = require!(resources(), "1.8.9.jar");
    let loc = ResourceLocation::parse("textures/items/fireball.png").expect("parse");
    assert!(rm.load(&loc).is_ok(), "items/fireball is not in the jar");
}

// ------------------------------------------------------------- real draws

/// An XP orb must produce pixels. Before A02-02 this entity type had no renderer at all, so the
/// frame was byte-identical to the empty one.
#[test]
fn an_xp_orb_reaches_the_framebuffer() {
    let mut gpu = require!(headless(SIZE, SIZE), "a GPU adapter");
    let empty = draw(&mut gpu, &[]);
    assert_eq!(lit(&empty), 0, "the control frame must be black");

    let mut verts = Vec::new();
    // Two blocks in front of a default camera (yaw 0 faces +Z), slightly below eye level.
    objects::emit_xp_orb(Vec3::new(0.0, -0.2, 2.0), 0.0, 0.0, 100, 0, 0.0, &mut verts);
    assert_eq!(verts.len(), 6, "one quad");
    let pixels = draw(&mut gpu, &verts);
    assert!(
        lit(&pixels) > 20,
        "an orb should cover pixels, got {}",
        lit(&pixels)
    );
    assert_ne!(pixels, empty);
}

/// An arrow's six quads must reach the screen and must be longer than they are tall — the shaft
/// is 16 texels and the vanes 4, so a side-on arrow is a thin streak, not a blob.
#[test]
fn an_arrow_reaches_the_framebuffer_as_a_streak() {
    let mut gpu = require!(headless(SIZE, SIZE), "a GPU adapter");
    let mut verts = Vec::new();
    // Yaw 90 turns the shaft across the view, so its length lands on the screen's X axis.
    objects::emit_arrow(Vec3::new(0.0, -0.1, 2.0), 90.0, 0.0, 0.0, &mut verts);
    assert_eq!(verts.len(), 36, "six quads");
    let pixels = draw(&mut gpu, &verts);
    let mut xs = (u32::MAX, 0u32);
    let mut ys = (u32::MAX, 0u32);
    let mut count = 0;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let i = ((y * SIZE + x) * 4) as usize;
            if pixels[i] > 8 || pixels[i + 1] > 8 || pixels[i + 2] > 8 {
                xs = (xs.0.min(x), xs.1.max(x));
                ys = (ys.0.min(y), ys.1.max(y));
                count += 1;
            }
        }
    }
    assert!(count > 20, "the arrow should cover pixels, got {count}");
    assert!(
        xs.1 - xs.0 > ys.1 - ys.0,
        "a side-on arrow is wider than it is tall: {}x{}",
        xs.1 - xs.0,
        ys.1 - ys.0
    );
}

/// A painting must reach the screen, and a bigger painting must cover more of it — which is the
/// property that proves `EnumArt`'s size table is actually being read rather than a constant.
#[test]
fn a_painting_reaches_the_framebuffer_and_scales_with_its_art() {
    let mut gpu = require!(headless(SIZE, SIZE), "a GPU adapter");
    let coverage = |title: &str, gpu: &mut Gpu| {
        let mut verts = Vec::new();
        // A north-facing painting (`hanging_yaw(North) == 180`) cancels `rotate(180 - yaw)`, so
        // the art's own front face — the one at local `-Z` — stays on `-Z` and points back at a
        // default camera sitting at the origin looking down `+Z`.
        objects::emit_painting(
            Vec3::new(0.0, 0.0, 6.0),
            objects::hanging_yaw(mc_math::Facing::North),
            objects::Art::by_title(title),
            &mut verts,
        );
        lit(&draw(gpu, &verts))
    };
    let kebab = coverage("Kebab", &mut gpu); // 1x1 blocks
    let kong = coverage("DonkeyKong", &mut gpu); // 4x3 blocks
    assert!(kebab > 10, "a 1x1 painting should be visible, got {kebab}");
    assert!(
        kong > kebab * 4,
        "a 4x3 painting covers ~12x the area of a 1x1: {kebab} vs {kong}"
    );
}

/// The dropped-item path end to end at the geometry level: `emit_baked_model` through
/// `emit_dropped_item`, with the jitter producing five distinct copies.
#[test]
fn a_dropped_stack_reaches_the_framebuffer_as_five_copies() {
    let mut gpu = require!(headless(SIZE, SIZE), "a GPU adapter");
    let model = full_cube();
    let mut one = Vec::new();
    mc_render::entity::item::emit_dropped_item(
        &mc_render::entity::item::DroppedItem {
            rel: Vec3::new(0.0, -0.3, 2.0),
            stack_size: 1,
            gui_3d: true,
            ..Default::default()
        },
        &model,
        Mat4::IDENTITY,
        0.0,
        &mut one,
    );
    let mut five = Vec::new();
    mc_render::entity::item::emit_dropped_item(
        &mc_render::entity::item::DroppedItem {
            rel: Vec3::new(0.0, -0.3, 2.0),
            stack_size: 64,
            gui_3d: true,
            ..Default::default()
        },
        &model,
        Mat4::IDENTITY,
        0.0,
        &mut five,
    );
    assert_eq!(
        five.len(),
        one.len() * 5,
        "java: func_177078_a's step function"
    );

    let single = lit(&draw(&mut gpu, &one));
    let stack = lit(&draw(&mut gpu, &five));
    assert!(
        single > 10,
        "a dropped block should be visible, got {single}"
    );
    // The 187-seeded jitter spreads the extra copies +-0.15 blocks, so a stack of 64 covers more
    // of the screen than a single item does.
    assert!(
        stack > single,
        "a stack should cover more than one item: {single} vs {stack}"
    );
}

/// A cube of six full faces in `0..1` space, matching how S09 bakes a full block.
fn full_cube() -> mc_assets::BakedModel {
    use mc_assets::{BakedQuad, Vertex};
    let mut model = mc_assets::BakedModel::default();
    let faces: [[[f32; 3]; 4]; 6] = [
        [[0., 0., 0.], [1., 0., 0.], [1., 0., 1.], [0., 0., 1.]],
        [[0., 1., 0.], [1., 1., 0.], [1., 1., 1.], [0., 1., 1.]],
        [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
        [[0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]],
        [[0., 0., 0.], [0., 0., 1.], [0., 1., 1.], [0., 1., 0.]],
        [[1., 0., 0.], [1., 0., 1.], [1., 1., 1.], [1., 1., 0.]],
    ];
    for f in faces {
        model.general_quads.push(BakedQuad {
            verts: core::array::from_fn(|i| Vertex {
                pos: f[i],
                uv: [0.0, 0.0],
            }),
            face: None,
            shade_face: None,
            tint_index: None,
            shade: true,
        });
    }
    model
}

/// The crystal's shells must stay put. With the old parameter punning the driver's
/// `age_in_ticks` became a height in blocks, so after a minute the crystal was 1200 blocks above
/// its base — off the top of any frame. Drawn here at two ages a minute apart, both must land on
/// the screen and cover a comparable number of pixels.
#[test]
fn an_ender_crystal_is_still_on_screen_after_a_minute() {
    let mut gpu = require!(headless(SIZE, SIZE), "a GPU adapter");
    let crystal = EnderCrystalModel::new(true);
    let at = |ticks: f32, gpu: &mut Gpu| {
        let mut verts = Vec::new();
        let params = AnimParams {
            age_in_ticks: ticks,
            ..AnimParams::still()
        };
        crystal.render(
            &params,
            objects::ender_crystal_matrix(Vec3::new(0.0, -1.0, 5.0)),
            &mut verts,
        );
        lit(&draw(gpu, &verts))
    };
    let early = at(1.0, &mut gpu);
    let late = at(1200.0, &mut gpu);
    assert!(early > 50, "the crystal should be visible, got {early}");
    assert!(
        late > 50,
        "the crystal should still be visible after 1200 ticks, got {late}"
    );
    // The bob's whole range is 0.4 blocks before the model's own scale(2); at five blocks that
    // cannot halve the silhouette.
    assert!(
        late * 2 > early && early * 2 > late,
        "coverage should be comparable: {early} vs {late}"
    );
    // And the spin really does turn it: two ages a third of a rotation apart differ.
    let (spin_a, _) = ender_crystal_motion(0.0);
    let (spin_b, _) = ender_crystal_motion(40.0);
    assert!(
        (spin_b - spin_a - 120.0).abs() < 1e-3,
        "three degrees a tick"
    );
}
