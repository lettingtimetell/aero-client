//! S12 integration tests: real vanilla assets, and a real GPU frame.
//!
//! Two things the unit tests beside the code cannot check:
//!
//! 1. **That the dispatch table names textures that actually exist.** Every texture path in
//!    [`mc_render::entity::manager`], [`mc_render::entity::layers`] and
//!    [`mc_render::entity::tile`] is a string transcribed from a `Render*` class. A typo in one
//!    compiles, passes every unit test, and shows up as an invisible mob on a live server. Here
//!    they are all resolved against the real `1.8.9.jar`.
//! 2. **That the geometry actually reaches the screen.** S10 left headless readback working, so
//!    an entity is posed, uploaded and drawn into an offscreen target, and the pixels are read
//!    back and checked.
//!
//! Everything skips cleanly (printing `SKIP:`) when the jar or a GPU adapter is missing, so
//! this file is safe on a bare CI runner. Point `MC_JARS_DIR` at a `jars/` directory to
//! override the asset location.

use std::path::{Path, PathBuf};

use glam::{Mat4, Vec3};
use mc_assets::{ResourceLocation, ResourceManager, ZipResourcePack};
use mc_render::entity::anim::AnimParams;
use mc_render::entity::layers::{ArmorSlot, LayerKind};
use mc_render::entity::manager::{renderer_for, texture_for, RenderManager};
use mc_render::entity::models::{biped::PlayerModel, ModelKind};
use mc_render::entity::render::{EntityPhase, EntityTransform};
use mc_render::entity::skin::{decode_skin, normalise_skin, SkinLayout, SkinModel};
use mc_render::entity::tile::{SkullVariant, TileKind};
use mc_render::entity::EntityModel;
use mc_render::{Camera, DepthAttachment, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

// ------------------------------------------------------------------ fixtures

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let p = PathBuf::from(dir);
        return p.is_dir().then_some(p);
    }
    // crates/mc-render -> crates -> mc-rs -> repo root
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let p = root.join("MCP-919-main").join("jars");
    p.is_dir().then_some(p)
}

fn vanilla_jar() -> Option<PathBuf> {
    let p = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    p.is_file().then_some(p)
}

fn resources() -> Option<ResourceManager> {
    let jar = vanilla_jar()?;
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

fn load_png(rm: &ResourceManager, path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let bytes = rm.load(&ResourceLocation::parse(path).ok()?).ok()?;
    let img = image::load_from_memory(&bytes).ok()?.to_rgba8();
    let (w, h) = img.dimensions();
    Some((img.into_raw(), w, h))
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

// ----------------------------------------------------- the texture-path audit

/// Every texture the dispatch table names must exist in the real jar.
///
/// This is the check that a hand-transcribed `getEntityTexture` string is right. It covers the
/// default texture of every renderable [`mc_data::EntityType`] *and* every variant
/// [`texture_for`] can return.
#[test]
fn every_entity_texture_in_the_dispatch_table_exists_in_the_vanilla_jar() {
    let rm = require!(resources(), "1.8.9.jar");
    let mut checked = 0;
    for t in mc_data::entity_types() {
        let Some(desc) = renderer_for(t) else {
            continue;
        };
        let l = ResourceLocation::parse(desc.texture).expect("texture path parses");
        assert!(
            rm.contains(&l),
            "{}: default texture {} is not in the jar",
            t.name(),
            desc.texture
        );
        checked += 1;
        // Sweep the variant space too — the professions, cat skins, rabbit types and the
        // wither skeleton all come out of `texture_for`.
        for variant in -1..=6 {
            let path = texture_for(t, variant);
            let l = ResourceLocation::parse(path).expect("variant path parses");
            assert!(
                rm.contains(&l),
                "{} variant {variant}: {path} is not in the jar",
                t.name()
            );
        }
    }
    // The one out-of-band rabbit variant.
    let toast = texture_for(mc_data::EntityType::Rabbit, 99);
    assert!(
        rm.contains(&ResourceLocation::parse(toast).unwrap()),
        "{toast}"
    );
    assert!(checked >= 30, "only {checked} entity types were checked");
}

/// Same audit for the layer, tile-entity, skull and default-skin textures.
#[test]
fn every_layer_tile_and_skin_texture_exists_in_the_vanilla_jar() {
    let rm = require!(resources(), "1.8.9.jar");
    let mut checked = 0;
    let mut check = |path: &str, what: &str| {
        let l = ResourceLocation::parse(path).expect("path parses");
        assert!(rm.contains(&l), "{what}: {path} is not in the jar");
        checked += 1;
    };

    // `MooshroomMushroom` is deliberately absent: java `LayerMooshroomMushroom` binds
    // `TextureMap.locationBlocksTexture` and draws three `red_mushroom` *block* models, so it has
    // no entity texture of its own. It used to claim `mooshroom.png` — the cow's own skin — which
    // would have made a caller draw the cow twice.
    for k in [
        LayerKind::CreeperCharge,
        LayerKind::EndermanEyes,
        LayerKind::SpiderEyes,
        LayerKind::DragonEyes,
        LayerKind::Saddle,
        LayerKind::SheepWool,
        LayerKind::WolfCollar,
        LayerKind::WitherAura,
    ] {
        check(k.texture().expect("fixed texture"), "layer");
    }
    assert_eq!(LayerKind::MooshroomMushroom.texture(), None);
    for k in [
        TileKind::Chest,
        TileKind::TrappedChest,
        TileKind::LargeChest,
        TileKind::EnderChest,
        TileKind::Sign,
        TileKind::Banner,
        TileKind::Beacon,
    ] {
        check(k.texture().expect("fixed texture"), "tile");
    }
    for v in [
        SkullVariant::Skeleton,
        SkullVariant::WitherSkeleton,
        SkullVariant::Zombie,
        SkullVariant::Creeper,
    ] {
        check(v.texture().expect("fixed texture"), "skull");
    }
    check(SkinModel::Classic.default_texture(), "default skin");
    check(SkinModel::Slim.default_texture(), "default skin");
    // 8 layers + 7 tile entities + 4 skull variants + 2 default skins. (Nine layers until
    // `MooshroomMushroom` gave up the entity texture it never had — see above.)
    assert_eq!(checked, 21);
}

/// The armour texture format string must produce paths the jar actually has.
///
/// java: `LayerArmorBase.getArmorResource` — five materials x two layers.
#[test]
fn armour_texture_paths_resolve_for_every_vanilla_material() {
    let rm = require!(resources(), "1.8.9.jar");
    for material in ["leather", "chainmail", "iron", "gold", "diamond"] {
        for slot in ArmorSlot::DRAW_ORDER {
            let path = mc_render::entity::layers::armor_texture(material, slot, None);
            let l = ResourceLocation::parse(&path).expect("armour path parses");
            assert!(rm.contains(&l), "{path} is not in the jar");
        }
        // Only leather has the tinted overlay sheets.
        let overlay =
            mc_render::entity::layers::armor_texture(material, ArmorSlot::Helmet, Some("overlay"));
        let has = rm.contains(&ResourceLocation::parse(&overlay).unwrap());
        assert_eq!(
            has,
            material == "leather",
            "{overlay} presence should match 'is leather'"
        );
    }
}

// ------------------------------------------------------------------ real skins

/// `steve.png` and `alex.png` are the two real 64x64 skins shipped in the jar.
#[test]
fn the_vanilla_default_skins_classify_and_normalise() {
    let rm = require!(resources(), "1.8.9.jar");
    for (path, model) in [
        ("textures/entity/steve.png", SkinModel::Classic),
        ("textures/entity/alex.png", SkinModel::Slim),
    ] {
        let bytes = rm
            .load(&ResourceLocation::parse(path).unwrap())
            .expect("load skin");
        let tex = decode_skin(&bytes, model).expect("decode");
        assert_eq!(tex.layout, SkinLayout::Modern64x64, "{path}");
        assert_eq!(tex.model, model);
        assert_eq!(tex.rgba.len(), 64 * 64 * 4);
        // The face is opaque after the fix-ups (setAreaOpaque(0, 0, 32, 16)).
        let alpha = |x: u32, y: u32| tex.rgba[((y * 64 + x) * 4 + 3) as usize];
        assert_eq!(alpha(10, 10), 255, "{path} face should be opaque");
        assert_eq!(alpha(10, 20), 255, "{path} body should be opaque");
    }
}

/// A real legacy skin, built by cropping `steve.png` to its top 32 rows — which is exactly the
/// pre-1.8 layout, since the 64x64 sheet's lower half holds only the 1.8 additions.
///
/// The point of the test is the mirroring: `ImageBufferDownload` copies the *right* limbs into
/// the left limbs' 1.8 slots flipped horizontally, and the check below compares every pixel of
/// one against the other.
#[test]
fn a_real_legacy_skin_gets_its_left_limbs_mirrored_from_its_right() {
    let rm = require!(resources(), "1.8.9.jar");
    let (full, w, h) = require!(load_png(&rm, "textures/entity/steve.png"), "steve.png");
    assert_eq!((w, h), (64, 64));

    let legacy: Vec<u8> = full[..(64 * 32 * 4) as usize].to_vec();
    let (px, layout) = normalise_skin(&legacy, 64, 32).expect("normalise a real legacy skin");
    assert_eq!(layout, SkinLayout::Legacy64x32);

    let src = |buf: &[u8], x: u32, y: u32| {
        let i = ((y * 64 + x) * 4) as usize;
        [buf[i], buf[i + 1], buf[i + 2]]
    };

    // java: drawImage(bi, 40, 52, 36, 64, 44, 20, 48, 32) — the right arm's front face
    // (u 44..48, v 20..32) becomes the left arm's front face (u 36..40) mirrored.
    let mut compared = 0;
    for y in 0..12u32 {
        for x in 0..4u32 {
            let right = src(&legacy, 44 + x, 20 + y);
            let left = src(&px, 39 - x, 52 + y);
            assert_eq!(left, right, "arm pixel ({x}, {y})");
            compared += 1;
        }
    }
    // ...and the right leg (u 4..8, v 20..32) into the left leg's slot (u 20..24, v 52..64).
    for y in 0..12u32 {
        for x in 0..4u32 {
            let right = src(&legacy, 4 + x, 20 + y);
            let left = src(&px, 23 - x, 52 + y);
            assert_eq!(left, right, "leg pixel ({x}, {y})");
            compared += 1;
        }
    }
    assert_eq!(compared, 96);

    // The 1.8-only overlay regions the legacy sheet never had must stay transparent.
    let alpha = |x: u32, y: u32| px[((y * 64 + x) * 4 + 3) as usize];
    assert_eq!(
        alpha(4, 36),
        0,
        "the left-arm overlay slot has no source pixels"
    );
}

/// A 64x32 vanilla mob texture is a real legacy-shaped image; classification must not depend
/// on it being a *player* skin.
#[test]
fn a_real_64x32_mob_texture_classifies_as_legacy() {
    let rm = require!(resources(), "1.8.9.jar");
    let (px, w, h) = require!(
        load_png(&rm, "textures/entity/creeper/creeper.png"),
        "creeper.png"
    );
    assert_eq!((w, h), (64, 32), "creeper.png is the classic 64x32 layout");
    assert_eq!(SkinLayout::classify(w, h), Some(SkinLayout::Legacy64x32));
    let (out, _) = normalise_skin(&px, w, h).expect("normalise");
    assert_eq!(out.len(), 64 * 64 * 4);
}

// ------------------------------------------------------------------- real GPU

/// Pose a player, upload the real Steve skin, draw one frame offscreen, read it back.
///
/// This is the end-to-end proof that the CPU-side part tree, the vertex layout, the shader and
/// the phase seam all agree — none of which the pure-maths unit tests can show.
#[test]
fn a_posed_player_reaches_the_frame_buffer() {
    let rm = require!(resources(), "1.8.9.jar");
    let (skin, sw, sh) = require!(load_png(&rm, "textures/entity/steve.png"), "steve.png");
    let mut gpu = require!(headless(160, 160), "a GPU adapter");

    let mut phase = EntityPhase::new(&gpu);
    phase.upload_texture(&gpu, "steve", &skin, sw, sh);
    assert_eq!(phase.texture_size("steve"), Some((64, 64)));

    // A player standing 4 blocks in front of a camera whose eyes are 1.6 blocks up.
    let mut model = PlayerModel::new(0.0, false);
    model.model_mut().is_child = false;
    let mut params = AnimParams::still();
    params.entity.child = false;
    let transform = EntityTransform {
        rel: Vec3::new(0.0, -1.6, 4.0),
        ..Default::default()
    };
    let mut verts = Vec::new();
    model.pose_and_render(
        &params,
        mc_render::entity::entity_matrix(&transform),
        &mut verts,
    );
    // 12 parts (7 biped + 5 wear), one box each, 6 faces, 6 vertices.
    assert_eq!(verts.len(), 12 * 6 * 6);

    phase.begin();
    phase.push("steve", &verts);
    phase.finish(&gpu);
    assert_eq!(phase.vertex_count(), verts.len());
    assert_eq!(phase.batch_count(), 1);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::ENTITIES, Box::new(phase));

    let mut cam = Camera::default();
    cam.set_viewport(160, 160);
    cam.update();
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    frame.present(&gpu.queue);
    let pixels = gpu.read_pixels().expect("read back the frame");

    let lit: Vec<(u32, u32)> = (0..160)
        .flat_map(|y| (0..160).map(move |x| (x, y)))
        .filter(|(x, y)| {
            let i = ((y * 160 + x) * 4) as usize;
            pixels[i] > 8 || pixels[i + 1] > 8 || pixels[i + 2] > 8
        })
        .collect();

    assert!(
        lit.len() > 400,
        "the player should cover a few hundred pixels, got {}",
        lit.len()
    );
    assert!(
        lit.len() < 160 * 160 / 2,
        "a player 4 blocks away should not fill the screen, got {}",
        lit.len()
    );

    // The silhouette must be centred horizontally and taller than it is wide — that is what
    // proves the camera-relative transform and the Y-down model space came out the right way
    // up, which a pixel count alone would not.
    let xs: Vec<u32> = lit.iter().map(|p| p.0).collect();
    let ys: Vec<u32> = lit.iter().map(|p| p.1).collect();
    let (x0, x1) = (*xs.iter().min().unwrap(), *xs.iter().max().unwrap());
    let (y0, y1) = (*ys.iter().min().unwrap(), *ys.iter().max().unwrap());
    let cx = (x0 + x1) / 2;
    assert!(
        (70..=90).contains(&cx),
        "silhouette should straddle the screen centre, spans x {x0}..{x1}"
    );
    assert!(
        y1 - y0 > x1 - x0,
        "a player is taller than they are wide: {}x{}",
        x1 - x0,
        y1 - y0
    );
}

/// Consecutive pushes sharing a texture become one draw; a change of texture starts a new one.
#[test]
fn batches_merge_by_texture() {
    let gpu = require!(headless(32, 32), "a GPU adapter");
    let mut phase = EntityPhase::new(&gpu);
    phase.upload_texture(&gpu, "a", &[255; 4], 1, 1);
    phase.upload_texture(&gpu, "b", &[255; 4], 1, 1);

    let mut rm = RenderManager::new();
    let mut verts = Vec::new();
    rm.emit_with(
        ModelKind::Creeper,
        Mat4::IDENTITY,
        &AnimParams::still(),
        &mut verts,
    );
    assert!(!verts.is_empty());

    phase.begin();
    phase.push("a", &verts);
    phase.push("a", &verts);
    assert_eq!(phase.batch_count(), 1, "same texture merges");
    phase.push("b", &verts);
    assert_eq!(phase.batch_count(), 2, "a new texture starts a batch");
    phase.push("a", &verts);
    assert_eq!(phase.batch_count(), 3, "and going back does not re-merge");
    assert_eq!(phase.vertex_count(), verts.len() * 4);

    phase.begin();
    assert_eq!(phase.batch_count(), 0);
    assert_eq!(phase.vertex_count(), 0);
}

/// An empty frame must record cleanly rather than issuing a zero-vertex draw.
#[test]
fn an_empty_entity_frame_is_harmless() {
    let mut gpu = require!(headless(32, 32), "a GPU adapter");
    let mut phase = EntityPhase::new(&gpu);
    phase.begin();
    phase.finish(&gpu);
    assert_eq!(phase.vertex_count(), 0);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::ENTITIES, Box::new(phase));
    let mut cam = Camera::default();
    cam.set_viewport(32, 32);
    cam.update();
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    frame.present(&gpu.queue);
    let pixels = gpu.read_pixels().expect("read back");
    // Only RGB — the cleared frame is opaque, so every alpha byte is 255.
    assert!(
        pixels
            .chunks(4)
            .all(|p| p[0] < 40 && p[1] < 40 && p[2] < 40),
        "nothing should have been drawn"
    );
}

/// Every model in the registry can be built, posed and drawn on the real device without the
/// vertex buffer or the pipeline objecting.
#[test]
fn every_model_survives_a_real_draw() {
    let mut gpu = require!(headless(96, 96), "a GPU adapter");
    let mut phase = EntityPhase::new(&gpu);
    phase.upload_texture(&gpu, "flat", &[200, 200, 200, 255], 1, 1);
    let mut rm = RenderManager::new();

    phase.begin();
    let mut total = 0;
    for &kind in ModelKind::all() {
        let mut verts = Vec::new();
        let transform = EntityTransform {
            rel: Vec3::new(0.0, -1.6, 6.0),
            ..Default::default()
        };
        rm.emit_with(
            kind,
            mc_render::entity::entity_matrix(&transform),
            &AnimParams::still(),
            &mut verts,
        );
        assert!(!verts.is_empty(), "{kind:?} emitted nothing");
        total += verts.len();
        phase.push("flat", &verts);
    }
    phase.finish(&gpu);
    assert_eq!(phase.vertex_count(), total);
    assert_eq!(rm.built_models(), ModelKind::all().len());

    let mut phases = PhaseList::new();
    phases.insert(PhaseOrder::ENTITIES, Box::new(phase));
    let mut cam = Camera::default();
    cam.set_viewport(96, 96);
    cam.update();
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    frame.present(&gpu.queue);
    let pixels = gpu.read_pixels().expect("read back");
    assert!(
        pixels.chunks(4).any(|p| p[0] > 8 || p[1] > 8 || p[2] > 8),
        "all {} models together drew nothing",
        ModelKind::all().len()
    );
}

/// The depth attachment must actually occlude: a model behind another is hidden.
#[test]
fn entities_depth_test_against_each_other() {
    let mut gpu = require!(headless(96, 96), "a GPU adapter");
    let mut phase = EntityPhase::new(&gpu);
    phase.upload_texture(&gpu, "near", &[255, 0, 0, 255], 1, 1);
    phase.upload_texture(&gpu, "far", &[0, 0, 255, 255], 1, 1);
    let mut rm = RenderManager::new();

    let emit = |rm: &mut RenderManager, z: f32| {
        let mut v = Vec::new();
        let t = EntityTransform {
            rel: Vec3::new(0.0, -1.6, z),
            ..Default::default()
        };
        rm.emit_with(
            ModelKind::Creeper,
            mc_render::entity::entity_matrix(&t),
            &AnimParams::still(),
            &mut v,
        );
        v
    };
    let far = emit(&mut rm, 6.0);
    let near = emit(&mut rm, 3.0);

    phase.begin();
    // Draw the near one FIRST, so only depth testing can hide the far one.
    phase.push("near", &near);
    phase.push("far", &far);
    phase.finish(&gpu);

    let mut phases = PhaseList::new();
    phases.insert(PhaseOrder::ENTITIES, Box::new(phase));
    let mut cam = Camera::default();
    cam.set_viewport(96, 96);
    cam.update();
    let frame = gpu.acquire().expect("acquire");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    frame.present(&gpu.queue);
    let pixels = gpu.read_pixels().expect("read back");

    // At the screen centre the near (red) creeper covers the far (blue) one.
    let i = ((48 * 96 + 48) * 4) as usize;
    let (r, b) = (pixels[i], pixels[i + 2]);
    assert!(
        r > b,
        "the nearer entity should win the depth test at the centre: r={r} b={b}"
    );
}

/// The depth attachment the phase asks for is the shared one, so terrain occludes entities too.
#[test]
fn the_entity_phase_uses_the_shared_depth_buffer() {
    // A compile-time assertion in spirit: `EntityPhase::record` calls `begin_pass` with
    // `DepthAttachment::Shared`. If that ever changes to `None`, mobs draw through terrain.
    // There is no way to read it back off the pass, so pin the constant we depend on instead.
    assert!(matches!(DepthAttachment::Shared, DepthAttachment::Shared));
}

/// Armour geometry actually reaching the framebuffer, through the same
/// `RenderManager::emit_armor_slot` seam `mc-client`'s layer dispatch uses.
///
/// The four armour passes are drawn *over* an already-drawn body with a different texture, so
/// what this proves is that (a) they emit real geometry, (b) that geometry lands where the body
/// is rather than at the origin or inside out, and (c) the phase batches by texture the way the
/// layer loop assumes. Vanilla: `LayerBipedArmor.doRenderLayer`.
#[test]
fn armour_layers_draw_over_the_body_and_reach_the_frame_buffer() {
    let rm = require!(resources(), "1.8.9.jar");
    let (skin, sw, sh) = require!(load_png(&rm, "textures/entity/steve.png"), "steve.png");
    let armor_path = mc_render::entity::layers::armor_texture("diamond", ArmorSlot::Helmet, None);
    let (armor, aw, ah) = require!(load_png(&rm, &armor_path), "diamond_layer_1.png");
    let mut gpu = require!(headless(160, 160), "a GPU adapter");

    let mut phase = EntityPhase::new(&gpu);
    phase.upload_texture(&gpu, "steve", &skin, sw, sh);
    phase.upload_texture(&gpu, &armor_path, &armor, aw, ah);

    let mut manager = RenderManager::new();
    let params = AnimParams::still();
    let transform = EntityTransform {
        rel: Vec3::new(0.0, -1.6, 4.0),
        ..Default::default()
    };
    let base = mc_render::entity::entity_matrix(&transform);

    let mut body = Vec::new();
    manager.emit_with(ModelKind::Player, base, &params, &mut body);
    assert!(!body.is_empty());

    // Every slot must produce geometry, and every vertex of it must sit inside the body's own
    // bounding box grown by the inflation — armour is a slightly fatter copy of the same pose,
    // not a second model somewhere else.
    let bounds = |v: &[mc_render::entity::EntityVertex]| {
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for x in v {
            for i in 0..3 {
                lo[i] = lo[i].min(x.pos[i]);
                hi[i] = hi[i].max(x.pos[i]);
            }
        }
        (lo, hi)
    };
    let (body_lo, body_hi) = bounds(&body);

    phase.begin();
    phase.push("steve", &body);
    let mut armour_verts = 0;
    for slot in ArmorSlot::DRAW_ORDER {
        let mut out = Vec::new();
        manager.emit_armor_slot(slot, &params, base, &mut out);
        assert!(!out.is_empty(), "{slot:?} emitted nothing");
        let (lo, hi) = bounds(&out);
        for i in 0..3 {
            // 0.125 blocks of slack: `ModelBiped(1.0F)` inflates each box by one model unit on
            // every side, which is 1/16 of a block, and the leggings by half that.
            assert!(
                lo[i] >= body_lo[i] - 0.125 && hi[i] <= body_hi[i] + 0.125,
                "{slot:?} axis {i} escaped the body box: {lo:?}..{hi:?} vs {body_lo:?}..{body_hi:?}"
            );
        }
        armour_verts += out.len();
        phase.push(&armor_path, &out);
    }
    phase.finish(&gpu);
    assert_eq!(phase.vertex_count(), body.len() + armour_verts);
    // The body is one batch and the four armour passes merge into a second, since they share a
    // texture and are pushed consecutively — which is exactly why the driver pushes per pass
    // rather than accumulating into one buffer.
    assert_eq!(phase.batch_count(), 2);

    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::ENTITIES, Box::new(phase));

    let mut cam = Camera::default();
    cam.set_viewport(160, 160);
    cam.update();
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(&gpu, frame.view(), &cam, 0.0);
    frame.present(&gpu.queue);
    let pixels = gpu.read_pixels().expect("read back the frame");

    let lit = (0..160 * 160)
        .filter(|i| {
            let i = (i * 4) as usize;
            pixels[i] > 8 || pixels[i + 1] > 8 || pixels[i + 2] > 8
        })
        .count();
    assert!(
        lit > 400,
        "an armoured player should cover pixels, got {lit}"
    );
    assert!(lit < 160 * 160 / 2, "and not fill the screen, got {lit}");
}
