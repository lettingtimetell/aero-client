//! Worn skulls and the slime gel, checked on geometry and on a
//! real GPU frame.
//!
//! * `LayerCustomHead` ends with `TileEntitySkullRenderer.renderSkull(-0.5, 0, -0.5, UP, 180,
//!   ...)`, which adds a `scale(-1, -1, 1)` flip and poses the skull at a fixed 180-degree yaw.
//!   With the layer's own `scale(1.1875, -1.1875, -1.1875)` that leaves the head box enlarged
//!   1.1875 times about the neck pivot. The old callers skipped both and hung the skull upside
//!   down below the neck, inside the torso.
//! * `LayerSlimeGel` blends with `blendFunc(770, 771)` but never calls `depthMask(false)`, and
//!   vanilla draws entities before translucent terrain. So water behind a slime fails the depth
//!   test wherever the gel is. The old pipeline turned the gel's depth writes off, which let the
//!   water draw over it.
//!
//! The GPU tests skip cleanly (printing `SKIP:`) without an adapter, and the slime test also
//! needs the vanilla jar (`MC_JARS_DIR` overrides its location).

use std::path::{Path, PathBuf};

use glam::{Mat4, Vec3};
use mc_assets::{ResourceLocation, ResourceManager, ZipResourcePack};
use mc_render::entity::anim::AnimParams;
use mc_render::entity::layers::{custom_head_matrix, CustomHeadItem, LayerBlend, LayerKind};
use mc_render::entity::manager::RenderManager;
use mc_render::entity::models::ModelKind;
use mc_render::entity::render::{
    entity_matrix, entity_matrix_scaled, slime_pre_render_scale, EntityPhase, EntityTransform,
};
use mc_render::entity::tile::SkullVariant;
use mc_render::entity::EntityVertex;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

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

fn resources() -> Option<ResourceManager> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    let pack = ZipResourcePack::open(jar).ok()?;
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(pack));
    Some(rm)
}

fn load_png(rm: &ResourceManager, path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let bytes = rm.load(&ResourceLocation::parse(path).ok()?).ok()?;
    let img = image::load_from_memory(&bytes).ok()?.to_rgba8();
    let (w, h) = img.dimensions();
    Some((img.into_raw(), w, h))
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

fn headless(size: u32) -> Option<Gpu> {
    match Gpu::headless(
        size,
        size,
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

/// Renders the given phases over a black clear and reads the frame back as RGBA rows.
fn render(gpu: &mut Gpu, size: u32, phases: Vec<(PhaseOrder, EntityPhase)>) -> Vec<u8> {
    let mut list = PhaseList::new();
    list.clear_color = wgpu::Color::BLACK;
    for (order, phase) in phases {
        list.insert(order, Box::new(phase));
    }
    let mut cam = Camera::default();
    cam.set_viewport(size, size);
    cam.update();
    let frame = gpu.acquire().expect("acquire offscreen frame");
    list.render(gpu, frame.view(), &cam, 0.0);
    frame.present(&gpu.queue);
    gpu.read_pixels().expect("read back the frame")
}

fn solid(width: u32, height: u32, rgba: [u8; 4]) -> Vec<u8> {
    rgba.repeat((width * height) as usize)
}

fn lit(px: &[u8]) -> bool {
    px[0] > 4 || px[1] > 4 || px[2] > 4
}

/// Bounding box of `verts` after mapping them back through `to_local`.
fn local_bounds(verts: &[EntityVertex], to_local: Mat4) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for v in verts {
        let p = to_local.transform_point3(Vec3::from_array(v.pos));
        lo = lo.min(p);
        hi = hi.max(p);
    }
    (lo, hi)
}

// ------------------------------------------------------------------ skull on a head

/// A posed player and the two matrices a worn skull needs: the head part's own
/// `postRender` frame, and `LayerCustomHead`'s skull-branch matrix built on it.
fn posed_player_head(
    manager: &mut RenderManager,
    params: &AnimParams,
    base: Mat4,
) -> (Mat4, Mat4, Vec<EntityVertex>) {
    let mut body = Vec::new();
    manager.emit_with(ModelKind::Player, base, params, &mut body);
    let model = manager.model_mut(ModelKind::Player).model();
    let head = model.find("head").expect("ModelPlayer has a head");
    let head_space = base * model.get(head).post_render_matrix(params.scale);
    let layer = custom_head_matrix(
        base,
        model,
        head,
        params.scale,
        false,
        false,
        false,
        CustomHeadItem::Skull,
    );
    (head_space, layer, body)
}

/// java: `ModelSkeletonHead` is one `addBox(-4, -8, -4, 8, 8, 8)` box; `ModelHumanoidHead` adds a
/// second at the same place inflated by `0.25`. `LayerCustomHead`'s `scale(1.1875, -1.1875,
/// -1.1875)`, `renderSkull`'s `scale(-1, -1, 1)` and the model's 180-degree yaw compose to a
/// uniform 1.1875, so in the head part's own frame the skull is that box, 1.1875 times larger,
/// still hanging up from the neck pivot (model Y is down), for every head pose.
#[test]
fn a_worn_skull_is_the_head_box_enlarged_about_the_neck_pivot() {
    for (yaw, pitch) in [(0.0, 0.0), (35.0, -20.0), (-50.0, 30.0)] {
        let mut manager = RenderManager::new();
        let mut params = AnimParams::still();
        params.net_head_yaw = yaw;
        params.head_pitch = pitch;
        let base = entity_matrix(&EntityTransform {
            rel: Vec3::new(0.3, -1.6, 4.0),
            body_yaw: 150.0,
            ..Default::default()
        });
        let (head_space, layer, _) = posed_player_head(&mut manager, &params, base);
        let to_head = head_space.inverse();

        for (variant, inflate) in [(SkullVariant::Skeleton, 0.0), (SkullVariant::Player, 0.25)] {
            let mut skull = Vec::new();
            manager.emit_custom_head_skull(variant.model(), layer, &mut skull);
            assert!(!skull.is_empty(), "{variant:?} emitted nothing");
            let (lo, hi) = local_bounds(&skull, to_head);
            let k = 1.1875 / 16.0;
            let want_lo = Vec3::new(-4.0 - inflate, -8.0 - inflate, -4.0 - inflate) * k;
            let want_hi = Vec3::new(4.0 + inflate, inflate, 4.0 + inflate) * k;
            assert!(
                lo.abs_diff_eq(want_lo, 1e-4) && hi.abs_diff_eq(want_hi, 1e-4),
                "{variant:?} at yaw {yaw} pitch {pitch}: skull spans {lo:?}..{hi:?} in head space, \
                 vanilla spans {want_lo:?}..{want_hi:?}"
            );
        }
    }
}

/// The old callers posed the skull with the wearer's own `AnimParams` under the layer matrix,
/// without `renderSkull`'s flip. Reproducing that here shows what the test above rules out: the
/// skull hung from the neck pivot downward, into the torso.
#[test]
fn the_old_skull_composition_hung_below_the_neck() {
    let mut manager = RenderManager::new();
    let params = AnimParams::still();
    let base = entity_matrix(&EntityTransform {
        rel: Vec3::new(0.0, -1.6, 4.0),
        ..Default::default()
    });
    let (head_space, layer, _) = posed_player_head(&mut manager, &params, base);
    let mut old = Vec::new();
    manager.emit_with(SkullVariant::Skeleton.model(), layer, &params, &mut old);
    let (lo, hi) = local_bounds(&old, head_space.inverse());
    // Model Y is down: a positive head-space Y is below the neck pivot.
    assert!(
        lo.y > -1e-4 && hi.y > 0.5,
        "the old composition sat below the neck: {lo:?}..{hi:?}"
    );
}

/// java: the same composition on screen. A player faces the camera wearing a skull; the skull
/// encloses the head, so the top of the figure is the skull's colour and the head's colour no
/// longer appears there. With the old composition the head showed at the top as if bare.
#[test]
fn a_worn_skull_covers_the_top_of_the_wearer_on_screen() {
    const SIZE: u32 = 160;
    let Some(mut gpu) = headless(SIZE) else {
        return;
    };
    let params = AnimParams::still();
    let base = entity_matrix(&EntityTransform {
        rel: Vec3::new(0.0, -1.6, 3.0),
        // Facing the camera, which looks down +Z.
        body_yaw: 180.0,
        ..Default::default()
    });

    let frame = |gpu: &mut Gpu, old: bool| {
        let mut manager = RenderManager::new();
        let (_, layer, body) = posed_player_head(&mut manager, &params, base);
        let mut skull = Vec::new();
        if old {
            manager.emit_with(SkullVariant::Skeleton.model(), layer, &params, &mut skull);
        } else {
            manager.emit_custom_head_skull(SkullVariant::Skeleton.model(), layer, &mut skull);
        }
        let mut phase = EntityPhase::new(gpu);
        phase.upload_texture(gpu, "body", &solid(64, 64, [220, 0, 0, 255]), 64, 64);
        phase.upload_texture(gpu, "skull", &solid(64, 32, [0, 220, 0, 255]), 64, 32);
        phase.begin();
        phase.push("body", &body);
        phase.push("skull", &skull);
        phase.finish(gpu);
        render(gpu, SIZE, vec![(PhaseOrder::ENTITIES, phase)])
    };

    // Rows `top .. top + 6` of the silhouette, as (red-dominant, green-dominant) pixel counts.
    let top_rows = |pixels: &[u8]| {
        let row_lit =
            |y: u32| (0..SIZE).any(|x| lit(&pixels[((y * SIZE + x) * 4) as usize..][..4]));
        let top = (0..SIZE)
            .find(|&y| row_lit(y))
            .expect("the figure reached the frame");
        let (mut red, mut green) = (0, 0);
        for y in top..(top + 6).min(SIZE) {
            for x in 0..SIZE {
                let px = &pixels[((y * SIZE + x) * 4) as usize..][..4];
                if px[0] > px[1] + 20 {
                    red += 1;
                } else if px[1] > px[0] + 20 {
                    green += 1;
                }
            }
        }
        (red, green)
    };

    let (red, green) = top_rows(&frame(&mut gpu, false));
    eprintln!("top rows with the skull: {green} skull pixels, {red} head pixels");
    assert!(
        green > 20 && red == 0,
        "the skull must cap the figure: {green} skull pixels, {red} head pixels in the top rows"
    );
    let (old_red, old_green) = top_rows(&frame(&mut gpu, true));
    eprintln!("top rows, old composition: {old_green} skull pixels, {old_red} head pixels");
    assert!(
        old_red > 20,
        "negative control: the old composition left the bare head on top ({old_red} head pixels)"
    );
}

// ------------------------------------------------------------------ slime gel vs water

/// java: `RenderSlime`'s `ModelSlime(16)` body, then `LayerSlimeGel`'s `ModelSlime(0)` shell with
/// `blendFunc(770, 771)` and the depth mask still on (`RendererLivingEntity:174`). Entities are
/// drawn before translucent terrain (`EntityRenderer:1402` vs `:1467`), and water draws with
/// `depthMask(false)` and an ordinary depth test, so a water surface behind the slime never
/// shows through the gel. Here a translucent "water" quad behind the slime is drawn in the later
/// `TERRAIN_TRANSLUCENT` phase. Every pixel the slime covered must be unchanged by it, for the
/// gel's ordinary blend and for its hurt-flash variant. The old state (`Translucent`, depth
/// writes off) is replayed as the negative control and must let the water through.
#[test]
fn slime_gel_hides_translucent_terrain_drawn_after_the_entity_pass() {
    const SIZE: u32 = 128;
    const SLIME: &str = "textures/entity/slime/slime.png";
    let rm = require!(resources(), "1.8.9.jar");
    let (slime, sw, sh) = require!(load_png(&rm, SLIME), "slime.png");
    let Some(mut gpu) = headless(SIZE) else {
        return;
    };

    let size = 2;
    let mut params = AnimParams::still();
    params.entity.variant = size;
    let base = entity_matrix_scaled(
        &EntityTransform {
            rel: Vec3::new(0.0, -0.55, 2.4),
            body_yaw: 200.0,
            ..Default::default()
        },
        slime_pre_render_scale(size, 0.0),
    );
    let mut manager = RenderManager::new();
    let mut body = Vec::new();
    manager.emit_with(ModelKind::Slime, base, &params, &mut body);
    let mut gel = Vec::new();
    manager.emit_layer(LayerKind::SlimeGel, &body, &params, base, &mut gel);
    assert!(!body.is_empty() && !gel.is_empty());

    // A screen-filling translucent blue quad behind the slime, facing the camera.
    let corners = [
        (-4.0, -4.0),
        (4.0, -4.0),
        (4.0, 4.0),
        (-4.0, -4.0),
        (4.0, 4.0),
        (-4.0, 4.0),
    ];
    let water: Vec<EntityVertex> = corners
        .into_iter()
        .map(|(x, y)| EntityVertex {
            pos: [x, y, 4.0],
            uv: [0.5, 0.5],
            normal: [0.0, 0.0, -1.0],
            color: EntityVertex::WHITE,
        })
        .collect();

    let slime_phase = |gpu: &Gpu, gel_blend: LayerBlend| {
        let mut phase = EntityPhase::new(gpu);
        phase.upload_texture(gpu, SLIME, &slime, sw, sh);
        phase.begin();
        phase.push_blended(SLIME, LayerBlend::Opaque, &body);
        phase.push_blended(SLIME, gel_blend, &gel);
        phase.finish(gpu);
        phase
    };
    let water_phase = |gpu: &Gpu| {
        let mut phase = EntityPhase::new(gpu);
        phase.upload_texture(gpu, "water", &[40, 80, 230, 180], 1, 1);
        phase.begin();
        phase.push_blended("water", LayerBlend::Translucent, &water);
        phase.finish(gpu);
        phase
    };

    // (slime pixels the later water changed, slime pixels, water-only pixels showing water)
    let mut measure = |gel_blend: LayerBlend| {
        let phases = vec![(PhaseOrder::ENTITIES, slime_phase(&gpu, gel_blend))];
        let alone = render(&mut gpu, SIZE, phases);
        let phases = vec![
            (PhaseOrder::ENTITIES, slime_phase(&gpu, gel_blend)),
            (PhaseOrder::TERRAIN_TRANSLUCENT, water_phase(&gpu)),
        ];
        let with_water = render(&mut gpu, SIZE, phases);
        let (mut changed, mut covered, mut open_water) = (0, 0, 0);
        for (a, b) in alone.chunks_exact(4).zip(with_water.chunks_exact(4)) {
            if lit(a) {
                covered += 1;
                if a.iter().zip(b).take(3).any(|(x, y)| x.abs_diff(*y) > 2) {
                    changed += 1;
                }
            } else if b[2] > b[0] + 40 {
                open_water += 1;
            }
        }
        (changed, covered, open_water)
    };

    let gel_blend = LayerKind::SlimeGel.blend();
    let hurt_blend = gel_blend
        .living_hurt()
        .expect("LayerSlimeGel combines with the hurt flash");
    for blend in [gel_blend, hurt_blend] {
        let (changed, covered, open_water) = measure(blend);
        eprintln!("{blend:?}: {changed} of {covered} slime pixels changed, {open_water} water");
        assert!(
            covered > 300,
            "{blend:?}: the slime covers {covered} pixels"
        );
        assert!(
            open_water > (SIZE * SIZE / 2) as usize,
            "{blend:?}: the water quad reached the frame around the slime ({open_water})"
        );
        assert_eq!(
            changed, 0,
            "{blend:?}: water drawn after the entity pass showed through {changed} of the \
             slime's {covered} pixels; the gel must write depth as in vanilla"
        );
    }

    let (old_changed, covered, _) = measure(LayerBlend::Translucent);
    eprintln!("old Translucent gel: {old_changed} of {covered} slime pixels changed");
    assert!(
        old_changed > 50,
        "negative control: with depth writes off the water drew over the gel \
         ({old_changed} of {covered} slime pixels changed)"
    );
}

// ------------------------------------------------------------------ dropped items

/// Two coplanar screen-filling quads wound opposite ways, like a pair of `ItemModelGenerator`
/// side quads. Both carry the same normal so the lighting cannot tell them apart; only the
/// winding (and so the cull) differs.
fn coplanar_pair(z: f32) -> [Vec<EntityVertex>; 2] {
    let r = 4.0;
    let v = |x: f32, y: f32| EntityVertex {
        pos: [x, y, z],
        uv: [0.5, 0.5],
        normal: [0.0, 0.0, -1.0],
        color: EntityVertex::WHITE,
    };
    [
        vec![v(-r, -r), v(r, -r), v(r, r), v(-r, -r), v(r, r), v(-r, r)],
        vec![v(-r, -r), v(r, r), v(r, -r), v(-r, -r), v(-r, r), v(r, r)],
    ]
}

/// java: `RenderEntityItem.doRender` never disables the world pass's `GL_CULL_FACE`, so of a
/// coplanar pair of opposite-facing side quads only the camera-facing one is drawn, by the model
/// pass and by `RenderItem.renderEffect`'s `GL_EQUAL` foil alike. Drawn two-sided, the foil lands
/// on such a pixel twice: the speckled double glint on a dropped enchanted bow's string.
#[test]
fn dropped_item_foil_lands_once_on_a_coplanar_opposite_facing_pair() {
    const SIZE: u32 = 64;
    let Some(mut gpu) = headless(SIZE) else {
        return;
    };
    let pair = coplanar_pair(3.0);
    let centre = |gpu: &mut Gpu, quads: &[&Vec<EntityVertex>], base, glint| {
        let mut phase = EntityPhase::new(gpu);
        phase.upload_texture(gpu, "item", &[90, 60, 30, 255], 1, 1);
        phase.upload_texture(gpu, "glint", &[160, 160, 160, 255], 1, 1);
        phase.begin();
        for quad in quads {
            phase.push_blended("item", base, quad);
        }
        for quad in quads {
            phase.push_blended("glint", glint, quad);
        }
        phase.finish(gpu);
        let pixels = render(gpu, SIZE, vec![(PhaseOrder::ENTITIES, phase)]);
        let i = (((SIZE / 2) * SIZE + SIZE / 2) * 4) as usize;
        [pixels[i], pixels[i + 1], pixels[i + 2]]
    };
    let (base, glint) = (LayerBlend::DroppedItem, LayerBlend::DroppedItemGlint);
    let pair_culled = centre(&mut gpu, &[&pair[0], &pair[1]], base, glint);
    let first = centre(&mut gpu, &[&pair[0]], base, glint);
    let second = centre(&mut gpu, &[&pair[1]], base, glint);
    let black = [0u8; 3];
    assert!(
        (first == black) != (second == black),
        "exactly one winding survives the cull: {first:?} / {second:?}"
    );
    let survivor = if first == black { second } else { first };
    assert_eq!(pair_culled, survivor, "the culled pair is foiled once");

    let two_sided = centre(
        &mut gpu,
        &[&pair[0], &pair[1]],
        LayerBlend::Opaque,
        LayerBlend::Glint,
    );
    eprintln!("culled pair {pair_culled:?}, two-sided pair {two_sided:?}");
    assert!(
        two_sided.iter().zip(&survivor).all(|(two, one)| two >= one)
            && two_sided.iter().zip(&survivor).any(|(two, one)| two > one),
        "negative control: two-sided, the foil lands twice ({two_sided:?} vs {survivor:?})"
    );
}

/// The cull keeps the faces the camera sees on real baked geometry. A dropped stone block (a
/// closed `ModelBakery` cube with no coplanar faces) seen at an angle is pixel-identical with and
/// without it. A flat `ItemModelGenerator` sword seen head-on differs only in the thin side-quad
/// slivers perspective exposes, where two-sided drawing lets the wrong one of a coplanar pair
/// win (the defect itself). The same meshes with their winding reversed, which the cull then
/// empties of camera-facing faces, are the control that shows an inverted cull would be caught.
#[test]
fn dropped_item_cull_keeps_the_camera_facing_faces_of_baked_items() {
    use mc_render::entity::item::{emit_dropped_item, DroppedItem};

    const SIZE: u32 = 96;
    let rm = require!(resources(), "1.8.9.jar");
    let (assets, _report) = mc_assets::bake_all(&rm, 2048).expect("bake_all");
    let atlas = assets.atlas.clone();
    mc_assets::model::install(assets);
    let Some(mut gpu) = headless(SIZE) else {
        return;
    };
    let frame = |gpu: &mut Gpu, verts: &[EntityVertex], blend| {
        let mut phase = EntityPhase::new(gpu);
        phase.upload_texture(gpu, "atlas", &atlas.texture, atlas.w, atlas.h);
        phase.begin();
        phase.push_blended("atlas", blend, verts);
        phase.finish(gpu);
        render(gpu, SIZE, vec![(PhaseOrder::ENTITIES, phase)])
    };

    // (item, age: 37 ticks turns the block to show three faces, 0 keeps the sword head-on)
    for (name, age) in [("stone", 37), ("diamond_sword", 0)] {
        let id = mc_data::item_by_name(name).expect("a vanilla item");
        let model = mc_assets::item_model_override_for(id, 0).model;
        assert!(!model.is_empty(), "{name} bakes to geometry");
        let item = DroppedItem {
            rel: Vec3::new(0.0, -0.2, 0.8),
            age,
            gui_3d: model.gui_3d,
            ground_scale: model.transforms.ground.scale,
            ..DroppedItem::default()
        };
        let mut verts = Vec::new();
        emit_dropped_item(&item, model, Mat4::IDENTITY, 0.0, &mut verts);
        let reversed: Vec<EntityVertex> = verts
            .chunks_exact(3)
            .flat_map(|tri| [tri[0], tri[2], tri[1]])
            .collect();

        let two_sided = frame(&mut gpu, &verts, LayerBlend::Opaque);
        let covered = two_sided.chunks_exact(4).filter(|px| lit(px)).count();
        assert!(
            covered > 50,
            "the dropped {name} reached the frame ({covered} pixels)"
        );
        let differing = |a: &[u8], b: &[u8]| {
            a.chunks_exact(4)
                .zip(b.chunks_exact(4))
                .filter(|(p, q)| p[..3] != q[..3])
                .count()
        };
        let culled = differing(
            &frame(&mut gpu, &verts, LayerBlend::DroppedItem),
            &two_sided,
        );
        let inverted = differing(
            &frame(&mut gpu, &reversed, LayerBlend::DroppedItem),
            &two_sided,
        );
        eprintln!("{name}: {covered} pixels, culled differs in {culled}, inverted in {inverted}");
        if model.gui_3d {
            assert_eq!(
                culled, 0,
                "culling a closed cube must not change what the camera sees"
            );
            assert!(
                inverted * 2 > covered,
                "control: culling the cube's camera-facing faces instead changes most of it"
            );
        } else {
            assert!(
                culled * 10 < covered,
                "culling a head-on flat item changes only side-quad slivers ({culled})"
            );
            // The back layer is the mirrored sprite, so from the front it nearly matches the
            // front layer; the inverted cull still shows clearly, through the depth offset and
            // the exposed far sides.
            assert!(
                inverted * 10 > covered,
                "control: culling the {name}'s camera-facing faces instead changes it visibly"
            );
        }
    }
}
