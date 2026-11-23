//! A05 integration tests: the tile-entity pass, on a real GPU and against the real jar.
//!
//! The audit's verdict for this area was "100% dead code" — `entity/tile.rs` and
//! `models/tileent.rs` were faithful and completely unreachable, and every TESR-only block was
//! an invisible hole in the world. The unit tests beside the code prove the geometry is *built*;
//! these prove it reaches a frame, which is the only claim that would have failed before.
//!
//! Every assertion here is one that could not pass if the pass were unregistered or fed nothing:
//! pixels present where a chest is and absent where it is not, a double chest wider on screen
//! than a single one, and a beam that reaches the top of the frame.
//!
//! Skips cleanly (`SKIP:`) with no GPU adapter or no `1.8.9.jar`.

use std::path::{Path, PathBuf};

use glam::{Mat4, Vec3};
use mc_assets::{ResourceLocation, ResourceManager, ZipResourcePack};
use mc_render::entity::tile::{self, is_christmas, BeaconPhase, SkullMount, TileKind};
use mc_render::entity::tilepass::{light_brightness, lit, TileBlend, TileModels, TilePhase};
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

/// Draws one frame at `size` and returns the lit pixel coordinates.
fn draw(gpu: &mut Gpu, phases: PhaseList, size: u32) -> Vec<(u32, u32)> {
    let mut phases = phases;
    phases.clear_color = wgpu::Color::BLACK;
    let mut cam = Camera::default();
    cam.set_viewport(size, size);
    cam.update();
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &cam, 0.0);
    frame.present(&gpu.queue);
    let pixels = gpu.read_pixels().expect("read back the frame");
    (0..size)
        .flat_map(|y| (0..size).map(move |x| (x, y)))
        .filter(|(x, y)| {
            let i = ((y * size + x) * 4) as usize;
            pixels[i] > 8 || pixels[i + 1] > 8 || pixels[i + 2] > 8
        })
        .collect()
}

// ------------------------------------------------------- the texture-path audit

/// The three chest textures `TileKind` could not name before A05-20, plus the book.
#[test]
fn the_chest_textures_a05_20_added_exist_in_the_vanilla_jar() {
    let rm = require!(resources(), "1.8.9.jar");
    for path in [
        "textures/entity/chest/trapped_double.png",
        "textures/entity/chest/christmas.png",
        "textures/entity/chest/christmas_double.png",
        "textures/entity/enchanting_table_book.png",
    ] {
        let l = ResourceLocation::parse(path).expect("path parses");
        assert!(rm.contains(&l), "{path} is not in the jar");
    }
    // The variant itself resolves, which is the half a string audit alone would miss.
    assert_eq!(
        TileKind::LargeTrappedChest.texture(),
        Some("textures/entity/chest/trapped_double.png")
    );
    assert_eq!(
        TileKind::chest(true, true),
        TileKind::LargeTrappedChest,
        "a double trapped chest must not fall back to LargeChest"
    );
    // java: `calendar.get(2) + 1 == 12 && calendar.get(5) >= 24 && <= 26`
    assert!(is_christmas(12, 25) && is_christmas(12, 24) && is_christmas(12, 26));
    assert!(!is_christmas(12, 23) && !is_christmas(12, 27) && !is_christmas(11, 25));
}

// ------------------------------------------------------------- the pass reaches a frame

/// **The** A05-02 test: a chest drawn through the registered phase puts pixels on the screen
/// where the block is, and a world with no chest in it puts none there.
///
/// Before this change, `RenderStack::install` registered no tile-entity phase of any kind and the
/// mesher deliberately emits nothing for `RenderType::TileEntity`, so the second half of this
/// assertion was the *only* observable behaviour: a hole.
#[test]
fn a_chest_draws_pixels_where_an_empty_world_draws_none() {
    let mut gpu = require!(headless(160, 160), "a GPU adapter");
    let mut models = TileModels::new(false);

    // Empty frame first, from an identically-configured phase with nothing pushed.
    let empty = {
        let mut phase = TilePhase::new(&gpu);
        phase.begin();
        phase.finish(&gpu);
        let mut phases = PhaseList::new();
        phases.insert(PhaseOrder::ENTITIES, Box::new(phase));
        draw(&mut gpu, phases, 160)
    };
    assert!(
        empty.is_empty(),
        "an empty tile pass must draw nothing, got {} pixels",
        empty.len()
    );

    let mut phase = TilePhase::new(&gpu);
    phase.upload_texture(&gpu, "chest", &[220, 170, 90, 255], 1, 1);
    let mut verts = Vec::new();
    // A chest one block below eye level, four blocks in front, facing south.
    models.emit_chest(
        Vec3::new(-0.5, -1.6, 4.0),
        2,
        false,
        false,
        Vec3::ZERO,
        0.0,
        0.0,
        &[],
        0.0,
        &mut verts,
    );
    assert!(!verts.is_empty());
    phase.begin();
    phase.push("chest", &lit(&verts, 1.0));
    phase.finish(&gpu);
    assert_eq!(phase.batch_count(), 1);
    assert_eq!(phase.vertex_count(), verts.len());

    let mut phases = PhaseList::new();
    phases.insert(PhaseOrder::ENTITIES, Box::new(phase));
    let drawn = draw(&mut gpu, phases, 160);

    assert!(
        drawn.len() > 300,
        "a chest four blocks away should cover a few hundred pixels, got {}",
        drawn.len()
    );
    assert!(
        drawn.len() < 160 * 160 / 3,
        "it should not fill the screen, got {}",
        drawn.len()
    );
    let xs: Vec<u32> = drawn.iter().map(|p| p.0).collect();
    let ys: Vec<u32> = drawn.iter().map(|p| p.1).collect();
    let (x0, x1) = (*xs.iter().min().unwrap(), *xs.iter().max().unwrap());
    let (y0, y1) = (*ys.iter().min().unwrap(), *ys.iter().max().unwrap());
    assert!(
        (60..=100).contains(&((x0 + x1) / 2)),
        "chest should straddle the centre, spans x {x0}..{x1}"
    );
    // It sits a block below the eye, so every pixel must be below the horizon. This is the
    // orientation proof: `chest_matrix`'s `scale(1, -1, -1)` on a Y-down model is exactly the
    // step that, got wrong, puts the chest above the camera or inside it.
    assert!(y0 > 80, "the chest should be below the horizon, y0 = {y0}");
    // A chest is a squat 14x10x14 box seen slightly from above, so its footprint is roughly
    // square — never a tall column, which is what a mis-scaled or mis-rotated model produces.
    assert!(
        (y1 - y0) < 2 * (x1 - x0),
        "a chest is not a tall column: {}x{}",
        x1 - x0,
        y1 - y0
    );
}

/// A double chest covers visibly more screen than a single one — the A05-07 branch that was
/// structurally present and unreachable.
#[test]
fn a_double_chest_covers_more_screen_than_a_single_one() {
    let mut gpu = require!(headless(160, 160), "a GPU adapter");
    let mut models = TileModels::new(false);

    let span = |gpu: &mut Gpu, models: &mut TileModels, double: bool| {
        let mut phase = TilePhase::new(gpu);
        phase.upload_texture(gpu, "chest", &[220, 170, 90, 255], 1, 1);
        let mut verts = Vec::new();
        models.emit_chest(
            Vec3::new(-0.5, -1.6, 5.0),
            2,
            false,
            double,
            Vec3::ZERO,
            0.0,
            0.0,
            &[],
            0.0,
            &mut verts,
        );
        phase.begin();
        phase.push("chest", &lit(&verts, 1.0));
        phase.finish(gpu);
        let mut phases = PhaseList::new();
        phases.insert(PhaseOrder::ENTITIES, Box::new(phase));
        let drawn = draw(gpu, phases, 160);
        let xs: Vec<u32> = drawn.iter().map(|p| p.0).collect();
        (
            drawn.len(),
            xs.iter().max().copied().unwrap_or(0) - xs.iter().min().copied().unwrap_or(0),
        )
    };

    let (single_px, single_w) = span(&mut gpu, &mut models, false);
    let (double_px, double_w) = span(&mut gpu, &mut models, true);
    assert!(single_px > 200 && double_px > 200);
    assert!(
        double_w > single_w + 10,
        "the large chest should be visibly wider on screen: {double_w} vs {single_w}"
    );
}

/// An open chest lid changes the pixels. Proves `chest_lid_angle` — which had no caller in the
/// whole tree — now reaches the GPU.
#[test]
fn opening_the_lid_changes_the_frame() {
    let mut gpu = require!(headless(128, 128), "a GPU adapter");
    let mut models = TileModels::new(false);

    let frame = |gpu: &mut Gpu, models: &mut TileModels, lid: f32| {
        let mut phase = TilePhase::new(gpu);
        phase.upload_texture(gpu, "chest", &[255, 255, 255, 255], 1, 1);
        let mut verts = Vec::new();
        models.emit_chest(
            Vec3::new(-0.5, -1.2, 2.5),
            2,
            false,
            false,
            Vec3::ZERO,
            lid,
            lid,
            &[],
            1.0,
            &mut verts,
        );
        phase.begin();
        phase.push("chest", &lit(&verts, 1.0));
        phase.finish(gpu);
        let mut phases = PhaseList::new();
        phases.insert(PhaseOrder::ENTITIES, Box::new(phase));
        draw(gpu, phases, 128)
    };

    let shut = frame(&mut gpu, &mut models, 0.0);
    let open = frame(&mut gpu, &mut models, 1.0);
    assert!(!shut.is_empty() && !open.is_empty());
    let differing = shut.iter().filter(|p| !open.contains(p)).count()
        + open.iter().filter(|p| !shut.contains(p)).count();
    assert!(
        differing > 50,
        "an open lid must change the silhouette, only {differing} pixels differ"
    );
}

/// The beacon beam reaches the top of the frame — the phase the audit found instantiated only
/// inside its own tests.
#[test]
fn a_beacon_beam_reaches_the_top_of_the_frame() {
    let mut gpu = require!(headless(128, 128), "a GPU adapter");
    let mut phase = BeaconPhase::new(&gpu);
    // A vertical white gradient, so the beam is bright everywhere the alpha test passes.
    phase.upload_texture(&gpu, &[255, 255, 255, 255], 1, 1);

    let beam = tile::beacon_beam(0, 0.0, 64, 1.0);
    let mut inner = Vec::new();
    let mut outer = Vec::new();
    // A beacon four blocks in front, its base two blocks below the eye, running 64 blocks up.
    tile::emit_beacon_inner_shell([-0.5, 4.0], -2.0, 62.0, &beam, [1.0, 1.0, 1.0], &mut inner);
    tile::emit_beacon_outer_shell([-0.5, 4.0], -2.0, 62.0, &beam, [1.0, 1.0, 1.0], &mut outer);
    assert_eq!(inner.len(), 24, "four quads of six vertices");
    assert_eq!(outer.len(), 24);

    phase.begin();
    phase.push_inner(&inner);
    phase.push_outer(&outer);
    phase.finish(&gpu);
    assert_eq!(phase.vertex_counts(), (24, 24));

    let mut phases = PhaseList::new();
    phases.insert(PhaseOrder::ENTITIES, Box::new(phase));
    let drawn = draw(&mut gpu, phases, 128);
    assert!(!drawn.is_empty(), "the beam drew nothing");
    let ys: Vec<u32> = drawn.iter().map(|p| p.1).collect();
    let top = *ys.iter().min().unwrap();
    let bottom = *ys.iter().max().unwrap();
    assert_eq!(top, 0, "a 64-block beam must leave the top of the frame");
    assert!(
        bottom > 80,
        "and start below the eye, lowest row is {bottom}"
    );
    // It must be a *column*, not a wash: narrow in x, tall in y.
    let xs: Vec<u32> = drawn.iter().map(|p| p.0).collect();
    let width = xs.iter().max().unwrap() - xs.iter().min().unwrap();
    assert!(
        width < 40,
        "the beam should be a narrow column, got {width}"
    );
}

/// The lightmap channel really modulates the frame — the first third, which had no
/// home before this phase existed.
#[test]
fn a_dark_tile_entity_is_dimmer_than_a_lit_one() {
    let mut gpu = require!(headless(96, 96), "a GPU adapter");
    let mut models = TileModels::new(false);

    let mean = |gpu: &mut Gpu, models: &mut TileModels, light: f32| {
        let mut phase = TilePhase::new(gpu);
        phase.upload_texture(gpu, "flat", &[255, 255, 255, 255], 1, 1);
        let mut verts = Vec::new();
        models.emit_chest(
            Vec3::new(-0.5, -1.2, 2.0),
            2,
            false,
            false,
            Vec3::ZERO,
            0.0,
            0.0,
            &[],
            0.0,
            &mut verts,
        );
        phase.begin();
        phase.push("flat", &lit(&verts, light));
        phase.finish(gpu);
        let mut phases = PhaseList::new();
        phases.clear_color = wgpu::Color::BLACK;
        phases.insert(PhaseOrder::ENTITIES, Box::new(phase));
        let mut cam = Camera::default();
        cam.set_viewport(96, 96);
        cam.update();
        let frame = gpu.acquire().expect("acquire");
        phases.render(gpu, frame.view(), &cam, 0.0);
        frame.present(&gpu.queue);
        let pixels = gpu.read_pixels().expect("read back");
        let sum: u64 = pixels.chunks(4).map(|p| u64::from(p[0])).sum();
        sum / (96 * 96)
    };

    let bright = mean(&mut gpu, &mut models, light_brightness(15));
    let dark = mean(&mut gpu, &mut models, light_brightness(0));
    assert!(bright > 0, "the lit chest drew nothing");
    assert!(
        dark * 3 < bright,
        "a chest in the dark should be much dimmer: {dark} vs {bright}"
    );
}

/// Every kind the pass can draw survives a real draw with real jar textures.
#[test]
fn every_tile_entity_kind_survives_a_real_draw() {
    let mut gpu = require!(headless(128, 128), "a GPU adapter");
    let rm = require!(resources(), "1.8.9.jar");
    let mut models = TileModels::new(false);
    let mut phase = TilePhase::new(&gpu);

    let mut total = 0;
    phase.begin();
    let push = |phase: &mut TilePhase, gpu: &Gpu, texture: &str, verts: &[_]| {
        assert!(
            phase.load_texture(gpu, &rm, texture),
            "{texture} did not load from the jar"
        );
        phase.push(texture, &lit(verts, 1.0));
    };

    let mut verts = Vec::new();
    for (trapped, double) in [(false, false), (true, false), (false, true), (true, true)] {
        verts.clear();
        let t = models.emit_chest(
            Vec3::new(-0.5, -1.6, 3.0),
            2,
            trapped,
            double,
            Vec3::ZERO,
            0.5,
            0.5,
            &[],
            0.5,
            &mut verts,
        );
        total += verts.len();
        push(&mut phase, &gpu, t, &verts);
    }
    verts.clear();
    let t = models.emit_ender_chest(Vec3::new(-0.5, -1.6, 3.0), 3, 0.0, 0.0, 0.0, &mut verts);
    total += verts.len();
    push(&mut phase, &gpu, t, &verts);

    for (standing, meta) in [(true, 4u8), (false, 2u8)] {
        verts.clear();
        let (t, _) = models.emit_sign(Vec3::new(-0.5, -1.0, 3.0), standing, meta, &mut verts);
        total += verts.len();
        push(&mut phase, &gpu, t, &verts);

        verts.clear();
        let t = models.emit_banner(
            Vec3::new(-0.5, -1.0, 3.0),
            standing,
            meta,
            1,
            64,
            2,
            120,
            0.5,
            &mut verts,
        );
        total += verts.len();
        push(&mut phase, &gpu, t, &verts);
    }

    for skull_type in 0..=4u8 {
        verts.clear();
        let t = models.emit_skull(Vec3::new(-0.5, -1.0, 3.0), 1, skull_type, 3, &mut verts);
        total += verts.len();
        // java: `case 3` has no fixed texture — it wears the profile's skin.
        let t = t.unwrap_or("textures/entity/steve.png");
        push(&mut phase, &gpu, t, &verts);
    }

    verts.clear();
    let base = tile::enchant_book_matrix(Vec3::new(-0.5, -1.0, 3.0), 40, 0.0, 0.5, 0.5);
    let t = models.emit_book(base, 40, 0.5, (0.2, 0.7), 0.6, &mut verts);
    total += verts.len();
    push(&mut phase, &gpu, t, &verts);

    phase.finish(&gpu);
    assert_eq!(phase.vertex_count(), total);
    assert!(total > 1000, "only {total} vertices across every kind");

    let mut phases = PhaseList::new();
    phases.insert(PhaseOrder::ENTITIES, Box::new(phase));
    let drawn = draw(&mut gpu, phases, 128);
    assert!(!drawn.is_empty(), "every kind together drew nothing");
}

/// The skull mount decode, which had no caller before this change.
#[test]
fn skull_metadata_selects_the_mounting_face() {
    // java: `EnumFacing.getFront(meta & 7)` then `renderSkull`'s switch — UP is the floor and
    // DOWN falls through the EAST/default arm.
    assert_eq!(SkullMount::from_meta(1), SkullMount::Floor);
    assert_eq!(SkullMount::from_meta(2), SkullMount::North);
    assert_eq!(SkullMount::from_meta(3), SkullMount::South);
    assert_eq!(SkullMount::from_meta(4), SkullMount::West);
    assert_eq!(SkullMount::from_meta(5), SkullMount::East);
    assert_eq!(
        SkullMount::from_meta(0),
        SkullMount::East,
        "DOWN -> default"
    );
    // The `& 7` matters: the top bit of a skull's metadata is not part of the facing.
    assert_eq!(SkullMount::from_meta(9), SkullMount::Floor);
    // java: `(float)(te.getSkullRotation() * 360) / 16.0F`
    assert_eq!(tile::skull_rotation_degrees(0), 0.0);
    assert_eq!(tile::skull_rotation_degrees(4), 90.0);
    assert_eq!(tile::skull_rotation_degrees(8), 180.0);
}

/// Pixel counts for [`book_cover_key`]'s two cover colours.
#[derive(Debug, Default)]
struct CoverSides {
    /// `ModelBox` quad 4 (`-Z`) of a cover: magenta.
    quad4: usize,
    /// `ModelBox` quad 5 (`+Z`) of a cover: red.
    quad5: usize,
}

/// A 64x32 `enchanting_table_book.png` stand-in keyed by cover side.
///
/// `coverRight` and `coverLeft` are 6x10x0 boxes at texture offsets `(0, 0)` and `(16, 0)`. A
/// zero-depth `ModelBox` has only its two Z quads: quad 4 samples `u = texU..texU+6`, quad 5
/// `u = texU+6..texU+12`. Those are magenta and red; everything else is white.
fn book_cover_key() -> Vec<u8> {
    let mut rgba = vec![255u8; 64 * 32 * 4];
    for v in 0..10usize {
        for u in 0..28usize {
            let colour = match u {
                0..=5 | 16..=21 => [255, 0, 255, 255],
                6..=11 | 22..=27 => [255, 0, 0, 255],
                _ => continue,
            };
            let i = (v * 64 + u) * 4;
            rgba[i..i + 4].copy_from_slice(&colour);
        }
    }
    rgba
}

/// Draws `verts` through one tile phase under `blend`, then counts the two cover colours.
fn draw_cover_sides(
    gpu: &mut Gpu,
    verts: &[mc_render::entity::EntityVertex],
    blend: TileBlend,
) -> CoverSides {
    let size = 160;
    let mut phase = TilePhase::new(gpu);
    phase.upload_texture(gpu, "book-key", &book_cover_key(), 64, 32);
    phase.begin();
    phase.push_blended("book-key", blend, &lit(verts, 1.0));
    phase.finish(gpu);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::ENTITIES, Box::new(phase));
    let mut cam = Camera::default();
    cam.set_viewport(size, size);
    cam.update();
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &cam, 0.0);
    frame.present(&gpu.queue);
    let pixels = gpu.read_pixels().expect("read back the frame");
    let mut sides = CoverSides::default();
    for px in pixels.chunks_exact(4) {
        let (r, g, b) = (u32::from(px[0]), u32::from(px[1]), u32::from(px[2]));
        if r > 40 && r > 2 * g && b > 40 && b > 2 * g {
            sides.quad4 += 1;
        } else if r > 40 && r > 2 * g && r > 2 * b {
            sides.quad5 += 1;
        }
    }
    sides
}

/// "It looks like theres z-fighting on the book on top of the
/// enchanting table block".
///
/// java: `TileEntityEnchantmentTableRenderer.renderTileEntityAt:64` calls
/// `GlStateManager.enableCull()` immediately before `ModelBook.render`. A cover is a zero-depth
/// box, so its quad 4 and quad 5 are coplanar with opposite windings. With the cull only the
/// camera-facing quad draws; without it both rasterise at one depth and whichever wins the depth
/// tie shows — the z-fighting.
///
/// The book is drawn open (`bookSpread` 1), turned so the side the covers open towards faces the
/// camera: what is visible of each cover is its inner side, which is quad 4. Quad 5 (red) on
/// screen is the far side of a cover bleeding through.
#[test]
fn the_enchanting_book_culls_its_cover_backs_like_vanilla() {
    let mut gpu = require!(headless(160, 160), "a GPU adapter");
    let mut models = TileModels::new(false);
    let mut verts = Vec::new();
    // 1.2 blocks in front of the eye; the model's +X (the opening of an open `ModelBook`) is
    // turned to face back at the camera.
    let base = Mat4::from_translation(Vec3::new(0.0, 0.0, 1.2))
        * Mat4::from_rotation_y(90f32.to_radians());
    models.emit_book(base, 0, 0.0, (0.0, 1.0), 1.0, &mut verts);

    let culled = draw_cover_sides(&mut gpu, &verts, TileBlend::OpaqueCulled);
    eprintln!("culled: {culled:?}");
    assert!(
        culled.quad4 > 200,
        "the covers' inner sides must face the camera: {culled:?}"
    );
    assert_eq!(
        culled.quad5, 0,
        "a culled cover never shows its back through its front: {culled:?}"
    );

    // Negative control: the pre-fix state, every TESR on the cull-less opaque pipeline.
    let unculled = draw_cover_sides(&mut gpu, &verts, TileBlend::Opaque);
    eprintln!("unculled: {unculled:?}");
    assert!(
        unculled.quad5 > 0,
        "without the cull the coplanar back quads win depth ties: {unculled:?}"
    );
}
