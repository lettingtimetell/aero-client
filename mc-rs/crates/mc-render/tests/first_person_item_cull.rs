//! Bug report: "z fighting on my enchanted bow on the diagonal string part",
//! seen with the bow held in first person.
//!
//! `ItemModelGenerator` gives a generated item one thin side quad per pixel-boundary span, and it
//! merges every span of a row (or column) into one quad even across gaps. A row's UP side and
//! the row above's DOWN side therefore lie in exactly the same plane, face opposite ways, and
//! overlap along the bow's diagonal string, where one of them samples the opaque string texel
//! and the other a transparent one.
//!
//! Vanilla draws the held item with `GL_CULL_FACE` still on: `EntityRenderer.renderWorldPass`
//! calls `enableCull()` before `renderHand`, only `ItemRenderer.renderPlayerArm(s)` bracket the
//! *arm* with `disableCull`/`enableCull`, and `RenderItem.renderItemModelTransform` merely picks
//! `cullFace(GL_BACK/GL_FRONT)`. So only the side facing the camera is drawn, and
//! `RenderItem.renderEffect`'s `GL_EQUAL` foil (whose texture is opaque, so the alpha test never
//! removes it) lands exactly once per foil pass on each item pixel. Drawn two-sided, the hidden
//! partner's foil also passes the equal test wherever the two coplanar quads rasterise to the
//! same depth: a speckled, doubled foil along the string.
//!
//! The render below uses the real 1.8.9 bow, enchanted, with every atlas texel's colour blacked
//! out (alpha kept, so the base pass still writes depth exactly where vanilla's alpha test
//! allows) and the phase's white 1x1 fallback foil texture. The foil's colour is `0x8040CC` and
//! its blend is `GL_SRC_COLOR, GL_ONE`, so each foil fragment adds `(128/255)^2` to red: a
//! pixel's red channel counts the foil fragments that landed on it (64, 128, 192, 255).
//!
//! Skips (loudly) without a wgpu adapter or the vanilla jar; set `MC_JARS_DIR`.

use std::path::{Path, PathBuf};

use mc_assets::{Atlas, ResourceManager, ZipResourcePack};
use mc_render::pipeline::FirstPersonPhase;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

const W: u32 = 320;
const H: u32 = 180;

fn gpu() -> Option<Gpu> {
    match Gpu::headless(
        W,
        H,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => Some(gpu),
        Err(RenderError::NoAdapter) => {
            eprintln!("SKIP: no wgpu adapter available on this machine");
            None
        }
        Err(error) => panic!("headless context creation failed: {error}"),
    }
}

fn jars_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("MC_JARS_DIR") {
        let dir = PathBuf::from(dir);
        return dir.is_dir().then_some(dir);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let dir = root.join("MCP-919-main").join("jars");
    dir.is_dir().then_some(dir)
}

/// Bake and install the vanilla jar's models and atlas once for this test process.
fn baked_atlas() -> Option<Atlas> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    if !jar.is_file() {
        return None;
    }
    let mut rm = ResourceManager::new();
    rm.push_pack(Box::new(ZipResourcePack::open(&jar).ok()?));
    let (baked, _report) = mc_assets::bake_all(&rm, 4096).expect("bake_all against the real jar");
    let atlas = baked.atlas.clone();
    mc_assets::install(baked);
    Some(atlas)
}

/// Render the held, enchanted `item` once and return the red channel of every pixel.
fn render_enchanted_held_item(gpu: &mut Gpu, atlas: &Atlas, item: &str) -> Vec<u8> {
    let id = mc_data::item_by_name(item).expect("a real 1.8.9 item");
    let model = mc_assets::item_model_for(id).clone();
    assert!(!model.is_empty(), "{item}: the real jar baked no model");

    let mut black = atlas.texture.clone();
    for texel in black.chunks_exact_mut(4) {
        texel[..3].fill(0);
    }
    let black = Atlas {
        texture: black,
        mipmaps: Vec::new(),
        w: atlas.w,
        h: atlas.h,
    };
    let pixels = render_held_model(gpu, &black, model, true);
    pixels.chunks_exact(4).map(|pixel| pixel[0]).collect()
}

/// Render `model` held in first person, fully raised, and return the RGBA framebuffer.
fn render_held_model(
    gpu: &mut Gpu,
    atlas: &Atlas,
    model: mc_assets::BakedModel,
    glint: bool,
) -> Vec<u8> {
    let gui_3d = model.gui_3d;
    let mut phase = FirstPersonPhase::new(gpu);
    phase.set_atlas(gpu, atlas);
    phase.set_item_model(Some(model), gui_3d);
    phase.set_item_glint_enabled(glint);
    // Fully raised: `equippedProgress == 1` on both ticks.
    phase.render_state.prev_equipped_progress = 1.0;
    phase.render_state.equipped_progress = 1.0;

    let mut camera = Camera::default();
    camera.set_viewport(W, H);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::FIRST_PERSON, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), &camera, 0.0);
    gpu.read_pixels().expect("readback")
}

/// Pixels with red at `0` / `~64` / `~128` / more: no foil, one foil fragment, one fragment per
/// pass (the vanilla result), and a pixel that some pass foiled twice.
fn foil_histogram(red: &[u8]) -> [usize; 4] {
    let mut histogram = [0; 4];
    for &value in red {
        let bucket = match value {
            0..=31 => 0,
            32..=95 => 1,
            96..=159 => 2,
            _ => 3,
        };
        histogram[bucket] += 1;
    }
    histogram
}

#[test]
fn a_held_enchanted_bow_takes_each_foil_pass_exactly_once_per_pixel() {
    let Some(mut gpu) = gpu() else { return };
    let Some(atlas) = baked_atlas() else {
        eprintln!("SKIP: vanilla jar not found; set MC_JARS_DIR");
        return;
    };

    let histogram = foil_histogram(&render_enchanted_held_item(&mut gpu, &atlas, "bow"));
    eprintln!("bow foil histogram [none, one, two, more]: {histogram:?}");
    assert!(
        histogram[2] > 1_000,
        "the bow must be on screen and foiled once per pass: {histogram:?}"
    );
    assert_eq!(
        histogram[3], 0,
        "a held item's foil may not land twice on a pixel: the coplanar, opposite-facing side \
         quads must be back-face culled as in vanilla ({histogram:?})"
    );
}

/// java: `RenderItem.isThereOneNegativeScale` is `x < 0 ^ y < 0 ^ z < 0` (an odd count, not
/// "exactly one"), and `renderItemModelTransform` culls `GL_FRONT` while it holds.
#[test]
fn the_held_item_cull_face_flips_only_for_an_odd_number_of_negative_display_scales() {
    use mc_render::pipeline::first_person_item_cull_face;
    let face = |scale: [f32; 3]| {
        first_person_item_cull_face(&mc_assets::model::ItemTransform {
            rotation: [0.0; 3],
            translation: [0.0; 3],
            scale,
        })
    };
    assert_eq!(
        face([1.7, 1.7, 1.7]),
        wgpu::Face::Back,
        "bow.json firstperson"
    );
    assert_eq!(face([1.0, 1.0, 1.0]), wgpu::Face::Back);
    assert_eq!(face([-1.0, 1.0, 1.0]), wgpu::Face::Front);
    assert_eq!(face([1.0, -1.0, 1.0]), wgpu::Face::Front);
    assert_eq!(face([1.0, 1.0, -1.0]), wgpu::Face::Front);
    assert_eq!(
        face([-1.0, -1.0, 1.0]),
        wgpu::Face::Back,
        "two mirrors cancel"
    );
    assert_eq!(face([-1.0, -1.0, -1.0]), wgpu::Face::Front, "three is odd");
    assert_eq!(
        face([0.0, 1.0, 1.0]),
        wgpu::Face::Back,
        "zero is not negative"
    );
}

/// The same contract for an ordinary flat tool and a GUI-3D block, so the cull face cannot be
/// right for the bow only. `diamond_sword` is a generated card; `stone` is a full cube whose
/// back faces would otherwise be foiled through the front ones only where depths tie.
#[test]
fn other_held_items_also_take_each_foil_pass_once() {
    let Some(mut gpu) = gpu() else { return };
    let Some(atlas) = baked_atlas() else {
        eprintln!("SKIP: vanilla jar not found; set MC_JARS_DIR");
        return;
    };

    for item in ["diamond_sword", "stone"] {
        let histogram = foil_histogram(&render_enchanted_held_item(&mut gpu, &atlas, item));
        eprintln!("{item} foil histogram [none, one, two, more]: {histogram:?}");
        assert!(
            histogram[2] > 1_000,
            "{item} must be on screen: {histogram:?}"
        );
        assert_eq!(histogram[3], 0, "{item}: foil landed twice: {histogram:?}");
    }
}

/// The culled held item must show the surface a two-sided, depth-tested draw shows: the near
/// side. Culling the wrong face (or a winding that disagrees with the hand projection) would
/// instead show the far side, e.g. the bow's mirrored back card or a block's inner faces.
///
/// The two-sided reference is the same model with every quad also emitted in reverse winding,
/// so no cull mode can remove it. The frames may differ only where the reference's back faces
/// tie the front ones under `GL_LEQUAL` along shared edges: measured 17 of the bow's ~4,500
/// pixels with the vanilla `GL_BACK` cull, against 926 with the cull face flipped, so the bound
/// below is 2% of the drawn pixels.
#[test]
fn held_item_culling_keeps_the_near_surface_a_two_sided_draw_shows() {
    let Some(mut gpu) = gpu() else { return };
    let Some(atlas) = baked_atlas() else {
        eprintln!("SKIP: vanilla jar not found; set MC_JARS_DIR");
        return;
    };
    let atlas = Atlas {
        mipmaps: Vec::new(),
        ..atlas
    };

    for item in ["bow", "crafting_table"] {
        let id = mc_data::item_by_name(item).expect("a real 1.8.9 item");
        let model = mc_assets::item_model_for(id).clone();
        let reversed = |quad: &mc_assets::BakedQuad| {
            let mut quad = quad.clone();
            quad.verts = [quad.verts[0], quad.verts[3], quad.verts[2], quad.verts[1]];
            quad
        };
        let mut two_sided = model.clone();
        let flipped: Vec<_> = model.general_quads.iter().map(reversed).collect();
        two_sided.general_quads.extend(flipped);
        for face in &mut two_sided.face_quads {
            let flipped: Vec<_> = face.iter().map(reversed).collect();
            face.extend(flipped);
        }

        let culled = render_held_model(&mut gpu, &atlas, model, false);
        let reference = render_held_model(&mut gpu, &atlas, two_sided, false);
        let drawn = culled
            .chunks_exact(4)
            .filter(|pixel| pixel[..3] != [0, 0, 0])
            .count();
        let differing = culled
            .chunks_exact(4)
            .zip(reference.chunks_exact(4))
            .filter(|(a, b)| a != b)
            .count();
        eprintln!("{item}: {drawn} px drawn, {differing} px differ from the two-sided draw");
        assert!(drawn > 1_000, "{item} must be on screen");
        assert!(
            differing * 50 <= drawn,
            "{item}: the culled draw shows a different surface on {differing} of {drawn} px"
        );
    }
}
