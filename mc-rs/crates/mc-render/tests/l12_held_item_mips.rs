//! h: held items sample only the atlas's base level.
//!
//! java: `RenderItem.renderItemModelTransform` brackets every held, framed or thrown item with
//! `getTexture(locationBlocksTexture).setBlurMipmap(false, false)` (`RenderItem.java:319`,
//! restored at `:340`), and `AbstractTexture.setBlurMipmapDirect(false, false)` sets
//! `GL_TEXTURE_MIN_FILTER` to `GL_NEAREST`: no mipmapping, the base level only. The first-person
//! phase kept the atlas's whole mip chain bound behind a `MipmapFilterMode::Nearest` sampler, so
//! a minified (or edge-on) held item sampled the mips.
//!
//! The test gives the phase an atlas whose base level is red and whose mip levels are green
//! (alpha kept from the real atlas, so the item's cutout is unchanged), holds the real `stone`
//! block, and renders it into a tiny frame so it is minified. Any green pixel is a mip sample.
//!
//! Skips (loudly) without a wgpu adapter or the vanilla jar; set `MC_JARS_DIR`.

use std::path::{Path, PathBuf};

use mc_assets::{Atlas, ResourceManager, ZipResourcePack};
use mc_render::pipeline::FirstPersonPhase;
use mc_render::{Camera, Gpu, GpuOptions, PhaseList, PhaseOrder, RenderError};

/// Small enough that the held block covers far fewer pixels than its 16x16 texels.
const W: u32 = 48;
const H: u32 = 27;

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

/// The real atlas's alpha, with a red base level and four green mip levels.
fn red_base_green_mips(atlas: &Atlas) -> Atlas {
    let mut base = atlas.texture.clone();
    for texel in base.chunks_exact_mut(4) {
        texel[..3].copy_from_slice(&[255, 0, 0]);
    }
    let mipmaps = (1..=4)
        .map(|level| {
            let w = (atlas.w >> level).max(1);
            let h = (atlas.h >> level).max(1);
            [0u8, 255, 0, 255].repeat((w * h) as usize)
        })
        .collect();
    Atlas {
        texture: base,
        mipmaps,
        w: atlas.w,
        h: atlas.h,
    }
}

#[test]
fn a_minified_held_item_samples_only_the_atlas_base_level() {
    let Some(atlas) = baked_atlas() else {
        eprintln!("SKIP: the vanilla jar is not available (set MC_JARS_DIR)");
        return;
    };
    let Some(mut gpu) = gpu() else {
        return;
    };
    let id = mc_data::item_by_name("stone").expect("a real 1.8.9 item");
    let model = mc_assets::item_model_for(id).clone();
    assert!(!model.is_empty(), "stone: the real jar baked no model");
    let gui_3d = model.gui_3d;

    let mut phase = FirstPersonPhase::new(&gpu);
    phase.set_atlas(&gpu, &red_base_green_mips(&atlas));
    phase.set_item_model(Some(model), gui_3d);
    // Fully raised: `equippedProgress == 1` on both ticks.
    phase.render_state.prev_equipped_progress = 1.0;
    phase.render_state.equipped_progress = 1.0;

    let mut camera = Camera::default();
    camera.set_viewport(W, H);
    let mut phases = PhaseList::new();
    phases.clear_color = wgpu::Color::BLACK;
    phases.insert(PhaseOrder::FIRST_PERSON, Box::new(phase));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(&gpu, frame.view(), &camera, 0.0);
    let pixels = gpu.read_pixels().expect("readback");

    let (mut base, mut mip) = (0, 0);
    for pixel in pixels.chunks_exact(4) {
        if pixel[0] > pixel[1] && pixel[0] > 8 {
            base += 1;
        } else if pixel[1] > pixel[0] && pixel[1] > 8 {
            mip += 1;
        }
    }
    eprintln!("held stone: {base} base-level pixels, {mip} mip-level pixels");
    assert!(
        base > 20,
        "the held block reached the frame ({base} base-level pixels)"
    );
    assert_eq!(
        mip, 0,
        "a held item never samples the atlas's mip levels ({mip} green pixels)"
    );
}
