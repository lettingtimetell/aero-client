//! A resource pack texture with a side over the graphics device's `max_texture_dimension_2d`
//! must read as a missing texture, never reach `create_texture` (a validation error, which
//! wgpu's default handler turns into a panic: the client would fail on every start, because the
//! pack stays selected). Vanilla's `glTexImage2D` only records a GL error there.
//!
//! These run on a real adapter through `Gpu::headless` and skip when there is none. wgpu's
//! default error handler panics on any validation error, so "no panic" is the proof that no
//! oversized texture was created.

use std::io::Cursor;

use mc_assets::{FolderResourcePack, ResourceManager};
use mc_render::{Gpu, GpuOptions, RenderError};
use mc_ui::widget::{UiRenderer, UiTexture};

fn gpu() -> Option<Gpu> {
    match Gpu::headless(
        64,
        64,
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
        Err(error) => panic!("headless GPU: {error}"),
    }
}

fn png(width: u32, height: u32) -> Vec<u8> {
    let image = image::RgbaImage::from_pixel(width, height, image::Rgba([200, 100, 50, 255]));
    let mut out = Cursor::new(Vec::new());
    image
        .write_to(&mut out, image::ImageFormat::Png)
        .expect("encode test PNG");
    out.into_inner()
}

#[test]
fn a_gui_texture_over_the_device_limit_is_missing_not_a_crash() {
    let Some(gpu) = gpu() else {
        return;
    };
    let max = gpu.device.limits().max_texture_dimension_2d;
    eprintln!("device max_texture_dimension_2d = {max}");

    let pack = tempfile::tempdir().expect("tempdir");
    let gui = pack.path().join("assets/minecraft/textures/gui");
    std::fs::create_dir_all(&gui).expect("pack folders");
    std::fs::write(gui.join("widgets.png"), png(max + 1, 2)).expect("wide PNG");
    std::fs::write(gui.join("icons.png"), png(2, max + 1)).expect("tall PNG");
    // At the limit exactly: still a texture (the control for "refuses everything").
    std::fs::write(gui.join("at_limit.png"), png(max, 1)).expect("PNG at the limit");

    let mut resources = ResourceManager::new();
    resources.push_pack(Box::new(
        FolderResourcePack::open(pack.path()).expect("open folder pack"),
    ));
    let mut renderer = UiRenderer::new(&gpu);

    for path in ["textures/gui/widgets.png", "textures/gui/icons.png"] {
        let key = UiTexture::gui(path);
        let result = renderer.ensure_texture(&gpu, &resources, &key);
        assert!(
            result.is_err(),
            "{path}: an oversized texture must read as missing"
        );
        assert!(!renderer.has_texture(&key), "{path}: nothing uploaded");
        // Asked again (every frame does): answered from the missing set, still no texture.
        assert!(renderer.ensure_texture(&gpu, &resources, &key).is_err());
    }

    let at_limit = UiTexture::gui("textures/gui/at_limit.png");
    renderer
        .ensure_texture(&gpu, &resources, &at_limit)
        .expect("a texture at the device limit still loads");
    assert!(renderer.has_texture(&at_limit));
}

/// A pack's font sheet (`textures/font/ascii.png`) is decoded with the
/// assets, not through the decode-time side check, and becomes a texture of its own in the UI
/// renderer and in the nameplate phase. One wider than the device allows (a multiple of 16, so
/// the font accepts it) must stay a blank texture, as vanilla's `glTexImage2D` leaves it, not
/// end the process with a validation error.
#[test]
fn a_pack_font_sheet_over_the_device_limit_is_blank_not_a_crash() {
    let Some(gpu) = gpu() else {
        return;
    };
    let max = gpu.device.limits().max_texture_dimension_2d;
    let data = mc_ui::font::FontData::from_parts(&png(max + 16, 16), &[0u8; 0x1_0000])
        .expect("the font accepts a 16x16 grid of cells");
    assert_eq!(data.ascii_size(), (max + 16, 16));

    let mut renderer = UiRenderer::new(&gpu);
    renderer.upload_ascii_font(&gpu, &data);
    assert!(renderer.has_texture(&UiTexture::Font(mc_ui::font::FontTexture::Ascii)));

    let mut phase = mc_render::entity::NameplatePhase::new(&gpu);
    mc_ui::nameplate::upload_ascii_font(&mut phase, &gpu, &data);

    // The control: a sheet at the limit is uploaded as it is.
    let fits = mc_ui::font::FontData::from_parts(&png(max, 16), &[0u8; 0x1_0000]).expect("fits");
    renderer.upload_ascii_font(&gpu, &fits);
    mc_ui::nameplate::upload_ascii_font(&mut phase, &gpu, &fits);
}

/// The first-person textures (the held map's background and icons, the
/// fallback skin, held block-entity sheets) are decoded on their own path: one over the device's
/// limit reads as missing (the fallback stays), never a validation error.
#[test]
fn a_first_person_texture_over_the_device_limit_is_missing_not_a_crash() {
    let Some(gpu) = gpu() else {
        return;
    };
    let max = gpu.device.limits().max_texture_dimension_2d;
    let pack = tempfile::tempdir().expect("tempdir");
    let map = pack.path().join("assets/minecraft/textures/map");
    let entity = pack.path().join("assets/minecraft/textures/entity");
    std::fs::create_dir_all(&map).expect("pack folders");
    std::fs::create_dir_all(&entity).expect("pack folders");
    std::fs::write(map.join("map_background.png"), png(max + 1, 1)).expect("wide PNG");
    std::fs::write(map.join("map_icons.png"), png(1, max + 1)).expect("tall PNG");
    std::fs::write(entity.join("steve.png"), png(max + 1, 2)).expect("wide skin");
    std::fs::write(entity.join("alex.png"), png(64, 64)).expect("a normal skin");
    let mut resources = ResourceManager::new();
    resources.push_pack(Box::new(
        FolderResourcePack::open(pack.path()).expect("open folder pack"),
    ));

    let mut phase = mc_render::pipeline::FirstPersonPhase::new(&gpu);
    phase.load_map_textures(&gpu, &resources);
    assert!(
        !phase.load_skin_texture(&gpu, &resources, "textures/entity/steve.png"),
        "an oversized skin reads as missing"
    );
    // The control: a texture inside the limit still loads.
    assert!(phase.load_skin_texture(&gpu, &resources, "textures/entity/alex.png"));
}
