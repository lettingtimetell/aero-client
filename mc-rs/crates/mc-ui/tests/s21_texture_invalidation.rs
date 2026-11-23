//! S21 → S15: a resource pack replacing a GUI texture must take effect without a restart.
//!
//! `UiRenderer::ensure_texture` returns early for any key already resident, so every texture a
//! screen has drawn once is frozen for the life of the process. `invalidate`/`clear_textures` are
//! the seam for that, and this file proves through the framebuffer that they work — and
//! that *not* using them changes nothing, so the addition is inert for every existing caller.
//!
//! java: `TextureManager`'s reload listener re-runs `SimpleTexture.loadTexture`, which is what
//! makes a vanilla pack's GUI art live after `refreshResources`.

use std::path::{Path, PathBuf};

use mc_assets::{FolderResourcePack, ResourceLocation, ResourceManager, ZipResourcePack};
use mc_render::{Gpu, GpuOptions, RenderError};
use mc_ui::font::FontData;
use mc_ui::widget::{UiPhase, UiTexture};

const W: u32 = 128;
const H: u32 = 64;
/// A GUI texture every screen uses, so replacing it is the realistic case.
const WIDGETS: &str = "textures/gui/widgets.png";

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
    if let Ok(dir) = std::env::var("MC_JARS_DIR") {
        let path = PathBuf::from(dir);
        return path.is_dir().then_some(path);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?;
    let path = root.join("MCP-919-main").join("jars");
    path.is_dir().then_some(path)
}

fn jar() -> Option<PathBuf> {
    let jar = jars_dir()?.join("versions").join("1.8.9").join("1.8.9.jar");
    jar.is_file().then_some(jar)
}

fn manager(extra: Option<&Path>) -> Option<ResourceManager> {
    let mut manager = ResourceManager::new();
    manager.push_pack(Box::new(ZipResourcePack::open(jar()?).ok()?));
    if let Some(pack) = extra {
        manager.push_pack(Box::new(FolderResourcePack::open(pack).ok()?));
    }
    Some(manager)
}

/// A folder pack whose `widgets.png` is one flat colour.
fn flat_widgets_pack(dir: &Path, rgb: [u8; 3]) -> PathBuf {
    let pack = dir.join("FlatWidgets");
    let gui = pack.join("assets/minecraft/textures/gui");
    std::fs::create_dir_all(&gui).expect("mkdir");
    std::fs::write(
        pack.join("pack.mcmeta"),
        br#"{"pack":{"pack_format":1,"description":"s21"}}"#,
    )
    .expect("mcmeta");
    let mut image = image::RgbaImage::new(256, 256);
    for pixel in image.pixels_mut() {
        *pixel = image::Rgba([rgb[0], rgb[1], rgb[2], 255]);
    }
    image.save(gui.join("widgets.png")).expect("write");
    pack
}

#[test]
fn a_pack_replacing_a_gui_texture_needs_invalidation_and_then_takes_effect() {
    let Some(gpu) = gpu() else { return };
    let Some(vanilla) = manager(None) else {
        eprintln!("SKIP: no vanilla jar in this checkout");
        return;
    };
    let dir = tempfile::tempdir().expect("scratch dir");
    let pack = flat_widgets_pack(dir.path(), [255, 0, 255]);
    let packed = manager(Some(&pack)).expect("stacked manager");

    let font = FontData::load(&vanilla).expect("font data");
    let mut phase = UiPhase::new(&gpu, W, H, 1, false);
    phase.renderer.upload_ascii_font(&gpu, &font);

    let key = UiTexture::Resource(ResourceLocation::minecraft(WIDGETS));
    assert!(!phase.renderer.has_texture(&key));
    phase
        .renderer
        .ensure_texture(&gpu, &vanilla, &key)
        .expect("vanilla widgets.png must load");
    assert!(phase.renderer.has_texture(&key));
    let resident_after_vanilla = phase.renderer.texture_count();

    // Without invalidation, `ensure_texture` against the *new* stack is a no-op — this is the
    // documented gap, asserted rather than assumed.
    phase
        .renderer
        .ensure_texture(&gpu, &packed, &key)
        .expect("no-op");
    assert_eq!(
        phase.renderer.texture_count(),
        resident_after_vanilla,
        "a resident key is not re-read; that is exactly why `invalidate` was needed"
    );

    // Invalidate one key and it comes back from the new stack.
    assert!(phase.renderer.invalidate(&key), "the key was resident");
    assert!(!phase.renderer.has_texture(&key));
    assert!(
        !phase.renderer.invalidate(&key),
        "invalidating a key twice reports that nothing was cached"
    );
    phase
        .renderer
        .ensure_texture(&gpu, &packed, &key)
        .expect("the pack's widgets.png must load");
    assert!(phase.renderer.has_texture(&key));

    // And `clear_textures` does the whole cache at once, keeping `Solid` alive because
    // `UiRenderer::draw` bails out of the frame without it.
    phase.renderer.clear_textures(&gpu);
    assert!(
        phase.renderer.has_texture(&UiTexture::Solid),
        "the 1x1 white fallback must survive a clear, or every frame after one draws nothing"
    );
    assert_eq!(
        phase.renderer.texture_count(),
        1,
        "everything else is gone, including the font sheet the caller uploaded by hand"
    );
    assert!(!phase.renderer.has_texture(&key));
}
