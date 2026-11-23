//! Loads the environment PNGs (`sun.png`, `moon_phases.png`, `clouds.png`, `rain.png`,
//! `snow.png`, `end_sky.png`) from an [`mc_assets::ResourceManager`] and uploads them as
//! `wgpu` textures.
//!
//! Decodes with `mc_assets::decode_texture_png` (a PNG the device cannot hold as one texture is
//! refused) rather than `mc_assets::sprite::load_sprite`: sky textures are not atlas sprites
//! (S09's `bake_all` never stitches `textures/environment/**`).

use mc_assets::{sprite::anaglyph_rgba, AssetError, ResourceLocation, ResourceManager};
use wgpu::util::DeviceExt;

use crate::Gpu;

/// Storage format used by Java's `TextureUtil.allocateTextureImpl`.
///
/// The 1.8.9 call is `glTexImage2D(..., GL_RGBA, ...)`: `GL_RGBA` is the plain normalized
/// colour format, not an sRGB internal format. Keeping the uploaded bytes in `Rgba8Unorm`
/// therefore matches the byte-domain sampling used by the vanilla sky shaders. In particular,
/// do not replace this with `Rgba8UnormSrgb` merely because PNG assets are conventionally called
/// sRGB; that would add a transfer-function conversion that the Java upload path never requests.
const JAVA_RGBA_UPLOAD_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Build the temporary RGBA8 upload array used by Java's `TextureUtil.copyToBufferPos`.
///
/// Sky and weather resources are standalone textures rather than atlas sprites, but they cross
/// the same Java upload boundary. Keeping this owner-local name makes the byte transform and its
/// alpha-preserving tests visible without changing the shared asset decoder.
fn sky_upload_rgba(source_rgba: &[u8], anaglyph: bool) -> Vec<u8> {
    anaglyph_rgba(source_rgba, anaglyph)
}

/// A decoded, GPU-resident texture plus its pixel dimensions.
pub struct SkyTexture {
    /// The uploaded texture.
    pub texture: wgpu::Texture,
    /// A full view of it.
    pub view: wgpu::TextureView,
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
    /// The decoded, unmodified RGBA8 source pixels.
    ///
    /// Java applies `TextureUtil.anaglyphColor` to a temporary upload buffer, not to the
    /// resource-pack source. Retaining this copy lets a live setting toggle rewrite the resident
    /// texture from the exact original bytes instead of trying to invert (or double-apply) the
    /// integer colour transform.
    source_rgba: Vec<u8>,
}

impl SkyTexture {
    /// Rewrite the resident texture with Java's upload-time anaglyph transform selected.
    ///
    /// The source bytes remain untouched, so `false -> true -> false` is byte-for-byte stable.
    pub(crate) fn set_anaglyph(&self, gpu: &Gpu, enabled: bool) {
        let upload_rgba = sky_upload_rgba(&self.source_rgba, enabled);
        gpu.queue.write_texture(
            self.texture.as_image_copy(),
            &upload_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.width * 4),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Raw decoded source bytes, for owner-local tests and deterministic reloads, and
    /// the fancy clouds' CPU-side texel coverage. Alpha is never touched by
    /// the anaglyph transform, so the coverage holds in both modes.
    pub(crate) fn source_rgba(&self) -> &[u8] {
        &self.source_rgba
    }
}

/// Decode `textures/environment/<name>.png` (or any other resource path) and upload it.
///
/// # Errors
/// [`AssetError`] if the resource is missing or is not a valid PNG.
pub fn load(
    gpu: &Gpu,
    rm: &ResourceManager,
    path: &str,
    label: &str,
) -> Result<SkyTexture, AssetError> {
    load_with_anaglyph(gpu, rm, path, label, false)
}

/// Decode and upload a sky/weather resource with Java's global upload-time anaglyph setting.
///
/// [`load`] remains the default, non-anaglyph convenience entry point used by callers that do
/// not own the setting. Sky and weather phases use this explicit variant so startup state is
/// applied before the first decoded resource reaches the GPU.
pub fn load_with_anaglyph(
    gpu: &Gpu,
    rm: &ResourceManager,
    path: &str,
    label: &str,
    anaglyph: bool,
) -> Result<SkyTexture, AssetError> {
    let loc = ResourceLocation::parse(path)?;
    let bytes = rm.load(&loc)?;
    // Refused (missing) when the device cannot hold it.
    let img = mc_assets::decode_texture_png(&bytes)
        .map_err(|e| AssetError::Image {
            location: loc.to_string(),
            message: e.to_string(),
        })?
        .into_rgba8();
    let (width, height) = img.dimensions();
    let source_rgba = img.into_raw();
    let upload_rgba = sky_upload_rgba(&source_rgba, anaglyph);

    let texture = gpu.device.create_texture_with_data(
        &gpu.queue,
        &wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: JAVA_RGBA_UPLOAD_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &upload_rgba,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Ok(SkyTexture {
        texture,
        view,
        width,
        height,
        source_rgba,
    })
}

/// A 1x1 white texture, for a phase that needs *a* bind group before its real texture loads (or
/// in a unit test that never touches a resource pack).
#[must_use]
pub fn white_pixel(gpu: &Gpu, label: &str) -> SkyTexture {
    let texture = gpu.device.create_texture_with_data(
        &gpu.queue,
        &wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: JAVA_RGBA_UPLOAD_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &[255, 255, 255, 255],
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    SkyTexture {
        texture,
        view,
        width: 1,
        height: 1,
        source_rgba: vec![255, 255, 255, 255],
    }
}

#[cfg(test)]
mod tests {
    use super::{sky_upload_rgba, JAVA_RGBA_UPLOAD_FORMAT};

    #[test]
    fn java_anaglyph_upload_mixes_colours_and_preserves_alpha() {
        let source = [
            255, 0, 0, 7, // red, translucent
            0, 255, 0, 19, // green
            0, 0, 255, 31, // blue
            128, 128, 128, 64,
        ];
        assert_eq!(sky_upload_rgba(&source, false), source);
        assert_eq!(
            sky_upload_rgba(&source, true),
            [76, 76, 76, 7, 178, 150, 0, 19, 0, 28, 178, 31, 128, 128, 128, 64,],
            "Java's integer anaglyph transform mixes RGB only"
        );
    }

    #[test]
    fn java_anaglyph_toggle_replays_raw_sky_bytes_exactly() {
        let source = [255, 7, 61, 3, 11, 222, 19, 247, 44, 55, 66, 77];
        let enabled = sky_upload_rgba(&source, true);
        assert_ne!(enabled, source);
        assert_eq!(sky_upload_rgba(&source, false), source);
        assert_eq!(sky_upload_rgba(&source, true), enabled);
    }

    #[test]
    fn decoded_resource_owner_retains_raw_source_for_startup_anaglyph() {
        use image::ImageEncoder;
        use std::io::Cursor;

        let Ok(gpu) = crate::Gpu::headless(
            8,
            8,
            crate::context::GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) else {
            eprintln!("SKIP: no wgpu adapter in this environment");
            return;
        };

        let mut png = Cursor::new(Vec::new());
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(&[255, 0, 0, 7], 1, 1, image::ExtendedColorType::Rgba8)
            .expect("encode one-pixel sky png");
        let temp = tempfile::tempdir().expect("temp resource pack");
        let root = temp.path().join("assets/minecraft/textures/environment");
        std::fs::create_dir_all(&root).expect("sky resource directory");
        std::fs::write(root.join("test.png"), png.into_inner()).expect("sky png");

        let mut resources = mc_assets::ResourceManager::new();
        resources.push_pack(Box::new(
            mc_assets::FolderResourcePack::open(temp.path()).expect("folder pack"),
        ));
        let texture = super::super::textures::load_with_anaglyph(
            &gpu,
            &resources,
            "textures/environment/test.png",
            "sky test",
            true,
        )
        .expect("decode sky png");
        assert_eq!(texture.source_rgba(), &[255, 0, 0, 7]);
        assert_eq!(
            sky_upload_rgba(texture.source_rgba(), true),
            [76, 76, 76, 7],
            "startup upload converts RGB while retaining source alpha"
        );
        assert_eq!(
            sky_upload_rgba(texture.source_rgba(), false),
            [255, 0, 0, 7],
            "the retained source can replay the exact non-anaglyph bytes"
        );
    }

    #[test]
    fn java_gl_rgba_upload_is_plain_unorm_not_srgb() {
        // TextureUtil.allocateTextureImpl passes GL_RGBA as the internal format. This is a
        // source-level control: changing either descriptor back to an sRGB format must require
        // changing this assertion too, instead of silently reintroducing a transfer function.
        assert_eq!(
            JAVA_RGBA_UPLOAD_FORMAT,
            wgpu::TextureFormat::Rgba8Unorm,
            "Java GL_RGBA must map to plain normalized storage"
        );
        assert_ne!(
            JAVA_RGBA_UPLOAD_FORMAT,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            "sky PNG uploads must not opt into an sRGB texture view"
        );
    }
}
