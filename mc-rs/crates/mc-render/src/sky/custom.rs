//! Custom Sky: resource-pack custom sky layers on the GPU.
//!
//! [`mc_assets::custom_sky`] finds and parses a pack's layers. This module decodes and uploads
//! their textures once per resource reload ([`CustomSkyAssets::load`]) and draws them every frame
//! ([`CustomSkyRenderer`]) inside [`crate::sky::dome::SkyPhase`]'s pass, where the format places
//! them: after the sunrise glow, before the sun, moon and stars (`RenderGlobal.renderSky`, right
//! after its `rotate(-90, 0, 1, 0)`). While the current world has custom layers the vanilla star
//! field is not drawn (the format's "disable default stars"): sky packs bring their own stars.
//!
//! # Geometry
//!
//! A layer is a cube around the eye, textured from a 3x2 grid of faces (see [`FACES`]):
//!
//! ```text
//! +--------+--------+--------+
//! | bottom |  top   | south  |
//! +--------+--------+--------+
//! |  west  | north  |  east  |
//! +--------+--------+--------+
//! ```
//!
//! The four walls stand upright and read left to right as seen from inside, so the bottom row
//! plus south is one panorama (west, north, east, south). The top face has south at the top of
//! its cell, the bottom face north. This is the format's labelled template, and it is also what
//! the format's GL transform sequence produces once vanilla's `rotate(-90, 0, 1, 0)` is applied.
//!
//! A rotating layer turns by `360 * (day_offset + celestial_angle * speed)` degrees about its
//! world-space axis, right-handed (seen looking along the axis, clockwise), so the default south
//! axis moves a layer exactly like the sun. `day_offset` keeps a fractional `speed` continuous
//! across the daily wrap of the celestial angle.
//!
//! # Blending
//!
//! Each `blend=` mode is the format's GL blend function, and the fade scales the
//! channels that mode fades ([`fade_color`]): `add`/`alpha`/`replace` fade alpha, `multiply` all
//! four, the rest RGB. `replace` draws unblended behind vanilla's `> 0.1` alpha test. The alpha
//! channel of the frame is never written.
//!
//! # Cost
//!
//! Off: [`crate::sky::dome::SkyPhase::custom_sky`] is `None`, one check per frame. On: per frame,
//! one uniform write and one 36-vertex draw per layer whose brightness is above zero; nothing is
//! allocated once warm. The textures are decoded (in parallel) and uploaded once per reload.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use glam::{Mat4, Vec3};
use mc_assets::custom_sky::{CustomSkyFiles, SkyBlend, SkyLayer, SkyWeather, TICKS_PER_DAY};
use mc_assets::{ResourceLocation, ResourceManager};
use rayon::prelude::*;
use wgpu::util::DeviceExt;

use crate::sky::environment::{Dimension, Environment};
use crate::{Camera, Gpu};

/// Half the cube's edge. Only directions matter for a sky box; at 16 the farthest corner (27.7
/// blocks) stays inside the sky projection's far plane at every render distance.
const CUBE_RADIUS: f32 = 16.0;

/// A layer whose brightness is below this is skipped.
pub const MIN_BRIGHTNESS: f32 = 1.0e-4;

/// `replace` layers keep a texel only if its faded alpha passes vanilla's alpha test,
/// `alphaFunc(GL_GREATER, 0.1)`.
pub const REPLACE_ALPHA_CUTOFF: f32 = 0.1;

/// One face of the sky cube.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Face {
    /// Which side of the sky it is.
    pub name: &'static str,
    /// Column of its cell in the 3x2 texture.
    pub col: u8,
    /// Row of its cell.
    pub row: u8,
    /// Corner directions in world space (+X east, +Y up, +Z south), in image order: top-left,
    /// top-right, bottom-right, bottom-left of the cell.
    pub corners: [[f32; 3]; 4],
}

/// The cube, in the format's template layout (see the module docs).
pub const FACES: [Face; 6] = [
    Face {
        name: "bottom",
        col: 0,
        row: 0,
        corners: [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0],
        ],
    },
    Face {
        name: "top",
        col: 1,
        row: 0,
        corners: [
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ],
    },
    Face {
        name: "south",
        col: 2,
        row: 0,
        corners: [
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
        ],
    },
    Face {
        name: "west",
        col: 0,
        row: 1,
        corners: [
            [-1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
    },
    Face {
        name: "north",
        col: 1,
        row: 1,
        corners: [
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
        ],
    },
    Face {
        name: "east",
        col: 2,
        row: 1,
        corners: [
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
        ],
    },
];

/// Vertices in the cube: six faces, two triangles each.
const CUBE_VERTICES: u32 = 36;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct CubeVertex {
    pos: [f32; 3],
    uv: [f32; 2],
    cell: [f32; 4],
}

impl CubeVertex {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<CubeVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x4],
    };
}

/// The cube as a triangle list, [`FACES`] order.
fn cube_vertices() -> Vec<CubeVertex> {
    let mut out = Vec::with_capacity(CUBE_VERTICES as usize);
    for face in &FACES {
        let (u0, v0) = (f32::from(face.col) / 3.0, f32::from(face.row) / 2.0);
        let (u1, v1) = (f32::from(face.col + 1) / 3.0, f32::from(face.row + 1) / 2.0);
        let uv = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]];
        let vertex = |i: usize| CubeVertex {
            pos: (Vec3::from(face.corners[i]) * CUBE_RADIUS).to_array(),
            uv: uv[i],
            cell: [u0, v0, u1, v1],
        };
        for i in [0, 1, 2, 0, 2, 3] {
            out.push(vertex(i));
        }
    }
    out
}

/// `custom_sky.wgsl`'s per-layer uniform.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerUniform {
    view_proj: [[f32; 4]; 4],
    color: [f32; 4],
    params: [f32; 4],
}

/// The vertex colour for a layer at `brightness`: which channels the fade scales.
#[must_use]
pub fn fade_color(blend: SkyBlend, brightness: f32) -> [f32; 4] {
    let b = brightness;
    match blend {
        SkyBlend::Alpha | SkyBlend::Add | SkyBlend::Replace => [1.0, 1.0, 1.0, b],
        SkyBlend::Multiply => [b, b, b, b],
        SkyBlend::Subtract
        | SkyBlend::Dodge
        | SkyBlend::Burn
        | SkyBlend::Screen
        | SkyBlend::Overlay => [b, b, b, 1.0],
    }
}

/// The colour blend for a mode (the format's GL factors), `None` for `replace`. Alpha is
/// irrelevant: the pipelines write RGB only.
#[must_use]
pub fn blend_state(blend: SkyBlend) -> Option<wgpu::BlendState> {
    use wgpu::BlendFactor as F;
    let (src_factor, dst_factor) = match blend {
        SkyBlend::Alpha => (F::SrcAlpha, F::OneMinusSrcAlpha),
        SkyBlend::Add => (F::SrcAlpha, F::One),
        SkyBlend::Subtract => (F::OneMinusDst, F::Zero),
        SkyBlend::Multiply => (F::Dst, F::OneMinusSrcAlpha),
        SkyBlend::Dodge => (F::One, F::One),
        SkyBlend::Burn => (F::Zero, F::OneMinusSrc),
        SkyBlend::Screen => (F::One, F::OneMinusSrc),
        SkyBlend::Overlay => (F::Dst, F::Src),
        SkyBlend::Replace => return None,
    };
    Some(wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor,
            dst_factor,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: F::Zero,
            dst_factor: F::One,
            operation: wgpu::BlendOperation::Add,
        },
    })
}

/// How much of a layer shows in this weather. `thunder` is vanilla's `getThunderStrength`, which
/// is already scaled by the rain, so clear + rain-only + thunder always add up to one. The
/// default (clear only) is `1 - rain`: every ordinary layer fades out as rain sets in.
#[must_use]
pub fn weather_brightness(weather: SkyWeather, rain: f32, thunder: f32) -> f32 {
    let rain = rain.clamp(0.0, 1.0);
    let thunder = thunder.clamp(0.0, rain);
    let mut b = 0.0;
    if weather.clear {
        b += 1.0 - rain;
    }
    if weather.rain {
        b += rain - thunder;
    }
    if weather.thunder {
        b += thunder;
    }
    b.clamp(0.0, 1.0)
}

/// A layer's brightness from its fade times and the weather, before any biome/height condition.
#[must_use]
pub fn layer_brightness(layer: &SkyLayer, time_of_day: i32, rain: f32, thunder: f32) -> f32 {
    let fade = layer.fade.map_or(1.0, |fade| fade.brightness(time_of_day));
    (fade * weather_brightness(layer.weather, rain, thunder)).clamp(0.0, 1.0)
}

/// How much of the sky behind it a layer drawn at `brightness` covers, 0..1: a `replace` layer
/// all of it while its texels pass the alpha test, every other mode its brightness. Layers
/// combine like stacked glass: the sky stays clear only where every layer leaves it clear
/// ([`CustomSkyRenderer::cover`]).
#[must_use]
pub fn layer_cover(blend: SkyBlend, brightness: f32) -> f32 {
    if blend == SkyBlend::Replace {
        f32::from(u8::from(brightness > REPLACE_ALPHA_CUTOFF))
    } else {
        brightness.clamp(0.0, 1.0)
    }
}

/// How far a rotating layer has turned, in turns (only the fraction matters).
///
/// `celestial_angle * speed` alone would jump at the daily wrap of the celestial angle (noon)
/// whenever `speed` is fractional, so the fractional part of `speed` is also accumulated over the
/// days elapsed (`(worldTime + 18000) / 24000`, the day count that steps at that wrap).
#[must_use]
pub fn layer_turns(layer: &SkyLayer, world_time: i64, celestial_angle: f32) -> f32 {
    let speed = f64::from(layer.speed);
    let mut turns = f64::from(celestial_angle) * speed;
    if speed.fract() != 0.0 {
        let day = world_time.saturating_add(18_000) / i64::from(TICKS_PER_DAY);
        turns += (day as f64 * speed.fract()).fract();
    }
    turns.rem_euclid(1.0) as f32
}

/// The layer's rotation this frame: identity unless `rotate`.
#[must_use]
pub fn layer_rotation(layer: &SkyLayer, world_time: i64, celestial_angle: f32) -> Mat4 {
    if !layer.rotate {
        return Mat4::IDENTITY;
    }
    let turns = layer_turns(layer, world_time, celestial_angle);
    Mat4::from_axis_angle(Vec3::from(layer.axis), turns * std::f32::consts::TAU)
}

/// Whether the camera's position satisfies a layer's `biomes`/`heights` (1 or 0). Unknown
/// position: 0 for a restricted layer, as when there is no render-view entity.
fn position_condition(layer: &SkyLayer, biome: Option<u8>, y: Option<i32>) -> f32 {
    if let Some(biomes) = &layer.biomes {
        if !biome.is_some_and(|b| biomes.contains(&b)) {
            return 0.0;
        }
    }
    if let Some(heights) = &layer.heights {
        if !y.is_some_and(|y| heights.iter().any(|&(lo, hi)| lo <= y && y <= hi)) {
            return 0.0;
        }
    }
    1.0
}

/// One uploaded texture.
struct LayerTexture {
    _texture: wgpu::Texture,
    nearest: wgpu::BindGroup,
    linear: wgpu::BindGroup,
    /// Its `.mcmeta` asks for linear filtering (`SimpleTexture`'s `blur`).
    blur: bool,
    /// Half a texel in UV units.
    half_texel: [f32; 2],
}

/// A layer with its texture.
struct GpuLayer {
    def: SkyLayer,
    texture: usize,
}

/// Facts about one load, for the log and the report.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CustomSkyStats {
    /// Drawable layers, all worlds.
    pub layers: usize,
    /// Layer files left out (bad times, a missing or undecodable texture).
    pub skipped: usize,
    /// Distinct textures uploaded.
    pub textures: usize,
    /// GPU memory of those textures, in bytes (RGBA8, no mipmaps).
    pub texture_bytes: u64,
    /// Finding, parsing and reading the files.
    pub read_ms: f32,
    /// PNG decoding (parallel).
    pub decode_ms: f32,
    /// Texture upload and pipeline creation.
    pub upload_ms: f32,
}

/// A pack's custom sky, on the GPU: every layer's texture plus the nine blend pipelines. Built
/// once per resource reload and shared (`Arc`) by every [`CustomSkyRenderer`] made from it, so a
/// phase rebuild does not decode anything again.
pub struct CustomSkyAssets {
    layers: Vec<GpuLayer>,
    textures: Vec<LayerTexture>,
    pipelines: Vec<wgpu::RenderPipeline>,
    uniform_layout: wgpu::BindGroupLayout,
    cube: wgpu::Buffer,
    /// Layer count per world: `[overworld, End]`.
    world_layers: [usize; 2],
    /// Some layer has `biomes`/`heights`, so the camera position matters.
    needs_position: bool,
    format: wgpu::TextureFormat,
    sample_count: u32,
    stats: CustomSkyStats,
}

impl std::fmt::Debug for CustomSkyAssets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomSkyAssets")
            .field("stats", &self.stats)
            .field("world_layers", &self.world_layers)
            .finish_non_exhaustive()
    }
}

/// A decoded texture, ready to upload.
struct Decoded {
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    blur: bool,
}

fn decode(bytes: &[u8], blur: bool, max_dimension: u32, anaglyph: bool) -> Result<Decoded, String> {
    let image = image::load_from_memory_with_format(bytes, image::ImageFormat::Png)
        .map_err(|error| error.to_string())?;
    let mut image = image.into_rgba8();
    if image.width() == 0 || image.height() == 0 {
        return Err("empty image".into());
    }
    // Larger than the GPU allows: halve until it fits (no real sky pack is anywhere near).
    while image.width() > max_dimension || image.height() > max_dimension {
        let (w, h) = ((image.width() / 2).max(1), (image.height() / 2).max(1));
        image = image::imageops::resize(&image, w, h, image::imageops::FilterType::Triangle);
    }
    let (width, height) = image.dimensions();
    let mut rgba = image.into_raw();
    // java: `TextureUtil` applies the anaglyph transform to the upload buffer.
    mc_assets::sprite::apply_anaglyph_rgba(&mut rgba, anaglyph);
    Ok(Decoded {
        rgba,
        width,
        height,
        blur,
    })
}

fn world_slot(world: i32) -> Option<usize> {
    match world {
        0 => Some(0),
        1 => Some(1),
        _ => None,
    }
}

fn dimension_world(dimension: Dimension) -> Option<i32> {
    match dimension {
        Dimension::Overworld => Some(0),
        Dimension::End => Some(1),
        Dimension::Nether => None,
    }
}

impl CustomSkyAssets {
    /// Find, parse, decode and upload the active packs' custom sky. `None` when no layer is
    /// drawable (the pack has no custom sky, or every layer is broken); the reason for each
    /// skipped layer is logged once, here.
    #[must_use]
    pub fn load(gpu: &Gpu, rm: &ResourceManager, anaglyph: bool) -> Option<Arc<CustomSkyAssets>> {
        let started = Instant::now();
        let files = CustomSkyFiles::load(rm);
        Self::from_files(gpu, rm, files, anaglyph, started)
    }

    /// [`CustomSkyAssets::load`] from already-parsed files (tests). `started` is when the read
    /// began, for [`CustomSkyStats::read_ms`].
    #[must_use]
    pub fn from_files(
        gpu: &Gpu,
        rm: &ResourceManager,
        files: CustomSkyFiles,
        anaglyph: bool,
        started: Instant,
    ) -> Option<Arc<CustomSkyAssets>> {
        mc_core::profile_scope!("sky.custom.load");
        let mut skipped = files.skipped.len();
        for (file, why) in &files.skipped {
            tracing::debug!(%file, why, "custom sky: layer skipped");
        }

        // Each distinct texture once, in first-use order.
        let mut sources: Vec<ResourceLocation> = Vec::new();
        let mut source_index: HashMap<ResourceLocation, usize> = HashMap::new();
        for layer in &files.layers {
            source_index.entry(layer.source.clone()).or_insert_with(|| {
                sources.push(layer.source.clone());
                sources.len() - 1
            });
        }
        let raw: Vec<Option<(Vec<u8>, bool)>> = sources
            .iter()
            .map(|location| match rm.resource(location) {
                Ok(resource) => {
                    let blur =
                        mc_assets::parse_texture_metadata(location, resource.metadata.as_deref())
                            .is_ok_and(|meta| meta.blur);
                    Some((resource.data, blur))
                }
                Err(error) => {
                    tracing::warn!(%location, %error, "custom sky: texture unreadable");
                    None
                }
            })
            .collect();
        let read_ms = started.elapsed().as_secs_f32() * 1000.0;

        let decode_started = Instant::now();
        let max_dimension = gpu.device.limits().max_texture_dimension_2d;
        let decoded: Vec<Option<Decoded>> = raw
            .into_par_iter()
            .zip(sources.par_iter())
            .map(|(raw, location)| {
                let (bytes, blur) = raw?;
                decode(&bytes, blur, max_dimension, anaglyph)
                    .map_err(|error| {
                        tracing::warn!(%location, %error, "custom sky: texture not decodable");
                    })
                    .ok()
            })
            .collect();
        let decode_ms = decode_started.elapsed().as_secs_f32() * 1000.0;

        let upload_started = Instant::now();
        let texture_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("custom sky texture"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });
        let sampler = |filter: wgpu::FilterMode, label: &str| {
            gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                // java: a pack texture loads through `SimpleTexture`: GL_NEAREST unless its
                // `.mcmeta` asks for `blur`. Clamped: the shader keeps samples inside the face.
                mag_filter: filter,
                min_filter: filter,
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                ..Default::default()
            })
        };
        let nearest = sampler(wgpu::FilterMode::Nearest, "custom sky nearest");
        let linear = sampler(wgpu::FilterMode::Linear, "custom sky linear");

        let mut texture_slot: Vec<Option<usize>> = vec![None; sources.len()];
        let mut textures = Vec::new();
        let mut texture_bytes = 0u64;
        for (source, decoded) in decoded.into_iter().enumerate() {
            let Some(decoded) = decoded else { continue };
            let texture = gpu.device.create_texture_with_data(
                &gpu.queue,
                &wgpu::TextureDescriptor {
                    label: Some("custom sky layer"),
                    size: wgpu::Extent3d {
                        width: decoded.width,
                        height: decoded.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    // java: `GL_RGBA`, plain normalised storage (see `sky::textures`).
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &decoded.rgba,
            );
            texture_bytes += decoded.rgba.len() as u64;
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let group = |sampler: &wgpu::Sampler| {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("custom sky texture"),
                    layout: &texture_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                    ],
                })
            };
            texture_slot[source] = Some(textures.len());
            textures.push(LayerTexture {
                nearest: group(&nearest),
                linear: group(&linear),
                _texture: texture,
                blur: decoded.blur,
                half_texel: [0.5 / decoded.width as f32, 0.5 / decoded.height as f32],
            });
        }

        let mut layers = Vec::new();
        for def in files.layers {
            match texture_slot[source_index[&def.source]] {
                Some(texture) => layers.push(GpuLayer { def, texture }),
                None => skipped += 1,
            }
        }
        if layers.is_empty() {
            tracing::info!(
                skipped,
                "custom sky: the resource packs have no drawable sky layer"
            );
            return None;
        }

        let uniform_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("custom sky layer uniform"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<LayerUniform>() as u64,
                            ),
                        },
                        count: None,
                    }],
                });
        let pipelines = create_pipelines(gpu, &texture_layout, &uniform_layout);
        let cube = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("custom sky cube"),
                contents: bytemuck::cast_slice(&cube_vertices()),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let mut world_layers = [0usize; 2];
        for layer in &layers {
            if let Some(slot) = world_slot(layer.def.world) {
                world_layers[slot] += 1;
            }
        }
        let needs_position = layers
            .iter()
            .any(|layer| layer.def.biomes.is_some() || layer.def.heights.is_some());
        let stats = CustomSkyStats {
            layers: layers.len(),
            skipped,
            textures: textures.len(),
            texture_bytes,
            read_ms,
            decode_ms,
            upload_ms: upload_started.elapsed().as_secs_f32() * 1000.0,
        };
        tracing::info!(
            layers = stats.layers,
            overworld = world_layers[0],
            end = world_layers[1],
            skipped = stats.skipped,
            textures = stats.textures,
            texture_mb = stats.texture_bytes as f32 / (1024.0 * 1024.0),
            read_ms = stats.read_ms,
            decode_ms = stats.decode_ms,
            upload_ms = stats.upload_ms,
            "custom sky loaded"
        );
        Some(Arc::new(CustomSkyAssets {
            layers,
            textures,
            pipelines,
            uniform_layout,
            cube,
            world_layers,
            needs_position,
            format: gpu.config.format,
            sample_count: gpu.sample_count(),
            stats,
        }))
    }

    /// What the load found and cost.
    #[must_use]
    pub fn stats(&self) -> CustomSkyStats {
        self.stats
    }

    /// Whether these pipelines fit `gpu`'s colour target (format and MSAA). A renderer must not
    /// be built from assets made for another target.
    #[must_use]
    pub fn fits(&self, gpu: &Gpu) -> bool {
        self.format == gpu.config.format && self.sample_count == gpu.sample_count()
    }

    /// Whether a layer restricts itself by biome or height (the client then supplies the
    /// camera's position, see [`CustomSkyRenderer::camera_biome`]).
    #[must_use]
    pub fn needs_position(&self) -> bool {
        self.needs_position
    }

    /// Layers drawn in a dimension.
    #[must_use]
    pub fn layer_count(&self, dimension: Dimension) -> usize {
        dimension_world(dimension)
            .and_then(world_slot)
            .map_or(0, |slot| self.world_layers[slot])
    }
}

fn create_pipelines(
    gpu: &Gpu,
    texture_layout: &wgpu::BindGroupLayout,
    uniform_layout: &wgpu::BindGroupLayout,
) -> Vec<wgpu::RenderPipeline> {
    let shader = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("custom sky"),
            source: wgpu::ShaderSource::Wgsl(include_str!("custom_sky.wgsl").into()),
        });
    let layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("custom sky layout"),
            bind_group_layouts: &[Some(texture_layout), Some(uniform_layout)],
            immediate_size: 0,
        });
    SkyBlend::ALL
        .iter()
        .map(|&blend| {
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("custom sky layer"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[Some(CubeVertex::LAYOUT)],
                    },
                    // Seen from inside; no culling, like the rest of the sky.
                    primitive: wgpu::PrimitiveState::default(),
                    // The sky draws before anything has depth.
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: gpu.sample_count(),
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.config.format,
                            blend: blend_state(blend),
                            write_mask: wgpu::ColorWrites::COLOR,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        })
        .collect()
}

/// Draws a [`CustomSkyAssets`] every frame. Lives in
/// [`crate::sky::dome::SkyPhase::custom_sky`]; owns the per-frame uniform buffer.
pub struct CustomSkyRenderer {
    assets: Arc<CustomSkyAssets>,
    uniform: wgpu::Buffer,
    uniform_group: wgpu::BindGroup,
    /// Bytes between two layers' uniforms (the device's dynamic-offset alignment).
    stride: usize,
    /// This frame's uniforms, `stride` apart.
    staging: Vec<u8>,
    /// This frame's draws: layer index and uniform offset.
    draws: Vec<(usize, u32)>,
    /// How much of the sky the last [`CustomSkyRenderer::draw`] covered ([`Self::cover`]).
    cover: f32,
    /// Each layer's biome/height condition, eased over its `transition` seconds.
    condition: Vec<f32>,
    /// The condition has its first value (no fade-in right after loading).
    condition_ready: bool,
    last_frame: Option<Instant>,
    /// Sample every layer with linear filtering instead of nearest.
    pub smooth: bool,
    /// The biome at the camera, for `biomes=` layers. `None`: unknown, and such layers hide.
    pub camera_biome: Option<u8>,
    /// The camera's block Y, for `heights=` layers. `None`: unknown, and such layers hide.
    pub camera_y: Option<i32>,
}

impl std::fmt::Debug for CustomSkyRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomSkyRenderer")
            .field("assets", &self.assets)
            .field("smooth", &self.smooth)
            .finish_non_exhaustive()
    }
}

impl CustomSkyRenderer {
    /// A renderer over `assets`, which must [`CustomSkyAssets::fits`] `gpu`.
    #[must_use]
    pub fn new(gpu: &Gpu, assets: Arc<CustomSkyAssets>) -> CustomSkyRenderer {
        let align = gpu.device.limits().min_uniform_buffer_offset_alignment as usize;
        let size = std::mem::size_of::<LayerUniform>();
        let stride = size.div_ceil(align.max(1)) * align.max(1);
        let slots = assets
            .world_layers
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .max(1);
        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("custom sky layers"),
            size: (slots * stride) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("custom sky layers"),
            layout: &assets.uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform,
                    offset: 0,
                    size: wgpu::BufferSize::new(size as u64),
                }),
            }],
        });
        let layer_count = assets.layers.len();
        CustomSkyRenderer {
            assets,
            uniform,
            uniform_group,
            stride,
            staging: Vec::with_capacity(slots * stride),
            draws: Vec::with_capacity(slots),
            cover: 0.0,
            condition: vec![1.0; layer_count],
            condition_ready: false,
            last_frame: None,
            smooth: false,
            camera_biome: None,
            camera_y: None,
        }
    }

    /// The shared textures and pipelines.
    #[must_use]
    pub fn assets(&self) -> &Arc<CustomSkyAssets> {
        &self.assets
    }

    /// Whether `dimension` has custom layers; the vanilla stars are then not drawn.
    #[must_use]
    pub fn has_layers(&self, dimension: Dimension) -> bool {
        self.assets.layer_count(dimension) > 0
    }

    /// Advance the biome/height conditions toward their targets.
    fn update_conditions(&mut self, now: Instant) {
        let dt = self
            .last_frame
            .map_or(0.0, |last| {
                now.saturating_duration_since(last).as_secs_f32()
            })
            .min(1.0);
        self.last_frame = Some(now);
        for (layer, condition) in self.assets.layers.iter().zip(&mut self.condition) {
            if layer.def.biomes.is_none() && layer.def.heights.is_none() {
                continue;
            }
            let target = position_condition(&layer.def, self.camera_biome, self.camera_y);
            *condition = if !self.condition_ready || layer.def.transition <= 0.0 {
                target
            } else {
                // Exponential approach: ~99 % of the way after `transition` seconds.
                let k = 1.0 - (-dt * 4.6 / layer.def.transition).exp();
                *condition + (target - *condition) * k
            };
        }
        self.condition_ready = true;
    }

    /// Draw this frame's visible layers into `pass`. `view_proj` is the sky projection times the
    /// camera's rotation-only view. Returns `false` when nothing was drawn, leaving the pass's
    /// pipeline and bind groups untouched; after `true` the caller must re-bind its own.
    pub fn draw(
        &mut self,
        gpu: &Gpu,
        pass: &mut wgpu::RenderPass<'_>,
        env: &Environment,
        partial_ticks: f32,
        view_proj: Mat4,
    ) -> bool {
        mc_core::profile_scope!("sky.custom.draw");
        self.cover = 0.0;
        let Some(world) = dimension_world(env.dimension) else {
            return false;
        };
        if self.assets.needs_position {
            self.update_conditions(Instant::now());
        }
        let time_of_day = env.world_time.rem_euclid(i64::from(TICKS_PER_DAY)) as i32;
        let celestial_angle = env.celestial_angle(partial_ticks);
        let (rain, thunder) = (env.weather.rain_strength, env.weather.thunder_strength);

        self.staging.clear();
        self.draws.clear();
        // The share of the sky that every drawn layer leaves clear.
        let mut clear = 1.0;
        for (index, layer) in self.assets.layers.iter().enumerate() {
            if layer.def.world != world {
                continue;
            }
            let brightness = (layer_brightness(&layer.def, time_of_day, rain, thunder)
                * self.condition[index])
                .clamp(0.0, 1.0);
            if brightness < MIN_BRIGHTNESS {
                continue;
            }
            clear *= 1.0 - layer_cover(layer.def.blend, brightness);
            let rotation = layer_rotation(&layer.def, env.world_time, celestial_angle);
            let texture = &self.assets.textures[layer.texture];
            let cutoff = if layer.def.blend == SkyBlend::Replace {
                REPLACE_ALPHA_CUTOFF
            } else {
                -1.0
            };
            let uniform = LayerUniform {
                view_proj: (view_proj * rotation).to_cols_array_2d(),
                color: fade_color(layer.def.blend, brightness),
                params: [cutoff, 0.0, texture.half_texel[0], texture.half_texel[1]],
            };
            let offset = self.staging.len();
            self.staging.extend_from_slice(bytemuck::bytes_of(&uniform));
            self.staging.resize(offset + self.stride, 0);
            self.draws.push((index, offset as u32));
        }
        self.cover = 1.0 - clear;
        if self.draws.is_empty() {
            return false;
        }
        gpu.write_buffer_staged(&self.uniform, 0, &self.staging);
        pass.set_vertex_buffer(0, self.assets.cube.slice(..));
        let mut bound = None;
        for &(index, offset) in &self.draws {
            let layer = &self.assets.layers[index];
            let pipeline = layer.def.blend.index();
            if bound != Some(pipeline) {
                pass.set_pipeline(&self.assets.pipelines[pipeline]);
                bound = Some(pipeline);
            }
            let texture = &self.assets.textures[layer.texture];
            let group = if self.smooth || texture.blur {
                &texture.linear
            } else {
                &texture.nearest
            };
            pass.set_bind_group(0, group, &[]);
            pass.set_bind_group(1, &self.uniform_group, &[offset]);
            pass.draw(0..CUBE_VERTICES, 0..1);
        }
        true
    }

    /// The same layers in a render pass of their own, with no vanilla sky under them (tests and
    /// tools; the game draws through [`CustomSkyRenderer::draw`] inside the sky pass).
    pub fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
        env: &Environment,
        partial_ticks: f32,
    ) {
        let view_proj = crate::sky::dome::sky_projection(cam) * cam.view;
        let mut pass = gpu.begin_pass(enc, view, "sky/custom", crate::DepthAttachment::None);
        self.draw(gpu, &mut pass, env, partial_ticks, view_proj);
    }

    /// How many layers the last [`CustomSkyRenderer::draw`] drew.
    #[must_use]
    pub fn last_draw_count(&self) -> usize {
        self.draws.len()
    }

    /// How much of the sky the last [`CustomSkyRenderer::draw`] covered, 0..1 ([`layer_cover`]
    /// of each drawn layer, combined): 0 when every layer was faded out by the time of day, the
    /// weather or its biome/height condition, and the vanilla sky shows.
    #[must_use]
    pub fn cover(&self) -> f32 {
        self.cover
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_assets::custom_sky::{parse_layer, parse_time, SkyFade};

    fn layer(text: &str) -> SkyLayer {
        parse_layer(
            0,
            1,
            &ResourceLocation::minecraft("mcpatcher/sky/world0/sky1.properties"),
            text,
        )
        .expect("valid layer")
    }

    #[test]
    fn custom_sky_wgsl_parses_and_validates() {
        let source = include_str!("custom_sky.wgsl");
        let module = wgpu::naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("custom_sky.wgsl validates");
    }

    /// Every face is a unit square on its own side of the cube, its image "up" and "right"
    /// are the directions a viewer inside sees as up and right (no mirroring), and neighbouring
    /// faces share their edges.
    #[test]
    fn faces_follow_the_formats_template() {
        let names: Vec<&str> = FACES.iter().map(|f| f.name).collect();
        assert_eq!(names, ["bottom", "top", "south", "west", "north", "east"]);
        let cells: Vec<(u8, u8)> = FACES.iter().map(|f| (f.col, f.row)).collect();
        assert_eq!(cells, [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]);
        let normal = |name: &str| -> Vec3 {
            match name {
                "bottom" => Vec3::NEG_Y,
                "top" => Vec3::Y,
                "south" => Vec3::Z,
                "north" => Vec3::NEG_Z,
                "west" => Vec3::NEG_X,
                "east" => Vec3::X,
                _ => unreachable!(),
            }
        };
        for face in &FACES {
            let c = face.corners.map(Vec3::from);
            let n = normal(face.name);
            let centre = (c[0] + c[1] + c[2] + c[3]) / 4.0;
            assert_eq!(centre, n, "{} sits on its side", face.name);
            let right = (c[1] - c[0]).normalize();
            let down = (c[3] - c[0]).normalize();
            // Seen from inside, looking along `n` with "up" = -down, right = n x up.
            let up = -down;
            assert!(
                (n.cross(up) - right).length() < 1e-6,
                "{} is not mirrored",
                face.name
            );
            match face.name {
                "top" => assert_eq!(up, Vec3::Z, "the top's image top is south"),
                "bottom" => assert_eq!(up, Vec3::NEG_Z, "the bottom's image top is north"),
                _ => assert_eq!(up, Vec3::Y, "{} stands upright", face.name),
            }
        }
        // The panorama: west -> north -> east -> south -> west, right edge to left edge.
        let face = |name: &str| FACES.iter().find(|f| f.name == name).unwrap();
        for (a, b) in [
            ("west", "north"),
            ("north", "east"),
            ("east", "south"),
            ("south", "west"),
        ] {
            assert_eq!(face(a).corners[1], face(b).corners[0], "{a} -> {b} top");
            assert_eq!(face(a).corners[2], face(b).corners[3], "{a} -> {b} bottom");
        }
        // The top's lower edge meets the north wall's upper edge, the bottom's upper edge its lower.
        assert_eq!(face("top").corners[3], face("north").corners[0]);
        assert_eq!(face("top").corners[2], face("north").corners[1]);
        assert_eq!(face("bottom").corners[0], face("north").corners[3]);
        assert_eq!(face("bottom").corners[1], face("north").corners[2]);
    }

    #[test]
    fn cube_vertices_cover_each_cell_exactly() {
        let vertices = cube_vertices();
        assert_eq!(vertices.len(), CUBE_VERTICES as usize);
        for (face, chunk) in FACES.iter().zip(vertices.chunks(6)) {
            let (u0, v0) = (f32::from(face.col) / 3.0, f32::from(face.row) / 2.0);
            for v in chunk {
                assert_eq!(v.cell[0], u0);
                assert_eq!(v.cell[1], v0);
                assert!(v.uv[0] >= v.cell[0] && v.uv[0] <= v.cell[2]);
                assert!(v.uv[1] >= v.cell[1] && v.uv[1] <= v.cell[3]);
                assert!((Vec3::from(v.pos).abs().max_element() - CUBE_RADIUS).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn fade_colours_scale_the_channels_each_mode_fades() {
        assert_eq!(fade_color(SkyBlend::Add, 0.25), [1.0, 1.0, 1.0, 0.25]);
        assert_eq!(fade_color(SkyBlend::Alpha, 0.25), [1.0, 1.0, 1.0, 0.25]);
        assert_eq!(fade_color(SkyBlend::Replace, 0.25), [1.0, 1.0, 1.0, 0.25]);
        assert_eq!(fade_color(SkyBlend::Multiply, 0.25), [0.25; 4]);
        for blend in [
            SkyBlend::Subtract,
            SkyBlend::Dodge,
            SkyBlend::Burn,
            SkyBlend::Screen,
            SkyBlend::Overlay,
        ] {
            assert_eq!(fade_color(blend, 0.25), [0.25, 0.25, 0.25, 1.0]);
        }
        assert!(blend_state(SkyBlend::Replace).is_none());
        for blend in SkyBlend::ALL {
            if blend != SkyBlend::Replace {
                let state = blend_state(blend).unwrap();
                assert_eq!(state.alpha.dst_factor, wgpu::BlendFactor::One, "alpha kept");
            }
        }
    }

    #[test]
    fn weather_shares_add_up_and_the_default_fades_out_in_rain() {
        let clear = SkyWeather::default();
        assert_eq!(weather_brightness(clear, 0.0, 0.0), 1.0);
        assert!((weather_brightness(clear, 0.25, 0.0) - 0.75).abs() < 1e-6);
        assert_eq!(weather_brightness(clear, 1.0, 1.0), 0.0);
        let all = SkyWeather {
            clear: true,
            rain: true,
            thunder: true,
        };
        for (rain, thunder) in [(0.0, 0.0), (0.4, 0.1), (1.0, 1.0), (0.7, 0.7)] {
            assert!((weather_brightness(all, rain, thunder) - 1.0).abs() < 1e-6);
        }
        let storm = SkyWeather {
            clear: false,
            rain: false,
            thunder: true,
        };
        assert!((weather_brightness(storm, 0.8, 0.5) - 0.5).abs() < 1e-6);
        let wet = SkyWeather {
            clear: false,
            rain: true,
            thunder: false,
        };
        assert!((weather_brightness(wet, 0.8, 0.5) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn layer_brightness_is_fade_times_weather() {
        let l = layer("startFadeIn=19:00\nendFadeIn=20:00\nendFadeOut=5:00");
        let t = |hhmm: &str| parse_time(hhmm).unwrap();
        assert!((layer_brightness(&l, t("19:30"), 0.0, 0.0) - 0.5).abs() < 1e-3);
        assert!((layer_brightness(&l, t("19:30"), 0.5, 0.0) - 0.25).abs() < 1e-3);
        assert_eq!(layer_brightness(&l, t("12:00"), 0.0, 0.0), 0.0);
        let always = layer("");
        assert_eq!(always.fade, None::<SkyFade>);
        assert_eq!(layer_brightness(&always, t("12:00"), 0.0, 0.0), 1.0);
    }

    /// A `replace` layer covers the sky while it passes the alpha test and not at all after;
    /// the other modes cover their brightness.
    #[test]
    fn layers_cover_the_sky_as_they_draw() {
        assert_eq!(layer_cover(SkyBlend::Replace, 1.0), 1.0);
        assert_eq!(layer_cover(SkyBlend::Replace, 0.11), 1.0);
        assert_eq!(layer_cover(SkyBlend::Replace, REPLACE_ALPHA_CUTOFF), 0.0);
        assert_eq!(layer_cover(SkyBlend::Replace, 0.0), 0.0);
        for blend in [
            SkyBlend::Alpha,
            SkyBlend::Add,
            SkyBlend::Multiply,
            SkyBlend::Screen,
        ] {
            assert_eq!(layer_cover(blend, 0.25), 0.25);
            assert_eq!(layer_cover(blend, 0.0), 0.0);
            assert_eq!(layer_cover(blend, 1.0), 1.0);
        }
    }

    #[test]
    fn rotation_follows_the_celestial_angle_and_stays_continuous_across_days() {
        let l = layer("speed=1");
        assert_eq!(layer_turns(&l, 6000, 0.0), 0.0);
        assert!((layer_turns(&l, 12_000, 0.25) - 0.25).abs() < 1e-6);
        // speed 200 (the user's shooting stars): only the fraction matters.
        let fast = layer("speed=200");
        assert!((layer_turns(&fast, 18_000, 0.5) - 0.0).abs() < 1e-4);
        assert!((layer_turns(&fast, 18_000, 0.501) - 0.2).abs() < 1e-3);
        // speed 1.3: the turn just before noon equals the turn at noon of the next day.
        let slow = layer("speed=1.3");
        let before = layer_turns(&slow, 24_000 + 5_999, 0.999_99);
        let after = layer_turns(&slow, 24_000 + 6_000, 0.0);
        let gap = (before - after).abs();
        assert!(gap.min(1.0 - gap) < 1e-3, "{before} vs {after}");
        // Without rotate the matrix is the identity.
        let fixed = layer("rotate=false");
        assert_eq!(layer_rotation(&fixed, 12_000, 0.25), Mat4::IDENTITY);
    }

    #[test]
    fn the_default_axis_turns_a_layer_like_the_sun() {
        let l = layer("");
        // A quarter turn (sunset): the top face's centre moves to the west, like the sun.
        let r = layer_rotation(&l, 12_000, 0.25);
        let top = r.transform_vector3(Vec3::Y);
        assert!((top - Vec3::NEG_X).length() < 1e-5, "{top}");
        // About "up", a quarter turn takes north to west (counterclockwise seen from above,
        // clockwise for a viewer looking up).
        let spin = layer("axis=0 1 0");
        let r = layer_rotation(&spin, 12_000, 0.25);
        let north = r.transform_vector3(Vec3::NEG_Z);
        assert!((north - Vec3::NEG_X).length() < 1e-5, "{north}");
    }

    #[test]
    fn biome_and_height_conditions() {
        let l = layer("biomes=Desert\nheights=60-80");
        assert_eq!(position_condition(&l, Some(2), Some(70)), 1.0);
        assert_eq!(position_condition(&l, Some(1), Some(70)), 0.0);
        assert_eq!(position_condition(&l, Some(2), Some(90)), 0.0);
        assert_eq!(position_condition(&l, None, Some(70)), 0.0);
        assert_eq!(position_condition(&layer(""), None, None), 1.0);
    }
}
