//! The GPU half of the menu scene: one shader module (`menu_scene.wgsl`), a static shadow map,
//! and per-frame passes.
//!
//! Per frame, all inside the game's frame encoder:
//!
//! 1. **scene** (4x MSAA, HDR): opaque blocks, alpha-to-coverage cutouts (leaves, plants, glass
//!    frames), the procedural sky behind them, then the glass panes' sheen. Resolved to `scene`.
//! 2. **depth**: the MSAA depth turned into planar view depth (`R32Float`).
//! 3. **water** (single sample): `scene` copied to `water`, then the waterfalls, pools and mist
//!    drawn over it, refracting `scene` and depth-testing against the planar depth by hand.
//! 4. **bloom**: a 5-level down/up chain from `water`.
//! 5. **shafts** (quarter resolution): the sun's shadow map ray-marched through the interior.
//! 6. **composite** into the frame: haze, shafts, bloom, exposure, tone map, grade, vignette,
//!    dither, sRGB encode.
//!
//! The finish phase is one more full-screen pass over the frame snapshot.

use std::cell::RefCell;
use std::rc::Rc;

use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use super::camera;
use super::greenhouse::{VOX_MAX, VOX_MIN};
use super::prepare::{PreparedScene, SceneVertex, WaterVertex};
use crate::pipeline::SceneTextures;
// Volumetric clouds in the title's sky.
use crate::sky::volumetric::{
    CloudFrame, CloudLayer, CloudLook, CloudQuality, CloudTarget, VolumetricClouds,
};
use crate::{DepthAttachment, Gpu};

const SHADER: &str = include_str!("menu_scene.wgsl");
const MSAA: u32 = 4;
const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const PLANAR: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const SHAFTS: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
/// 16 bits over the ~150-block light-space depth range is ~0.002 blocks per step: plenty, at
/// half the memory of `Depth32Float` (32 MiB at 4096²).
const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth16Unorm;
const SHADOW_SIZE: u32 = 4096;
const BLOOM_LEVELS: usize = 6;
const NEAR: f32 = 0.08;

/// Toward the sun: high, to the camera's right (-x) and a little ahead.
const SUN: [f32; 3] = [-0.42, 0.80, 0.42];

/// The grading knobs. The defaults are the tuned look; [`super::MenuScene::set_look`] exists
/// for the render harness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Look {
    /// Exposure before the tone map.
    pub exposure: f32,
    /// Linear sunlight colour.
    pub sun: [f32; 3],
    /// Haze extinction per block inside the greenhouse.
    pub haze: f32,
    /// How far the haze may pull a pixel toward the haze colour, `0..=1`.
    pub haze_mix: f32,
    /// Light-shaft strength.
    pub shafts: f32,
    /// Bloom added before the tone map.
    pub bloom: f32,
    /// The soft "Orton" glow screened over the tone-mapped image, `0..=1`.
    pub glow: f32,
    /// `1` shows the scene without haze, shafts, bloom or glow (for tuning).
    pub debug: f32,
    /// How much the left of the frame, where the title and buttons sit, is darkened, `0..=1`.
    pub scrim: f32,
}

impl Default for Look {
    fn default() -> Self {
        Self {
            exposure: 0.8,
            sun: [2.6, 2.42, 2.1],
            haze: 0.004,
            haze_mix: 0.55,
            shafts: 0.0022,
            bloom: 0.05,
            glow: 0.18,
            debug: 0.0,
            scrim: 0.12,
        }
    }
}

/// The title's volumetric clouds: lit by the scene's own sun, in its HDR
/// scale (`Look::sun`), with a blue sky ambient, drifting on the scene clock. Fair-weather
/// cumulus with plenty of blue between them (the reference image's sky; with the roof's white
/// frames, more cover read as a white ceiling).
fn title_cloud_look(time: f64, look: &Look) -> CloudLook {
    CloudLook {
        coverage: 0.4,
        rain: 0.0,
        type_bias: 0.15,
        sun_dir: Vec3::from(SUN).normalize(),
        sun_color: Vec3::from(look.sun) * 1.15,
        ambient_top: Vec3::new(0.34, 0.50, 0.86),
        ambient_bottom: Vec3::new(0.16, 0.21, 0.30),
        ambient: 1.0,
        powder: 0.7,
        exposure: 1.0,
        wind: [time * 9.0, time * 3.0],
        drift: time * 1.2,
    }
}

/// Must match `struct Frame` in `menu_scene.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    shadow_mat: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    cam_fwd: [f32; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    sun_dir: [f32; 4],
    sun_color: [f32; 4],
    screen: [f32; 4],
    box_min: [f32; 4],
    box_max: [f32; 4],
    params: [f32; 4],
    look: [f32; 4],
    /// x = how much the volumetric clouds replace the painted ones (0..1,
    /// 0 until they are ready), y/z = 1 / the frame pixels their trace spans.
    clouds: [f32; 4],
}

struct Pipelines {
    shadow_opaque: wgpu::RenderPipeline,
    shadow_cutout: wgpu::RenderPipeline,
    opaque: wgpu::RenderPipeline,
    cutout: wgpu::RenderPipeline,
    sky: wgpu::RenderPipeline,
    glass: wgpu::RenderPipeline,
    depth: wgpu::RenderPipeline,
    water: wgpu::RenderPipeline,
    mist: wgpu::RenderPipeline,
    bloom_first: wgpu::RenderPipeline,
    bloom_down: wgpu::RenderPipeline,
    bloom_up: wgpu::RenderPipeline,
    shafts: wgpu::RenderPipeline,
}

/// The two pipelines that write the game's frame, keyed by its format and sample count.
struct OutputPipelines {
    key: (wgpu::TextureFormat, u32),
    composite: wgpu::RenderPipeline,
    finish: wgpu::RenderPipeline,
}

struct Targets {
    size: (u32, u32),
    msaa_color: wgpu::TextureView,
    msaa_depth: wgpu::TextureView,
    scene: wgpu::Texture,
    scene_view: wgpu::TextureView,
    water: wgpu::Texture,
    water_view: wgpu::TextureView,
    planar: wgpu::TextureView,
    bloom: Vec<(wgpu::TextureView, (u32, u32))>,
    shafts: wgpu::TextureView,
    g_water: wgpu::BindGroup,
    g_bloom_first: wgpu::BindGroup,
    g_bloom_down: Vec<wgpu::BindGroup>,
    g_bloom_up: Vec<wgpu::BindGroup>,
    g_shafts: wgpu::BindGroup,
    g_composite: wgpu::BindGroup,
    g_ms: wgpu::BindGroup,
    /// Group 1 with nothing bound, for the pass that writes the planar depth.
    g_none: wgpu::BindGroup,
}

/// Quads per draw through the shared `u16` quad index buffer (65,536 vertices).
const QUAD_BATCH: u32 = 16_384;

/// The GPU objects that do not depend on the visit: the shader module, layouts, pipelines,
/// samplers, the noise texture, stand-in textures and the shared quad index buffer.
///
/// Owned by one [`Renderer`] and dropped with it. Deliberately **not** cached in a
/// thread-local across visits: measured (aero-title leak follow-up, 2026-11-07), caching them
/// did not change the per-visit resident-set growth, and a thread-local holding wgpu objects
/// (and so the device) past the `Gpu` made the process spin forever in thread-local teardown.
struct Statics {
    module: wgpu::ShaderModule,
    frame_layout: wgpu::BindGroupLayout,
    input_layout: wgpu::BindGroupLayout,
    ms_layout: wgpu::BindGroupLayout,
    layouts: [wgpu::PipelineLayout; 3],
    pipes: Pipelines,
    /// The frame-writing pipelines, one entry per (format, sample count) seen; usually one.
    output: RefCell<Vec<Rc<OutputPipelines>>>,
    atlas_sampler: wgpu::Sampler,
    shadow_sampler: wgpu::Sampler,
    noise_sampler: wgpu::Sampler,
    linear_sampler: wgpu::Sampler,
    noise_view: wgpu::TextureView,
    dummy_hdr: wgpu::TextureView,
    dummy_planar: wgpu::TextureView,
    dummy_depth: wgpu::TextureView,
    /// `0,1,2, 0,2,3` per quad for [`QUAD_BATCH`] quads, drawn with a base vertex.
    quad_indices: wgpu::Buffer,
    _textures: Vec<wgpu::Texture>,
}

/// The scene's GPU side. Meant to live as long as the client wants a title screen: the client
/// keeps one [`super::MenuScene`] across visits (aero-title leak follow-up, 2026-11-07).
///
/// Why: building and freeing all of this on every visit made the GPU driver's own user-mode
/// memory (not wgpu's, not Rust's: both were measured flat) ramp up by about 23 MB over the
/// first ~14 visits before levelling off, which the client's leak test reads as a leak. Kept
/// alive, nothing is rebuilt per visit. [`Renderer::release`] gives the big memory back when the
/// title screen has been gone a while.
pub(super) struct Renderer {
    prepared: &'static PreparedScene,
    statics: Rc<Statics>,
    time: f64,
    frame: u32,
    pub(super) vhs_strength: f32,
    pub(super) look: Look,
    uniform: wgpu::Buffer,
    shadow_mat: Mat4,
    /// The atlas, geometry, shadow map and their bind groups: built on the first frame drawn,
    /// dropped by [`Renderer::release`].
    resident: Option<Resident>,
    output: Option<Rc<OutputPipelines>>,
    targets: Option<Targets>,
    /// The finish phase's group 1 and the frame snapshot it was built for, reused while the
    /// phase list hands over the same snapshot.
    finish_group: Option<(wgpu::TextureView, wgpu::BindGroup)>,
    /// The volumetric clouds in the sky: built off-thread the first time
    /// the background draws; the painted clouds show until they are ready.
    clouds: VolumetricClouds,
    /// The sky's group 1 over the clouds' output, and the output generation it was built for.
    cloud_group: Option<(u64, wgpu::BindGroup)>,
    /// The cross-fade from the painted clouds to the volumetric ones, 0..1 over 0.6 s.
    cloud_fade: f32,
    /// The scene time the clouds last traced at.
    cloud_clock: f64,
    /// Whether the sky draws them at all (`MenuScene::set_volumetric_clouds`).
    clouds_enabled: bool,
}

/// The big, rebuildable part of the scene's GPU memory (about 50 MB, mostly the shadow map and
/// the vertex buffer).
struct Resident {
    frame_group: wgpu::BindGroup,
    /// `frame_group` with a stand-in for the shadow map, for the pass that draws the shadow map.
    shadow_pass_group: wgpu::BindGroup,
    vertices: wgpu::Buffer,
    water_vertices: wgpu::Buffer,
    shadow_view: wgpu::TextureView,
    shadow_ready: bool,
    /// The atlas and the shadow map.
    _textures: [wgpu::Texture; 2],
}

/// Draw quads `range` of the bound vertex buffer through the shared quad index buffer, in
/// batches of [`QUAD_BATCH`] with a base vertex.
fn draw_quads(pass: &mut wgpu::RenderPass<'_>, range: (u32, u32)) {
    let mut first = range.0;
    while first < range.1 {
        let count = (range.1 - first).min(QUAD_BATCH);
        pass.draw_indexed(0..count * 6, (first * 4) as i32, 0..1);
        first += count;
    }
}

fn texture(
    gpu: &Gpu,
    label: &str,
    size: (u32, u32),
    format: wgpu::TextureFormat,
    samples: u32,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0.max(1),
            height: size.1.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn view(t: &wgpu::Texture) -> wgpu::TextureView {
    t.create_view(&wgpu::TextureViewDescriptor::default())
}

/// The most pixels the scene renders at: 1920 x 1080. Larger frames render the scene at this
/// pixel count with the frame's aspect and the composite upscales it, so the cost stays at the
/// 1080p budget on a 1440p or 4K monitor.
const MAX_SCENE_PIXELS: f64 = 1920.0 * 1080.0;

fn internal_size(width: u32, height: u32) -> (u32, u32) {
    let pixels = f64::from(width) * f64::from(height);
    if pixels <= MAX_SCENE_PIXELS {
        return (width, height);
    }
    let s = (MAX_SCENE_PIXELS / pixels).sqrt();
    (
        ((f64::from(width) * s) as u32).max(1),
        ((f64::from(height) * s) as u32).max(1),
    )
}

/// A tileable 256x256 noise texture: R and G are gradient noise with 16 and 32 cells per tile,
/// B is inverted cellular noise with 8 cells, A is white noise.
fn noise_pixels() -> Vec<u8> {
    const N: usize = 256;
    let grad = |ix: i32, iy: i32, period: i32, salt: u32| -> (f32, f32) {
        let h = super::greenhouse::hash(ix.rem_euclid(period), iy.rem_euclid(period), 0, salt);
        let a = (h >> 8) as f32 / 16_777_216.0 * std::f32::consts::TAU;
        (a.cos(), a.sin())
    };
    let perlin = |x: f32, y: f32, period: i32, salt: u32| -> f32 {
        let (ix, iy) = (x.floor() as i32, y.floor() as i32);
        let (fx, fy) = (x - x.floor(), y - y.floor());
        let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        let dot = |gx: i32, gy: i32, dx: f32, dy: f32| {
            let g = grad(gx, gy, period, salt);
            g.0 * dx + g.1 * dy
        };
        let (u, v) = (fade(fx), fade(fy));
        let a = dot(ix, iy, fx, fy) + (dot(ix + 1, iy, fx - 1.0, fy) - dot(ix, iy, fx, fy)) * u;
        let b = dot(ix, iy + 1, fx, fy - 1.0)
            + (dot(ix + 1, iy + 1, fx - 1.0, fy - 1.0) - dot(ix, iy + 1, fx, fy - 1.0)) * u;
        (a + (b - a) * v) * 0.75 + 0.5
    };
    let cellular = |x: f32, y: f32, period: i32, salt: u32| -> f32 {
        let (ix, iy) = (x.floor() as i32, y.floor() as i32);
        let mut best = 9.0f32;
        for oy in -1..=1 {
            for ox in -1..=1 {
                let (cx, cy) = (ix + ox, iy + oy);
                let h =
                    super::greenhouse::hash(cx.rem_euclid(period), cy.rem_euclid(period), 1, salt);
                let px = cx as f32 + (h & 0xFFFF) as f32 / 65_536.0;
                let py = cy as f32 + (h >> 16) as f32 / 65_536.0;
                best = best.min((px - x).powi(2) + (py - y).powi(2));
            }
        }
        1.0 - best.sqrt().min(1.0)
    };
    let mut out = vec![0u8; N * N * 4];
    for y in 0..N {
        for x in 0..N {
            let (u, v) = (x as f32 / N as f32, y as f32 / N as f32);
            let r = perlin(u * 16.0, v * 16.0, 16, 11);
            let g = perlin(u * 32.0, v * 32.0, 32, 23);
            let b = cellular(u * 8.0, v * 8.0, 8, 37);
            let a = (super::greenhouse::hash(x as i32, y as i32, 7, 41) >> 24) as f32 / 255.0;
            let i = (y * N + x) * 4;
            for (k, c) in [r, g, b, a].into_iter().enumerate() {
                out[i + k] = (c.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    out
}

impl Statics {
    #[allow(clippy::too_many_lines)] // One constructor: textures, layouts, pipelines.
    fn new(gpu: &Gpu) -> Self {
        let device = &gpu.device;
        let mut keep = Vec::new();

        let noise_tex = texture(
            gpu,
            "menu scene noise",
            (256, 256),
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        gpu.queue.write_texture(
            noise_tex.as_image_copy(),
            &noise_pixels(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4),
                rows_per_image: Some(256),
            },
            wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
        );
        let noise_view = view(&noise_tex);
        keep.push(noise_tex);

        let dummy_hdr_tex = texture(
            gpu,
            "menu scene dummy",
            (1, 1),
            HDR,
            1,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let dummy_planar_tex = texture(
            gpu,
            "menu scene dummy depth",
            (1, 1),
            PLANAR,
            1,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let dummy_hdr = view(&dummy_hdr_tex);
        let dummy_planar = view(&dummy_planar_tex);
        keep.push(dummy_hdr_tex);
        keep.push(dummy_planar_tex);

        let sampler = |label: &str,
                       address: wgpu::AddressMode,
                       filter: wgpu::FilterMode,
                       mip: wgpu::MipmapFilterMode,
                       compare: Option<wgpu::CompareFunction>| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                address_mode_u: address,
                address_mode_v: address,
                address_mode_w: address,
                mag_filter: filter,
                min_filter: filter,
                mipmap_filter: mip,
                compare,
                ..wgpu::SamplerDescriptor::default()
            })
        };
        // Minecraft's look: nearest texels up close, sprite-local mips at a distance.
        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("menu scene atlas sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..wgpu::SamplerDescriptor::default()
        });
        let shadow_sampler = sampler(
            "menu scene shadow sampler",
            wgpu::AddressMode::ClampToEdge,
            wgpu::FilterMode::Linear,
            wgpu::MipmapFilterMode::Nearest,
            Some(wgpu::CompareFunction::LessEqual),
        );
        let noise_sampler = sampler(
            "menu scene noise sampler",
            wgpu::AddressMode::Repeat,
            wgpu::FilterMode::Linear,
            wgpu::MipmapFilterMode::Nearest,
            None,
        );
        let linear_sampler = sampler(
            "menu scene linear sampler",
            wgpu::AddressMode::ClampToEdge,
            wgpu::FilterMode::Linear,
            wgpu::MipmapFilterMode::Nearest,
            None,
        );

        let tex_entry = |binding: u32, filterable: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let samp_entry = |binding: u32, ty: wgpu::SamplerBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Sampler(ty),
            count: None,
        };
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("menu scene group0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex_entry(1, true),
                samp_entry(2, wgpu::SamplerBindingType::Filtering),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                samp_entry(4, wgpu::SamplerBindingType::Comparison),
                tex_entry(5, true),
                samp_entry(6, wgpu::SamplerBindingType::Filtering),
                samp_entry(7, wgpu::SamplerBindingType::Filtering),
            ],
        });
        let input_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("menu scene group1"),
            entries: &[
                tex_entry(0, true),
                tex_entry(1, false),
                tex_entry(2, true),
                tex_entry(3, true),
            ],
        });
        let ms_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("menu scene group2"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: true,
                },
                count: None,
            }],
        });
        // The shadow pass writes the shadow map, so it binds a stand-in depth texture there.
        let dummy_depth_tex = texture(
            gpu,
            "menu scene dummy shadow",
            (1, 1),
            SHADOW_FORMAT,
            1,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let dummy_depth = view(&dummy_depth_tex);
        keep.push(dummy_depth_tex);

        let layout = |label: &str, groups: &[&wgpu::BindGroupLayout]| {
            let groups: Vec<Option<&wgpu::BindGroupLayout>> =
                groups.iter().map(|g| Some(*g)).collect();
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &groups,
                immediate_size: 0,
            })
        };
        let layouts = [
            layout("menu scene layout g0", &[&frame_layout]),
            layout("menu scene layout g0 g1", &[&frame_layout, &input_layout]),
            layout(
                "menu scene layout g0 g1 g2",
                &[&frame_layout, &input_layout, &ms_layout],
            ),
        ];
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("menu scene"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let pipes = Self::pipelines(gpu, &module, &layouts);

        let indices: Vec<u16> = (0..QUAD_BATCH as u16)
            .flat_map(|q| {
                let b = q.wrapping_mul(4);
                [b, b + 1, b + 2, b, b + 2, b + 3]
            })
            .collect();
        let quad_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("menu scene quad indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            module,
            frame_layout,
            input_layout,
            ms_layout,
            layouts,
            pipes,
            output: RefCell::new(Vec::new()),
            atlas_sampler,
            shadow_sampler,
            noise_sampler,
            linear_sampler,
            noise_view,
            dummy_hdr,
            dummy_planar,
            dummy_depth,
            quad_indices,
            _textures: keep,
        }
    }
}

impl Resident {
    #[allow(clippy::too_many_lines)] // One constructor: the atlas, buffers and bind groups.
    fn new(
        gpu: &Gpu,
        statics: &Statics,
        uniform: &wgpu::Buffer,
        prepared: &'static PreparedScene,
    ) -> Self {
        let device = &gpu.device;

        // --- the vanilla atlas, sRGB so sampling linearises, with its sprite-local mips.
        let atlas = prepared.atlas;
        let levels = 1 + atlas.mipmaps.len() as u32;
        let atlas_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("menu scene atlas"),
            size: wgpu::Extent3d {
                width: atlas.w,
                height: atlas.h,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (level, pixels) in std::iter::once(&atlas.texture)
            .chain(atlas.mipmaps.iter())
            .enumerate()
        {
            let (w, h) = ((atlas.w >> level).max(1), (atlas.h >> level).max(1));
            if pixels.len() != (w * h * 4) as usize {
                break;
            }
            let mut dst = atlas_tex.as_image_copy();
            dst.mip_level = level as u32;
            gpu.queue.write_texture(
                dst,
                pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 4),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
        }
        let atlas_view = view(&atlas_tex);

        let shadow_tex = texture(
            gpu,
            "menu scene shadow map",
            (SHADOW_SIZE, SHADOW_SIZE),
            SHADOW_FORMAT,
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let shadow_view = view(&shadow_tex);

        let make_group = |shadow: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("menu scene group0"),
                layout: &statics.frame_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&atlas_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&statics.atlas_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(shadow),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&statics.shadow_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&statics.noise_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&statics.noise_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::Sampler(&statics.linear_sampler),
                    },
                ],
            })
        };
        let frame_group = make_group(&shadow_view);
        // The shadow pass writes the shadow map, so it binds a stand-in depth texture there.
        let shadow_pass_group = make_group(&statics.dummy_depth);

        // --- geometry
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("menu scene vertices"),
            contents: bytemuck::cast_slice::<SceneVertex, u8>(&prepared.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let water_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("menu scene water vertices"),
            contents: bytemuck::cast_slice::<WaterVertex, u8>(&prepared.water),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            frame_group,
            shadow_pass_group,
            vertices,
            water_vertices,
            shadow_view,
            shadow_ready: false,
            _textures: [atlas_tex, shadow_tex],
        }
    }
}

impl Renderer {
    /// The scene's GPU side over fresh [`Statics`]; the [`Resident`] part is built on the first
    /// frame drawn.
    pub(super) fn new(gpu: &Gpu, prepared: &'static PreparedScene) -> Self {
        let statics = Rc::new(Statics::new(gpu));
        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("menu scene frame uniform"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- the sun's orthographic view over the whole voxel box, fixed for the scene's life.
        let sun = Vec3::from(SUN).normalize();
        let (lo, hi) = (
            Vec3::new(VOX_MIN[0] as f32, VOX_MIN[1] as f32, VOX_MIN[2] as f32),
            Vec3::new(
                VOX_MAX[0] as f32 + 1.0,
                VOX_MAX[1] as f32 + 1.0,
                VOX_MAX[2] as f32 + 1.0,
            ),
        );
        let centre = (lo + hi) * 0.5;
        let light_view =
            glam::camera::rh::view::look_at_mat4(centre + sun * 200.0, centre, Vec3::Y);
        let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for i in 0..8 {
            let c = Vec3::new(
                if i & 1 == 0 { lo.x } else { hi.x },
                if i & 2 == 0 { lo.y } else { hi.y },
                if i & 4 == 0 { lo.z } else { hi.z },
            );
            let p = light_view.transform_point3(c);
            mn = mn.min(p);
            mx = mx.max(p);
        }
        let ortho = glam::camera::rh::proj::directx::orthographic(
            mn.x,
            mx.x,
            mn.y,
            mx.y,
            -mx.z - 1.0,
            -mn.z + 1.0,
        );
        let shadow_mat = ortho * light_view;

        Self {
            prepared,
            statics,
            time: 0.0,
            frame: 0,
            vhs_strength: 1.0,
            look: Look::default(),
            uniform,
            shadow_mat,
            resident: None,
            output: None,
            targets: None,
            finish_group: None,
            clouds: VolumetricClouds::new(),
            cloud_group: None,
            cloud_fade: 0.0,
            cloud_clock: 0.0,
            clouds_enabled: true,
        }
    }

    /// Switch the volumetric clouds; off gives everything of theirs back.
    pub(super) fn set_volumetric_clouds(&mut self, enabled: bool) {
        self.clouds_enabled = enabled;
        if !enabled {
            self.clouds.release();
            self.cloud_group = None;
            self.cloud_fade = 0.0;
        }
    }

    /// Block until the volumetric clouds are built (at most `timeout`),
    /// then show them at full strength without the cross-fade (offline renders, tests).
    pub(super) fn wait_for_clouds(&mut self, gpu: &Gpu, timeout: std::time::Duration) -> bool {
        if !self.clouds_enabled {
            return false;
        }
        self.clouds.request(gpu, CloudTarget::Offscreen);
        let ready = self.clouds.wait_ready(timeout);
        if ready {
            self.cloud_fade = 1.0;
        }
        ready
    }

    /// Free the big, rebuildable GPU memory: the screen-sized targets, the shadow map, the
    /// geometry and the atlas (about 200 MB at 1080p), and the clouds' noise and targets.
    /// Everything is rebuilt, and the shadow map redrawn, on the next frame drawn. The shader,
    /// pipelines and other small objects stay.
    pub(super) fn release(&mut self) {
        self.resident = None;
        self.targets = None;
        self.finish_group = None;
        self.clouds.release_memory();
        self.cloud_group = None;
    }

    /// Trace this frame's volumetric clouds for the sky (into their own
    /// small target, before the scene pass samples it). Returns the uniform's `clouds` row: zeros
    /// until they are ready, so the sky keeps its painted clouds meanwhile.
    fn trace_clouds(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        size: (u32, u32),
    ) -> [f32; 4] {
        if !self.clouds_enabled {
            return [0.0; 4];
        }
        self.clouds.request(gpu, CloudTarget::Offscreen);
        if self.time < self.cloud_clock - 1.0e-3 {
            // The scene's clock went back (a new visit): last frames' clouds no longer match.
            self.clouds.reset_history();
        }
        let shot = camera::shot(self.time);
        let aspect = size.0 as f32 / size.1.max(1) as f32;
        let view =
            glam::camera::rh::view::look_at_mat4(Vec3::ZERO, shot.target - shot.eye, Vec3::Y);
        let proj =
            glam::camera::rh::proj::directx::perspective_infinite_reverse(shot.fov_y, aspect, NEAR);
        let frame = CloudFrame {
            view_proj: proj * view,
            eye: shot.eye.as_dvec3(),
            // Reverse-Z with an infinite far plane: the near plane is NDC z 1.
            ray_z: (1.0, 0.5),
            frame_size: size,
            layer: CloudLayer::title(),
            look: title_cloud_look(self.time, &self.look),
            quality: CloudQuality::Medium,
            scene_depth: None,
            history_weight: 0.08,
        };
        let Some(info) = self.clouds.trace(gpu, enc, &frame) else {
            return [0.0; 4];
        };
        if self
            .cloud_group
            .as_ref()
            .is_none_or(|(generation, _)| *generation != info.generation)
        {
            let Some((color, _)) = self.clouds.out_views() else {
                return [0.0; 4];
            };
            let group = self.statics.input_group(gpu, Some(color), None, None, None);
            self.cloud_group = Some((info.generation, group));
        }
        let step = (self.time - self.cloud_clock).clamp(0.0, 0.1) as f32;
        self.cloud_clock = self.time;
        self.cloud_fade = (self.cloud_fade + step / 0.6).min(1.0);
        [
            self.cloud_fade,
            1.0 / (info.trace_size.0 * info.divisor) as f32,
            1.0 / (info.trace_size.1 * info.divisor) as f32,
            0.0,
        ]
    }

    /// Whether [`Renderer::release`] left the big memory freed (tests).
    pub(super) fn is_released(&self) -> bool {
        self.resident.is_none() && self.targets.is_none()
    }

    fn ensure_resident(&mut self, gpu: &Gpu) {
        if self.resident.is_none() {
            self.resident = Some(Resident::new(
                gpu,
                &self.statics,
                &self.uniform,
                self.prepared,
            ));
        }
    }
}

impl Statics {
    #[allow(clippy::too_many_lines)] // A table of pipelines.
    fn pipelines(
        gpu: &Gpu,
        module: &wgpu::ShaderModule,
        layouts: &[wgpu::PipelineLayout; 3],
    ) -> Pipelines {
        let device = &gpu.device;
        struct Desc<'a> {
            label: &'a str,
            layout: usize,
            vs: &'a str,
            fs: Option<&'a str>,
            buffers: Vec<Option<wgpu::VertexBufferLayout<'a>>>,
            format: Option<wgpu::TextureFormat>,
            blend: Option<wgpu::BlendState>,
            samples: u32,
            depth: Option<wgpu::DepthStencilState>,
            cull: Option<wgpu::Face>,
            a2c: bool,
        }
        let build = |d: Desc<'_>| {
            let targets = [d.format.map(|format| wgpu::ColorTargetState {
                format,
                blend: d.blend,
                write_mask: wgpu::ColorWrites::ALL,
            })];
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(d.label),
                layout: Some(&layouts[d.layout]),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some(d.vs),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &d.buffers,
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: d.cull,
                    ..wgpu::PrimitiveState::default()
                },
                depth_stencil: d.depth,
                multisample: wgpu::MultisampleState {
                    count: d.samples,
                    mask: !0,
                    alpha_to_coverage_enabled: d.a2c,
                },
                fragment: d.fs.map(|fs| wgpu::FragmentState {
                    module,
                    entry_point: Some(fs),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: if d.format.is_some() {
                        &targets[..]
                    } else {
                        &[]
                    },
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let scene_depth = |write: bool, compare: wgpu::CompareFunction| wgpu::DepthStencilState {
            format: DEPTH,
            depth_write_enabled: Some(write),
            depth_compare: Some(compare),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        let shadow_depth = wgpu::DepthStencilState {
            format: SHADOW_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 2.0,
                clamp: 0.0,
            },
        };
        let premultiplied = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let scene_buffers = || vec![Some(SceneVertex::layout())];
        let water_buffers = || vec![Some(WaterVertex::layout())];
        let post = |label: &'static str, fs: &'static str, format, blend| Desc {
            label,
            layout: 1,
            vs: "vs_fullscreen",
            fs: Some(fs),
            buffers: Vec::new(),
            format: Some(format),
            blend,
            samples: 1,
            depth: None,
            cull: None,
            a2c: false,
        };
        Pipelines {
            shadow_opaque: build(Desc {
                label: "menu scene shadow opaque",
                layout: 0,
                vs: "vs_shadow",
                fs: None,
                buffers: scene_buffers(),
                format: None,
                blend: None,
                samples: 1,
                depth: Some(shadow_depth.clone()),
                cull: None,
                a2c: false,
            }),
            shadow_cutout: build(Desc {
                label: "menu scene shadow cutout",
                layout: 0,
                vs: "vs_shadow",
                fs: Some("fs_shadow_cutout"),
                buffers: scene_buffers(),
                format: None,
                blend: None,
                samples: 1,
                depth: Some(shadow_depth),
                cull: None,
                a2c: false,
            }),
            opaque: build(Desc {
                label: "menu scene opaque",
                layout: 0,
                vs: "vs_scene",
                fs: Some("fs_opaque"),
                buffers: scene_buffers(),
                format: Some(HDR),
                blend: None,
                samples: MSAA,
                depth: Some(scene_depth(true, wgpu::CompareFunction::Greater)),
                cull: Some(wgpu::Face::Back),
                a2c: false,
            }),
            cutout: build(Desc {
                label: "menu scene cutout",
                layout: 0,
                vs: "vs_scene",
                fs: Some("fs_cutout"),
                buffers: scene_buffers(),
                format: Some(HDR),
                blend: None,
                samples: MSAA,
                depth: Some(scene_depth(true, wgpu::CompareFunction::Greater)),
                cull: Some(wgpu::Face::Back),
                a2c: true,
            }),
            sky: build(Desc {
                label: "menu scene sky",
                // Group 1 holds the volumetric clouds' output.
                layout: 1,
                vs: "vs_fullscreen",
                fs: Some("fs_sky"),
                buffers: Vec::new(),
                format: Some(HDR),
                blend: None,
                samples: MSAA,
                depth: Some(scene_depth(false, wgpu::CompareFunction::GreaterEqual)),
                cull: None,
                a2c: false,
            }),
            glass: build(Desc {
                label: "menu scene glass",
                layout: 0,
                vs: "vs_scene",
                fs: Some("fs_glass"),
                buffers: scene_buffers(),
                format: Some(HDR),
                blend: Some(premultiplied),
                samples: MSAA,
                // Writes depth: the glass is the house's skin, so haze and light shafts must
                // stop at it rather than run on into the sky behind.
                depth: Some(scene_depth(true, wgpu::CompareFunction::Greater)),
                cull: Some(wgpu::Face::Back),
                a2c: false,
            }),
            depth: build(Desc {
                layout: 2,
                ..post("menu scene planar depth", "fs_depth", PLANAR, None)
            }),
            water: build(Desc {
                label: "menu scene water",
                layout: 1,
                vs: "vs_water",
                fs: Some("fs_water"),
                buffers: water_buffers(),
                format: Some(HDR),
                blend: Some(premultiplied),
                samples: 1,
                depth: None,
                cull: Some(wgpu::Face::Back),
                a2c: false,
            }),
            mist: build(Desc {
                label: "menu scene mist",
                layout: 1,
                vs: "vs_water",
                fs: Some("fs_mist"),
                buffers: water_buffers(),
                format: Some(HDR),
                blend: Some(additive),
                samples: 1,
                depth: None,
                cull: None,
                a2c: false,
            }),
            bloom_first: build(post("menu scene bloom first", "fs_bloom_first", HDR, None)),
            bloom_down: build(post("menu scene bloom down", "fs_bloom_down", HDR, None)),
            bloom_up: build(post(
                "menu scene bloom up",
                "fs_bloom_up",
                HDR,
                Some(additive),
            )),
            shafts: build(post("menu scene shafts", "fs_shafts", SHAFTS, None)),
        }
    }

    /// The frame-writing pipelines for `format` at `gpu`'s sample count, built on first use.
    fn output_pipelines(&self, gpu: &Gpu, format: wgpu::TextureFormat) -> Rc<OutputPipelines> {
        let key = (format, gpu.sample_count());
        if let Some(found) = self.output.borrow().iter().find(|o| o.key == key) {
            return Rc::clone(found);
        }
        let build = |label: &str, fs: &str| {
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&self.layouts[1]),
                    vertex: wgpu::VertexState {
                        module: &self.module,
                        entry_point: Some("vs_fullscreen"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: key.1,
                        ..wgpu::MultisampleState::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &self.module,
                        entry_point: Some(fs),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        };
        let built = Rc::new(OutputPipelines {
            key,
            composite: build("menu scene composite", "fs_composite"),
            finish: build("menu scene finish", "fs_finish"),
        });
        self.output.borrow_mut().push(Rc::clone(&built));
        built
    }

    fn input_group(
        &self,
        gpu: &Gpu,
        a: Option<&wgpu::TextureView>,
        b: Option<&wgpu::TextureView>,
        c: Option<&wgpu::TextureView>,
        d: Option<&wgpu::TextureView>,
    ) -> wgpu::BindGroup {
        fn entry(binding: u32, v: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::TextureView(v),
            }
        }
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("menu scene group1"),
            layout: &self.input_layout,
            entries: &[
                entry(0, a.unwrap_or(&self.dummy_hdr)),
                entry(1, b.unwrap_or(&self.dummy_planar)),
                entry(2, c.unwrap_or(&self.dummy_hdr)),
                entry(3, d.unwrap_or(&self.dummy_hdr)),
            ],
        })
    }
}

impl Renderer {
    /// Keep this visit's handle on the frame-writing pipelines current.
    fn output_pipelines(&mut self, gpu: &Gpu, format: wgpu::TextureFormat) {
        let key = (format, gpu.sample_count());
        if self.output.as_ref().is_none_or(|o| o.key != key) {
            self.output = Some(self.statics.output_pipelines(gpu, format));
        }
    }

    fn input_group(
        &self,
        gpu: &Gpu,
        a: Option<&wgpu::TextureView>,
        b: Option<&wgpu::TextureView>,
        c: Option<&wgpu::TextureView>,
        d: Option<&wgpu::TextureView>,
    ) -> wgpu::BindGroup {
        self.statics.input_group(gpu, a, b, c, d)
    }

    fn ensure_targets(&mut self, gpu: &Gpu, size: (u32, u32)) {
        if self.targets.as_ref().is_some_and(|t| t.size == size) {
            return;
        }
        self.targets = None;
        let u = wgpu::TextureUsages::RENDER_ATTACHMENT;
        let sampled = u | wgpu::TextureUsages::TEXTURE_BINDING;
        let msaa_color = view(&texture(gpu, "menu scene msaa colour", size, HDR, MSAA, u));
        let msaa_depth = view(&texture(
            gpu,
            "menu scene msaa depth",
            size,
            DEPTH,
            MSAA,
            sampled,
        ));
        let scene = texture(
            gpu,
            "menu scene hdr",
            size,
            HDR,
            1,
            sampled | wgpu::TextureUsages::COPY_SRC,
        );
        let water = texture(
            gpu,
            "menu scene hdr + water",
            size,
            HDR,
            1,
            sampled | wgpu::TextureUsages::COPY_DST,
        );
        let planar = view(&texture(
            gpu,
            "menu scene planar depth",
            size,
            PLANAR,
            1,
            sampled,
        ));
        let mut bloom = Vec::new();
        let mut s = size;
        for _ in 0..BLOOM_LEVELS {
            s = ((s.0 / 2).max(1), (s.1 / 2).max(1));
            bloom.push((
                view(&texture(gpu, "menu scene bloom", s, HDR, 1, sampled)),
                s,
            ));
        }
        let shafts = view(&texture(
            gpu,
            "menu scene shafts",
            ((size.0 / 4).max(1), (size.1 / 4).max(1)),
            SHAFTS,
            1,
            sampled,
        ));
        let (scene_view, water_view) = (view(&scene), view(&water));
        let g_water = self.input_group(gpu, Some(&scene_view), Some(&planar), None, None);
        let g_bloom_first = self.input_group(gpu, Some(&water_view), None, None, None);
        let g_bloom_down = (1..BLOOM_LEVELS)
            .map(|i| self.input_group(gpu, Some(&bloom[i - 1].0), None, None, None))
            .collect();
        let g_bloom_up = (0..BLOOM_LEVELS - 1)
            .map(|i| self.input_group(gpu, Some(&bloom[i + 1].0), None, None, None))
            .collect();
        let g_shafts = self.input_group(gpu, None, Some(&planar), None, None);
        let g_composite = self.input_group(
            gpu,
            Some(&water_view),
            Some(&planar),
            Some(&bloom[0].0),
            Some(&shafts),
        );
        let g_none = self.input_group(gpu, None, None, None, None);
        let g_ms = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("menu scene group2"),
            layout: &self.statics.ms_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&msaa_depth),
            }],
        });
        self.targets = Some(Targets {
            size,
            msaa_color,
            msaa_depth,
            scene,
            scene_view,
            water,
            water_view,
            planar,
            bloom,
            shafts,
            g_water,
            g_bloom_first,
            g_bloom_down,
            g_bloom_up,
            g_shafts,
            g_composite,
            g_ms,
            g_none,
        });
    }

    pub(super) fn update(&mut self, time_s: f64, _width: u32, _height: u32) {
        self.time = time_s.max(0.0);
    }

    fn write_uniform(&mut self, gpu: &Gpu, size: (u32, u32), clouds: [f32; 4]) {
        let shot = camera::shot(self.time);
        let aspect = size.0 as f32 / size.1.max(1) as f32;
        let view = glam::camera::rh::view::look_at_mat4(shot.eye, shot.target, Vec3::Y);
        let proj =
            glam::camera::rh::proj::directx::perspective_infinite_reverse(shot.fov_y, aspect, NEAR);
        let view_proj = proj * view;
        let fwd = (shot.target - shot.eye).normalize();
        let right = fwd.cross(Vec3::Y).normalize();
        let up = right.cross(fwd);
        let tan_y = (shot.fov_y * 0.5).tan();
        let sun = Vec3::from(SUN).normalize();
        let intro = camera::ease(self.time / camera::INTRO_SECONDS * 0.85);
        let (lo, hi) = self.prepared.interior;
        self.frame = self.frame.wrapping_add(1);
        let u = FrameUniform {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            shadow_mat: self.shadow_mat.to_cols_array_2d(),
            cam_pos: [shot.eye.x, shot.eye.y, shot.eye.z, self.time as f32],
            cam_fwd: [fwd.x, fwd.y, fwd.z, NEAR],
            cam_right: [right.x, right.y, right.z, tan_y * aspect],
            cam_up: [up.x, up.y, up.z, tan_y],
            sun_dir: [sun.x, sun.y, sun.z, intro],
            sun_color: [
                self.look.sun[0],
                self.look.sun[1],
                self.look.sun[2],
                self.look.exposure,
            ],
            screen: [
                size.0 as f32,
                size.1 as f32,
                1.0 / size.0.max(1) as f32,
                1.0 / size.1.max(1) as f32,
            ],
            box_min: [lo[0], lo[1], lo[2], self.look.haze],
            box_max: [hi[0], hi[1], hi[2], self.look.shafts],
            params: [
                1.0 / SHADOW_SIZE as f32,
                self.look.bloom,
                self.vhs_strength,
                (self.frame % 64) as f32,
            ],
            look: [
                self.look.haze_mix,
                self.look.glow,
                self.look.debug,
                self.look.scrim,
            ],
            clouds,
        };
        gpu.queue
            .write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
    }

    fn render_shadow(&mut self, enc: &mut wgpu::CommandEncoder) {
        let Some(res) = self.resident.as_mut() else {
            return;
        };
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("menu scene shadow map"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &res.shadow_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let r = self.prepared.ranges;
        pass.set_bind_group(0, &res.shadow_pass_group, &[]);
        pass.set_vertex_buffer(0, res.vertices.slice(..));
        pass.set_index_buffer(
            self.statics.quad_indices.slice(..),
            wgpu::IndexFormat::Uint16,
        );
        pass.set_pipeline(&self.statics.pipes.shadow_opaque);
        draw_quads(&mut pass, r.opaque);
        pass.set_pipeline(&self.statics.pipes.shadow_cutout);
        draw_quads(&mut pass, r.cutout);
        drop(pass);
        res.shadow_ready = true;
    }

    fn fullscreen(
        enc: &mut wgpu::CommandEncoder,
        label: &str,
        target: &wgpu::TextureView,
        load: wgpu::LoadOp<wgpu::Color>,
        pipeline: &wgpu::RenderPipeline,
        groups: &[&wgpu::BindGroup],
    ) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        for (i, g) in groups.iter().enumerate() {
            pass.set_bind_group(i as u32, *g, &[]);
        }
        pass.draw(0..3, 0..1);
    }

    #[allow(clippy::too_many_lines)] // The frame, pass by pass.
    pub(super) fn record_background(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        frame_view: &wgpu::TextureView,
    ) {
        let size = internal_size(gpu.config.width.max(1), gpu.config.height.max(1));
        self.ensure_resident(gpu);
        self.ensure_targets(gpu, size);
        self.output_pipelines(gpu, gpu.config.format);
        // The volumetric clouds first: the scene pass's sky samples them.
        let clouds = self.trace_clouds(gpu, enc, size);
        self.write_uniform(gpu, size, clouds);
        if self.resident.as_ref().is_some_and(|res| !res.shadow_ready) {
            self.render_shadow(enc);
        }
        let res = self.resident.as_ref().expect("ensured above");
        let t = self.targets.as_ref().expect("ensured above");
        let p = &self.statics.pipes;
        let indices = self.statics.quad_indices.slice(..);
        let r = self.prepared.ranges;

        // 1. The scene, multisampled, resolved into `scene`.
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("menu scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &t.msaa_color,
                    depth_slice: None,
                    resolve_target: Some(&t.scene_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &t.msaa_depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &res.frame_group, &[]);
            pass.set_vertex_buffer(0, res.vertices.slice(..));
            pass.set_index_buffer(indices, wgpu::IndexFormat::Uint16);
            pass.set_pipeline(&p.opaque);
            draw_quads(&mut pass, r.opaque);
            pass.set_pipeline(&p.cutout);
            draw_quads(&mut pass, r.cutout);
            // The clouds' output, or nothing while they are not ready.
            let cloud_input = match &self.cloud_group {
                Some((_, group)) if clouds[0] > 0.0 => group,
                _ => &t.g_none,
            };
            pass.set_bind_group(1, cloud_input, &[]);
            pass.set_pipeline(&p.sky);
            pass.draw(0..3, 0..1);
            pass.set_pipeline(&p.glass);
            draw_quads(&mut pass, r.glass);
        }

        // 2. Planar depth.
        Self::fullscreen(
            enc,
            "menu scene planar depth",
            &t.planar,
            wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            &p.depth,
            &[&res.frame_group, &t.g_none, &t.g_ms],
        );

        // 3. Water over a copy of the scene.
        enc.copy_texture_to_texture(
            t.scene.as_image_copy(),
            t.water.as_image_copy(),
            wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
        );
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("menu scene water"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &t.water_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &res.frame_group, &[]);
            pass.set_bind_group(1, &t.g_water, &[]);
            pass.set_vertex_buffer(0, res.water_vertices.slice(..));
            pass.set_index_buffer(indices, wgpu::IndexFormat::Uint16);
            pass.set_pipeline(&p.water);
            draw_quads(&mut pass, (0, self.prepared.mist_first_quad));
            pass.set_pipeline(&p.mist);
            draw_quads(
                &mut pass,
                (self.prepared.mist_first_quad, self.prepared.water_quads),
            );
        }

        // 4. Bloom.
        let clear = wgpu::LoadOp::Clear(wgpu::Color::BLACK);
        Self::fullscreen(
            enc,
            "menu scene bloom",
            &t.bloom[0].0,
            clear,
            &p.bloom_first,
            &[&res.frame_group, &t.g_bloom_first],
        );
        for i in 1..BLOOM_LEVELS {
            Self::fullscreen(
                enc,
                "menu scene bloom",
                &t.bloom[i].0,
                clear,
                &p.bloom_down,
                &[&res.frame_group, &t.g_bloom_down[i - 1]],
            );
        }
        for i in (0..BLOOM_LEVELS - 1).rev() {
            Self::fullscreen(
                enc,
                "menu scene bloom",
                &t.bloom[i].0,
                wgpu::LoadOp::Load,
                &p.bloom_up,
                &[&res.frame_group, &t.g_bloom_up[i]],
            );
        }

        // 5. Light shafts.
        Self::fullscreen(
            enc,
            "menu scene shafts",
            &t.shafts,
            clear,
            &p.shafts,
            &[&res.frame_group, &t.g_shafts],
        );

        // 6. Composite into the game's frame.
        let out = self.output.as_ref().expect("built above");
        let mut pass = gpu.begin_pass(enc, frame_view, "menu/background", DepthAttachment::None);
        pass.set_pipeline(&out.composite);
        pass.set_bind_group(0, &res.frame_group, &[]);
        pass.set_bind_group(1, &t.g_composite, &[]);
        pass.draw(0..3, 0..1);
    }

    pub(super) fn record_finish(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        frame_view: &wgpu::TextureView,
        scene: &SceneTextures<'_>,
    ) {
        // The finish only reads the frame and the uniform; without the background's resident
        // part (it was released and nothing has drawn since) there is nothing to finish.
        if self.resident.is_none() {
            return;
        }
        self.output_pipelines(gpu, scene.format);
        if self
            .finish_group
            .as_ref()
            .is_none_or(|(snapshot, _)| snapshot != scene.color)
        {
            let group = self.input_group(gpu, Some(scene.color), None, None, None);
            self.finish_group = Some((scene.color.clone(), group));
        }
        let (Some(out), Some(res), Some((_, group))) = (
            self.output.as_ref(),
            self.resident.as_ref(),
            self.finish_group.as_ref(),
        ) else {
            return;
        };
        let mut pass = gpu.begin_pass(enc, frame_view, "menu/finish", DepthAttachment::None);
        pass.set_pipeline(&out.finish);
        pass.set_bind_group(0, &res.frame_group, &[]);
        pass.set_bind_group(1, group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::internal_size;

    #[test]
    fn the_scene_never_renders_more_than_1080p_worth_of_pixels() {
        assert_eq!(internal_size(1920, 1080), (1920, 1080));
        assert_eq!(internal_size(1280, 720), (1280, 720));
        let (w, h) = internal_size(3840, 2160);
        assert!((1919..=1920).contains(&w) && (1079..=1080).contains(&h));
        for (fw, fh) in [(2560, 1440), (3440, 1440), (2560, 1600)] {
            let (w, h) = internal_size(fw, fh);
            assert!(f64::from(w) * f64::from(h) <= 1920.0 * 1080.0, "{fw}x{fh}");
            let aspect = |a: u32, b: u32| f64::from(a) / f64::from(b);
            assert!((aspect(w, h) - aspect(fw, fh)).abs() < 0.01, "{fw}x{fh}");
        }
    }
}
