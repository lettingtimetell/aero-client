//! The glass scene: a Minecraft plant swaying gently behind a pane
//! of hammered, frosted, wired glass, with the faint reflection of a spinning ceiling fan in the
//! shiny spot where a ceiling light catches the glass. The user's reference is
//!
//! # How it is drawn
//!
//! 0. **Once per window size, the glass's static fields** (`fs_bake`, two full-size textures): the
//!    hammered front face's slope (fine pebbles, a medium ripple and the sheet's slow waviness),
//!    the condensation fog (thick at the left edge, in patches elsewhere), how much the etched
//!    back face scatters, and the wire mesh's mask. None of it ever moves, so the per-frame pass
//!    only reads it (this took the scene from 0.68 to 0.24 ms a frame at 1080p).
//! 1. **The plants**, as flat sprites with the real 1.8.9 textures read from the jar (tall grass,
//!    ferns, double plants, four kinds of leaves as rounded clusters, a flower or two, tinted with
//!    the jar's own grass and foliage colormaps), into three depth layers at half the window's
//!    size: near (pressed against the glass), mid (a bush a hand's width behind it) and far
//!    (foliage further back). Each sprite sways from its root, so the grass bends and the leaves
//!    flutter.
//! 2. **A blur pyramid** of each layer (five half-size steps, all three layers per pass), sampled
//!    through a cubic B-spline so even the coarsest level is smooth: the glass can pick any amount
//!    of blur per pixel.
//! 3. **The glass**, one full-screen pass. Per pixel: the pebbles' slope refracts each layer by an
//!    amount that grows with the layer's distance behind the glass, and the frost blurs it by an
//!    amount that grows the same way (so leaves touching the glass stay readable and the far ones
//!    melt into light); the layers are composited back to front over a daylight backdrop, with
//!    the frost's haze; the wire mesh, set in the middle of the pane, stays crisp but wobbles
//!    slightly; the fog veils everything behind it, wires included; and the glossy front face
//!    reflects the room: a lit ceiling whose bright heart breaks into glitter on the pebbles, and
//!    a ceiling fan turning in front of it, whose blades sweep faintly through the shiny spot
//!    (motion-blurred analytically over a fixed shutter, so it never strobes at any frame rate),
//!    plus a soft glow around the spot.
//!
//! No VHS, no grain beyond a sub-LSB dither against banding.

use bytemuck::{Pod, Zeroable};

use super::{JarTextures, Scene, SceneFrame, SceneInit, SCENE_FORMAT};
use crate::Gpu;

/// The vanilla textures the scene reads from the jar.
pub(super) const JAR_TEXTURES: &[&str] = &[
    "textures/blocks/tallgrass.png",
    "textures/blocks/fern.png",
    "textures/blocks/double_plant_grass_bottom.png",
    "textures/blocks/double_plant_grass_top.png",
    "textures/blocks/double_plant_fern_bottom.png",
    "textures/blocks/double_plant_fern_top.png",
    "textures/blocks/leaves_oak.png",
    "textures/blocks/leaves_jungle.png",
    "textures/blocks/leaves_big_oak.png",
    "textures/blocks/leaves_birch.png",
    "textures/blocks/flower_rose.png",
    "textures/blocks/flower_dandelion.png",
    "textures/blocks/flower_oxeye_daisy.png",
    "textures/colormap/grass.png",
    "textures/colormap/foliage.png",
];

/// The sprite textures, in texture-array order (the first 13 entries of [`JAR_TEXTURES`]).
const SPRITE_TEXTURES: usize = 13;

/// Indices into the sprite texture array.
mod tex {
    pub(super) const TALLGRASS: u32 = 0;
    pub(super) const FERN: u32 = 1;
    pub(super) const GRASS_BOTTOM: u32 = 2;
    pub(super) const GRASS_TOP: u32 = 3;
    pub(super) const FERN_BOTTOM: u32 = 4;
    pub(super) const FERN_TOP: u32 = 5;
    pub(super) const LEAVES_OAK: u32 = 6;
    pub(super) const LEAVES_JUNGLE: u32 = 7;
    pub(super) const LEAVES_DARK_OAK: u32 = 8;
    pub(super) const LEAVES_BIRCH: u32 = 9;
    pub(super) const POPPY: u32 = 10;
    pub(super) const DANDELION: u32 = 11;
    pub(super) const OXEYE: u32 = 12;
}

/// The plant layers, nearest first.
const LAYERS: u32 = 3;
/// Mip levels of each layer's blur pyramid (level 0 is the sharp half-size layer).
const MIPS: u32 = 6;
/// The plant layers' size as a fraction of the window's.
const LAYER_SCALE: f32 = 0.5;
/// The format of the plant layers (premultiplied linear colour).
const LAYER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The baked static fields: the slope and form (signed, fine), then fog, frost, the wire mask.
const BAKED_FORMATS: [wgpu::TextureFormat; 2] = [
    wgpu::TextureFormat::Rgba16Float,
    wgpu::TextureFormat::Rgba8Unorm,
];

/// The glass's tuning knobs. [`GlassLook::default`] is the tuned look; the render harness can
/// override any of them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlassLook {
    /// Overall exposure before the tone curve.
    pub exposure: f32,
    /// How frosted the back face is (blur and milkiness), around 1.
    pub frost: f32,
    /// How strongly the dimples refract, around 1.
    pub refraction: f32,
    /// The wire mesh's visibility, `0..=1`.
    pub wire: f32,
    /// The glitter of the ceiling light on the dimples.
    pub sparkle: f32,
    /// The fan's reflection: how dark its blades make the shiny spot, `0..=1`.
    pub fan: f32,
    /// The soft glow around the shiny spot.
    pub glow: f32,
    /// The fan's speed, turns per second.
    pub fan_speed: f32,
    /// Pebbles of the hammered face per screen height.
    pub dimples: f32,
    /// The pebbles' steepness (slope).
    pub dimple_depth: f32,
    /// The condensation fog on the front face (thick at the left edge, in patches elsewhere).
    pub fog: f32,
    /// The plants' sway, `0..=1` and beyond.
    pub sway: f32,
    /// Where the ceiling light's mirror image sits, in screen fractions (`x`, `y` from the top).
    pub spot: [f32; 2],
    /// How large the shiny spot is, in screen heights.
    pub spot_size: f32,
    /// Debug view: 0 off, 1 the plant layers unblurred, 2 the slope field, 3 the frost map,
    /// 4 the reflection alone.
    pub debug: f32,
}

impl Default for GlassLook {
    fn default() -> Self {
        GlassLook {
            exposure: 0.85,
            frost: 1.0,
            refraction: 1.0,
            wire: 1.0,
            sparkle: 1.0,
            fan: 1.0,
            glow: 1.4,
            fan_speed: 0.9,
            dimples: 95.0,
            dimple_depth: 1.0,
            fog: 1.0,
            sway: 1.0,
            spot: [0.48, 0.15],
            spot_size: 0.32,
            debug: 0.0,
        }
    }
}

/// The glass pass's uniform (`Params` in `glass.wgsl`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Uniform {
    /// time, aspect, width, height
    frame: [f32; 4],
    /// layer width, layer height, mips, seed fraction
    layer: [f32; 4],
    /// exposure, frost, refraction, wire
    look0: [f32; 4],
    /// sparkle, fan, glow, fan speed
    look1: [f32; 4],
    /// pebbles, pebble depth, fog, debug
    look2: [f32; 4],
    /// the logo rect, pixels
    logo: [f32; 4],
    /// spot x, spot y, spot size, sway
    spot: [f32; 4],
    /// loaded progress, opacity, unused, unused
    extra: [f32; 4],
}

/// One plant sprite (`Sprite` in `glass.wgsl`): 64 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Sprite {
    /// Root (bottom-anchored) or centre, in screen heights from the screen's centre (y down);
    /// width and height in screen heights.
    pos_size: [f32; 4],
    /// rotation (radians), sway amplitude (screen heights), sway phase, anchor (0 root, 1 centre)
    motion: [f32; 4],
    /// linear tint (rgb), texture index
    look: [f32; 4],
    /// mirror (0/1), brightness, backlight, unused
    extra: [f32; 4],
}

impl Sprite {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x4,
        1 => Float32x4,
        2 => Float32x4,
        3 => Float32x4,
    ];
}

/// The window-size-dependent part.
struct Targets {
    size: (u32, u32),
    layer_size: (u32, u32),
    /// One 2D view per (mip, layer), for rendering: index `mip * LAYERS + layer`.
    attach: Vec<wgpu::TextureView>,
    /// Bind groups reading the array at one mip (for the downsample into the next).
    down_groups: Vec<wgpu::BindGroup>,
    /// The glass pass's bind group (the whole array, every mip, and the baked fields).
    glass_group: wgpu::BindGroup,
    /// The baked static fields (`fs_bake`): slope and form; fog, frost and the wire mask.
    baked_views: [wgpu::TextureView; 2],
    /// Whether the static fields have been baked at this size.
    baked: bool,
    _layers: wgpu::Texture,
    _baked: [wgpu::Texture; 2],
}

/// The glass scene.
pub(super) struct GlassScene {
    look: GlassLook,
    seed: u32,
    uniform: wgpu::Buffer,
    sprites: wgpu::Buffer,
    /// Instance ranges per layer, nearest first.
    ranges: [(u32, u32); LAYERS as usize],
    sprite_group: wgpu::BindGroup,
    sprite_pipeline: wgpu::RenderPipeline,
    down_pipeline: wgpu::RenderPipeline,
    bake_pipeline: wgpu::RenderPipeline,
    glass_pipeline: wgpu::RenderPipeline,
    down_layout: wgpu::BindGroupLayout,
    glass_layout: wgpu::BindGroupLayout,
    linear: wgpu::Sampler,
    targets: Option<Targets>,
    _plants: wgpu::Texture,
}

impl GlassScene {
    /// Build the scene with the compiled-in shader and the tuned look.
    pub(super) fn new(gpu: &Gpu, init: &SceneInit<'_>) -> GlassScene {
        GlassScene::with_shader(gpu, init, include_str!("glass.wgsl"), GlassLook::default())
    }

    /// Build the scene from `wgsl` (the render harness's edited copy of `glass.wgsl`) and `look`.
    pub(super) fn with_shader(
        gpu: &Gpu,
        init: &SceneInit<'_>,
        wgsl: &str,
        look: GlassLook,
    ) -> GlassScene {
        let device = &gpu.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("loading/glass"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("loading/glass uniform"),
            size: std::mem::size_of::<Uniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // The plants' textures, and the biome colours from the jar's colormaps.
        let (plants, plant_view) = upload_plants(gpu, init.textures);
        let colors = BiomeColors::from_jar(init.textures);
        let (sprite_list, ranges) = layout_plants(init.seed, &colors);
        let sprites = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("loading/glass sprites"),
            size: (std::mem::size_of::<Sprite>() * sprite_list.len().max(1)) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&sprites, 0, bytemuck::cast_slice(&sprite_list));

        let nearest = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("loading/glass nearest"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..wgpu::SamplerDescriptor::default()
        });
        let linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("loading/glass linear"),
            address_mode_u: wgpu::AddressMode::MirrorRepeat,
            address_mode_v: wgpu::AddressMode::MirrorRepeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..wgpu::SamplerDescriptor::default()
        });

        let uniform_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let array_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        };
        let sampler_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        // One layout shape for all three passes: uniform, a texture array, a sampler.
        let make_layout = |label| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[uniform_entry(0), array_entry(1), sampler_entry(2)],
            })
        };
        let sprite_layout = make_layout("loading/glass sprites");
        let down_layout = make_layout("loading/glass down");
        let baked_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let glass_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("loading/glass"),
            entries: &[
                uniform_entry(0),
                array_entry(1),
                sampler_entry(2),
                baked_entry(3),
                baked_entry(4),
            ],
        });
        let sprite_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("loading/glass sprites"),
            layout: &sprite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&plant_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&nearest),
                },
            ],
        });

        let pipeline_layout = |layout: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("loading/glass"),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            })
        };
        let premultiplied_over = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let sprite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("loading/glass sprites"),
            layout: Some(&pipeline_layout(&sprite_layout)),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_sprite"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Sprite>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &Sprite::ATTRIBUTES,
                })],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_sprite"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: LAYER_FORMAT,
                    blend: Some(premultiplied_over),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let layer_target = Some(wgpu::ColorTargetState {
            format: LAYER_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        });
        let down_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("loading/glass down"),
            layout: Some(&pipeline_layout(&down_layout)),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_fullscreen"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_down"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[layer_target.clone(), layer_target.clone(), layer_target],
            }),
            multiview_mask: None,
            cache: None,
        });
        let bake_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("loading/glass bake"),
            layout: Some(&pipeline_layout(&sprite_layout)),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_fullscreen"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_bake"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: BAKED_FORMATS[0],
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: BAKED_FORMATS[1],
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });
        let glass_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("loading/glass"),
            layout: Some(&pipeline_layout(&glass_layout)),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_fullscreen"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_glass"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        GlassScene {
            look,
            seed: init.seed,
            uniform,
            sprites,
            ranges,
            sprite_group,
            sprite_pipeline,
            down_pipeline,
            bake_pipeline,
            glass_pipeline,
            down_layout,
            glass_layout,
            linear,
            targets: None,
            _plants: plants,
        }
    }

    fn ensure_targets(&mut self, gpu: &Gpu, size: (u32, u32)) {
        if self.targets.as_ref().is_some_and(|t| t.size == size) {
            return;
        }
        self.targets = None;
        let layer_size = (
            ((size.0 as f32 * LAYER_SCALE).round() as u32).max(1 << (MIPS - 1)),
            ((size.1 as f32 * LAYER_SCALE).round() as u32).max(1 << (MIPS - 1)),
        );
        let layers = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("loading/glass layers"),
            size: wgpu::Extent3d {
                width: layer_size.0,
                height: layer_size.1,
                depth_or_array_layers: LAYERS,
            },
            mip_level_count: MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: LAYER_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mut attach = Vec::with_capacity((MIPS * LAYERS) as usize);
        for mip in 0..MIPS {
            for layer in 0..LAYERS {
                attach.push(layers.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("loading/glass layer"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..wgpu::TextureViewDescriptor::default()
                }));
            }
        }
        let group = |layout: &wgpu::BindGroupLayout, view: &wgpu::TextureView| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("loading/glass"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.linear),
                    },
                ],
            })
        };
        let mut down_groups = Vec::with_capacity((MIPS - 1) as usize);
        for mip in 0..MIPS - 1 {
            let view = layers.create_view(&wgpu::TextureViewDescriptor {
                label: Some("loading/glass down source"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..wgpu::TextureViewDescriptor::default()
            });
            down_groups.push(group(&self.down_layout, &view));
        }
        let all = layers.create_view(&wgpu::TextureViewDescriptor {
            label: Some("loading/glass layers"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..wgpu::TextureViewDescriptor::default()
        });
        let baked = BAKED_FORMATS.map(|format| {
            gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("loading/glass baked"),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        });
        let baked_views = [
            baked[0].create_view(&wgpu::TextureViewDescriptor::default()),
            baked[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];
        let glass_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("loading/glass"),
            layout: &self.glass_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&all),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.linear),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&baked_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&baked_views[1]),
                },
            ],
        });
        self.targets = Some(Targets {
            size,
            layer_size,
            attach,
            down_groups,
            glass_group,
            baked_views,
            baked: false,
            _layers: layers,
            _baked: baked,
        });
    }

    fn write_uniform(&self, gpu: &Gpu, frame: &SceneFrame, layer_size: (u32, u32)) {
        let look = &self.look;
        let (w, h) = (frame.width.max(1) as f32, frame.height.max(1) as f32);
        let uniform = Uniform {
            frame: [frame.time_s, w / h, w, h],
            layer: [
                layer_size.0 as f32,
                layer_size.1 as f32,
                MIPS as f32,
                (self.seed % 10_007) as f32 / 10_007.0,
            ],
            look0: [look.exposure, look.frost, look.refraction, look.wire],
            look1: [look.sparkle, look.fan, look.glow, look.fan_speed],
            look2: [look.dimples, look.dimple_depth, look.fog, look.debug],
            logo: frame.logo,
            spot: [look.spot[0], look.spot[1], look.spot_size, look.sway],
            extra: [frame.progress, frame.opacity, 0.0, 0.0],
        };
        gpu.queue
            .write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
    }
}

impl Scene for GlassScene {
    fn name(&self) -> &'static str {
        "glass"
    }

    fn render(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        frame: &SceneFrame,
    ) {
        let size = (frame.width.max(1), frame.height.max(1));
        self.ensure_targets(gpu, size);
        let layer_size = self.targets.as_ref().expect("ensured above").layer_size;
        self.write_uniform(gpu, frame, layer_size);
        // 0. Once per size: bake the glass's static fields (its noise never moves).
        let bake = self
            .targets
            .as_mut()
            .is_some_and(|t| !std::mem::replace(&mut t.baked, true));
        let t = self.targets.as_ref().expect("ensured above");
        if bake {
            let attachment = |view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })
            };
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("loading/glass bake"),
                color_attachments: &[attachment(&t.baked_views[0]), attachment(&t.baked_views[1])],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.bake_pipeline);
            pass.set_bind_group(0, &self.sprite_group, &[]);
            pass.draw(0..3, 0..1);
        }
        let clear = wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        };

        // 1. The plant layers.
        for (layer, &(first, count)) in self.ranges.iter().enumerate() {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("loading/glass plants"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &t.attach[layer],
                    depth_slice: None,
                    resolve_target: None,
                    ops: clear,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if count > 0 {
                pass.set_pipeline(&self.sprite_pipeline);
                pass.set_bind_group(0, &self.sprite_group, &[]);
                pass.set_vertex_buffer(0, self.sprites.slice(..));
                pass.draw(0..6, first..first + count);
            }
        }

        // 2. The blur pyramid: each pass reads one mip of all three layers, writes the next.
        for mip in 1..MIPS {
            let base = (mip * LAYERS) as usize;
            let attachment = |i: usize| {
                Some(wgpu::RenderPassColorAttachment {
                    view: &t.attach[base + i],
                    depth_slice: None,
                    resolve_target: None,
                    ops: clear,
                })
            };
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("loading/glass blur"),
                color_attachments: &[attachment(0), attachment(1), attachment(2)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.down_pipeline);
            pass.set_bind_group(0, &t.down_groups[(mip - 1) as usize], &[]);
            pass.draw(0..3, 0..1);
        }

        // 3. The glass.
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("loading/glass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.glass_pipeline);
        pass.set_bind_group(0, &t.glass_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// The plant layers and the baked fields (~41 MB at 1080p): rebuilt, and the fields baked
    /// again, on the next frame.
    fn release_gpu_memory(&mut self) {
        self.targets = None;
    }
}

/// The plant textures as one 16x16 texture array (sRGB, so sampling returns linear colour). A
/// texture the jar lacks becomes a soft leafy blob, so the scene still reads without the jar.
fn upload_plants(gpu: &Gpu, textures: &JarTextures) -> (wgpu::Texture, wgpu::TextureView) {
    const SIZE: u32 = 16;
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("loading/glass plants"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: SPRITE_TEXTURES as u32,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (index, path) in JAR_TEXTURES.iter().take(SPRITE_TEXTURES).enumerate() {
        let mut pixels = vec![0_u8; (SIZE * SIZE * 4) as usize];
        match textures.get(path) {
            Some(image) => {
                // Nearest-resample the first 16x16 frame (vanilla's are exactly 16x16).
                let frame = image.width.min(image.height).max(1);
                for y in 0..SIZE {
                    for x in 0..SIZE {
                        let texel = image.texel(x * frame / SIZE, y * frame / SIZE);
                        let i = ((y * SIZE + x) * 4) as usize;
                        pixels[i..i + 4].copy_from_slice(&texel);
                    }
                }
            }
            None => fallback_blob(&mut pixels, SIZE, index as u32),
        }
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: index as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SIZE * 4),
                rows_per_image: Some(SIZE),
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("loading/glass plants"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..wgpu::TextureViewDescriptor::default()
    });
    (texture, view)
}

/// A stand-in for a missing texture: a gray, ragged disc (tinted like a leaf by the sprite).
fn fallback_blob(pixels: &mut [u8], size: u32, salt: u32) {
    let mut rng = Rng::new(0xB10B_0000 ^ salt);
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 + 0.5 - size as f32 * 0.5;
            let dy = y as f32 + 0.5 - size as f32 * 0.5;
            let r = (dx * dx + dy * dy).sqrt() / (size as f32 * 0.5);
            let inside = r < 0.85 + rng.next_f32() * 0.2;
            let v = (150.0 + rng.next_f32() * 80.0) as u8;
            let i = ((y * size + x) * 4) as usize;
            pixels[i..i + 4].copy_from_slice(&[v, v, v, if inside { 255 } else { 0 }]);
        }
    }
}

/// The biome colours the gray textures are tinted with, read from the jar's colormaps the way
/// vanilla's `ColorizerGrass`/`ColorizerFoliage` do, at a lush forest (temperature 0.7,
/// rainfall 0.8). Linear RGB.
#[derive(Debug, Clone, Copy)]
struct BiomeColors {
    grass: [f32; 3],
    foliage: [f32; 3],
}

impl BiomeColors {
    fn from_jar(textures: &JarTextures) -> BiomeColors {
        // java: ColorizerGrass.getGrassColor(t, r): rainfall *= temperature;
        // i = (int)((1 - t) * 255), j = (int)((1 - r) * 255), buffer[j << 8 | i].
        let lookup = |path: &str, fallback: [u8; 3]| -> [f32; 3] {
            let (temperature, rainfall) = (0.7_f32, 0.8_f32);
            let rainfall = rainfall * temperature;
            let i = ((1.0 - temperature) * 255.0) as u32;
            let j = ((1.0 - rainfall) * 255.0) as u32;
            let texel = textures
                .get(path)
                .filter(|image| image.width >= 256 && image.height >= 256)
                .map_or(fallback, |image| {
                    let t = image.texel(i, j);
                    [t[0], t[1], t[2]]
                });
            texel.map(|c| srgb_to_linear(f32::from(c) / 255.0))
        };
        BiomeColors {
            grass: lookup("textures/colormap/grass.png", [0x79, 0xC0, 0x5A]),
            foliage: lookup("textures/colormap/foliage.png", [0x59, 0xAE, 0x30]),
        }
    }
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// A small deterministic generator for the layout (xorshift32).
struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9) ^ 0x6A09_E667 | 1)
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// `0..1`.
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[(self.next_u32() as usize) % items.len()]
    }
}

/// Lay out the plants for `seed`: a different arrangement every launch, the same composition.
/// Returns the sprites and each layer's `(first, count)`, nearest layer first.
#[allow(clippy::too_many_lines)] // One composition, layer by layer.
fn layout_plants(seed: u32, colors: &BiomeColors) -> (Vec<Sprite>, [(u32, u32); 3]) {
    let mut rng = Rng::new(seed);
    let mut layers: [Vec<Sprite>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let shade = |c: [f32; 3], k: f32| [c[0] * k, c[1] * k, c[2] * k];
    let leaf = |layer: &mut Vec<Sprite>,
                rng: &mut Rng,
                (x, y, size): (f32, f32, f32),
                brightness: f32,
                sway: f32| {
        push_leaf(layer, rng, colors, (x, y, size), brightness, sway);
    };

    // Far layer (index 2): foliage further back, thick on the right and along the bottom, thin
    // at the top left where the daylight comes through.
    for _ in 0..170 {
        let x = rng.range(-1.15, 1.15);
        let y = rng.range(-0.62, 0.62);
        // Sparser toward the upper left.
        let openness = ((-x * 0.75 - y * 1.1) + 0.3).clamp(0.0, 1.0);
        if rng.next_f32() < openness * 0.92 {
            continue;
        }
        let size = rng.range(0.09, 0.17);
        let brightness = rng.range(0.3, 0.68);
        leaf(&mut layers[2], &mut rng, (x, y, size), brightness, 0.0022);
    }
    // Mid layer (index 1): a leafy bush right of centre, reaching up and to the left.
    let bush = (rng.range(0.18, 0.36), rng.range(0.0, 0.12));
    for i in 0..76 {
        let a = rng.range(0.0, std::f32::consts::TAU);
        let r = rng.range(0.0, 1.0).sqrt();
        let (rx, ry) = (0.95, 0.62);
        let mut x = bush.0 + a.cos() * r * rx;
        let mut y = bush.1 + a.sin() * r * ry;
        if i % 6 == 0 {
            // A few sprigs reaching up and to the left.
            let t = rng.range(0.25, 1.0);
            x = bush.0 - t * 0.7 + rng.range(-0.07, 0.07);
            y = bush.1 - t * 0.45 + rng.range(-0.05, 0.05);
        }
        let size = rng.range(0.09, 0.15);
        // Brighter toward the top (light from above and behind).
        let brightness = (0.8 - y * 0.55).clamp(0.42, 1.15);
        leaf(&mut layers[1], &mut rng, (x, y, size), brightness, 0.0035);
    }
    // A few clusters further left, softened by the fog there.
    for _ in 0..16 {
        let x = rng.range(-0.95, -0.15);
        let y = rng.range(-0.42, 0.4);
        let size = rng.range(0.09, 0.15);
        let brightness = rng.range(0.5, 0.85);
        leaf(&mut layers[1], &mut rng, (x, y, size), brightness, 0.0035);
    }
    // Ferns at the mid layer's foot.
    for _ in 0..7 {
        let x = rng.range(-0.55, 1.0);
        let h = rng.range(0.2, 0.28);
        let double = rng.next_f32() < 0.5;
        push_plant(
            &mut layers[1],
            &mut rng,
            x,
            0.6,
            h,
            if double { tex::FERN_BOTTOM } else { tex::FERN },
            double.then_some(tex::FERN_TOP),
            shade(colors.grass, 0.62),
            0.005,
        );
    }

    // Near layer (index 0): tall grass and ferns rising from the bottom edge, pressed against
    // the glass (fewer toward the foggy left), a flower or two among them, and a few leaf
    // clusters of the bush touching the glass on the right.
    let count = 11;
    for i in 0..count {
        let x = -0.62 + 1.8 * (i as f32 + rng.range(0.05, 0.95)) / count as f32;
        if rng.next_f32() < 0.18 {
            continue;
        }
        let kind = rng.next_f32();
        let h = rng.range(0.11, 0.22);
        let tint = shade(colors.grass, rng.range(0.78, 1.0));
        let root = 0.56 + rng.range(0.0, 0.06);
        let (bottom, top, sway) = if kind < 0.45 {
            (tex::GRASS_BOTTOM, Some(tex::GRASS_TOP), 0.011)
        } else if kind < 0.72 {
            (tex::TALLGRASS, None, 0.009)
        } else {
            (tex::FERN_BOTTOM, Some(tex::FERN_TOP), 0.01)
        };
        push_plant(
            &mut layers[0],
            &mut rng,
            x,
            root,
            h,
            bottom,
            top,
            tint,
            sway,
        );
    }
    let flowers = [tex::POPPY, tex::DANDELION, tex::POPPY, tex::OXEYE];
    for _ in 0..2 {
        let x = rng.range(-0.2, 0.9);
        let kind = rng.pick(&flowers);
        let h = rng.range(0.14, 0.19);
        push_plant(
            &mut layers[0],
            &mut rng,
            x,
            0.545,
            h,
            kind,
            None,
            [0.85, 0.85, 0.85],
            0.012,
        );
    }
    for _ in 0..30 {
        // Denser toward the right edge.
        let x = 1.15 - rng.next_f32().powf(1.6) * 1.0;
        let y = rng.range(-0.56, 0.32);
        let size = rng.range(0.065, 0.12);
        let brightness = rng.range(0.85, 1.3) * (1.0 - 0.25 * (y + 0.2).max(0.0));
        leaf(&mut layers[0], &mut rng, (x, y, size), brightness, 0.005);
    }

    let mut sprites = Vec::new();
    let mut ranges = [(0, 0); 3];
    for (i, layer) in layers.into_iter().enumerate() {
        ranges[i] = (sprites.len() as u32, layer.len() as u32);
        sprites.extend(layer);
    }
    (sprites, ranges)
}

/// One leaf card (a whole leaves texture) centred at `(x, y)`, `size` screen heights square.
fn push_leaf(
    layer: &mut Vec<Sprite>,
    rng: &mut Rng,
    colors: &BiomeColors,
    (x, y, size): (f32, f32, f32),
    brightness: f32,
    sway: f32,
) {
    const KINDS: [u32; 6] = [
        tex::LEAVES_OAK,
        tex::LEAVES_OAK,
        tex::LEAVES_JUNGLE,
        tex::LEAVES_JUNGLE,
        tex::LEAVES_DARK_OAK,
        tex::LEAVES_BIRCH,
    ];
    let kind = rng.pick(&KINDS);
    let base = if kind == tex::LEAVES_BIRCH {
        // java: ColorizerFoliage.getFoliageColorBirch() = 0x80A755.
        [0x80_u8, 0xA7, 0x55].map(|c| srgb_to_linear(f32::from(c) / 255.0))
    } else {
        colors.foliage
    };
    // Variety: some leaves bluer (in shade), many yellower (the sun shining through them).
    let warm = if rng.next_f32() < 0.35 {
        rng.range(0.3, 0.75)
    } else {
        rng.range(-0.15, 0.2)
    };
    let tint = [
        base[0] * (1.0 + warm * 2.2),
        base[1] * (1.0 + warm * 0.45),
        base[2] * (1.0 - warm * 0.7).max(0.2),
    ];
    layer.push(Sprite {
        pos_size: [x, y, size, size],
        motion: [
            rng.range(-0.5, 0.5),
            sway * rng.range(0.6, 1.2),
            rng.range(0.0, std::f32::consts::TAU),
            1.0,
        ],
        look: [tint[0], tint[1], tint[2], kind as f32],
        extra: [
            if rng.next_f32() < 0.5 { 1.0 } else { 0.0 },
            brightness * rng.range(0.82, 1.12),
            rng.range(0.0, 1.0),
            0.0,
        ],
    });
}

/// One plant rooted at `(x, root)` (`root` is its bottom edge), `h` screen heights tall per
/// block; a double plant gets its top half above.
#[allow(clippy::too_many_arguments)] // A plant is all of these.
fn push_plant(
    layer: &mut Vec<Sprite>,
    rng: &mut Rng,
    x: f32,
    root: f32,
    h: f32,
    bottom: u32,
    top: Option<u32>,
    tint: [f32; 3],
    sway: f32,
) {
    let phase = rng.range(0.0, std::f32::consts::TAU);
    let mirror = if rng.next_f32() < 0.5 { 1.0 } else { 0.0 };
    let brightness = rng.range(0.85, 1.1);
    let lean = rng.range(-0.06, 0.06);
    let blocks = if top.is_some() { 2.0 } else { 1.0 };
    for (i, kind) in std::iter::once(bottom).chain(top).enumerate() {
        // Both halves sway as one stem: the shader bends by height above the root, so the top
        // half carries the root's position and its own offset up the stem.
        layer.push(Sprite {
            pos_size: [x, root - h * i as f32, h, h],
            motion: [lean, sway, phase, 0.0],
            look: [tint[0], tint[1], tint[2], kind as f32],
            extra: [mirror, brightness, i as f32 / blocks, blocks],
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layout_is_stable_per_seed_and_varies_between_seeds() {
        let colors = BiomeColors {
            grass: [0.2, 0.5, 0.1],
            foliage: [0.1, 0.4, 0.05],
        };
        let (a, ra) = layout_plants(7, &colors);
        let (b, rb) = layout_plants(7, &colors);
        assert_eq!(ra, rb);
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&a),
            bytemuck::cast_slice::<_, u8>(&b)
        );
        let (c, _) = layout_plants(8, &colors);
        assert_ne!(
            bytemuck::cast_slice::<_, u8>(&a),
            bytemuck::cast_slice::<_, u8>(&c)
        );
        // Every layer has plants, and the ranges tile the buffer.
        let mut next = 0;
        for (first, count) in ra {
            assert_eq!(first, next);
            assert!(count > 0);
            next += count;
        }
        assert_eq!(next as usize, a.len());
        // Every sprite names a real texture.
        for sprite in &a {
            assert!((sprite.look[3] as usize) < SPRITE_TEXTURES);
        }
    }

    #[test]
    fn the_jar_list_starts_with_the_sprite_textures() {
        assert!(JAR_TEXTURES.len() > SPRITE_TEXTURES);
        assert!(JAR_TEXTURES[..SPRITE_TEXTURES]
            .iter()
            .all(|p| p.starts_with("textures/blocks/")));
        assert_eq!(
            JAR_TEXTURES[tex::OXEYE as usize],
            "textures/blocks/flower_oxeye_daisy.png"
        );
    }

    #[test]
    fn the_colormap_lookup_falls_back_without_the_jar() {
        let colors = BiomeColors::from_jar(&JarTextures::empty());
        assert!(colors.grass[1] > colors.grass[0] && colors.grass[1] > colors.grass[2]);
        assert!(colors.foliage[1] > colors.foliage[2]);
    }
}
