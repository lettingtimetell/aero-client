//! One GPU block atlas.
//!
//! The five full-mip block-atlas owners (terrain, particles, block overlay, first person, screen
//! overlay) installed inside an `AtlasShareScope` share one texture. An animation flush in the
//! driver's owner order then records each record once instead of five times, and every owner's
//! texels stay byte-identical to the private-copy path at every mip, through an anaglyph on/off
//! cycle (after which the particle and screen-overlay replays go back to private textures).
//!
//! Texels are read with a compute shader's `textureLoad`, which is what the owners' shaders see
//! and needs no `COPY_SRC` (the owners' textures are `TEXTURE_BINDING | COPY_DST` only).

use mc_assets::{Atlas, AtlasAnimationUpload};
use mc_render::atlas_transfer::{AtlasShareScope, AtlasStaging, AtlasTransferBatch, BatchStats};
use mc_render::particle::ParticlePhase;
use mc_render::pipeline::FirstPersonPhase;
use mc_render::post::OverlayPhase;
use mc_render::terrain::{BlockOverlayPhase, TerrainRenderer};
use mc_render::{Gpu, GpuOptions};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 32;
const LEVELS: u32 = 3;

fn level_size(level: u32) -> (u32, u32) {
    (WIDTH >> level, HEIGHT >> level)
}

fn pattern(width: u32, height: u32, seed: u32) -> Vec<u8> {
    (0..width * height * 4)
        .map(|i| {
            (i.wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(97))
                >> 13) as u8
        })
        .collect()
}

fn atlas() -> Atlas {
    let level = |level: u32, seed: u32| {
        let (width, height) = level_size(level);
        pattern(width, height, seed)
    };
    Atlas {
        texture: level(0, 9),
        mipmaps: (1..LEVELS).map(|l| level(l, 31 * l + 7)).collect(),
        w: WIDTH,
        h: HEIGHT,
    }
}

fn record(level: u32, x: u32, y: u32, width: u32, height: u32, seed: u32) -> AtlasAnimationUpload {
    AtlasAnimationUpload {
        mip_level: level,
        x,
        y,
        width,
        height,
        rgba: pattern(width, height, seed),
    }
}

/// One tick's dirty rectangles: two 16x16 sprites at every level (as the animator emits them),
/// plus a mip-0 rectangle overlapping the first sprite, so record order matters.
fn tick_records(step: u32) -> Vec<AtlasAnimationUpload> {
    let mut uploads = Vec::new();
    for (sprite, (x, y)) in [(16_u32, 0_u32), (32, 16)].into_iter().enumerate() {
        for level in 0..LEVELS {
            let seed = step * 1000 + sprite as u32 * 100 + level;
            uploads.push(record(
                level,
                x >> level,
                y >> level,
                16 >> level,
                16 >> level,
                seed,
            ));
        }
    }
    uploads.push(record(0, 20, 4, 8, 8, step * 1000 + 999));
    uploads
}

fn apply(levels: &mut [Vec<u8>], upload: &AtlasAnimationUpload) {
    let (level_width, _) = level_size(upload.mip_level);
    let row_bytes = (upload.width * 4) as usize;
    for (row, bytes) in upload.rgba.chunks_exact(row_bytes).enumerate() {
        let start = (((upload.y as usize + row) * level_width as usize) + upload.x as usize) * 4;
        levels[upload.mip_level as usize][start..start + row_bytes].copy_from_slice(bytes);
    }
}

struct Owners {
    terrain: TerrainRenderer,
    particles: ParticlePhase,
    block_overlay: BlockOverlayPhase,
    first_person: FirstPersonPhase,
    overlay: OverlayPhase,
}

struct Flush {
    stats: BatchStats,
    /// Each owner's texture as its sink handed it over, in driver order.
    textures: Vec<wgpu::Texture>,
}

impl Owners {
    /// The client's install order (`RenderStack::install`): terrain first.
    fn install(gpu: &Gpu, atlas: &Atlas, anaglyph: bool) -> Self {
        let terrain = TerrainRenderer::new_with_anaglyph(gpu, atlas, anaglyph);
        let block_overlay = BlockOverlayPhase::new_with_anaglyph(gpu, Some(atlas), anaglyph);
        let mut particles = ParticlePhase::new_with_anaglyph(gpu, anaglyph);
        particles.install_block_atlas(gpu, atlas);
        let mut first_person = FirstPersonPhase::new_with_anaglyph(gpu, anaglyph);
        first_person.set_atlas(gpu, atlas);
        let mut overlay = OverlayPhase::new_with_anaglyph(gpu, anaglyph);
        overlay.set_block_atlas(gpu, atlas);
        Self {
            terrain,
            particles,
            block_overlay,
            first_person,
            overlay,
        }
    }

    fn set_anaglyph(&mut self, gpu: &Gpu, enabled: bool) {
        self.terrain.set_anaglyph_with_gpu(gpu, enabled);
        self.block_overlay.set_anaglyph_with_gpu(gpu, enabled);
        self.particles.set_anaglyph_with_gpu(gpu, enabled);
        self.first_person.set_anaglyph_with_gpu(gpu, enabled);
        self.overlay.set_anaglyph_with_gpu(gpu, enabled);
    }

    /// `Client::flush_atlas_animation_uploads`'s batch branch: every owner for each record.
    fn flush(
        &self,
        gpu: &Gpu,
        staging: &mut AtlasStaging,
        uploads: &[AtlasAnimationUpload],
        anaglyph: bool,
    ) -> Flush {
        let mut batch = AtlasTransferBatch::new_with_staging(gpu, staging, uploads, anaglyph)
            .expect("valid transfer plan");
        let mut textures = Vec::new();
        for (index, upload) in uploads.iter().enumerate() {
            let mut sink = |texture: &wgpu::Texture, enabled: bool| {
                if index == 0 {
                    textures.push(texture.clone());
                }
                batch.copy_to(index, texture, enabled);
            };
            assert!(self
                .terrain
                .upload_atlas_animation_with_sink(upload, &mut sink));
            assert!(self
                .particles
                .upload_atlas_animation_with_sink(upload, &mut sink));
            assert!(self
                .block_overlay
                .upload_atlas_animation_with_sink(upload, &mut sink));
            assert!(self
                .first_person
                .upload_atlas_animation_with_sink(upload, &mut sink));
            assert!(self
                .overlay
                .upload_block_atlas_animation_with_sink(upload, &mut sink));
        }
        Flush {
            stats: batch.finish(gpu),
            textures,
        }
    }
}

/// Reads one mip level of any `TEXTURE_BINDING` RGBA8 texture through `textureLoad`.
struct TexelReader {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

const READER_WGSL: &str = r"
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> texels: array<u32>;
struct Params { level: u32, width: u32, height: u32, pad: u32 }
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let color = textureLoad(source, vec2<i32>(id.xy), i32(params.level));
    texels[id.y * params.width + id.x] = pack4x8unorm(color);
}
";

impl TexelReader {
    fn new(gpu: &Gpu) -> Self {
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("x14 texel reader"),
                source: wgpu::ShaderSource::Wgsl(READER_WGSL.into()),
            });
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("x14 texel reader"),
                entries: &[
                    entry(
                        0,
                        wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                    ),
                    entry(
                        1,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    ),
                    entry(
                        2,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    ),
                ],
            });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("x14 texel reader"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("x14 texel reader"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Self { pipeline, layout }
    }

    fn read(&self, gpu: &Gpu, texture: &wgpu::Texture, level: u32) -> Vec<u8> {
        let (width, height) = level_size(level);
        let bytes = u64::from(width * height * 4);
        let storage = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("x14 texels"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("x14 texel readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("x14 reader params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let words: Vec<u8> = [level, width, height, 0]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        gpu.queue.write_buffer(&params, 0, &words);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("x14 texel reader"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: storage.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, bytes);
        gpu.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            tx.send(result).expect("map result receiver");
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("readback poll");
        rx.recv().expect("map callback").expect("readback map");
        let texels = readback
            .get_mapped_range(..)
            .expect("readback view")
            .to_vec();
        readback.unmap();
        texels
    }
}

fn distinct(textures: &[wgpu::Texture]) -> usize {
    let mut seen: Vec<&wgpu::Texture> = Vec::new();
    for texture in textures {
        if !seen.contains(&texture) {
            seen.push(texture);
        }
    }
    seen.len()
}

#[test]
fn shared_block_atlas_matches_private_copies_and_copies_each_record_once() {
    let gpu = Gpu::headless(8, 8, GpuOptions::default())
        .expect("this qualification requires a working GPU adapter");
    let errors = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let atlas = atlas();
    let reader = TexelReader::new(&gpu);
    let mut private = Owners::install(&gpu, &atlas, false);
    let mut shared = {
        let scope = AtlasShareScope::begin_if(true, &atlas, false);
        let owners = Owners::install(&gpu, &atlas, false);
        assert_eq!(
            scope.adopted(),
            4,
            "terrain creates the texture, the other four adopt it"
        );
        owners
    };
    let mut expected: Vec<Vec<u8>> = std::iter::once(atlas.texture.clone())
        .chain(atlas.mipmaps.iter().cloned())
        .collect();
    let mut private_staging = AtlasStaging::new(&gpu);
    let mut shared_staging = AtlasStaging::new(&gpu);
    let mut anaglyph = false;
    let mut shared_copies = Vec::new();
    for (step, toggle) in [None, None, Some(true), None, Some(false), None]
        .into_iter()
        .enumerate()
    {
        if let Some(enabled) = toggle {
            private.set_anaglyph(&gpu, enabled);
            shared.set_anaglyph(&gpu, enabled);
            anaglyph = enabled;
        }
        let uploads = tick_records(step as u32);
        let records = uploads.len() as u64;
        let private_flush = private.flush(&gpu, &mut private_staging, &uploads, anaglyph);
        let shared_flush = shared.flush(&gpu, &mut shared_staging, &uploads, anaglyph);
        for upload in &uploads {
            apply(&mut expected, upload);
        }

        assert_eq!(
            distinct(&private_flush.textures),
            5,
            "step {step}: five private copies"
        );
        assert_eq!(
            (
                private_flush.stats.copy_commands,
                private_flush.stats.deduplicated_copies
            ),
            (5 * records, 0),
            "step {step}"
        );
        // Before any toggle all five share one texture. The particle and screen-overlay replays
        // rebuild a private texture outside any scope, so three remain after the first toggle.
        let textures = distinct(&shared_flush.textures) as u64;
        assert_eq!(
            textures,
            if step < 2 { 1 } else { 3 },
            "step {step}: shared textures"
        );
        assert_eq!(
            (
                shared_flush.stats.copy_commands,
                shared_flush.stats.deduplicated_copies
            ),
            (textures * records, (5 - textures) * records),
            "step {step}"
        );
        shared_copies.push(shared_flush.stats.copy_commands);

        for level in 0..LEVELS {
            let want = mc_assets::sprite::anaglyph_rgba(&expected[level as usize], anaglyph);
            for owner in 0..5 {
                let private_texels = reader.read(&gpu, &private_flush.textures[owner], level);
                let shared_texels = reader.read(&gpu, &shared_flush.textures[owner], level);
                assert!(
                    private_texels == want,
                    "step {step} owner {owner} level {level}: private copy differs from CPU"
                );
                assert!(
                    shared_texels == private_texels,
                    "step {step} owner {owner} level {level}: shared atlas differs from private"
                );
            }
        }
    }
    assert_eq!(
        shared_copies,
        [7, 7, 21, 21, 21, 21],
        "copy commands per flush, shared"
    );
    assert!(
        pollster::block_on(errors.pop()).is_none(),
        "unexpected wgpu validation error"
    );
}

/// Outside a scope (or with the switch off) every owner keeps its own texture, exactly as before.
#[test]
fn without_a_share_scope_every_owner_keeps_a_private_atlas() {
    let gpu = Gpu::headless(8, 8, GpuOptions::default())
        .expect("this qualification requires a working GPU adapter");
    let atlas = atlas();
    let uploads = tick_records(0);
    for enabled in [None, Some(false)] {
        let owners = match enabled {
            None => Owners::install(&gpu, &atlas, false),
            Some(switch) => {
                let scope = AtlasShareScope::begin_if(switch, &atlas, false);
                assert!(!scope.is_active());
                let owners = Owners::install(&gpu, &atlas, false);
                assert_eq!(scope.adopted(), 0);
                owners
            }
        };
        let mut staging = AtlasStaging::new(&gpu);
        let flush = owners.flush(&gpu, &mut staging, &uploads, false);
        assert_eq!(distinct(&flush.textures), 5);
        assert_eq!(flush.stats.copy_commands, 5 * uploads.len() as u64);
        assert_eq!(flush.stats.deduplicated_copies, 0);
    }
}
