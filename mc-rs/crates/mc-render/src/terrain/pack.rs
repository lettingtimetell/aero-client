//! The shader packs' terrain variant: sun shading, shadows,
//! waving plants and shader water for the four terrain layers, and the shadow-map pass.
//!
//! A child module of `terrain::phase` (declared there with `#[path]`, like `aniso`), so it shares
//! that file's private vertex-offset layout, section buffers and layer helpers.
//!
//! # What changes while a pack runs
//!
//! With a [`ShaderCore`] attached ([`TerrainRenderer::set_shader_core`]) and the running pack's
//! config using a terrain feature ([`CoreConfig::uses_terrain_variant`]), every layer draws with a
//! second set of pipelines built from `shaderpack/common.wgsl` + `terrain.wgsl` + `pack.wgsl`:
//! the same vertex buffers, offsets, index buffer, group-1 bind groups and draw plans, plus a
//! group 2 (frame, shadow and pack uniforms, the shadow map). The translucent layer additionally
//! reads the scene and the opaque depth (group 3) while shader water is on, through a pass whose
//! depth attachment is read-only.
//!
//! # Off costs nothing
//!
//! No core attached (every client without a pack ever switched on) is one `Option` test per layer
//! draw. A core with no pack running is one `RefCell` borrow and a `bool`. The pipelines and every
//! pack resource are built the first frame a pack runs and dropped when it stops; the vanilla
//! pipelines, bind groups and draw plans are never touched, so the pack-off frame is the frame
//! without this module, byte for byte (`tests/sa_pack_terrain.rs`).
//!
//! # The shadow pass
//!
//! [`TerrainRenderer::record_pack_shadows`] draws the opaque and alpha-tested layers of every
//! resident section inside the shadow box into the core's shadow map, from the light, relative to
//! a section-aligned caster origin. The caster list is rebuilt when the origin or the quantised
//! light direction changes, or (at most every [`RELIST_FRAMES`] frames) when the resident
//! geometry changes; the map is redrawn only when the list or the matrices changed. On every other
//! frame the pass records nothing at all.

use std::rc::Rc;

use mc_data::RenderLayer;

use super::{layer_index, ChunkOffset, TerrainRenderer, LAYER_COUNT};
use crate::shaderpack::config::{CoreConfig, PackMesh};
use crate::shaderpack::shadow::{ShadowKey, ShadowUniform, SHADOW_FORMAT};
use crate::shaderpack::{ShaderCore, COMMON_WGSL};
use crate::terrain::vertex::{PackedTerrainVertex, TerrainVertex};
use crate::Gpu;

/// The material codes the mesher writes into a vertex's `light.z` while [`PackMesh::flags`] is
/// on. Zero (every other vertex) is no material.
pub mod material {
    /// Grass, flowers, crops, saplings: sway, anchored at the root (`light.w` = height 0..255).
    pub const PLANT: u8 = 1;
    /// Leaves and vines: the whole block sways gently.
    pub const LEAVES: u8 = 2;
    /// Water: shader water.
    pub const WATER: u8 = 3;
}

/// Frames between caster re-lists while only the resident geometry changes (chunks loading).
pub const RELIST_FRAMES: u32 = 8;

/// What the mesher writes differently for one block while a pack runs. Computed once per block;
/// [`PackQuad::VANILLA`] (always, with no pack) changes nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PackQuad {
    /// Leave vanilla's per-face shade out of the vertex colour.
    pub flat_shade: bool,
    /// The material code for `light.z` (0 = none).
    pub material: u8,
    /// 0 = a one-block plant, 1 = the lower and 2 = the upper half of a double plant (the sway
    /// weight runs 0..1 over the whole plant).
    pub half: u8,
}

impl PackQuad {
    /// Vanilla meshing.
    pub const VANILLA: PackQuad = PackQuad {
        flat_shade: false,
        material: 0,
        half: 0,
    };

    /// The block's pack meshing under `mesh`. One compare while `mesh` is vanilla.
    #[inline]
    #[must_use]
    pub fn for_block(state: mc_data::BlockState, mesh: PackMesh) -> PackQuad {
        if mesh.is_vanilla() {
            return PackQuad::VANILLA;
        }
        let (material, half) = if mesh.flags {
            match state.block().0 {
                // leaves, leaves2
                18 | 161 => (material::LEAVES, 0),
                // sapling, tall grass, dead bush, flowers, wheat, pumpkin and melon stems,
                // carrots, potatoes
                6 | 31 | 32 | 37 | 38 | 59 | 104 | 105 | 141 | 142 => (material::PLANT, 0),
                // double plant: meta bit 8 is the upper half
                175 => (material::PLANT, if state.meta() & 8 != 0 { 2 } else { 1 }),
                _ => (0, 0),
            }
        } else {
            (0, 0)
        };
        PackQuad {
            flat_shade: mesh.flat_shade,
            material,
            half,
        }
    }

    /// The baked face shade byte to use (`255` = none while flat shading).
    #[inline]
    #[must_use]
    pub fn shade(self, vanilla: f32) -> f32 {
        if self.flat_shade {
            255.0
        } else {
            vanilla
        }
    }

    /// The vertex's two spare light bytes: zero, as vanilla's, unless this block has a material
    /// code. `model_y` is the vertex's height inside the block model (0 = bottom).
    #[inline]
    #[must_use]
    pub fn light_meta(self, model_y: f32) -> [u8; 2] {
        match self.material {
            0 => [0, 0],
            material::PLANT => {
                let y = model_y.clamp(0.0, 1.0);
                let weight = match self.half {
                    1 => 0.5 * y,
                    2 => 0.5 + 0.5 * y,
                    _ => y,
                };
                [material::PLANT, (weight * 255.0).round() as u8]
            }
            code => [code, 255],
        }
    }
}

/// The module source the pack pipelines compile.
#[must_use]
pub fn shader_source() -> String {
    format!(
        "{}\n{}\n{}",
        COMMON_WGSL,
        include_str!("terrain.wgsl"),
        include_str!("pack.wgsl")
    )
}

/// `pack.wgsl`'s `TerrainPackUniform`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct TerrainPackUniform {
    light: [f32; 4],
    waving: [f32; 4],
    water: [f32; 4],
    water_tint: [f32; 4],
    caster_origin: [f32; 4],
}

impl TerrainPackUniform {
    fn new(config: &CoreConfig, caster_origin: [f32; 3]) -> Self {
        let flag = |on: bool| if on { 1.0 } else { 0.0 };
        let lighting = config.lighting.unwrap_or_default();
        let shadows = config.shadows.unwrap_or_default();
        let waving = config.waving.unwrap_or_default();
        let water = config.water.unwrap_or_default();
        // Without sun shading, shadows still take the sun's share of the sky light.
        let share = if config.lighting.is_some() {
            lighting.sun_share
        } else {
            0.55
        };
        TerrainPackUniform {
            light: [
                flag(config.shadows.is_some()),
                share,
                lighting.shade_floor,
                flag(config.lighting.is_some()),
            ],
            waving: [
                flag(config.waving.is_some_and(|w| w.plants)),
                flag(config.waving.is_some_and(|w| w.leaves)),
                waving.strength,
                waving.speed,
            ],
            water: [
                flag(config.water.is_some()),
                water.waves,
                water.clarity,
                flag(config.water.is_some_and(|w| w.reflections)),
            ],
            water_tint: [
                water.tint[0],
                water.tint[1],
                water.tint[2],
                shadows.strength,
            ],
            caster_origin: [caster_origin[0], caster_origin[1], caster_origin[2], 0.0],
        }
    }
}

/// One section's draw in the shadow pass.
struct CasterItem {
    buffer: Rc<wgpu::Buffer>,
    base_vertex: i32,
    quads: u32,
    instance: u32,
}

/// The shadow casters and what the map currently holds.
#[derive(Default)]
struct Casters {
    /// The key the list was built for (origin and direction) and the geometry generations.
    listed: Option<(ShadowKey, [u64; 3])>,
    frames_since_list: u32,
    /// Per shadow-casting layer (solid, cutout mipped, cutout).
    items: [Vec<CasterItem>; 3],
    offsets: Option<wgpu::Buffer>,
    offsets_capacity: u32,
    /// `(key, geometry, map generation, cutout casters)` of the last render.
    rendered: Option<(ShadowKey, [u64; 3], u64, bool)>,
    /// Sections and quads drawn by the last shadow render (for tests and the F3 line).
    last_drawn: (usize, usize),
    renders: u64,
}

/// The GPU half, built on first use.
struct PackGpu {
    sample_count: u32,
    format: wgpu::TextureFormat,
    layout2: wgpu::BindGroupLayout,
    layout3: wgpu::BindGroupLayout,
    /// Solid, cutout mipped, cutout, translucent without scene inputs.
    pipelines: [wgpu::RenderPipeline; LAYER_COUNT],
    /// Translucent with scene colour and depth (shader water).
    translucent_scene: wgpu::RenderPipeline,
    shadow_solid: wgpu::RenderPipeline,
    shadow_cutout: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    uploaded: Option<TerrainPackUniform>,
    compare_sampler: wgpu::Sampler,
    /// Bound in place of the shadow map and its uniform while shadows are off.
    dummy_shadow_view: wgpu::TextureView,
    dummy_shadow_uniform: wgpu::Buffer,
    /// Group 2 and what it was built from: the frame buffer and the shadow map generation.
    group2: Option<(wgpu::Buffer, u64, wgpu::BindGroup)>,
    /// Group 2 for the shadow pass itself: the same uniforms, with the dummy map bound in place
    /// of the shadow map (a texture cannot be sampled and rendered into in one pass).
    group2_casters: Option<wgpu::BindGroup>,
    /// Group 3 and the scene view it reads.
    group3: Option<(wgpu::TextureView, wgpu::BindGroup)>,
}

/// The pack state a [`TerrainRenderer`] carries.
#[derive(Default)]
pub(super) struct PackState {
    core: Option<ShaderCore>,
    gpu: Option<PackGpu>,
    casters: Casters,
    /// The scene snapshot handed to the translucent phase this frame (shader water).
    scene: Option<wgpu::TextureView>,
}

impl PackState {
    /// The pack pipeline for a layer (translucent: the scene-reading one when `scene`).
    pub(super) fn pipeline(&self, layer: usize, scene: bool) -> Option<&wgpu::RenderPipeline> {
        let built = self.gpu.as_ref()?;
        if scene && layer == layer_index(RenderLayer::Translucent) {
            Some(&built.translucent_scene)
        } else {
            Some(&built.pipelines[layer])
        }
    }

    /// Group 2 of the pack pipelines.
    pub(super) fn group(&self) -> Option<&wgpu::BindGroup> {
        self.gpu.as_ref()?.group2.as_ref().map(|(_, _, g)| g)
    }
}

impl TerrainRenderer {
    /// Attach the shader packs' shared state (once, at install). `None` detaches it. Nothing is
    /// built until a pack runs.
    pub fn set_shader_core(&mut self, core: Option<ShaderCore>) {
        self.pack.core = core;
        if self.pack.core.is_none() {
            self.pack.gpu = None;
            self.pack.casters = Casters::default();
        }
    }

    /// The running pack's terrain config, or `None` while no pack runs a terrain feature. With no
    /// pack running: one `RefCell` borrow and two `bool` tests.
    fn pack_config(&self) -> Option<CoreConfig> {
        let core = self.pack.core.as_ref()?;
        if !core.any_running() {
            return None;
        }
        let config = core.config();
        config.uses_terrain_variant().then_some(config)
    }

    /// Whether frames draw with the pack pipelines.
    #[must_use]
    pub fn shader_pack_active(&self) -> bool {
        self.pack_config().is_some()
    }

    /// Whether the pack resources exist (only after a frame drew with a pack on).
    #[must_use]
    pub fn shader_pack_allocated(&self) -> bool {
        self.pack.gpu.is_some() || self.pack.casters.offsets.is_some()
    }

    /// The mesher mode the running pack needs ([`PackMesh::VANILLA`] while none runs a terrain
    /// feature).
    #[must_use]
    pub fn shader_pack_mesh(&self) -> PackMesh {
        self.pack_config().map_or(PackMesh::VANILLA, |c| c.mesh())
    }

    /// `(sections, quads)` the last shadow render drew, and how many renders there were.
    #[must_use]
    pub fn shadow_pass_stats(&self) -> ((usize, usize), u64) {
        (self.pack.casters.last_drawn, self.pack.casters.renders)
    }

    /// Re-list and re-render the shadow casters on the next frame,
    /// whatever the cache says. A video export calls it before every frame it renders, so each
    /// frame's map is that frame's own: the relist otherwise waits [`RELIST_FRAMES`] frames, and
    /// a cached map keeps the matrices of the frame that drew it, so what a video showed at a
    /// replay time would depend on how many frames came before it. Normal play never calls it.
    pub fn invalidate_pack_shadows(&mut self) {
        self.pack.casters.listed = None;
        self.pack.casters.rendered = None;
    }

    /// Whether the translucent layer wants the scene snapshot this frame (shader water).
    pub(super) fn pack_wants_scene(&self) -> bool {
        self.pack_config().is_some_and(|c| c.water.is_some())
    }

    /// The scene snapshot for this frame's translucent draw.
    pub(super) fn set_pack_scene(&mut self, scene: &wgpu::TextureView) {
        self.pack.scene = Some(scene.clone());
    }

    /// Build what a pack frame needs and upload the uniform. Returns whether this layer draw
    /// uses the pack pipelines. No core: one `Option` test.
    pub(super) fn prepare_pack(&mut self, gpu: &Gpu) -> bool {
        if self.pack.core.is_none() {
            return false;
        }
        let Some(config) = self.pack_config() else {
            if self.pack.gpu.is_some() {
                self.pack.gpu = None;
                self.pack.casters = Casters::default();
            }
            return false;
        };
        let core = self.pack.core.clone().expect("checked above");
        let Some(frame_buffer) = core.frame_buffer() else {
            return false;
        };
        if self
            .pack
            .gpu
            .as_ref()
            .is_none_or(|g| g.sample_count != gpu.sample_count() || g.format != gpu.config.format)
        {
            self.pack.gpu = Some(build(gpu, &self.group1_layout, self.options));
        }
        let shadow = core.shadow();
        let origin = shadow.as_ref().map_or([0.0; 3], |s| {
            let o = s.frame.caster_origin;
            let wrap = |v: f64| {
                let w = crate::shaderpack::frame::EYE_WRAP;
                (v - (v / w).floor() * w) as f32
            };
            [wrap(o.x), wrap(o.y), wrap(o.z)]
        });
        let shadow_generation = core.with_shadow(|map| map.generation);
        let built = self.pack.gpu.as_mut().expect("built above");
        let uniform = TerrainPackUniform::new(&config, origin);
        if built.uploaded != Some(uniform) {
            gpu.write_buffer_staged(&built.uniform, 0, bytemuck::bytes_of(&uniform));
            built.uploaded = Some(uniform);
        }
        let group_current = built
            .group2
            .as_ref()
            .is_some_and(|(buffer, generation, _)| {
                *buffer == frame_buffer && *generation == shadow_generation
            });
        if !group_current {
            let (shadow_view, shadow_uniform) = match &shadow {
                Some(s) => (s.view.clone(), s.uniform.clone()),
                None => (
                    built.dummy_shadow_view.clone(),
                    built.dummy_shadow_uniform.clone(),
                ),
            };
            let make = |map: &wgpu::TextureView, label: &str| {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &built.layout2,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: frame_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: shadow_uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(map),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&built.compare_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: built.uniform.as_entire_binding(),
                        },
                    ],
                })
            };
            let group = make(&shadow_view, "terrain pack group2");
            let casters = make(
                &built.dummy_shadow_view,
                "terrain pack group2 (shadow pass)",
            );
            built.group2 = Some((frame_buffer, shadow_generation, group));
            built.group2_casters = Some(casters);
        }
        true
    }

    /// Take this frame's scene snapshot and return group 3 for it (shader water), building it
    /// when the snapshot view changed. `None` when no snapshot was handed over.
    pub(super) fn take_pack_scene_group(&mut self, gpu: &Gpu) -> Option<wgpu::BindGroup> {
        let scene = self.pack.scene.take()?;
        if !self.pack_wants_scene() {
            return None;
        }
        let built = self.pack.gpu.as_mut()?;
        if built.group3.as_ref().is_none_or(|(view, _)| *view != scene) {
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain pack group3"),
                layout: &built.layout3,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&scene),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(gpu.depth_view()),
                    },
                ],
            });
            built.group3 = Some((scene, group));
        }
        built.group3.as_ref().map(|(_, g)| g.clone())
    }

    /// Open the translucent pass for shader water: like [`Gpu::begin_pass`] with the shared
    /// depth buffer, but read-only, so group 3 can sample it in the same pass. Packs never run
    /// with MSAA, so there is no resolve target.
    pub(super) fn begin_pack_scene_pass<'e>(
        gpu: &Gpu,
        enc: &'e mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        label: &str,
    ) -> wgpu::RenderPass<'e> {
        enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: gpu.depth_view(),
                depth_ops: None,
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    /// Draw the shadow casters into the core's shadow map, if anything changed since the last
    /// render. Called by `shaderpack::ShadowPhase` before the sky.
    pub fn record_pack_shadows(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        core: &ShaderCore,
    ) {
        if !self.pack.core.as_ref().is_some_and(|c| c.same(core)) {
            return;
        }
        let Some(config) = self.pack_config() else {
            return;
        };
        let Some(settings) = config.shadows else {
            return;
        };
        if !self.prepare_pack(gpu) {
            return;
        }
        let Some(shadow) = core.shadow() else {
            return;
        };
        let frame = shadow.frame;
        let geometry = [
            self.layer_generations[layer_index(RenderLayer::Solid)],
            self.layer_generations[layer_index(RenderLayer::CutoutMipped)],
            self.layer_generations[layer_index(RenderLayer::Cutout)],
        ];
        let casters = &mut self.pack.casters;
        casters.frames_since_list = casters.frames_since_list.saturating_add(1);
        let relist = match casters.listed {
            None => true,
            Some((key, listed_geometry)) => {
                key.origin != frame.key.origin
                    || key.direction_step != frame.key.direction_step
                    || key.tilt_bits != frame.key.tilt_bits
                    || key.distance_bits != frame.key.distance_bits
                    || (listed_geometry != geometry && casters.frames_since_list >= RELIST_FRAMES)
            }
        };
        if relist {
            self.list_casters(gpu, &frame, geometry);
        }
        let map_generation = core.with_shadow(|map| map.generation);
        let casters = &self.pack.casters;
        let listed_geometry = casters.listed.map_or(geometry, |(_, g)| g);
        let wanted = (
            frame.key,
            listed_geometry,
            map_generation,
            settings.cutout_casters,
        );
        if casters.rendered == Some(wanted) {
            return;
        }
        let built = self.pack.gpu.as_ref().expect("prepared above");
        let Some(group2) = built.group2_casters.as_ref() else {
            return;
        };
        let Some(offsets) = casters.offsets.as_ref() else {
            return;
        };
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shaderpack/shadow terrain"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &shadow.terrain_view,
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
        pass.set_index_buffer(self.index_buffer.slice(..), self.index_format);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.set_bind_group(2, group2, &[]);
        pass.set_vertex_buffer(1, offsets.slice(..));
        let mut sections = 0usize;
        let mut quads = 0usize;
        for (slot, layer) in [
            RenderLayer::Solid,
            RenderLayer::CutoutMipped,
            RenderLayer::Cutout,
        ]
        .into_iter()
        .enumerate()
        {
            if slot > 0 && !settings.cutout_casters {
                continue;
            }
            let items = &casters.items[slot];
            if items.is_empty() {
                continue;
            }
            let li = layer_index(layer);
            pass.set_pipeline(if slot == 0 {
                &built.shadow_solid
            } else {
                &built.shadow_cutout
            });
            pass.set_bind_group(1, &self.bind_groups[li], &[]);
            let mut bound: Option<usize> = None;
            for item in items {
                let id = Rc::as_ptr(&item.buffer) as usize;
                if bound != Some(id) {
                    pass.set_vertex_buffer(0, item.buffer.slice(..));
                    bound = Some(id);
                }
                pass.draw_indexed(
                    0..item.quads * 6,
                    item.base_vertex,
                    item.instance..item.instance + 1,
                );
                sections += 1;
                quads += item.quads as usize;
            }
        }
        drop(pass);
        let casters = &mut self.pack.casters;
        casters.rendered = Some(wanted);
        casters.last_drawn = (sections, quads);
        casters.renders += 1;
        tracing::debug!(
            sections,
            quads,
            renders = casters.renders,
            key = ?frame.key,
            "shader pack shadow map rendered"
        );
    }

    /// Rebuild the caster list for `frame`: every resident section with opaque or alpha-tested
    /// geometry whose bounding sphere overlaps the shadow box.
    fn list_casters(
        &mut self,
        gpu: &Gpu,
        frame: &crate::shaderpack::shadow::ShadowFrame,
        geometry: [u64; 3],
    ) {
        let origin = frame.caster_origin;
        let m = frame.caster_matrix;
        // A section's bounding sphere (radius 8 sqrt 3) in NDC units of the box.
        let half = f64::from(frame.settings.distance) as f32;
        let r_xy = 13.9 / half;
        let mut offsets: Vec<ChunkOffset> = Vec::new();
        let mut items: [Vec<CasterItem>; 3] = Default::default();
        let layers = [
            layer_index(RenderLayer::Solid),
            layer_index(RenderLayer::CutoutMipped),
            layer_index(RenderLayer::Cutout),
        ];
        for (pos, chunk) in &self.chunks {
            if layers.iter().all(|&li| chunk.layers[li].is_none()) {
                continue;
            }
            let o = pos.origin();
            let rel = glam::Vec3::new(
                (f64::from(o.x) - origin.x) as f32,
                (f64::from(o.y) - origin.y) as f32,
                (f64::from(o.z) - origin.z) as f32,
            );
            let center = m * (rel + glam::Vec3::splat(8.0)).extend(1.0);
            if center.x.abs() > 1.0 + r_xy || center.y.abs() > 1.0 + r_xy {
                continue;
            }
            if center.z < -0.1 || center.z > 1.1 {
                continue;
            }
            let instance = offsets.len() as u32;
            offsets.push(ChunkOffset {
                offset: [rel.x, rel.y, rel.z, 0.0],
            });
            for (slot, &li) in layers.iter().enumerate() {
                if let Some(slice) = &chunk.layers[li] {
                    items[slot].push(CasterItem {
                        buffer: slice.buffer.clone(),
                        base_vertex: slice.base_vertex(),
                        quads: slice.quads,
                        instance,
                    });
                }
            }
        }
        // Group draws by vertex buffer so the pass rebinds as rarely as the main passes do.
        for list in &mut items {
            list.sort_by_key(|item| Rc::as_ptr(&item.buffer) as usize);
        }
        let casters = &mut self.pack.casters;
        let needed = (offsets.len() as u32).max(1);
        if casters.offsets.is_none() || casters.offsets_capacity < needed {
            let capacity = needed.next_power_of_two().max(256);
            casters.offsets = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("terrain pack shadow offsets"),
                size: u64::from(capacity) * std::mem::size_of::<ChunkOffset>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            casters.offsets_capacity = capacity;
        }
        if !offsets.is_empty() {
            gpu.write_buffer_staged(
                casters.offsets.as_ref().expect("built above"),
                0,
                bytemuck::cast_slice(&offsets),
            );
        }
        casters.items = items;
        casters.listed = Some((frame.key, geometry));
        casters.frames_since_list = 0;
    }
}

/// The pack module, its layouts, samplers and uniform, and every pipeline, each with the
/// original layer pipeline's state except the shader entry points and the extra groups.
fn build(
    gpu: &Gpu,
    group1_layout: &wgpu::BindGroupLayout,
    options: super::TerrainOptions,
) -> PackGpu {
    let device = &gpu.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("terrain pack"),
        source: wgpu::ShaderSource::Wgsl(shader_source().into()),
    });
    let uniform_entry = |binding, visibility| wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
    let layout2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("terrain pack group2"),
        entries: &[
            uniform_entry(0, vf),
            uniform_entry(1, vf),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
            uniform_entry(4, vf),
        ],
    });
    let layout3 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("terrain pack group3"),
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
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("terrain pack layout"),
        bind_group_layouts: &[
            Some(gpu.camera_layout()),
            Some(group1_layout),
            Some(&layout2),
        ],
        immediate_size: 0,
    });
    let layout_scene = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("terrain pack scene layout"),
        bind_group_layouts: &[
            Some(gpu.camera_layout()),
            Some(group1_layout),
            Some(&layout2),
            Some(&layout3),
        ],
        immediate_size: 0,
    });
    let vertex_layouts = [
        Some(if options.packed_vertices {
            PackedTerrainVertex::layout()
        } else {
            TerrainVertex::layout()
        }),
        Some(ChunkOffset::LAYOUT),
    ];
    let pipeline = |layout: &wgpu::PipelineLayout, fragment: &str, translucent: bool| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain pack"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_pack"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &vertex_layouts,
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: options.cull_backfaces.then_some(wgpu::Face::Back),
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::DEPTH_FORMAT,
                depth_write_enabled: Some(!translucent),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: gpu.sample_count(),
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(fragment),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: translucent.then_some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    let pipelines = std::array::from_fn(|i| match i {
        0 => pipeline(&layout, "fs_pack_solid", false),
        3 => pipeline(&layout, "fs_pack_translucent_plain", true),
        _ => pipeline(&layout, "fs_pack_cutout", false),
    });
    let translucent_scene = pipeline(&layout_scene, "fs_pack_translucent", true);
    let shadow_pipeline = |fragment: Option<&str>| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain pack shadow"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_shadow"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &vertex_layouts,
            },
            // Both faces cast: a sealed surface's back faces are what the light sees at grazing
            // angles, and drawing them costs nothing in a depth-only pass.
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: SHADOW_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 1.75,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: fragment.map(|entry| wgpu::FragmentState {
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[],
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    let shadow_solid = shadow_pipeline(None);
    let shadow_cutout = shadow_pipeline(Some("fs_shadow_cutout"));
    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain pack uniform"),
        size: std::mem::size_of::<TerrainPackUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let compare_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("terrain pack shadow sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    });
    let dummy_shadow_view = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain pack no shadow map"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADOW_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default());
    let dummy_shadow_uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain pack no shadow uniform"),
        size: std::mem::size_of::<ShadowUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });
    PackGpu {
        sample_count: gpu.sample_count(),
        format: gpu.config.format,
        layout2,
        layout3,
        pipelines,
        translucent_scene,
        shadow_solid,
        shadow_cutout,
        uniform,
        uploaded: None,
        compare_sampler,
        dummy_shadow_view,
        dummy_shadow_uniform,
        group2: None,
        group2_casters: None,
        group3: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_wgsl_validates_with_the_terrain_shader() {
        let source = shader_source();
        let module = wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
    }

    #[test]
    fn the_off_path_module_is_terrain_wgsl_unchanged() {
        let original = include_str!("terrain.wgsl");
        assert!(shader_source().contains(original));
        for entry in ["vs_pack(", "fs_pack_", "vs_shadow", "TerrainPackUniform"] {
            assert!(
                !original.contains(entry),
                "terrain.wgsl must not name {entry}"
            );
        }
    }

    #[test]
    fn material_codes_are_distinct_and_never_zero() {
        let codes = [material::PLANT, material::LEAVES, material::WATER];
        for (index, code) in codes.iter().enumerate() {
            assert_ne!(*code, 0, "zero is no material");
            assert!(!codes[index + 1..].contains(code), "{code} twice");
        }
        // No material (stone under a pack, or no pack): both spare bytes stay zero, whatever the
        // vertex's height.
        let none = PackQuad {
            flat_shade: true,
            material: 0,
            half: 0,
        };
        for y in [0.0, 0.5, 1.0] {
            assert_eq!(none.light_meta(y), [0, 0]);
            assert_eq!(PackQuad::VANILLA.light_meta(y), [0, 0]);
        }
    }
}
