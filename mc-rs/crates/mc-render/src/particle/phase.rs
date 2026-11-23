//! The `PARTICLES` render phase: texture upload, per-frame batching, and the wgpu draw calls.
//!
//! java: `client/particle/EffectRenderer.renderParticles`/`renderLitParticles` — vanilla's GL
//! state dance (bind texture, `depthMask`, draw, repeat per layer) becomes two pipelines
//! (depth-write on/off, matching the alpha-bucket split) drawn once per non-empty
//! (texture, bucket) group, with no fixed-function GL state.
//!
//! # The two-step contract
//!
//! [`crate::pipeline::RenderPhase::record`]'s signature is fixed by S10 — `(gpu, enc, view,
//! cam)`, no `partial_ticks`, no world, no particle list. So, exactly like S12's `EntityPhase`
//! (`begin`/`push`/`finish`), building this frame's geometry is a **separate** step the game loop
//! runs before [`crate::pipeline::PhaseList::render`]: call [`ParticlePhase::upload`] once per
//! frame with the live particle slice, the camera, `partial_ticks`, and the world (for per-particle
//! light); `record` then just replays whatever batches that produced.

// These parity checks mirror Java's remainder expressions. `is_multiple_of` is clearer for
// ordinary non-negative ages, but keeping `%` preserves the exact source semantics at boundaries.
#![allow(clippy::manual_is_multiple_of)]

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use crate::context::{Gpu, DEPTH_FORMAT};
use crate::frame::Camera;
use crate::pipeline::{PassTarget, RenderPhase};

use mc_assets::AtlasAnimationUpload;

use super::billboard::{self, ParticleVertex};
use super::types::{AlphaBucket, FxLayer, Particle, ParticleKind};

use glam::DVec3;
use mc_math::BlockPos;
use mc_world::world::BiomeSource;
use mc_world::World;

use crate::sky::lightmap::{Lightmap, LightmapParams};

const PARTICLE_WGSL: &str = r#"
struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen:     vec4<f32>,
    frame:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;
@group(2) @binding(0) var lightmap_tex: texture_2d<f32>;
@group(2) @binding(1) var lightmap_samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

// Java's particle quads carry OpenGL lightmap coordinates (sky/block * 16), and the active
// EntityRenderer.lightmapTexture is a clamped, linearly-filtered 16x16 table. The CPU stores
// those coordinates normalized to 0..1 in ParticleVertex, so put them back at texel centres
// here. This preserves mixed sky/block colour instead of collapsing both channels through a
// max() scalar curve. Vertex shaders cannot use implicit-LOD `textureSample`, so the explicit
// level-0 form is intentional: it is legal in the vertex stage while retaining the active
// lightmap sampler's clamp-to-edge + linear filtering semantics.
fn particle_lightmap(light: vec2<f32>) -> vec3<f32> {
    let uv = vec2<f32>(
        (light.y * 240.0 + 8.0) / 256.0,
        (light.x * 240.0 + 8.0) / 256.0,
    );
    return textureSampleLevel(lightmap_tex, lightmap_samp, uv, 0.0).rgb;
}

@vertex
fn vs(
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) light: vec2<f32>,
    @location(4) lightmap_mode: f32,
) -> VsOut {
    var out: VsOut;
    out.clip = camera.view_proj * vec4<f32>(pos, 1.0);
    out.uv = uv;
    if (lightmap_mode > 0.5) {
        out.color = color;
    } else {
        out.color = vec4<f32>(color.rgb * particle_lightmap(light), color.a);
    }
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv);
    let a = texel.a * in.color.a;
    // java: GlStateManager.alphaFunc(516 /* GL_GREATER */, 0.003921569F /* 1/255 */)
    if (a <= 0.003921569) {
        discard;
    }
    return vec4<f32>(texel.rgb * in.color.rgb, a);
}

// java: `EffectRenderer.renderLitParticles` (layer 3) runs from `EntityRenderer:1443`, six lines
// after `GlStateManager.disableBlend()` at `:1437`, and sets no alpha func of its own — so it
// inherits the `alphaFunc(516, 0.1F)` the previous `renderParticles` left. Alpha is a cutout
// here, never a blend weight.
@fragment
fn fs_lit(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv);
    let a = texel.a * in.color.a;
    if (a <= 0.1) {
        discard;
    }
    return vec4<f32>(texel.rgb * in.color.rgb, 1.0);
}
"#;

/// Which atlas a batch of particles draws from. java: `EffectRenderer.renderParticles`'s
/// `bindTexture` switch (`i == 0/1`) plus the separate `renderLitParticles` texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TextureSlot {
    Misc,
    Terrain,
    LitExplosion,
    Footprint,
}

fn slot_of(layer: FxLayer) -> TextureSlot {
    match layer {
        FxLayer::Misc => TextureSlot::Misc,
        FxLayer::Terrain => TextureSlot::Terrain,
        FxLayer::LitExplosion => TextureSlot::LitExplosion,
        FxLayer::Footprint => TextureSlot::Footprint,
    }
}

/// Whether this particle family emits billboard vertices. `EntityHugeExplodeFX` is an invisible
/// controller: its empty `renderParticle` exists only to let `onUpdate` spawn six large-explosion
/// children per tick.
#[must_use]
fn emits_billboard(p: &Particle) -> bool {
    match &p.kind {
        ParticleKind::HugeExplode { .. } => false,
        // `MobAppearance.renderParticle` submits an elder-guardian entity through its own
        // renderer. [`super::appearance`] exposes that entity-phase handoff; it is never a
        // particle billboard or explosion-sheet quad.
        ParticleKind::MobAppearance => false,
        ParticleKind::FireworkSpark { twinkle: true, .. } => {
            p.age < p.max_age / 3 || (p.age + p.max_age) / 3 % 2 == 0
        }
        _ => true,
    }
}

struct TextureEntry {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
    /// Raw RGBA8 bytes for level zero and each resident mip. Java's anaglyph transform runs on
    /// the temporary upload buffer, so retaining these source levels lets a live toggle replay
    /// every GPU copy without applying the integer transform twice.
    /// Shared by the owners of the shared atlas.
    source_levels: std::rc::Rc<RefCell<Vec<Vec<u8>>>>,
}

fn mip_dimensions(width: u32, height: u32, mip_level: u32) -> (u32, u32) {
    (
        width.checked_shr(mip_level).unwrap_or(0).max(1),
        height.checked_shr(mip_level).unwrap_or(0).max(1),
    )
}

fn atlas_animation_rect_valid(
    upload: &AtlasAnimationUpload,
    width: u32,
    height: u32,
    mip_level_count: u32,
) -> bool {
    if upload.mip_level >= mip_level_count
        || upload.width == 0
        || upload.height == 0
        || upload.rgba.len() != (u64::from(upload.width) * u64::from(upload.height) * 4) as usize
    {
        return false;
    }
    let (level_width, level_height) = mip_dimensions(width, height, upload.mip_level);
    upload
        .x
        .checked_add(upload.width)
        .is_some_and(|end| end <= level_width)
        && upload
            .y
            .checked_add(upload.height)
            .is_some_and(|end| end <= level_height)
}

/// Reads one PNG out of the pack stack and returns `(rgba8, width, height)`.
///
/// Mirrors `crate::sky::textures::load`'s decode step rather than reusing it: that helper
/// uploads a `wgpu::Texture` of its own with its own bind-group layout, whereas
/// [`ParticlePhase::install`] wants the raw pixels so both pack textures and the stitched block
/// atlas travel the same path.
fn decode_rgba(
    rm: &mc_assets::ResourceManager,
    path: &str,
) -> Result<(Vec<u8>, u32, u32), mc_assets::AssetError> {
    let loc = mc_assets::ResourceLocation::parse(path)?;
    let bytes = rm.load(&loc)?;
    // Refused (missing) when the device cannot hold it.
    let img = mc_assets::decode_texture_png(&bytes)
        .map_err(|e| mc_assets::AssetError::Image {
            location: loc.to_string(),
            message: e.to_string(),
        })?
        .into_rgba8();
    let (w, h) = img.dimensions();
    Ok((img.into_raw(), w, h))
}

/// Build the temporary RGBA8 upload array used by Java's `TextureUtil.copyToBufferPos`.
///
/// Keeping this one boundary helper shared by pack textures, stitched atlas levels, and dirty
/// rectangles makes the integer channel weights and alpha preservation impossible to drift.
fn particle_upload_rgba(source_rgba: &[u8], anaglyph: bool) -> Vec<u8> {
    mc_assets::sprite::anaglyph_rgba(source_rgba, anaglyph)
}

const LIGHTMAP_SIDE: u32 = 16;

fn upload_lightmap_texture(gpu: &Gpu, texture: &wgpu::Texture, rgba: &[u8]) {
    assert_eq!(
        rgba.len(),
        (LIGHTMAP_SIDE * LIGHTMAP_SIDE * 4) as usize,
        "invalid particle lightmap payload"
    );
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(LIGHTMAP_SIDE * 4),
            rows_per_image: Some(LIGHTMAP_SIDE),
        },
        wgpu::Extent3d {
            width: LIGHTMAP_SIDE,
            height: LIGHTMAP_SIDE,
            depth_or_array_layers: 1,
        },
    );
}

fn create_lightmap_texture(gpu: &Gpu, rgba: &[u8], anaglyph: bool) -> wgpu::Texture {
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mc-render particle lightmap"),
        size: wgpu::Extent3d {
            width: LIGHTMAP_SIDE,
            height: LIGHTMAP_SIDE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let upload_rgba = particle_upload_rgba(rgba, anaglyph);
    upload_lightmap_texture(gpu, &texture, &upload_rgba);
    texture
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DrawBatch {
    slot: TextureSlot,
    depth_write: bool,
    start: u32,
    end: u32,
}

/// Return the raw packed light channels consumed by one particle's renderer.
///
/// Java has two intentionally different sample paths here:
///
/// * `EntityFootStepFX.renderParticle` calls `World.getLightBrightness(new BlockPos(this))`, so
///   its scalar is based on the particle's current block, not `Entity.getEyeHeight()`.
/// * `EntityDiggingFX.getBrightnessForRender` inherits the eye-cell packed sample, but returns
///   the source cell's packed light when that primary result is zero.  `Entity` returns zero for
///   an unloaded primary cell, and the digging override only reads `sourcePos` when that source
///   block is loaded.
///
/// Other particle families retain the phase's existing generic eye-cell path. Keeping this
/// dispatch here, before `billboard::quad`, means the vertex still carries the two packed
/// coordinates (`[sky, block]`) instead of collapsing them into a single CPU brightness.
///
/// Production uploads use [`probe_light_samples`], which answers the same questions from one
/// cached column lookup; this three-probe form is kept as its parity oracle.
#[cfg(test)]
#[must_use]
fn particle_light_samples(world: &World, p: &Particle) -> (u8, u8) {
    let current_cell = BlockPos::from_vec3(p.pos);
    match &p.kind {
        ParticleKind::Footstep => (
            world.sky_light_at(current_cell),
            world.block_light_at(current_cell),
        ),
        ParticleKind::Digging { source_pos, .. } => {
            // java: Entity.getBrightnessForRender returns literal 0 when the eye block is not
            // loaded. World::sky_light_at intentionally returns the EmptyChunk sky default (15)
            // for an absent overworld column, so do the Java loaded check before reading it.
            let eye_cell =
                BlockPos::from_vec3(p.pos + DVec3::new(0.0, f64::from(p.size.1) * 0.85, 0.0));
            let eye_loaded =
                (0..256).contains(&eye_cell.y) && world.is_chunk_loaded(eye_cell.chunk_pos());
            let (sky, block) = if eye_loaded {
                (world.sky_light_at(eye_cell), world.block_light_at(eye_cell))
            } else {
                (0, 0)
            };
            if sky == 0 && block == 0 {
                // java: EntityDiggingFX checks World.isBlockLoaded(sourcePos) before
                // getCombinedLight(sourcePos, 0); never borrow an EmptyChunk's provisional sky.
                let source_loaded = (0..256).contains(&source_pos.y)
                    && world.is_chunk_loaded(source_pos.chunk_pos());
                if source_loaded {
                    return (
                        world.sky_light_at(*source_pos),
                        world.block_light_at(*source_pos),
                    );
                }
            }
            (sky, block)
        }
        _ => {
            let eye_cell =
                BlockPos::from_vec3(p.pos + DVec3::new(0.0, f64::from(p.size.1) * 0.85, 0.0));
            // java: Entity.getBrightnessForRender returns literal 0 when the eye cell is not
            // loaded. World::sky_light_at intentionally exposes EmptyChunk's provisional sky
            // value (15), so gate the raw channels before the generic path consumes them.
            if (0..256).contains(&eye_cell.y) && world.is_chunk_loaded(eye_cell.chunk_pos()) {
                (world.sky_light_at(eye_cell), world.block_light_at(eye_cell))
            } else {
                (0, 0)
            }
        }
    }
}

/// Per-upload chunk-column cache for the particle light probes (PART-3).
///
/// `upload` borrows the world immutably for its whole duration, so a column resolved once is
/// valid for every particle of this frame; nothing is carried across frames, where light can
/// change (vanilla also reads world light at render time). Each method reproduces the exact
/// `World::sky_light_at` / `block_light_at` / `is_chunk_loaded` answers, so the samples are
/// bit-identical to `particle_light_samples` (the test oracle) while costing at most one
/// chunk-map lookup per distinct column instead of three per particle.
struct LightProbe<'w> {
    world: &'w World,
    has_sky: bool,
    /// Direct-mapped on the low bit of each chunk coordinate, so any 2x2 group of columns (a
    /// burst straddling a chunk corner) stays resident.
    slots: [Option<(mc_math::ChunkPos, Option<&'w mc_world::Chunk>)>; 4],
}

impl<'w> LightProbe<'w> {
    fn new(world: &'w World) -> Self {
        Self {
            world,
            has_sky: world.has_sky_light(),
            slots: [None; 4],
        }
    }

    fn chunk(&mut self, pos: mc_math::ChunkPos) -> Option<&'w mc_world::Chunk> {
        let slot = ((pos.x & 1) | ((pos.z & 1) << 1)) as usize;
        if let Some((cached, chunk)) = self.slots[slot] {
            if cached == pos {
                return chunk;
            }
        }
        let chunk = self.world.chunk(pos);
        self.slots[slot] = Some((pos, chunk));
        chunk
    }

    /// `(sky_light_at(p), block_light_at(p))` for `p.y` in `0..256` inside a resolved column.
    fn levels_in(&self, chunk: &mc_world::Chunk, p: BlockPos) -> (u8, u8) {
        let (x, z) = ((p.x & 15) as u8, (p.z & 15) as u8);
        let ly = (p.y & 15) as u8;
        let section = chunk.section((p.y >> 4) as u8);
        let sky = if !self.has_sky {
            0
        } else {
            match section {
                Some(section) => section.sky_light(x, ly, z),
                None if chunk.can_see_sky(x, p.y, z) => 15,
                None => 0,
            }
        };
        let block = section.map_or(0, |section| section.block_light(x, ly, z));
        (sky, block)
    }

    /// `(world.sky_light_at(p), world.block_light_at(p))`, including the unloaded-column sky
    /// default of 15 — the footstep path, which has no loaded gate.
    fn raw(&mut self, p: BlockPos) -> (u8, u8) {
        if !(0..256).contains(&p.y) {
            return (0, 0);
        }
        match self.chunk(p.chunk_pos()) {
            Some(chunk) => self.levels_in(chunk, p),
            None => (if self.has_sky { 15 } else { 0 }, 0),
        }
    }

    /// `Entity.getBrightnessForRender`'s gate: `(0, 0)` unless `p` is in range and loaded,
    /// then the raw channels.
    fn loaded(&mut self, p: BlockPos) -> (u8, u8) {
        if !(0..256).contains(&p.y) {
            return (0, 0);
        }
        match self.chunk(p.chunk_pos()) {
            Some(chunk) => self.levels_in(chunk, p),
            None => (0, 0),
        }
    }
}

/// `particle_light_samples` (now the test oracle) answered through a [`LightProbe`]; same three
/// Java paths.
#[must_use]
fn probe_light_samples(probe: &mut LightProbe<'_>, p: &Particle) -> (u8, u8) {
    match &p.kind {
        ParticleKind::Footstep => probe.raw(BlockPos::from_vec3(p.pos)),
        ParticleKind::Digging { source_pos, .. } => {
            let eye_cell =
                BlockPos::from_vec3(p.pos + DVec3::new(0.0, f64::from(p.size.1) * 0.85, 0.0));
            let (sky, block) = probe.loaded(eye_cell);
            if sky == 0 && block == 0 {
                // An unloaded source answers (0, 0), which is also the fall-through value.
                return probe.loaded(*source_pos);
            }
            (sky, block)
        }
        _ => {
            let eye_cell =
                BlockPos::from_vec3(p.pos + DVec3::new(0.0, f64::from(p.size.1) * 0.85, 0.0));
            probe.loaded(eye_cell)
        }
    }
}

/// The six draw groups in vanilla's draw order (see `upload`): `(slot, depth_write)`.
const DRAW_GROUPS: [(TextureSlot, bool); 6] = [
    (TextureSlot::LitExplosion, true),
    (TextureSlot::Footprint, true),
    (TextureSlot::Misc, false),
    (TextureSlot::Misc, true),
    (TextureSlot::Terrain, false),
    (TextureSlot::Terrain, true),
];

/// The shared quad index list: `[4q, 4q+1, 4q+2, 4q, 4q+2, 4q+3]` for `q` in `0..quads` —
/// the same two triangles, in the same order, that the non-indexed upload spelled out.
fn quad_indices(quads: u32) -> Vec<u32> {
    (0..quads)
        .flat_map(|q| {
            let base = q * 4;
            [base, base + 1, base + 2, base, base + 2, base + 3]
        })
        .collect()
}

/// Which [`DRAW_GROUPS`] entry draws `p`, or `None` — exactly the predicates the old six filter
/// passes applied: billboard families only; layer 3 ignores the alpha bucket; a footprint is
/// only drawn from the opaque bucket (its translucent bucket has no pass).
fn draw_group(p: &Particle) -> Option<usize> {
    if !emits_billboard(p) {
        return None;
    }
    let opaque = p.alpha_bucket() == AlphaBucket::Opaque;
    match slot_of(p.layer) {
        TextureSlot::LitExplosion => Some(0),
        TextureSlot::Footprint => opaque.then_some(1),
        TextureSlot::Misc => Some(if opaque { 3 } else { 2 }),
        TextureSlot::Terrain => Some(if opaque { 5 } else { 4 }),
    }
}

/// Conservative clip-volume test against the matrix the GPU draws particles with (PART-2).
///
/// A quad whose every corner lies strictly outside one clip plane rasterises no fragment (no
/// colour, no depth), so skipping it cannot change a pixel, and skipping preserves the relative
/// order of everything still drawn. The test is against a sphere that contains all four
/// corners (`billboard::placement`), inflated by 0.1% plus 1 cm to absorb f32 rounding in
/// both this evaluation and the GPU's.
struct ParticleFrustum {
    /// `xyz·p + w >= 0` inside, for camera-relative `p`, with `|xyz|` precomputed.
    planes: [(glam::Vec4, f32); 6],
}

impl ParticleFrustum {
    /// The camera is the upload handoff copy: its `view.x_axis.w` carries the F5-mode tag
    /// (`billboard::particle_camera_handoff`). The real view is affine (rotations, translations
    /// and scales only), so that element is exactly `0.0` in the camera the uniform is built
    /// from (`CameraUniform::new` -> `Camera::view_proj`).
    fn from_camera(cam: &Camera) -> Self {
        let mut view = cam.view;
        view.x_axis.w = 0.0;
        let m = cam.proj * view;
        let (r0, r1, r2, r3) = (m.row(0), m.row(1), m.row(2), m.row(3));
        // wgpu clip volume: -w <= x <= w, -w <= y <= w, 0 <= z <= w.
        let planes = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2]
            .map(|plane| (plane, plane.truncate().length()));
        Self { planes }
    }

    fn may_be_visible(&self, center: glam::Vec3, radius: f32) -> bool {
        let slack = radius * 1.001 + 0.01;
        // NaN anywhere compares false and keeps the particle, as before.
        !self
            .planes
            .iter()
            .any(|(plane, norm)| plane.truncate().dot(center) + plane.w < -slack * norm)
    }
}

/// The CPU half of [`ParticlePhase::upload`]: fills `verts` (4 per quad) and `batches` (index
/// ranges, 6 per quad) and returns the quad count. Free of GPU state so its output can be
/// compared against the pre-PART-2 six-pass build in tests.
#[allow(clippy::too_many_arguments)]
fn build_particle_quads(
    particles: &[Particle],
    cam: &Camera,
    partial_ticks: f32,
    world: &World,
    skylight_subtracted: u8,
    frustum_culling: bool,
    group_members: &mut [Vec<u32>; 6],
    verts: &mut Vec<ParticleVertex>,
    batches: &mut Vec<DrawBatch>,
) -> u32 {
    verts.clear();
    batches.clear();

    // One classification pass, then one build per draw group (PART-2). This replaces six
    // filter passes (3 texture slots x 2 alpha buckets) over the whole arena with one, and
    // yields exactly the same runs: each group's members in arena order — the same "group by
    // draw state, not by spawn order" approach EntityPhase uses, and for the same reason:
    // one draw call per draw state instead of a full sort.
    //
    // The **group order** is vanilla's, and it is load-bearing rather than cosmetic because
    // layer 3 now writes depth opaquely: `EntityRenderer:1443` calls `renderLitParticles`
    // (layer 3) *before* `:1446`'s `renderParticles`, and `renderParticles` is layer-major
    // with the no-depth-write bucket first inside each layer
    // (`EffectRenderer.renderParticles:253-268`). Layer 3 is one group because
    // `renderLitParticles` iterates both of its bucket lists without touching `depthMask`;
    // footsteps retain layer 3's depth write but enable alpha blending themselves.
    let basis = billboard::QuadBasis::new(cam);
    let frustum = frustum_culling.then(|| ParticleFrustum::from_camera(cam));
    for members in group_members.iter_mut() {
        members.clear();
    }
    for (index, p) in particles.iter().enumerate() {
        if let Some(group) = draw_group(p) {
            group_members[group].push(index as u32);
        }
    }

    let mut light = LightProbe::new(world);
    let mut quads = 0u32;
    for (group, &(slot, depth_write)) in DRAW_GROUPS.iter().enumerate() {
        let start = quads;
        for &index in &group_members[group] {
            let p = &particles[index as usize];
            // Centre and half-size once, shared by the frustum test and the quad itself.
            let placed = billboard::placement(p, cam, partial_ticks, &basis);
            if let Some(frustum) = &frustum {
                if !frustum.may_be_visible(placed.center, placed.radius) {
                    continue;
                }
            }
            // `EntityFX`'s height is `0.2 * scale`, so this is a fraction of a block —
            // but it is the difference between reading the cell a particle is standing
            // in and the one it is standing *under*, which is where a torch's light is.
            let (sky, block) = probe_light_samples(&mut light, p);
            // java: `EntityDiggingFX.getBrightnessForRender` — `return i == 0 ? j : i`,
            // The sample is now converted to the vertex's normalized packed channels;
            // the WGSL stage applies the shared provider light curve.
            let (sky, block) = (f32::from(sky), f32::from(block));
            let mut quad = billboard::quad_at(p, cam, partial_ticks, sky, block, &basis, &placed);
            if matches!(p.kind, ParticleKind::Footstep) {
                // Footsteps supply a scalar vertex colour in Java and do not consume the
                // active packed lightmap texture. Keep the packed sample in `light` for
                // inspection/debugging, but mark this vertex path so WGSL does not apply
                // a second multiplier.
                let brightness = footstep_brightness(world, p, skylight_subtracted);
                for vertex in &mut quad {
                    vertex.color[0] = brightness;
                    vertex.color[1] = brightness;
                    vertex.color[2] = brightness;
                    vertex.lightmap_mode = 1.0;
                }
            }
            // Four vertices; the index buffer draws (0,1,2) and (0,2,3), `quad`'s doc'd order.
            verts.extend_from_slice(&quad);
            quads += 1;
        }
        if quads != start {
            batches.push(DrawBatch {
                slot,
                depth_write,
                start: start * 6,
                end: quads * 6,
            });
        }
    }

    quads
}

/// `EntityFootStepFX.renderParticle` bypasses `getBrightnessForRender` and writes the provider's
/// scalar `World.getLightBrightness(BlockPos)` directly into RGB. That value is the dimension's
/// 16-entry brightness table, not the dynamic RGB lightmap used by ordinary particles.
#[must_use]
fn footstep_brightness(world: &World, p: &Particle, skylight_subtracted: u8) -> f32 {
    let ambient = if matches!(world.biome_source(), BiomeSource::Nether) {
        0.1
    } else {
        0.0
    };
    // Java's `World.getLightBrightness` delegates to `getLightFromNeighbors`, which uses
    // `getLight(pos, true)` with the world's current sky subtraction. This differs from the raw
    // `light_at` shortcut: neighbor-brightness blocks (slabs/stairs/etc.) take the maximum of
    // the five adjacent cells, and an unloaded EmptyChunk returns zero for this query.
    crate::sky::lightmap::light_brightness_table(ambient)
        [usize::from(world.light_subtracted(BlockPos::from_vec3(p.pos), skylight_subtracted))]
}

/// The `PARTICLES` phase. java: the particle half of `EffectRenderer`.
pub struct ParticlePhase {
    pipeline_translucent: wgpu::RenderPipeline,
    pipeline_opaque: wgpu::RenderPipeline,
    /// java: layer 3's `renderLitParticles` pass — blending off, 0.1 alpha cutout.
    pipeline_lit: wgpu::RenderPipeline,
    pipeline_footstep: wgpu::RenderPipeline,
    texture_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    lightmap_texture: wgpu::Texture,
    lightmap_bind_group: wgpu::BindGroup,
    /// Raw bytes for the current Java `EntityRenderer.lightmapTexture`. Keeping this source
    /// lets anaglyph toggles replay the exact table without transforming it twice.
    lightmap_source_rgba: Vec<u8>,
    /// What the lightmap texture currently holds, when it came from parameters.
    uploaded_lightmap_params: Option<LightmapParams>,
    textures: HashMap<TextureSlot, TextureEntry>,
    /// Java's `GameSettings.anaglyph`, applied at every particle texture upload boundary.
    anaglyph: bool,
    /// Four vertices per particle quad, in draw order; the shared `index_buffer` expands each
    /// quad to vanilla's two triangles `(0,1,2)`/`(0,2,3)`.
    verts: Vec<ParticleVertex>,
    /// `DrawBatch::start/end` are *index* ranges (6 per particle), which are exactly the
    /// triangle-expanded vertex ranges the non-indexed upload used to record.
    batches: Vec<DrawBatch>,
    buffer: Option<wgpu::Buffer>,
    capacity: u64,
    /// Grow-only `u32` quad index list, `[4q, 4q+1, 4q+2, 4q, 4q+2, 4q+3]` per quad `q`.
    index_buffer: Option<wgpu::Buffer>,
    index_capacity_quads: u32,
    /// Per-upload scratch: arena indices of the particles in each [`DRAW_GROUPS`] entry.
    group_members: [Vec<u32>; 6],
    /// Skip particles entirely outside the view volume (pixel-identical; see
    /// [`ParticleFrustum`]). Off by default so direct callers whose `Camera` is not the one the
    /// frame renders with keep the old "build everything" behaviour; the client enables it.
    frustum_culling: bool,
    uploaded: u32,
    enabled: bool,
    /// Slots [`RenderPhase::record`] has already complained about. Dropping a batch because its
    /// atlas was never installed used to be completely silent, which is how the misc and
    /// explosion atlases stayed uninstalled for a whole port cycle without anyone noticing
    /// — every non-terrain particle was built, ticked, batched and then
    /// thrown away at draw time. One warning per slot per phase, not one per frame.
    warned_missing: HashSet<TextureSlot>,
    /// Tick-domain World.skylightSubtracted, used by scalar-lit footprints only.
    skylight_subtracted: u8,
}

impl ParticlePhase {
    /// Builds the pipelines. No texture is resident yet — call
    /// [`ParticlePhase::install_misc_atlas`] etc. before the first [`ParticlePhase::upload`].
    #[must_use]
    pub fn new(gpu: &Gpu) -> Self {
        Self::new_with_anaglyph(gpu, false)
    }

    /// Builds the pipelines with Java's global upload-time anaglyph switch selected.
    ///
    /// The ordinary [`Self::new`] constructor remains disabled for compatibility. A client that
    /// owns `GameSettings` can seed the phase here, or call [`Self::set_anaglyph`] before its
    /// first resource-backed upload.
    #[must_use]
    pub fn new_with_anaglyph(gpu: &Gpu, anaglyph: bool) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mc-render particle"),
                source: wgpu::ShaderSource::Wgsl(PARTICLE_WGSL.into()),
            });
        let texture_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("mc-render particle texture"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                            count: None,
                        },
                    ],
                });
        let lightmap_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("mc-render particle lightmap"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mc-render particle layout"),
                bind_group_layouts: &[
                    Some(gpu.camera_layout()),
                    Some(&texture_layout),
                    Some(&lightmap_layout),
                ],
                immediate_size: 0,
            });

        // `blended = false` is layer 3's pipeline: `renderLitParticles` draws with blending off
        // and a 0.1 alpha cutout. `F-P2-client.particle-01-6`.
        let make_pipeline = |depth_write: bool, blended: bool| {
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(if !blended {
                        "mc-render particle pipeline (lit, unblended)"
                    } else if depth_write {
                        "mc-render particle pipeline (opaque)"
                    } else {
                        "mc-render particle pipeline (translucent)"
                    }),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs"),
                        compilation_options: Default::default(),
                        buffers: &[Some(ParticleVertex::layout())],
                    },
                    // java: no `GlStateManager.enableCull`/`disableCull` call around particle
                    // rendering — vanilla draws with whatever cull state the previous phase left,
                    // which by the time particles render is disabled. Two-sided is also simply
                    // correct for a billboard: its winding depends on the camera-facing basis, not
                    // on world geometry.
                    primitive: wgpu::PrimitiveState {
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        // java: `GlStateManager.depthMask(false/true)` per alpha bucket. Depth
                        // *testing* stays on for both (vanilla never disables `GL_DEPTH_TEST` here).
                        depth_write_enabled: Some(depth_write),
                        depth_compare: Some(wgpu::CompareFunction::Less),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: gpu.sample_count(),
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(if blended { "fs" } else { "fs_lit" }),
                        compilation_options: Default::default(),
                        // java: `GlStateManager.blendFunc(770 /* SRC_ALPHA */, 771 /* ONE_MINUS_SRC_ALPHA */)`
                        // for layers 0-2; layer 3 runs with `disableBlend()` still in effect.
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.config.format,
                            blend: blended.then_some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        };

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mc-render particle sampler"),
            // Vanilla's particle atlas is point-sampled, same as every other 1.8.9 texture.
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            // GL's default wrap is `GL_REPEAT`, and 1.8.9 never sets `GL_CLAMP_TO_EDGE` on
            // `particles.png`. That is load-bearing exactly once, and systematically:
            // `setParticleTextureIndex(7 - age*8/maxAge)` is `-1` on the last live tick of every
            // animated family, which Java's truncating `%`//` stores as column `-1`, row `0`.
            // Vanilla's `u ∈ [-0.0625, -0.0000625]` then *wraps* to cell (15, 0); `wgpu`'s
            // default `ClampToEdge` instead pins it to the leftmost texel and smears a
            // one-pixel column across the quad. `F-P2-client.particle-SWEEP-12`.
            //
            // Safe for the terrain and explosion atlases too: every other UV this crate emits is
            // inside `[0, 1]`, where `Repeat` and `ClampToEdge` are identical.
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        let lightmap = Lightmap::generate(LightmapParams::default());
        let lightmap_source_rgba = lightmap.rgba.clone();
        let lightmap_texture = create_lightmap_texture(gpu, &lightmap_source_rgba, anaglyph);
        let lightmap_view = lightmap_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let lightmap_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mc-render particle lightmap sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let lightmap_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mc-render particle lightmap bind group"),
            layout: &lightmap_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&lightmap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&lightmap_sampler),
                },
            ],
        });

        Self {
            pipeline_translucent: make_pipeline(false, true),
            pipeline_opaque: make_pipeline(true, true),
            pipeline_lit: make_pipeline(true, false),
            // EntityFootStepFX is nominally layer 3 but enables alpha blending itself while
            // retaining that layer's depth writes; the normal layer-3 pipeline is a cutout.
            pipeline_footstep: make_pipeline(true, true),
            texture_layout,
            sampler,
            lightmap_texture,
            lightmap_bind_group,
            lightmap_source_rgba,
            uploaded_lightmap_params: None,
            textures: HashMap::new(),
            anaglyph,
            verts: Vec::new(),
            batches: Vec::new(),
            buffer: None,
            capacity: 0,
            index_buffer: None,
            index_capacity_quads: 0,
            group_members: Default::default(),
            frustum_culling: false,
            uploaded: 0,
            enabled: true,
            warned_missing: HashSet::new(),
            skylight_subtracted: 0,
        }
    }

    /// Enables the conservative frustum test in [`Self::upload`]. Only enable it when the
    /// `cam` passed to `upload` has the same `proj`/`view` (apart from the particle handoff's
    /// `x_axis.w` tag) as the camera the frame is recorded with — true for the client, whose
    /// `upload_particles` runs after `place_camera` on the same camera the shell renders.
    pub fn set_frustum_culling(&mut self, on: bool) {
        self.frustum_culling = on;
    }

    fn install(
        &mut self,
        gpu: &Gpu,
        slot: TextureSlot,
        label: &str,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) {
        self.install_levels(gpu, slot, label, vec![rgba.to_vec()], width, height);
    }

    fn install_levels(
        &mut self,
        gpu: &Gpu,
        slot: TextureSlot,
        label: &str,
        source_levels: Vec<Vec<u8>>,
        width: u32,
        height: u32,
    ) {
        assert!(
            width > 0 && height > 0,
            "{label} must have non-zero dimensions"
        );
        assert!(
            !source_levels.is_empty(),
            "{label} must have a base mip level"
        );
        for (level, rgba) in source_levels.iter().enumerate() {
            let level = level as u32;
            let (level_width, level_height) = mip_dimensions(width, height, level);
            assert_eq!(
                rgba.len(),
                (u64::from(level_width) * u64::from(level_height) * 4) as usize,
                "{label} mip {level} is {} bytes, expected {level_width}x{level_height}x4",
                rgba.len()
            );
        }
        // Inside the client's install sweep the block atlas slot adopts
        // the one shared GPU atlas; every other slot (and any use outside it) is a plain create.
        let texture = crate::atlas_transfer::share_atlas_texture(
            gpu,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: source_levels.len() as u32,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // Deliberately *not* sRGB: 1.8.9's fixed-function GL does no colour-space
                // conversion, so an sRGB view would darken every particle (same reasoning S15
                // used for font textures).
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            &source_levels,
            self.anaglyph,
        );
        for (level, source_rgba) in source_levels.iter().enumerate() {
            let level = level as u32;
            let (level_width, level_height) = mip_dimensions(width, height, level);
            let upload_rgba = particle_upload_rgba(source_rgba, self.anaglyph);
            let mut destination = texture.as_image_copy();
            destination.mip_level = level;
            gpu.queue.write_texture(
                destination,
                &upload_rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(level_width * 4),
                    rows_per_image: Some(level_height),
                },
                wgpu::Extent3d {
                    width: level_width,
                    height: level_height,
                    depth_or_array_layers: 1,
                },
            );
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        // One raw shadow for every owner of the shared atlas (only the
        // block atlas slot can hold it; every other slot keeps its own levels as before).
        let source_levels = crate::atlas_transfer::share_atlas_shadow(&texture, source_levels);
        self.textures.insert(
            slot,
            TextureEntry {
                texture,
                bind_group,
                width,
                height,
                source_levels,
            },
        );
    }

    /// `textures/particle/particles.png` — the 16x16 misc-atlas grid smoke/flame/lava/portal/
    /// reddust/crit/bubble/rain/drip/spell/firework-spark all draw from.
    pub fn install_misc_atlas(&mut self, gpu: &Gpu, rgba: &[u8], width: u32, height: u32) {
        self.install(
            gpu,
            TextureSlot::Misc,
            "mc-render particle misc atlas",
            rgba,
            width,
            height,
        );
    }

    /// `textures/entity/explosion.png` — the 4x4 animated grid `EXPLOSION_LARGE` uses.
    pub fn install_explosion_atlas(&mut self, gpu: &Gpu, rgba: &[u8], width: u32, height: u32) {
        self.install(
            gpu,
            TextureSlot::LitExplosion,
            "mc-render particle explosion atlas",
            rgba,
            width,
            height,
        );
    }

    /// `textures/particle/footprint.png` — `EntityFootStepFX`'s standalone ground-quad sprite.
    pub fn install_footprint_texture(&mut self, gpu: &Gpu, rgba: &[u8], width: u32, height: u32) {
        self.install(
            gpu,
            TextureSlot::Footprint,
            "mc-render particle footprint texture",
            rgba,
            width,
            height,
        );
    }

    /// Loads and installs all **pack-supplied** particle textures in one call:
    /// `textures/particle/particles.png` for [`FxLayer::Misc`], `textures/entity/explosion.png`
    /// for [`FxLayer::LitExplosion`], and `textures/particle/footprint.png` for
    /// [`FxLayer::Footprint`].
    ///
    /// java: `EffectRenderer.renderParticles` binds `particleTextures` (`textures/particle/
    /// particles.png`) for `fxLayer` 0/1 and `EntityLargeExplodeFX:17`/`:44` binds
    /// `textures/entity/explosion.png` for layer 3 (`renderLitParticles`). Vanilla binds them
    /// lazily per draw through the `TextureManager`; here they are resident bind groups, so the
    /// binding happens once, at render-stack install.
    ///
    /// The block/item atlas is deliberately **not** loaded here — it is stitched by
    /// `mc_assets::bake_all` and is not a file in any pack, so it still arrives through
    /// [`ParticlePhase::install_block_atlas`].
    ///
    /// Best-effort per texture, matching [`crate::sky::WeatherPhase::load_textures`]: a pack
    /// missing one leaves that slot uninstalled and logs, rather than failing the
    /// whole render-stack build. [`RenderPhase::record`] then warns once for the batches it has
    /// to drop.
    pub fn load_textures(&mut self, gpu: &Gpu, rm: &mc_assets::ResourceManager) {
        for (path, slot, label) in [
            (
                "textures/particle/particles.png",
                TextureSlot::Misc,
                "mc-render particle misc atlas",
            ),
            (
                "textures/entity/explosion.png",
                TextureSlot::LitExplosion,
                "mc-render particle explosion atlas",
            ),
            (
                "textures/particle/footprint.png",
                TextureSlot::Footprint,
                "mc-render particle footprint texture",
            ),
        ] {
            match decode_rgba(rm, path) {
                Ok((rgba, w, h)) => self.install(gpu, slot, label, &rgba, w, h),
                Err(e) => {
                    tracing::warn!(texture = path, error = %e, "particle atlas not loaded");
                }
            }
        }
    }

    /// The block/item atlas, for block-break/block-dust/item-break particles.
    ///
    /// `particle/**` does not itself decide *which* texture that is for a given block or item
    /// (that is the baked-model territory `mc_assets::bake_all`/`model_for` own — S09/S11); a
    /// caller supplies the finished atlas here and resolves each particle's [`super::types::ParticleIcon`]
    /// separately (see [`super::packet::IconLookup`]).
    pub fn install_block_atlas(&mut self, gpu: &Gpu, atlas: &mc_assets::Atlas) {
        let mut source_levels = Vec::with_capacity(1 + atlas.mipmaps.len());
        source_levels.push(atlas.texture.clone());
        source_levels.extend(atlas.mipmaps.iter().cloned());
        self.install_levels(
            gpu,
            TextureSlot::Terrain,
            "mc-render particle block atlas",
            source_levels,
            atlas.w,
            atlas.h,
        );
    }

    /// Apply one S09 animated-atlas dirty rectangle to the particle phase's terrain copy.
    ///
    /// The block atlas is resident with the same sprite-local mip chain as the terrain copy, so
    /// level-specific records are accepted and retained as raw source pixels for a later live
    /// anaglyph toggle. Resource-backed particle textures remain single-level.
    pub fn upload_atlas_animation(&self, gpu: &Gpu, upload: &AtlasAnimationUpload) -> bool {
        self.upload_atlas_animation_with_sink(upload, |texture, anaglyph| {
            let upload_rgba = particle_upload_rgba(&upload.rgba, anaglyph);
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: upload.mip_level,
                    origin: wgpu::Origin3d {
                        x: upload.x,
                        y: upload.y,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &upload_rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(upload.width * 4),
                    rows_per_image: Some(upload.height),
                },
                wgpu::Extent3d {
                    width: upload.width,
                    height: upload.height,
                    depth_or_array_layers: 1,
                },
            );
        })
    }

    /// Experimental transfer seam: validate and retain raw pixels before invoking the sink.
    /// The sink receives this owner's exact texture and current transform setting.
    #[doc(hidden)]
    pub fn upload_atlas_animation_with_sink(
        &self,
        upload: &AtlasAnimationUpload,
        sink: impl FnOnce(&wgpu::Texture, bool),
    ) -> bool {
        let Some(entry) = self.textures.get(&TextureSlot::Terrain) else {
            return false;
        };
        let mip_level_count = entry.source_levels.borrow().len() as u32;
        if !atlas_animation_rect_valid(upload, entry.width, entry.height, mip_level_count) {
            return false;
        }
        let (level_width, _) = mip_dimensions(entry.width, entry.height, upload.mip_level);
        let mut source_levels = entry.source_levels.borrow_mut();
        let source = &mut source_levels[upload.mip_level as usize];
        for row in 0..upload.height {
            let src_start = (row * upload.width * 4) as usize;
            let dst_start = (((upload.y + row) * level_width + upload.x) * 4) as usize;
            let count = (upload.width * 4) as usize;
            source[dst_start..dst_start + count]
                .copy_from_slice(&upload.rgba[src_start..src_start + count]);
        }
        drop(source_levels);
        sink(&entry.texture, self.anaglyph);
        true
    }

    /// Whether this phase applies Java's global upload-time anaglyph transform.
    #[must_use]
    pub const fn anaglyph_enabled(&self) -> bool {
        self.anaglyph
    }

    /// Toggle Java's global transform and rebuild every resident particle texture from raw
    /// source levels. Replaying the retained source avoids applying the integer transform twice,
    /// and also preserves the latest animated frame in every atlas mip.
    pub fn set_anaglyph_with_gpu(&mut self, gpu: &Gpu, enabled: bool) {
        if self.anaglyph == enabled {
            return;
        }
        self.anaglyph = enabled;
        let uploads: Vec<_> = self
            .textures
            .iter()
            .map(|(slot, entry)| {
                (
                    *slot,
                    entry.source_levels.borrow().clone(),
                    entry.width,
                    entry.height,
                )
            })
            .collect();
        for (slot, source_levels, width, height) in uploads {
            let label = match slot {
                TextureSlot::Misc => "mc-render particle misc atlas",
                TextureSlot::Terrain => "mc-render particle block atlas",
                TextureSlot::LitExplosion => "mc-render particle explosion atlas",
                TextureSlot::Footprint => "mc-render particle footprint texture",
            };
            self.install_levels(gpu, slot, label, source_levels, width, height);
        }
        let upload_rgba = particle_upload_rgba(&self.lightmap_source_rgba, self.anaglyph);
        upload_lightmap_texture(gpu, &self.lightmap_texture, &upload_rgba);
    }

    /// Upload the frame's shared Java lightmap table for particle draws.
    ///
    /// Vanilla keeps one `EntityRenderer.lightmapTexture` active while terrain, entities and
    /// particles render. The particle phase owns a compatible binding so callers that already
    /// regenerate the table can hand the same [`Lightmap`] to this phase without falling back to
    /// a scalar max-channel approximation.
    pub fn set_lightmap(&mut self, gpu: &Gpu, lightmap: &Lightmap) {
        assert_eq!(
            lightmap.rgba.len(),
            (LIGHTMAP_SIDE * LIGHTMAP_SIDE * 4) as usize,
            "invalid particle lightmap payload"
        );
        self.uploaded_lightmap_params = None;
        self.lightmap_source_rgba.clone_from(&lightmap.rgba);
        let upload_rgba = particle_upload_rgba(&self.lightmap_source_rgba, self.anaglyph);
        upload_lightmap_texture(gpu, &self.lightmap_texture, &upload_rgba);
    }

    /// Regenerate and upload the shared Java lightmap from the current frame parameters.
    pub fn set_lightmap_params(&mut self, gpu: &Gpu, params: LightmapParams) {
        if self.uploaded_lightmap_params == Some(params) {
            return;
        }
        self.set_lightmap(gpu, &Lightmap::generate_cached(params));
        self.uploaded_lightmap_params = Some(params);
    }

    /// Set the transform for subsequent uploads without touching resident GPU textures.
    /// Use [`Self::set_anaglyph_with_gpu`] when an already-installed texture must change now.
    pub fn set_anaglyph(&mut self, enabled: bool) {
        self.anaglyph = enabled;
    }

    /// Supply the world's tick-domain sky darkening before uploading scalar-lit particles.
    /// Packed lightmap coordinates for ordinary particles remain unmodified.
    pub fn set_skylight_subtracted(&mut self, amount: u8) {
        self.skylight_subtracted = amount.min(15);
    }

    /// Whether every texture slot a set of particles could need has been installed. Cheap sanity
    /// check for callers wiring this phase up.
    #[must_use]
    pub fn has_texture(&self, layer: FxLayer) -> bool {
        self.textures.contains_key(&slot_of(layer))
    }

    /// Builds this frame's vertex buffer and batch list from the live particle set. Call once
    /// per frame, before [`crate::pipeline::PhaseList::render`] (see the module docs).
    ///
    /// `world` supplies each particle's light level (`World::sky_light_at`/`block_light_at`);
    /// look-ups happen once per particle, not once per vertex.
    pub fn upload(
        &mut self,
        gpu: &Gpu,
        particles: &[Particle],
        cam: &Camera,
        partial_ticks: f32,
        world: &World,
    ) {
        let quads = build_particle_quads(
            particles,
            cam,
            partial_ticks,
            world,
            self.skylight_subtracted,
            self.frustum_culling,
            &mut self.group_members,
            &mut self.verts,
            &mut self.batches,
        );
        self.uploaded = quads * 6;
        if self.verts.is_empty() {
            return;
        }
        let needed = std::mem::size_of_val(&self.verts[..]) as u64;
        if self.capacity < needed || self.buffer.is_none() {
            let capacity = needed.next_power_of_two().max(4096);
            self.buffer = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc-render particle vertices"),
                size: capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.capacity = capacity;
        }
        if let Some(buf) = &self.buffer {
            // Per-frame upload ring; the index buffer below is written once, when created.
            gpu.write_buffer_staged(buf, 0, bytemuck::cast_slice(&self.verts));
        }
        if self.index_capacity_quads < quads || self.index_buffer.is_none() {
            let capacity = quads.next_power_of_two().max(1024);
            let indices = quad_indices(capacity);
            let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc-render particle quad indices"),
                size: std::mem::size_of_val(&indices[..]) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(&indices));
            self.index_buffer = Some(buffer);
            self.index_capacity_quads = capacity;
        }
    }

    /// Vertices the last [`ParticlePhase::upload`] draws, triangle-expanded: 6 per particle
    /// (the index count). The vertex buffer itself holds 4 per particle.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.verts.len() / 4 * 6
    }

    /// Draw calls the next [`RenderPhase::record`] will issue.
    #[must_use]
    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    /// Drop the active frame geometry without destroying the reusable GPU allocation.
    ///
    /// java: `EffectRenderer.clearEffects` on a world unload. The client calls this at the
    /// `Minecraft.loadWorld(null)` boundary because a disconnected frame does not have a world
    /// to pass through [`ParticlePhase::upload`]. Keeping the buffer capacity avoids a needless
    /// device allocation when the next world joins, while `uploaded = 0` makes the next record
    /// a no-op until live particles are uploaded again.
    pub fn clear(&mut self) {
        self.verts.clear();
        self.batches.clear();
        self.uploaded = 0;
    }

    /// Turns the phase off without unregistering it (vanilla's minimal-particles-still-draws-
    /// server-driven-ones is handled by [`super::system::ParticleDensity`], not this — this is
    /// the coarser "particles off entirely" escape hatch, e.g. for a screenshot mode).
    pub fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
    }
}

impl RenderPhase for ParticlePhase {
    fn name(&self) -> &'static str {
        "particles"
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        self.record_to(gpu, PassTarget::Own { enc, view });
    }

    // One pass, nothing else on the encoder: it joins the world pass
    // (the draws are untouched).
    fn shares_world_pass(&self) -> bool {
        true
    }

    fn record_into_pass(&mut self, gpu: &Gpu, pass: &mut wgpu::RenderPass<'static>, _cam: &Camera) {
        self.record_to(gpu, PassTarget::Shared(pass));
    }
}

impl ParticlePhase {
    /// The phase's draws, into a pass of its own or the shared world pass.
    fn record_to(&mut self, gpu: &Gpu, target: PassTarget<'_>) {
        if self.buffer.is_none() || self.uploaded == 0 {
            return;
        }
        // The draw loop below used to `continue` past a batch whose slot had no texture
        // without a word, which is how the misc and explosion atlases stayed uninstalled without
        // anyone noticing. Destructured so `warned_missing` can be written while `batches`/
        // `textures` are read (disjoint field borrows); once per slot, not once per frame.
        let Self {
            batches,
            textures,
            warned_missing,
            ..
        } = self;
        for batch in batches.iter() {
            if !textures.contains_key(&batch.slot) && warned_missing.insert(batch.slot) {
                tracing::warn!(
                    slot = ?batch.slot,
                    verts = batch.end - batch.start,
                    "particle batch dropped: no atlas installed for this slot \
                     (call ParticlePhase::load_textures / install_block_atlas)"
                );
            }
        }

        let buffer = self.buffer.as_ref().expect("checked above");
        let Some(indices) = self.index_buffer.as_ref() else {
            return;
        };
        let mut own = None;
        let pass = target.begin(gpu, "particles", &mut own);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.set_bind_group(2, &self.lightmap_bind_group, &[]);
        pass.set_vertex_buffer(0, buffer.slice(..));
        pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
        for batch in &self.batches {
            let Some(entry) = self.textures.get(&batch.slot) else {
                continue;
            };
            pass.set_pipeline(if batch.slot == TextureSlot::LitExplosion {
                &self.pipeline_lit
            } else if batch.slot == TextureSlot::Footprint {
                &self.pipeline_footstep
            } else if batch.depth_write {
                &self.pipeline_opaque
            } else {
                &self.pipeline_translucent
            });
            pass.set_bind_group(1, &entry.bind_group, &[]);
            pass.draw_indexed(batch.start..batch.end, 0, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn white_pixels(w: u32, h: u32) -> Vec<u8> {
        vec![255u8; (w * h * 4) as usize]
    }

    #[test]
    fn particle_shader_samples_mixed_sky_and_block_channels_from_the_lightmap() {
        assert!(PARTICLE_WGSL.contains("textureSampleLevel(lightmap_tex, lightmap_samp, uv, 0.0)"));
        assert!(
            !PARTICLE_WGSL.contains("textureSample(lightmap_tex, lightmap_samp"),
            "implicit-LOD textureSample is fragment-only and invalid in the vertex stage"
        );
        assert!(PARTICLE_WGSL.contains("light.y * 240.0"));
        assert!(PARTICLE_WGSL.contains("light.x * 240.0"));
        assert!(PARTICLE_WGSL.contains("(light.y * 240.0 + 8.0) / 256.0"));
        assert!(PARTICLE_WGSL.contains("(light.x * 240.0 + 8.0) / 256.0"));
        assert!(
            !PARTICLE_WGSL.contains("max(light.x, light.y)"),
            "particle shading must not collapse packed light into one max-channel curve"
        );
    }

    #[test]
    fn footstep_brightness_uses_the_provider_table_and_dimension_ambient_floor() {
        let mut overworld = World::new();
        let mut rng = mc_math::JavaRandom::new(1);
        let p = super::super::spawn::footstep(&mut rng, glam::DVec3::new(0.5, 70.0, 0.5));
        assert_eq!(
            footstep_brightness(&overworld, &p, 0),
            crate::sky::lightmap::light_brightness_table(0.0)[0],
            "Java's unloaded EmptyChunk returns zero to getLightBrightness"
        );

        overworld.set_dimension_id(-1);
        assert_eq!(
            footstep_brightness(&overworld, &p, 11),
            crate::sky::lightmap::light_brightness_table(0.1)[0]
        );
    }

    #[test]
    fn footstep_brightness_uses_java_neighbor_brightness_query() {
        let mut world = World::new();
        let payload = vec![0u8; mc_world::expected_chunk_data_len(1 << 4, true, true)];
        world
            .apply_chunk_data(mc_math::ChunkPos::new(0, 0), true, 1 << 4, &payload)
            .expect("blank resident section");

        let feet = BlockPos::new(0, 70, 0);
        let slab = mc_data::block_by_name("stone_slab").expect("1.8.9 stone_slab");
        world.set_block(feet, mc_data::BlockState::new(slab, 0));
        world.set_light_for(mc_world::SkyBlock::Block, feet.up(), 11);

        let mut rng = mc_math::JavaRandom::new(2);
        let p = super::super::spawn::footstep(&mut rng, glam::DVec3::new(0.5, 70.0, 0.5));
        assert_eq!(
            world.light_subtracted(feet, 0),
            11,
            "slab uses the brightest adjacent cell in Java getLightFromNeighbors"
        );
        assert_eq!(
            footstep_brightness(&world, &p, 0),
            crate::sky::lightmap::light_brightness_table(0.0)[11]
        );
    }

    #[test]
    fn footstep_brightness_subtracts_night_sky_but_preserves_block_light() {
        let mut world = World::new();
        let payload = vec![0u8; mc_world::expected_chunk_data_len(1, true, true)];
        world
            .apply_chunk_data(mc_math::ChunkPos::new(0, 0), true, 1, &payload)
            .unwrap();
        let pos = BlockPos::new(0, 8, 0);
        world.set_light_for(mc_world::SkyBlock::Sky, pos, 15);
        world.set_light_for(mc_world::SkyBlock::Block, pos, 7);
        let mut random = mc_math::JavaRandom::new(3);
        let p = super::super::spawn::footstep(&mut random, glam::DVec3::new(0.5, 8.0, 0.5));
        let table = crate::sky::lightmap::light_brightness_table(0.0);
        assert_eq!(footstep_brightness(&world, &p, 0), table[15]);
        assert_eq!(footstep_brightness(&world, &p, 11), table[7]);
        world.set_light_for(mc_world::SkyBlock::Block, pos, 0);
        assert_eq!(footstep_brightness(&world, &p, 11), table[4]);
        assert_eq!(
            particle_light_samples(&world, &p),
            (15, 0),
            "the packed sky channel must not be darkened twice"
        );
    }

    #[test]
    fn footstep_upload_applies_current_sky_subtraction_without_changing_packed_light() {
        let Ok(gpu) = Gpu::headless(64, 64, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let mut world = World::new();
        let payload = vec![0; mc_world::expected_chunk_data_len(1, true, true)];
        world
            .apply_chunk_data(mc_math::ChunkPos::new(0, 0), true, 1, &payload)
            .unwrap();
        world.set_light_for(mc_world::SkyBlock::Sky, BlockPos::new(0, 8, 0), 15);
        let mut random = mc_math::JavaRandom::new(4);
        let p = super::super::spawn::footstep(&mut random, DVec3::new(0.5, 8.0, 0.5));
        let mut phase = ParticlePhase::new(&gpu);
        phase.install_footprint_texture(&gpu, &white_pixels(16, 16), 16, 16);
        phase.upload(
            &gpu,
            std::slice::from_ref(&p),
            &Camera::default(),
            0.0,
            &world,
        );
        let day = phase.verts[0];
        phase.set_skylight_subtracted(11);
        phase.upload(&gpu, &[p], &Camera::default(), 0.5, &world);
        let night = phase.verts[0];
        assert_eq!(day.color[0], 1.0);
        assert_eq!(
            night.color[0],
            crate::sky::lightmap::light_brightness_table(0.0)[4]
        );
        assert_eq!(day.light, night.light);
        assert_eq!(night.lightmap_mode, 1.0);
        assert_eq!(night.color[0], night.color[1]);
        assert_eq!(night.color[1], night.color[2]);
    }

    #[test]
    fn generic_particle_eye_light_is_zero_until_its_chunk_is_loaded() {
        // World::sky_light_at deliberately reports EmptyChunk's overworld sky default (15), but
        // Java Entity.getBrightnessForRender has an explicit isBlockLoaded gate and returns zero
        // before consulting that default. Keep this CPU-only oracle separate from GPU upload tests
        // so the unloaded negative control cannot be masked by a missing adapter.
        let world = World::new();
        let mut random = mc_math::JavaRandom::new(7);
        let p = super::super::spawn::smoke(
            &mut random,
            glam::DVec3::new(-16.25, 70.0, 0.25),
            glam::DVec3::ZERO,
            1.0,
        );
        assert_eq!(
            particle_light_samples(&world, &p),
            (0, 0),
            "generic EntityFX eye sample must retain Java's unloaded-cell zero"
        );

        // Footsteps are intentionally different: their custom renderer asks World.getLightBrightness
        // directly, whose EmptyChunk path supplies the provider's full sky cell.
        let footstep = super::super::spawn::footstep(&mut random, p.pos);
        assert_eq!(
            particle_light_samples(&world, &footstep),
            (15, 0),
            "footstep current-block sampling must preserve EmptyChunk sky semantics"
        );
    }

    #[test]
    fn anaglyph_particle_upload_matches_java_channels_and_preserves_alpha() {
        let source = [
            255, 0, 0, 7, // red, translucent
            0, 255, 0, 19, // green
            0, 0, 255, 31, // blue
            128, 128, 128, 64,
        ];
        assert_eq!(particle_upload_rgba(&source, false), source);
        assert_eq!(
            particle_upload_rgba(&source, true),
            [76, 76, 76, 7, 178, 150, 0, 19, 0, 28, 178, 31, 128, 128, 128, 64,],
            "Java's integer anaglyph transform mixes RGB only"
        );
    }

    #[test]
    fn animated_particle_atlas_rects_validate_each_mip_level() {
        let valid = AtlasAnimationUpload {
            mip_level: 1,
            x: 1,
            y: 0,
            width: 1,
            height: 2,
            rgba: vec![0xAB; 8],
        };
        assert!(atlas_animation_rect_valid(&valid, 4, 4, 2));

        let mut out_of_bounds = valid.clone();
        out_of_bounds.x = 2;
        assert!(!atlas_animation_rect_valid(&out_of_bounds, 4, 4, 2));
        let mut wrong_level = valid.clone();
        wrong_level.mip_level = 2;
        assert!(!atlas_animation_rect_valid(&wrong_level, 4, 4, 2));
        let mut wrong_payload = valid;
        wrong_payload.rgba.pop();
        assert!(!atlas_animation_rect_valid(&wrong_payload, 4, 4, 2));
    }

    #[test]
    fn particle_atlas_keeps_raw_mips_across_animation_and_live_anaglyph_toggles() {
        let Ok(gpu) = Gpu::headless(16, 16, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let base = [255u8, 0, 0, 7].repeat(4 * 4);
        let mip = [0u8, 0, 255, 19].repeat(2 * 2);
        let atlas = mc_assets::Atlas {
            texture: base.clone(),
            mipmaps: vec![mip.clone()],
            w: 4,
            h: 4,
        };
        let mut phase = ParticlePhase::new(&gpu);
        phase.install_block_atlas(&gpu, &atlas);

        let animated = AtlasAnimationUpload {
            mip_level: 1,
            x: 0,
            y: 0,
            width: 1,
            height: 1,
            rgba: vec![0, 255, 0, 23],
        };
        assert!(phase.upload_atlas_animation(&gpu, &animated));
        let expected_mip = {
            let mut expected = mip.clone();
            expected[..4].copy_from_slice(&animated.rgba);
            expected
        };
        assert_eq!(
            phase
                .textures
                .get(&TextureSlot::Terrain)
                .expect("block atlas")
                .source_levels
                .borrow()
                .as_slice(),
            &[base.clone(), expected_mip.clone()],
            "animation writes raw bytes into the retained level source"
        );

        phase.set_anaglyph_with_gpu(&gpu, true);
        assert!(phase.anaglyph_enabled());
        assert_eq!(
            phase
                .textures
                .get(&TextureSlot::Terrain)
                .expect("replayed block atlas")
                .source_levels
                .borrow()
                .as_slice(),
            &[base.clone(), expected_mip.clone()],
            "enabling anaglyph must not replace the raw source cache"
        );
        phase.set_anaglyph_with_gpu(&gpu, false);
        assert!(!phase.anaglyph_enabled());
        assert_eq!(
            phase
                .textures
                .get(&TextureSlot::Terrain)
                .expect("restored block atlas")
                .source_levels
                .borrow()
                .as_slice(),
            &[base, expected_mip],
            "disabling anaglyph restores the animated raw mip bytes"
        );
    }

    /// `F-P2-client.particle-SWEEP-5`: `EntityHugeExplodeFX.renderParticle` is empty. The family
    /// remains alive so its tick path can spawn children, but phase upload must not make a quad
    /// from its icon-less Terrain-layer state.
    #[test]
    fn huge_explosion_emitter_is_an_invisible_non_billboard_family() {
        let mut r = mc_math::JavaRandom::new(41);
        let huge = super::super::spawn::huge_explode(&mut r, glam::DVec3::ZERO);
        assert!(matches!(huge.kind, ParticleKind::HugeExplode { .. }));
        assert!(
            !emits_billboard(&huge),
            "the invisible controller must not contribute phase vertices"
        );

        // Negative controls from both ordinary render layers: neither may be swallowed by the
        // special-case gate.
        let smoke = super::super::spawn::smoke(&mut r, glam::DVec3::ZERO, glam::DVec3::ZERO, 1.0);
        let large = super::super::spawn::explode_large(&mut r, glam::DVec3::ZERO, 0.0);
        assert!(emits_billboard(&smoke));
        assert!(emits_billboard(&large));
    }

    #[test]
    fn footstep_uses_its_own_texture_slot_with_blended_depth_writing() {
        let Ok(gpu) = Gpu::headless(64, 64, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let mut phase = ParticlePhase::new(&gpu);
        phase.install_footprint_texture(&gpu, &white_pixels(16, 16), 16, 16);
        let mut rng = mc_math::JavaRandom::new(43);
        let footstep = super::super::spawn::footstep(&mut rng, glam::DVec3::new(0.0, 70.0, 0.0));
        phase.upload(&gpu, &[footstep], &Camera::default(), 0.0, &World::new());
        assert_eq!(phase.vertex_count(), 6);
        assert_eq!(phase.batches.len(), 1);
        assert_eq!(phase.batches[0].slot, TextureSlot::Footprint);
        assert!(phase.batches[0].depth_write);
    }

    /// `F-P2-client.particle-02-5`: SparkFX conditionally skips its base renderer to make a
    /// twinkle. Java uses integer division in both thirds, so for `maxAge = 12` the visible ages
    /// are exactly `0..=3` (the unconditional first third) and `6..=8` (the next even band).
    #[test]
    fn twinkling_firework_spark_emits_only_on_the_java_blink_frames() {
        let mut r = mc_math::JavaRandom::new(42);
        let mut spark = super::super::spawn::firework_spark(
            &mut r,
            glam::DVec3::ZERO,
            glam::DVec3::ZERO,
            [1.0, 0.0, 0.0],
            None,
            false,
            true,
        );
        spark.max_age = 12;
        let visible: Vec<u32> = (0..12)
            .filter(|&age| {
                spark.age = age;
                emits_billboard(&spark)
            })
            .collect();
        assert_eq!(visible, vec![0, 1, 2, 3, 6, 7, 8]);

        // Negative control: the identical non-twinkling spark renders continuously.
        spark.kind = ParticleKind::FireworkSpark {
            base_tex: 160,
            trail: false,
            twinkle: false,
            fade_color: None,
        };
        assert!((0..12).all(|age| {
            spark.age = age;
            emits_billboard(&spark)
        }));
    }

    #[test]
    fn phase_builds_with_no_gpu_texture_installed() {
        let Ok(gpu) = Gpu::headless(64, 64, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let phase = ParticlePhase::new(&gpu);
        assert!(!phase.has_texture(FxLayer::Misc));
        assert_eq!(phase.batch_count(), 0);
    }

    #[test]
    fn upload_with_no_particles_produces_no_batches() {
        let Ok(gpu) = Gpu::headless(64, 64, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let mut phase = ParticlePhase::new(&gpu);
        phase.install_misc_atlas(&gpu, &white_pixels(16, 16), 16, 16);
        let world = World::new();
        let cam = Camera::default();
        phase.upload(&gpu, &[], &cam, 0.0, &world);
        assert_eq!(phase.vertex_count(), 0);
        assert_eq!(phase.batch_count(), 0);
    }

    #[test]
    fn upload_groups_particles_into_one_batch_per_populated_slot() {
        let Ok(gpu) = Gpu::headless(64, 64, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let mut phase = ParticlePhase::new(&gpu);
        phase.install_misc_atlas(&gpu, &white_pixels(16, 16), 16, 16);
        let world = World::new();
        let cam = Camera::default();
        let mut r = mc_math::JavaRandom::new(1);
        let particles: Vec<Particle> = (0..50)
            .map(|_| {
                super::super::spawn::smoke(
                    &mut r,
                    glam::DVec3::new(0.0, 70.0, 0.0),
                    glam::DVec3::ZERO,
                    1.0,
                )
            })
            .collect();
        phase.upload(&gpu, &particles, &cam, 0.0, &world);
        assert_eq!(phase.vertex_count(), 50 * 6);
        assert_eq!(
            phase.batch_count(),
            1,
            "all 50 share one (Misc, bucket) group"
        );
    }

    /// java: `EntityFootStepFX.renderParticle` samples `World.getLightBrightness(new BlockPos(this))`;
    /// ordinary particles retain `Entity.getBrightnessForRender`'s eye-cell sample, while
    /// `EntityDiggingFX` returns the source cell's packed light when its primary result is zero.
    ///
    /// The world below is lit only at `y = 71`: a particle whose feet are at `y = 70.9` has its
    /// eye in the lit cell and its feet in the dark one. The oracle covers that straddle, then
    /// checks a digging particle drifting into an unloaded neighboring chunk.
    #[test]
    fn family_brightness_uses_foot_cells_and_digging_source_fallback() {
        let Ok(gpu) = Gpu::headless(64, 64, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let mut phase = ParticlePhase::new(&gpu);
        phase.install_misc_atlas(&gpu, &white_pixels(16, 16), 16, 16);
        // `upload` builds vertices for every slot whether or not its atlas is resident; only
        // `record` needs the texture, and this test never records.
        // A no-sky-light chunk whose block light is 15 at exactly two cells, so every other
        // cell reads 0 and the comparisons below are unambiguous.
        let mut world = World::new();
        world.set_has_sky_light(false);
        let mut payload = vec![0u8; 16 * 4096 * 2 + 16 * 2048];
        let mut light = |x: usize, y: usize, z: usize| {
            let section = y / 16;
            let idx = ((y % 16) << 8) | (z << 4) | x;
            let byte = 16 * 4096 * 2 + section * 2048 + idx / 2;
            payload[byte] |= if idx % 2 == 0 { 0x0F } else { 0xF0 };
        };
        light(0, 71, 0);
        light(5, 71, 5);
        payload.resize(payload.len() + 256, 1);
        world
            .apply_chunk_data(mc_math::ChunkPos::new(0, 0), true, 0xFFFF, &payload)
            .expect("payload length");
        assert_eq!(world.block_light_at(BlockPos::new(0, 71, 0)), 15);
        assert_eq!(world.block_light_at(BlockPos::new(0, 70, 0)), 0);
        assert_eq!(world.sky_light_at(BlockPos::new(0, 71, 0)), 0);
        let cam = Camera::default();
        let mut r = mc_math::JavaRandom::new(1);

        let brightness_of = |phase: &mut ParticlePhase, p: Particle| {
            phase.upload(&gpu, std::slice::from_ref(&p), &cam, 0.0, &world);
            phase.verts[0].light
        };

        // A misc particle standing just under the lit cell: `0.2 * 0.85 = 0.17` puts its eye in
        // it.
        let mut lit = super::super::spawn::smoke(
            &mut r,
            glam::DVec3::new(0.5, 70.9, 0.5),
            glam::DVec3::ZERO,
            1.0,
        );
        lit.size = (0.2, 0.2);
        let mut flat = lit.clone();
        flat.size = (0.2, 0.0);
        let with_eye = brightness_of(&mut phase, lit);
        // Same particle with no height: reads its feet, which are dark.
        let without_eye = brightness_of(&mut phase, flat);
        assert!(
            with_eye[1] > without_eye[1],
            "eye {with_eye:?} vs feet {without_eye:?}"
        );

        // EntityFootStepFX does not call Entity.getBrightnessForRender: its custom renderer
        // samples `new BlockPos(this)`. The same straddling position is therefore dark, while a
        // footprint whose current block is the lit y=71 cell is bright.
        let foot_dark = super::super::spawn::footstep(&mut r, glam::DVec3::new(0.5, 70.9, 0.5));
        let foot_lit = super::super::spawn::footstep(&mut r, glam::DVec3::new(0.5, 71.0, 0.5));
        assert_eq!(brightness_of(&mut phase, foot_dark), [0.0, 0.0]);
        assert_eq!(brightness_of(&mut phase, foot_lit), [0.0, 1.0]);

        // A digging particle in a fully dark cell borrows its source block's light.
        let icon = super::super::types::ParticleIcon {
            min_u: 0.0,
            max_u: 1.0,
            min_v: 0.0,
            max_v: 1.0,
        };
        let dark = glam::DVec3::new(5.5, 60.0, 5.5);
        let borrowed = super::super::spawn::digging(
            &mut r,
            dark,
            glam::DVec3::ZERO,
            icon,
            super::super::spawn::DEFAULT_DIGGING_TINT,
            false,
            BlockPos::new(5, 71, 5),
        );
        let orphan = super::super::spawn::digging(
            &mut r,
            dark,
            glam::DVec3::ZERO,
            icon,
            super::super::spawn::DEFAULT_DIGGING_TINT,
            false,
            BlockPos::new(5, 60, 5),
        );
        let borrowed = brightness_of(&mut phase, borrowed);
        let orphan = brightness_of(&mut phase, orphan);
        assert!(
            borrowed[1] > orphan[1],
            "source light not borrowed: {borrowed:?} vs {orphan:?}"
        );

        // A particle that has drifted into the unloaded neighboring chunk must still borrow its
        // loaded source cell. This guards the EmptyChunk sky default from bypassing Java's
        // `isBlockLoaded` check on both sides of the override.
        let drifting = super::super::spawn::digging(
            &mut r,
            glam::DVec3::new(20.5, 60.0, 5.5),
            glam::DVec3::ZERO,
            icon,
            super::super::spawn::DEFAULT_DIGGING_TINT,
            false,
            BlockPos::new(5, 71, 5),
        );
        let drifting = brightness_of(&mut phase, drifting);
        assert_eq!(drifting, [0.0, 1.0], "unloaded drift lost source light");
    }

    /// java: `EffectRenderer.renderLitParticles` (layer 3) is one draw state — it walks both of
    /// layer 3's bucket lists without touching `depthMask` — and `EntityRenderer:1443` runs it
    /// **before** `:1446`'s `renderParticles`. `renderParticles` itself is layer-major with the
    /// no-depth-write bucket first inside each layer.
    ///
    /// Negative control: the misc/terrain layers must still split into two buckets, or the
    /// assertion below would pass for a phase that had stopped bucketing altogether.
    #[test]
    fn layer_three_is_one_unsplit_batch_and_is_drawn_first() {
        let Ok(gpu) = Gpu::headless(64, 64, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let mut phase = ParticlePhase::new(&gpu);
        phase.install_misc_atlas(&gpu, &white_pixels(16, 16), 16, 16);
        let world = World::new();
        let cam = Camera::default();
        let mut r = mc_math::JavaRandom::new(1);
        let at = glam::DVec3::new(0.0, 70.0, 0.0);
        let mut particles: Vec<Particle> = Vec::new();
        for _ in 0..4 {
            particles.push(super::super::spawn::explode_large(&mut r, at, 1.0));
        }
        // Two misc particles, one in each alpha bucket.
        let mut opaque = super::super::spawn::smoke(&mut r, at, glam::DVec3::ZERO, 1.0);
        opaque.color[3] = 1.0;
        let mut translucent = super::super::spawn::smoke(&mut r, at, glam::DVec3::ZERO, 1.0);
        translucent.color[3] = 0.5;
        particles.push(opaque);
        particles.push(translucent);

        phase.upload(&gpu, &particles, &cam, 0.0, &world);
        let batches: Vec<(TextureSlot, bool)> = phase
            .batches
            .iter()
            .map(|b| (b.slot, b.depth_write))
            .collect();
        assert_eq!(
            batches,
            vec![
                (TextureSlot::LitExplosion, true),
                (TextureSlot::Misc, false),
                (TextureSlot::Misc, true),
            ],
            "layer 3 first and unsplit; misc still splits, translucent bucket first"
        );
        // All four explosion particles land in that one batch.
        assert_eq!(phase.batches[0].end - phase.batches[0].start, 4 * 6);
    }

    // ---- The pack-supplied atlases -----------------------------------------------

    /// The vanilla jar as a one-pack [`mc_assets::ResourceManager`], or `None` when it is not on
    /// disk. Same skip convention (and same search path) `tests/s18_particle.rs` uses.
    fn vanilla_resources() -> Option<mc_assets::ResourceManager> {
        let dir = match std::env::var_os("MC_JARS_DIR") {
            Some(d) => std::path::PathBuf::from(d),
            None => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .ancestors()
                .nth(3)?
                .join("MCP-919-main")
                .join("jars"),
        };
        let jar = dir.join("versions").join("1.8.9").join("1.8.9.jar");
        if !jar.is_file() {
            eprintln!("SKIP: {} not found; set MC_JARS_DIR", jar.display());
            return None;
        }
        let mut rm = mc_assets::ResourceManager::new();
        rm.push_pack(Box::new(
            mc_assets::ZipResourcePack::open(&jar).expect("open 1.8.9.jar"),
        ));
        Some(rm)
    }

    /// Before this existed, `RenderStack::install` only ever called
    /// `install_block_atlas`, so `Misc`, `LitExplosion`, and `Footprint` had no bind group and `record`
    /// dropped their batches without a word.
    #[test]
    fn load_textures_installs_every_pack_supplied_particle_texture() {
        let Ok(gpu) = Gpu::headless(64, 64, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let Some(rm) = vanilla_resources() else {
            return;
        };

        let mut phase = ParticlePhase::new(&gpu);
        assert!(!phase.has_texture(FxLayer::Misc));
        assert!(!phase.has_texture(FxLayer::LitExplosion));
        assert!(!phase.has_texture(FxLayer::Footprint));

        phase.load_textures(&gpu, &rm);
        assert!(
            phase.has_texture(FxLayer::Misc),
            "textures/particle/particles.png should be resident"
        );
        assert!(
            phase.has_texture(FxLayer::LitExplosion),
            "textures/entity/explosion.png should be resident"
        );
        assert!(
            phase.has_texture(FxLayer::Footprint),
            "textures/particle/footprint.png should be resident"
        );
        // Not loaded here on purpose: the block atlas is stitched, not a file in the pack.
        assert!(!phase.has_texture(FxLayer::Terrain));
    }

    /// A pack with neither PNG must leave both slots empty and must not panic — vanilla logs a
    /// missing texture and carries on, and so does the render-stack build.
    #[test]
    fn load_textures_on_an_empty_pack_stack_installs_nothing() {
        let Ok(gpu) = Gpu::headless(64, 64, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let mut phase = ParticlePhase::new(&gpu);
        phase.load_textures(&gpu, &mc_assets::ResourceManager::new());
        assert!(!phase.has_texture(FxLayer::Misc));
        assert!(!phase.has_texture(FxLayer::LitExplosion));
    }

    /// Renders one full-bright `Misc`-layer particle over a blue clear and reads the frame back.
    ///
    /// This is A14-05's real acceptance criterion: `upload` batching a particle proves nothing on
    /// its own, because `record`'s `self.textures.get(&batch.slot)` used to swallow the batch
    /// afterwards. The paired uninstalled case below is the negative control — same particle,
    /// same upload, nothing on screen.
    ///
    /// `spawn::lava` is the subject because `billboard::light_boost` returns `1.0` for
    /// `ParticleKind::Lava` (java: `EntityLavaFX.getBrightnessForRender` is full-bright), so the
    /// result does not depend on this empty `World` reporting zero light.
    fn render_one_misc_particle(install_atlas: bool) -> (Vec<u8>, u32) {
        const SIZE: u32 = 64;
        let mut gpu = Gpu::headless(
            SIZE,
            SIZE,
            crate::context::GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        )
        .expect("caller checked for an adapter");

        let mut phase = ParticlePhase::new(&gpu);
        if install_atlas {
            // Opaque white everywhere, so every texel survives the shader's 1/255 alpha test and
            // the readback assertion is about "did the draw happen", not about atlas contents —
            // `the_real_particles_png_loads_and_renders_a_recognisable_flame` covers the real PNG.
            phase.install_misc_atlas(&gpu, &white_pixels(16, 16), 16, 16);
        }

        let world = World::new();
        let cam = Camera::default();
        let mut r = mc_math::JavaRandom::new(4);
        let mut p = super::super::spawn::lava(&mut r, cam.pos + cam.look_vec() * 3.0);
        p.motion = glam::DVec3::ZERO;
        p.prev_pos = p.pos;
        // Big enough to cover the middle of a 64x64 frame from 3 blocks away.
        p.kind = super::super::types::ParticleKind::Lava { peak_scale: 8.0 };
        phase.upload(&gpu, std::slice::from_ref(&p), &cam, 0.0, &world);
        assert_eq!(phase.batch_count(), 1, "one Misc batch either way");

        let mut phases = crate::pipeline::PhaseList::new();
        phases.clear_color = wgpu::Color {
            r: 0.0,
            g: 0.0,
            b: 1.0,
            a: 1.0,
        };
        phases.insert(crate::pipeline::PhaseOrder::PARTICLES, Box::new(phase));
        let frame = gpu.acquire().expect("acquire");
        phases.render(&gpu, frame.view(), &cam, 0.0);
        (gpu.read_pixels().expect("readback"), SIZE)
    }

    fn non_background_pixels(pixels: &[u8], size: u32) -> usize {
        let mut n = 0;
        for y in (size / 4)..(size * 3 / 4) {
            for x in (size / 4)..(size * 3 / 4) {
                let i = ((y * size + x) * 4) as usize;
                // Background is pure blue; anything the particle wrote raises R and G.
                if pixels[i] > 16 || pixels[i + 1] > 16 {
                    n += 1;
                }
            }
        }
        n
    }

    #[test]
    fn a_misc_batch_reaches_the_framebuffer_once_its_atlas_is_installed() {
        if Gpu::headless(8, 8, crate::context::GpuOptions::default()).is_err() {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        }
        let (with, size) = render_one_misc_particle(true);
        let (without, _) = render_one_misc_particle(false);

        let drawn = non_background_pixels(&with, size);
        let dropped = non_background_pixels(&without, size);
        assert!(
            drawn > 100,
            "a Misc-layer particle should cover the centre once install_misc_atlas has run, \
             got {drawn} changed pixels"
        );
        assert_eq!(
            dropped, 0,
            "without the atlas `record` drops the batch — this is the A14-05 defect, and it \
             must stay reproducible so the positive case above means something"
        );
    }
}

#[cfg(test)]
mod atlas_batch_owner_contract_tests {
    use super::*;

    // Fixed three-level atlas and explicit byte-offset expectations below form the oracle;
    // no production rectangle validator or source-copy helper computes expected output.
    fn atlas() -> mc_assets::Atlas {
        mc_assets::Atlas {
            texture: [17, 29, 43, 7].repeat(16),
            mipmaps: vec![[53, 67, 79, 19].repeat(4), vec![83, 97, 109, 31]],
            w: 4,
            h: 4,
        }
    }

    fn records() -> Vec<mc_assets::AtlasAnimationUpload> {
        let base = mc_assets::AtlasAnimationUpload {
            mip_level: 0,
            x: 1,
            y: 1,
            width: 2,
            height: 2,
            rgba: vec![
                151, 13, 79, 23, 2, 173, 61, 37, 83, 7, 199, 47, 113, 157, 11, 59,
            ],
        };
        let mip = mc_assets::AtlasAnimationUpload {
            mip_level: 1,
            x: 1,
            y: 0,
            width: 1,
            height: 2,
            rgba: vec![3, 211, 41, 67, 227, 17, 89, 71],
        };
        let tail = mc_assets::AtlasAnimationUpload {
            mip_level: 2,
            x: 0,
            y: 0,
            width: 1,
            height: 1,
            rgba: vec![251, 101, 5, 83],
        };
        let mut short = base.clone();
        short.rgba.pop();
        let mut right = mip.clone();
        right.x = 2;
        let mut zero_width = base.clone();
        zero_width.width = 0;
        zero_width.rgba.clear();
        let mut level = tail.clone();
        level.mip_level = 3;
        let mut overflow = tail.clone();
        overflow.x = u32::MAX;
        let mut overlap = base.clone();
        overlap.rgba.reverse();
        let mut bottom = mip.clone();
        bottom.y = 1;
        let mut zero_height = base.clone();
        zero_height.height = 0;
        zero_height.rgba.clear();
        let mut long = tail.clone();
        long.rgba.push(1);
        vec![
            base,
            short,
            mip,
            right,
            zero_width,
            tail,
            level,
            overflow,
            overlap,
            bottom,
            zero_height,
            long,
        ]
    }

    #[test]
    fn actual_owner_sink_stream_shadow_settings_and_standalone_contract() {
        let gpu = Gpu::headless(16, 16, crate::context::GpuOptions::default())
            .expect("atlas owner qualification requires a working GPU adapter");
        let atlas = atlas();
        let mut direct = ParticlePhase::new(&gpu);
        let mut captured = ParticlePhase::new(&gpu);
        direct.install_block_atlas(&gpu, &atlas);
        captured.install_block_atlas(&gpu, &atlas);
        let mut expected = vec![
            atlas.texture.clone(),
            atlas.mipmaps[0].clone(),
            atlas.mipmaps[1].clone(),
        ];
        let stream = records();
        let mut receipts = Vec::new();
        for enabled in [false, true, false] {
            direct.set_anaglyph_with_gpu(&gpu, enabled);
            captured.set_anaglyph_with_gpu(&gpu, enabled);
            assert_eq!(
                *direct
                    .textures
                    .get(&TextureSlot::Terrain)
                    .expect("terrain atlas")
                    .source_levels
                    .borrow(),
                expected,
                "toggle must retain raw bytes"
            );
            assert_eq!(
                *captured
                    .textures
                    .get(&TextureSlot::Terrain)
                    .expect("terrain atlas")
                    .source_levels
                    .borrow(),
                expected,
                "toggle must retain raw bytes"
            );
            for (index, upload) in stream.iter().enumerate() {
                let accepted = matches!(index, 0 | 2 | 5 | 8);
                match index {
                    0 | 8 => {
                        expected[0][20..28].copy_from_slice(&upload.rgba[..8]);
                        expected[0][36..44].copy_from_slice(&upload.rgba[8..]);
                    }
                    2 => {
                        expected[1][4..8].copy_from_slice(&upload.rgba[..4]);
                        expected[1][12..16].copy_from_slice(&upload.rgba[4..]);
                    }
                    5 => expected[2].copy_from_slice(&upload.rgba),
                    _ => {}
                }
                let before_count = receipts.len();
                let actual =
                    captured.upload_atlas_animation_with_sink(upload, |texture, setting| {
                        assert!(
                            std::ptr::eq(
                                texture,
                                &captured
                                    .textures
                                    .get(&TextureSlot::Terrain)
                                    .expect("terrain atlas")
                                    .texture
                            ),
                            "sink must receive this owner's texture"
                        );
                        assert_eq!(
                            setting, enabled,
                            "sink must receive the owner's current setting"
                        );
                        // This borrow also proves the mutation borrow ended before the callback.
                        assert_eq!(
                            *captured
                                .textures
                                .get(&TextureSlot::Terrain)
                                .expect("terrain atlas")
                                .source_levels
                                .borrow(),
                            expected,
                            "raw shadow must be ready before transfer"
                        );
                        receipts.push((enabled, index, upload.mip_level));
                    });
                assert_eq!(actual, accepted, "capture acceptance at record {index}");
                assert_eq!(
                    direct.upload_atlas_animation(&gpu, upload),
                    accepted,
                    "standalone acceptance at record {index}"
                );
                assert_eq!(
                    receipts.len() - before_count,
                    usize::from(accepted),
                    "one sink call per accepted record"
                );
                assert_eq!(
                    *captured
                        .textures
                        .get(&TextureSlot::Terrain)
                        .expect("terrain atlas")
                        .source_levels
                        .borrow(),
                    expected,
                    "capture raw bytes at record {index}"
                );
                assert_eq!(
                    *direct
                        .textures
                        .get(&TextureSlot::Terrain)
                        .expect("terrain atlas")
                        .source_levels
                        .borrow(),
                    expected,
                    "standalone raw bytes at record {index}"
                );
            }
        }
        assert_eq!(
            receipts,
            vec![
                (false, 0, 0),
                (false, 2, 1),
                (false, 5, 2),
                (false, 8, 0),
                (true, 0, 0),
                (true, 2, 1),
                (true, 5, 2),
                (true, 8, 0),
                (false, 0, 0),
                (false, 2, 1),
                (false, 5, 2),
                (false, 8, 0),
            ]
        );
        // Exercise the actual optional-owner absence gate, not a synthetic validator.
        let absent = ParticlePhase::new(&gpu);
        assert!(
            !absent.upload_atlas_animation_with_sink(&stream[0], |_, _| panic!(
                "absent owner called sink"
            ))
        );
        assert!(!absent.upload_atlas_animation(&gpu, &stream[0]));
    }
}

/// PART-2/PART-3 parity: the one-pass, cached-light, indexed build against the pre-change
/// six-pass build, and the frustum test against exact corner clipping. CPU-only, so they run
/// without a GPU adapter.
#[cfg(test)]
mod one_pass_build_parity_tests {
    use super::*;
    use crate::particle::billboard::{particle_camera_handoff, ParticleCameraMode};
    use crate::particle::spawn;
    use mc_math::JavaRandom;

    /// The pre-PART-2 `upload` body, verbatim except for the GPU write: six filter passes,
    /// three-probe light, per-particle basis, triangle-expanded vertices.
    fn six_pass_reference(
        particles: &[Particle],
        cam: &Camera,
        partial_ticks: f32,
        world: &World,
        skylight_subtracted: u8,
    ) -> (Vec<ParticleVertex>, Vec<DrawBatch>) {
        let mut verts = Vec::new();
        let mut batches = Vec::new();
        for (slot, buckets) in [
            (TextureSlot::LitExplosion, &[true][..]),
            (TextureSlot::Footprint, &[true][..]),
            (TextureSlot::Misc, &[false, true][..]),
            (TextureSlot::Terrain, &[false, true][..]),
        ] {
            for &depth_write in buckets {
                let bucket = if depth_write {
                    AlphaBucket::Opaque
                } else {
                    AlphaBucket::Translucent
                };
                let start = verts.len() as u32;
                for p in particles {
                    if !emits_billboard(p) || slot_of(p.layer) != slot {
                        continue;
                    }
                    if slot != TextureSlot::LitExplosion && p.alpha_bucket() != bucket {
                        continue;
                    }
                    let (sky, block) = particle_light_samples(world, p);
                    let (sky, block) = (f32::from(sky), f32::from(block));
                    let mut quad = billboard::quad(p, cam, partial_ticks, sky, block);
                    if matches!(p.kind, ParticleKind::Footstep) {
                        let brightness = footstep_brightness(world, p, skylight_subtracted);
                        for vertex in &mut quad {
                            vertex.color[0] = brightness;
                            vertex.color[1] = brightness;
                            vertex.color[2] = brightness;
                            vertex.lightmap_mode = 1.0;
                        }
                    }
                    verts
                        .extend_from_slice(&[quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]]);
                }
                let end = verts.len() as u32;
                if end != start {
                    batches.push(DrawBatch {
                        slot,
                        depth_write,
                        start,
                        end,
                    });
                }
            }
        }
        (verts, batches)
    }

    fn stone() -> mc_data::BlockState {
        mc_data::BlockState::new(mc_data::block_by_name("stone").expect("stone"), 0)
    }

    /// Three loaded columns with sections, light and an overhang (so absent sections below the
    /// height map read sky 0); everything else unloaded.
    fn lit_world(has_sky: bool) -> World {
        let mut world = World::new();
        world.set_has_sky_light(has_sky);
        for (x, z) in [(0, 0), (-16, 0), (0, -16)] {
            world.load_empty_chunk(BlockPos::new(x, 0, z).chunk_pos());
        }
        world.set_block(BlockPos::new(2, 90, 2), stone());
        world.set_block(BlockPos::new(-5, 40, 3), stone());
        let mut n = 0u8;
        for x in [-14, -5, 0, 2, 7, 15] {
            for y in [0, 15, 16, 39, 41, 64, 70, 71, 89, 91, 255] {
                for z in [-9, 0, 2, 3, 13] {
                    n = n.wrapping_add(7);
                    let p = BlockPos::new(x, y, z);
                    world.set_light_for(mc_world::SkyBlock::Block, p, n % 16);
                    world.set_light_for(mc_world::SkyBlock::Sky, p, (n / 3) % 16);
                }
            }
        }
        world
    }

    fn particle_mix(rng: &mut JavaRandom, around: DVec3, n: usize) -> Vec<Particle> {
        let icon = super::super::types::ParticleIcon {
            min_u: 0.25,
            max_u: 0.5,
            min_v: 0.0,
            max_v: 0.25,
        };
        (0..n)
            .map(|i| {
                let jitter = DVec3::new(
                    f64::from(rng.next_f32()) * 40.0 - 20.0,
                    f64::from(rng.next_f32()) * 40.0 - 20.0,
                    f64::from(rng.next_f32()) * 40.0 - 20.0,
                );
                let pos = around + jitter;
                let mut p = match i % 9 {
                    0 => spawn::explode_large(rng, pos, 1.0),
                    1 => spawn::footstep(rng, pos),
                    2 => spawn::digging(
                        rng,
                        pos,
                        DVec3::ZERO,
                        icon,
                        spawn::DEFAULT_DIGGING_TINT,
                        false,
                        BlockPos::new(2, 71, 3),
                    ),
                    3 => spawn::huge_explode(rng, pos),
                    4 => spawn::mob_appearance(rng, pos),
                    5 => spawn::firework_spark(rng, pos, DVec3::ZERO, [1.0; 3], None, false, true),
                    _ => spawn::smoke(rng, pos, DVec3::new(0.1, 0.0, -0.1), 1.0),
                };
                if i % 4 == 0 {
                    p.color[3] = 0.5;
                }
                p.prev_pos = p.pos - DVec3::new(0.05, 0.02, -0.03);
                p.age = (i % 7) as u32;
                p
            })
            .collect()
    }

    fn handoff_camera(pos: DVec3, yaw: f32, pitch: f32, mode: ParticleCameraMode) -> Camera {
        let mut cam = Camera {
            pos,
            yaw,
            pitch,
            ..Camera::default()
        };
        cam.update();
        particle_camera_handoff(&cam, yaw + 7.0, pitch * 0.5, mode)
    }

    #[test]
    fn one_pass_indexed_build_expands_to_the_six_pass_vertices_exactly() {
        let mut rng = JavaRandom::new(0x0B1);
        for (has_sky, mode) in [
            (true, ParticleCameraMode::FirstPerson),
            (true, ParticleCameraMode::ThirdPersonFront),
            (false, ParticleCameraMode::ThirdPersonRear),
        ] {
            let world = lit_world(has_sky);
            let around = DVec3::new(0.0, 70.0, 0.0);
            let particles = particle_mix(&mut rng, around, 400);
            let cam = handoff_camera(around, 33.0, -21.0, mode);
            let (old_verts, old_batches) = six_pass_reference(&particles, &cam, 0.37, &world, 4);

            let mut members: [Vec<u32>; 6] = Default::default();
            let (mut verts, mut batches) = (Vec::new(), Vec::new());
            let quads = build_particle_quads(
                &particles,
                &cam,
                0.37,
                &world,
                4,
                false,
                &mut members,
                &mut verts,
                &mut batches,
            );
            assert_eq!(verts.len(), quads as usize * 4);
            let expanded: Vec<ParticleVertex> = quad_indices(quads)
                .into_iter()
                .map(|i| verts[i as usize])
                .collect();
            assert_eq!(
                expanded, old_verts,
                "{mode:?}: vertex stream must be bit-identical"
            );
            assert_eq!(
                batches, old_batches,
                "{mode:?}: same groups, same index ranges"
            );
            assert!(batches.len() >= 4, "the mix must populate several groups");

            // With the frustum test on (the camera sits inside the cloud, so most particles are
            // behind or beside it), the kept quads must be the full build's quads, byte for
            // byte and in the same order, and the batches a subsequence of the full batches.
            let (mut culled_verts, mut culled_batches) = (Vec::new(), Vec::new());
            let kept = build_particle_quads(
                &particles,
                &cam,
                0.37,
                &world,
                4,
                true,
                &mut members,
                &mut culled_verts,
                &mut culled_batches,
            );
            assert!(kept > 0 && kept < quads, "{mode:?}: kept {kept} of {quads}");
            let mut full = verts.chunks_exact(4);
            for kept_quad in culled_verts.chunks_exact(4) {
                assert!(
                    full.any(|quad| quad == kept_quad),
                    "{mode:?}: the culled build changed or reordered a quad"
                );
            }
            let mut full_groups = batches.iter().map(|b| (b.slot, b.depth_write));
            for batch in &culled_batches {
                assert!(full_groups.any(|group| group == (batch.slot, batch.depth_write)));
            }
        }
    }

    #[test]
    fn light_probe_matches_the_three_probe_oracle_everywhere() {
        let mut rng = JavaRandom::new(0x11);
        for has_sky in [true, false] {
            let world = lit_world(has_sky);
            // One probe across the whole sweep, so cache hits and slot replacement both happen.
            let mut probe = LightProbe::new(&world);
            for x in (-40..40).step_by(3) {
                for z in (-40..40).step_by(5) {
                    for y in [-1, 0, 15, 16, 39, 40, 41, 64, 70, 71, 89, 90, 91, 255, 256] {
                        let pos = DVec3::new(f64::from(x) + 0.3, f64::from(y) - 0.1, f64::from(z));
                        let source = BlockPos::new(-x, y.clamp(0, 255), z / 2);
                        let icon = super::super::types::ParticleIcon {
                            min_u: 0.0,
                            max_u: 1.0,
                            min_v: 0.0,
                            max_v: 1.0,
                        };
                        for p in [
                            spawn::smoke(&mut rng, pos, DVec3::ZERO, 1.0),
                            spawn::footstep(&mut rng, pos),
                            spawn::digging(
                                &mut rng,
                                pos,
                                DVec3::ZERO,
                                icon,
                                spawn::DEFAULT_DIGGING_TINT,
                                false,
                                source,
                            ),
                        ] {
                            assert_eq!(
                                probe_light_samples(&mut probe, &p),
                                particle_light_samples(&world, &p),
                                "{:?} at {pos:?} (sky {has_sky})",
                                p.kind
                            );
                        }
                    }
                }
            }
        }
    }

    /// PART-2/3 timing: CPU build of 1,400 particles (a TNT burst) around a lit column, old
    /// six-pass build vs the new one, with and without the frustum test. Printed, not asserted.
    #[test]
    #[ignore = "microbenchmark: run with --include-ignored --nocapture"]
    fn microbench_part2_particle_build() {
        use std::time::Instant;
        let world = lit_world(true);
        let mut rng = JavaRandom::new(0xB);
        let around = DVec3::new(1.0, 70.0, 1.0);
        let particles: Vec<Particle> = (0..1_400)
            .map(|i| {
                let jitter = DVec3::new(
                    f64::from(rng.next_f32()) * 8.0 - 4.0,
                    f64::from(rng.next_f32()) * 8.0 - 4.0,
                    f64::from(rng.next_f32()) * 8.0 - 4.0,
                );
                if i % 2 == 0 {
                    spawn::explode(&mut rng, around + jitter, DVec3::ZERO)
                } else {
                    spawn::smoke(&mut rng, around + jitter, DVec3::ZERO, 1.0)
                }
            })
            .collect();
        // Standing 6 blocks south of the blast, looking at it.
        let cam = handoff_camera(
            around + DVec3::new(0.0, 0.0, 6.0),
            180.0,
            0.0,
            ParticleCameraMode::FirstPerson,
        );
        const N: u32 = 50;
        let t = Instant::now();
        for _ in 0..N {
            std::hint::black_box(six_pass_reference(&particles, &cam, 0.5, &world, 0));
        }
        let old = t.elapsed() / N;
        let mut members: [Vec<u32>; 6] = Default::default();
        let (mut verts, mut batches) = (Vec::new(), Vec::new());
        let mut timed = |cull: bool| {
            let t = Instant::now();
            let mut quads = 0;
            for _ in 0..N {
                quads = build_particle_quads(
                    &particles,
                    &cam,
                    0.5,
                    &world,
                    0,
                    cull,
                    &mut members,
                    &mut verts,
                    &mut batches,
                );
            }
            (t.elapsed() / N, quads)
        };
        let (new_all, all_quads) = timed(false);
        let (new_culled, kept_quads) = timed(true);
        eprintln!(
            "PART-2 build of 1400: old {old:?} ({} B), new {new_all:?} ({} B), new+frustum \
             {new_culled:?} ({kept_quads}/{all_quads} quads kept)",
            1_400 * 6 * ParticleVertex::SIZE,
            all_quads as usize * 4 * ParticleVertex::SIZE,
        );
    }

    #[test]
    fn quad_index_pattern_is_the_two_documented_triangles() {
        assert_eq!(quad_indices(2), vec![0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]);
    }

    /// Every particle the frustum drops must have all four emitted corners strictly outside one
    /// clip plane of the real (untagged) `proj * view`, checked in f64; and the test must both
    /// keep and drop particles, or it proves nothing.
    #[test]
    fn frustum_drops_only_quads_wholly_outside_one_clip_plane() {
        let mut rng = JavaRandom::new(0xF0);
        let (mut kept, mut dropped) = (0, 0);
        for (yaw, pitch, mode) in [
            (0.0, 0.0, ParticleCameraMode::FirstPerson),
            (137.0, 62.0, ParticleCameraMode::ThirdPersonFront),
            (-71.0, -88.0, ParticleCameraMode::ThirdPersonRear),
        ] {
            let around = DVec3::new(100.5, 64.0, -200.25);
            let cam = handoff_camera(around, yaw, pitch, mode);
            let frustum = ParticleFrustum::from_camera(&cam);
            let basis = billboard::QuadBasis::new(&cam);
            let mut real_view = cam.view;
            real_view.x_axis.w = 0.0;
            let m = (cam.proj * real_view).as_dmat4();
            for p in particle_mix(&mut rng, around, 600) {
                if draw_group(&p).is_none() {
                    continue;
                }
                let (center, radius) = billboard::cull_sphere(&p, &cam, 0.5, &basis);
                let quad = billboard::quad_with_basis(&p, &cam, 0.5, 15.0, 0.0, &basis);
                for v in &quad {
                    let d = glam::Vec3::from(v.pos).distance(center);
                    // f32 corner rounding at |centre| <= ~35 blocks is ~1e-5; the frustum's own
                    // slack is `radius * 1.001 + 0.01`, 100x looser than this bound.
                    assert!(d <= radius * 1.0001 + 1e-4, "sphere must contain corner");
                }
                if frustum.may_be_visible(center, radius) {
                    kept += 1;
                    continue;
                }
                dropped += 1;
                let clips: Vec<glam::DVec4> = quad
                    .iter()
                    .map(|v| {
                        m * glam::DVec4::new(v.pos[0].into(), v.pos[1].into(), v.pos[2].into(), 1.0)
                    })
                    .collect();
                let outside = |f: fn(&glam::DVec4) -> f64| clips.iter().all(|c| f(c) < 0.0);
                assert!(
                    outside(|c| c.w + c.x)
                        || outside(|c| c.w - c.x)
                        || outside(|c| c.w + c.y)
                        || outside(|c| c.w - c.y)
                        || outside(|c| c.z)
                        || outside(|c| c.w - c.z),
                    "dropped a quad that touches the clip volume: {:?}",
                    p.kind
                );
            }
        }
        assert!(kept > 50 && dropped > 50, "kept {kept}, dropped {dropped}");
    }
}
